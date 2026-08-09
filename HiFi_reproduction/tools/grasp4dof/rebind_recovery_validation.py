#!/usr/bin/env python3
"""Rebind copied validation metadata to a hash-identical sibling recovery run.

The prediction tables and metric payloads are never rewritten.  Only G1/C1
configuration paths and their content-addressed validation metadata change,
after proving that the copied configuration differs solely by the location of
the hash-identical fine-tuned checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


METHODS = ("G1", "C1")
PAYLOADS = (
    "metrics.json",
    "runtime_metrics.json",
    "memory_metrics.json",
    "per_sample_predictions.parquet",
    "per_candidate_predictions.parquet",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _replace_root(value: Any, *, old: str, new: str) -> Any:
    if isinstance(value, str):
        return new + value[len(old) :] if value.startswith(old) else value
    if isinstance(value, list):
        return [_replace_root(item, old=old, new=new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_root(item, old=old, new=new) for key, item in value.items()
        }
    return value


def _assert_semantic_config_rebind(
    parent_config: Path, recovery_config: Path, *, parent: Path, recovery: Path
) -> dict[str, Any]:
    before = _json(parent_config)
    after = _json(recovery_config)
    restored = _replace_root(after, old=str(recovery), new=str(parent))
    if restored != before:
        raise ValueError(
            f"recovery config changed beyond a root-path rebind: {recovery_config}"
        )
    old_checkpoint = Path(str(before["finetuned_checkpoint"])).resolve()
    new_checkpoint = Path(str(after["finetuned_checkpoint"])).resolve()
    expected_new = recovery / old_checkpoint.relative_to(parent)
    if new_checkpoint != expected_new or not new_checkpoint.is_file():
        raise ValueError(
            f"fine-tuned checkpoint was not rebound into sibling: {new_checkpoint}"
        )
    old_sha = _sha256(old_checkpoint)
    new_sha = _sha256(new_checkpoint)
    declared = str(after["finetuned_checkpoint_sha256"])
    if old_sha != new_sha or new_sha != declared:
        raise ValueError(f"fine-tuned checkpoint bytes drifted: {recovery_config}")
    return {
        "parent_config": str(parent_config),
        "parent_config_sha256": _sha256(parent_config),
        "recovery_config": str(recovery_config),
        "recovery_config_sha256": _sha256(recovery_config),
        "only_root_paths_changed": True,
        "parent_checkpoint": str(old_checkpoint),
        "recovery_checkpoint": str(new_checkpoint),
        "checkpoint_sha256": new_sha,
    }


def _rebind_output(
    parent_output: Path,
    recovery_output: Path,
    *,
    parent: Path,
    recovery: Path,
    config: Path,
    complete_filename: str,
) -> dict[str, Any]:
    if parent_output.resolve() == recovery_output.resolve():
        raise ValueError("recovery output must be distinct from its parent")
    payload_hashes: dict[str, str] = {}
    for name in PAYLOADS:
        source = parent_output / name
        copied = recovery_output / name
        if (
            not source.is_file()
            or not copied.is_file()
            or _sha256(source) != _sha256(copied)
        ):
            raise ValueError(
                f"copied validation payload drifted: {recovery_output}/{name}"
            )
        payload_hashes[name] = _sha256(copied)

    parent_run_config = parent_output / "run_config.json"
    recovery_run_config = recovery_output / "run_config.json"
    if _sha256(parent_run_config) != _sha256(recovery_run_config):
        raise ValueError(
            f"validation run config already drifted: {recovery_run_config}"
        )
    run_config = _replace_root(
        _json(recovery_run_config), old=str(parent), new=str(recovery)
    )
    run_config["config_sha256"] = _sha256(config)
    _atomic_json(recovery_run_config, run_config)

    parent_complete = parent_output / complete_filename
    recovery_complete = recovery_output / complete_filename
    if _sha256(parent_complete) != _sha256(recovery_complete):
        raise ValueError(
            f"validation completion marker already drifted: {recovery_complete}"
        )
    complete = _json(recovery_complete)
    complete["config_sha256"] = _sha256(config)
    artifacts = complete.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"completion artifact inventory missing: {recovery_complete}")
    complete["artifacts"]["run_config.json"] = _sha256(recovery_run_config)
    _atomic_json(recovery_complete, complete)
    return {
        "parent_output": str(parent_output),
        "recovery_output": str(recovery_output),
        "oracle": bool(run_config["oracle"]),
        "payloads_exactly_reused": payload_hashes,
        "parent_run_config_sha256": _sha256(parent_run_config),
        "recovery_run_config_sha256": _sha256(recovery_run_config),
        "parent_complete_sha256": _sha256(parent_complete),
        "recovery_complete_sha256": _sha256(recovery_complete),
        "recovery_config_sha256": _sha256(config),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    parent = args.parent_run.expanduser().resolve()
    recovery = args.run_dir.expanduser().resolve()
    if parent == recovery or not parent.is_dir() or not recovery.is_dir():
        raise ValueError("parent and recovery must be distinct existing directories")
    if (recovery / "manifests/experiment_lock.json").exists():
        raise RuntimeError("validation recovery rebind is forbidden after lock")
    output_path = recovery / "audit/recovery_lineage.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    snapshot = recovery / "audit/parent_failed_lock_snapshot"
    selection = _json(snapshot / "primary_validation_selection.json")
    parent_lock = snapshot / "manifests/experiment_lock.json"
    parent_marker = snapshot / "EXPERIMENT_LOCKED.json"
    parent_failure_log = snapshot / "logs/formal_R0.log"
    parent_commands = snapshot / "commands.log"
    for path in (parent_lock, parent_marker, parent_failure_log, parent_commands):
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(path.is_file() for path in (recovery / "formal_test").rglob("*")):
        raise RuntimeError("recovery formal-test directory must contain no artifacts")

    config_records: dict[str, Any] = {}
    output_records: dict[str, Any] = {}
    for method_id in METHODS:
        parent_config = Path(
            str(selection["selected_configs"][method_id]["path"])
        ).resolve()
        config = recovery / parent_config.relative_to(parent)
        config_records[method_id] = _assert_semantic_config_rebind(
            parent_config, config, parent=parent, recovery=recovery
        )
        source = selection["sources"][method_id]
        records = []
        for raw in (source, source["oracle"]):
            parent_output = Path(str(raw["directory"])).resolve()
            recovery_output = recovery / parent_output.relative_to(parent)
            records.append(
                _rebind_output(
                    parent_output,
                    recovery_output,
                    parent=parent,
                    recovery=recovery,
                    config=config,
                    complete_filename=str(raw["complete_filename"]),
                )
            )
        output_records[method_id] = records

    evidence = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "recovery_reason": "R0 legacy success flags used an evaluator-incompatible angle convention",
        "parent_run": str(parent),
        "recovery_run": str(recovery),
        "parent_lock_sha256": _sha256(parent_lock),
        "parent_lock_marker_sha256": _sha256(parent_marker),
        "parent_formal_failure_log_sha256": _sha256(parent_failure_log),
        "parent_commands_sha256": _sha256(parent_commands),
        "parent_failed_formal_attempt_preserved": True,
        "test_metrics_read_for_selection": False,
        "validation_prediction_payloads_recomputed": False,
        "config_rebinds": config_records,
        "validation_output_rebinds": output_records,
    }
    _atomic_json(output_path, evidence)
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
