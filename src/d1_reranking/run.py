"""Run bootstrap and immutable source records for the D1 extension."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unified_reranking.hashing import atomic_json, atomic_text, sha256_file
from unified_reranking.ledger import initialize_ledger

from .contracts import (
    EXPECTED_EVALUATOR_SHA256,
    EXPECTED_UNIFIED_FINAL_LOCK_SHA256,
    RUN_DIRECTORIES,
    RunState,
)
from .io import atomic_copy


_STATE_ORDER = {
    state.value: index
    for index, state in enumerate(
        (
            RunState.AUDIT,
            RunState.CANDIDATES_FROZEN,
            RunState.FEATURES_READY,
            RunState.TRAIN_OOF,
            RunState.VALIDATION_SCREEN,
            RunState.PRELOCK_LABEL_FREE,
            RunState.FORMAL_LOCKED,
            RunState.FORMAL_EXECUTED,
            RunState.POSTFORMAL,
            RunState.COMPLETE,
        )
    )
}


def transition_pipeline_status(
    run_dir: str | Path,
    *,
    status: RunState | str,
    first_incomplete_stage: str | None,
    formal_test_executed: bool,
    test_candidate_labels_read: bool,
    formal_test_execution_count: int,
) -> dict[str, Any]:
    """Atomically advance the run lifecycle without permitting rollback."""

    root = Path(run_dir).expanduser().resolve()
    path = root / "pipeline_status.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"D1 pipeline status is not a regular file: {path}")
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("D1 pipeline status is invalid JSON") from error
    if not isinstance(current, dict):
        raise RuntimeError("D1 pipeline status is not an object")
    target = str(status)
    observed = str(current.get("status", ""))
    if (
        observed in {RunState.COMPLETE.value, RunState.FAILED.value}
        and observed != target
    ):
        raise PermissionError(
            f"D1 terminal pipeline status cannot transition: {observed} -> {target}"
        )
    if (
        target != RunState.FAILED.value
        and observed in _STATE_ORDER
        and target in _STATE_ORDER
        and _STATE_ORDER[target] < _STATE_ORDER[observed]
    ):
        raise PermissionError(
            f"D1 pipeline status cannot roll back: {observed} -> {target}"
        )
    if formal_test_execution_count not in (0, 1):
        raise ValueError("D1 formal Test execution count must be 0 or 1")
    if formal_test_executed and formal_test_execution_count != 1:
        raise ValueError("D1 executed state requires exactly one formal execution")
    if test_candidate_labels_read and not formal_test_executed:
        raise ValueError("D1 Test-label state cannot precede formal execution")
    if (
        observed == target
        and current.get("first_incomplete_stage") == first_incomplete_stage
        and current.get("formal_test_executed") is bool(formal_test_executed)
        and current.get("test_candidate_labels_read")
        is bool(test_candidate_labels_read)
        and int(current.get("formal_test_execution_count", 0))
        == formal_test_execution_count
    ):
        return _artifact(path)
    value = {
        **current,
        "status": target,
        "first_incomplete_stage": first_incomplete_stage,
        "formal_test_executed": bool(formal_test_executed),
        "test_candidate_labels_read": bool(test_candidate_labels_read),
        "formal_test_execution_count": int(formal_test_execution_count),
        "previous_status": observed,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(path, value)
    return _artifact(path)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True, text=True
    )
    return (result.stdout + result.stderr).strip()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def assert_writable_prelock(run_dir: Path) -> None:
    locks = [
        run_dir / "08_lock" / "FORMAL_TEST_LOCK.json",
        run_dir / "FINAL_RUN_LOCK.json",
        run_dir / "COMPLETE",
    ]
    present = [str(path) for path in locks if path.exists()]
    if present:
        raise PermissionError(
            f"D1 bootstrap refuses to write after lock/completion: {present}"
        )


def bootstrap_run(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    unified_run: str | Path,
    snapshot_a: str | Path,
    evaluator_path: str | Path,
    resume: bool,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    run = Path(run_dir).resolve()
    unified = Path(unified_run).resolve()
    snapshot = Path(snapshot_a).resolve()
    evaluator = Path(evaluator_path).resolve()
    if run.exists() and any(run.iterdir()) and not resume:
        raise FileExistsError(f"D1 run is non-empty; pass --resume: {run}")
    assert_writable_prelock(run)
    for directory in RUN_DIRECTORIES:
        (run / directory).mkdir(parents=True, exist_ok=True)
    initialize_ledger(run / "run_ledger.sqlite")

    unified_lock = unified / "FINAL_RUN_LOCK.json"
    observed_lock = sha256_file(unified_lock)
    if observed_lock != EXPECTED_UNIFIED_FINAL_LOCK_SHA256:
        raise RuntimeError(
            f"completed three-route lock drift: {observed_lock} != {EXPECTED_UNIFIED_FINAL_LOCK_SHA256}"
        )
    observed_evaluator = sha256_file(evaluator)
    if observed_evaluator != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            f"canonical evaluator drift: {observed_evaluator} != {EXPECTED_EVALUATOR_SHA256}"
        )
    required_snapshot = {
        "run_manifest": snapshot / "run_manifest.json",
        "frozen_protocol": snapshot / "frozen_protocol.yaml",
        "final_output_manifest": snapshot / "final_output_manifest.json",
        "completed": snapshot / "COMPLETED",
        "test_nms_candidates": snapshot
        / "candidates"
        / "dexnet_nms_candidates.parquet",
    }
    sources = {name: _artifact(path) for name, path in required_snapshot.items()}
    sources["unified_final_lock"] = _artifact(unified_lock)
    sources["canonical_evaluator"] = _artifact(evaluator)

    paired_sources = {
        split: unified / "01_manifests" / f"paired_{split}.parquet"
        for split in ("train", "validation", "test")
    }
    for split, path in paired_sources.items():
        sources[f"paired_{split}"] = _artifact(path)

    copied_evaluator = atomic_copy(
        evaluator, run / "configs" / "canonical_evaluator.py"
    )
    if sha256_file(copied_evaluator) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("copied evaluator is not byte-identical")
    paired_copies = {}
    for split, source in paired_sources.items():
        name = (
            "d1_paired_manifest.parquet"
            if split == "test"
            else f"d1_paired_{split}.parquet"
        )
        copied = atomic_copy(source, run / "01_manifests" / name)
        if sha256_file(copied) != sha256_file(source):
            raise RuntimeError(f"D1 {split} paired manifest is not byte-identical")
        paired_copies[split] = _artifact(copied)
    atomic_json(
        run / "01_manifests" / "d1_pairing_manifest.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "join_key": "sample_id",
            "same_denominator_and_assets_as_unified_run": True,
            "candidate_test_labels_read": False,
            "artifacts": paired_copies,
        },
    )
    atomic_text(
        run / "01_manifests" / "d1_pairing_audit.md",
        "# D1 pairing audit\n\n"
        "The Train, Validation and Test paired manifests are byte-identical copies "
        "of the completed unified-run manifests. Candidate joins use `sample_id`; "
        "row number and basename matching are forbidden. Test candidate labels "
        "were not opened.\n",
    )
    atomic_text(
        run / "01_manifests" / "excluded_or_mismatched_samples.csv",
        "split,sample_id,reason\n",
    )
    source_immutability = {
        "status": "PASS",
        "scope": "locked retrospective D1 extension; source runs are read-only",
        "source_records": sources,
    }
    atomic_json(
        run / "00_audit" / "SOURCE_RUN_IMMUTABILITY_BEFORE.json", source_immutability
    )

    created_at = datetime.now(timezone.utc).isoformat()
    manifest_path = run / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != "fair-d1-reranking-retrospective-extension-v1":
            raise RuntimeError("existing D1 run has a different protocol")
    else:
        manifest = {
            "schema_version": 1,
            "protocol": "fair-d1-reranking-retrospective-extension-v1",
            "status": RunState.AUDIT,
            "created_at_utc": created_at,
            "run_dir": str(run),
            "repo_root": str(root),
            "route": "D1",
            "denominator_contract": {
                "test_samples": 7675,
                "source": "unified paired Test",
            },
            "retrospective_extension": True,
            "formal_test_execution_count": 0,
            "test_label_state": "LOCKED_PREVALIDATION",
            "source_run_records": sources,
        }
        atomic_json(manifest_path, manifest)
    pipeline = {
        "schema_version": 1,
        "status": RunState.AUDIT,
        "first_incomplete_stage": "P1_SOURCE_RECONCILIATION",
        "formal_test_executed": False,
        "test_candidate_labels_read": False,
    }
    atomic_json(run / "pipeline_status.json", pipeline)
    environment = (
        "\n".join(
            [
                f"captured_at_utc={created_at}",
                f"cwd={root}",
                f"python={sys.version.replace(chr(10), ' ')}",
                f"platform={platform.platform()}",
                f"machine={platform.machine()}",
            ]
        )
        + "\n"
    )
    atomic_text(run / "environment.txt", environment)
    atomic_text(
        run / "git_state.txt",
        f"HEAD\n{_git(root, 'rev-parse', 'HEAD')}\n\nSTATUS\n{_git(root, 'status', '--short')}\n",
    )
    readme = f"""# Locked retrospective D1 reranking extension

This run adds `HiFi-CS -> Dex-Net -> GQ-CNN` as route D1 without modifying the
completed three-route run. It is retrospective and must not be described as a
pristine blind Test experiment.

Source authority:

- completed three-route run: `{unified}`
- Snapshot A candidate source: `{snapshot}`
- canonical evaluator SHA-256: `{EXPECTED_EVALUATOR_SHA256}`

Before the formal lock, Test candidate membership, geometry, GQ-CNN q/rank and
label-free features may be frozen. Test ground truth/candidate labels may not be
opened as a table. Use `pipeline_status.json` as the current stage authority.
"""
    atomic_text(run / "README_REPRODUCE.md", readme)
    if not (run / "commands.log").exists():
        atomic_text(run / "commands.log", "")
    audit_md = f"""# Source and environment audit

- Status: PASS (lightweight preflight)
- Protocol: locked retrospective D1 extension
- Completed three-route lock SHA-256: `{observed_lock}`
- Canonical evaluator SHA-256: `{observed_evaluator}`
- Snapshot A: `{snapshot}`
- Git HEAD: `{_git(root, "rev-parse", "HEAD")}`
- Python: `{sys.version.split()[0]}`
- Test candidate labels opened: no

Heavy candidate/training work remains blocked until the project resource gate
passes. Existing frozen candidates will be reused; Dex-Net/GQ-CNN generation is
not authorized by this bootstrap.
"""
    atomic_text(run / "00_audit" / "SOURCE_AND_ENVIRONMENT_AUDIT.md", audit_md)
    return manifest
