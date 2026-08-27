#!/usr/bin/env python3
"""Run label-free G1/C1 predicted replay and compare the frozen candidate table."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HIFI = ROOT / "HiFi_reproduction"
for entry in (str(SRC), str(HIFI)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from gtmask_counterfactual.audit import require_verified_source  # noqa: E402
from gtmask_counterfactual.execution import FROZEN_G1_C1_SOURCE  # noqa: E402
from gtmask_counterfactual.io import (  # noqa: E402
    artifact_record,
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from gtmask_counterfactual.native_replay import (  # noqa: E402
    NativeReplayMissingError,
    assert_exact_native_replay,
    load_replay_sample,
    load_frozen_native_module,
    replay_label_free_samples,
    write_canonical_replay_frames,
    write_replay_sample,
)
from gtmask_counterfactual.resource import (  # noqa: E402
    collect_fresh_three_by_five_gate,
    exclusive_d1_flock,
    validate_fresh_gate,
    validate_live_resources,
)
from src.grasping.backends import BackendSample  # noqa: E402
from src.grasping.backends.training import load_finetuned_model  # noqa: E402
from src.grasping.common.sample_io import CompactSampleLoader, read_deployment_manifest  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--unified-run", type=Path, required=True)
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--resource-gate", type=Path)
    gate.add_argument("--collect-resource-gate", action="store_true")
    parser.add_argument("--route", choices=("g1", "c1"), required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--branch", choices=("predicted",), default="predicted")
    parser.add_argument("--pool", choices=("allnms",), default="allnms")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )
    return parser.parse_args()


def _load_gate(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    if path.expanduser().is_symlink() or not source.is_file():
        raise ValueError(f"resource gate must be a regular non-symlink file: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("resource gate must contain a JSON object")
    return value


def _prepare_gate(
    args: argparse.Namespace, *, run_dir: Path
) -> tuple[dict[str, object], Path]:
    """Collect or load a fresh gate while the global heavy lease is held."""

    if args.collect_resource_gate:
        value = collect_fresh_three_by_five_gate(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir.expanduser().resolve(),
        )
        path = (
            run_dir
            / "00_audit/resource_gates"
            / f"predicted_{args.route}_{value['content_sha256'][:20]}.json"
        )
        if path.exists():
            raise FileExistsError(f"resource gate already exists: {path}")
        atomic_json(path, value)
    else:
        path = args.resource_gate.expanduser().resolve()
        value = _load_gate(path)
    validate_fresh_gate(value)
    return value, path


def _load_locked_source_records(unified_run: Path) -> dict[str, dict[str, Any]]:
    """Verify the G1/C1 source records already frozen by the unified run."""

    path = unified_run / "00_audit/source_run_hashes.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("unified source_run_hashes.json must be an object")
    required = {"test_samples", "g1_selected", "c1_selected"}
    if not required.issubset(value):
        raise ValueError("unified source hash inventory is incomplete")
    records: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        record = value[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"unified source record is invalid: {name}")
        observed = artifact_record(str(record.get("path", "")))
        if any(record.get(key) != observed[key] for key in ("path", "sha256", "bytes")):
            raise ValueError(f"unified source record changed: {name}")
        records[name] = observed
    return records


def _validate_deployment(deployment: Sequence[Mapping[str, Any]]) -> list[str]:
    sample_ids = [str(row.get("sample_id", "")) for row in deployment]
    if len(sample_ids) != 7675 or any(not value for value in sample_ids):
        raise ValueError("predicted replay denominator must be exactly 7,675")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("predicted replay sample IDs are duplicated")
    return sample_ids


def _load_existing_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("predicted replay manifest must be an object")
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise ValueError("predicted replay manifest self hash differs")
    return value


def main() -> int:
    args = parse_args()
    if args.workers != 1:
        raise ValueError("frozen replay is single-worker")
    run = args.run_dir.expanduser().resolve()
    require_verified_source(run, source_name="unified", source_run=args.unified_run)
    module = load_frozen_native_module()
    source = args.source_run.expanduser().resolve()
    if source != FROZEN_G1_C1_SOURCE.resolve():
        raise ValueError("--source-run must be the frozen G1/C1 modular source")
    source_records = _load_locked_source_records(
        args.unified_run.expanduser().resolve()
    )
    samples_path = source / "manifests/test_samples.parquet"
    selected_path = source / "selected_configs" / f"{args.route.upper()}.json"
    if args.config.expanduser().resolve() != selected_path:
        raise ValueError("--config must be the frozen selected route configuration")
    frozen_path = (
        args.unified_run.expanduser().resolve()
        / f"02_candidates/{args.route}_test_all.parquet"
    )
    output = run / f"04_predicted_replay/{args.route}"
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError("predicted replay manifest exists; pass --resume")
    command = " ".join(shlex.quote(value) for value in sys.argv)
    with ledger_stage(
        run / "run_ledger.sqlite",
        stage="P3_PREDICTED_REPLAY",
        substage=args.route,
        route=args.route,
        pool=args.pool,
        command=command,
    ) as ledger:
        with exclusive_d1_flock(run, purpose=f"{args.route} predicted-mask replay"):
            gate, gate_path = _prepare_gate(args, run_dir=run)
            validate_live_resources(
                repo_root=ROOT,
                rank1_run_dir=args.rank1_run_dir,
                prefix=f"gtmask_{args.route}_predicted_replay_launch",
            )
            deployment = read_deployment_manifest(samples_path)
            deployment_ids = _validate_deployment(deployment)
            expected_selected = source_records[f"{args.route}_selected"]
            if artifact_record(selected_path) != expected_selected:
                raise ValueError(
                    "selected route configuration differs from unified authority"
                )
            if artifact_record(samples_path) != source_records["test_samples"]:
                raise ValueError(
                    "Test deployment manifest differs from unified authority"
                )
            config, selected = module.load_config(args.route, selected_path, False)
            checkpoint = Path(selected["finetuned_checkpoint"]).expanduser().resolve()
            rows: list[dict[str, object]] = []
            candidates: list[dict[str, object]] = []

            def save_sample(
                row: Mapping[str, Any], sample_candidates: Sequence[Mapping[str, Any]]
            ) -> None:
                rows.append(dict(row))
                candidates.extend(map(dict, sample_candidates))
                write_replay_sample(
                    run,
                    route=args.route,
                    sample_id=str(row["sample_id"]),
                    sample=row,
                    candidates=sample_candidates,
                )

            pending: list[Mapping[str, Any]] = []
            for deployment_row in deployment:
                sample_id = str(deployment_row["sample_id"])
                if args.resume:
                    try:
                        saved_row, saved_candidates = load_replay_sample(
                            run, route=args.route, sample_id=sample_id
                        )
                    except NativeReplayMissingError:
                        pending.append(deployment_row)
                    else:
                        rows.append(saved_row)
                        candidates.extend(saved_candidates)
                else:
                    pending.append(deployment_row)
            if pending:
                model, observed_sha, _ = load_finetuned_model(
                    checkpoint,
                    backend="grconvnet" if args.route == "g1" else "ggcnn2",
                    device="mps",
                )
                if observed_sha != selected["finetuned_checkpoint_sha256"]:
                    raise RuntimeError("loaded checkpoint hash differs")
                replay_label_free_samples(
                    pending,
                    route=args.route,
                    module=module,
                    model=model,
                    device=torch.device("mps"),
                    config=config,
                    sample_loader=CompactSampleLoader(),
                    sample_factory=BackendSample,
                    on_sample=save_sample,
                )
            observed_sample_ids = [str(row.get("sample_id", "")) for row in rows]
            if len(observed_sample_ids) != 7675 or set(observed_sample_ids) != set(
                deployment_ids
            ):
                raise RuntimeError("predicted replay sample coverage differs")
            replay_frame = pd.DataFrame(candidates)
            frozen_frame = pd.read_parquet(frozen_path)
            result = assert_exact_native_replay(
                replay_frame,
                frozen_frame,
                route=args.route,
            )
            candidate_path, sample_path = write_canonical_replay_frames(
                run,
                route=args.route,
                sample_ids=deployment_ids,
                candidates=replay_frame,
            )
            result.update(
                {
                    "branch": "predicted",
                    "sample_count": len(rows),
                    "no_output_count": 7675 - result["sample_count_with_output"],
                    "candidates": artifact_record(candidate_path),
                    "per_sample": artifact_record(sample_path),
                    "source_samples": artifact_record(samples_path),
                    "selected_config": artifact_record(selected_path),
                    "checkpoint": artifact_record(checkpoint),
                    "frozen_native_inference": artifact_record(module.__file__),
                    "frozen_candidates": artifact_record(frozen_path),
                    "resource_gate": artifact_record(gate_path),
                    "command": command,
                }
            )
            expected_no_output = 41 if args.route == "g1" else 13
            if result["no_output_count"] != expected_no_output:
                raise RuntimeError("predicted replay no-output count differs")
            result["content_sha256"] = canonical_sha256(result)
            if manifest_path.exists():
                existing = _load_existing_manifest(manifest_path)
                if existing != result:
                    raise RuntimeError("existing predicted replay manifest differs")
            atomic_json(manifest_path, result)
            ledger["artifact_path"] = str(manifest_path)
            ledger["artifact_sha256"] = sha256_file(manifest_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
