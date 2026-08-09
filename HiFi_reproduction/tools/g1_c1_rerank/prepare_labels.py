#!/usr/bin/env python3
"""Project offline labels into a physically separate candidate-label table.

Validation labels may be prepared during development.  Test labels require an
immutable primary lock and the explicit ``--unlock-test-once`` flag.  This
module never writes inference features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.g1_c1_safe_rerank.contracts import sha256_file  # noqa: E402
from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json as _safe_atomic_json  # noqa: E402
from src.grasping.g1_c1_safe_rerank.pools import (  # noqa: E402
    SOURCE_LABEL_COLUMNS,
    adapt_source_labels,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("G1", "C1", "all"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--unlock-test-once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    run = args.run_dir.expanduser().resolve()
    base = args.base_run.expanduser().resolve()
    split = str(args.split)
    backends = ("G1", "C1") if args.backend == "all" else (str(args.backend),)
    access_token: Path | None = None
    locked_source_labels: dict[Path, str] = {}
    if split == "test":
        if args.backend != "all":
            raise PermissionError(
                "formal test labels must be opened for G1 and C1 in one transaction"
            )
        lock = run / "08_lock" / "PRIMARY_METHOD_LOCK.json"
        if not lock.is_file() or not args.unlock_test_once:
            raise PermissionError(
                "test labels require PRIMARY_METHOD_LOCK.json and --unlock-test-once"
            )
        primary = json.loads(lock.read_text(encoding="utf-8"))
        if Path(str(primary.get("base_run", ""))).expanduser().resolve() != base:
            raise PermissionError("Test label source differs from the locked base run")
        prediction_marker = run / "09_formal_test" / "PREDICTIONS_COMPLETE.json"
        if not prediction_marker.is_file():
            raise PermissionError("formal Test labels require sealed label-free predictions")
        prediction_state = json.loads(prediction_marker.read_text(encoding="utf-8"))
        if (
            prediction_state.get("status") != "COMPLETE"
            or prediction_state.get("test_labels_accessed") is not False
            or prediction_state.get("primary_lock_sha256") != sha256_file(lock)
            or Path(str(prediction_state.get("base_run", ""))).resolve() != base
            or prediction_state.get("test_universe_sha256")
            != sha256_file(base / "manifests" / "test_samples.parquet")
        ):
            raise PermissionError("formal prediction marker is not bound to the current lock")
        for artifact in prediction_state.get("prediction_artifacts", []):
            path = Path(str(artifact["path"])).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != str(artifact["sha256"]):
                raise PermissionError(f"sealed formal prediction drift: {path}")
        required_lock = {
            "status",
            "locked_models",
            "candidate_artifacts",
            "feature_artifacts",
            "code_artifacts",
            "validation_selection_artifacts",
            "source_manifest_artifacts",
            "source_label_artifacts",
            "audit_artifacts",
            "ranker_calibrators",
            "lock_support_artifacts",
            "evaluator",
            "prediction_plan",
        }
        if primary.get("status") != "LOCKED" or not required_lock.issubset(primary):
            raise PermissionError("primary method lock is not immutable/LOCKED")
        for section in (
            "locked_models",
            "candidate_artifacts",
            "feature_artifacts",
            "code_artifacts",
            "validation_selection_artifacts",
            "source_manifest_artifacts",
            "source_label_artifacts",
            "audit_artifacts",
            "ranker_calibrators",
            "lock_support_artifacts",
        ):
            for artifact in primary[section]:
                path = Path(str(artifact["path"])).expanduser().resolve()
                if not path.is_file() or sha256_file(path) != str(artifact["sha256"]):
                    raise PermissionError(f"locked artifact hash mismatch: {path}")
        evaluator = primary["evaluator"]
        evaluator_path = Path(str(evaluator["path"])).expanduser().resolve()
        if not evaluator_path.is_file() or sha256_file(evaluator_path) != str(evaluator["sha256"]):
            raise PermissionError(f"locked evaluator hash mismatch: {evaluator_path}")
        plan = Path(str(primary["prediction_plan"]["path"])).expanduser().resolve()
        if not plan.is_file() or sha256_file(plan) != str(primary["prediction_plan"]["sha256"]):
            raise PermissionError("formal prediction plan hash mismatch")
        mirror = run / "08_lock" / "FORMAL_TEST_LOCK.json"
        if not mirror.is_file() or sha256_file(mirror) != sha256_file(lock):
            raise PermissionError("FORMAL_TEST_LOCK mirror differs from PRIMARY_METHOD_LOCK")
        locked_source_labels = {
            Path(str(item["path"])).expanduser().resolve(): str(item["sha256"])
            for item in primary["source_label_artifacts"]
        }
        access_token = run / "09_formal_test" / "TEST_ACCESS_TRANSACTION.json"
        access_token.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                access_token,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise RuntimeError("formal test access transaction was already claimed") from error
        opening = json.dumps(
            {
                "status": "OPENING",
                "primary_lock_sha256": sha256_file(lock),
                "predictions_complete_sha256": sha256_file(prediction_marker),
                "backend_opened": list(backends),
                "base_run": str(base),
                "test_universe_sha256": sha256_file(base / "manifests" / "test_samples.parquet"),
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(opening)
            stream.flush()
            os.fsync(stream.fileno())
        manifest_path = run / "MANIFEST.json"
        manifest_state = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest_state.get("formal_run_count", 0)) != 0:
            raise RuntimeError("formal_run_count is already non-zero")
        manifest_state["formal_run_count"] = 1
        manifest_state["formal_test_status"] = "OPENING"
        _safe_atomic_json(manifest_path, manifest_state)
    manifests: list[dict[str, Any]] = []
    phase = "validation/final" if split == "validation" else "formal_test"
    for backend in backends:
        source = base / phase / backend / "per_candidate_predictions.parquet"
        if split == "test":
            expected_sha = locked_source_labels.get(source.resolve())
            if expected_sha is None or sha256_file(source) != expected_sha:
                raise PermissionError(f"canonical Test label source drift: {source}")
        # This is the only read in this process and it projects only label identity
        # plus evaluator outcomes; no model feature table is reachable here.
        raw = pd.read_parquet(source, columns=list(SOURCE_LABEL_COLUMNS))
        labels = adapt_source_labels(raw, backend=backend)
        destination = run / ("07_validation" if split == "validation" else "09_formal_test") / f"{backend.lower()}_candidate_labels.parquet"
        _atomic_parquet(destination, labels)
        manifest = {
            "status": "COMPLETE",
            "split": split,
            "backend": backend,
            "candidate_rows": len(labels),
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "source_columns_loaded": list(SOURCE_LABEL_COLUMNS),
            "label_path": str(destination),
            "label_sha256": sha256_file(destination),
        }
        _atomic_json(destination.with_suffix(".manifest.json"), manifest)
        manifests.append(manifest)
    if split == "test":
        assert access_token is not None
        # The qualitative GT geometry is projected during the same one-time
        # transaction. Downstream galleries must consume this sealed local
        # artifact and may not reopen the source Test label manifest.
        analysis_source = base / "manifests" / "test_labels.parquet"
        expected_sha = locked_source_labels.get(analysis_source.resolve())
        if expected_sha is None or sha256_file(analysis_source) != expected_sha:
            raise PermissionError(f"canonical Test analysis source drift: {analysis_source}")
        analysis_labels = pd.read_parquet(
            analysis_source,
            columns=["sample_id", "gt_grasp_rectangles"],
        )
        analysis_destination = run / "09_formal_test" / "test_analysis_labels.parquet"
        _atomic_parquet(analysis_destination, analysis_labels)
        analysis_manifest = {
            "status": "COMPLETE",
            "split": "test",
            "backend": "ANALYSIS_ONLY",
            "candidate_rows": len(analysis_labels),
            "source_path": str(analysis_source),
            "source_sha256": sha256_file(analysis_source),
            "source_columns_loaded": ["sample_id", "gt_grasp_rectangles"],
            "label_path": str(analysis_destination),
            "label_sha256": sha256_file(analysis_destination),
        }
        _atomic_json(
            analysis_destination.with_suffix(".manifest.json"),
            analysis_manifest,
        )
        manifests.append(analysis_manifest)
        access_log_path = run / "09_formal_test" / "test_access.log"
        access_record = {
            "event": "FORMAL_TEST_LABELS_OPENED_ONCE",
            "backend_opened": list(backends),
            "base_run": str(base),
            "primary_lock_sha256": sha256_file(
                run / "08_lock" / "PRIMARY_METHOD_LOCK.json"
            ),
            "predictions_complete_sha256": sha256_file(
                run / "09_formal_test" / "PREDICTIONS_COMPLETE.json"
            ),
            "source_label_sha256": {
                str(path): value for path, value in sorted(locked_source_labels.items())
            },
        }
        try:
            descriptor = os.open(
                access_log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as error:
            raise RuntimeError("formal test access log already exists") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(access_record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        manifests.append(
            {
                "status": "COMPLETE",
                "split": "test",
                "backend": "ACCESS_AUDIT",
                "path": str(access_log_path),
                "sha256": sha256_file(access_log_path),
            }
        )
        _atomic_json(
            access_token,
            {
                "status": "CONSUMED",
                "primary_lock_sha256": sha256_file(run / "08_lock" / "PRIMARY_METHOD_LOCK.json"),
                "predictions_complete_sha256": sha256_file(
                    run / "09_formal_test" / "PREDICTIONS_COMPLETE.json"
                ),
                "backend_opened": list(backends),
                "base_run": str(base),
                "test_universe_sha256": sha256_file(base / "manifests" / "test_samples.parquet"),
                "label_artifacts": manifests,
            },
        )
        manifest_path = run / "MANIFEST.json"
        manifest_state = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_state["formal_test_status"] = "LABELS_OPENED_ONCE"
        _safe_atomic_json(manifest_path, manifest_state)
    print(json.dumps({"status": "COMPLETE", "artifacts": manifests}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
