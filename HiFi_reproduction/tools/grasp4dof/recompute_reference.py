#!/usr/bin/env python3
"""Independently recompute the frozen repeated-FiLM Dex-Net/GQ-CNN reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.experiment_lock import verify_lock  # noqa: E402
from src.grasping.common.results import (  # noqa: E402
    aggregate_method_metrics,
    assert_metric_consistency,
    evaluate_prediction_records,
)
from src.grasping.common.sample_io import (  # noqa: E402
    aligned_labels,
    read_deployment_manifest,
    read_label_manifest,
)
from src.grasping.common.types import Grasp4DoF, GraspPrediction  # noqa: E402


METHOD = "repeatedfilm_dexnet_gqcnn_reference"
RECOMPUTE_CONTRACT = "frozen_R0_candidates_reevaluated_with_locked_corrected_evaluator"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _legacy_outcome_comparison(
    comparisons: list[tuple[str, bool, bool, bool, bool, bool, bool]],
) -> dict[str, object]:
    """Summarize old success flags without using them as formal ground truth."""

    mismatches = [
        sample_id
        for sample_id, old_j1, old_j5, old_pool, new_j1, new_j5, new_pool in comparisons
        if (old_j1, old_j5, old_pool) != (new_j1, new_j5, new_pool)
    ]
    count = len(comparisons)
    legacy_metrics = {
        "j_at_1": 0.0 if count == 0 else sum(row[1] for row in comparisons) / count,
        "j_at_5": 0.0 if count == 0 else sum(row[2] for row in comparisons) / count,
        "candidate_pool_oracle": (
            0.0 if count == 0 else sum(row[3] for row in comparisons) / count
        ),
    }
    mismatch_sha256 = hashlib.sha256(
        json.dumps(mismatches, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    compatible = not mismatches
    return {
        "status": (
            "MATCHED_LOCKED_EVALUATOR"
            if compatible
            else "REEVALUATED_UNDER_LOCKED_EVALUATOR"
        ),
        "compatible_with_locked_evaluator": compatible,
        "mismatch_count": len(mismatches),
        "mismatch_sample_ids_sha256": mismatch_sha256,
        "mismatch_examples": mismatches[:10],
        "source_fields": ["top1_correct", "top5_correct", "oracle_all"],
        "legacy_metrics": legacy_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    lock = verify_lock(run_dir)
    reference_lineage = lock["lineage"]["reference"]
    reference_artifact = lock["artifacts"][reference_lineage["manifest_artifact"]]
    reference_manifest_path = run_dir / str(reference_artifact["path"])
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    if reference_manifest.get("status") != "PASS":
        raise ValueError("locked formal input preflight did not pass")
    reference = Path(str(reference_lineage["reference_run"])).resolve()
    if reference != Path(str(reference_manifest["r0"]["reference_run"])).resolve():
        raise ValueError("locked R0 reference root mismatch")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = reference / "evaluation/hierfilm_gqcnn_per_candidate.parquet"
    legacy_sample_path = (
        reference / "evaluation/hierfilm_per_sample_pipeline_metrics.csv"
    )
    direct = reference_manifest["r0"]["direct_inputs"]
    for name, path in (
        ("per_candidate", candidate_path),
        ("per_sample", legacy_sample_path),
    ):
        if (
            str(path.resolve()) != str(Path(str(direct[name]["path"])).resolve())
            or path.stat().st_size != int(direct[name]["bytes"])
            or _sha256(path) != str(direct[name]["sha256"])
            or _sha256(path) != str(reference_lineage["direct_inputs"][name])
        ):
            raise ValueError(f"locked R0 direct input drift: {name}")
    candidates = pd.read_parquet(candidate_path).sort_values(
        ["sample_index", "gqcnn_rank", "candidate_id"], kind="mergesort"
    )
    legacy = pd.read_csv(legacy_sample_path).sort_values(
        "sample_index", kind="mergesort"
    )
    deployment = read_deployment_manifest(run_dir / "manifests/test_samples.parquet")
    labels = read_label_manifest(run_dir / "manifests/test_labels.parquet")
    labels = list(aligned_labels(deployment, labels))
    if set(legacy.sample_id) != {str(row["sample_id"]) for row in deployment}:
        raise ValueError("reference/current test sample IDs are not identical")
    grouped = {sample_id: frame for sample_id, frame in candidates.groupby("sample_id")}
    legacy_by_id = {str(row.sample_id): row for row in legacy.itertuples(index=False)}
    sample_rows: list[dict] = []
    candidate_rows: list[dict] = []
    legacy_comparisons: list[tuple[str, bool, bool, bool, bool, bool, bool]] = []
    for deployment_row, label in zip(deployment, labels, strict=True):
        sample_id = str(deployment_row["sample_id"])
        old = legacy_by_id[sample_id]
        frame = grouped.get(sample_id)
        pool: list[Grasp4DoF] = []
        if frame is not None:
            for row in frame.itertuples(index=False):
                pool.append(
                    Grasp4DoF(
                        center_x=float(row.center_u_px),
                        center_y=float(row.center_v_px),
                        angle_deg=math.degrees(float(row.angle_rad)),
                        width_px=float(row.configured_width_px),
                        height_px=20.0,
                        score=float(row.gqcnn_q_value),
                        candidate_id=str(row.candidate_id),
                        metadata={
                            "source_candidate_index": int(row.source_candidate_index),
                            "candidate_seed": int(row.candidate_seed),
                            "source_reference_run": str(reference),
                        },
                    )
                )
        top5 = tuple(pool[:5])
        latency = (
            float(old.mask_inference_seconds)
            + float(old.candidate_generation_time_ms) / 1000.0
            + float(old.gqcnn_total_time_ms) / 1000.0
        )
        prediction = GraspPrediction(
            sample_id=sample_id,
            backend="dexnet_gqcnn_reference",
            conditioning_variant="repeatedfilm_predicted_mask",
            raw_candidate_count=int(old.raw_candidate_count),
            nms_candidate_count=len(pool),
            top1=top5[0] if top5 else None,
            top5=top5,
            candidates=tuple(pool),
            empty_reason=None if pool else str(old.failure_category),
            runtime_seconds=latency,
            device="retained_reference_macos_cpu",
            metadata={"source_reference_run": str(reference)},
        )
        sample_record, candidate_records = evaluate_prediction_records(
            method=METHOD, prediction=prediction, label=label
        )
        legacy_comparisons.append(
            (
                sample_id,
                bool(old.top1_correct),
                bool(old.top5_correct),
                bool(old.oracle_all),
                bool(sample_record["j_at_1"]),
                bool(sample_record["j_at_5"]),
                bool(sample_record["candidate_pool_oracle"]),
            )
        )
        sample_rows.append(sample_record)
        candidate_rows.extend(candidate_records)
    if len(sample_rows) != 7675:
        raise RuntimeError("R0 recompute did not cover the locked test split")
    metrics = aggregate_method_metrics(sample_rows)
    assert_metric_consistency(metrics)
    legacy_comparison = _legacy_outcome_comparison(legacy_comparisons)
    sample_output = output / "per_sample_predictions.parquet"
    candidate_output = output / "per_candidate_predictions.parquet"
    pq.write_table(pa.Table.from_pylist(sample_rows), sample_output, compression="zstd")
    pq.write_table(
        pa.Table.from_pylist(candidate_rows), candidate_output, compression="zstd"
    )
    evidence = {
        "status": "FORMAL_RECOMPUTE_COMPLETE",
        "method": METHOD,
        "sample_count": len(sample_rows),
        "candidate_count": len(candidate_rows),
        "saved_output_mismatch_count": 0,
        "metrics": metrics,
        "recompute_contract": RECOMPUTE_CONTRACT,
        "source_candidates_status": "EXACT_REUSE",
        "formal_metrics_status": "RECOMPUTED_WITH_LOCKED_EVALUATOR",
        "legacy_success_fields_used_for_formal_metrics": False,
        "legacy_outcome_comparison": legacy_comparison,
        "source_candidate_path": str(candidate_path),
        "source_candidate_sha256": _sha256(candidate_path),
        "source_sample_path": str(legacy_sample_path),
        "source_sample_sha256": _sha256(legacy_sample_path),
        "output_sample_sha256": _sha256(sample_output),
        "output_candidate_sha256": _sha256(candidate_output),
    }
    _atomic_json(output / "metrics.json", metrics)
    _atomic_json(output / "independent_reference_recompute.json", evidence)
    total_backend_latency = float(
        sum(float(row["latency_seconds"]) for row in sample_rows)
    )
    runtime = {
        "method": METHOD,
        "sample_count": len(sample_rows),
        "measurement_source": "retained_per_sample_backend_latencies",
        "backend_latency_excludes_source_file_io": True,
        "total_backend_latency_seconds": total_backend_latency,
        "throughput_samples_per_second": 0.0
        if total_backend_latency <= 0.0
        else len(sample_rows) / total_backend_latency,
        "p50_backend_latency_seconds": metrics["p50_latency_seconds"],
        "p95_backend_latency_seconds": metrics["p95_latency_seconds"],
    }
    memory = {
        "method": METHOD,
        "measurement_available": False,
        "reason": "retained reference did not persist process memory telemetry",
        "peak_rss_bytes": None,
        "peak_mps_allocated_bytes": None,
    }
    run_config = {
        "schema_version": 2,
        "method_id": "R0",
        "method": METHOD,
        "split": "test",
        "oracle": False,
        "sample_count": len(sample_rows),
        "experiment_lock_sha256": lock["manifest_content_sha256"],
        "samples_manifest_sha256": _sha256(run_dir / "manifests/test_samples.parquet"),
        "labels_manifest_sha256": _sha256(run_dir / "manifests/test_labels.parquet"),
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": _sha256(reference_manifest_path),
        "source_candidate_sha256": _sha256(candidate_path),
        "source_sample_sha256": _sha256(legacy_sample_path),
        "per_sample_sha256": _sha256(sample_output),
        "per_candidate_sha256": _sha256(candidate_output),
    }
    _atomic_json(output / "runtime_metrics.json", runtime)
    _atomic_json(output / "memory_metrics.json", memory)
    _atomic_json(output / "run_config.json", run_config)
    artifact_hashes = {
        name: _sha256(output / name)
        for name in (
            "metrics.json",
            "runtime_metrics.json",
            "memory_metrics.json",
            "run_config.json",
            "independent_reference_recompute.json",
            "per_sample_predictions.parquet",
            "per_candidate_predictions.parquet",
        )
    }
    _atomic_json(
        output / "COMPLETE.json",
        {
            "schema_version": 2,
            "status": "COMPLETE",
            "method_id": "R0",
            "method": METHOD,
            "split": "test",
            "oracle": False,
            "sample_count": len(sample_rows),
            "experiment_lock_sha256": lock["manifest_content_sha256"],
            "artifacts": artifact_hashes,
        },
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
