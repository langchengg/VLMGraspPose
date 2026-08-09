"""Machine-readable per-candidate evaluation and aggregate 4-DoF metrics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np

from .candidate_decoder import rank_candidates
from .evaluator import EvaluatorConfig, evaluate_ocid_predictions
from .types import GraspPrediction


def evaluate_prediction_records(
    *,
    method: str,
    prediction: GraspPrediction,
    label: Mapping[str, Any],
    config: EvaluatorConfig = EvaluatorConfig(),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one stored prediction without consulting inference inputs."""

    if str(label["sample_id"]) != prediction.sample_id:
        raise ValueError("prediction/label sample ID mismatch")
    # Keep the serialized rows and the evaluator on exactly the same canonical
    # ordering even when an adapter supplies an otherwise valid unsorted pool.
    pool = rank_candidates(prediction.candidates or prediction.top5)
    evaluated = evaluate_ocid_predictions(pool, label["gt_grasp_rectangles"], config)
    candidate_rows: list[dict[str, Any]] = []
    for rank, (candidate, outcome) in enumerate(zip(pool, evaluated.candidates), 1):
        candidate_rows.append(
            {
                "method": method,
                "sample_id": prediction.sample_id,
                "scene_id": str(label["scene_id"]),
                "candidate_id": candidate.candidate_id,
                "rank": rank,
                "center_x": candidate.center_x,
                "center_y": candidate.center_y,
                "angle_deg": candidate.angle_deg,
                "width_px": candidate.width_px,
                "height_px": config.fixed_height_px,
                "score": candidate.score,
                "candidate_success": outcome.candidate_success,
                "best_gt_index": outcome.best_gt_index,
                "best_rectangle_iou": outcome.best_rectangle_iou,
                "best_angle_difference_deg": outcome.best_angle_difference_deg,
                "candidate_metadata_json": json.dumps(
                    dict(candidate.metadata), sort_keys=True, separators=(",", ":")
                ),
                "pairwise_json": json.dumps(
                    [asdict(pair) for pair in outcome.pairwise],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    top1_row = candidate_rows[0] if candidate_rows else None
    sample_row = {
        "method": method,
        "sample_id": prediction.sample_id,
        "scene_id": str(label["scene_id"]),
        "conditioning": prediction.conditioning_variant,
        "device": prediction.device,
        "j_at_1": bool(evaluated.j_at_1),
        "j_at_5": bool(any(row["candidate_success"] for row in candidate_rows[:5])),
        "candidate_pool_oracle": bool(
            any(row["candidate_success"] for row in candidate_rows)
        ),
        "first_valid_rank": evaluated.first_valid_rank,
        "reciprocal_rank": evaluated.reciprocal_rank,
        "non_empty": bool(pool),
        "raw_candidate_count": prediction.raw_candidate_count,
        "nms_candidate_count": prediction.nms_candidate_count,
        "empty_reason": prediction.empty_reason,
        "latency_seconds": prediction.runtime_seconds,
        "top1_candidate_id": None if top1_row is None else top1_row["candidate_id"],
        "top1_json": None
        if not pool
        else json.dumps(pool[0].to_dict(rank=1), sort_keys=True, separators=(",", ":")),
        "top1_rectangle_iou": None
        if top1_row is None
        else top1_row["best_rectangle_iou"],
        "top1_angle_difference_deg": None
        if top1_row is None
        else top1_row["best_angle_difference_deg"],
        "top5_candidate_ids_json": json.dumps(
            [row["candidate_id"] for row in candidate_rows[:5]], separators=(",", ":")
        ),
        "top5_json": json.dumps(
            [item.to_dict(rank=rank) for rank, item in enumerate(pool[:5], 1)],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "prediction_metadata_json": json.dumps(
            dict(prediction.metadata), sort_keys=True, separators=(",", ":")
        ),
    }
    return sample_row, candidate_rows


def aggregate_method_metrics(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate all-sample primary metrics; empty samples remain failures."""

    if not sample_rows:
        raise ValueError("cannot aggregate an empty method")
    methods = {str(row["method"]) for row in sample_rows}
    if len(methods) != 1:
        raise ValueError("sample rows contain multiple methods")
    count = len(sample_rows)
    non_empty_rows = [row for row in sample_rows if bool(row["non_empty"])]
    positive_ranks = [
        int(row["first_valid_rank"])
        for row in sample_rows
        if row.get("first_valid_rank") is not None
    ]
    latencies = np.asarray(
        [float(row["latency_seconds"]) for row in sample_rows], dtype=np.float64
    )
    if not np.all(np.isfinite(latencies)) or np.any(latencies < 0.0):
        raise ValueError("latencies must be finite and non-negative")

    def mean_bool(key: str, rows: Sequence[Mapping[str, Any]]) -> float:
        return 0.0 if not rows else float(np.mean([bool(row[key]) for row in rows]))

    return {
        "method": next(iter(methods)),
        "sample_count": count,
        "j_at_1": mean_bool("j_at_1", sample_rows),
        "j_at_5": mean_bool("j_at_5", sample_rows),
        "recall_at_5": mean_bool("j_at_5", sample_rows),
        "candidate_pool_oracle": mean_bool("candidate_pool_oracle", sample_rows),
        "mrr": float(np.mean([float(row["reciprocal_rank"]) for row in sample_rows])),
        "mean_first_valid_rank": None
        if not positive_ranks
        else float(np.mean(positive_ranks)),
        "median_first_valid_rank": None
        if not positive_ranks
        else float(np.median(positive_ranks)),
        "non_empty_rate": len(non_empty_rows) / count,
        "no_grasp_rate": 1.0 - len(non_empty_rows) / count,
        "non_empty_j_at_1": mean_bool("j_at_1", non_empty_rows),
        "non_empty_j_at_5": mean_bool("j_at_5", non_empty_rows),
        "mean_raw_candidates": float(
            np.mean([int(row["raw_candidate_count"]) for row in sample_rows])
        ),
        "mean_nms_candidates": float(
            np.mean([int(row["nms_candidate_count"]) for row in sample_rows])
        ),
        "p50_latency_seconds": float(np.percentile(latencies, 50)),
        "p95_latency_seconds": float(np.percentile(latencies, 95)),
    }


def assert_metric_consistency(metrics: Mapping[str, Any]) -> None:
    """Fail if basic metric inequalities or all-sample accounting drift."""

    rates = (
        "j_at_1",
        "j_at_5",
        "candidate_pool_oracle",
        "mrr",
        "non_empty_rate",
        "no_grasp_rate",
    )
    if any(
        not math.isfinite(float(metrics[key])) or not 0.0 <= float(metrics[key]) <= 1.0
        for key in rates
    ):
        raise ValueError("aggregate rates must be finite in [0, 1]")
    if float(metrics["j_at_1"]) > float(metrics["j_at_5"]):
        raise ValueError("J@1 cannot exceed J@5")
    if float(metrics["j_at_5"]) > float(metrics["candidate_pool_oracle"]):
        raise ValueError("J@5 cannot exceed candidate-pool oracle")
    if not math.isclose(
        float(metrics["non_empty_rate"]) + float(metrics["no_grasp_rate"]),
        1.0,
        abs_tol=1e-12,
    ):
        raise ValueError("non-empty/no-grasp accounting mismatch")


AGGREGATE_METRIC_FIELDS = (
    "sample_count",
    "j_at_1",
    "j_at_5",
    "recall_at_5",
    "candidate_pool_oracle",
    "mrr",
    "mean_first_valid_rank",
    "median_first_valid_rank",
    "non_empty_rate",
    "no_grasp_rate",
    "non_empty_j_at_1",
    "non_empty_j_at_5",
    "mean_raw_candidates",
    "mean_nms_candidates",
    "p50_latency_seconds",
    "p95_latency_seconds",
)


def assert_aggregate_matches_sample_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    declared_metrics: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Recompute every aggregate and reject a stale or fabricated summary.

    The comparison is deliberately independent of candidate correctness fields:
    it uses only the already persisted per-sample outcomes. Candidate geometry is
    independently re-evaluated later by the formal recomputation stage.
    """

    recomputed = aggregate_method_metrics(sample_rows)
    if str(declared_metrics.get("method")) != str(recomputed["method"]):
        raise ValueError("declared aggregate method does not match per-sample rows")
    missing = sorted(set(AGGREGATE_METRIC_FIELDS) - set(declared_metrics))
    if missing:
        raise ValueError(f"declared aggregate metrics are incomplete: {missing}")
    for field in AGGREGATE_METRIC_FIELDS:
        observed = declared_metrics[field]
        expected = recomputed[field]
        observed_missing = observed is None or (
            isinstance(observed, (float, np.floating)) and bool(np.isnan(observed))
        )
        expected_missing = expected is None or (
            isinstance(expected, (float, np.floating)) and bool(np.isnan(expected))
        )
        if observed_missing or expected_missing:
            if observed_missing is not expected_missing:
                raise ValueError(f"aggregate metric mismatch: {field}")
            continue
        if field == "sample_count":
            if int(observed) != int(expected):
                raise ValueError(f"aggregate metric mismatch: {field}")
            continue
        observed_number = float(observed)
        expected_number = float(expected)
        if not math.isfinite(observed_number) or not math.isclose(
            observed_number,
            expected_number,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(f"aggregate metric mismatch: {field}")
    assert_metric_consistency(declared_metrics)
    return recomputed
