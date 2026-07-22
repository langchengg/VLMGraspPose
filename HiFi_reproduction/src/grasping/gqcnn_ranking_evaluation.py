"""Evaluate stored GQ-CNN q-value rankings on frozen Dex-Net candidates.

This module never imports TensorFlow, calls a sampler, or changes a candidate.
It treats ``candidates.npz`` as the pose source of truth and accepts q-values
only after the scored NPZ/JSON are cross-checked against that frozen source.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ruamel.yaml import YAML

from .camera_geometry import T_CAMERA_GRASP_FIXED_APPROACH_KEY
from .geometric_ranker import (
    evaluate_planar_annotation_consistency,
    load_frozen_candidates,
    load_ocid_vlg_annotation_record,
    make_candidate_evaluation_rectangle,
    make_ocid_vlg_evaluation_rectangles,
    save_strict_json,
    sha256_file,
)


BASELINE_NAME = "GQ-CNN q-value ranking on frozen Dex-Net candidates"
METRIC_NAME = "2D consistency with OCID-VLG planar grasp annotations"
POSE_ARRAY_KEYS = (
    "center_uv",
    "center_depth_m",
    "center_camera_xyz_m",
    "angle_rad",
    "width_m",
    "width_px",
    "endpoints_uv",
    T_CAMERA_GRASP_FIXED_APPROACH_KEY,
)
PER_SAMPLE_FIELDS = (
    "sample_id",
    "query",
    "candidate_count",
    "finite_q_count",
    "top1_candidate_id",
    "top1_q_value",
    "top1_consistent",
    "top5_consistent",
    "first_valid_rank",
    "first_valid_candidate_id",
    "first_valid_q_value",
    "total_valid_candidate_count",
    "best_valid_q_value",
    "candidate_generation_success",
    "failure_type",
    "failure_reason",
    "annotation_count",
    "consistency_iou_threshold",
    "consistency_angle_threshold_deg",
)
SUMMARY_FIELDS = (
    "method",
    "evaluable_samples",
    "top1_numerator",
    "top1_denominator",
    "top1_decimal",
    "top1_percentage",
    "top5_numerator",
    "top5_denominator",
    "top5_decimal",
    "top5_percentage",
    "mean_valid_grasp_rank",
    "median_valid_grasp_rank",
    "minimum_first_valid_rank",
    "maximum_first_valid_rank",
    "population_std_first_valid_rank",
    "valid_rank_samples_included",
    "valid_rank_samples_excluded",
    "top1_failures",
    "top5_ranking_failures",
    "candidate_generation_failures",
    "invalid_data_samples",
)


class EvaluationDataError(ValueError):
    """A classified input-integrity or annotation error."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def load_evaluation_config(path: Path | str) -> dict[str, Any]:
    config = YAML(typ="safe").load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a mapping")
    required = {
        "metric_name",
        "predicted_rectangle_height_px",
        "ground_truth_rectangle_height_px",
        "ground_truth_width_clip_px",
        "angle_threshold_deg",
        "iou_threshold",
        "top_k",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"evaluation config is missing {missing}")
    if str(config["metric_name"]) != METRIC_NAME:
        raise ValueError(f"metric_name must be exactly {METRIC_NAME!r}")
    if int(config["top_k"]) <= 0:
        raise ValueError("top_k must be positive")
    result = dict(config)
    result["label"] = result["metric_name"]
    return result


def _read_candidate_payload(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationDataError("invalid_gqcnn_scores", f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise EvaluationDataError("invalid_gqcnn_scores", f"{path} has no candidates list")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise EvaluationDataError("invalid_gqcnn_scores", f"{path} metadata is not an object")
    return [dict(item) for item in payload["candidates"]], dict(metadata)


def _load_npz_arrays(path: Path, required: Sequence[str], category: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(required) - set(archive.files))
            if missing:
                raise EvaluationDataError(category, f"{path} is missing arrays {missing}")
            return {name: np.asarray(archive[name]) for name in required}
    except EvaluationDataError:
        raise
    except Exception as error:
        raise EvaluationDataError(category, f"cannot load {path}: {error}") from error


def _record_pose_value(record: Mapping[str, Any], name: str) -> Any:
    if name == "center_uv":
        return record.get("center_uv", [record.get("center_u_px"), record.get("center_v_px")])
    if name == "endpoints_uv":
        return record.get(
            "endpoints_uv",
            [record.get("endpoint_1_uv"), record.get("endpoint_2_uv")],
        )
    return record.get(name)


def _assert_record_pose_matches(
    record: Mapping[str, Any],
    source_arrays: Mapping[str, np.ndarray],
    source_index: int,
    *,
    label: str,
) -> None:
    for name in POSE_ARRAY_KEYS:
        actual = source_arrays[name][source_index]
        try:
            candidate = np.asarray(_record_pose_value(record, name), dtype=actual.dtype)
        except (TypeError, ValueError) as error:
            raise EvaluationDataError(
                "mapping_or_geometry_error",
                f"{label} has invalid {name}",
            ) from error
        if actual.shape != candidate.shape or not np.array_equal(actual, candidate, equal_nan=True):
            raise EvaluationDataError(
                "mapping_or_geometry_error",
                f"{label} pose mismatch for {name}",
            )


def rank_stored_q_values(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort raw finite q-values descending, breaking exact ties by ID."""

    ranked: list[dict[str, Any]] = []
    ids = []
    for source in records:
        record = dict(source)
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise EvaluationDataError("invalid_gqcnn_scores", "candidate ID is missing")
        ids.append(candidate_id)
        try:
            q_value = float(record["gqcnn_q_value"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationDataError(
                "invalid_gqcnn_scores",
                f"candidate {candidate_id} has no numeric q-value",
            ) from error
        if not math.isfinite(q_value):
            raise EvaluationDataError(
                "invalid_gqcnn_scores",
                f"candidate {candidate_id} has non-finite q-value",
            )
        record["gqcnn_q_value"] = q_value
        ranked.append(record)
    if len(set(ids)) != len(ids):
        raise EvaluationDataError("mapping_or_geometry_error", "candidate IDs are not unique")
    ranked.sort(key=lambda item: (-item["gqcnn_q_value"], item["candidate_id"]))
    for rank, record in enumerate(ranked, start=1):
        record["gqcnn_rank"] = rank
    return ranked


def load_verified_scored_candidates(sample_dir: Path | str) -> dict[str, Any]:
    """Cross-check stored q-values and poses, then reconstruct exact q ranking."""

    sample_dir = Path(sample_dir)
    metadata_path = sample_dir / "metadata.json"
    try:
        sample_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationDataError("mapping_or_geometry_error", f"invalid {metadata_path}: {error}") from error
    source_npz = sample_dir / "candidates.npz"
    source_json = sample_dir / "candidates.json"
    try:
        source_records, candidate_metadata, source_hashes = load_frozen_candidates(source_npz, source_json)
    except Exception as error:
        raise EvaluationDataError("mapping_or_geometry_error", str(error)) from error
    sample_id = str(sample_metadata.get("sample_id", sample_dir.name))
    query = sample_metadata.get("query")
    if sample_id != sample_dir.name:
        raise EvaluationDataError(
            "mapping_or_geometry_error",
            f"directory {sample_dir.name} disagrees with metadata sample_id {sample_id}",
        )
    for field, expected in (("sample_id", sample_id), ("query", query)):
        if candidate_metadata and candidate_metadata.get(field) != expected:
            raise EvaluationDataError(
                "mapping_or_geometry_error",
                f"candidate metadata {field} disagrees with sample metadata",
            )

    source_arrays = _load_npz_arrays(source_npz, POSE_ARRAY_KEYS, "mapping_or_geometry_error")
    scored_npz_path = sample_dir / "gqcnn_scored_candidates.npz"
    scored_json_path = sample_dir / "gqcnn_scored_candidates.json"
    scored_required = ("candidate_id", "gqcnn_q_value", "gqcnn_rank") + POSE_ARRAY_KEYS
    scored_arrays = _load_npz_arrays(scored_npz_path, scored_required, "invalid_gqcnn_scores")
    source_ids = [record["candidate_id"] for record in source_records]
    scored_ids = [str(value) for value in scored_arrays["candidate_id"].tolist()]
    if scored_ids != source_ids:
        raise EvaluationDataError(
            "mapping_or_geometry_error",
            "scored NPZ candidate IDs/order differ from frozen source order",
        )
    count = len(source_ids)
    q_values = np.asarray(scored_arrays["gqcnn_q_value"], dtype=np.float64)
    if q_values.shape != (count,) or not np.all(np.isfinite(q_values)):
        raise EvaluationDataError(
            "invalid_gqcnn_scores",
            f"expected {count} finite q-values, got shape {q_values.shape}",
        )
    stored_ranks = np.asarray(scored_arrays["gqcnn_rank"])
    if stored_ranks.shape != (count,) or set(stored_ranks.astype(int).tolist()) != set(range(1, count + 1)):
        raise EvaluationDataError("invalid_gqcnn_scores", "stored GQ-CNN ranks are not a 1..N permutation")
    for name in POSE_ARRAY_KEYS:
        if (
            source_arrays[name].shape != scored_arrays[name].shape
            or source_arrays[name].dtype != scored_arrays[name].dtype
            or not np.array_equal(source_arrays[name], scored_arrays[name], equal_nan=True)
        ):
            raise EvaluationDataError(
                "mapping_or_geometry_error",
                f"scored NPZ differs from frozen source pose array {name}",
            )

    scored_json_records, scored_metadata = _read_candidate_payload(scored_json_path)
    by_json_id = {record.get("candidate_id"): record for record in scored_json_records}
    if len(by_json_id) != count or set(by_json_id) != set(source_ids):
        raise EvaluationDataError(
            "mapping_or_geometry_error",
            "scored JSON candidate IDs differ from frozen source IDs",
        )
    if scored_metadata.get("sample_id") != sample_id or scored_metadata.get("query") != query:
        raise EvaluationDataError("mapping_or_geometry_error", "scored JSON sample/query mapping disagrees")

    q_by_id = dict(zip(source_ids, q_values.astype(float).tolist()))
    stored_rank_by_id = dict(zip(source_ids, stored_ranks.astype(int).tolist()))
    reconstructed = []
    for index, source in enumerate(source_records):
        candidate_id = source_ids[index]
        json_record = by_json_id[candidate_id]
        _assert_record_pose_matches(
            json_record,
            source_arrays,
            index,
            label=f"scored JSON candidate {candidate_id}",
        )
        try:
            json_q = float(json_record["gqcnn_q_value"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationDataError(
                "invalid_gqcnn_scores",
                f"scored JSON candidate {candidate_id} has invalid q-value",
            ) from error
        if not math.isfinite(json_q) or json_q != q_by_id[candidate_id]:
            raise EvaluationDataError(
                "invalid_gqcnn_scores",
                f"JSON/NPZ q-value mismatch for {candidate_id}",
            )
        record = dict(source)
        record["gqcnn_q_value"] = q_by_id[candidate_id]
        record["gqcnn_model_name"] = json_record.get("gqcnn_model_name")
        reconstructed.append(record)

    ranked = rank_stored_q_values(reconstructed)
    reconstructed_rank_by_id = {item["candidate_id"]: item["gqcnn_rank"] for item in ranked}
    stored_rank_mismatches = [
        {
            "candidate_id": candidate_id,
            "stored_rank": stored_rank_by_id[candidate_id],
            "raw_q_reconstructed_rank": reconstructed_rank_by_id[candidate_id],
        }
        for candidate_id in source_ids
        if stored_rank_by_id[candidate_id] != reconstructed_rank_by_id[candidate_id]
    ]
    stored_rank_matches = not stored_rank_mismatches
    return {
        "sample_dir": sample_dir,
        "sample_metadata": sample_metadata,
        "source_records": source_records,
        "ranked": ranked,
        "source_hashes": {
            **source_hashes,
            "gqcnn_scored_candidates_npz_sha256": sha256_file(scored_npz_path),
            "gqcnn_scored_candidates_json_sha256": sha256_file(scored_json_path),
            "gqcnn_scored_candidates_csv_sha256": sha256_file(
                sample_dir / "gqcnn_scored_candidates.csv"
            ),
        },
        "finite_q_count": int(np.count_nonzero(np.isfinite(q_values))),
        "storage_order": "frozen source candidate order; not q-value rank order",
        "stored_rank_matches_exact_q_reconstruction": stored_rank_matches,
        "stored_rank_mismatch_count": len(stored_rank_mismatches),
        "stored_rank_mismatches": stored_rank_mismatches,
    }


def classify_failures(
    *,
    candidate_count: int,
    top1_consistent: bool,
    top5_consistent: bool,
    first_valid_rank: int | None,
    data_error: str | None = None,
) -> list[str]:
    if data_error is not None:
        return [data_error]
    if candidate_count == 0:
        return ["no_candidates"]
    if first_valid_rank is None:
        return ["candidate_generation_failure"]
    failures = []
    if not top1_consistent:
        failures.append("top1_ranking_failure")
    if not top5_consistent:
        failures.append("top5_ranking_failure")
    return failures or ["none"]


def _failure_reason(categories: Sequence[str], first_valid_rank: int | None) -> str:
    reasons = {
        "none": "",
        "top1_ranking_failure": "rank 1 is inconsistent but a consistent candidate exists later",
        "top5_ranking_failure": f"no consistent candidate in Top-5; first occurs at rank {first_valid_rank}",
        "candidate_generation_failure": "no consistent candidate exists anywhere in the frozen candidate set",
        "no_candidates": "frozen candidate set is empty",
        "invalid_gqcnn_scores": "stored q-values are missing, non-finite, or inconsistent",
        "annotation_unavailable": "no usable OCID-VLG planar grasp annotations",
        "mapping_or_geometry_error": "sample, annotation, candidate ID, or pose mapping is inconsistent",
    }
    return "; ".join(reasons[item] for item in categories if reasons.get(item))


def _metrics_from_consistency(
    *,
    sample_id: str,
    query: str,
    ranked: Sequence[Mapping[str, Any]],
    finite_q_count: int,
    evaluation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    per_candidate = evaluation["per_candidate"]
    valid = [item for item in per_candidate if item["rectangle_match"]]
    valid_by_id = {item["candidate_id"]: item for item in valid}
    ranked_by_id = {item["candidate_id"]: item for item in ranked}
    first_valid_rank = evaluation["first_matching_rank"]
    first_valid = None
    if first_valid_rank is not None:
        first_valid = ranked[int(first_valid_rank) - 1]
    top1 = ranked[0] if ranked else None
    top1_consistent = bool(evaluation["top1_rectangle_accuracy"])
    top5_consistent = bool(evaluation["topk_recall"])
    categories = classify_failures(
        candidate_count=len(ranked),
        top1_consistent=top1_consistent,
        top5_consistent=top5_consistent,
        first_valid_rank=first_valid_rank,
    )
    best_valid_q = None
    if valid:
        best_valid_q = max(
            float(ranked_by_id[item["candidate_id"]]["gqcnn_q_value"])
            for item in valid
        )
    return {
        "sample_id": sample_id,
        "query": query,
        "candidate_count": len(ranked),
        "finite_q_count": finite_q_count,
        "top1_candidate_id": None if top1 is None else top1["candidate_id"],
        "top1_q_value": None if top1 is None else float(top1["gqcnn_q_value"]),
        "top1_consistent": top1_consistent,
        "top5_consistent": top5_consistent,
        "first_valid_rank": first_valid_rank,
        "first_valid_candidate_id": None if first_valid is None else first_valid["candidate_id"],
        "first_valid_q_value": None if first_valid is None else float(first_valid["gqcnn_q_value"]),
        "total_valid_candidate_count": len(valid),
        "best_valid_q_value": best_valid_q,
        "candidate_generation_success": first_valid_rank is not None,
        "failure_categories": categories,
        "failure_type": "|".join(categories),
        "failure_reason": _failure_reason(categories, first_valid_rank),
        "annotation_count": int(evaluation["annotation_count"]),
        "consistency_iou_threshold": float(config["iou_threshold"]),
        "consistency_angle_threshold_deg": float(config["angle_threshold_deg"]),
        "valid_candidate_ids": sorted(valid_by_id),
    }


def evaluate_sample(
    sample_dir: Path | str,
    annotation_file: Path | str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    verified = load_verified_scored_candidates(sample_dir)
    metadata = verified["sample_metadata"]
    sample_id = str(metadata["sample_id"])
    query = str(metadata["query"])
    question_index = int(metadata["question_index"])
    try:
        annotation = load_ocid_vlg_annotation_record(annotation_file, question_index)
    except Exception as error:
        raise EvaluationDataError("annotation_unavailable", str(error)) from error
    grasps = annotation.get("grasps")
    if not isinstance(grasps, list) or not grasps:
        raise EvaluationDataError(
            "annotation_unavailable",
            f"question_index {question_index} has no grasp rectangles",
        )
    if annotation.get("question") != query:
        raise EvaluationDataError("mapping_or_geometry_error", "annotation query disagrees")
    if annotation.get("image_filename") != metadata.get("scene_id"):
        raise EvaluationDataError("mapping_or_geometry_error", "annotation scene disagrees")

    ranked = verified["ranked"]
    evaluation = evaluate_planar_annotation_consistency(
        ranked,
        grasps,
        config,
        rank_field="gqcnn_rank",
    )
    by_id = {item["candidate_id"]: item for item in evaluation["per_candidate"]}
    for record in ranked:
        record["ocid_vlg_2d_consistency"] = by_id[record["candidate_id"]]
    metrics = _metrics_from_consistency(
        sample_id=sample_id,
        query=query,
        ranked=ranked,
        finite_q_count=verified["finite_q_count"],
        evaluation=evaluation,
        config=config,
    )

    geometric_path = Path(sample_dir) / "geometrically_ranked_candidates.json"
    try:
        geometric_payload = json.loads(geometric_path.read_text(encoding="utf-8"))
        geometric = [dict(item) for item in geometric_payload["candidates"]]
    except Exception as error:
        raise EvaluationDataError(
            "mapping_or_geometry_error",
            f"cannot load geometric reference ranking: {error}",
        ) from error
    if {item.get("candidate_id") for item in geometric} != {
        item["candidate_id"] for item in verified["source_records"]
    }:
        raise EvaluationDataError("mapping_or_geometry_error", "geometric reference ID set differs")
    geometric_evaluation = evaluate_planar_annotation_consistency(
        geometric,
        grasps,
        config,
        rank_field="geometric_rank",
    )
    geometric_metrics = {
        "sample_id": sample_id,
        "candidate_count": len(geometric),
        "top1_consistent": bool(geometric_evaluation["top1_rectangle_accuracy"]),
        "top5_consistent": bool(geometric_evaluation["topk_recall"]),
        "first_valid_rank": geometric_evaluation["first_matching_rank"],
    }
    verified["source_hashes"]["geometrically_ranked_candidates_json_sha256"] = sha256_file(
        geometric_path
    )
    return {
        **verified,
        "annotation": annotation,
        "annotation_count": len(grasps),
        "annotation_file": str(Path(annotation_file).resolve()),
        "evaluation": evaluation,
        "metrics": metrics,
        "geometric_reference_metrics": geometric_metrics,
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]], *, method: str) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("data_valid", True)]
    denominator = len(evaluable)
    top1_success = [row["sample_id"] for row in evaluable if row["top1_consistent"]]
    top1_failed = [row["sample_id"] for row in evaluable if not row["top1_consistent"]]
    top5_success = [row["sample_id"] for row in evaluable if row["top5_consistent"]]
    top5_failed = [row["sample_id"] for row in evaluable if not row["top5_consistent"]]
    first_ranks = [int(row["first_valid_rank"]) for row in evaluable if row["first_valid_rank"] is not None]
    excluded = [row["sample_id"] for row in evaluable if row["first_valid_rank"] is None]
    penalized = [
        int(row["first_valid_rank"])
        if row["first_valid_rank"] is not None
        else int(row["candidate_count"]) + 1
        for row in evaluable
    ]
    top1_count = len(top1_success)
    top5_count = len(top5_success)
    return {
        "method": method,
        "evaluable_samples": denominator,
        "evaluable_sample_ids": [row["sample_id"] for row in evaluable],
        "top1_consistency": {
            "numerator": top1_count,
            "denominator": denominator,
            "decimal": None if denominator == 0 else top1_count / denominator,
            "percentage": None if denominator == 0 else 100.0 * top1_count / denominator,
            "successful_sample_ids": top1_success,
            "failed_sample_ids": top1_failed,
        },
        "top5_recall": {
            "numerator": top5_count,
            "denominator": denominator,
            "decimal": None if denominator == 0 else top5_count / denominator,
            "percentage": None if denominator == 0 else 100.0 * top5_count / denominator,
            "successful_sample_ids": top5_success,
            "failed_sample_ids": top5_failed,
        },
        "first_valid_rank": {
            "mean": None if not first_ranks else float(statistics.mean(first_ranks)),
            "median": None if not first_ranks else float(statistics.median(first_ranks)),
            "minimum": None if not first_ranks else min(first_ranks),
            "maximum": None if not first_ranks else max(first_ranks),
            "population_standard_deviation": None
            if not first_ranks
            else float(statistics.pstdev(first_ranks)),
            "samples_included": len(first_ranks),
            "samples_excluded": len(excluded),
            "excluded_sample_ids": excluded,
            "per_sample": {row["sample_id"]: row["first_valid_rank"] for row in evaluable},
        },
        "penalized_mean_rank": None if not penalized else float(statistics.mean(penalized)),
        "failures": {
            "samples_with_any_failure": sum(
                row.get("failure_type", "none") != "none" for row in rows
            ),
            "top1_failures": len(top1_failed),
            "top1_failure_sample_ids": top1_failed,
            "top5_ranking_failures": sum(
                "top5_ranking_failure" in row.get("failure_categories", []) for row in evaluable
            ),
            "top5_ranking_failure_sample_ids": [
                row["sample_id"]
                for row in evaluable
                if "top5_ranking_failure" in row.get("failure_categories", [])
            ],
            "candidate_generation_failures": sum(
                row["first_valid_rank"] is None for row in evaluable
            ),
            "candidate_generation_failure_sample_ids": excluded,
            "invalid_data_samples": len(rows) - denominator,
            "invalid_data_sample_ids": [row["sample_id"] for row in rows if not row.get("data_valid", True)],
            "taxonomy_counts": {
                category: sum(category in row.get("failure_categories", []) for row in rows)
                for category in (
                    "top1_ranking_failure",
                    "top5_ranking_failure",
                    "candidate_generation_failure",
                    "no_candidates",
                    "invalid_gqcnn_scores",
                    "annotation_unavailable",
                    "mapping_or_geometry_error",
                )
            },
        },
    }


def summary_csv_row(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    first = aggregate["first_valid_rank"]
    failures = aggregate["failures"]
    top1 = aggregate["top1_consistency"]
    top5 = aggregate["top5_recall"]
    return {
        "method": aggregate["method"],
        "evaluable_samples": aggregate["evaluable_samples"],
        "top1_numerator": top1["numerator"],
        "top1_denominator": top1["denominator"],
        "top1_decimal": top1["decimal"],
        "top1_percentage": top1["percentage"],
        "top5_numerator": top5["numerator"],
        "top5_denominator": top5["denominator"],
        "top5_decimal": top5["decimal"],
        "top5_percentage": top5["percentage"],
        "mean_valid_grasp_rank": first["mean"],
        "median_valid_grasp_rank": first["median"],
        "minimum_first_valid_rank": first["minimum"],
        "maximum_first_valid_rank": first["maximum"],
        "population_std_first_valid_rank": first["population_standard_deviation"],
        "valid_rank_samples_included": first["samples_included"],
        "valid_rank_samples_excluded": first["samples_excluded"],
        "top1_failures": failures["top1_failures"],
        "top5_ranking_failures": failures["top5_ranking_failures"],
        "candidate_generation_failures": failures["candidate_generation_failures"],
        "invalid_data_samples": failures["invalid_data_samples"],
    }


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source in rows:
            row = {}
            for field in fields:
                value = source.get(field)
                if value is None:
                    value = ""
                elif isinstance(value, bool):
                    value = int(value)
                elif isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                row[field] = value
            writer.writerow(row)
    return destination


def _draw_ground_truth(axis: Any, grasps: Sequence[Any], config: Mapping[str, Any]) -> None:
    for index, item in enumerate(make_ocid_vlg_evaluation_rectangles(grasps, config), start=1):
        polygon = np.vstack([item["polygon"], item["polygon"][0]])
        axis.plot(
            polygon[:, 0],
            polygon[:, 1],
            color="#00e5ff",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
        )
        axis.text(
            item["center_uv"][0],
            item["center_uv"][1],
            f"GT{index}",
            color="#00e5ff",
            fontsize=5,
            zorder=3,
        )


def _draw_ranked_candidate(
    axis: Any,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label_prefix: str = "",
) -> None:
    geometry = make_candidate_evaluation_rectangle(record, config)
    polygon = np.vstack([geometry["polygon"], geometry["polygon"][0]])
    axis.plot(
        polygon[:, 0],
        polygon[:, 1],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=5,
    )
    consistent = bool(record["ocid_vlg_2d_consistency"]["rectangle_match"])
    text = (
        f"{label_prefix}r{record['gqcnn_rank']} {record['candidate_id']} "
        f"q={record['gqcnn_q_value']:.6g} {'valid' if consistent else 'invalid'}"
    )
    axis.text(
        geometry["center_uv"][0] + 3,
        geometry["center_uv"][1] - 3,
        text,
        color="white",
        fontsize=6,
        zorder=7,
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": color, "pad": 1.0},
    )


def _base_failure_axis(sample_dir: Path, grasps: Sequence[Any], config: Mapping[str, Any]):
    rgb = np.asarray(Image.open(sample_dir / "rgb.png").convert("RGB"))
    mask = np.asarray(Image.open(sample_dir / "hifics_mask_processed.png").convert("L")) > 0
    if mask.shape != rgb.shape[:2]:
        raise EvaluationDataError("mapping_or_geometry_error", "RGB/mask dimensions differ")
    figure, axis = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    axis.imshow(rgb)
    if np.any(mask) and not np.all(mask):
        axis.contour(mask.astype(np.uint8), levels=[0.5], colors=["#ffea00"], linewidths=1.4)
    _draw_ground_truth(axis, grasps, config)
    axis.set_xlim(-0.5, rgb.shape[1] - 0.5)
    axis.set_ylim(rgb.shape[0] - 0.5, -0.5)
    axis.set_axis_off()
    return figure, axis


def _save_figure(figure: Any, path: Path, title: str) -> None:
    figure.axes[0].set_title(title)
    figure.tight_layout(pad=0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def save_failure_visualizations(outcome: Mapping[str, Any], output_root: Path | str) -> list[Path]:
    metrics = outcome["metrics"]
    if metrics["top1_consistent"]:
        return []
    sample_dir = Path(outcome["sample_dir"])
    destination = Path(output_root) / "failures" / metrics["sample_id"]
    ranked = outcome["ranked"]
    grasps = outcome["annotation"]["grasps"]
    config = outcome["evaluation_config"]
    paths = []

    figure, axis = _base_failure_axis(sample_dir, grasps, config)
    if ranked:
        _draw_ranked_candidate(
            axis,
            ranked[0],
            config,
            color="#ff1744",
            linestyle="-",
            linewidth=3.0,
            label_prefix="TOP1 ",
        )
    if metrics["first_valid_rank"] is not None and metrics["first_valid_rank"] != 1:
        first_valid = ranked[int(metrics["first_valid_rank"]) - 1]
        _draw_ranked_candidate(
            axis,
            first_valid,
            config,
            color="#00e676",
            linestyle="--",
            linewidth=2.6,
            label_prefix="FIRST_VALID ",
        )
    path = destination / "top1_vs_first_valid.png"
    _save_figure(figure, path, f"{metrics['sample_id']}: GQ-CNN Top-1 vs first valid")
    paths.append(path)

    figure, axis = _base_failure_axis(sample_dir, grasps, config)
    visible = ranked[: min(int(config["top_k"]), len(ranked))]
    if metrics["first_valid_rank"] is None:
        visible = ranked
    for record in visible:
        valid = bool(record["ocid_vlg_2d_consistency"]["rectangle_match"])
        _draw_ranked_candidate(
            axis,
            record,
            config,
            color="#00e676" if valid else "#ff9100",
            linestyle="-" if valid else ":",
            linewidth=2.0 if valid else 1.3,
        )
    path = destination / "top5_with_ground_truth.png"
    _save_figure(figure, path, f"{metrics['sample_id']}: q-ranked Top-5 and normalized GT")
    paths.append(path)

    details = {
        "baseline": BASELINE_NAME,
        "metric": METRIC_NAME,
        "is_physical_grasp_success": False,
        "metrics": metrics,
        "evaluation": outcome["evaluation"],
        "source_files": outcome["source_hashes"],
        "legend": {
            "target_mask_contour": "yellow",
            "normalized_ground_truth_rectangles": "cyan dashed",
            "gqcnn_top1": "red solid",
            "first_valid_candidate": "green dashed",
            "top5_valid": "green solid",
            "top5_invalid": "orange dotted",
        },
    }
    details_path = save_strict_json(destination / "failure_details.json", details)
    paths.append(details_path)
    return paths


def save_invalid_diagnostic(
    sample_dir: Path | str,
    output_root: Path | str,
    error_row: Mapping[str, Any],
) -> list[Path]:
    """Save a non-metric diagnostic for a sample that cannot be evaluated."""

    sample_dir = Path(sample_dir)
    destination = Path(output_root) / "failures" / str(error_row["sample_id"])
    rgb_path = sample_dir / "rgb.png"
    figure, axis = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    if rgb_path.is_file():
        axis.imshow(np.asarray(Image.open(rgb_path).convert("RGB")))
    else:
        axis.set_facecolor("#202020")
        axis.text(
            0.5,
            0.5,
            "RGB image unavailable",
            color="white",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.text(
        0.01,
        0.01,
        f"{error_row['failure_type']}: {error_row['failure_reason']}",
        color="white",
        fontsize=7,
        va="bottom",
        wrap=True,
        transform=axis.transAxes,
        bbox={"facecolor": "black", "alpha": 0.75, "edgecolor": "red", "pad": 2.0},
    )
    axis.set_axis_off()
    diagnostic_path = destination / "diagnostic.png"
    _save_figure(
        figure,
        diagnostic_path,
        f"{error_row['sample_id']}: evaluation unavailable (not scored)",
    )
    details_path = save_strict_json(
        destination / "failure_details.json",
        {
            "baseline": BASELINE_NAME,
            "metric": METRIC_NAME,
            "is_physical_grasp_success": False,
            "accuracy_visualization_generated": False,
            "error": dict(error_row),
        },
    )
    return [diagnostic_path, details_path]
