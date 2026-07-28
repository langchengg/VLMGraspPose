#!/usr/bin/env python3
"""Evaluate frozen GQ-CNN rankings with CROG's corrected rectangle kernel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grasping.geometric_ranker import (  # noqa: E402
    make_candidate_evaluation_rectangle,
    make_ocid_vlg_evaluation_rectangles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--mask-metadata", type=Path, required=True)
    parser.add_argument("--official-annotations", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--crog-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=7675)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, allow_nan=False)
                        if isinstance(value, (dict, list))
                        else ("" if value is None else value)
                    )
                    for key, value in source.items()
                }
            )
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "predicted_rectangle_height_px",
        "ground_truth_rectangle_height_px",
        "ground_truth_width_clip_px",
        "angle_threshold_deg",
        "iou_threshold",
        "top_k",
    }
    if not required.issubset(payload):
        raise ValueError(f"evaluation config missing {sorted(required - set(payload))}")
    expected_semantics = {
        "evaluator_version": "corrected_geometric_v2",
        "iou_comparison": ">",
        "angle_comparison": "<=",
    }
    observed = {key: payload.get(key) for key in expected_semantics}
    if observed != expected_semantics:
        raise ValueError(
            "evaluation config does not freeze corrected semantics: "
            f"observed={observed}, expected={expected_semantics}"
        )
    convention = payload.get("coordinate_convention", {})
    if (
        convention.get("polygon_vertices") != "[x,y]"
        or convention.get("rasterization") != "row=y, column=x"
        or "same GT" not in convention.get("multi_gt_match", "")
    ):
        raise ValueError("evaluation config coordinate/multi-GT semantics mismatch")
    return payload


def candidate_grasp(
    candidate: dict[str, Any], config: dict[str, Any]
) -> tuple[list[float], dict[str, Any]]:
    normalized = dict(candidate)
    normalized.setdefault(
        "center_uv",
        [candidate["center_u_px"], candidate["center_v_px"]],
    )
    geometry = make_candidate_evaluation_rectangle(normalized, config)
    center = np.asarray(geometry["center_uv"], dtype=np.float64)
    grasp = [
        float(center[0]),
        float(center[1]),
        float(geometry["width_px"]),
        float(geometry["height_px"]),
        float(math.degrees(geometry["angle_rad"])),
    ]
    return grasp, geometry


def ground_truth_grasps(
    corners: list[Any], config: dict[str, Any]
) -> list[list[float]]:
    geometries = make_ocid_vlg_evaluation_rectangles(corners, config)
    return [
        [
            float(item["center_uv"][0]),
            float(item["center_uv"][1]),
            float(item["width_px"]),
            float(item["height_px"]),
            float(math.degrees(item["angle_rad"])),
        ]
        for item in geometries
    ]


PER_CANDIDATE_SCHEMA = pa.schema(
    [
        ("pipeline", pa.string()),
        ("sample_index", pa.int32()),
        ("sample_id", pa.string()),
        ("question_index", pa.int32()),
        ("scene_id", pa.string()),
        ("candidate_id", pa.string()),
        ("gqcnn_rank", pa.int32()),
        ("gqcnn_q_value", pa.float64()),
        ("source_candidate_index", pa.int32()),
        ("center_u_px", pa.float64()),
        ("center_v_px", pa.float64()),
        ("center_depth_m", pa.float64()),
        ("angle_rad", pa.float64()),
        ("angle_deg_contact_span", pa.float64()),
        ("width_m", pa.float64()),
        ("contact_span_width_px", pa.float64()),
        ("configured_width_px", pa.float64()),
        ("endpoints_uv_json", pa.string()),
        ("contact_points_uv_json", pa.string()),
        ("contact_normals_json", pa.string()),
        ("center_camera_xyz_m_json", pa.string()),
        ("pose_matrix_json", pa.string()),
        ("candidate_seed", pa.int64()),
        ("candidate_success", pa.bool_()),
        ("best_gt_id", pa.string()),
        ("best_gt_index", pa.int32()),
        ("rectangle_iou", pa.float64()),
        ("angle_difference_deg", pa.float64()),
        ("iou_ok", pa.bool_()),
        ("angle_ok", pa.bool_()),
        ("joint_success", pa.bool_()),
        ("failure_mode", pa.string()),
        ("evaluator_version", pa.string()),
    ]
)


def parquet_row(
    *,
    pipeline: str,
    sample_index: int,
    question_index: int,
    scene_id: str,
    candidate: dict[str, Any],
    grasp: list[float],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    best = evaluation["best_gt"]
    return {
        "pipeline": pipeline,
        "sample_index": int(sample_index),
        "sample_id": str(candidate["sample_id"]),
        "question_index": int(question_index),
        "scene_id": scene_id,
        "candidate_id": str(candidate["candidate_id"]),
        "gqcnn_rank": int(candidate["gqcnn_rank"]),
        "gqcnn_q_value": float(candidate["gqcnn_q_value"]),
        "source_candidate_index": int(candidate.get("source_candidate_index", -1)),
        "center_u_px": float(grasp[0]),
        "center_v_px": float(grasp[1]),
        "center_depth_m": float(candidate["center_depth_m"]),
        "angle_rad": float(candidate["angle_rad"]),
        "angle_deg_contact_span": float(grasp[4]),
        "width_m": float(candidate["width_m"]),
        "contact_span_width_px": float(grasp[2]),
        "configured_width_px": float(candidate["width_px"]),
        "endpoints_uv_json": json.dumps(
            [candidate["endpoint_1_uv"], candidate["endpoint_2_uv"]],
            separators=(",", ":"),
        ),
        "contact_points_uv_json": json.dumps(
            candidate["contact_points_uv"], separators=(",", ":")
        ),
        "contact_normals_json": json.dumps(
            candidate["contact_normals"], separators=(",", ":")
        ),
        "center_camera_xyz_m_json": json.dumps(
            candidate["center_camera_xyz_m"], separators=(",", ":")
        ),
        "pose_matrix_json": json.dumps(
            candidate["T_camera_grasp_fixed_approach"], separators=(",", ":")
        ),
        "candidate_seed": int(candidate["seed"]),
        "candidate_success": bool(evaluation["candidate_success"]),
        "best_gt_id": None if best is None else str(best["gt_id"]),
        "best_gt_index": None if best is None else int(best["gt_index"]),
        "rectangle_iou": None if best is None else float(best["rectangle_iou"]),
        "angle_difference_deg": (
            None if best is None else float(best["angle_difference_deg"])
        ),
        "iou_ok": None if best is None else bool(best["iou_ok"]),
        "angle_ok": None if best is None else bool(best["angle_ok"]),
        "joint_success": None if best is None else bool(best["joint_success"]),
        "failure_mode": evaluation["failure_mode"],
        "evaluator_version": evaluation["evaluator_version"],
    }


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p5": None,
            "q1": None,
            "median": None,
            "q3": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p5": float(np.quantile(array, 0.05)),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def aggregate(
    rows: list[dict[str, Any]],
    *,
    pipeline: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    total = len(rows)
    scored = [row for row in rows if row["terminal_status"] == "scored"]
    empties_mask = [row for row in rows if row["terminal_status"] == "valid_empty_mask"]
    empties_candidates = [
        row for row in rows if row["terminal_status"] == "valid_empty_candidates"
    ]
    failures = [row for row in rows if row["terminal_status"] == "failed"]
    top1 = sum(bool(row["top1_correct"]) for row in rows)
    top5 = sum(bool(row["top5_correct"]) for row in rows)
    oracle = sum(bool(row["oracle_all"]) for row in rows)
    first_ranks = [
        int(row["first_correct_rank"])
        for row in rows
        if row["first_correct_rank"] is not None
    ]
    mrr = sum(
        0.0 if row["first_correct_rank"] is None else 1.0 / row["first_correct_rank"]
        for row in rows
    ) / total
    failure_flow = Counter(row["failure_category"] for row in rows)
    strata_edges = [
        (0.0, 0.25, "[0,0.25)"),
        (0.25, 0.50, "[0.25,0.50)"),
        (0.50, 0.70, "[0.50,0.70)"),
        (0.70, 0.80, "[0.70,0.80)"),
        (0.80, 0.90, "[0.80,0.90)"),
        (0.90, 1.0000001, "[0.90,1.00]"),
    ]
    strata = []
    for low, high, label in strata_edges:
        selected = [
            row
            for row in rows
            if low <= float(row["mask_iou"]) < high
        ]
        denominator = len(selected)
        strata.append(
            {
                "stratum": label,
                "denominator": denominator,
                "top1_numerator": sum(row["top1_correct"] for row in selected),
                "top5_numerator": sum(row["top5_correct"] for row in selected),
                "oracle_all_numerator": sum(row["oracle_all"] for row in selected),
                "top1": (
                    None
                    if not denominator
                    else sum(row["top1_correct"] for row in selected) / denominator
                ),
                "top5": (
                    None
                    if not denominator
                    else sum(row["top5_correct"] for row in selected) / denominator
                ),
                "oracle_all": (
                    None
                    if not denominator
                    else sum(row["oracle_all"] for row in selected) / denominator
                ),
            }
        )
    later_top5 = sum(
        row["first_correct_rank"] is not None
        and 2 <= int(row["first_correct_rank"]) <= 5
        for row in rows
    )
    later_after5 = sum(
        row["first_correct_rank"] is not None
        and int(row["first_correct_rank"]) > 5
        for row in rows
    )
    scored_denominator = len(scored)
    return {
        "schema_version": 1,
        "pipeline": pipeline,
        "metric_label": "corrected offline 2D rectangle consistency",
        "is_physical_grasp_success": False,
        "primary_denominator": total,
        "terminal_counts": {
            "scored": scored_denominator,
            "valid_empty_mask": len(empties_mask),
            "valid_empty_candidates": len(empties_candidates),
            "failed": len(failures),
        },
        "technical_failure_count": len(failures),
        "any_candidate": {
            "numerator": scored_denominator,
            "denominator": total,
            "value": scored_denominator / total,
        },
        "top1": {
            "numerator": top1,
            "denominator": total,
            "value": top1 / total,
        },
        "top5": {
            "numerator": top5,
            "denominator": total,
            "value": top5 / total,
        },
        "oracle_all": {
            "numerator": oracle,
            "denominator": total,
            "value": oracle / total,
        },
        "scored_only": {
            "denominator": scored_denominator,
            "top1_numerator": top1,
            "top1": None if not scored_denominator else top1 / scored_denominator,
            "top5_numerator": top5,
            "top5": None if not scored_denominator else top5 / scored_denominator,
            "oracle_all_numerator": oracle,
            "oracle_all": None if not scored_denominator else oracle / scored_denominator,
        },
        "top1_wrong_but_later_correct": {
            "rank_2_to_5": later_top5,
            "rank_after_5": later_after5,
            "all_later": later_top5 + later_after5,
        },
        "first_correct_rank": {
            **quantiles([float(value) for value in first_ranks]),
            "distribution": dict(
                sorted(Counter(str(value) for value in first_ranks).items())
            ),
        },
        "mrr": {
            "denominator": total,
            "value": float(mrr),
        },
        "top1_q_value": quantiles(
            [float(row["top1_q_value"]) for row in scored]
        ),
        "raw_candidate_count": quantiles(
            [float(row["raw_candidate_count"]) for row in rows]
        ),
        "nms_candidate_count": quantiles(
            [float(row["nms_candidate_count"]) for row in rows]
        ),
        "failure_flow": {
            "predicted_mask_empty": int(failure_flow["predicted_mask_empty"]),
            "mask_nonempty_no_candidate": int(
                failure_flow["mask_nonempty_no_candidate"]
            ),
            "candidate_pool_no_correct": int(
                failure_flow["candidate_pool_no_correct"]
            ),
            "ranking_failure": int(failure_flow["ranking_failure"]),
            "top1_correct": int(failure_flow["top1_correct"]),
            "technical_failure": int(failure_flow["technical_failure"]),
        },
        "mask_iou_strata": strata,
        "evaluator": {
            "version": "corrected_geometric_v2",
            "rectangle_coordinates": "[x,y] pixel coordinates",
            "rasterization": "row=y, column=x",
            "angle_periodicity": "180 degrees",
            "iou_threshold": float(config["iou_threshold"]),
            "iou_comparison": ">",
            "angle_threshold_deg": float(config["angle_threshold_deg"]),
            "angle_comparison": "<=",
            "multi_gt": "one same GT rectangle must jointly pass IoU and angle",
            "empty_policy": "failure in the all-7675 primary denominator",
        },
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    candidate_root = args.candidate_root.expanduser().resolve()
    scored_root = args.scored_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.evaluation_config.expanduser().resolve()
    config = load_config(config_path)
    crog_root = args.crog_root.expanduser().resolve()
    sys.path.insert(0, str(crog_root))
    from utils.grasp_metrics import (  # noqa: E402
        CORRECTED_EVALUATOR_VERSION,
        evaluate_candidate,
    )

    if CORRECTED_EVALUATOR_VERSION != "corrected_geometric_v2":
        raise RuntimeError("unexpected CROG corrected evaluator version")
    mask_rows = read_csv(args.mask_metadata.expanduser().resolve())
    if len(mask_rows) != args.expected_samples:
        raise ValueError(f"mask metadata count {len(mask_rows)} != {args.expected_samples}")
    mask_by_id = {row["sample_id"]: row for row in mask_rows}
    candidate_rows = read_csv(candidate_root / "summary.csv")
    if len(candidate_rows) != args.expected_samples:
        raise ValueError(
            f"candidate summary count {len(candidate_rows)} != {args.expected_samples}"
        )
    annotations_path = args.official_annotations.expanduser().resolve()
    annotations = load_json(annotations_path)["data"]
    annotation_by_question = {
        int(row["question_index"]): row for row in annotations
    }
    per_sample: list[dict[str, Any]] = []
    parquet_path = output_dir / f"{args.pipeline}_gqcnn_per_candidate.parquet"
    parquet_writer = pq.ParquetWriter(
        parquet_path,
        PER_CANDIDATE_SCHEMA,
        compression="zstd",
        use_dictionary=True,
    )
    parquet_batch: list[dict[str, Any]] = []
    total_candidates = 0
    try:
        for sample_index, candidate_summary in enumerate(candidate_rows):
            sample_id = candidate_summary["sample_id"]
            mask = mask_by_id.get(sample_id)
            if mask is None:
                raise ValueError(f"mask metadata missing: {sample_id}")
            candidate_dir = candidate_root / sample_id
            scored_dir = scored_root / sample_id
            candidate_metadata = load_json(candidate_dir / "metadata.json")
            question_index = int(candidate_metadata["question_index"])
            annotation = annotation_by_question.get(question_index)
            if annotation is None:
                raise ValueError(f"annotation missing: {question_index}")
            if annotation["image_filename"] != candidate_metadata["scene_id"]:
                raise ValueError(f"annotation scene mismatch: {sample_id}")
            raw_count = int(candidate_metadata["counts"]["raw"])
            nms_count = int(candidate_metadata["counts"]["post_nms"])
            mask_empty = str(mask["empty_mask"]).lower() == "true"
            top1_correct = False
            top5_correct = False
            oracle_all = False
            first_correct_rank = None
            top1_q = None
            scored_count = 0
            scoring_time_ms = 0.0
            scoring_total_ms = 0.0
            if nms_count > 0:
                scored_payload = load_json(
                    scored_dir / "gqcnn_scored_candidates.json"
                )
                ranked = list(scored_payload["candidates"])
                reconstructed = sorted(
                    ranked,
                    key=lambda item: (
                        -float(item["gqcnn_q_value"]),
                        str(item["candidate_id"]),
                    ),
                )
                if [row["candidate_id"] for row in reconstructed] != [
                    row["candidate_id"] for row in ranked
                ]:
                    raise RuntimeError(f"stored GQ-CNN rank order mismatch: {sample_id}")
                if any(
                    int(row["gqcnn_rank"]) != index
                    for index, row in enumerate(ranked, 1)
                ):
                    raise RuntimeError(f"stored GQ-CNN ranks invalid: {sample_id}")
                if not all(
                    math.isfinite(float(row["gqcnn_q_value"])) for row in ranked
                ):
                    raise RuntimeError(f"non-finite GQ-CNN Q: {sample_id}")
                if len(ranked) != nms_count:
                    raise RuntimeError(f"scored/NMS count mismatch: {sample_id}")
                gt_grasps = ground_truth_grasps(annotation["grasps"], config)
                evaluations = []
                for candidate in ranked:
                    grasp, _ = candidate_grasp(candidate, config)
                    evaluated = evaluate_candidate(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "legacy_grasp": grasp,
                        },
                        gt_grasps,
                        shape=(480, 640),
                        evaluator_version=CORRECTED_EVALUATOR_VERSION,
                        iou_threshold=float(config["iou_threshold"]),
                        angle_threshold=float(config["angle_threshold_deg"]),
                    )
                    evaluations.append(evaluated)
                    parquet_batch.append(
                        parquet_row(
                            pipeline=args.pipeline,
                            sample_index=sample_index,
                            question_index=question_index,
                            scene_id=candidate_metadata["scene_id"],
                            candidate=candidate,
                            grasp=grasp,
                            evaluation=evaluated,
                        )
                    )
                    if len(parquet_batch) >= 5000:
                        parquet_writer.write_table(
                            pa.Table.from_pylist(
                                parquet_batch, schema=PER_CANDIDATE_SCHEMA
                            )
                        )
                        parquet_batch.clear()
                correct = [bool(item["candidate_success"]) for item in evaluations]
                first_correct_rank = next(
                    (index for index, value in enumerate(correct, 1) if value),
                    None,
                )
                top1_correct = bool(correct[0])
                top5_correct = any(correct[: int(config["top_k"])])
                oracle_all = any(correct)
                top1_q = float(ranked[0]["gqcnn_q_value"])
                scored_count = len(ranked)
                scoring_metadata = load_json(scored_dir / "scoring_metadata.json")
                scoring_time_ms = float(scoring_metadata["timing_ms"]["model_inference"])
                scoring_total_ms = float(scoring_metadata["timing_ms"]["total"])
                terminal_status = "scored"
            elif mask_empty:
                terminal_status = "valid_empty_mask"
            else:
                terminal_status = "valid_empty_candidates"
            if terminal_status == "valid_empty_mask":
                failure_category = "predicted_mask_empty"
            elif terminal_status == "valid_empty_candidates":
                failure_category = "mask_nonempty_no_candidate"
            elif not oracle_all:
                failure_category = "candidate_pool_no_correct"
            elif not top1_correct:
                failure_category = "ranking_failure"
            else:
                failure_category = "top1_correct"
            row = {
                "pipeline": args.pipeline,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "question_index": question_index,
                "scene_id": candidate_metadata["scene_id"],
                "query": candidate_metadata["query"],
                "terminal_status": terminal_status,
                "technical_failure": False,
                "mask_iou": float(mask["standard_iou_352"]),
                "mask_area_352_px": int(mask["mask_area_352_px"]),
                "mask_area_native_px": int(mask["mask_area_native_px"]),
                "valid_depth_fraction": float(mask["valid_depth_fraction"]),
                "raw_candidate_count": raw_count,
                "nms_candidate_count": nms_count,
                "scored_candidate_count": scored_count,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "oracle_all": oracle_all,
                "first_correct_rank": first_correct_rank,
                "reciprocal_rank": (
                    0.0 if first_correct_rank is None else 1.0 / first_correct_rank
                ),
                "top1_q_value": top1_q,
                "failure_category": failure_category,
                "mask_inference_seconds": float(mask["inference_seconds"]),
                "candidate_generation_time_ms": float(
                    candidate_metadata["timing_ms"]["generation"]
                ),
                "gqcnn_inference_time_ms": scoring_time_ms,
                "gqcnn_total_time_ms": scoring_total_ms,
                "candidate_seed": int(candidate_metadata["seed"]),
            }
            per_sample.append(row)
            total_candidates += scored_count
        if parquet_batch:
            parquet_writer.write_table(
                pa.Table.from_pylist(parquet_batch, schema=PER_CANDIDATE_SCHEMA)
            )
            parquet_batch.clear()
    finally:
        parquet_writer.close()
    if len(per_sample) != args.expected_samples:
        raise RuntimeError("per-sample evaluation count mismatch")
    metrics = aggregate(per_sample, pipeline=args.pipeline, config=config)
    metrics.update(
        {
            "inputs": {
                "candidate_root": str(candidate_root),
                "candidate_run_config_sha256": sha256_file(
                    candidate_root / "run_config.json"
                ),
                "scored_root": str(scored_root),
                "scoring_run_config_sha256": sha256_file(
                    scored_root / "run_config.json"
                ),
                "mask_metadata": str(args.mask_metadata.expanduser().resolve()),
                "mask_metadata_sha256": sha256_file(
                    args.mask_metadata.expanduser().resolve()
                ),
                "official_annotations": str(annotations_path),
                "official_annotations_sha256": sha256_file(annotations_path),
                "evaluation_config": str(config_path),
                "evaluation_config_sha256": sha256_file(config_path),
                "corrected_evaluator_source": str(
                    crog_root / "utils" / "grasp_metrics.py"
                ),
                "corrected_evaluator_source_sha256": sha256_file(
                    crog_root / "utils" / "grasp_metrics.py"
                ),
            },
            "per_candidate_rows": total_candidates,
            "per_candidate_parquet": str(parquet_path),
            "per_candidate_parquet_sha256": sha256_file(parquet_path),
            "total_evaluation_seconds": time.perf_counter() - started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    per_sample_path = output_dir / f"{args.pipeline}_per_sample_pipeline_metrics.csv"
    metrics_path = output_dir / f"{args.pipeline}_pipeline_metrics.json"
    write_csv(per_sample_path, per_sample)
    metrics["per_sample_metrics"] = str(per_sample_path)
    metrics["per_sample_metrics_sha256"] = sha256_file(per_sample_path)
    write_json(metrics_path, metrics)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "pipeline": args.pipeline,
                "samples": len(per_sample),
                "candidates": total_candidates,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "oracle_all": metrics["oracle_all"],
                "output": str(metrics_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
