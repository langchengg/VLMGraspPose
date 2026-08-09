#!/usr/bin/env python3
"""Build a deterministic, test-only 4-DoF failure gallery.

The tool consumes only frozen test manifests/labels and already-written formal
prediction tables.  It recomputes candidate outcomes from stored geometry, so
post-hoc diagnostic labels cannot influence inference or configuration choice.
Ground truth is drawn only in explicitly labelled evaluation panels.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
import shutil
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.candidate_decoder import rank_candidates  # noqa: E402
from src.grasping.common.evaluator import (  # noqa: E402
    EvaluatorConfig,
    evaluate_ocid_predictions,
    grasp_from_ocid_corners,
)
from src.grasping.common.geometry import rectangle_corners  # noqa: E402
from src.grasping.common.types import Grasp4DoF  # noqa: E402
from src.grasping.backends import BackendSample  # noqa: E402
from src.grasping.backends.base import prepare_conditioned_input  # noqa: E402
from src.grasping.backends.network_utils import (  # noqa: E402
    gated_quality_map,
    official_gaussian_post_process,
    synchronize_device,
)
from src.grasping.common.sample_io import CompactSampleLoader  # noqa: E402
from tools.grasp4dof.run_method import _build_backend  # noqa: E402


SCHEMA_VERSION = 1
ANALYSIS_SCOPE = "formal_test_only"
FAILURE_STAGES = (
    "grounding_wrong_target",
    "grounding_fragmented_mask",
    "empty_mask",
    "invalid_depth",
    "no_candidate_generated",
    "candidate_pool_has_no_positive",
    "ranking_failure",
    "angle_failure",
    "width_failure",
    "crop_mapping_failure",
    "visible_collision_proxy_failure",
    "evaluator_ambiguity",
    "successful",
)
GALLERY_QUOTAS: Mapping[str, int] = {
    "success": 20,
    "ranking_failure": 20,
    "no_candidate": 15,
    "wrong_mask": 15,
    "angle_failure": 10,
    "width_failure": 10,
}
CROSS_METHOD_QUOTA = 30

SAMPLE_COLUMNS = (
    "sample_id",
    "scene_id",
    "language",
    "source_rgb_path",
    "source_depth_path",
    "predicted_mask_path",
)
LABEL_COLUMNS = (
    "sample_id",
    "scene_id",
    "prepared_gt_mask_path",
    "gt_grasp_rectangles",
)
PREDICTION_COLUMNS = (
    "sample_id",
    "scene_id",
    "j_at_1",
    "j_at_5",
    "candidate_pool_oracle",
    "first_valid_rank",
    "non_empty",
    "empty_reason",
)
CANDIDATE_COLUMNS = (
    "sample_id",
    "candidate_id",
    "rank",
    "center_x",
    "center_y",
    "angle_deg",
    "width_px",
    "height_px",
    "score",
)

PANEL_WIDTH = 420
PANEL_HEIGHT = 315
PANEL_GAP = 16
MARGIN = 20


class FailureGalleryError(ValueError):
    """Raised when formal artifacts violate the gallery input contract."""


def _read_parquet(path: Path, *, label: str, allow_empty: bool = False) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {path}")
    frame = pd.read_parquet(path)
    if frame.empty and not allow_empty:
        raise FailureGalleryError(f"{label} must not be empty")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FailureGalleryError(f"{label} missing columns: {missing}")


def _strict_test_split(frame: pd.DataFrame, *, label: str) -> None:
    if "split" not in frame.columns:
        return
    splits = {str(value).strip().lower() for value in frame["split"].tolist()}
    if splits != {"test"}:
        raise FailureGalleryError(
            f"{label} is not test-only; observed split values: {sorted(splits)}"
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    result = str(value)
    return result if result else None


def _strict_integer(value: Any, *, field: str, sample_id: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise FailureGalleryError(f"{sample_id}: {field} must be an integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise FailureGalleryError(f"{sample_id}: {field} must be an integer")
    return int(number)


def _load_mask(path: str | Path, *, size: tuple[int, int], label: str) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    with Image.open(resolved) as image:
        mask = image.convert("L")
        if mask.size != size:
            mask = mask.resize(size, resample=Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 0


def _load_rgb(path: str | Path) -> Image.Image:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty source RGB: {resolved}")
    with Image.open(resolved) as image:
        return image.convert("RGB")


def _load_depth_valid(path: str | Path, *, size: tuple[int, int]) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty source depth: {resolved}")
    with Image.open(resolved) as image:
        if image.size != size:
            image = image.resize(size, resample=Image.Resampling.NEAREST)
        depth = np.asarray(image)
    if depth.ndim == 3:
        depth = depth[..., 0]
    numeric = np.asarray(depth, dtype=np.float64)
    return np.isfinite(numeric) & (numeric > 0.0)


def _mask_diagnostics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    if predicted.shape != ground_truth.shape:
        raise FailureGalleryError("predicted and GT masks must have equal shapes")
    pred_area = int(predicted.sum())
    gt_area = int(ground_truth.sum())
    if gt_area == 0:
        raise FailureGalleryError("frozen GT mask contains no positive pixel")
    intersection = int(np.logical_and(predicted, ground_truth).sum())
    union = int(np.logical_or(predicted, ground_truth).sum())
    mask_iou = 0.0 if union == 0 else float(intersection / union)

    component_count = 0
    largest_ratio = 0.0
    if pred_area:
        labels, component_count = ndimage.label(predicted)
        sizes = np.bincount(labels.reshape(-1))[1:]
        largest_ratio = float(sizes.max() / pred_area) if sizes.size else 0.0

    centroid_distance: float | None = None
    gt_bbox_diagonal: float | None = None
    if pred_area and gt_area:
        pred_rows, pred_cols = np.nonzero(predicted)
        gt_rows, gt_cols = np.nonzero(ground_truth)
        centroid_distance = float(
            math.hypot(
                float(pred_cols.mean() - gt_cols.mean()),
                float(pred_rows.mean() - gt_rows.mean()),
            )
        )
        gt_bbox_diagonal = float(
            math.hypot(
                float(gt_cols.max() - gt_cols.min() + 1),
                float(gt_rows.max() - gt_rows.min() + 1),
            )
        )

    zero_overlap = bool(pred_area and gt_area and intersection == 0)
    centroid_far = bool(
        centroid_distance is not None
        and gt_bbox_diagonal is not None
        and centroid_distance > 0.25 * gt_bbox_diagonal
    )
    wrong_mask = bool(zero_overlap or (pred_area and mask_iou < 0.25 and centroid_far))
    fragmented = bool(component_count > 1 and largest_ratio < 0.90)
    return {
        "predicted_mask_area_px": pred_area,
        "gt_mask_area_px": gt_area,
        "mask_intersection_px": intersection,
        "mask_iou": mask_iou,
        "predicted_mask_component_count": int(component_count),
        "predicted_mask_largest_component_ratio": largest_ratio,
        "mask_centroid_distance_px": centroid_distance,
        "gt_mask_bbox_diagonal_px": gt_bbox_diagonal,
        "wrong_mask_flag": wrong_mask,
        "fragmented_mask_flag": fragmented,
        "empty_mask_flag": pred_area == 0,
    }


def _candidate_pool(frame: pd.DataFrame, *, sample_id: str) -> list[Grasp4DoF]:
    if frame.empty:
        return []
    _require_columns(frame, CANDIDATE_COLUMNS, label=f"{sample_id} candidates")
    trusted = frame.loc[:, list(CANDIDATE_COLUMNS)]
    if trusted.isna().any().any():
        raise FailureGalleryError(f"{sample_id}: candidate geometry contains nulls")
    ranks: list[int] = []
    candidates: list[Grasp4DoF] = []
    for row in frame.itertuples(index=False):
        rank = _strict_integer(row.rank, field="rank", sample_id=sample_id)
        if rank <= 0:
            raise FailureGalleryError(f"{sample_id}: candidate rank must be positive")
        candidate_id = str(row.candidate_id)
        if not candidate_id:
            raise FailureGalleryError(f"{sample_id}: empty candidate_id")
        ranks.append(rank)
        candidates.append(
            Grasp4DoF(
                center_x=float(row.center_x),
                center_y=float(row.center_y),
                angle_deg=float(row.angle_deg),
                width_px=float(row.width_px),
                height_px=float(row.height_px),
                score=float(row.score),
                candidate_id=candidate_id,
            )
        )
    order = np.argsort(np.asarray(ranks), kind="stable")
    ordered_ranks = [ranks[index] for index in order]
    if ordered_ranks != list(range(1, len(ranks) + 1)):
        raise FailureGalleryError(
            f"{sample_id}: candidate ranks are duplicate, missing, or non-contiguous"
        )
    ordered = [candidates[index] for index in order]
    if len({item.candidate_id for item in ordered}) != len(ordered):
        raise FailureGalleryError(f"{sample_id}: duplicate candidate_id")
    if [item.candidate_id for item in rank_candidates(ordered)] != [
        item.candidate_id for item in ordered
    ]:
        raise FailureGalleryError(
            f"{sample_id}: saved rank disagrees with deterministic score order"
        )
    return ordered


def _validate_rectangles(value: Any, *, sample_id: str) -> list[list[list[float]]]:
    rectangles: list[list[list[float]]] = []
    for rectangle in value:
        points = [
            np.asarray(point, dtype=np.float64).reshape(-1) for point in rectangle
        ]
        if len(points) != 4 or any(point.shape != (2,) for point in points):
            raise FailureGalleryError(f"{sample_id}: malformed frozen GT rectangle")
        array = np.stack(points)
        if not np.all(np.isfinite(array)):
            raise FailureGalleryError(f"{sample_id}: non-finite frozen GT rectangle")
        rectangles.append(array.tolist())
    if not rectangles:
        raise FailureGalleryError(f"{sample_id}: frozen label has no GT grasps")
    return rectangles


def _specific_candidate_failure(
    pool: Sequence[Grasp4DoF],
    gt_rectangles: Sequence[Sequence[Sequence[float]]],
    outcome: Any,
    *,
    config: EvaluatorConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "angle_failure_flag": False,
        "width_failure_flag": False,
        "top1_width_relative_error": None,
        "top1_center_distance_px": None,
    }
    if not pool or outcome.j_at_1:
        return result
    top_evaluation = outcome.candidates[0]
    result["angle_failure_flag"] = any(
        pair.iou_ok and not pair.angle_ok for pair in top_evaluation.pairwise
    )

    ground_truth = [
        grasp_from_ocid_corners(
            corners,
            fixed_height_px=config.fixed_height_px,
            width_clip_px=config.gt_width_clip_px,
        )
        for corners in gt_rectangles
    ]
    angle_compatible = [
        pair
        for pair in top_evaluation.pairwise
        if pair.angle_ok and not pair.iou_ok
    ]
    if not angle_compatible:
        return result
    best = min(
        angle_compatible,
        key=lambda pair: (-pair.rectangle_iou, pair.angle_difference_deg, pair.gt_index),
    )
    gt = ground_truth[best.gt_index]
    candidate = pool[0]
    width_relative_error = abs(candidate.width_px - gt.width_px) / gt.width_px
    center_distance = math.hypot(
        candidate.center_x - gt.center_x, candidate.center_y - gt.center_y
    )
    result["top1_width_relative_error"] = float(width_relative_error)
    result["top1_center_distance_px"] = float(center_distance)
    # This is explicitly a diagnostic proxy, not a benchmark criterion: the
    # angle is compatible, the centre is local, and opening width differs
    # substantially while rectangle IoU fails.
    result["width_failure_flag"] = bool(
        width_relative_error >= 0.25
        and center_distance <= max(config.fixed_height_px, 0.5 * gt.width_px)
    )
    return result


def _stage(row: Mapping[str, Any]) -> tuple[str, str]:
    if bool(row["j_at_1"]):
        return "successful", "recomputed Top-1 satisfies frozen IoU and angle criteria"
    if bool(row["wrong_mask_flag"]):
        return (
            "grounding_wrong_target",
            "diagnostic proxy: zero GT overlap, or IoU<0.25 with a distant centroid",
        )
    if bool(row["fragmented_mask_flag"]):
        return (
            "grounding_fragmented_mask",
            "predicted mask has multiple components and largest-component ratio<0.90",
        )
    if bool(row["empty_mask_flag"]):
        return "empty_mask", "repeated-FiLM predicted mask contains no positive pixel"
    if bool(row["invalid_depth_flag"]):
        return "invalid_depth", "predicted target mask contains no valid positive depth"
    if bool(row["no_candidate_flag"]):
        return "no_candidate_generated", "formal candidate table has no row for sample"
    if bool(row["ranking_failure_flag"]):
        return (
            "ranking_failure",
            "candidate pool contains a positive but recomputed Top-1 is negative",
        )
    if bool(row["angle_failure_flag"]):
        return (
            "angle_failure",
            "Top-1 overlaps a GT rectangle above threshold but exceeds angle tolerance",
        )
    if bool(row["width_failure_flag"]):
        return (
            "width_failure",
            "diagnostic proxy: angle-compatible local Top-1 has >=25% width error and fails IoU",
        )
    if bool(row["candidate_pool_has_no_positive_flag"]):
        return (
            "candidate_pool_has_no_positive",
            "non-empty candidate pool contains no candidate satisfying both criteria",
        )
    return (
        "evaluator_ambiguity",
        "saved artifacts did not support a more specific deterministic stage",
    )


def _load_frozen_inputs(run_dir: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    samples = _read_parquet(
        run_dir / "manifests" / "test_samples.parquet", label="frozen test manifest"
    )
    labels = _read_parquet(
        run_dir / "manifests" / "test_labels.parquet", label="frozen test labels"
    )
    _require_columns(samples, SAMPLE_COLUMNS, label="frozen test manifest")
    _require_columns(labels, LABEL_COLUMNS, label="frozen test labels")
    _strict_test_split(samples, label="frozen test manifest")
    _strict_test_split(labels, label="frozen test labels")
    if samples.duplicated("sample_id", keep=False).any():
        raise FailureGalleryError("duplicate sample_id in frozen test manifest")
    if labels.duplicated("sample_id", keep=False).any():
        raise FailureGalleryError("duplicate sample_id in frozen test labels")
    sample_ids = [str(value) for value in samples["sample_id"].tolist()]
    label_ids = [str(value) for value in labels["sample_id"].tolist()]
    if len(sample_ids) != len(set(sample_ids)):
        raise FailureGalleryError("duplicate string-normalized sample_id in frozen test manifest")
    if len(label_ids) != len(set(label_ids)):
        raise FailureGalleryError("duplicate string-normalized sample_id in frozen test labels")
    if set(sample_ids) != set(label_ids):
        raise FailureGalleryError("frozen test manifest/label sample coverage mismatch")
    sample_by_id = {
        str(row.sample_id): row._asdict() for row in samples.itertuples(index=False)
    }
    label_by_id = {
        str(row.sample_id): row._asdict() for row in labels.itertuples(index=False)
    }
    merged: dict[str, dict[str, Any]] = {}
    for sample_id in sample_ids:
        sample = sample_by_id[sample_id]
        label = label_by_id[sample_id]
        if str(sample["scene_id"]) != str(label["scene_id"]):
            raise FailureGalleryError(f"{sample_id}: frozen scene_id mismatch")
        merged[sample_id] = {
            **sample,
            "gt_grasp_rectangles": _validate_rectangles(
                label["gt_grasp_rectangles"], sample_id=sample_id
            ),
            "prepared_gt_mask_path": label["prepared_gt_mask_path"],
        }
    return sample_ids, merged


def _analyse_method(
    *,
    method: str,
    method_dir: Path,
    sample_ids: Sequence[str],
    frozen: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[Grasp4DoF]]]:
    sample_path = method_dir / "per_sample_predictions.parquet"
    candidate_path = method_dir / "per_candidate_predictions.parquet"
    predictions = _read_parquet(sample_path, label=f"{method} per-sample predictions")
    candidates = _read_parquet(
        candidate_path, label=f"{method} per-candidate predictions", allow_empty=True
    )
    _require_columns(predictions, PREDICTION_COLUMNS, label=f"{method} per-sample predictions")
    if candidates.empty and not set(CANDIDATE_COLUMNS).issubset(candidates.columns):
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    else:
        _require_columns(candidates, CANDIDATE_COLUMNS, label=f"{method} candidates")
    if predictions.duplicated("sample_id", keep=False).any():
        raise FailureGalleryError(f"{method}: duplicate sample_id in per-sample predictions")
    saved_ids = [str(value) for value in predictions["sample_id"].tolist()]
    if set(saved_ids) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(saved_ids))
        extra = sorted(set(saved_ids) - set(sample_ids))
        raise FailureGalleryError(
            f"{method}: sample coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    outside = sorted(set(candidates["sample_id"].astype(str)) - set(sample_ids))
    if outside:
        raise FailureGalleryError(f"{method}: candidates outside frozen test set: {outside[:5]}")
    if candidates.duplicated(["sample_id", "candidate_id"], keep=False).any():
        raise FailureGalleryError(f"{method}: duplicate sample/candidate IDs")

    predictions_by_id = {
        str(row.sample_id): row._asdict() for row in predictions.itertuples(index=False)
    }
    candidate_groups = {
        str(sample_id): frame
        for sample_id, frame in candidates.groupby("sample_id", sort=False)
    }
    empty_candidates = candidates.iloc[0:0]
    config = EvaluatorConfig()
    rows: list[dict[str, Any]] = []
    pools: dict[str, list[Grasp4DoF]] = {}
    for sample_id in sample_ids:
        item = frozen[sample_id]
        saved = predictions_by_id[sample_id]
        if str(saved["scene_id"]) != str(item["scene_id"]):
            raise FailureGalleryError(f"{method}/{sample_id}: scene_id mismatch")
        rgb = _load_rgb(item["source_rgb_path"])
        predicted_mask = _load_mask(
            item["predicted_mask_path"], size=rgb.size, label="predicted mask"
        )
        gt_mask = _load_mask(
            item["prepared_gt_mask_path"], size=rgb.size, label="GT mask"
        )
        valid_depth = _load_depth_valid(item["source_depth_path"], size=rgb.size)
        mask = _mask_diagnostics(predicted_mask, gt_mask)
        valid_fraction: float | None = None
        if mask["predicted_mask_area_px"]:
            valid_fraction = float(valid_depth[predicted_mask].mean())

        pool = _candidate_pool(
            candidate_groups.get(sample_id, empty_candidates), sample_id=sample_id
        )
        pools[sample_id] = pool
        outcome = evaluate_ocid_predictions(
            pool, item["gt_grasp_rectangles"], config=config
        )
        pool_oracle = bool(any(entry.candidate_success for entry in outcome.candidates))
        specific = _specific_candidate_failure(
            pool, item["gt_grasp_rectangles"], outcome, config=config
        )
        top = outcome.candidates[0] if outcome.candidates else None
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "analysis_scope": ANALYSIS_SCOPE,
            "configuration_selection_performed": False,
            "method": method,
            "sample_id": sample_id,
            "scene_id": str(item["scene_id"]),
            "language": str(item["language"]),
            "j_at_1": bool(outcome.j_at_1),
            "j_at_5": bool(outcome.j_at_5),
            "candidate_pool_oracle": pool_oracle,
            "first_valid_rank": outcome.first_valid_rank,
            "candidate_count": len(pool),
            "successful_flag": bool(outcome.j_at_1),
            "invalid_depth_flag": bool(valid_fraction is not None and valid_fraction == 0.0),
            "no_candidate_flag": not pool,
            "candidate_pool_has_no_positive_flag": bool(pool and not pool_oracle),
            "ranking_failure_flag": bool(pool_oracle and not outcome.j_at_1),
            "top1_rectangle_iou": None if top is None else top.best_rectangle_iou,
            "top1_angle_difference_deg": None
            if top is None
            else top.best_angle_difference_deg,
            "valid_target_depth_fraction": valid_fraction,
            "empty_reason": _optional_string(saved.get("empty_reason")),
            "dense_maps_status": "unavailable",
            "dense_maps_reason": (
                "formal output schema stores candidates and map summary metadata, "
                "not persisted dense-map arrays"
            ),
            "crop_mapping_status": "unavailable_from_saved_artifacts",
            "visible_collision_proxy_status": "unavailable_from_saved_artifacts",
            "evaluator_ambiguity_status": "not_independently_inferred",
            **mask,
            **specific,
        }
        failure_stage, failure_rule = _stage(row)
        row["failure_stage"] = failure_stage
        row["failure_rule"] = failure_rule
        rows.append(row)
    return rows, pools


def _finite_sort_value(value: Any, *, descending: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf
    if not math.isfinite(number):
        return math.inf
    return -number if descending else number


def select_gallery_cases(analysis: pd.DataFrame) -> dict[str, Any]:
    """Select deterministic per-method and cross-method qualitative cases."""

    required = {
        "method",
        "sample_id",
        "j_at_1",
        "ranking_failure_flag",
        "no_candidate_flag",
        "wrong_mask_flag",
        "angle_failure_flag",
        "width_failure_flag",
        "mask_iou",
        "top1_angle_difference_deg",
        "top1_width_relative_error",
        "failure_stage",
    }
    _require_columns(analysis, tuple(required), label="failure analysis")
    selections: list[dict[str, str]] = []
    counts: dict[str, dict[str, dict[str, int]]] = {}
    category_specs = {
        "success": ("j_at_1", "sample_id", False),
        "ranking_failure": ("ranking_failure_flag", "first_valid_rank", True),
        "no_candidate": ("no_candidate_flag", "sample_id", False),
        "wrong_mask": ("wrong_mask_flag", "mask_iou", False),
        "angle_failure": ("angle_failure_flag", "top1_angle_difference_deg", True),
        "width_failure": ("width_failure_flag", "top1_width_relative_error", True),
    }
    methods = sorted({str(value) for value in analysis["method"].tolist()})
    for method in methods:
        method_frame = analysis.loc[analysis["method"].astype(str) == method].copy()
        counts[method] = {}
        for category, quota in GALLERY_QUOTAS.items():
            flag, sort_column, descending = category_specs[category]
            subset = method_frame.loc[method_frame[flag].astype(bool)].copy()
            records = subset.to_dict("records")
            if sort_column == "sample_id":
                records.sort(key=lambda row: str(row["sample_id"]))
            else:
                records.sort(
                    key=lambda row: (
                        _finite_sort_value(row.get(sort_column), descending=descending),
                        str(row["sample_id"]),
                    )
                )
            chosen = records[:quota]
            selections.extend(
                {
                    "method": method,
                    "sample_id": str(row["sample_id"]),
                    "category": category,
                }
                for row in chosen
            )
            counts[method][category] = {
                "requested": quota,
                "available": len(records),
                "actual": len(chosen),
            }

    cross_records: list[dict[str, Any]] = []
    if len(methods) >= 2:
        for sample_id, frame in analysis.groupby("sample_id", sort=False):
            present = {str(value) for value in frame["method"].tolist()}
            if present != set(methods):
                raise FailureGalleryError(
                    f"{sample_id}: cross-method sample coverage is incomplete"
                )
            outcomes = {bool(value) for value in frame["j_at_1"].tolist()}
            stages = {str(value) for value in frame["failure_stage"].tolist()}
            cross_records.append(
                {
                    "sample_id": str(sample_id),
                    "outcome_disagreement": len(outcomes) > 1,
                    "stage_diversity": len(stages),
                }
            )
        cross_records.sort(
            key=lambda row: (
                -int(row["outcome_disagreement"]),
                -int(row["stage_diversity"]),
                row["sample_id"],
            )
        )
    cross_available = len(cross_records)
    cross_records = cross_records[:CROSS_METHOD_QUOTA]
    return {
        "per_method": selections,
        "counts": counts,
        "cross_method": cross_records,
        "cross_method_count": {
            "requested": CROSS_METHOD_QUOTA,
            "available": cross_available,
            "actual": len(cross_records),
        },
    }


class DenseMapExtractor:
    """Re-run only selected network cases with their locked formal configs."""

    def __init__(self, configs: Mapping[str, Path]) -> None:
        unsupported = sorted(set(configs) - {"G0", "G1", "C0", "C1"})
        if unsupported:
            raise FailureGalleryError(
                f"dense-map configs are supported only for G0/G1/C0/C1: {unsupported}"
            )
        self.configs = {str(name): Path(path).resolve() for name, path in configs.items()}
        self.backends: dict[str, Any] = {}
        self.loader = CompactSampleLoader()

    def _backend(self, method: str) -> Any:
        if method not in self.backends:
            path = self.configs[method]
            if not path.is_file():
                raise FileNotFoundError(path)
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise FailureGalleryError(f"dense-map config is not an object: {path}")
            backend, _ = _build_backend(method_id=method, config=value, oracle=False)
            self.backends[method] = backend
        return self.backends[method]

    def extract(
        self, method: str, deployment: Mapping[str, Any]
    ) -> tuple[dict[str, np.ndarray] | None, str]:
        if method not in self.configs:
            return None, "no locked dense-map config supplied"
        try:
            backend = self._backend(method)
            arrays = self.loader.load(deployment, mask_source="predicted", labels=None)
            sample = BackendSample(
                sample_id=arrays.sample_id,
                rgb=arrays.rgb,
                depth_m=arrays.depth_m,
                predicted_mask=arrays.binary_mask,
                probability_map=arrays.probability,
                mask_source="predicted",
                metadata={"scene_id": arrays.scene_id},
            )
            conditioned = prepare_conditioned_input(sample, backend.config)
            model_input = conditioned.depth_chw
            if method in {"G0", "G1"} and int(backend.config.input_channels) == 4:
                model_input = np.concatenate(
                    (conditioned.depth_chw, conditioned.rgb_chw), axis=0
                )
            tensor = torch.from_numpy(np.ascontiguousarray(model_input)).unsqueeze(0)
            tensor = tensor.to(device=backend.device, dtype=torch.float32)
            with torch.inference_mode():
                outputs = backend.model(tensor)
            synchronize_device(backend.device)
            network_quality, cos_map, sin_map, width_map = official_gaussian_post_process(
                outputs,
                expected_spatial_shape=(
                    backend.config.input_size,
                    backend.config.input_size,
                ),
            )
            quality = gated_quality_map(
                network_quality,
                probability_gate=conditioned.gate_map,
                valid_depth_gate=conditioned.valid_depth_map,
                center_gate_exponent=backend.config.center_gate_exponent,
            )
            return (
                {
                    "quality": np.asarray(quality, dtype=np.float32),
                    "angle_deg": np.rad2deg(
                        0.5 * np.arctan2(sin_map, cos_map)
                    ).astype(np.float32),
                    "width_px": np.asarray(width_map, dtype=np.float32),
                },
                "recomputed from locked formal model/config for selected gallery case",
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return None, f"selected-case dense-map rerun failed: {type(error).__name__}:{error}"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    resized = image.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.BILINEAR)
    scaled_mask = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(
        (PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.NEAREST
    )
    array = np.asarray(resized, dtype=np.float32).copy()
    active = np.asarray(scaled_mask, dtype=np.uint8) > 0
    array[active] = 0.62 * array[active] + 0.38 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _draw_grasp(
    draw: ImageDraw.ImageDraw,
    grasp: Grasp4DoF,
    *,
    native_size: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    corners = rectangle_corners(
        grasp.center_x,
        grasp.center_y,
        grasp.width_px,
        grasp.height_px,
        grasp.angle_deg,
    )
    scale_x = PANEL_WIDTH / native_size[0]
    scale_y = PANEL_HEIGHT / native_size[1]
    points = [
        (float(point[0] * scale_x), float(point[1] * scale_y)) for point in corners
    ]
    draw.line(points + [points[0]], fill=color, width=width, joint="curve")


def _candidate_panel(
    rgb: Image.Image,
    predicted_mask: np.ndarray,
    pool: Sequence[Grasp4DoF],
    *,
    evaluation_outcome: bool | None = None,
) -> Image.Image:
    panel = _overlay_mask(rgb, predicted_mask, (0, 220, 90))
    draw = ImageDraw.Draw(panel)
    neutral = ((255, 214, 10), (0, 210, 255), (175, 110, 255), (255, 145, 0), (240, 240, 240))
    for rank, candidate in reversed(list(enumerate(pool[:5], 1))):
        color = neutral[rank - 1]
        if rank == 1 and evaluation_outcome is not None:
            color = (20, 235, 90) if evaluation_outcome else (255, 70, 70)
        _draw_grasp(
            draw,
            candidate,
            native_size=rgb.size,
            color=color,
            width=4 if rank == 1 else 2,
        )
        if rank == 1:
            x = int(candidate.center_x * PANEL_WIDTH / rgb.size[0])
            y = int(candidate.center_y * PANEL_HEIGHT / rgb.size[1])
            draw.text((x + 4, y + 3), "Top-1", fill=color, font=_font(15, bold=True))
    return panel


def _evaluation_panel(
    rgb: Image.Image,
    gt_mask: np.ndarray,
    gt_rectangles: Sequence[Sequence[Sequence[float]]],
    pool: Sequence[Grasp4DoF],
    *,
    top1_success: bool,
) -> Image.Image:
    panel = _overlay_mask(rgb, gt_mask, (255, 0, 210))
    draw = ImageDraw.Draw(panel)
    for corners in gt_rectangles:
        gt = grasp_from_ocid_corners(corners)
        _draw_grasp(draw, gt, native_size=rgb.size, color=(0, 235, 255), width=3)
    if pool:
        _draw_grasp(
            draw,
            pool[0],
            native_size=rgb.size,
            color=(20, 235, 90) if top1_success else (255, 70, 70),
            width=4,
        )
    return panel


def _label_panel(panel: Image.Image, title: str) -> Image.Image:
    result = panel.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, PANEL_WIDTH, 34), fill=(0, 0, 0, 170))
    draw.text(
        (7, 7),
        title,
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
        font=_font(15, bold=True),
    )
    return result


def _linear_heatmap(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(array)
    scaled = np.zeros(array.shape, dtype=np.float32)
    if finite.any():
        low, high = np.percentile(array[finite], [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        scaled[finite] = np.clip((array[finite] - low) / (high - low), 0.0, 1.0)
    red = np.clip(1.5 * scaled, 0.0, 1.0)
    green = np.clip(1.5 - 2.0 * np.abs(scaled - 0.55), 0.0, 1.0)
    blue = np.clip(1.4 * (1.0 - scaled), 0.0, 1.0)
    return (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)


def _angle_heatmap(angle_deg: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=np.float32)
    hue = np.mod(angle + 90.0, 180.0) / 180.0
    hsv = np.stack(
        (
            hue * 255.0,
            np.full_like(hue, 225.0),
            np.full_like(hue, 235.0),
        ),
        axis=-1,
    ).astype(np.uint8)
    return np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"))


def _heatmap_panel(value: np.ndarray, *, title: str, angle: bool = False) -> Image.Image:
    rgb = _angle_heatmap(value) if angle else _linear_heatmap(value)
    panel = Image.fromarray(rgb).resize(
        (PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.BILINEAR
    )
    return _label_panel(panel, title)


def _wrapped_lines(text: str, width: int = 105) -> list[str]:
    return textwrap.wrap(str(text), width=width, break_long_words=False) or [""]


def _render_case(
    *,
    method: str,
    row: Mapping[str, Any],
    frozen: Mapping[str, Any],
    pool: Sequence[Grasp4DoF],
    categories: Sequence[str],
    dense_maps: Mapping[str, np.ndarray] | None = None,
    dense_maps_reason: str = "formal artifacts do not contain dense-map arrays",
) -> Image.Image:
    rgb = _load_rgb(frozen["source_rgb_path"])
    predicted_mask = _load_mask(
        frozen["predicted_mask_path"], size=rgb.size, label="predicted mask"
    )
    gt_mask = _load_mask(
        frozen["prepared_gt_mask_path"], size=rgb.size, label="GT mask"
    )
    lines = _wrapped_lines(f"Language: {frozen['language']}")
    header_height = 94 + 22 * len(lines)
    footer_height = 92
    columns = 3 if dense_maps is not None else 2
    panel_rows = 2 if dense_maps is not None else 1
    canvas = Image.new(
        "RGB",
        (
            columns * PANEL_WIDTH + (columns - 1) * PANEL_GAP + 2 * MARGIN,
            header_height
            + panel_rows * PANEL_HEIGHT
            + (panel_rows - 1) * PANEL_GAP
            + footer_height,
        ),
        (18, 22, 29),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (MARGIN, 14),
        f"{method} | {row['sample_id']} | {', '.join(categories)}",
        fill=(245, 247, 250),
        font=_font(21, bold=True),
    )
    y = 45
    for line in lines:
        draw.text((MARGIN, y), line, fill=(205, 213, 224), font=_font(17))
        y += 21
    left = _label_panel(
        _candidate_panel(rgb, predicted_mask, pool),
        "INFERENCE: predicted mask + Top-5",
    )
    right = _label_panel(
        _evaluation_panel(
        rgb,
        gt_mask,
        frozen["gt_grasp_rectangles"],
        pool,
        top1_success=bool(row["j_at_1"]),
        ),
        "EVALUATION ONLY: GT mask + GT grasps + Top-1",
    )
    panel_y = header_height
    canvas.paste(left, (MARGIN, panel_y))
    if dense_maps is None:
        canvas.paste(right, (MARGIN + PANEL_WIDTH + PANEL_GAP, panel_y))
    else:
        quality = _heatmap_panel(dense_maps["quality"], title="GATED QUALITY MAP")
        angle = _heatmap_panel(
            dense_maps["angle_deg"], title="ANGLE MAP (180° periodic)", angle=True
        )
        width = _heatmap_panel(dense_maps["width_px"], title="OPENING WIDTH MAP (px)")
        second_y = panel_y + PANEL_HEIGHT + PANEL_GAP
        canvas.paste(quality, (MARGIN + PANEL_WIDTH + PANEL_GAP, panel_y))
        canvas.paste(right, (MARGIN + 2 * (PANEL_WIDTH + PANEL_GAP), panel_y))
        canvas.paste(angle, (MARGIN, second_y))
        canvas.paste(width, (MARGIN + PANEL_WIDTH + PANEL_GAP, second_y))
        note = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), (28, 34, 44))
        note_draw = ImageDraw.Draw(note)
        note_draw.text((18, 24), "DENSE MAP PROVENANCE", fill=(245, 232, 135), font=_font(18, bold=True))
        for index, line in enumerate(_wrapped_lines(dense_maps_reason, width=47)[:8]):
            note_draw.text((18, 62 + 25 * index), line, fill=(210, 218, 230), font=_font(16))
        note_draw.text((18, 270), "Selected cases only; no GT used in rerun.", fill=(125, 225, 170), font=_font(15, bold=True))
        canvas.paste(note, (MARGIN + 2 * (PANEL_WIDTH + PANEL_GAP), second_y))
    footer_y = (
        panel_y
        + panel_rows * PANEL_HEIGHT
        + (panel_rows - 1) * PANEL_GAP
        + 12
    )
    metrics = (
        f"Outcome={'success' if row['j_at_1'] else 'failure'} | stage={row['failure_stage']} | "
        f"IoU={_fmt(row['top1_rectangle_iou'])} | angle error={_fmt(row['top1_angle_difference_deg'])}° | "
        f"candidates={row['candidate_count']}"
    )
    draw.text((MARGIN, footer_y), metrics, fill=(244, 226, 118), font=_font(16, bold=True))
    draw.text(
        (MARGIN, footer_y + 27),
        (
            f"Dense maps: available — {dense_maps_reason}"
            if dense_maps is not None
            else f"Dense maps: unavailable — {dense_maps_reason}; no map image was fabricated."
        ),
        fill=(125, 225, 170) if dense_maps is not None else (255, 150, 145),
        font=_font(15),
    )
    draw.text(
        (MARGIN, footer_y + 51),
        str(row["failure_rule"]),
        fill=(195, 205, 219),
        font=_font(14),
    )
    return canvas


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(number) else f"{number:.3f}"


def _render_cross_case(
    *,
    sample_id: str,
    methods: Sequence[str],
    rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    frozen: Mapping[str, Any],
    pools_by_method: Mapping[str, Mapping[str, Sequence[Grasp4DoF]]],
) -> Image.Image:
    rgb = _load_rgb(frozen["source_rgb_path"])
    predicted_mask = _load_mask(
        frozen["predicted_mask_path"], size=rgb.size, label="predicted mask"
    )
    gt_mask = _load_mask(
        frozen["prepared_gt_mask_path"], size=rgb.size, label="GT mask"
    )
    panel_count = len(methods) + 1
    columns = min(3, panel_count)
    rows = int(math.ceil(panel_count / columns))
    header_height = 100
    caption_height = 54
    cell_height = PANEL_HEIGHT + caption_height
    canvas = Image.new(
        "RGB",
        (
            columns * PANEL_WIDTH + (columns - 1) * PANEL_GAP + 2 * MARGIN,
            header_height + rows * cell_height + (rows - 1) * PANEL_GAP + MARGIN,
        ),
        (18, 22, 29),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (MARGIN, 13),
        f"Cross-method evaluation | {sample_id}",
        fill=(245, 247, 250),
        font=_font(22, bold=True),
    )
    language_lines = _wrapped_lines(f"Language: {frozen['language']}", width=115)
    for index, line in enumerate(language_lines[:2]):
        draw.text((MARGIN, 45 + 20 * index), line, fill=(205, 213, 224), font=_font(16))

    for index, method in enumerate(methods):
        grid_row, grid_col = divmod(index, columns)
        x = MARGIN + grid_col * (PANEL_WIDTH + PANEL_GAP)
        y = header_height + grid_row * (cell_height + PANEL_GAP)
        row = rows_by_key[(method, sample_id)]
        panel = _candidate_panel(
            rgb,
            predicted_mask,
            pools_by_method[method][sample_id],
            evaluation_outcome=bool(row["j_at_1"]),
        )
        canvas.paste(panel, (x, y))
        draw.text(
            (x, y + PANEL_HEIGHT + 5),
            f"{method}: {'success' if row['j_at_1'] else 'failure'} | {row['failure_stage']}",
            fill=(245, 232, 135),
            font=_font(15, bold=True),
        )
        draw.text(
            (x, y + PANEL_HEIGHT + 27),
            f"Top-1 IoU={_fmt(row['top1_rectangle_iou'])}, angle={_fmt(row['top1_angle_difference_deg'])}°; maps unavailable",
            fill=(200, 210, 222),
            font=_font(13),
        )

    index = len(methods)
    grid_row, grid_col = divmod(index, columns)
    x = MARGIN + grid_col * (PANEL_WIDTH + PANEL_GAP)
    y = header_height + grid_row * (cell_height + PANEL_GAP)
    gt_panel = _evaluation_panel(
        rgb,
        gt_mask,
        frozen["gt_grasp_rectangles"],
        (),
        top1_success=False,
    )
    canvas.paste(gt_panel, (x, y))
    draw.text(
        (x, y + PANEL_HEIGHT + 5),
        "EVALUATION ONLY: GT mask + GT grasps",
        fill=(0, 235, 255),
        font=_font(15, bold=True),
    )
    return canvas


def _safe_image_name(*parts: str) -> str:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"case_{digest}.png"


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _build_html(
    *,
    gallery_dir: Path,
    method_cards: Sequence[Mapping[str, Any]],
    cross_cards: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    stage_counts: Mapping[str, Mapping[str, int]],
) -> str:
    css = """
body{background:#0f131a;color:#edf1f7;font:15px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;padding:24px}
h1,h2,h3{color:#fff}.note{background:#242b36;border-left:4px solid #ff8a80;padding:12px 16px;margin:16px 0}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}.summary pre{white-space:pre-wrap;background:#171d26;padding:12px;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:18px}.card{background:#171d26;border:1px solid #333d4b;border-radius:10px;padding:12px}.card img{width:100%;height:auto;border-radius:6px}.meta{color:#bcc7d6;margin-top:8px;overflow-wrap:anywhere}.tag{display:inline-block;background:#30405a;color:#fff;border-radius:12px;padding:3px 9px;margin:2px}
"""
    sections: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<style>{css}</style><title>Formal 4-DoF failure gallery</title></head><body>",
        "<h1>Formal 4-DoF failure gallery</h1>",
        "<div class='note'><strong>Post-hoc test-only analysis.</strong> Ground truth appears only in evaluation panels. "
        "This gallery performs no threshold, model, checkpoint, or configuration selection. Dense maps in formal artifacts "
        "are <code>unavailable</code>; when locked configs are supplied, selected network maps are rerun from that exact "
        "model/config and labelled accordingly. No map is fabricated.</div>",
        "<div class='summary'>",
    ]
    for method in sorted(selection["counts"]):
        count_text = "\n".join(
            f"{category}: {values['actual']}/{values['requested']}"
            for category, values in selection["counts"][method].items()
        )
        stages = "\n".join(
            f"{stage}: {count}" for stage, count in sorted(stage_counts[method].items())
        )
        sections.append(
            f"<pre><strong>{html.escape(method)}</strong>\nSelected:\n{html.escape(count_text)}\n\nStages:\n{html.escape(stages)}</pre>"
        )
    sections.append("</div><h2>Per-method selected cases</h2><div class='grid'>")
    for card in method_cards:
        image_uri = _data_uri(gallery_dir / str(card["image_path"]))
        tags = "".join(
            f"<span class='tag'>{html.escape(str(tag))}</span>" for tag in card["categories"]
        )
        sections.append(
            "<article class='card'>"
            f"<h3>{html.escape(str(card['method']))} · {html.escape(str(card['sample_id']))}</h3>"
            f"<div>{tags}</div><img loading='lazy' src='{image_uri}' alt='selected failure case'>"
            f"<div class='meta'>stage={html.escape(str(card['failure_stage']))}; "
            f"saved PNG: {html.escape(str(card['image_path']))}</div></article>"
        )
    sections.append("</div><h2>Cross-method comparisons</h2><div class='grid'>")
    for card in cross_cards:
        image_uri = _data_uri(gallery_dir / str(card["image_path"]))
        sections.append(
            "<article class='card'>"
            f"<h3>{html.escape(str(card['sample_id']))}</h3>"
            f"<img loading='lazy' src='{image_uri}' alt='cross-method comparison'>"
            f"<div class='meta'>outcome disagreement={str(bool(card['outcome_disagreement'])).lower()}; "
            f"stage diversity={int(card['stage_diversity'])}; saved PNG: "
            f"{html.escape(str(card['image_path']))}</div></article>"
        )
    sections.append("</div></body></html>")
    return "".join(sections)


def build_failure_gallery(
    *,
    run_dir: str | Path,
    method_dirs: Sequence[tuple[str, str | Path]],
    dense_map_configs: Sequence[tuple[str, str | Path]] = (),
) -> dict[str, Any]:
    """Build the analysis table, selected PNGs, manifest, and standalone HTML."""

    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    if not method_dirs:
        raise FailureGalleryError("at least one method directory is required")
    normalized: list[tuple[str, Path]] = []
    for raw_name, raw_path in method_dirs:
        name = str(raw_name).strip()
        path = Path(raw_path).expanduser().resolve()
        if not name or "=" in name:
            raise FailureGalleryError(f"invalid method name: {raw_name!r}")
        if not path.is_dir():
            raise FileNotFoundError(f"method directory does not exist: {path}")
        normalized.append((name, path))
    normalized.sort(key=lambda item: item[0])
    if len({name for name, _ in normalized}) != len(normalized):
        raise FailureGalleryError("duplicate method name")
    dense_configs: dict[str, Path] = {}
    for raw_name, raw_path in dense_map_configs:
        name = str(raw_name).strip()
        if name in dense_configs:
            raise FailureGalleryError(f"duplicate dense-map config: {name}")
        if name not in {item[0] for item in normalized}:
            raise FailureGalleryError(f"dense-map config has no method directory: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        dense_configs[name] = path
    dense_extractor = DenseMapExtractor(dense_configs)

    analysis_path = run / "per_sample_failure_stage.parquet"
    gallery_path = run / "gallery"
    if analysis_path.exists() or gallery_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing failure analysis/gallery output"
        )

    sample_ids, frozen = _load_frozen_inputs(run)
    analysis_rows: list[dict[str, Any]] = []
    pools_by_method: dict[str, dict[str, list[Grasp4DoF]]] = {}
    for method, directory in normalized:
        rows, pools = _analyse_method(
            method=method,
            method_dir=directory,
            sample_ids=sample_ids,
            frozen=frozen,
        )
        analysis_rows.extend(rows)
        pools_by_method[method] = pools
    analysis = pd.DataFrame(analysis_rows)
    if dense_configs:
        selected_map_methods = analysis["method"].astype(str).isin(dense_configs)
        analysis.loc[selected_map_methods, "dense_maps_status"] = (
            "available_for_selected_gallery_rerun"
        )
        analysis.loc[selected_map_methods, "dense_maps_reason"] = (
            "locked formal model/config supplied; arrays recomputed only for selected gallery cases"
        )
    if not set(analysis["failure_stage"]).issubset(set(FAILURE_STAGES)):
        raise AssertionError("internal failure-stage taxonomy drift")
    selection = select_gallery_cases(analysis)
    rows_by_key = {
        (str(row["method"]), str(row["sample_id"])): row
        for row in analysis.to_dict("records")
    }

    temporary_gallery = Path(tempfile.mkdtemp(prefix=".failure_gallery.", dir=run))
    temporary_analysis = run / f".per_sample_failure_stage.{os.getpid()}.tmp.parquet"
    try:
        images = temporary_gallery / "images"
        images.mkdir()
        category_by_key: dict[tuple[str, str], list[str]] = {}
        for selected in selection["per_method"]:
            key = (selected["method"], selected["sample_id"])
            category_by_key.setdefault(key, []).append(selected["category"])
        method_cards: list[dict[str, Any]] = []
        for method, sample_id in sorted(category_by_key):
            categories = sorted(category_by_key[(method, sample_id)])
            row = rows_by_key[(method, sample_id)]
            filename = _safe_image_name("method", method, sample_id)
            relative = Path("images") / filename
            dense_maps, dense_reason = dense_extractor.extract(
                method, frozen[sample_id]
            )
            image = _render_case(
                method=method,
                row=row,
                frozen=frozen[sample_id],
                pool=pools_by_method[method][sample_id],
                categories=categories,
                dense_maps=dense_maps,
                dense_maps_reason=dense_reason,
            )
            _save_png(image, temporary_gallery / relative)
            method_cards.append(
                {
                    "method": method,
                    "sample_id": sample_id,
                    "categories": categories,
                    "failure_stage": row["failure_stage"],
                    "image_path": relative.as_posix(),
                }
            )

        methods = [name for name, _ in normalized]
        cross_cards: list[dict[str, Any]] = []
        for selected in selection["cross_method"]:
            sample_id = str(selected["sample_id"])
            filename = _safe_image_name("cross", sample_id, *methods)
            relative = Path("images") / filename
            image = _render_cross_case(
                sample_id=sample_id,
                methods=methods,
                rows_by_key=rows_by_key,
                frozen=frozen[sample_id],
                pools_by_method=pools_by_method,
            )
            _save_png(image, temporary_gallery / relative)
            cross_cards.append({**selected, "image_path": relative.as_posix()})

        stage_counts = {
            method: dict(
                sorted(
                    Counter(
                        analysis.loc[analysis["method"] == method, "failure_stage"].astype(str)
                    ).items()
                )
            )
            for method in methods
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "analysis_scope": ANALYSIS_SCOPE,
            "configuration_selection_performed": False,
            "selection_policy": (
                "deterministic diagnostic-severity ordering with sample_id tie-break; "
                "cross-method disagreements first"
            ),
            "failure_stage_taxonomy": list(FAILURE_STAGES),
            "assignment_precedence": [
                "successful",
                "grounding_wrong_target",
                "grounding_fragmented_mask",
                "empty_mask",
                "invalid_depth",
                "no_candidate_generated",
                "ranking_failure",
                "angle_failure",
                "width_failure",
                "candidate_pool_has_no_positive",
                "evaluator_ambiguity",
            ],
            "dense_maps_status": (
                "available_for_selected_network_cases" if dense_configs else "unavailable"
            ),
            "dense_maps_reason": (
                "network maps are recomputed from locked model/config only for selected cases; "
                "formal candidate outputs remain unchanged"
                if dense_configs
                else "formal output schema does not persist reconstructable dense-map arrays"
            ),
            "dense_map_configs": {name: str(path) for name, path in dense_configs.items()},
            "method_directories": {name: str(path) for name, path in normalized},
            "sample_count": len(sample_ids),
            "method_count": len(methods),
            "stage_counts": stage_counts,
            "selection": selection,
            "rendered_method_image_count": len(method_cards),
            "rendered_cross_method_image_count": len(cross_cards),
            "html_dependencies": "embedded CSS and selected PNGs embedded as data URIs",
        }
        (temporary_gallery / "selection_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (temporary_gallery / "index.html").write_text(
            _build_html(
                gallery_dir=temporary_gallery,
                method_cards=method_cards,
                cross_cards=cross_cards,
                selection=selection,
                stage_counts=stage_counts,
            ),
            encoding="utf-8",
        )
        analysis.to_parquet(temporary_analysis, index=False, compression="zstd")
        os.replace(temporary_analysis, analysis_path)
        os.replace(temporary_gallery, gallery_path)
    except BaseException:
        temporary_analysis.unlink(missing_ok=True)
        if temporary_gallery.exists():
            shutil.rmtree(temporary_gallery)
        raise

    return {
        "status": "COMPLETE",
        "analysis_scope": ANALYSIS_SCOPE,
        "configuration_selection_performed": False,
        "analysis_path": str(analysis_path),
        "gallery_path": str(gallery_path),
        "index_path": str(gallery_path / "index.html"),
        "sample_count": len(sample_ids),
        "method_count": len(normalized),
        "analysis_row_count": len(analysis),
        "rendered_method_image_count": len(method_cards),
        "rendered_cross_method_image_count": len(cross_cards),
        "selection": selection,
    }


def _method_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name.strip(), Path(path).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--method-dir",
        required=True,
        action="append",
        type=_method_spec,
        metavar="NAME=PATH",
        help="formal method output directory; repeat for every method",
    )
    parser.add_argument(
        "--dense-map-config",
        action="append",
        type=_method_spec,
        default=[],
        metavar="NAME=PATH",
        help="locked G0/G1/C0/C1 config used to recompute maps for selected cases",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_failure_gallery(
        run_dir=args.run_dir,
        method_dirs=args.method_dir,
        dense_map_configs=args.dense_map_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
