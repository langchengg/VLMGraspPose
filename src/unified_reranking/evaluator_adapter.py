"""Adapter around the byte-frozen fair evaluator and development-only labels."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from .hashing import sha256_file


def load_frozen_evaluator(path: str | Path, expected_sha256: str | None = None) -> ModuleType:
    source = Path(path).resolve()
    observed = sha256_file(source)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            f"frozen evaluator hash mismatch: observed={observed} expected={expected_sha256}"
        )
    module_name = f"_unified_reranking_frozen_evaluator_{observed[:16]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # A failed import must not poison the process-wide cache.  Otherwise a
        # later call can receive a partially initialised evaluator module and
        # fail in a misleading place.
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    return module


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _candidate_grasp(module: ModuleType, row: Any) -> Any:
    return module.CanonicalGrasp(
        cx_px=float(row.cx_px),
        cy_px=float(row.cy_px),
        theta_deg=float(row.theta_deg),
        jaw_width_px=float(row.width_px),
        rectangle_height_px=float(row.height_px),
        native_score=float(row.native_score),
        native_rank=int(row.native_rank),
        source_method=str(row.route),
        sample_id=str(row.sample_id),
    )


def _label_one(module: ModuleType, candidate: Any, gt_corners: Any) -> dict[str, Any]:
    # pandas/PyArrow materialises nested fixed-size lists as object arrays of
    # numeric arrays under NumPy 2.x.  Normalise each rectangle explicitly so
    # the frozen evaluator receives a conventional finite 4x2 float array.
    normalized = (
        np.stack([np.asarray(point, dtype=np.float64) for point in rectangle])
        for rectangle in gt_corners
    )
    ground_truth = tuple(module.gt_from_corners(item) for item in normalized)
    evaluated = module.evaluate_candidate(candidate, ground_truth)
    pairwise = evaluated["pairwise"]
    margins = [
        float(
            np.clip(
                min(
                    (float(pair["iou"]) - module.IOU_THRESHOLD) / module.IOU_THRESHOLD,
                    (module.ANGLE_THRESHOLD_DEG - float(pair["angle_error_deg"]))
                    / module.ANGLE_THRESHOLD_DEG,
                ),
                -1.0,
                1.0,
            )
        )
        for pair in pairwise
    ]
    if pairwise:
        index = max(
            range(len(pairwise)),
            key=lambda item: (
                margins[item],
                float(pairwise[item]["iou"]),
                -float(pairwise[item]["angle_error_deg"]),
                -int(pairwise[item]["gt_index"]),
            ),
        )
        match = pairwise[index]
        margin = margins[index]
        matched_index: int | None = int(match["gt_index"])
        matched_iou = float(match["iou"])
        matched_angle = float(match["angle_error_deg"])
    else:
        matched_index = None
        matched_iou = math.nan
        matched_angle = math.nan
        margin = -1.0
    return {
        "candidate_success": bool(evaluated["success"]),
        "best_same_gt_iou": matched_iou,
        "best_same_gt_angle_error_deg": matched_angle,
        "matched_gt_index": matched_index,
        "jacquard_margin": margin,
    }


def annotate_pool_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach pool-specific solvability and first-positive rank."""

    result = frame.drop(columns=["pool_solvable", "first_positive_rank"], errors="ignore").copy()
    if result.empty:
        result["pool_solvable"] = pd.Series(dtype=bool)
        result["first_positive_rank"] = pd.Series(dtype="Int16")
        return result
    positive_rank = result["native_rank"].where(result["candidate_success"].astype(bool))
    first_positive = positive_rank.groupby(result["sample_id"], sort=False).min()
    result["pool_solvable"] = result["sample_id"].map(first_positive.notna()).astype(bool)
    result["first_positive_rank"] = result["sample_id"].map(first_positive).astype("Int16")
    return result


def evaluate_candidate_rows_with_frozen_evaluator(
    candidates: pd.DataFrame,
    sample_labels: pd.DataFrame,
    evaluator_path: Path,
    evaluator_sha256: str,
) -> pd.DataFrame:
    """Evaluate authorized rows with the frozen evaluator, preserving angles.

    This primitive intentionally makes no split/access decision.  A caller
    handling Test ground truth must establish the formal execution claim before
    loading ``sample_labels``.
    """

    candidate_required = {
        "sample_id",
        "route",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    }
    label_required = {"sample_id", "gt_grasp_rectangles"}
    missing_candidates = sorted(candidate_required.difference(candidates.columns))
    missing_labels = sorted(label_required.difference(sample_labels.columns))
    if missing_candidates:
        raise ValueError(f"candidate evaluator input misses columns: {missing_candidates}")
    if missing_labels:
        raise ValueError(f"sample evaluator input misses columns: {missing_labels}")
    labels = sample_labels[list(label_required)].copy()
    labels["sample_id"] = labels["sample_id"].astype(str)
    if (
        labels.empty
        or labels["sample_id"].eq("").any()
        or labels["sample_id"].duplicated().any()
    ):
        raise ValueError("sample evaluator input must have unique non-empty sample IDs")
    work = candidates.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    joined = work.merge(labels, on="sample_id", how="left", validate="many_to_one")
    if joined["gt_grasp_rectangles"].isna().any():
        raise ValueError("candidate evaluator input references a sample without ground truth")
    module = load_frozen_evaluator(evaluator_path, evaluator_sha256)
    evaluated = [
        _label_one(module, _candidate_grasp(module, row), row.gt_grasp_rectangles)
        for row in joined.itertuples(index=False)
    ]
    return pd.concat(
        [
            work.reset_index(drop=True),
            pd.DataFrame(evaluated).reset_index(drop=True),
        ],
        axis=1,
    )


def build_candidate_labels(
    candidates_path: Path,
    sample_labels_path: Path,
    destination: Path,
    evaluator_path: Path,
    evaluator_sha256: str,
    *,
    split: str,
) -> dict[str, Any]:
    """Evaluate Train/Validation candidates; Test is denied until the formal evaluator."""

    if split not in {"train", "validation"}:
        raise PermissionError("development label builder may read Train/Validation labels only")
    candidates = pd.read_parquet(candidates_path)
    labels = pd.read_parquet(
        sample_labels_path,
        columns=["sample_id", "gt_grasp_rectangles"],
    )
    if labels["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample labels: {sample_labels_path}")
    gt_by_sample = labels.set_index("sample_id")["gt_grasp_rectangles"].to_dict()
    missing = sorted(set(candidates["sample_id"]).difference(gt_by_sample))
    if missing:
        raise ValueError(f"candidate samples missing GT labels: {missing[:5]}")
    module = load_frozen_evaluator(evaluator_path, evaluator_sha256)
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        result = _label_one(module, _candidate_grasp(module, row), gt_by_sample[row.sample_id])
        rows.append(
            {
                "sample_id": row.sample_id,
                "candidate_id": row.candidate_id,
                "native_rank": int(row.native_rank),
                **result,
            }
        )
    frame = pd.DataFrame(rows)
    frame = annotate_pool_labels(frame)
    _atomic_parquet(frame, destination)
    return {
        "candidate_rows": len(frame),
        "samples": int(frame["sample_id"].nunique()) if len(frame) else 0,
        "positive_candidates": int(frame["candidate_success"].sum()) if len(frame) else 0,
        "solvable_samples": int(frame.loc[frame["pool_solvable"], "sample_id"].nunique())
        if len(frame)
        else 0,
        "destination": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "evaluator_sha256": evaluator_sha256,
    }
