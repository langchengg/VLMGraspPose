"""Binary visual-grounding metrics with the frozen evaluator conventions."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy import ndimage


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def evaluator_iou_float32(prediction: np.ndarray, target: np.ndarray) -> float:
    """Reproduce the authoritative per-sample float32 IoU calculation."""

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"mask shapes differ: {prediction.shape} != {target.shape}")
    intersection = np.float32(np.logical_and(prediction, target).sum())
    union = np.float32(np.logical_or(prediction, target).sum())
    return 1.0 if union == 0 else float(np.float32(intersection / union))


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(mask)
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))


def boundary_fscore(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    tolerance_px: int = 2,
) -> float:
    return float(
        boundary_fscores([prediction], target, tolerance_px=tolerance_px)[0]
    )


def boundary_fscores(
    predictions: Iterable[np.ndarray],
    target: np.ndarray,
    *,
    tolerance_px: int = 2,
) -> np.ndarray:
    """Compute multiple boundary F-scores while reusing the target distance map."""

    target_boundary = _boundary(target)
    target_count = int(target_boundary.sum())
    target_distance = (
        ndimage.distance_transform_edt(~target_boundary)
        if target_count
        else None
    )
    results: list[float] = []
    for prediction in predictions:
        pred_boundary = _boundary(prediction)
        pred_count = int(pred_boundary.sum())
        if pred_count == 0 and target_count == 0:
            results.append(1.0)
            continue
        if pred_count == 0 or target_count == 0 or target_distance is None:
            results.append(0.0)
            continue
        pred_distance = ndimage.distance_transform_edt(~pred_boundary)
        precision = float(
            np.count_nonzero(pred_boundary & (target_distance <= tolerance_px))
            / pred_count
        )
        recall = float(
            np.count_nonzero(target_boundary & (pred_distance <= tolerance_px))
            / target_count
        )
        results.append(
            0.0
            if precision + recall == 0.0
            else float(2.0 * precision * recall / (precision + recall))
        )
    return np.asarray(results, dtype=np.float64)


def binary_mask_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    boundary_tolerance_px: int = 2,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target masks must be aligned")
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    pred_area = true_positive + false_positive
    target_area = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    iou = 1.0 if union == 0 else float(true_positive / union)
    precision = 1.0 if pred_area == 0 and target_area == 0 else (
        0.0 if pred_area == 0 else float(true_positive / pred_area)
    )
    recall = 1.0 if target_area == 0 else float(true_positive / target_area)
    dice_denom = pred_area + target_area
    dice = 1.0 if dice_denom == 0 else float(2 * true_positive / dice_denom)
    labels, component_count = ndimage.label(prediction)
    component_areas = np.bincount(labels.ravel())[1:] if component_count else np.asarray([], dtype=int)
    largest_component_ratio = (
        0.0 if pred_area == 0 else float(component_areas.max(initial=0) / pred_area)
    )
    return {
        "iou": iou,
        "evaluator_iou_float32": evaluator_iou_float32(prediction, target),
        "dice": dice,
        "mask_precision": precision,
        "mask_recall": recall,
        "boundary_fscore": boundary_fscore(
            prediction, target, tolerance_px=boundary_tolerance_px
        ),
        "true_positive_area_px": true_positive,
        "false_positive_area_px": false_positive,
        "false_negative_area_px": false_negative,
        "predicted_area_px": pred_area,
        "target_area_px": target_area,
        "connected_component_count": int(component_count),
        "largest_component_ratio": largest_component_ratio,
    }


def summarize_ious(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("IoU values must be a non-empty finite vector")
    result: dict[str, Any] = {
        "samples": int(len(array)),
        "mean_iou": float(array.mean()),
        "median_iou": float(np.median(array)),
        "min_iou": float(array.min()),
        "max_iou": float(array.max()),
        "zero_iou_count": int(np.count_nonzero(array == 0.0)),
        "iou_le_0_25_count": int(np.count_nonzero(array <= 0.25)),
    }
    for threshold in THRESHOLDS:
        percent = int(round(threshold * 100))
        numerator = int(np.count_nonzero(array > threshold))
        result[f"p_at_{percent}"] = float(numerator / len(array))
        result[f"p_at_{percent}_numerator"] = numerator
        result[f"p_at_{percent}_denominator"] = int(len(array))
    return result


def paper_gap_requirements(
    baseline: dict[str, Any],
    *,
    paper_mean_iou: float,
    paper_p_at: dict[int, float],
) -> dict[str, Any]:
    denominator = int(baseline["samples"])
    thresholds: dict[str, Any] = {}
    for percent, paper_value in sorted(paper_p_at.items()):
        current = int(baseline[f"p_at_{percent}_numerator"])
        required = int(math.ceil(float(paper_value) * denominator))
        thresholds[f"p_at_{percent}"] = {
            "paper_numeric_reference": float(paper_value),
            "required_successes": required,
            "baseline_successes": current,
            "required_additional_successes": required - current,
            "denominator": denominator,
        }
    return {
        "denominator": denominator,
        "paper_mean_iou_numeric_reference": float(paper_mean_iou),
        "baseline_mean_iou": float(baseline["mean_iou"]),
        "required_total_iou_gain": float(
            (float(paper_mean_iou) - float(baseline["mean_iou"])) * denominator
        ),
        "thresholds": thresholds,
    }
