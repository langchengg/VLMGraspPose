"""Mask metrics used only after SAM hypothesis selection is frozen."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def _binary_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be aligned 2D masks")
    return prediction, target


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _binary_pair(prediction, target)
    union = int(np.count_nonzero(prediction | target))
    return 1.0 if union == 0 else float(np.count_nonzero(prediction & target) / union)


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _binary_pair(prediction, target)
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 1.0 if denominator == 0 else float(2 * np.count_nonzero(prediction & target) / denominator)


def boundary_fscore(prediction: np.ndarray, target: np.ndarray, tolerance_px: int = 2) -> float:
    prediction, target = _binary_pair(prediction, target)
    pred_boundary = prediction ^ ndimage.binary_erosion(prediction)
    target_boundary = target ^ ndimage.binary_erosion(target)
    if not np.any(pred_boundary) and not np.any(target_boundary):
        return 1.0
    if not np.any(pred_boundary) or not np.any(target_boundary):
        return 0.0
    pred_dilated = ndimage.binary_dilation(pred_boundary, iterations=max(1, int(tolerance_px)))
    target_dilated = ndimage.binary_dilation(target_boundary, iterations=max(1, int(tolerance_px)))
    precision = float(np.mean(target_dilated[pred_boundary]))
    recall = float(np.mean(pred_dilated[target_boundary]))
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def component_statistics(mask: np.ndarray) -> tuple[int, float]:
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0, 0.0
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return int(count), float(np.max(sizes) / area)


def evaluate_mask(prediction: np.ndarray, target: np.ndarray, *, boundary_tolerance_px: int = 2) -> dict[str, Any]:
    prediction, target = _binary_pair(prediction, target)
    false_positive = int(np.count_nonzero(prediction & ~target))
    false_negative = int(np.count_nonzero(~prediction & target))
    components, largest = component_statistics(prediction)
    return {
        "iou": mask_iou(prediction, target),
        "dice": dice_score(prediction, target),
        "boundary_fscore": boundary_fscore(prediction, target, boundary_tolerance_px),
        "false_positive_area_px": false_positive,
        "false_negative_area_px": false_negative,
        "connected_component_count": components,
        "main_component_ratio": largest,
        "prediction_area_px": int(np.count_nonzero(prediction)),
        "ground_truth_area_px": int(np.count_nonzero(target)),
    }
