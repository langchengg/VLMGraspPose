"""Canonical CROG grasp and mask evaluation kernels.

``legacy_official_impl_v1`` preserves the historical public implementation for
auditing only.  All new success/failure decisions must use
``corrected_geometric_v2``.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
from skimage.draw import polygon as draw_polygon


LEGACY_EVALUATOR_VERSION = "legacy_official_impl_v1"
CORRECTED_EVALUATOR_VERSION = "corrected_geometric_v2"
DEFAULT_IMAGE_SHAPE = (480, 640)
DEFAULT_IOU_THRESHOLD = 0.25
DEFAULT_ANGLE_THRESHOLD_DEG = 30.0
OFFICIAL_GT_WIDTH_LIMIT_PX = 100.0
OFFICIAL_RECTANGLE_HEIGHT_PX = 20.0


def periodic_angle_difference_deg(angle_a, angle_b):
    """Return the parallel-jaw angle difference in ``[0, 90]`` degrees."""
    return abs(((float(angle_a) - float(angle_b) + 90.0) % 180.0) - 90.0)


def periodic_angle_difference_rad(angle_a, angle_b):
    """Return the parallel-jaw angle difference in ``[0, pi/2]`` radians."""
    return abs(((float(angle_a) - float(angle_b) + math.pi / 2.0) % math.pi) - math.pi / 2.0)


def legacy_angle_compatible(angle_a, angle_b, threshold=DEFAULT_ANGLE_THRESHOLD_DEG):
    return bool(
        abs(float(angle_a) - float(angle_b)) <= float(threshold)
        or abs(float(angle_a) + float(angle_b)) <= float(threshold)
    )


def joint_success(
    rectangle_iou_value,
    angle_difference_deg_value,
    *,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
    angle_threshold=DEFAULT_ANGLE_THRESHOLD_DEG,
):
    """Apply CROG's strict IoU and inclusive angle boundaries without rounding."""
    return bool(
        float(rectangle_iou_value) > float(iou_threshold)
        and float(angle_difference_deg_value) <= float(angle_threshold)
    )


def _as_grasp(grasp):
    if isinstance(grasp, Mapping):
        legacy = grasp.get("legacy_grasp")
        if legacy is not None:
            return [float(value) for value in legacy[:5]]
        return [
            float(grasp["cx"]),
            float(grasp["cy"]),
            float(grasp["width_px"]),
            float(grasp["height_px"]),
            float(grasp["angle_deg"]),
        ]
    values = list(grasp)
    if len(values) < 5:
        raise ValueError("a grasp must contain x, y, width, height, and angle")
    return [float(value) for value in values[:5]]


def normalize_official_gt_rectangle(grasp):
    """Preserve the official evaluator's documented GT width/height convention."""
    values = _as_grasp(grasp)
    values[2] = float(np.clip(values[2], 0.0, OFFICIAL_GT_WIDTH_LIMIT_PX))
    values[3] = OFFICIAL_RECTANGLE_HEIGHT_PX
    return values


def rectangle_vertices(grasp):
    x, y, width, height, angle = _as_grasp(grasp)
    return cv2.boxPoints(((x, y), (width, height), -angle))


def _corrected_rectangle_pixels(grasp, shape):
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("shape must contain positive height and width")
    box = np.asarray(rectangle_vertices(grasp), dtype=np.intp)
    # OpenCV vertices are [x, y]; skimage expects [row=y, column=x].
    rows, cols = draw_polygon(box[:, 1], box[:, 0], shape=(height, width))
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(rows.astype(np.int64) * width + cols.astype(np.int64))


def _legacy_rectangle_pixels(grasp, shape):
    """Reproduce the historical x/y clipping defect for migration tables only."""
    height, width = (int(shape[0]), int(shape[1]))
    box = np.asarray(rectangle_vertices(grasp), dtype=np.intp)
    wrong_rows, wrong_cols = draw_polygon(box[:, 0], box[:, 1], shape=(height, width))
    keep = (wrong_rows < width) & (wrong_cols < height)
    wrong_rows = wrong_rows[keep]
    wrong_cols = wrong_cols[keep]
    if wrong_rows.size == 0:
        return np.empty(0, dtype=np.int64)
    # The legacy implementation wrote area[cc, rr], which partially cancels
    # the swap only after draw_polygon has already clipped x to image height.
    return np.unique(wrong_cols.astype(np.int64) * width + wrong_rows.astype(np.int64))


def rasterize_rectangle(grasp, shape=DEFAULT_IMAGE_SHAPE):
    height, width = (int(shape[0]), int(shape[1]))
    mask = np.zeros((height, width), dtype=bool)
    pixels = _corrected_rectangle_pixels(grasp, (height, width))
    if pixels.size:
        mask.reshape(-1)[pixels] = True
    return mask


def _pixel_iou(pixels_a, pixels_b):
    if pixels_a.size == 0 and pixels_b.size == 0:
        return 0.0
    intersection = np.intersect1d(pixels_a, pixels_b, assume_unique=True).size
    union = int(pixels_a.size + pixels_b.size - intersection)
    return float(intersection / union) if union else 0.0


def rectangle_iou(prediction, ground_truth, shape=DEFAULT_IMAGE_SHAPE, *, normalize_gt=True):
    pred = _as_grasp(prediction)
    gt = normalize_official_gt_rectangle(ground_truth) if normalize_gt else _as_grasp(ground_truth)
    return _pixel_iou(
        _corrected_rectangle_pixels(pred, shape),
        _corrected_rectangle_pixels(gt, shape),
    )


def legacy_rectangle_iou(prediction, ground_truth, shape=DEFAULT_IMAGE_SHAPE, *, normalize_gt=True):
    pred = _as_grasp(prediction)
    gt = normalize_official_gt_rectangle(ground_truth) if normalize_gt else _as_grasp(ground_truth)
    return _pixel_iou(
        _legacy_rectangle_pixels(pred, shape),
        _legacy_rectangle_pixels(gt, shape),
    )


def stable_gt_id(ground_truth):
    raw = [float(value) for value in list(ground_truth)]
    encoded = json.dumps(raw, separators=(",", ":"), allow_nan=False)
    return "gt_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _candidate_id(candidate):
    if isinstance(candidate, Mapping) and candidate.get("candidate_id") is not None:
        return str(candidate["candidate_id"])
    return "candidate"


def evaluate_candidate(
    candidate,
    gt_grasps: Iterable[Sequence[float]],
    *,
    shape=DEFAULT_IMAGE_SHAPE,
    evaluator_version=CORRECTED_EVALUATOR_VERSION,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
    angle_threshold=DEFAULT_ANGLE_THRESHOLD_DEG,
):
    """Evaluate one candidate against every GT using one pairwise kernel."""
    pred = _as_grasp(candidate)
    pairwise = []
    gt_iterable = [] if gt_grasps is None else gt_grasps
    for gt_index, raw_gt in enumerate(gt_iterable):
        gt = [float(value) for value in list(raw_gt)]
        gt_id = stable_gt_id(gt)
        if evaluator_version == LEGACY_EVALUATOR_VERSION:
            iou_value = legacy_rectangle_iou(pred, gt, shape=shape)
            angle_difference = periodic_angle_difference_deg(pred[4], gt[4])
            angle_ok = legacy_angle_compatible(pred[4], gt[4], angle_threshold)
        elif evaluator_version == "xy_only_geometric_sensitivity":
            iou_value = rectangle_iou(pred, gt, shape=shape)
            angle_difference = periodic_angle_difference_deg(pred[4], gt[4])
            angle_ok = legacy_angle_compatible(pred[4], gt[4], angle_threshold)
        elif evaluator_version == CORRECTED_EVALUATOR_VERSION:
            iou_value = rectangle_iou(pred, gt, shape=shape)
            angle_difference = periodic_angle_difference_deg(pred[4], gt[4])
            angle_ok = angle_difference <= float(angle_threshold)
        else:
            raise ValueError(f"unknown evaluator version: {evaluator_version}")
        iou_ok = iou_value > float(iou_threshold)
        pairwise.append(
            {
                "gt_index": int(gt_index),
                "gt_id": gt_id,
                "rectangle_iou": float(iou_value),
                "angle_difference_deg": float(angle_difference),
                "iou_ok": bool(iou_ok),
                "angle_ok": bool(angle_ok),
                "joint_success": bool(iou_ok and angle_ok),
                "center_distance_px": float(math.hypot(pred[0] - gt[0], pred[1] - gt[1])),
                "width_difference_px": float(abs(pred[2] - gt[2])),
            }
        )

    any_iou = any(item["iou_ok"] for item in pairwise)
    any_angle = any(item["angle_ok"] for item in pairwise)
    successes = [item for item in pairwise if item["joint_success"]]
    selectable = successes or pairwise
    best_gt = (
        min(
            selectable,
            key=lambda item: (
                -item["rectangle_iou"],
                item["angle_difference_deg"],
                item["gt_id"],
            ),
        )
        if selectable
        else None
    )
    candidate_success = bool(successes)
    if candidate_success:
        failure_mode = None
    elif not pairwise:
        failure_mode = "no_gt"
    elif any_angle and not any_iou:
        failure_mode = "geometry_iou_failure"
    elif any_iou and not any_angle:
        failure_mode = "angle_failure"
    elif any_iou and any_angle:
        failure_mode = "joint_mismatch"
    else:
        failure_mode = "both_failure"
    return {
        "evaluator_version": evaluator_version,
        "candidate_id": _candidate_id(candidate),
        "candidate_success": candidate_success,
        "any_iou_compatible": bool(any_iou),
        "any_angle_compatible": bool(any_angle),
        "failure_mode": failure_mode,
        "best_gt": best_gt,
        "pairwise": pairwise,
    }


def evaluate_candidates(candidates, gt_grasps, **kwargs):
    return [evaluate_candidate(candidate, gt_grasps, **kwargs) for candidate in candidates]


def grasp_set_success(candidates, gt_grasps, **kwargs):
    return any(item["candidate_success"] for item in evaluate_candidates(candidates, gt_grasps, **kwargs))


def validate_binary_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"binary mask must be 2-D, received shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("binary mask contains NaN or Inf")
    unique = set(np.unique(array).tolist())
    if not unique.issubset({0, 1, 255, False, True}):
        raise ValueError(f"mask is not binary; values={sorted(unique)[:10]}")
    return array != 0


def binary_mask_iou(predicted_mask, ground_truth_mask):
    pred = validate_binary_mask(predicted_mask)
    gt = validate_binary_mask(ground_truth_mask)
    if pred.shape != gt.shape:
        raise ValueError(f"mask shape mismatch: predicted={pred.shape}, gt={gt.shape}")
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return float(intersection / union) if union else 0.0


def load_raw_binary_target_mask(mask_path, object_id):
    """Load the original-resolution instance mask without interpolation."""
    instance_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if instance_mask is None:
        raise FileNotFoundError(f"unable to read ground-truth mask: {mask_path}")
    if instance_mask.ndim == 3:
        if not np.array_equal(instance_mask[..., 0], instance_mask[..., 1]) or not np.array_equal(
            instance_mask[..., 0], instance_mask[..., 2]
        ):
            raise ValueError(f"multi-channel instance mask is not channel-identical: {mask_path}")
        instance_mask = instance_mask[..., 0]
    target = instance_mask == int(object_id)
    if not target.any():
        raise ValueError(f"object_id {object_id} is absent from ground-truth mask: {mask_path}")
    return validate_binary_mask(target)
