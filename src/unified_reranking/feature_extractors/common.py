"""Candidate-aligned common HiFi/RGB-D evidence in original image coordinates.

These quantities are offline single-view 2.5-D proxies.  They are not force
closure, robot reachability, or full collision-checking claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage


def periodic_angle_error(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def _axis(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    # The byte-frozen fair evaluator defines candidate vertices with
    # ``cv2.boxPoints(..., -theta_deg)``.  Feature geometry must use that same
    # image-coordinate convention; otherwise candidate-aligned mask/RGB-D
    # evidence is sampled from the mirror-rotated rectangle for every
    # non-axis-aligned grasp.
    radians = math.radians(-float(angle_deg))
    closing = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    return closing, np.asarray([-closing[1], closing[0]], dtype=np.float64)


def _corners(
    center: np.ndarray,
    angle_deg: float,
    width_px: float,
    height_px: float,
) -> np.ndarray:
    closing, grasp = _axis(angle_deg)
    half_w, half_h = 0.5 * float(width_px), 0.5 * float(height_px)
    return np.stack(
        [
            center - half_w * closing - half_h * grasp,
            center + half_w * closing - half_h * grasp,
            center + half_w * closing + half_h * grasp,
            center - half_w * closing + half_h * grasp,
        ]
    )


def _roi_mask(shape: tuple[int, int], corners: np.ndarray) -> tuple[slice, slice, np.ndarray, float]:
    height, width = shape
    x0 = max(0, int(math.floor(float(corners[:, 0].min()))) - 1)
    x1 = min(width, int(math.ceil(float(corners[:, 0].max()))) + 2)
    y0 = max(0, int(math.floor(float(corners[:, 1].min()))) - 1)
    y1 = min(height, int(math.ceil(float(corners[:, 1].max()))) + 2)
    intended = max(abs(float(cv2.contourArea(corners.astype(np.float32)))), 1.0)
    if x1 <= x0 or y1 <= y0:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), dtype=bool), 1.0
    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = np.rint(corners - np.asarray([x0, y0])).astype(np.int32)
    cv2.fillConvexPoly(local, shifted, 1)
    overflow = float(np.clip(1.0 - float(local.sum()) / intended, 0.0, 1.0))
    return slice(y0, y1), slice(x0, x1), local.astype(bool), overflow


def _region_values(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, float]:
    ys, xs, mask, overflow = _roi_mask(image.shape, corners)
    return np.asarray(image)[ys, xs][mask], overflow


def _outer_context_values(image: np.ndarray, inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    ys, xs, outer_mask, _ = _roi_mask(image.shape, outer)
    if not outer_mask.size:
        return np.asarray([], dtype=float)
    x0, y0 = xs.start or 0, ys.start or 0
    shifted = np.rint(inner - np.asarray([x0, y0])).astype(np.int32)
    inner_mask = np.zeros_like(outer_mask, dtype=np.uint8)
    cv2.fillConvexPoly(inner_mask, shifted, 1)
    ring = outer_mask & ~inner_mask.astype(bool)
    return np.asarray(image)[ys, xs][ring]


def _bilinear(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    height, width = array.shape
    x = np.clip(points[:, 0], 0.0, width - 1.0)
    y = np.clip(points[:, 1], 0.0, height - 1.0)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    return (
        (1 - dx) * (1 - dy) * array[y0, x0]
        + dx * (1 - dy) * array[y0, x1]
        + (1 - dx) * dy * array[y1, x0]
        + dx * dy * array[y1, x1]
    )


def _line(center: np.ndarray, axis: np.ndarray, length: float) -> np.ndarray:
    count = max(3, int(math.ceil(abs(float(length)))) + 1)
    offsets = np.linspace(-0.5 * length, 0.5 * length, count)
    return center[None, :] + offsets[:, None] * axis[None, :]


def _patch(image: np.ndarray, point: np.ndarray, radius: int) -> np.ndarray:
    height, width = image.shape
    x, y = np.rint(point).astype(int)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    return np.asarray(image)[y0:y1, x0:x1]


def _finite_depth(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values) & (values > 0)]


def _depth_summary(values: np.ndarray) -> tuple[float, float, float, float, float]:
    raw = np.asarray(values).reshape(-1)
    valid = _finite_depth(raw)
    ratio = float(len(valid) / max(len(raw), 1))
    if not len(valid):
        return math.nan, math.nan, math.nan, math.nan, ratio
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return median, mad, float(valid.min()), float(valid.max()), ratio


def _plane_residual(depth: np.ndarray, corners: np.ndarray) -> float:
    ys, xs, mask, _ = _roi_mask(depth.shape, corners)
    values = np.asarray(depth)[ys, xs]
    valid = mask & np.isfinite(values) & (values > 0)
    if int(valid.sum()) < 10:
        return math.nan
    yy, xx = np.nonzero(valid)
    design = np.column_stack([xx, yy, np.ones(len(xx))])
    coefficient, *_ = np.linalg.lstsq(design, values[valid].astype(float), rcond=None)
    residual = values[valid] - design @ coefficient
    return float(np.median(np.abs(residual)))


def _continuous_iou(first: np.ndarray, second: np.ndarray) -> float:
    area_a = abs(float(cv2.contourArea(first.astype(np.float32))))
    area_b = abs(float(cv2.contourArea(second.astype(np.float32))))
    intersection, _ = cv2.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))
    union = area_a + area_b - float(intersection)
    return 0.0 if union <= 0 else float(intersection / union)


@dataclass(slots=True)
class EvidenceContext:
    probability: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    component_labels: np.ndarray
    signed_distance: np.ndarray
    mask_centroid: np.ndarray
    mask_principal_angle: float
    mask_reliability: float
    global_depth_median: float
    gradient_x: np.ndarray
    gradient_y: np.ndarray


def _context(probability: np.ndarray, binary_mask: np.ndarray, depth_m: np.ndarray) -> EvidenceContext:
    probability = np.asarray(probability, dtype=np.float32)
    mask = np.asarray(binary_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    if probability.shape != mask.shape or mask.shape != depth.shape:
        raise ValueError("probability, mask, and depth must share native HxW shape")
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("probability must be a finite two-dimensional array")
    probability = np.clip(probability, 0.0, 1.0)
    labels, _ = ndimage.label(mask)
    signed = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
    yy, xx = np.nonzero(mask)
    if len(xx):
        centroid = np.asarray([xx.mean(), yy.mean()], dtype=np.float64)
        centered = np.column_stack([xx - xx.mean(), yy - yy.mean()])
        covariance = np.cov(centered, rowvar=False) if len(xx) > 1 else np.eye(2)
        values, vectors = np.linalg.eigh(covariance)
        principal = vectors[:, int(np.argmax(values))]
        principal_angle = math.degrees(math.atan2(principal[1], principal[0]))
        inside = probability[mask]
        mask_reliability = float(np.mean(np.maximum(inside, 1.0 - inside)))
    else:
        centroid = np.asarray([mask.shape[1] / 2.0, mask.shape[0] / 2.0])
        principal_angle = 0.0
        mask_reliability = 0.0
    valid_depth = _finite_depth(depth)
    median = float(np.median(valid_depth)) if len(valid_depth) else 0.0
    filled = depth.astype(np.float64).copy()
    invalid = ~np.isfinite(filled) | (filled <= 0)
    filled[invalid] = median
    gy, gx = np.gradient(filled)
    return EvidenceContext(
        probability=probability,
        mask=mask,
        depth=depth,
        component_labels=labels,
        signed_distance=signed,
        mask_centroid=centroid,
        mask_principal_angle=principal_angle,
        mask_reliability=mask_reliability,
        global_depth_median=median,
        gradient_x=gx,
        gradient_y=gy,
    )


def _point_label(labels: np.ndarray, point: np.ndarray) -> int:
    y = int(np.clip(round(float(point[1])), 0, labels.shape[0] - 1))
    x = int(np.clip(round(float(point[0])), 0, labels.shape[1] - 1))
    return int(labels[y, x])


def _one_candidate(row: Any, context: EvidenceContext) -> tuple[dict[str, Any], dict[str, Any]]:
    p, mask, depth = context.probability, context.mask, context.depth
    height, width = mask.shape
    center = np.asarray([float(row.cx_px), float(row.cy_px)])
    closing, grasp_axis = _axis(float(row.theta_deg))
    jaw_width, short_side = float(row.width_px), float(row.height_px)
    corners = _corners(center, row.theta_deg, jaw_width, short_side)
    left = center - 0.5 * jaw_width * closing
    right = center + 0.5 * jaw_width * closing
    radius = max(2, int(round(jaw_width * 0.06)))
    rectangle_p, overflow = _region_values(p, corners)
    rectangle_mask, _ = _region_values(mask.astype(float), corners)
    rectangle_depth, _ = _region_values(depth, corners)
    outer = _corners(center, row.theta_deg, 1.4 * jaw_width, 2.0 * short_side)
    outer_p = _outer_context_values(p, corners, outer)
    closing_line = _line(center, closing, jaw_width)
    grasp_line = _line(center, grasp_axis, max(2.0 * short_side, 4.0))
    closing_p = _bilinear(p, closing_line)
    left_p = _patch(p, left, radius)
    right_p = _patch(p, right, radius)
    left_m = _patch(mask, left, radius)
    right_m = _patch(mask, right, radius)
    left_d = _patch(depth, left, radius)
    right_d = _patch(depth, right, radius)
    rectangle_depth_stats = _depth_summary(rectangle_depth)
    left_depth_stats = _depth_summary(left_d)
    right_depth_stats = _depth_summary(right_d)
    center_depth_stats = _depth_summary(_patch(depth, center, 1))
    center_x = int(np.clip(round(center[0]), 0, width - 1))
    center_y = int(np.clip(round(center[1]), 0, height - 1))
    depth_grad = np.asarray([context.gradient_x[center_y, center_x], context.gradient_y[center_y, center_x]])
    closing_mask = _bilinear(mask.astype(float), closing_line) >= 0.5
    grasp_mask = _bilinear(mask.astype(float), grasp_line) >= 0.5
    closing_span = float(closing_mask.mean() * jaw_width)
    grasp_span = float(grasp_mask.mean() * max(2.0 * short_side, 4.0))
    rectangle_probability_mean = float(rectangle_p.mean()) if len(rectangle_p) else 0.0
    entropy_values = np.clip(rectangle_p, 1e-6, 1 - 1e-6)
    entropy = float(np.mean(-(entropy_values * np.log(entropy_values) + (1 - entropy_values) * np.log(1 - entropy_values)))) if len(entropy_values) else 0.0
    signed_distance = float(context.signed_distance[center_y, center_x])
    same_component = (
        _point_label(context.component_labels, left) > 0
        and _point_label(context.component_labels, left) == _point_label(context.component_labels, right)
    )

    threshold = max(0.01, 2.0 * (rectangle_depth_stats[1] if math.isfinite(rectangle_depth_stats[1]) else 0.0))
    center_z = center_depth_stats[0]
    left_sweep_center = left - 0.5 * max(short_side, 8.0) * closing
    right_sweep_center = right + 0.5 * max(short_side, 8.0) * closing
    left_sweep = _corners(left_sweep_center, row.theta_deg, max(short_side, 8.0), 2.0 * short_side)
    right_sweep = _corners(right_sweep_center, row.theta_deg, max(short_side, 8.0), 2.0 * short_side)
    approach = _corners(center, row.theta_deg, 1.2 * jaw_width, 3.0 * short_side)

    def obstacle_ratio(polygon: np.ndarray) -> tuple[float, float]:
        values, _ = _region_values(depth, polygon)
        valid = _finite_depth(values)
        invalid_ratio = 1.0 - float(len(valid) / max(len(values), 1))
        if not len(valid) or not math.isfinite(center_z):
            return math.nan, invalid_ratio
        return float(np.mean(valid < center_z - threshold)), invalid_ratio

    left_obstacle, left_invalid = obstacle_ratio(left_sweep)
    right_obstacle, right_invalid = obstacle_ratio(right_sweep)
    approach_obstacle, approach_invalid = obstacle_ratio(approach)
    valid_rect_depth = _finite_depth(rectangle_depth)
    nearer = (
        math.nan
        if not len(valid_rect_depth) or not math.isfinite(center_z)
        else float(np.mean(valid_rect_depth < center_z - threshold))
    )

    variants = [
        (0, 0, 0, 1.0), (-2, 0, 0, 1.0), (2, 0, 0, 1.0),
        (0, -2, 0, 1.0), (0, 2, 0, 1.0), (0, 0, -5, 1.0),
        (0, 0, 5, 1.0), (0, 0, 0, 0.95), (0, 0, 0, 1.05),
    ]
    perturbed_scores: list[float] = []
    perturbed_valid: list[float] = []
    for dx, dy, dtheta, scale in variants:
        variant_corners = _corners(
            center + np.asarray([dx, dy]),
            float(row.theta_deg) + dtheta,
            jaw_width * scale,
            short_side,
        )
        values, variant_overflow = _region_values(p, variant_corners)
        perturbed_scores.append(float(values.mean()) if len(values) else 0.0)
        perturbed_valid.append(float(variant_overflow <= 0.5))

    feature = {
        "sample_id": str(row.sample_id),
        "candidate_id": str(row.candidate_id),
        "route": str(row.route),
        "native_score_raw": float(row.native_score),
        "native_rank": int(row.native_rank),
        "p_center": float(_bilinear(p, center[None, :])[0]),
        "center_inside_mask": float(bool(mask[center_y, center_x])),
        "rectangle_probability_mean": rectangle_probability_mean,
        "rectangle_probability_q10": float(np.quantile(rectangle_p, 0.1)) if len(rectangle_p) else 0.0,
        "rectangle_probability_min": float(rectangle_p.min()) if len(rectangle_p) else 0.0,
        "rectangle_binary_coverage": float(rectangle_mask.mean()) if len(rectangle_mask) else 0.0,
        "signed_distance_to_mask_boundary_px": signed_distance,
        "signed_distance_to_mask_boundary_over_width": signed_distance / max(jaw_width, 1e-6),
        "closing_axis_probability_mean": float(closing_p.mean()),
        "left_contact_probability_mean": float(left_p.mean()) if left_p.size else 0.0,
        "right_contact_probability_mean": float(right_p.mean()) if right_p.size else 0.0,
        "jaw_probability_min": min(float(left_p.mean()) if left_p.size else 0.0, float(right_p.mean()) if right_p.size else 0.0),
        "jaw_probability_asymmetry": abs((float(left_p.mean()) if left_p.size else 0.0) - (float(right_p.mean()) if right_p.size else 0.0)),
        "left_contact_mask_coverage": float(left_m.mean()) if left_m.size else 0.0,
        "right_contact_mask_coverage": float(right_m.mean()) if right_m.size else 0.0,
        "same_component_for_contacts": float(same_component),
        "background_fraction_inside_rectangle": 1.0 - (float(rectangle_mask.mean()) if len(rectangle_mask) else 0.0),
        "outer_context_probability_mean": float(outer_p.mean()) if len(outer_p) else 0.0,
        "local_probability_entropy": entropy,
        "mask_reliability": context.mask_reliability,
        "mask_span_on_closing_axis": closing_span,
        "mask_span_on_grasp_axis": grasp_span,
        "width_to_closing_span_ratio": jaw_width / max(closing_span, 1.0),
        "width_to_grasp_span_ratio": jaw_width / max(grasp_span, 1.0),
        "candidate_angle_to_mask_principal_axis": periodic_angle_error(row.theta_deg, context.mask_principal_angle),
        "z_center": center_z,
        "valid_depth_ratio_rectangle": rectangle_depth_stats[4],
        "valid_depth_ratio_contacts": 0.5 * (left_depth_stats[4] + right_depth_stats[4]),
        "local_depth_median": rectangle_depth_stats[0],
        "local_depth_mad": rectangle_depth_stats[1],
        "local_depth_range": rectangle_depth_stats[3] - rectangle_depth_stats[2] if math.isfinite(rectangle_depth_stats[2]) else math.nan,
        "local_plane_residual": _plane_residual(depth, corners),
        "left_contact_depth_mean": float(np.mean(_finite_depth(left_d))) if len(_finite_depth(left_d)) else math.nan,
        "left_contact_depth_std": float(np.std(_finite_depth(left_d))) if len(_finite_depth(left_d)) else math.nan,
        "right_contact_depth_mean": float(np.mean(_finite_depth(right_d))) if len(_finite_depth(right_d)) else math.nan,
        "right_contact_depth_std": float(np.std(_finite_depth(right_d))) if len(_finite_depth(right_d)) else math.nan,
        "contact_depth_abs_difference": abs(left_depth_stats[0] - right_depth_stats[0]) if math.isfinite(left_depth_stats[0]) and math.isfinite(right_depth_stats[0]) else math.nan,
        "contact_depth_symmetry": math.exp(-abs(left_depth_stats[0] - right_depth_stats[0]) / 0.02) if math.isfinite(left_depth_stats[0]) and math.isfinite(right_depth_stats[0]) else math.nan,
        "closing_axis_depth_edge_strength": abs(float(np.dot(depth_grad, closing))),
        "grasp_axis_depth_edge_strength": abs(float(np.dot(depth_grad, grasp_axis))),
        "interior_vs_exterior_depth_gap": rectangle_depth_stats[0] - context.global_depth_median if math.isfinite(rectangle_depth_stats[0]) else math.nan,
        "left_finger_sweep_obstacle_ratio": left_obstacle,
        "right_finger_sweep_obstacle_ratio": right_obstacle,
        "finger_sweep_obstacle_max": float(np.nanmax([left_obstacle, right_obstacle])) if any(math.isfinite(x) for x in (left_obstacle, right_obstacle)) else math.nan,
        "approach_context_obstacle_ratio": approach_obstacle,
        "nearer_than_center_depth_fraction": nearer,
        "invalid_depth_in_sweep": float(np.mean([left_invalid, right_invalid, approach_invalid])),
        "background_intrusion_fraction": 1.0 - (float(rectangle_mask.mean()) if len(rectangle_mask) else 0.0),
        "border_clearance_px": float(min(center[0], center[1], width - 1 - center[0], height - 1 - center[1])),
        "border_clearance_over_width": float(min(center[0], center[1], width - 1 - center[0], height - 1 - center[1]) / max(jaw_width, 1e-6)),
        "perturbed_score_mean": float(np.mean(perturbed_scores)),
        "perturbed_score_min": float(np.min(perturbed_scores)),
        "perturbed_score_std": float(np.std(perturbed_scores)),
        "perturbed_valid_fraction": float(np.mean(perturbed_valid)),
        "peak_retention_rate": float(np.mean(np.asarray(perturbed_scores) >= 0.9 * max(perturbed_scores[0], 1e-6))),
        "candidate_crop_overflow_ratio": overflow,
    }
    reliable = [
        context.mask_reliability,
        rectangle_depth_stats[4],
        left_depth_stats[4],
        right_depth_stats[4],
        float(np.mean(perturbed_valid)),
        float(overflow <= 0.5),
    ]
    feature["overall_feature_reliability"] = float(np.mean(reliable))
    feature["depth_missing"] = float(not math.isfinite(center_z))
    feature["contact_depth_missing"] = float(not math.isfinite(feature["contact_depth_abs_difference"]))
    feature["local_plane_missing"] = float(not math.isfinite(feature["local_plane_residual"]))
    feature["sweep_depth_missing"] = float(not math.isfinite(feature["finger_sweep_obstacle_max"]))
    geometry = {
        "center": center,
        "closing": closing,
        "corners": corners,
        "mask_support": feature["rectangle_probability_mean"],
        "depth": feature["z_center"],
        "collision": feature["finger_sweep_obstacle_max"],
    }
    return feature, geometry


def _list_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    for sample_id, indices in result.groupby("sample_id", sort=False).groups.items():
        index = np.asarray(list(indices), dtype=int)
        scores = result.loc[index, "native_score_raw"].to_numpy(float)
        ranks = result.loc[index, "native_rank"].to_numpy(int)
        order = np.argsort(ranks, kind="stable")
        sorted_scores = scores[order]
        mean, std = float(scores.mean()), float(scores.std())
        softmax = np.exp(scores - scores.max())
        softmax /= softmax.sum()
        entropy = float(-np.sum(softmax * np.log(np.maximum(softmax, 1e-12))))
        for position, local in enumerate(order):
            row_index = index[local]
            rank = position + 1
            result.loc[row_index, "rank_percentile"] = 1.0 if len(index) == 1 else (len(index) - rank) / (len(index) - 1)
            result.loc[row_index, "score_zscore_within_pool"] = (scores[local] - mean) / std if std > 1e-12 else 0.0
            result.loc[row_index, "score_percentile_within_pool"] = result.loc[row_index, "rank_percentile"]
            result.loc[row_index, "delta_to_top1"] = scores[local] - sorted_scores[0]
            result.loc[row_index, "delta_to_previous"] = 0.0 if position == 0 else scores[local] - sorted_scores[position - 1]
            result.loc[row_index, "delta_to_next"] = 0.0 if position + 1 == len(index) else scores[local] - sorted_scores[position + 1]
            result.loc[row_index, "top1_top2_margin"] = sorted_scores[0] - (sorted_scores[1] if len(index) > 1 else sorted_scores[0])
            result.loc[row_index, "pool_score_mean"] = mean
            result.loc[row_index, "pool_score_std"] = std
            result.loc[row_index, "pool_score_entropy"] = entropy
            result.loc[row_index, "candidate_count"] = len(index)
            result.loc[row_index, "is_native_top1"] = float(rank == 1)
    return result


def _relations(features: pd.DataFrame, geometry: dict[str, dict[str, Any]], image_shape: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    relation_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    diagonal = math.hypot(image_shape[0], image_shape[1])
    for sample_id, group in features.groupby("sample_id", sort=False):
        records = group.to_dict(orient="records")
        for row in records:
            source = geometry[f"{sample_id}\0{row['candidate_id']}"]
            local_relations: list[dict[str, Any]] = []
            for other in records:
                if other["candidate_id"] == row["candidate_id"]:
                    continue
                target = geometry[f"{sample_id}\0{other['candidate_id']}"]
                delta = target["center"] - source["center"]
                angle = periodic_angle_error(row["theta_deg"], other["theta_deg"])
                iou = _continuous_iou(source["corners"], target["corners"])
                relation = {
                    "sample_id": sample_id,
                    "source_candidate_id": row["candidate_id"],
                    "target_candidate_id": other["candidate_id"],
                    "normalized_delta_x": float(delta[0] / image_shape[1]),
                    "normalized_delta_y": float(delta[1] / image_shape[0]),
                    "normalized_center_distance": float(np.linalg.norm(delta) / diagonal),
                    # Candidate theta is stored in evaluator convention, while
                    # physical image-space axes use ``-theta`` (see ``_axis``).
                    # Therefore target-minus-source in physical coordinates is
                    # source.theta - target.theta.  The cosine/absolute terms
                    # are sign-invariant, but the directed sine is not.
                    "sin_2_delta_angle": math.sin(2 * math.radians(float(row["theta_deg"] - other["theta_deg"]))),
                    "cos_2_delta_angle": math.cos(2 * math.radians(float(other["theta_deg"] - row["theta_deg"]))),
                    "absolute_periodic_angle_difference": angle,
                    "log_width_ratio": math.log(max(float(other["width_px"]), 1e-6) / max(float(row["width_px"]), 1e-6)),
                    "native_score_difference": float(other["native_score_raw"] - row["native_score_raw"]),
                    "rotated_rectangle_iou": iou,
                    "closing_axis_overlap": max(0.0, 1.0 - float(np.linalg.norm(delta)) / max(float(row["width_px"]), float(other["width_px"]), 1.0)),
                    "jaw_region_overlap": iou,
                    "mask_support_difference": float(target["mask_support"] - source["mask_support"]),
                    "depth_difference": float(target["depth"] - source["depth"]) if math.isfinite(target["depth"]) and math.isfinite(source["depth"]) else math.nan,
                    "same_spatial_cluster": float(np.linalg.norm(delta) <= 24.0),
                    "same_orientation_cluster": float(angle <= 15.0),
                    "sweep_corridor_conflict": float(
                        iou
                        * max(
                            0.0 if not math.isfinite(source["collision"]) else source["collision"],
                            0.0 if not math.isfinite(target["collision"]) else target["collision"],
                        )
                    ),
                }
                relation_rows.append(relation)
                local_relations.append(relation)
            distances = [item["normalized_center_distance"] for item in local_relations]
            higher = [item["normalized_center_distance"] for item in local_relations if item["native_score_difference"] > 0]
            overlaps = [item["rotated_rectangle_iou"] for item in local_relations]
            nearby = [item for item in local_relations if item["same_spatial_cluster"]]
            cluster_scores = [row["native_score_raw"]] + [
                other["native_score_raw"] for other in records
                if other["candidate_id"] != row["candidate_id"]
                and np.linalg.norm(geometry[f"{sample_id}\0{other['candidate_id']}"]["center"] - source["center"]) <= 24.0
            ]
            aggregate_rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": row["candidate_id"],
                    "nearest_candidate_distance": min(distances, default=0.0),
                    "nearest_higher_score_distance": min(higher, default=0.0),
                    "number_of_nearby_candidates": len(nearby),
                    "candidate_cluster_size": 1 + len(nearby),
                    "candidate_rank_in_cluster": 1 + sum(value > row["native_score_raw"] for value in cluster_scores),
                    "cluster_best_score": max(cluster_scores),
                    "candidate_uniqueness": min(distances, default=1.0) * (1.0 - max(overlaps, default=0.0)),
                    "max_iou_with_other_candidate": max(overlaps, default=0.0),
                    "mean_iou_with_other_candidate": float(np.mean(overlaps)) if overlaps else 0.0,
                    "number_of_overlapping_candidates": sum(value > 0.1 for value in overlaps),
                }
            )
    return pd.DataFrame(relation_rows), pd.DataFrame(aggregate_rows)


def extract_common_evidence(
    candidates: pd.DataFrame,
    *,
    probability: np.ndarray,
    binary_mask: np.ndarray,
    depth_m: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return separate candidate-feature and directed relation tables."""

    required = {
        "sample_id", "candidate_id", "route", "native_rank", "native_score",
        "cx_px", "cy_px", "theta_deg", "width_px", "height_px",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidate table missing fields: {missing}")
    if candidates["sample_id"].nunique() != 1:
        raise ValueError("extract_common_evidence accepts exactly one sample")
    context = _context(probability, binary_mask, depth_m)
    rows: list[dict[str, Any]] = []
    geometry: dict[str, dict[str, Any]] = {}
    for row in candidates.sort_values("native_rank").itertuples(index=False):
        feature, candidate_geometry = _one_candidate(row, context)
        feature.update(
            {
                "cx_px": float(row.cx_px), "cy_px": float(row.cy_px),
                "theta_deg": float(row.theta_deg), "width_px": float(row.width_px),
                "height_px": float(row.height_px),
            }
        )
        rows.append(feature)
        geometry[f"{row.sample_id}\0{row.candidate_id}"] = candidate_geometry
    features = _list_features(pd.DataFrame(rows))
    relations, aggregates = _relations(features, geometry, context.mask.shape)
    features = features.merge(aggregates, on=["sample_id", "candidate_id"], validate="one_to_one")
    return features, relations
