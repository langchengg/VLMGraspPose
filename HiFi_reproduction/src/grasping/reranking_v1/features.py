"""Leakage-safe candidate features for frozen planar grasp re-ranking.

The functions in this module consume only deployable inputs: a frozen candidate
pose, GQ-CNN score, predicted HiFi-CS mask/probability, depth, and camera
intrinsics. Ground-truth diagnostics are deliberately handled by
``join_candidate_labels`` after inference features have been computed.

Collision-related values are *visible-surface proxies*. They describe observed
depth points in simple gripper-frame volumes and are not a collision-free,
force-closure, reachability, or lift-success guarantee.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from src.grasping.camera_geometry import (
    T_CAMERA_GRASP_FIXED_APPROACH_KEY,
    CameraIntrinsicsData,
    backproject_pixels,
    width_pixels_to_meters,
)
from src.grasping.geometric_ranker import make_candidate_evaluation_rectangle
from src.grasping.reranking_v1.identity import candidate_identity_sha256
from src.grasping.reranking_v1.labels import periodic_angle_difference_deg, polygon_iou

SCHEMA_VERSION = 1
MASK_THRESHOLD = 0.15
MAX_GRIPPER_WIDTH_M = 0.08
LOCAL_NEIGHBOUR_CENTER_PX = 24.0
LOCAL_NEIGHBOUR_ANGLE_DEG = 30.0
CLUSTER_CENTER_PX = 16.0
CLUSTER_ANGLE_DEG = 20.0
PREDICTED_RECTANGLE_HEIGHT_PX = 20.0

IDENTIFIER_COLUMNS = (
    "sample_id",
    "scene_id",
    "candidate_id",
    "split",
    "query_type",
    "original_gqcnn_rank",
    "candidate_identity_sha256",
)

QUALITY_FEATURES = (
    "q_raw",
    "q_log",
    "q_percentile_within_sample",
    "q_rank_normalized",
    "q_gap_to_top1",
    "q_gap_to_previous",
    "q_gap_to_next",
    "local_neighbour_q_mean",
    "local_neighbour_q_std",
    "local_neighbour_q_max",
    "local_q_contrast",
)

SOFT_MASK_FEATURES = (
    "p_center",
    "p_axis_mean",
    "p_axis_min",
    "p_axis_p10",
    "p_left_jaw",
    "p_right_jaw",
    "p_contact_min",
    "p_contact_imbalance",
    "p_grasp_rectangle_mean",
    "p_grasp_rectangle_min",
    "signed_distance_to_mask_boundary",
    "mask_entropy_local",
    "grasp_axis_mask_support",
    "grasp_axis_mask_support_margin",
    "valid_depth_support",
    "valid_depth_support_margin",
    "centre_inside_mask",
)

WIDTH_FEATURES = (
    "width_m",
    "width_px",
    "width_ratio_to_max_gripper",
    "width_margin_to_max",
    "estimated_local_object_thickness",
    "width_minus_thickness",
    "normalized_width_mismatch",
)

DEPTH_GEOMETRY_FEATURES = (
    "center_depth",
    "left_contact_depth",
    "right_contact_depth",
    "jaw_depth_difference",
    "local_depth_mean",
    "local_depth_std",
    "depth_gradient_along_closing_axis",
    "depth_gradient_across_axis",
    "distance_to_depth_edge",
    "left_normal_x",
    "left_normal_y",
    "right_normal_x",
    "right_normal_y",
    "normal_opposition",
    "normal_closing_axis_alignment",
    "contact_symmetry",
)

VISIBLE_SURFACE_PROXY_FEATURES = (
    "left_finger_occupancy",
    "right_finger_occupancy",
    "palm_occupancy",
    "approach_corridor_occupancy",
    "minimum_visible_obstacle_distance",
    "approach_clearance",
    "collision_proxy_total",
)

RELATION_FEATURES = (
    "nearest_candidate_center_distance",
    "nearest_candidate_angle_difference",
    "nearest_candidate_width_difference",
    "max_rectangle_iou_with_other_candidate",
    "local_candidate_count",
    "pose_cluster_id",
    "pose_cluster_size",
    "distance_to_cluster_medoid",
    "cluster_q_mean",
    "cluster_q_std",
    "candidate_uniqueness",
)

INFERENCE_FEATURE_ALLOWLIST = (
    QUALITY_FEATURES
    + SOFT_MASK_FEATURES
    + WIDTH_FEATURES
    + DEPTH_GEOMETRY_FEATURES
    + VISIBLE_SURFACE_PROXY_FEATURES
    + RELATION_FEATURES
)

FORBIDDEN_GT_COLUMNS = (
    "candidate_positive",
    "best_gt_id",
    "candidate_gt_iou",
    "candidate_gt_angle_error",
    "candidate_gt_angle_error_deg",
    "maximum_rectangle_iou_with_angle_gate",
    "legacy_geq_positive",
    "exact_iou_threshold_pair_count",
    "gt_mask",
    "gt_grasp",
    "first_valid_rank",
    "center_error",
    "angle_error",
    "iou_with_gt",
    "correct_candidate_id",
    "j_at_1",
    "j_at_any",
    "target_gt_object_id",
)


def resize_probability_to_native(
    probability: np.ndarray, output_shape: tuple[int, int]
) -> np.ndarray:
    """Resize HiFi logits/probability with its exact bilinear convention."""

    source = np.asarray(probability, dtype=np.float32)
    if source.ndim != 2 or not np.all(np.isfinite(source)):
        raise ValueError("probability must be a finite HxW array")
    height, width = (int(output_shape[0]), int(output_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("output_shape must be positive")
    if source.shape == (height, width):
        return np.clip(source, 0.0, 1.0).astype(np.float32, copy=True)
    # Torch is the source-of-truth inference implementation. Import lazily so
    # geometry-only unit tests do not pay its import/startup cost.
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(source)[None, None]
    resized = F.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)


def _bilinear_value(image: np.ndarray, point_uv: Sequence[float]) -> float:
    point = np.asarray(point_uv, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return 0.0
    h, w = image.shape
    u = float(np.clip(point[0], 0.0, w - 1.0))
    v = float(np.clip(point[1], 0.0, h - 1.0))
    u0, v0 = int(math.floor(u)), int(math.floor(v))
    u1, v1 = min(u0 + 1, w - 1), min(v0 + 1, h - 1)
    du, dv = u - u0, v - v0
    return float(
        (1.0 - du) * (1.0 - dv) * image[v0, u0]
        + du * (1.0 - dv) * image[v0, u1]
        + (1.0 - du) * dv * image[v1, u0]
        + du * dv * image[v1, u1]
    )


def _line_points(
    first_uv: Sequence[float],
    second_uv: Sequence[float],
    shape: tuple[int, int],
    *,
    spacing_px: float = 1.0,
) -> np.ndarray:
    first, second = np.asarray(first_uv, float), np.asarray(second_uv, float)
    length = float(np.linalg.norm(second - first))
    count = max(2, int(math.ceil(length / max(spacing_px, 1e-6))) + 1)
    points = np.linspace(first, second, count)
    points[:, 0] = np.clip(points[:, 0], 0.0, shape[1] - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, shape[0] - 1.0)
    return points


def _sample_points(image: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    return np.asarray([_bilinear_value(image, point) for point in points_uv], float)


def _disk_values(
    image: np.ndarray, point_uv: Sequence[float], radius_px: int
) -> np.ndarray:
    u, v = np.rint(np.asarray(point_uv, float)).astype(int)
    h, w = image.shape
    x0, x1 = max(0, u - radius_px), min(w, u + radius_px + 1)
    y0, y1 = max(0, v - radius_px), min(h, v + radius_px + 1)
    if x0 >= x1 or y0 >= y1:
        return np.empty(0, dtype=float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    keep = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px**2
    return np.asarray(image[y0:y1, x0:x1][keep], dtype=float)


def _rectangle_values(image: np.ndarray, record: Mapping[str, Any]) -> np.ndarray:
    rectangle = make_candidate_evaluation_rectangle(
        record, {"predicted_rectangle_height_px": PREDICTED_RECTANGLE_HEIGHT_PX}
    )
    polygon = np.rint(rectangle["polygon"]).astype(np.int32)
    selection = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillConvexPoly(selection, polygon, 1)
    return np.asarray(image[selection.astype(bool)], dtype=float)


def _closing_axis(record: Mapping[str, Any]) -> np.ndarray:
    contacts = np.asarray(record["contact_points_uv"], dtype=float)
    delta = contacts[1] - contacts[0]
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-9:
        angle = float(record["angle_rad"])
        return np.array([math.cos(angle), math.sin(angle)], dtype=float)
    return delta / norm


def _safe_depth_at(
    depth_m: np.ndarray, point_uv: Sequence[float], fallback: float
) -> float:
    value = _bilinear_value(depth_m, point_uv)
    return float(value if np.isfinite(value) and value > 0 else fallback)


def soft_mask_features(
    record: Mapping[str, Any],
    *,
    probability: np.ndarray,
    binary_mask: np.ndarray,
    depth_m: np.ndarray,
) -> dict[str, float | bool]:
    """Extract probability and binary support without any GT access."""

    probability = np.asarray(probability, dtype=np.float32)
    binary_mask = np.asarray(binary_mask, dtype=bool)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if probability.shape != binary_mask.shape or probability.shape != depth_m.shape:
        raise ValueError("probability, mask, and depth must have identical HxW shape")
    center = np.asarray(record["center_uv"], float)
    contacts = np.asarray(record["contact_points_uv"], float)
    jaws = np.asarray(record["endpoints_uv"], float)
    axis_points = _line_points(contacts[0], contacts[1], probability.shape)
    axis_probability = _sample_points(probability, axis_points)
    axis_mask = _sample_points(binary_mask.astype(np.float32), axis_points)
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    axis_depth = _sample_points(valid_depth.astype(np.float32), axis_points)

    left = _disk_values(probability, jaws[0], 3)
    right = _disk_values(probability, jaws[1], 3)
    p_left = float(np.mean(left)) if left.size else 0.0
    p_right = float(np.mean(right)) if right.size else 0.0
    contact_probabilities = [_bilinear_value(probability, point) for point in contacts]
    rectangle_values = _rectangle_values(probability, record)
    signed_distance = ndimage.distance_transform_edt(
        binary_mask
    ) - ndimage.distance_transform_edt(~binary_mask)
    center_u = int(np.clip(round(center[0]), 0, probability.shape[1] - 1))
    center_v = int(np.clip(round(center[1]), 0, probability.shape[0] - 1))
    local = probability[
        max(0, center_v - 5) : min(probability.shape[0], center_v + 6),
        max(0, center_u - 5) : min(probability.shape[1], center_u + 6),
    ].astype(float)
    entropy = -(local * np.log(np.clip(local, 1e-8, 1.0))) - (
        (1.0 - local) * np.log(np.clip(1.0 - local, 1e-8, 1.0))
    )
    mask_support = float(np.mean(axis_mask >= 0.5))
    depth_support = float(np.mean(axis_depth >= 0.5))
    return {
        "p_center": _bilinear_value(probability, center),
        "p_axis_mean": float(np.mean(axis_probability)),
        "p_axis_min": float(np.min(axis_probability)),
        "p_axis_p10": float(np.quantile(axis_probability, 0.10)),
        "p_left_jaw": p_left,
        "p_right_jaw": p_right,
        "p_contact_min": float(min(contact_probabilities)),
        "p_contact_imbalance": float(
            abs(contact_probabilities[0] - contact_probabilities[1])
        ),
        "p_grasp_rectangle_mean": (
            float(np.mean(rectangle_values)) if rectangle_values.size else 0.0
        ),
        "p_grasp_rectangle_min": (
            float(np.min(rectangle_values)) if rectangle_values.size else 0.0
        ),
        "signed_distance_to_mask_boundary": float(signed_distance[center_v, center_u]),
        "mask_entropy_local": float(np.mean(entropy)) if entropy.size else 0.0,
        "grasp_axis_mask_support": mask_support,
        "grasp_axis_mask_support_margin": mask_support - 0.20,
        "valid_depth_support": depth_support,
        "valid_depth_support_margin": depth_support - 0.80,
        "centre_inside_mask": bool(binary_mask[center_v, center_u]),
    }


def width_and_depth_features(
    record: Mapping[str, Any],
    *,
    binary_mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsicsData,
    max_gripper_width_m: float = MAX_GRIPPER_WIDTH_M,
) -> dict[str, float]:
    """Extract width compatibility, local depth, and contact-normal geometry."""

    mask = np.asarray(binary_mask, bool)
    depth = np.asarray(depth_m, np.float32)
    center = np.asarray(record["center_uv"], float)
    contacts = np.asarray(record["contact_points_uv"], float)
    normals = np.asarray(record["contact_normals"], float)
    axis = _closing_axis(record)
    cross = np.array([-axis[1], axis[0]], dtype=float)
    center_depth = _safe_depth_at(depth, center, float(record["center_depth_m"]))
    contact_depths = [_safe_depth_at(depth, point, center_depth) for point in contacts]
    u, v = np.rint(center).astype(int)
    h, w = depth.shape
    x0, x1 = max(0, u - 6), min(w, u + 7)
    y0, y1 = max(0, v - 6), min(h, v + 7)
    local = depth[y0:y1, x0:x1].astype(float)
    local_valid = local[np.isfinite(local) & (local > 0)]
    local_mean = float(np.mean(local_valid)) if local_valid.size else center_depth
    local_std = float(np.std(local_valid)) if local_valid.size else 0.0

    filled = depth.astype(float).copy()
    invalid = ~np.isfinite(filled) | (filled <= 0)
    filled[invalid] = local_mean
    grad_v, grad_u = np.gradient(filled)
    gu = _bilinear_value(grad_u, center)
    gv = _bilinear_value(grad_v, center)
    gradient = np.array([gu, gv], dtype=float)
    edge = invalid | (np.hypot(grad_u, grad_v) >= 0.01)
    distance_to_edge = ndimage.distance_transform_edt(~edge)
    cu, cv = int(np.clip(round(center[0]), 0, w - 1)), int(
        np.clip(round(center[1]), 0, h - 1)
    )

    # Estimate target thickness along the closing axis using the predicted mask.
    max_span_px = max(float(record["width_px"]) * 1.5, 8.0)
    thickness_points = np.linspace(
        center - axis * max_span_px / 2.0,
        center + axis * max_span_px / 2.0,
        max(9, int(math.ceil(max_span_px)) + 1),
    )
    support = _sample_points(mask.astype(np.float32), thickness_points) >= 0.5
    if np.any(support):
        projected = np.linspace(-max_span_px / 2.0, max_span_px / 2.0, len(support))
        thickness_px = float(projected[support].max() - projected[support].min())
    else:
        thickness_px = 0.0
    thickness_m = width_pixels_to_meters(
        thickness_px, center_depth, intrinsics, math.atan2(axis[1], axis[0])
    )
    width_m = float(record["width_m"])

    normalized_normals = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
    )
    normal_opposition = float(
        np.clip(-np.dot(normalized_normals[0], normalized_normals[1]), -1.0, 1.0)
    )
    alignment_left = abs(float(np.dot(normalized_normals[0], axis)))
    alignment_right = abs(float(np.dot(normalized_normals[1], axis)))
    normal_alignment = float(min(alignment_left, alignment_right))
    jaw_difference = abs(contact_depths[0] - contact_depths[1])
    depth_symmetry = math.exp(-jaw_difference / 0.01)
    contact_symmetry = float(
        np.clip(0.5 * depth_symmetry + 0.5 * normal_alignment, 0.0, 1.0)
    )
    return {
        "width_m": width_m,
        "width_px": float(record["width_px"]),
        "width_ratio_to_max_gripper": width_m / max(max_gripper_width_m, 1e-12),
        "width_margin_to_max": max_gripper_width_m - width_m,
        "estimated_local_object_thickness": float(thickness_m),
        "width_minus_thickness": width_m - float(thickness_m),
        "normalized_width_mismatch": abs(width_m - float(thickness_m))
        / max(width_m, 1e-12),
        "center_depth": center_depth,
        "left_contact_depth": float(contact_depths[0]),
        "right_contact_depth": float(contact_depths[1]),
        "jaw_depth_difference": float(jaw_difference),
        "local_depth_mean": local_mean,
        "local_depth_std": local_std,
        "depth_gradient_along_closing_axis": float(np.dot(gradient, axis)),
        "depth_gradient_across_axis": float(np.dot(gradient, cross)),
        "distance_to_depth_edge": float(distance_to_edge[cv, cu]),
        "left_normal_x": float(normalized_normals[0, 0]),
        "left_normal_y": float(normalized_normals[0, 1]),
        "right_normal_x": float(normalized_normals[1, 0]),
        "right_normal_y": float(normalized_normals[1, 1]),
        "normal_opposition": normal_opposition,
        "normal_closing_axis_alignment": normal_alignment,
        "contact_symmetry": contact_symmetry,
    }


def transform_visible_points_to_grasp(
    points_camera: np.ndarray, transform_camera_from_grasp: np.ndarray
) -> np.ndarray:
    """Transform Nx3 observed camera points into the frozen grasp frame."""

    points = np.asarray(points_camera, dtype=np.float64)
    transform = np.asarray(transform_camera_from_grasp, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera must have shape (N,3)")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4x4 matrix")
    inverse = np.linalg.inv(transform)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=float)))
    return (homogeneous @ inverse.T)[:, :3]


def _voxel_occupancy(
    points: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    voxel_m: float = 0.005,
) -> float:
    lower_array, upper_array = np.asarray(lower, float), np.asarray(upper, float)
    keep = np.all((points >= lower_array) & (points <= upper_array), axis=1)
    selected = points[keep]
    bins = np.maximum(np.ceil((upper_array - lower_array) / voxel_m).astype(int), 1)
    total = int(np.prod(bins))
    if selected.size == 0 or total <= 0:
        return 0.0
    indices = np.floor((selected - lower_array) / voxel_m).astype(int)
    indices = np.clip(indices, 0, bins - 1)
    unique = np.unique(indices, axis=0)
    return float(np.clip(len(unique) / total, 0.0, 1.0))


def visible_surface_collision_features(
    record: Mapping[str, Any],
    *,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsicsData,
    pixel_stride: int = 3,
) -> dict[str, float]:
    """Compute finite observed-depth occupancy/clearance proxies in grasp frame."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth shape and intrinsics disagree")
    center = np.asarray(record["center_uv"], float)
    center_depth = max(float(record["center_depth_m"]), 1e-6)
    radius = int(
        np.clip(
            math.ceil(max(intrinsics.fx, intrinsics.fy) * 0.12 / center_depth), 24, 120
        )
    )
    u, v = np.rint(center).astype(int)
    x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
    y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
    vv, uu = np.mgrid[y0:y1:pixel_stride, x0:x1:pixel_stride]
    values = depth[vv, uu]
    valid = np.isfinite(values) & (values > 0)
    if np.any(valid):
        pixels = np.column_stack((uu[valid], vv[valid])).astype(float)
        points_camera = backproject_pixels(pixels, values[valid], intrinsics)
        points = transform_visible_points_to_grasp(
            points_camera,
            np.asarray(record[T_CAMERA_GRASP_FIXED_APPROACH_KEY], dtype=float),
        )
    else:
        points = np.empty((0, 3), dtype=float)

    half_width = max(float(record["width_m"]) / 2.0, 0.004)
    finger_half_thickness = 0.006
    left = _voxel_occupancy(
        points,
        (-0.035, -half_width - finger_half_thickness, -0.012),
        (0.015, -half_width + finger_half_thickness, 0.012),
    )
    right = _voxel_occupancy(
        points,
        (-0.035, half_width - finger_half_thickness, -0.012),
        (0.015, half_width + finger_half_thickness, 0.012),
    )
    palm = _voxel_occupancy(
        points,
        (-0.055, -half_width - 0.008, -0.016),
        (-0.035, half_width + 0.008, 0.016),
    )
    approach_lower = np.array([-0.080, -half_width - 0.012, -0.020])
    approach_upper = np.array([-0.015, half_width + 0.012, 0.020])
    approach = _voxel_occupancy(points, approach_lower, approach_upper)
    in_corridor = (
        np.all((points >= approach_lower) & (points <= approach_upper), axis=1)
        if len(points)
        else np.zeros(0, dtype=bool)
    )
    if np.any(in_corridor):
        # Distance remaining along the approach direction before the nearest
        # visible point. The upper cap is the 65-mm corridor length.
        clearance_m = float(
            np.clip(np.min(-points[in_corridor, 0] - 0.015), 0.0, 0.065)
        )
        minimum_distance = float(np.min(np.linalg.norm(points[in_corridor], axis=1)))
    else:
        clearance_m = 0.065
        minimum_distance = 0.080
    clearance_score = clearance_m / 0.065
    total = float(
        np.clip(
            0.25 * left + 0.25 * right + 0.20 * palm + 0.30 * approach,
            0.0,
            1.0,
        )
    )
    result = {
        "left_finger_occupancy": left,
        "right_finger_occupancy": right,
        "palm_occupancy": palm,
        "approach_corridor_occupancy": approach,
        "minimum_visible_obstacle_distance": minimum_distance,
        "approach_clearance": clearance_score,
        "collision_proxy_total": total,
    }
    if not np.all(np.isfinite(list(result.values()))):
        raise AssertionError(
            "visible-surface collision proxy produced non-finite values"
        )
    return result


def quality_and_relation_features(
    records: Sequence[Mapping[str, Any]],
    q_values: Sequence[float],
    q_ranks: Sequence[int],
) -> list[dict[str, float | int]]:
    """Compute within-sample q, neighbourhood, and deterministic cluster features."""

    count = len(records)
    q = np.asarray(q_values, dtype=float)
    ranks = np.asarray(q_ranks, dtype=int)
    if q.shape != (count,) or ranks.shape != (count,) or not np.all(np.isfinite(q)):
        raise ValueError("q values/ranks must align with finite candidate rows")
    if sorted(ranks.tolist()) != list(range(1, count + 1)):
        raise ValueError("GQ-CNN ranks must be a one-based permutation")
    centers = np.asarray([row["center_uv"] for row in records], dtype=float)
    angles = np.asarray([float(row["angle_rad"]) for row in records], dtype=float)
    widths = np.asarray([float(row["width_m"]) for row in records], dtype=float)
    center_distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    angle_distance = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            value = periodic_angle_difference_deg(angles[i], angles[j])
            angle_distance[i, j] = angle_distance[j, i] = value
    width_distance = np.abs(widths[:, None] - widths[None, :])

    rectangles = [
        make_candidate_evaluation_rectangle(
            row, {"predicted_rectangle_height_px": PREDICTED_RECTANGLE_HEIGHT_PX}
        )["polygon"]
        for row in records
    ]
    rectangle_iou = np.eye(count, dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            value = polygon_iou(rectangles[i], rectangles[j])
            rectangle_iou[i, j] = rectangle_iou[j, i] = value

    adjacency = (center_distance <= CLUSTER_CENTER_PX) & (
        angle_distance <= CLUSTER_ANGLE_DEG
    )
    np.fill_diagonal(adjacency, True)
    components: list[list[int]] = []
    unseen = set(range(count))
    while unseen:
        stack = [min(unseen)]
        component: set[int] = set()
        while stack:
            item = stack.pop()
            if item in component:
                continue
            component.add(item)
            unseen.discard(item)
            stack.extend(
                int(x)
                for x in np.flatnonzero(adjacency[item])
                if int(x) not in component
            )
        components.append(sorted(component))
    components.sort(
        key=lambda group: min(str(records[i]["candidate_id"]) for i in group)
    )
    cluster_by_index: dict[int, tuple[int, list[int], int]] = {}
    for cluster_id, component in enumerate(components):
        metric = (
            center_distance[np.ix_(component, component)] / CLUSTER_CENTER_PX
            + angle_distance[np.ix_(component, component)] / CLUSTER_ANGLE_DEG
            + width_distance[np.ix_(component, component)] / 0.01
        )
        totals = np.sum(metric, axis=1)
        medoid_position = min(
            range(len(component)),
            key=lambda k: (
                float(totals[k]),
                str(records[component[k]]["candidate_id"]),
            ),
        )
        medoid = component[medoid_position]
        for index in component:
            cluster_by_index[index] = (cluster_id, component, medoid)

    rank_to_index = {int(rank): index for index, rank in enumerate(ranks)}
    top_q = float(q[rank_to_index[1]])
    results: list[dict[str, float | int]] = []
    for i in range(count):
        local = (center_distance[i] <= LOCAL_NEIGHBOUR_CENTER_PX) & (
            angle_distance[i] <= LOCAL_NEIGHBOUR_ANGLE_DEG
        )
        local[i] = False
        local_indices = np.flatnonzero(local)
        neighbour_q = q[local_indices] if local_indices.size else np.asarray([q[i]])
        other = np.arange(count) != i
        if np.any(other):
            metric = center_distance[i].copy()
            metric[i] = np.inf
            nearest = min(
                np.flatnonzero(other),
                key=lambda j: (float(metric[j]), str(records[int(j)]["candidate_id"])),
            )
            max_iou = float(np.max(rectangle_iou[i, other]))
        else:
            nearest = i
            max_iou = 0.0
        cluster_id, component, medoid = cluster_by_index[i]
        rank = int(ranks[i])
        previous_q = top_q if rank == 1 else float(q[rank_to_index[rank - 1]])
        next_q = float(q[i]) if rank == count else float(q[rank_to_index[rank + 1]])
        cluster_q = q[component]
        distance_medoid = float(
            center_distance[i, medoid]
            + 0.25 * angle_distance[i, medoid]
            + 100.0 * width_distance[i, medoid]
        )
        results.append(
            {
                "q_raw": float(q[i]),
                "q_log": float(math.log(max(float(q[i]), 1e-12))),
                "q_percentile_within_sample": (
                    1.0 if count == 1 else float((count - rank) / (count - 1))
                ),
                "q_rank_normalized": (
                    1.0 if count == 1 else float((count - rank) / (count - 1))
                ),
                "q_gap_to_top1": top_q - float(q[i]),
                "q_gap_to_previous": previous_q - float(q[i]),
                "q_gap_to_next": float(q[i]) - next_q,
                "local_neighbour_q_mean": float(np.mean(neighbour_q)),
                "local_neighbour_q_std": float(np.std(neighbour_q)),
                "local_neighbour_q_max": float(np.max(neighbour_q)),
                "local_q_contrast": float(q[i] - np.mean(neighbour_q)),
                "nearest_candidate_center_distance": float(center_distance[i, nearest]),
                "nearest_candidate_angle_difference": float(angle_distance[i, nearest]),
                "nearest_candidate_width_difference": float(width_distance[i, nearest]),
                "max_rectangle_iou_with_other_candidate": max_iou,
                "local_candidate_count": int(local_indices.size),
                "pose_cluster_id": int(cluster_id),
                "pose_cluster_size": int(len(component)),
                "distance_to_cluster_medoid": distance_medoid,
                "cluster_q_mean": float(np.mean(cluster_q)),
                "cluster_q_std": float(np.std(cluster_q)),
                "candidate_uniqueness": float(
                    (1.0 - max_iou) / (1.0 + local_indices.size)
                ),
            }
        )
    return results


def extract_sample_features(
    records: Sequence[Mapping[str, Any]],
    *,
    q_values: Sequence[float],
    q_ranks: Sequence[int],
    probability: np.ndarray,
    binary_mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsicsData,
    sample_id: str,
    scene_id: str,
    split: str,
    query_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract one complete frozen sample without reading ground truth."""

    if not records:
        return [], {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "split": split,
            "query_type": query_type,
            "candidate_count": 0,
            "mask_area_px": int(np.count_nonzero(binary_mask)),
        }
    q_relation = quality_and_relation_features(records, q_values, q_ranks)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if str(record.get("sample_id")) != sample_id:
            raise ValueError("candidate sample_id disagrees with requested sample")
        identity = candidate_identity_sha256(record)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "candidate_id": str(record["candidate_id"]),
            "split": split,
            "query_type": query_type,
            "original_gqcnn_rank": int(q_ranks[index]),
            "candidate_identity_sha256": identity,
        }
        row.update(q_relation[index])
        row.update(
            soft_mask_features(
                record,
                probability=probability,
                binary_mask=binary_mask,
                depth_m=depth_m,
            )
        )
        row.update(
            width_and_depth_features(
                record,
                binary_mask=binary_mask,
                depth_m=depth_m,
                intrinsics=intrinsics,
            )
        )
        row.update(
            visible_surface_collision_features(
                record, depth_m=depth_m, intrinsics=intrinsics
            )
        )
        rows.append(row)
    feature_matrix = np.asarray(
        [[float(row[name]) for name in INFERENCE_FEATURE_ALLOWLIST] for row in rows],
        dtype=float,
    )
    if not np.all(np.isfinite(feature_matrix)):
        bad = np.argwhere(~np.isfinite(feature_matrix))[0]
        raise ValueError(
            f"non-finite feature {INFERENCE_FEATURE_ALLOWLIST[int(bad[1])]} "
            f"for {sample_id}/{rows[int(bad[0])]['candidate_id']}"
        )
    sample_row = {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "split": split,
        "query_type": query_type,
        "candidate_count": len(rows),
        "mask_area_px": int(np.count_nonzero(binary_mask)),
        "finite_feature_fraction": float(np.mean(np.isfinite(feature_matrix))),
        "q_top1": float(max(q_values)),
        "q_mean": float(np.mean(q_values)),
        "q_std": float(np.std(q_values)),
        "q_top1_gap": float(
            0.0
            if len(q_values) == 1
            else np.sort(np.asarray(q_values, float))[-1]
            - np.sort(np.asarray(q_values, float))[-2]
        ),
    }
    return rows, sample_row


def reject_forbidden_features(columns: Sequence[str]) -> None:
    """Fail closed if a requested inference feature can encode GT."""

    forbidden = {name.lower() for name in FORBIDDEN_GT_COLUMNS}
    rejected = []
    for column in columns:
        lowered = str(column).lower()
        if (
            lowered in forbidden
            or lowered.startswith("gt_")
            or lowered.endswith("_with_gt")
            or "ground_truth" in lowered
        ):
            rejected.append(str(column))
    if rejected:
        raise ValueError(
            f"forbidden GT inference features requested: {sorted(rejected)}"
        )


def validate_inference_allowlist(columns: Sequence[str]) -> tuple[str, ...]:
    """Validate a selected feature list against the frozen allowlist."""

    reject_forbidden_features(columns)
    unknown = sorted(set(map(str, columns)) - set(INFERENCE_FEATURE_ALLOWLIST))
    if unknown:
        raise ValueError(f"features are not in the inference allowlist: {unknown}")
    return tuple(map(str, columns))


def join_candidate_labels(
    feature_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Strictly join GT-only labels by sample_id+candidate_id after extraction."""

    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in label_rows:
        key = (str(row["sample_id"]), str(row["candidate_id"]))
        if key in index:
            raise ValueError(f"duplicate candidate label: {key}")
        index[key] = row
    output: list[dict[str, Any]] = []
    for source in feature_rows:
        key = (str(source["sample_id"]), str(source["candidate_id"]))
        label = index.get(key)
        if label is None:
            raise ValueError(f"candidate label missing: {key}")
        joined = dict(source)
        for column in FORBIDDEN_GT_COLUMNS:
            if column in label:
                joined[column] = label[column]
        # The evaluator records the unit-explicit name; retain it and also emit
        # the exact thesis contract column (documented as degrees by the schema).
        if (
            "candidate_gt_angle_error" not in joined
            and "candidate_gt_angle_error_deg" in label
        ):
            joined["candidate_gt_angle_error"] = label["candidate_gt_angle_error_deg"]
        output.append(joined)
    extras = sorted(
        index.keys()
        - {(str(r["sample_id"]), str(r["candidate_id"])) for r in feature_rows}
    )
    if extras:
        raise ValueError(f"labels include unknown candidates: {extras[:5]}")
    return output


def compute_train_only_statistics(frame: pd.DataFrame, *, split: str) -> dict[str, Any]:
    """Fit normalization statistics only on development/train features."""

    if split not in {"train", "development"}:
        raise ValueError("feature statistics may only be fitted on train/development")
    validate_inference_allowlist(INFERENCE_FEATURE_ALLOWLIST)
    statistics: dict[str, Any] = {}
    for column in INFERENCE_FEATURE_ALLOWLIST:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"cannot fit non-finite train feature: {column}")
        mean = float(np.mean(values))
        std = float(np.std(values))
        statistics[column] = {
            "count": int(len(values)),
            "mean": mean,
            "std": std,
            "scale": std if std > 1e-12 else 1.0,
            "minimum": float(np.min(values)),
            "q01": float(np.quantile(values, 0.01)),
            "median": float(np.median(values)),
            "q99": float(np.quantile(values, 0.99)),
            "maximum": float(np.max(values)),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_split": split,
        "fit_scope": "development/train candidates only",
        "feature_columns": list(INFERENCE_FEATURE_ALLOWLIST),
        "statistics": statistics,
    }


def feature_schema(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a machine-readable schema with source/leakage semantics."""

    label_columns = set(frame.columns) & set(FORBIDDEN_GT_COLUMNS)
    columns = []
    for name in frame.columns:
        if name in label_columns:
            role, source = "label_or_gt_diagnostic", "ground_truth_post_extraction"
        elif name in INFERENCE_FEATURE_ALLOWLIST:
            role, source = "inference_feature", "deployable_predicted_or_observed_input"
        else:
            role, source = "identifier_or_audit", "frozen_artifact_metadata"
        columns.append(
            {
                "name": str(name),
                "pandas_dtype": str(frame[name].dtype),
                "role": role,
                "source": source,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": int(len(frame)),
        "columns": columns,
        "visible_surface_proxy_disclaimer": (
            "Occupancy and clearance use only visible depth points; they are not "
            "a complete collision, force-closure, reachability, or lift guarantee."
        ),
        "constants": {
            "mask_threshold": MASK_THRESHOLD,
            "max_gripper_width_m": MAX_GRIPPER_WIDTH_M,
            "local_neighbour_center_px": LOCAL_NEIGHBOUR_CENTER_PX,
            "local_neighbour_angle_deg": LOCAL_NEIGHBOUR_ANGLE_DEG,
            "cluster_center_px": CLUSTER_CENTER_PX,
            "cluster_angle_deg": CLUSTER_ANGLE_DEG,
            "predicted_rectangle_height_px": PREDICTED_RECTANGLE_HEIGHT_PX,
        },
    }


def stable_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash JSON-canonical row content for dataset manifests."""

    payload = json.dumps(
        list(rows),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
