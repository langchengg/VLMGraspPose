from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy import ndimage


MASK_THRESHOLD = 0.35
FIXED_GRASP_HEIGHT_PX = 20.0


def axial_angle_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def axial_angle_difference_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def decode_angle_map(sin_2theta: np.ndarray, cos_2theta: np.ndarray) -> np.ndarray:
    return np.rad2deg(0.5 * np.arctan2(sin_2theta, cos_2theta)).astype(np.float32)


def angle_vector_magnitude(
    sin_2theta: np.ndarray, cos_2theta: np.ndarray
) -> np.ndarray:
    return np.hypot(sin_2theta, cos_2theta).astype(np.float32)


def circular_concentration(
    sin_2theta: np.ndarray,
    cos_2theta: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    sin_values = np.asarray(sin_2theta, dtype=np.float64).ravel()
    cos_values = np.asarray(cos_2theta, dtype=np.float64).ravel()
    magnitude = np.hypot(sin_values, cos_values)
    valid = np.isfinite(sin_values) & np.isfinite(cos_values) & (magnitude > 1e-8)
    if weights is None:
        weights_array = magnitude
    else:
        weights_array = np.asarray(weights, dtype=np.float64).ravel() * magnitude
        valid &= np.isfinite(weights_array) & (weights_array > 0)
    if not valid.any():
        return 0.0, 0.0
    unit_sin = sin_values[valid] / magnitude[valid]
    unit_cos = cos_values[valid] / magnitude[valid]
    w = weights_array[valid]
    mean_sin = float(np.average(unit_sin, weights=w))
    mean_cos = float(np.average(unit_cos, weights=w))
    concentration = float(np.hypot(mean_sin, mean_cos))
    dominant = axial_angle_deg(math.degrees(0.5 * math.atan2(mean_sin, mean_cos)))
    return concentration, dominant


def _patch(array: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    radius = int(size) // 2
    y0, y1 = max(0, row - radius), min(array.shape[0], row + radius + 1)
    x0, x1 = max(0, col - radius), min(array.shape[1], col + radius + 1)
    return np.asarray(array[y0:y1, x0:x1])


def _sample_rotated_region(
    array: np.ndarray,
    candidate: dict[str, Any],
    *,
    output_size: int = 64,
    width_scale: float = 1.0,
    height_scale: float = 1.0,
) -> np.ndarray:
    normalized = np.linspace(-1.0, 1.0, output_size, dtype=np.float32)
    vv, uu = np.meshgrid(normalized, normalized, indexing="ij")
    theta = math.radians(axial_angle_deg(candidate["angle_deg"]))
    opening = np.asarray([math.cos(theta), -math.sin(theta)], dtype=np.float32)
    normal = np.asarray([-opening[1], opening[0]], dtype=np.float32)
    half_width = max(float(candidate["width_px"]) * width_scale / 2.0, 1.0)
    half_height = max(float(candidate.get("height_px", 20.0)) * height_scale / 2.0, 1.0)
    map_x = np.float32(
        float(candidate["cx"]) + uu * half_width * opening[0] + vv * half_height * normal[0]
    )
    map_y = np.float32(
        float(candidate["cy"]) + uu * half_width * opening[1] + vv * half_height * normal[1]
    )
    return cv2.remap(
        np.asarray(array, dtype=np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _polygon_mask(shape: tuple[int, int], polygon: Any) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, points, 1)
    return mask.astype(bool)


def _polygon_iou(left: Any, right: Any) -> float:
    left_poly = np.asarray(left, dtype=np.float32)
    right_poly = np.asarray(right, dtype=np.float32)
    left_area = abs(float(cv2.contourArea(left_poly)))
    right_area = abs(float(cv2.contourArea(right_poly)))
    intersection, _ = cv2.intersectConvexConvex(left_poly, right_poly)
    union = left_area + right_area - float(intersection)
    return 0.0 if union <= 0 else float(intersection) / union


def _normalized_entropy(values: list[float]) -> float:
    scores = np.asarray(values, dtype=np.float64)
    scores = scores - scores.max(initial=0.0)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return entropy / math.log(max(2, len(values)))


def _candidate_relations(candidates: list[dict[str, Any]], index: int) -> dict[str, float | int]:
    current = candidates[index]
    distances, angles, widths, overlaps = [], [], [], []
    for other_index, other in enumerate(candidates):
        if other_index == index:
            continue
        distances.append(
            math.hypot(float(current["cx"]) - float(other["cx"]), float(current["cy"]) - float(other["cy"]))
        )
        angles.append(axial_angle_difference_deg(current["angle_deg"], other["angle_deg"]))
        widths.append(abs(float(current["width_px"]) - float(other["width_px"])))
        overlaps.append(_polygon_iou(current["polygon"], other["polygon"]))
    nearest = int(np.argmin(distances))
    cluster_size = 1 + sum(value >= 0.25 for value in overlaps)
    maximum_overlap = max(overlaps, default=0.0)
    return {
        "distance_to_nearest_candidate": float(distances[nearest]),
        "nearest_candidate_angle_difference": float(angles[nearest]),
        "nearest_candidate_width_difference": float(widths[nearest]),
        "max_rectangle_iou_with_other_candidate": float(maximum_overlap),
        "mean_rectangle_iou_with_other_candidates": float(np.mean(overlaps)),
        "candidate_cluster_size": int(cluster_size),
        "candidate_uniqueness": float(1.0 - maximum_overlap),
    }


def extract_candidate_evidence(
    *,
    sample_id: str,
    frame_id: str,
    candidates: list[dict[str, Any]],
    mask_probability: np.ndarray,
    quality_probability: np.ndarray,
    sin_2theta: np.ndarray,
    cos_2theta: np.ndarray,
    width_probability: np.ndarray,
    mask_logit: np.ndarray | None = None,
    quality_raw: np.ndarray | None = None,
    width_raw: np.ndarray | None = None,
    mask_threshold: float = MASK_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract GT-free evidence from frozen candidate coordinates and CROG maps."""
    arrays = [mask_probability, quality_probability, sin_2theta, cos_2theta, width_probability]
    shape = np.asarray(mask_probability).shape
    if len(shape) != 2 or any(np.asarray(value).shape != shape for value in arrays):
        raise ValueError("all CROG maps must share one HxW shape")
    if len(candidates) != 5:
        raise ValueError("the experiment requires exactly five frozen candidates")
    for value in arrays:
        if not np.isfinite(value).all():
            raise ValueError("CROG evidence maps contain non-finite values")
    # Bicubic replay can overshoot slightly even though these three maps were
    # sigmoid outputs before upsampling.  Their probability semantics require
    # clipping before statistics or visualization.
    mask_probability = np.clip(np.asarray(mask_probability, dtype=np.float32), 0.0, 1.0)
    q_map = np.clip(np.asarray(quality_probability, dtype=np.float32), 0.0, 1.0)
    sin_map = np.asarray(sin_2theta, dtype=np.float32)
    cos_map = np.asarray(cos_2theta, dtype=np.float32)
    w_map = np.clip(np.asarray(width_probability, dtype=np.float32), 0.0, 1.0)
    angle_map = decode_angle_map(sin_map, cos_map)
    magnitude_map = angle_vector_magnitude(sin_map, cos_map)
    binary_mask = mask_probability > float(mask_threshold)
    inside_distance = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~binary_mask).astype(np.uint8), cv2.DIST_L2, 5)
    signed_distance = inside_distance - outside_distance
    centroid_yx = (
        np.asarray(ndimage.center_of_mass(binary_mask), dtype=np.float64)
        if binary_mask.any()
        else np.asarray([shape[0] / 2.0, shape[1] / 2.0], dtype=np.float64)
    )
    q_values = [float(q_map[int(item["row"]), int(item["col"])]) for item in candidates]
    q_min, q_max = min(q_values), max(q_values)
    q_entropy = _normalized_entropy(q_values)
    q_sorted = sorted(
        range(5), key=lambda idx: (-float(q_values[idx]), str(candidates[idx]["candidate_id"]))
    )
    q_ranks = {candidate_index: rank for rank, candidate_index in enumerate(q_sorted)}
    records: list[dict[str, Any]] = []
    mask_rank_values: list[float] = []
    angle_rank_values: list[float] = []
    width_rank_values: list[float] = []
    for index, candidate in enumerate(candidates):
        row, col = int(candidate["row"]), int(candidate["col"])
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            raise ValueError(f"candidate center outside map: {candidate['candidate_id']}")
        q3, q7, q15 = (_patch(q_map, row, col, size) for size in (3, 7, 15))
        stronger = np.argwhere(q_map > q_map[row, col])
        nearest_stronger = (
            float(np.min(np.linalg.norm(stronger - np.asarray([row, col]), axis=1)))
            if stronger.size
            else float(math.hypot(*shape))
        )
        local_m = _sample_rotated_region(mask_probability, candidate)
        local_sin = _sample_rotated_region(sin_map, candidate)
        local_cos = _sample_rotated_region(cos_map, candidate)
        local_w = _sample_rotated_region(w_map, candidate)
        coordinates = np.linspace(-1.0, 1.0, local_m.shape[0], dtype=np.float32)
        vv, uu = np.meshgrid(coordinates, coordinates, indexing="ij")
        rectangle = (np.abs(uu) <= 1.0) & (np.abs(vv) <= 1.0)
        center_strip = np.abs(vv) <= 0.22
        grasp_axis = np.abs(vv) <= 0.12
        left_jaw = (uu >= -0.85) & (uu <= -0.55) & (np.abs(vv) <= 0.5)
        right_jaw = (uu <= 0.85) & (uu >= 0.55) & (np.abs(vv) <= 0.5)
        left_support = float(local_m[left_jaw].mean())
        right_support = float(local_m[right_jaw].mean())
        concentration, dominant_angle = circular_concentration(
            local_sin[rectangle], local_cos[rectangle], weights=local_m[rectangle]
        )
        local_angles = decode_angle_map(local_sin, local_cos)
        angle_differences = np.asarray(
            [axial_angle_difference_deg(value, dominant_angle) for value in local_angles[rectangle]],
            dtype=np.float64,
        )
        candidate_angle_difference = axial_angle_difference_deg(candidate["angle_deg"], dominant_angle)
        angle_consistency = float(np.clip(concentration * (1.0 - candidate_angle_difference / 90.0), 0.0, 1.0))
        decoded_width = local_w * 100.0
        candidate_width = float(candidate["width_px"])
        local_width_mean = float(decoded_width.mean())
        width_difference = abs(candidate_width - local_width_mean)
        width_consistency = float(np.exp(-width_difference / max(candidate_width, 10.0)))
        gy, gx = np.gradient(w_map * 100.0)
        polygon_mask = _polygon_mask(shape, candidate["polygon"])
        polygon_values = mask_probability[polygon_mask]
        probability = float(mask_probability[row, col])
        local_entropy = -float(
            np.mean(
                probability_array * np.log(np.maximum(probability_array, 1e-8))
                + (1.0 - probability_array)
                * np.log(np.maximum(1.0 - probability_array, 1e-8))
            )
        ) if (probability_array := local_m[rectangle]).size else 0.0
        q_rank = q_ranks[index]
        previous = q_values[q_sorted[q_rank - 1]] if q_rank > 0 else q_values[index]
        following = q_values[q_sorted[q_rank + 1]] if q_rank < 4 else q_values[index]
        relation = _candidate_relations(candidates, index)
        record = {
            "sample_id": str(sample_id),
            "frame_id": str(frame_id),
            "scene_id": str(frame_id),
            "stable_candidate_id": f"{sample_id}/{candidate['candidate_id']}",
            "display_candidate_id": None,
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_checksum": str(candidate["candidate_checksum"]),
            "original_q_rank": int(candidate.get("q_rank", candidate.get("legacy_rank", q_rank))),
            "center_x_px": float(candidate["cx"]),
            "center_y_px": float(candidate["cy"]),
            "center_x_normalized": float(candidate["cx"]) / float(shape[1]),
            "center_y_normalized": float(candidate["cy"]) / float(shape[0]),
            "angle_deg_periodic_180": axial_angle_deg(candidate["angle_deg"]),
            "width_px": candidate_width,
            "width_normalized": candidate_width / float(shape[1]),
            "fixed_height_px": float(candidate.get("height_px", FIXED_GRASP_HEIGHT_PX)),
            "rectangle_corners": candidate["polygon"],
            "q_raw_at_center": float(q_map[row, col] if quality_raw is None else quality_raw[row, col]),
            "q_probability_at_center": float(q_map[row, col]),
            "q_original_value": float(candidate["q_raw"]),
            "q_rank": int(q_rank),
            "q_percentile_within_top5": float((4 - q_rank) / 4.0),
            "q_relative_minmax": float((q_values[index] - q_min) / max(q_max - q_min, 1e-12)),
            "q_gap_to_rank1": float(q_values[q_sorted[0]] - q_values[index]),
            "q_gap_to_previous": float(previous - q_values[index]),
            "q_gap_to_next": float(q_values[index] - following),
            "q_local_mean_3x3": float(q3.mean()),
            "q_local_mean_7x7": float(q7.mean()),
            "q_local_mean_15x15": float(q15.mean()),
            "q_local_max_7x7": float(q7.max()),
            "q_local_std_7x7": float(q7.std()),
            "q_local_std_15x15": float(q15.std()),
            "q_local_contrast": float(q_map[row, col] - q15.mean()),
            "q_peak_prominence": float(q_map[row, col] - np.median(q15)),
            "q_distance_to_nearest_stronger_peak": nearest_stronger,
            "q_entropy_across_top5": q_entropy,
            "mask_logit_at_center": None if mask_logit is None else float(mask_logit[row, col]),
            "mask_probability_at_center": probability,
            "center_inside_predicted_mask": bool(binary_mask[row, col]),
            "rectangle_mask_probability_mean": float(polygon_values.mean()) if polygon_values.size else 0.0,
            "rectangle_mask_probability_min": float(polygon_values.min()) if polygon_values.size else 0.0,
            "rectangle_foreground_fraction": float((polygon_values > mask_threshold).mean()) if polygon_values.size else 0.0,
            "center_strip_mask_mean": float(local_m[center_strip].mean()),
            "grasp_axis_mask_mean": float(local_m[grasp_axis].mean()),
            "grasp_axis_mask_min": float(local_m[grasp_axis].min()),
            "left_jaw_region_mask_mean": left_support,
            "right_jaw_region_mask_mean": right_support,
            "min_jaw_mask_support": min(left_support, right_support),
            "jaw_mask_imbalance": abs(left_support - right_support),
            "signed_distance_to_predicted_mask_boundary": float(signed_distance[row, col]),
            "distance_to_predicted_mask_centroid": float(np.linalg.norm(np.asarray([row, col]) - centroid_yx)),
            "rectangle_overlap_with_predicted_mask": float((polygon_mask & binary_mask).sum() / max(1, polygon_mask.sum())),
            "predicted_mask_local_entropy": local_entropy,
            "sin2theta_at_center": float(sin_map[row, col]),
            "cos2theta_at_center": float(cos_map[row, col]),
            "decoded_angle_at_center": float(angle_map[row, col]),
            "angle_vector_magnitude": float(magnitude_map[row, col]),
            "local_angle_vector_magnitude_mean": float(np.hypot(local_sin, local_cos).mean()),
            "local_angle_vector_magnitude_min": float(np.hypot(local_sin, local_cos).min()),
            "local_circular_concentration": concentration,
            "local_dominant_angle": dominant_angle,
            "candidate_vs_local_angle_difference": candidate_angle_difference,
            "angle_variance_inside_rectangle": float(np.var(angle_differences)),
            "angle_consistency_score": angle_consistency,
            "width_raw_at_center": float(w_map[row, col] if width_raw is None else width_raw[row, col]),
            "width_probability_at_center": float(w_map[row, col]),
            "decoded_width_at_center": float(w_map[row, col] * 100.0),
            "width_local_mean": local_width_mean,
            "width_local_std": float(decoded_width.std()),
            "width_rectangle_mean": float(decoded_width[rectangle].mean()),
            "width_rectangle_std": float(decoded_width[rectangle].std()),
            "candidate_vs_local_width_difference": width_difference,
            "width_map_gradient_at_center": float(math.hypot(gx[row, col], gy[row, col])),
            "width_consistency_score": width_consistency,
            **relation,
        }
        records.append(record)
        mask_rank_values.append(float(record["rectangle_mask_probability_mean"]))
        angle_rank_values.append(angle_consistency)
        width_rank_values.append(width_consistency)
    def ranks(values: list[float]) -> dict[int, int]:
        order = sorted(range(5), key=lambda idx: (-values[idx], str(candidates[idx]["candidate_id"])))
        return {candidate_index: rank for rank, candidate_index in enumerate(order)}
    mask_ranks, angle_ranks, width_ranks = map(ranks, (mask_rank_values, angle_rank_values, width_rank_values))
    for index, record in enumerate(records):
        record["q_rank_disagreement_with_mask_rank"] = int(abs(q_ranks[index] - mask_ranks[index]))
        record["q_rank_disagreement_with_angle_consistency_rank"] = int(abs(q_ranks[index] - angle_ranks[index]))
        record["q_rank_disagreement_with_width_consistency_rank"] = int(abs(q_ranks[index] - width_ranks[index]))
    sample = {
        "sample_id": str(sample_id),
        "frame_id": str(frame_id),
        "candidate_count": 5,
        "mask_threshold": float(mask_threshold),
        "predicted_mask_area": int(binary_mask.sum()),
        "predicted_mask_area_fraction": float(binary_mask.mean()),
        "q_entropy_across_top5": q_entropy,
        "original_q_top1_candidate_id": str(candidates[q_sorted[0]]["candidate_id"]),
        "map_height": int(shape[0]),
        "map_width": int(shape[1]),
    }
    return records, sample
