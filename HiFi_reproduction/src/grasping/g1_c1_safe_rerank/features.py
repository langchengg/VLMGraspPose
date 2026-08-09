"""Leakage-safe image-space evidence for frozen G1/C1 candidates.

Depth and collision values are single-view 2.5-D proxies.  They do not provide
robot reachability, force closure, or collision-free physical execution proof.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from src.grasping.common.sample_io import CompactSampleLoader
from src.grasping.backends.conditioning import resize_probability_to_native

from .contracts import assert_inference_columns, canonical_sha256, validate_frozen_candidates
from .pools import periodic_angle_difference_deg


FEATURE_SCHEMA_VERSION = 1
CONTINUOUS_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "score": (
        "original_score",
        "pool_score",
        "raw_network_quality",
        "score_decomposition_residual",
        "score_margin_to_top1",
        "score_margin_to_previous",
        "score_margin_to_next",
        "score_z_within_set",
        "score_percentile_within_set",
        "score_entropy",
        "source_score_calibrated",
    ),
    "geometry": (
        "center_x_normalized",
        "center_y_normalized",
        "angle_sin2",
        "angle_cos2",
        "width_px_normalized",
        "rectangle_area_normalized",
        "distance_to_image_boundary_normalized",
        "distance_to_mask_centroid_normalized",
        "angle_to_mask_principal_axis_deg",
    ),
    "mask": (
        "center_probability",
        "local_probability_mean",
        "local_probability_max",
        "local_probability_min",
        "local_probability_std",
        "rectangle_mask_coverage",
        "grasp_axis_mask_support",
        "left_contact_mask_support",
        "right_contact_mask_support",
        "contact_support_minimum",
        "contact_support_imbalance",
        "signed_distance_to_mask_boundary_normalized",
        "mask_area_normalized",
        "mask_compactness",
        "largest_component_ratio",
        "foreground_probability_mean",
        "foreground_probability_entropy",
        "candidate_rectangle_overflow_ratio",
        "sweep_region_mask_support",
    ),
    "width": (
        "target_extent_along_closing_axis_px",
        "target_extent_orthogonal_axis_px",
        "candidate_width_over_target_extent",
        "width_compatibility",
    ),
    "depth": (
        "center_depth_m",
        "local_valid_depth_fraction",
        "left_contact_valid_fraction",
        "right_contact_valid_fraction",
        "left_contact_median_depth_m",
        "right_contact_median_depth_m",
        "absolute_contact_depth_difference_m",
        "signed_contact_depth_difference_m",
        "contact_depth_mad_m",
        "local_depth_mean_m",
        "local_depth_std_m",
        "local_depth_mad_m",
        "depth_gradient_along_closing_axis",
        "depth_gradient_orthogonal_axis",
    ),
    "clearance": (
        "gripper_sweep_valid_fraction",
        "sweep_foreground_fraction",
        "background_intrusion_ratio",
        "approach_collision_proxy",
        "sweep_minimum_clearance_proxy",
    ),
    "relations": (
        "nearest_candidate_center_distance_normalized",
        "nearest_candidate_angle_difference_normalized",
        "nearest_candidate_width_ratio",
        "maximum_rectangle_iou_with_other_candidate",
        "candidate_density",
        "similar_pose_fraction",
        "backend_consensus_count",
        "candidate_count_normalized",
    ),
    "reliability": (
        "feature_reliability_score",
        "depth_available",
        "probability_available",
        "left_contact_reliable",
        "right_contact_reliable",
        "sweep_reliable",
        "candidate_crop_in_bounds",
    ),
    "semantic": (
        "query_length_tokens",
        "query_attribute_count",
        "query_type_name",
        "query_type_attribute",
        "query_type_relation",
        "query_type_location",
        "query_type_mixed",
    ),
}
DEFAULT_FEATURE_COLUMNS = tuple(
    column for group in CONTINUOUS_FEATURE_GROUPS.values() for column in group
)


def deterministic_query_type(language: str) -> str:
    text = " " + re.sub(r"\s+", " ", str(language).strip().lower()) + " "
    relation = any(
        token in text
        for token in (
            " next to ", " beside ", " between ", " behind ", " in front of ",
            " near ", " touching ", " closest to ", " furthest from ",
        )
    )
    location = any(
        token in text
        for token in (
            " left ", " right ", " top ", " bottom ", " middle ", " center ",
            " front ", " back ",
        )
    )
    attribute = any(
        token in text
        for token in (
            " red ", " green ", " blue ", " yellow ", " orange ", " white ",
            " black ", " small ", " large ", " big ", " long ", " short ",
            " round ", " square ", " plastic ", " metal ",
        )
    )
    count = int(relation) + int(location) + int(attribute)
    if count > 1:
        return "mixed"
    if relation:
        return "relation"
    if location:
        return "location"
    if attribute:
        return "attribute"
    return "name"


def _attribute_count(language: str) -> int:
    text = " " + str(language).lower() + " "
    vocabulary = (
        " red ", " green ", " blue ", " yellow ", " orange ", " white ",
        " black ", " small ", " large ", " big ", " long ", " short ",
        " round ", " square ", " left ", " right ", " front ", " back ",
    )
    return sum(token in text for token in vocabulary)


def _axis(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    radians = math.radians(float(angle_deg))
    closing = np.array([math.cos(radians), math.sin(radians)], dtype=float)
    return closing, np.array([-closing[1], closing[0]], dtype=float)


def _line(center: np.ndarray, axis: np.ndarray, span: float, count: int | None = None) -> np.ndarray:
    steps = max(int(math.ceil(span)) + 1, 3) if count is None else max(int(count), 3)
    offsets = np.linspace(-span / 2.0, span / 2.0, steps)
    return center[None, :] + offsets[:, None] * axis[None, :]


def _sample_bilinear(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=float)
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


def _patch(image: np.ndarray, point: np.ndarray, radius: int) -> np.ndarray:
    height, width = image.shape
    x, y = np.rint(point).astype(int)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    return np.asarray(image[y0:y1, x0:x1])


def _corners(row: Mapping[str, Any], *, width_scale: float = 1.0, height_scale: float = 1.0) -> np.ndarray:
    center = np.array([float(row["center_x"]), float(row["center_y"])])
    axis, cross = _axis(float(row["angle_deg"]))
    half_width = float(row["width_px"]) * width_scale / 2.0
    half_height = float(row["height_px"]) * height_scale / 2.0
    return np.stack(
        [
            center - axis * half_width - cross * half_height,
            center + axis * half_width - cross * half_height,
            center + axis * half_width + cross * half_height,
            center - axis * half_width + cross * half_height,
        ]
    )


def _region_values(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape
    polygon = np.rint(corners).astype(np.int32)
    clipped = polygon.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, clipped, 1)
    intended = max(abs(float(cv2.contourArea(polygon.astype(np.float32)))), 1.0)
    observed = float(mask.sum())
    overflow = float(np.clip(1.0 - observed / intended, 0.0, 1.0))
    return np.asarray(image)[mask.astype(bool)], overflow


def _depth_stats(depth: np.ndarray, point: np.ndarray, radius: int) -> tuple[float, float, float, float]:
    values = _patch(depth, point, radius).astype(float).ravel()
    valid = values[np.isfinite(values) & (values > 0)]
    fraction = float(len(valid) / max(len(values), 1))
    if not len(valid):
        return math.nan, math.nan, math.nan, fraction
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return median, float(valid.mean()), mad, fraction


def _mask_summary(mask: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    height, width = mask_bool.shape
    area = int(mask_bool.sum())
    labels, count = ndimage.label(mask_bool)
    sizes = np.bincount(labels.ravel())[1:] if count else np.asarray([], dtype=int)
    largest = int(sizes.max(initial=0))
    perimeter = int(np.count_nonzero(mask_bool ^ ndimage.binary_erosion(mask_bool)))
    compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1))
    if area:
        y, x = np.nonzero(mask_bool)
        centroid = np.array([x.mean(), y.mean()])
        centered = np.column_stack([x - x.mean(), y - y.mean()])
        covariance = np.cov(centered, rowvar=False) if len(centered) > 1 else np.eye(2)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        principal_angle = math.degrees(math.atan2(principal[1], principal[0]))
    else:
        centroid = np.array([width / 2.0, height / 2.0])
        principal_angle = 0.0
    inside_distance = ndimage.distance_transform_edt(mask_bool)
    outside_distance = ndimage.distance_transform_edt(~mask_bool)
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return {
        "area": area,
        "component_count": int(count),
        "largest_component_ratio": float(largest / area) if area else 0.0,
        "compactness": compactness,
        "centroid": centroid,
        "principal_angle": principal_angle,
        "signed_distance": inside_distance - outside_distance,
        "foreground_probability_mean": float(np.mean(probability[mask_bool])) if area else 0.0,
        "foreground_probability_entropy": float(np.mean(entropy[mask_bool])) if area else 0.0,
    }


def _relation_features(group: pd.DataFrame) -> dict[str, dict[str, float]]:
    rows = group.to_dict(orient="records")
    result: dict[str, dict[str, float]] = {}
    for index, row in enumerate(rows):
        nearest = math.inf
        nearest_angle = 90.0
        nearest_width_ratio = 1.0
        max_iou = 0.0
        similar = 0
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            distance = float(
                np.linalg.norm(
                    np.array([row["center_x"], row["center_y"]], dtype=float)
                    - np.array([other["center_x"], other["center_y"]], dtype=float)
                )
            )
            angle = periodic_angle_difference_deg(row["angle_deg"], other["angle_deg"])
            width_ratio = min(float(row["width_px"]), float(other["width_px"])) / max(
                max(float(row["width_px"]), float(other["width_px"])), 1e-9
            )
            if distance < nearest:
                nearest, nearest_angle, nearest_width_ratio = distance, angle, width_ratio
            if distance <= 24.0 and angle <= 30.0:
                similar += 1
            # OpenCV polygon IoU without importing evaluator/labels.
            first, second = _corners(row), _corners(other)
            intersection, _ = cv2.intersectConvexConvex(
                first.astype(np.float32), second.astype(np.float32)
            )
            union = abs(cv2.contourArea(first.astype(np.float32))) + abs(
                cv2.contourArea(second.astype(np.float32))
            ) - float(intersection)
            max_iou = max(max_iou, 0.0 if union <= 0 else float(intersection / union))
        backend_consensus = 0
        if "source_backend" in group.columns:
            backend_consensus = int(
                group.loc[
                    group["stable_candidate_id"].eq(row["stable_candidate_id"]),
                    "source_backend",
                ].nunique()
            )
        result[str(row["stable_candidate_id"])] = {
            "nearest_candidate_center_distance_normalized": 0.0 if not math.isfinite(nearest) else nearest / 800.0,
            "nearest_candidate_angle_difference_normalized": nearest_angle / 90.0,
            "nearest_candidate_width_ratio": nearest_width_ratio,
            "maximum_rectangle_iou_with_other_candidate": max_iou,
            "candidate_density": float(similar),
            "similar_pose_fraction": float(similar / max(len(rows) - 1, 1)),
            "backend_consensus_count": float(backend_consensus),
            "candidate_count_normalized": float(len(rows) / 10.0),
        }
    return result


def extract_sample_features(
    candidates: pd.DataFrame,
    *,
    language: str,
    probability: np.ndarray,
    binary_mask: np.ndarray,
    depth_m: np.ndarray,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    is_union = "union_rank" in candidates.columns or "pool_rank" in candidates.columns
    validate_frozen_candidates(candidates, require_top5=False, allow_union=is_union)
    probability = np.asarray(probability, dtype=np.float32)
    mask = np.asarray(binary_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    if probability.shape != mask.shape or mask.shape != depth.shape:
        raise ValueError("probability/mask/depth shape mismatch")
    height, width = mask.shape
    summary = _mask_summary(mask, probability)
    relations = _relation_features(candidates)
    query_type = deterministic_query_type(language)
    score_column = "pool_score" if "pool_score" in candidates.columns else "original_score"
    scores = candidates[score_column].to_numpy(dtype=float)
    order = np.argsort(-scores, kind="stable")
    rank_by_index = np.empty(len(scores), dtype=int)
    rank_by_index[order] = np.arange(1, len(scores) + 1)
    score_mean, score_std = float(scores.mean()), float(scores.std())
    shifted = scores - scores.max()
    softmax = np.exp(shifted) / np.exp(shifted).sum()
    entropy = float(-np.sum(softmax * np.log(np.maximum(softmax, 1e-12))))
    filled_depth = depth.astype(float).copy()
    valid_depth = np.isfinite(filled_depth) & (filled_depth > 0)
    global_median = float(np.median(filled_depth[valid_depth])) if valid_depth.any() else 0.0
    filled_depth[~valid_depth] = global_median
    gradient_y, gradient_x = np.gradient(filled_depth)
    rows: list[dict[str, Any]] = []
    for position, (_, candidate) in enumerate(candidates.iterrows()):
        row = candidate.to_dict()
        center = np.array([float(row["center_x"]), float(row["center_y"])])
        axis, cross = _axis(float(row["angle_deg"]))
        half_width = float(row["width_px"]) / 2.0
        left, right = center - axis * half_width, center + axis * half_width
        line = _line(center, axis, max(float(row["width_px"]), 2.0))
        orthogonal = _line(center, cross, max(float(row["height_px"]) * 2.0, 4.0))
        local_p = _patch(probability, center, max(2, int(round(float(row["width_px"]) * 0.10)))).astype(float)
        left_p = _patch(probability, left, max(2, int(round(float(row["width_px"]) * 0.08)))).astype(float)
        right_p = _patch(probability, right, max(2, int(round(float(row["width_px"]) * 0.08)))).astype(float)
        left_probability_mean = float(left_p.mean()) if left_p.size else 0.0
        right_probability_mean = float(right_p.mean()) if right_p.size else 0.0
        rectangle_values, overflow = _region_values(mask.astype(float), _corners(row))
        sweep_values, _ = _region_values(mask.astype(float), _corners(row, width_scale=1.25, height_scale=2.5))
        _, local_depth_mean, local_depth_mad, local_valid = _depth_stats(depth, center, 6)
        left_depth, _, left_mad, left_valid = _depth_stats(depth, left, 3)
        right_depth, _, right_mad, right_valid = _depth_stats(depth, right, 3)
        center_depth, _, _, center_valid = _depth_stats(depth, center, 1)
        local_depth_values = _patch(depth, center, 6).astype(float).ravel()
        local_depth_values = local_depth_values[
            np.isfinite(local_depth_values) & (local_depth_values > 0)
        ]
        depth_std = float(local_depth_values.std()) if len(local_depth_values) else math.nan
        line_mask = _sample_bilinear(mask.astype(float), line) >= 0.5
        along_extent = float(line_mask.sum() / max(len(line_mask), 1) * max(float(row["width_px"]), 2.0))
        orth_mask = _sample_bilinear(mask.astype(float), orthogonal) >= 0.5
        orth_extent = float(orth_mask.sum() / max(len(orth_mask), 1) * max(float(row["height_px"]) * 2.0, 4.0))
        x, y = int(np.clip(round(center[0]), 0, width - 1)), int(np.clip(round(center[1]), 0, height - 1))
        grad = np.array([gradient_x[y, x], gradient_y[y, x]], dtype=float)
        raw_quality = float(row.get("raw_network_quality", math.nan))
        center_probability = float(_sample_bilinear(probability, center[None, :])[0])
        rank = int(rank_by_index[position])
        previous = scores[order[max(rank - 2, 0)]]
        following = scores[order[min(rank, len(scores) - 1)]]
        target_extent = max(along_extent, 1.0)
        width_ratio = float(row["width_px"]) / target_extent
        width_compatibility = float(math.exp(-abs(math.log(max(width_ratio, 1e-6)))))
        depth_available = float(valid_depth.mean() > 0.01)
        left_reliable = float(left_valid >= 0.5)
        right_reliable = float(right_valid >= 0.5)
        sweep_valid_values, _ = _region_values(valid_depth.astype(float), _corners(row, width_scale=1.25, height_scale=2.5))
        sweep_valid = float(sweep_valid_values.mean()) if len(sweep_valid_values) else 0.0
        signed_difference = (
            float(right_depth - left_depth)
            if math.isfinite(left_depth) and math.isfinite(right_depth)
            else math.nan
        )
        abs_difference = abs(signed_difference) if math.isfinite(signed_difference) else math.nan
        collision = float(
            np.clip(
                (1.0 - (float(sweep_values.mean()) if len(sweep_values) else 0.0))
                * (1.0 if sweep_valid >= 0.25 else 0.5)
                + min((abs_difference if math.isfinite(abs_difference) else 0.05) / 0.05, 1.0) * 0.25,
                0.0,
                1.0,
            )
        )
        reliability_flags = [
            math.isfinite(raw_quality),
            probability.size > 0,
            depth_available > 0,
            left_reliable > 0,
            right_reliable > 0,
            sweep_valid >= 0.5,
            0 <= center[0] < width and 0 <= center[1] < height,
        ]
        missingness = sum((0 if flag else 1) << index for index, flag in enumerate(reliability_flags))
        rank_value = int(rank_by_index[position])
        declared_pool_rank = int(row.get("pool_rank", rank_value))
        if declared_pool_rank != rank_value:
            raise ValueError("pool rank/score ordering drift")
        row.update(
            {
                "query_type": query_type,
                "pool_rank": rank_value,
                "pool_score": float(row.get("pool_score", row["original_score"])),
                "score_decomposition_residual": float(row["original_score"] - raw_quality * center_probability) if math.isfinite(raw_quality) else math.nan,
                "score_margin_to_top1": float(scores.max() - scores[position]),
                "score_margin_to_previous": float(previous - scores[position]),
                "score_margin_to_next": float(scores[position] - following),
                "score_z_within_set": float((scores[position] - score_mean) / score_std) if score_std > 1e-12 else 0.0,
                "score_percentile_within_set": 1.0 if len(scores) == 1 else float((len(scores) - rank) / (len(scores) - 1)),
                "score_entropy": entropy,
                "source_score_calibrated": float(row.get("source_score_calibrated", math.nan)),
                "center_x_normalized": float(center[0] / width),
                "center_y_normalized": float(center[1] / height),
                "angle_sin2": math.sin(2.0 * math.radians(float(row["angle_deg"]))),
                "angle_cos2": math.cos(2.0 * math.radians(float(row["angle_deg"]))),
                "width_px_normalized": float(row["width_px"] / width),
                "rectangle_area_normalized": float(row["width_px"] * row["height_px"] / (width * height)),
                "distance_to_image_boundary_normalized": float(min(center[0], center[1], width - 1 - center[0], height - 1 - center[1]) / max(width, height)),
                "distance_to_mask_centroid_normalized": float(np.linalg.norm(center - summary["centroid"]) / math.hypot(width, height)),
                "angle_to_mask_principal_axis_deg": periodic_angle_difference_deg(row["angle_deg"], summary["principal_angle"]),
                "center_probability": center_probability,
                "local_probability_mean": float(local_p.mean()) if local_p.size else 0.0,
                "local_probability_max": float(local_p.max(initial=0.0)),
                "local_probability_min": float(local_p.min(initial=0.0)),
                "local_probability_std": float(local_p.std()) if local_p.size else 0.0,
                "rectangle_mask_coverage": float(rectangle_values.mean()) if len(rectangle_values) else 0.0,
                "grasp_axis_mask_support": float(line_mask.mean()),
                "left_contact_mask_support": left_probability_mean,
                "right_contact_mask_support": right_probability_mean,
                "contact_support_minimum": min(left_probability_mean, right_probability_mean),
                "contact_support_imbalance": abs(left_probability_mean - right_probability_mean),
                "signed_distance_to_mask_boundary_normalized": float(summary["signed_distance"][y, x] / max(width, height)),
                "mask_area_normalized": float(summary["area"] / (width * height)),
                "mask_compactness": float(summary["compactness"]),
                "largest_component_ratio": float(summary["largest_component_ratio"]),
                "foreground_probability_mean": float(summary["foreground_probability_mean"]),
                "foreground_probability_entropy": float(summary["foreground_probability_entropy"]),
                "candidate_rectangle_overflow_ratio": overflow,
                "sweep_region_mask_support": float(sweep_values.mean()) if len(sweep_values) else 0.0,
                "target_extent_along_closing_axis_px": along_extent,
                "target_extent_orthogonal_axis_px": orth_extent,
                "candidate_width_over_target_extent": width_ratio,
                "width_compatibility": width_compatibility,
                "center_depth_m": center_depth,
                "local_valid_depth_fraction": local_valid,
                "left_contact_valid_fraction": left_valid,
                "right_contact_valid_fraction": right_valid,
                "left_contact_median_depth_m": left_depth,
                "right_contact_median_depth_m": right_depth,
                "absolute_contact_depth_difference_m": abs_difference,
                "signed_contact_depth_difference_m": signed_difference,
                "contact_depth_mad_m": float(np.nanmean([left_mad, right_mad])) if any(math.isfinite(value) for value in (left_mad, right_mad)) else math.nan,
                "local_depth_mean_m": local_depth_mean,
                "local_depth_std_m": depth_std,
                "local_depth_mad_m": local_depth_mad,
                "depth_gradient_along_closing_axis": float(np.dot(grad, axis)),
                "depth_gradient_orthogonal_axis": float(np.dot(grad, cross)),
                "gripper_sweep_valid_fraction": sweep_valid,
                "sweep_foreground_fraction": float(sweep_values.mean()) if len(sweep_values) else 0.0,
                "background_intrusion_ratio": 1.0 - (float(sweep_values.mean()) if len(sweep_values) else 0.0),
                "approach_collision_proxy": collision,
                "sweep_minimum_clearance_proxy": 1.0 - collision,
                "feature_reliability_score": float(np.mean(reliability_flags)),
                "raw_quality_available": float(math.isfinite(raw_quality)),
                "probability_available": 1.0,
                "depth_available": depth_available,
                "left_contact_reliable": left_reliable,
                "right_contact_reliable": right_reliable,
                "sweep_reliable": float(sweep_valid >= 0.5),
                "candidate_crop_in_bounds": float(0 <= center[0] < width and 0 <= center[1] < height),
                "missingness_bitmask": int(missingness),
                "query_length_tokens": float(len(str(language).split())),
                "query_attribute_count": float(_attribute_count(language)),
                **{f"query_type_{kind}": float(query_type == kind) for kind in ("name", "attribute", "relation", "location", "mixed")},
                **relations[str(row["stable_candidate_id"])],
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    assert_inference_columns(output.columns)
    return output


def extract_feature_table(
    candidates: pd.DataFrame,
    deployment_rows: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_frozen_candidates(
        candidates,
        require_top5=False,
        allow_union="union_rank" in candidates.columns or "pool_rank" in candidates.columns,
    )
    by_sample = {
        str(sample_id): group.copy()
        for sample_id, group in candidates.groupby("sample_id", sort=False)
    }
    loader = CompactSampleLoader()
    feature_parts: list[pd.DataFrame] = []
    sample_rows: list[dict[str, Any]] = []
    for deployment in deployment_rows:
        sample_id = str(deployment["sample_id"])
        group = by_sample.get(sample_id)
        if group is None or group.empty:
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "scene_id": str(deployment["scene_id"]),
                    "split": str(candidates["split"].iloc[0]) if not candidates.empty else "unknown",
                    "candidate_count": 0,
                    "query_type": deterministic_query_type(str(deployment["language"])),
                }
            )
            continue
        arrays = loader.load(deployment, mask_source="predicted", load_intrinsics=False)
        probability = resize_probability_to_native(
            arrays.probability,
            arrays.depth_m.shape,
        )
        features = extract_sample_features(
            group,
            language=arrays.language,
            probability=probability,
            binary_mask=arrays.binary_mask,
            depth_m=arrays.depth_m,
        )
        feature_parts.append(features)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "scene_id": arrays.scene_id,
                "split": str(group["split"].iloc[0]),
                "candidate_count": int(len(features)),
                "query_type": deterministic_query_type(arrays.language),
            }
        )
    feature_table = pd.concat(feature_parts, ignore_index=True) if feature_parts else candidates.iloc[:0].copy()
    return feature_table, pd.DataFrame(sample_rows)


def feature_schema() -> dict[str, Any]:
    fields = []
    units = {
        "center_depth_m": "metre",
        "left_contact_median_depth_m": "metre",
        "right_contact_median_depth_m": "metre",
        "absolute_contact_depth_difference_m": "metre",
        "signed_contact_depth_difference_m": "metre",
        "contact_depth_mad_m": "metre",
        "local_depth_mean_m": "metre",
        "local_depth_std_m": "metre",
        "local_depth_mad_m": "metre",
        "target_extent_along_closing_axis_px": "pixel",
        "target_extent_orthogonal_axis_px": "pixel",
        "angle_to_mask_principal_axis_deg": "degree_180_periodic",
    }
    for group, columns in CONTINUOUS_FEATURE_GROUPS.items():
        for name in columns:
            fields.append(
                {
                    "name": name,
                    "dtype": "float64",
                    "units": units.get(name, "unitless"),
                    "source": group,
                    "computation": "g1_c1_safe_rerank.features_v1",
                    "missing_value_rule": "NaN plus explicit availability/reliability flags; train-only median imputation",
                    "allowed_at_inference": True,
                    "backend_applicability": ["G1", "C1", "union"],
                }
            )
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "coordinate_convention": "original 640x480; x=column rightward, y=row downward",
        "angle_convention": "degrees in [-90,90), theta equivalent to theta+180",
        "physical_claim": False,
        "fields": fields,
    }
    payload["schema_sha256"] = canonical_sha256(payload)
    return payload
