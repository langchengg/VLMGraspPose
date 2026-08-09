"""CPU-only mask/depth analytic planar grasp backend.

The occupancy terms in this module are visible-surface clearance proxies.  They
are not a collision-free guarantee, a force-closure guarantee, or a robot
reachability check.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure, morphology

from ..common.candidate_decoder import (
    NMSConfig,
    non_maximum_suppression,
    rank_candidates,
    stable_candidate_id,
)
from ..common.geometry import normalize_angle_deg
from ..common.types import Grasp4DoF, GraspPrediction


BACKEND_NAME = "repeatedfilm_mask_depth_analytic"
CLEARANCE_PROXY_SEMANTICS = (
    "visible-surface collision/clearance proxy; not a complete collision check"
)
ANALYTIC_FEATURE_NAMES = (
    "antipodal_normal_alignment",
    "left_mask_probability",
    "right_mask_probability",
    "minimum_jaw_mask_support",
    "axis_mask_support",
    "center_mask_probability",
    "boundary_margin",
    "left_depth",
    "right_depth",
    "left_depth_gradient",
    "right_depth_gradient",
    "left_boundary_curvature",
    "right_boundary_curvature",
    "left_valid_depth_support",
    "right_valid_depth_support",
    "jaw_depth_difference",
    "local_depth_variance",
    "depth_continuity_along_axis",
    "depth_edge_crossing",
    "contact_symmetry",
    "width_px",
    "width_m",
    "width_margin",
    "left_finger_occupancy",
    "right_finger_occupancy",
    "palm_occupancy",
    "approach_corridor_occupancy",
    "visible_clearance",
    "target_centroid_distance",
    "candidate_uniqueness",
)


@dataclass(frozen=True, slots=True)
class AnalyticGraspConfig:
    """Small, validation-tunable search surface for the analytic backend."""

    probability_threshold: float = 0.5
    min_component_area_px: int = 50
    max_hole_area_px: int = 64
    opening_radius_px: int = 1
    closing_radius_px: int = 2
    component_policy: str = "all"
    contour_spacing_px: float = 4.0
    max_contour_points: int = 256
    min_width_px: float = 10.0
    max_width_px: float = 100.0
    min_width_m: float = 0.005
    max_width_m: float = 0.08
    antipodal_alignment_min: float = 0.55
    normal_axis_alignment_min: float = 0.45
    min_contact_probability: float = 0.1
    min_center_probability: float = 0.1
    min_axis_mask_support: float = 0.55
    min_contact_valid_depth_support: float = 0.5
    min_axis_valid_depth_fraction: float = 0.8
    max_jaw_depth_difference_m: float = 0.04
    max_axis_depth_jump_m: float = 0.05
    contact_patch_radius_px: int = 2
    axis_sample_spacing_px: float = 1.5
    finger_length_px: float = 20.0
    finger_thickness_px: float = 5.0
    visible_depth_margin_m: float = 0.01
    fixed_height_px: float = 20.0
    allow_oracle: bool = False
    max_raw_candidates: int = 500
    weight_antipodal: float = 1.4
    weight_mask_support: float = 1.2
    weight_contact_symmetry: float = 0.7
    weight_boundary_margin: float = 0.5
    weight_visible_clearance: float = 0.8
    weight_width_compatibility: float = 0.5
    weight_jaw_depth_difference: float = 0.9
    weight_depth_edge_crossing: float = 1.0
    weight_occupancy_risk: float = 0.8
    nms: NMSConfig = field(
        default_factory=lambda: NMSConfig(
            center_distance_px=8.0,
            angle_distance_deg=15.0,
            width_distance_px=10.0,
            rectangle_iou_threshold=0.25,
            max_output=100,
        )
    )

    def __post_init__(self) -> None:
        if self.component_policy not in {"all", "largest"}:
            raise ValueError("component_policy must be 'all' or 'largest'")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must be in [0, 1]")
        integer_nonnegative = (
            self.min_component_area_px,
            self.max_hole_area_px,
            self.opening_radius_px,
            self.closing_radius_px,
        )
        if any(int(value) < 0 for value in integer_nonnegative):
            raise ValueError("morphology parameters must be non-negative")
        if self.max_contour_points < 4 or self.max_raw_candidates <= 0:
            raise ValueError("candidate limits must be positive")
        positive = (
            self.contour_spacing_px,
            self.min_width_px,
            self.max_width_px,
            self.min_width_m,
            self.max_width_m,
            self.max_jaw_depth_difference_m,
            self.max_axis_depth_jump_m,
            self.axis_sample_spacing_px,
            self.finger_length_px,
            self.finger_thickness_px,
            self.fixed_height_px,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("positive analytic parameters must be finite")
        if self.min_width_px >= self.max_width_px or self.min_width_m >= self.max_width_m:
            raise ValueError("minimum grasp width must be below maximum")
        unit_interval = (
            self.antipodal_alignment_min,
            self.normal_axis_alignment_min,
            self.min_contact_probability,
            self.min_center_probability,
            self.min_axis_mask_support,
            self.min_contact_valid_depth_support,
            self.min_axis_valid_depth_fraction,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in unit_interval):
            raise ValueError("support/alignment thresholds must be in [0, 1]")
        if self.contact_patch_radius_px < 0 or self.visible_depth_margin_m < 0.0:
            raise ValueError("patch radius and depth margin must be non-negative")
        weights = self.score_weights()
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("analytic score weights must be finite and non-negative")
        if sum(weights.values()) <= 0.0:
            raise ValueError("at least one analytic score weight must be positive")

    def score_weights(self) -> dict[str, float]:
        return {
            "antipodal_alignment": float(self.weight_antipodal),
            "target_mask_support": float(self.weight_mask_support),
            "contact_symmetry": float(self.weight_contact_symmetry),
            "boundary_margin": float(self.weight_boundary_margin),
            "visible_clearance": float(self.weight_visible_clearance),
            "width_compatibility": float(self.weight_width_compatibility),
            "jaw_depth_difference": float(self.weight_jaw_depth_difference),
            "depth_edge_crossing": float(self.weight_depth_edge_crossing),
            "occupancy_risk": float(self.weight_occupancy_risk),
        }

    @classmethod
    def validation_search_space(cls) -> dict[str, tuple[Any, ...]]:
        """Return a deliberately small validation-only search grid."""

        return {
            "component_policy": ("all", "largest"),
            "opening_radius_px": (0, 1),
            "closing_radius_px": (1, 2),
            "contour_spacing_px": (3.0, 5.0),
            "antipodal_alignment_min": (0.55, 0.7),
            "min_axis_mask_support": (0.55, 0.7),
            "max_axis_depth_jump_m": (0.03, 0.05),
        }


def clean_predicted_mask(
    binary_mask: np.ndarray, config: AnalyticGraspConfig
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove small components, fill holes, and perform open/close cleanup."""

    original = np.asarray(binary_mask)
    if original.ndim != 2:
        raise ValueError("binary mask must be 2D")
    mask = original.astype(bool, copy=True)
    input_labels, input_components = ndimage.label(mask)
    _ = input_labels
    if config.min_component_area_px > 0:
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
        keep = sizes >= config.min_component_area_px
        keep[0] = False
        mask = keep[labels]
    if config.max_hole_area_px > 0:
        holes = ndimage.binary_fill_holes(mask) & ~mask
        labels, count = ndimage.label(holes, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
        fill = sizes <= config.max_hole_area_px
        fill[0] = False
        mask = mask | fill[labels]
    if config.opening_radius_px > 0:
        mask = ndimage.binary_opening(
            mask, structure=morphology.disk(config.opening_radius_px)
        )
    if config.closing_radius_px > 0:
        mask = ndimage.binary_closing(
            mask, structure=morphology.disk(config.closing_radius_px)
        )
    if config.min_component_area_px > 0:
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
        keep = sizes >= config.min_component_area_px
        keep[0] = False
        mask = keep[labels]
    labels, component_count = ndimage.label(mask)
    if config.component_policy == "largest" and component_count > 1:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, component_count + 1))
        keep = int(np.argmax(sizes)) + 1
        mask = labels == keep
        labels, component_count = ndimage.label(mask)
        _ = labels
    input_area = int(np.count_nonzero(original))
    cleaned_area = int(np.count_nonzero(mask))
    original_binary = original.astype(bool, copy=False)
    return mask.astype(bool, copy=False), {
        "original_mask_area_px": input_area,
        "cleaned_mask_area_px": cleaned_area,
        "input_component_count": int(input_components),
        "cleaned_component_count": int(component_count),
        "removed_area_px": int(np.count_nonzero(original_binary & ~mask)),
        "filled_area_px": int(np.count_nonzero(~original_binary & mask)),
    }


def sample_contour_by_arclength(
    contour_xy: np.ndarray, *, spacing_px: float, max_points: int
) -> np.ndarray:
    """Uniformly resample a closed contour by cumulative arc length."""

    points = np.asarray(contour_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("contour must have shape [N,2] with N>=3")
    if spacing_px <= 0.0 or max_points < 3:
        raise ValueError("contour spacing and point limit must be positive")
    if not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    segment = np.diff(points, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    valid = lengths > 1e-9
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError("contour has insufficient nonzero arc length")
    start = points[:-1][valid]
    segment = segment[valid]
    lengths = lengths[valid]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    count = min(max_points, max(3, int(math.ceil(total / spacing_px))))
    distances = np.linspace(0.0, total, count, endpoint=False)
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(lengths) - 1)
    fractions = (distances - cumulative[indices]) / lengths[indices]
    return start[indices] + fractions[:, None] * segment[indices]


def _sample_bilinear(array: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    return ndimage.map_coordinates(
        np.asarray(array, dtype=np.float64),
        [points[:, 1], points[:, 0]],
        order=1,
        mode="nearest",
    )


def estimate_inward_normals(points_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Estimate tangent normals and orient them toward the mask interior."""

    points = np.asarray(points_xy, dtype=np.float64)
    target = np.asarray(mask, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("points must be a closed ordered contour sample")
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    norms = np.linalg.norm(tangent, axis=1)
    if np.any(norms <= 1e-9):
        raise ValueError("contour contains degenerate tangents")
    tangent /= norms[:, None]
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    distance = ndimage.distance_transform_edt(target)
    plus = _sample_bilinear(distance, points + 1.5 * normals)
    minus = _sample_bilinear(distance, points - 1.5 * normals)
    flip = minus > plus
    ties = np.isclose(plus, minus)
    if np.any(ties):
        rows, columns = np.nonzero(target)
        centroid = np.asarray([columns.mean(), rows.mean()], dtype=np.float64)
        toward_centroid = np.einsum("ij,ij->i", normals, centroid - points) >= 0.0
        flip[ties] = ~toward_centroid[ties]
    normals[flip] *= -1.0
    return normals


def estimate_boundary_curvature(points_xy: np.ndarray) -> np.ndarray:
    """Estimate unsigned local turning curvature in inverse pixels."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("points must have shape [N,2] with N>=3")
    previous = points - np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0) - points
    previous_length = np.linalg.norm(previous, axis=1)
    following_length = np.linalg.norm(following, axis=1)
    if np.any(previous_length <= 1e-9) or np.any(following_length <= 1e-9):
        raise ValueError("curvature points contain duplicate neighbours")
    previous /= previous_length[:, None]
    following /= following_length[:, None]
    turning = np.arccos(np.clip(np.einsum("ij,ij->i", previous, following), -1, 1))
    scale = 0.5 * (previous_length + following_length)
    return turning / scale


def generate_antipodal_pairs(
    points_xy: np.ndarray,
    inward_normals: np.ndarray,
    *,
    min_width_px: float,
    max_width_px: float,
    antipodal_alignment_min: float,
    normal_axis_alignment_min: float,
) -> list[dict[str, Any]]:
    """Use a cKDTree radius query over sampled contacts, never mask pixels."""

    points = np.asarray(points_xy, dtype=np.float64)
    normals = np.asarray(inward_normals, dtype=np.float64)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("contact points and normals must have matching [N,2] shape")
    if len(points) < 2:
        return []
    tree = cKDTree(points)
    pair_array = tree.query_pairs(float(max_width_px), output_type="ndarray")
    if pair_array.size == 0:
        return []
    result: list[dict[str, Any]] = []
    for first, second in np.asarray(pair_array, dtype=np.int64).reshape(-1, 2):
        delta = points[second] - points[first]
        width = float(np.linalg.norm(delta))
        if width < min_width_px or width > max_width_px:
            continue
        axis = delta / width
        forward = (
            float(np.dot(normals[first], axis)),
            float(np.dot(normals[second], -axis)),
        )
        reverse = (
            float(np.dot(normals[second], -axis)),
            float(np.dot(normals[first], axis)),
        )
        # Reverse only changes contact naming.  Keeping this explicit makes the
        # left/right depth and probability fields deterministic.
        if sum(reverse) > sum(forward):
            left, right = int(second), int(first)
            oriented_axis = -axis
            alignments = reverse
        else:
            left, right = int(first), int(second)
            oriented_axis = axis
            alignments = forward
        antipodal = float(np.mean(alignments))
        if min(alignments) < normal_axis_alignment_min:
            continue
        if antipodal < antipodal_alignment_min:
            continue
        result.append(
            {
                "left_index": left,
                "right_index": right,
                "width_px": width,
                "axis": oriented_axis,
                "antipodal_normal_alignment": antipodal,
                "minimum_normal_axis_alignment": float(min(alignments)),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            -row["antipodal_normal_alignment"],
            row["width_px"],
            row["left_index"],
            row["right_index"],
        ),
    )


def _line_points(start: np.ndarray, end: np.ndarray, spacing: float) -> np.ndarray:
    length = float(np.linalg.norm(end - start))
    count = max(2, int(math.ceil(length / spacing)) + 1)
    return np.linspace(start, end, count)


def _local_values(
    array: np.ndarray, center_xy: np.ndarray, radius: int, *, valid_only: bool = False
) -> np.ndarray:
    height, width = array.shape
    x = int(round(float(center_xy[0])))
    y = int(round(float(center_xy[1])))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    values = np.asarray(array[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if valid_only:
        values = values[values > 0.0]
    return values


def _local_valid_depth_support(
    depth_m: np.ndarray, center_xy: np.ndarray, radius: int
) -> float:
    height, width = depth_m.shape
    x = int(round(float(center_xy[0])))
    y = int(round(float(center_xy[1])))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64)
    if patch.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(patch) & (patch > 0.0)))


def _oriented_grid(
    center_xy: np.ndarray,
    axis: np.ndarray,
    *,
    half_axis_px: float,
    half_normal_px: float,
    spacing_px: float = 2.0,
) -> np.ndarray:
    perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    along = np.arange(-half_axis_px, half_axis_px + 0.5 * spacing_px, spacing_px)
    across = np.arange(-half_normal_px, half_normal_px + 0.5 * spacing_px, spacing_px)
    aa, bb = np.meshgrid(along, across, indexing="xy")
    return (
        center_xy[None, None, :]
        + aa[..., None] * axis[None, None, :]
        + bb[..., None] * perpendicular[None, None, :]
    ).reshape(-1, 2)


def visible_surface_occupancy(
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    points_xy: np.ndarray,
    *,
    reference_depth_m: float,
    closer_margin_m: float,
) -> float:
    """Visible foreground occupancy proxy over an image-plane gripper region."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.size == 0:
        return 0.0
    height, width = depth_m.shape
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    depth = np.asarray(depth_m, dtype=np.float64)[y, x]
    target = np.asarray(target_mask, dtype=bool)[y, x]
    eligible = np.isfinite(depth) & (depth > 0.0) & ~target
    if not np.any(eligible):
        return 0.0
    obstructing = depth[eligible] < float(reference_depth_m) - float(closer_margin_m)
    return float(np.mean(obstructing))


def score_candidate_features(
    features: Mapping[str, float], config: AnalyticGraspConfig
) -> tuple[float, dict[str, float]]:
    """Normalize analytic terms and return a finite score in ``[0,1]``."""

    missing = sorted(set(ANALYTIC_FEATURE_NAMES) - set(features))
    if missing:
        raise ValueError(f"analytic candidate is missing features: {missing}")
    if any(not math.isfinite(float(features[name])) for name in ANALYTIC_FEATURE_NAMES):
        raise ValueError("analytic candidate contains non-finite features")
    target_support = float(
        np.mean(
            [
                features["minimum_jaw_mask_support"],
                features["axis_mask_support"],
                features["center_mask_probability"],
            ]
        )
    )
    occupancy_risk = float(
        max(
            features["left_finger_occupancy"],
            features["right_finger_occupancy"],
            features["palm_occupancy"],
            features["approach_corridor_occupancy"],
        )
    )
    normalized = {
        "antipodal_alignment": float(np.clip(features["antipodal_normal_alignment"], 0, 1)),
        "target_mask_support": float(np.clip(target_support, 0, 1)),
        "contact_symmetry": float(np.clip(features["contact_symmetry"], 0, 1)),
        "boundary_margin": float(np.clip(features["boundary_margin"], 0, 1)),
        "visible_clearance": float(np.clip(features["visible_clearance"], 0, 1)),
        "width_compatibility": float(np.clip(features["width_margin"], 0, 1)),
        "jaw_depth_difference": float(
            np.clip(
                features["jaw_depth_difference"] / config.max_jaw_depth_difference_m,
                0,
                1,
            )
        ),
        "depth_edge_crossing": float(np.clip(features["depth_edge_crossing"], 0, 1)),
        "occupancy_risk": float(np.clip(occupancy_risk, 0, 1)),
    }
    weights = config.score_weights()
    positive_names = (
        "antipodal_alignment",
        "target_mask_support",
        "contact_symmetry",
        "boundary_margin",
        "visible_clearance",
        "width_compatibility",
    )
    negative_names = (
        "jaw_depth_difference",
        "depth_edge_crossing",
        "occupancy_risk",
    )
    positive = sum(weights[name] * normalized[name] for name in positive_names)
    negative = sum(weights[name] * normalized[name] for name in negative_names)
    negative_capacity = sum(weights[name] for name in negative_names)
    capacity = sum(weights.values())
    score = float(np.clip((positive - negative + negative_capacity) / capacity, 0.0, 1.0))
    if not math.isfinite(score):
        raise ValueError("analytic score is non-finite")
    normalized["weighted_positive"] = float(positive)
    normalized["weighted_penalty"] = float(negative)
    normalized["standardized_score"] = score
    return score, normalized


def _candidate_features(
    *,
    pair: Mapping[str, Any],
    points: np.ndarray,
    normals: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
    depth_m: np.ndarray,
    depth_gradient: np.ndarray,
    boundary_curvature: np.ndarray,
    distance_inside: np.ndarray,
    centroid_xy: np.ndarray,
    fx: float,
    config: AnalyticGraspConfig,
) -> tuple[dict[str, float], dict[str, Any]] | None:
    left_index = int(pair["left_index"])
    right_index = int(pair["right_index"])
    left = points[left_index]
    right = points[right_index]
    axis = np.asarray(pair["axis"], dtype=np.float64)
    width_px = float(pair["width_px"])
    center = 0.5 * (left + right)
    height, width = mask.shape
    center_x = int(np.clip(round(float(center[0])), 0, width - 1))
    center_y = int(np.clip(round(float(center[1])), 0, height - 1))
    if not mask[center_y, center_x]:
        return None
    center_probability = float(_sample_bilinear(probability, center[None, :])[0])
    if center_probability < config.min_center_probability:
        return None
    left_probability = float(_sample_bilinear(probability, left[None, :])[0])
    right_probability = float(_sample_bilinear(probability, right[None, :])[0])
    if min(left_probability, right_probability) < config.min_contact_probability:
        return None

    left_contact = left + 1.5 * normals[left_index]
    right_contact = right + 1.5 * normals[right_index]
    left_depth_values = _local_values(
        depth_m, left_contact, config.contact_patch_radius_px, valid_only=True
    )
    right_depth_values = _local_values(
        depth_m, right_contact, config.contact_patch_radius_px, valid_only=True
    )
    if left_depth_values.size == 0 or right_depth_values.size == 0:
        return None
    left_valid_depth_support = _local_valid_depth_support(
        depth_m, left_contact, config.contact_patch_radius_px
    )
    right_valid_depth_support = _local_valid_depth_support(
        depth_m, right_contact, config.contact_patch_radius_px
    )
    if (
        min(left_valid_depth_support, right_valid_depth_support)
        < config.min_contact_valid_depth_support
    ):
        return None
    left_depth = float(np.median(left_depth_values))
    right_depth = float(np.median(right_depth_values))
    left_depth_gradient = float(
        _sample_bilinear(depth_gradient, left_contact[None, :])[0]
    )
    right_depth_gradient = float(
        _sample_bilinear(depth_gradient, right_contact[None, :])[0]
    )
    jaw_depth_difference = abs(left_depth - right_depth)
    if jaw_depth_difference > config.max_jaw_depth_difference_m:
        return None
    mean_depth = 0.5 * (left_depth + right_depth)
    width_m = width_px * mean_depth / fx
    if not config.min_width_m <= width_m <= config.max_width_m:
        return None

    line = _line_points(left_contact, right_contact, config.axis_sample_spacing_px)
    line_mask = _sample_bilinear(mask.astype(np.float32), line)
    axis_mask_support = float(np.mean(line_mask >= 0.5))
    if axis_mask_support < config.min_axis_mask_support:
        return None
    line_depth = _sample_bilinear(depth_m, line)
    valid_depth = np.isfinite(line_depth) & (line_depth > 0.0)
    if float(np.mean(valid_depth)) < config.min_axis_valid_depth_fraction:
        return None
    valid_line_depth = line_depth[valid_depth]
    jumps = np.abs(np.diff(valid_line_depth))
    maximum_jump = float(jumps.max(initial=0.0))
    if maximum_jump > config.max_axis_depth_jump_m:
        return None
    depth_edge_crossing = (
        0.0
        if jumps.size == 0
        else float(np.mean(jumps > 0.5 * config.max_axis_depth_jump_m))
    )
    depth_continuity = float(
        1.0 - np.clip(maximum_jump / config.max_axis_depth_jump_m, 0.0, 1.0)
    )

    left_support_values = _local_values(
        mask.astype(np.float32), left_contact, config.contact_patch_radius_px
    )
    right_support_values = _local_values(
        mask.astype(np.float32), right_contact, config.contact_patch_radius_px
    )
    left_support = float(np.mean(left_support_values))
    right_support = float(np.mean(right_support_values))
    minimum_jaw_support = min(left_support, right_support)
    local_depth = np.concatenate((left_depth_values, right_depth_values))
    local_depth_variance = float(np.var(local_depth))
    probability_symmetry = 1.0 - abs(left_probability - right_probability)
    depth_symmetry = math.exp(
        -jaw_depth_difference / max(config.max_jaw_depth_difference_m, 1e-9)
    )
    contact_symmetry = float(np.clip(0.5 * (probability_symmetry + depth_symmetry), 0, 1))
    boundary_distance = float(distance_inside[center_y, center_x])
    boundary_margin = float(np.clip(boundary_distance / max(0.5 * width_px, 1.0), 0, 1))
    width_position = (width_m - config.min_width_m) / (
        config.max_width_m - config.min_width_m
    )
    width_margin = float(np.clip(1.0 - 2.0 * abs(width_position - 0.5), 0, 1))

    finger_half_axis = 0.5 * config.finger_thickness_px
    finger_half_normal = 0.5 * config.finger_length_px
    left_finger_center = left - axis * (finger_half_axis + 1.0)
    right_finger_center = right + axis * (finger_half_axis + 1.0)
    left_finger_points = _oriented_grid(
        left_finger_center,
        axis,
        half_axis_px=finger_half_axis,
        half_normal_px=finger_half_normal,
    )
    right_finger_points = _oriented_grid(
        right_finger_center,
        axis,
        half_axis_px=finger_half_axis,
        half_normal_px=finger_half_normal,
    )
    left_finger_occupancy = visible_surface_occupancy(
        depth_m,
        mask,
        left_finger_points,
        reference_depth_m=mean_depth,
        closer_margin_m=config.visible_depth_margin_m,
    )
    right_finger_occupancy = visible_surface_occupancy(
        depth_m,
        mask,
        right_finger_points,
        reference_depth_m=mean_depth,
        closer_margin_m=config.visible_depth_margin_m,
    )
    perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    palm_offset = 0.5 * config.fixed_height_px + finger_half_axis
    palm_values = []
    for sign in (-1.0, 1.0):
        palm_points = _oriented_grid(
            center + sign * perpendicular * palm_offset,
            axis,
            half_axis_px=0.5 * width_px,
            half_normal_px=finger_half_axis,
        )
        palm_values.append(
            visible_surface_occupancy(
                depth_m,
                mask,
                palm_points,
                reference_depth_m=mean_depth,
                closer_margin_m=config.visible_depth_margin_m,
            )
        )
    palm_occupancy = float(max(palm_values))
    corridor_points = _oriented_grid(
        center,
        axis,
        half_axis_px=0.5 * width_px,
        half_normal_px=0.5 * config.fixed_height_px,
    )
    approach_corridor_occupancy = visible_surface_occupancy(
        depth_m,
        mask,
        corridor_points,
        reference_depth_m=mean_depth,
        closer_margin_m=config.visible_depth_margin_m,
    )
    occupancy_risk = max(
        left_finger_occupancy,
        right_finger_occupancy,
        palm_occupancy,
        approach_corridor_occupancy,
    )
    visible_clearance = float(1.0 - np.clip(occupancy_risk, 0.0, 1.0))
    image_diagonal = max(math.hypot(width, height), 1.0)
    centroid_distance = float(np.linalg.norm(center - centroid_xy) / image_diagonal)
    features = {
        "antipodal_normal_alignment": float(pair["antipodal_normal_alignment"]),
        "left_mask_probability": left_probability,
        "right_mask_probability": right_probability,
        "minimum_jaw_mask_support": minimum_jaw_support,
        "axis_mask_support": axis_mask_support,
        "center_mask_probability": center_probability,
        "boundary_margin": boundary_margin,
        "left_depth": left_depth,
        "right_depth": right_depth,
        "left_depth_gradient": left_depth_gradient,
        "right_depth_gradient": right_depth_gradient,
        "left_boundary_curvature": float(boundary_curvature[left_index]),
        "right_boundary_curvature": float(boundary_curvature[right_index]),
        "left_valid_depth_support": left_valid_depth_support,
        "right_valid_depth_support": right_valid_depth_support,
        "jaw_depth_difference": jaw_depth_difference,
        "local_depth_variance": local_depth_variance,
        "depth_continuity_along_axis": depth_continuity,
        "depth_edge_crossing": depth_edge_crossing,
        "contact_symmetry": contact_symmetry,
        "width_px": width_px,
        "width_m": width_m,
        "width_margin": width_margin,
        "left_finger_occupancy": left_finger_occupancy,
        "right_finger_occupancy": right_finger_occupancy,
        "palm_occupancy": palm_occupancy,
        "approach_corridor_occupancy": approach_corridor_occupancy,
        "visible_clearance": visible_clearance,
        "target_centroid_distance": centroid_distance,
        "candidate_uniqueness": 0.0,
    }
    geometry = {
        "center": center,
        "axis": axis,
        "left_contact": left,
        "right_contact": right,
        "source_key": f"contacts:{left_index}:{right_index}",
    }
    return features, geometry


def _resize_to_shape(array: np.ndarray, shape: tuple[int, int], *, binary: bool) -> np.ndarray:
    if array.shape == shape:
        return array
    interpolation = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    resized = cv2.resize(
        np.asarray(array), (int(shape[1]), int(shape[0])), interpolation=interpolation
    )
    return resized


def _sample_value(sample: object, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        for name in names:
            if name in sample and sample[name] is not None:
                return sample[name]
    else:
        for name in names:
            value = getattr(sample, name, None)
            if value is not None:
                return value
    return default


def _load_array(value: Any, *, unchanged_image: bool = True) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(f"ambiguous array archive: {path}")
            return np.asarray(archive[archive.files[0]])
    flag = cv2.IMREAD_UNCHANGED if unchanged_image else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"unable to load image array: {path}")
    return np.asarray(image)


def _load_intrinsics(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = Path(str(value)).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for key in ("fx", "fy", "cx", "cy", "depth_scale"):
        if key in payload and payload[key] is not None:
            result[key] = float(payload[key])
    if "fx" not in result or not math.isfinite(result["fx"]) or result["fx"] <= 0.0:
        raise ValueError("intrinsics must contain a finite positive fx")
    result.setdefault("fy", result["fx"])
    result.setdefault("depth_scale", 1000.0)
    return result


class MaskDepthAnalyticBackend:
    """Generate and rank planar antipodal grasps without a learned model."""

    backend_name = BACKEND_NAME

    def __init__(self, config: AnalyticGraspConfig | None = None) -> None:
        self.config = config or AnalyticGraspConfig()

    def _empty_prediction(
        self,
        *,
        sample_id: str,
        reason: str,
        started: float,
        metadata: Mapping[str, Any] | None = None,
        conditioning_variant: str = "predicted_probability_binary_mask_depth",
    ) -> GraspPrediction:
        return GraspPrediction(
            sample_id=sample_id,
            backend=self.backend_name,
            conditioning_variant=conditioning_variant,
            raw_candidate_count=0,
            nms_candidate_count=0,
            top1=None,
            top5=(),
            candidates=(),
            empty_reason=reason,
            runtime_seconds=time.perf_counter() - started,
            device="cpu",
            metadata=dict(metadata or {}),
        )

    def generate_candidates(
        self,
        *,
        sample_id: str,
        probability: np.ndarray,
        binary_mask: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: Mapping[str, float],
    ) -> tuple[list[Grasp4DoF], dict[str, Any]]:
        probability = np.asarray(probability, dtype=np.float32)
        depth = np.asarray(depth_m, dtype=np.float32)
        mask_input = np.asarray(binary_mask)
        if depth.ndim != 2 or probability.ndim != 2 or mask_input.ndim != 2:
            raise ValueError("probability, mask, and depth must all be 2D")
        probability = _resize_to_shape(probability, depth.shape, binary=False).astype(
            np.float32, copy=False
        )
        mask_input = _resize_to_shape(mask_input, depth.shape, binary=True)
        if not np.isfinite(probability).all() or float(probability.min()) < 0.0 or float(
            probability.max()
        ) > 1.0:
            raise ValueError("predicted probability must be finite in [0,1]")
        binary = np.asarray(mask_input) > 0
        cleaned, mask_stats = clean_predicted_mask(binary, self.config)
        stats: dict[str, Any] = {
            **mask_stats,
            "clearance_proxy_semantics": CLEARANCE_PROXY_SEMANTICS,
        }
        if not np.any(cleaned):
            stats["empty_reason"] = (
                "empty_mask" if not np.any(binary) else "mask_below_min_area"
            )
            return [], stats
        valid_depth_in_mask = cleaned & np.isfinite(depth) & (depth > 0.0)
        stats["valid_depth_in_mask_px"] = int(np.count_nonzero(valid_depth_in_mask))
        stats["valid_depth_fraction"] = float(
            np.count_nonzero(valid_depth_in_mask) / np.count_nonzero(cleaned)
        )
        if not np.any(valid_depth_in_mask):
            stats["empty_reason"] = "invalid_depth"
            return [], stats

        contours_rc = measure.find_contours(cleaned.astype(np.float32), 0.5)
        if not contours_rc:
            stats["empty_reason"] = "no_contour"
            return [], stats
        contours_xy = [np.column_stack((item[:, 1], item[:, 0])) for item in contours_rc]
        contours_xy.sort(key=lambda item: -len(item))
        point_groups: list[np.ndarray] = []
        remaining = self.config.max_contour_points
        for contour in contours_xy:
            if remaining < 3:
                break
            try:
                sampled = sample_contour_by_arclength(
                    contour,
                    spacing_px=self.config.contour_spacing_px,
                    max_points=remaining,
                )
            except ValueError:
                continue
            point_groups.append(sampled)
            remaining -= len(sampled)
        if not point_groups:
            stats["empty_reason"] = "no_contour"
            return [], stats
        points = np.concatenate(point_groups, axis=0)
        # Estimate normals separately per connected contour so roll() cannot
        # construct a tangent between unrelated components.
        normal_groups = [estimate_inward_normals(group, cleaned) for group in point_groups]
        normals = np.concatenate(normal_groups, axis=0)
        curvature_groups = [estimate_boundary_curvature(group) for group in point_groups]
        boundary_curvature = np.concatenate(curvature_groups, axis=0)
        pairs = generate_antipodal_pairs(
            points,
            normals,
            min_width_px=self.config.min_width_px,
            max_width_px=self.config.max_width_px,
            antipodal_alignment_min=self.config.antipodal_alignment_min,
            normal_axis_alignment_min=self.config.normal_axis_alignment_min,
        )
        stats["sampled_contour_points"] = len(points)
        stats["kd_tree_width_pairs"] = len(pairs)
        if not pairs:
            stats["empty_reason"] = "no_candidate_generated"
            return [], stats
        fx = float(intrinsics["fx"])
        if not math.isfinite(fx) or fx <= 0.0:
            raise ValueError("intrinsics fx must be finite and positive")
        rows, columns = np.nonzero(cleaned)
        centroid = np.asarray([columns.mean(), rows.mean()], dtype=np.float64)
        distance_inside = ndimage.distance_transform_edt(cleaned)
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        fill_depth = float(np.median(depth[valid_depth]))
        depth_for_gradient = np.where(valid_depth, depth, fill_depth)
        depth_for_gradient = ndimage.gaussian_filter(depth_for_gradient, sigma=1.0)
        gradient_y, gradient_x = np.gradient(depth_for_gradient)
        depth_gradient = np.hypot(gradient_x, gradient_y)
        staged: list[tuple[dict[str, float], dict[str, Any]]] = []
        for pair in pairs:
            result = _candidate_features(
                pair=pair,
                points=points,
                normals=normals,
                probability=probability,
                mask=cleaned,
                depth_m=depth,
                depth_gradient=depth_gradient,
                boundary_curvature=boundary_curvature,
                distance_inside=distance_inside,
                centroid_xy=centroid,
                fx=fx,
                config=self.config,
            )
            if result is not None:
                staged.append(result)
        if not staged:
            stats["empty_reason"] = "no_candidate_generated"
            return [], stats

        centers = np.asarray([item[1]["center"] for item in staged], dtype=np.float64)
        if len(staged) == 1:
            uniqueness = np.ones(1, dtype=np.float64)
        else:
            tree = cKDTree(centers)
            distances, _ = tree.query(centers, k=2)
            uniqueness = np.clip(
                distances[:, 1] / max(self.config.max_width_px, 1.0), 0.0, 1.0
            )
        candidates: list[Grasp4DoF] = []
        for index, ((features, geometry), unique) in enumerate(
            zip(staged, uniqueness, strict=True)
        ):
            features["candidate_uniqueness"] = float(unique)
            score, normalized = score_candidate_features(features, self.config)
            center = geometry["center"]
            axis = geometry["axis"]
            angle = normalize_angle_deg(math.degrees(math.atan2(axis[1], axis[0])))
            candidate_id = stable_candidate_id(
                sample_id=sample_id,
                backend=self.backend_name,
                center_x=float(center[0]),
                center_y=float(center[1]),
                angle_deg=angle,
                width_px=features["width_px"],
                source_key=geometry["source_key"],
            )
            candidates.append(
                Grasp4DoF(
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    angle_deg=angle,
                    width_px=features["width_px"],
                    height_px=self.config.fixed_height_px,
                    score=score,
                    candidate_id=candidate_id,
                    metadata={
                        "features": dict(features),
                        "normalized_score_terms": normalized,
                        "left_contact_xy": geometry["left_contact"].tolist(),
                        "right_contact_xy": geometry["right_contact"].tolist(),
                        "clearance_proxy_semantics": CLEARANCE_PROXY_SEMANTICS,
                    },
                )
            )
        ranked = rank_candidates(candidates)[: self.config.max_raw_candidates]
        stats["geometrically_valid_candidates"] = len(staged)
        stats["retained_raw_candidates"] = len(ranked)
        stats["empty_reason"] = None
        return ranked, stats

    def predict(self, sample: object) -> GraspPrediction:
        """Predict Top-1/Top-5 from duck-typed sample arrays or file paths."""

        started = time.perf_counter()
        sample_id = str(_sample_value(sample, ("sample_id",), ""))
        if not sample_id:
            raise ValueError("sample must expose a non-empty sample_id")
        mask_source = str(_sample_value(sample, ("mask_source",), "predicted"))
        if mask_source == "gt_mask_oracle" and not self.config.allow_oracle:
            from .base import GroundTruthAccessError

            raise GroundTruthAccessError(
                "gt_mask_oracle is forbidden by the main predicted-mask protocol"
            )
        if mask_source not in ("predicted", "gt_mask_oracle"):
            raise ValueError(f"unsupported mask_source: {mask_source}")
        conditioning_label = (
            "predicted_probability_binary_mask_depth"
            if mask_source == "predicted"
            else "gt_mask_oracle_binary_mask_depth"
        )
        intrinsics_value = _sample_value(
            sample, ("intrinsics", "camera_intrinsics", "intrinsics_path")
        )
        if intrinsics_value is None:
            return self._empty_prediction(
                sample_id=sample_id,
                reason="missing_intrinsics",
                started=started,
                metadata={"clearance_proxy_semantics": CLEARANCE_PROXY_SEMANTICS},
                conditioning_variant=conditioning_label,
            )
        probability_value = _sample_value(
            sample,
            (
                "predicted_probability",
                "probability",
                "predicted_probability_path",
                "probability_path",
            ),
        )
        if probability_value is None:
            return self._empty_prediction(
                sample_id=sample_id,
                reason="missing_probability",
                started=started,
                conditioning_variant=conditioning_label,
            )
        mask_value = _sample_value(
            sample,
            ("predicted_mask", "binary_mask", "predicted_mask_path", "mask_path"),
        )
        if mask_value is None:
            return self._empty_prediction(
                sample_id=sample_id, reason="missing_mask", started=started
                , conditioning_variant=conditioning_label
            )
        depth_value = _sample_value(
            sample, ("depth", "depth_m", "depth_path", "source_depth_path")
        )
        if depth_value is None:
            return self._empty_prediction(
                sample_id=sample_id, reason="missing_depth", started=started
                , conditioning_variant=conditioning_label
            )
        intrinsics = _load_intrinsics(intrinsics_value)
        probability = _load_array(probability_value)
        binary_mask = _load_array(mask_value)
        raw_depth = _load_array(depth_value)
        if raw_depth.ndim == 3:
            raw_depth = raw_depth[..., 0]
        depth = np.asarray(raw_depth, dtype=np.float32)
        if np.issubdtype(np.asarray(raw_depth).dtype, np.integer) or float(
            np.nanmax(depth, initial=0.0)
        ) > 20.0:
            depth /= float(intrinsics.get("depth_scale", 1000.0))
        raw_candidates, stats = self.generate_candidates(
            sample_id=sample_id,
            probability=probability,
            binary_mask=binary_mask,
            depth_m=depth,
            intrinsics=intrinsics,
        )
        if not raw_candidates:
            return self._empty_prediction(
                sample_id=sample_id,
                reason=str(stats.get("empty_reason") or "no_candidate_generated"),
                started=started,
                metadata=stats,
                conditioning_variant=conditioning_label,
            )
        nms_candidates = non_maximum_suppression(raw_candidates, self.config.nms)
        ranked = tuple(rank_candidates(nms_candidates)[:5])
        if not ranked:
            return self._empty_prediction(
                sample_id=sample_id,
                reason="no_candidate_after_nms",
                started=started,
                metadata=stats,
                conditioning_variant=conditioning_label,
            )
        return GraspPrediction(
            sample_id=sample_id,
            backend=self.backend_name,
            conditioning_variant=conditioning_label,
            raw_candidate_count=len(raw_candidates),
            nms_candidate_count=len(nms_candidates),
            top1=ranked[0],
            top5=ranked,
            candidates=tuple(rank_candidates(nms_candidates)),
            empty_reason=None,
            runtime_seconds=time.perf_counter() - started,
            device="cpu",
            metadata=stats,
        )


GraspBackend = MaskDepthAnalyticBackend
