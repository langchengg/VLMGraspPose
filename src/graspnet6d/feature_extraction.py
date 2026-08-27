"""Runtime-only geometric features for immutable GraspNet-frame candidates.

The official evaluator is not called here. Collision, contact, and target
support values are observational proxies from RGB-D plus the selected mask and
must never be confused with official collision/friction supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from .contracts import Candidate6D, validate_candidate_pool
from .features import default_feature_schema
from .geometry import CameraIntrinsics, project_points


FINGER_THICKNESS_M = 0.01
PALM_DEPTH_M = 0.02
APPROACH_SWEEP_M = 0.10
EPS = 1e-9


@dataclass(frozen=True)
class RuntimeObservation:
    intrinsics: CameraIntrinsics
    mask_probability: np.ndarray
    depth_m: np.ndarray
    scene_points_camera_m: np.ndarray
    table_normal_camera: np.ndarray
    gravity_camera: np.ndarray
    grounding_condition: str
    mask_source_sha256: str
    depth_source_sha256: str
    scene_points_source_sha256: str
    scene_points_source_kind: str = "full_depth_backprojection"

    def validated(self) -> "RuntimeObservation":
        probability = np.asarray(self.mask_probability, dtype=np.float64)
        depth = np.asarray(self.depth_m, dtype=np.float64)
        scene = np.asarray(self.scene_points_camera_m, dtype=np.float64)
        if probability.ndim != 2 or depth.shape != probability.shape:
            raise ValueError("mask probability and depth must share HxW shape")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("mask probability must be finite in [0,1]")
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError("depth must be finite and non-negative metres")
        if scene.ndim != 2 or scene.shape[1] != 3 or not np.isfinite(scene).all() or not len(scene):
            raise ValueError("scene points must be a non-empty finite Nx3 array")
        for name, vector in (("table normal", self.table_normal_camera), ("gravity", self.gravity_camera)):
            value = np.asarray(vector, dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) <= EPS:
                raise ValueError(f"{name} must be a finite non-zero 3-vector")
        if self.intrinsics.width != probability.shape[1] or self.intrinsics.height != probability.shape[0]:
            raise ValueError("intrinsics dimensions disagree with mask")
        conditions = {
            "oracle_gt_mask",
            "hifics_zero_shot_mask",
            "hifics_adapted_mask",
        }
        if self.grounding_condition not in conditions:
            raise ValueError(f"unsupported grounding condition: {self.grounding_condition!r}")
        if self.grounding_condition == "oracle_gt_mask" and not np.isin(probability, [0.0, 1.0]).all():
            raise ValueError("oracle_gt_mask probabilities must be exactly binary")
        digest = re.compile(r"^[0-9a-f]{64}$")
        for name, value in (
            ("mask_source_sha256", self.mask_source_sha256),
            ("depth_source_sha256", self.depth_source_sha256),
            ("scene_points_source_sha256", self.scene_points_source_sha256),
        ):
            if not isinstance(value, str) or digest.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.scene_points_source_kind not in {
            "full_depth_backprojection",
            "deterministic_voxel_downsample_of_full_depth",
        }:
            raise ValueError("scene points must derive from the complete selected depth image")
        # Reject an independently supplied or stale scene cloud. A deterministic
        # sample must project to valid pixels whose stored depth agrees with Z.
        sample_indices = np.linspace(0, len(scene) - 1, min(len(scene), 4096), dtype=np.int64)
        sample = scene[sample_indices]
        if np.any(sample[:, 2] <= EPS):
            raise ValueError("scene points must have positive camera-frame depth")
        pixels = project_points(sample, self.intrinsics)
        columns = np.rint(pixels[:, 0]).astype(np.int64)
        rows = np.rint(pixels[:, 1]).astype(np.int64)
        inside = (
            (rows >= 0)
            & (rows < depth.shape[0])
            & (columns >= 0)
            & (columns < depth.shape[1])
        )
        if not inside.all():
            raise ValueError("scene points do not project into the source depth image")
        observed_depth = depth[rows, columns]
        tolerance = np.maximum(0.002, 0.005 * sample[:, 2])
        if not np.all((observed_depth > 0) & (np.abs(observed_depth - sample[:, 2]) <= tolerance)):
            raise ValueError("scene points disagree with the source depth image")
        if not np.any((probability >= 0.5) & (depth > 0)):
            raise ValueError("selected grounding mask contains no valid target depth")
        return self

    def target_points_camera_m(self) -> np.ndarray:
        """Backproject target geometry only from this condition's mask and depth."""

        probability = np.asarray(self.mask_probability, dtype=np.float64)
        depth = np.asarray(self.depth_m, dtype=np.float64)
        rows, columns = np.nonzero((probability >= 0.5) & (depth > 0))
        z = depth[rows, columns]
        x = (columns - self.intrinsics.cx) * z / self.intrinsics.fx
        y = (rows - self.intrinsics.cy) * z / self.intrinsics.fy
        return np.column_stack((x, y, z))


def _angle(left: np.ndarray, right: np.ndarray) -> float:
    a = left / max(np.linalg.norm(left), EPS)
    b = right / max(np.linalg.norm(right), EPS)
    return float(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and abs(denominator) > EPS else np.nan


def _sample_probability(probability: np.ndarray, pixels_uv: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    result = np.full(len(pixels), np.nan, dtype=np.float64)
    finite = np.isfinite(pixels).all(axis=1)
    columns = np.zeros(len(pixels), dtype=np.int64)
    rows = np.zeros(len(pixels), dtype=np.int64)
    columns[finite] = np.rint(pixels[finite, 0]).astype(np.int64)
    rows[finite] = np.rint(pixels[finite, 1]).astype(np.int64)
    inside = finite & (rows >= 0) & (rows < probability.shape[0]) & (columns >= 0) & (columns < probability.shape[1])
    result[inside] = probability[rows[inside], columns[inside]]
    return result


def _project_local_samples(
    candidate: Candidate6D,
    intrinsics: CameraIntrinsics,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    counts: tuple[int, int, int] = (5, 5, 3),
) -> np.ndarray:
    coordinates = [np.linspace(low, high, count) for (low, high), count in zip((x_range, y_range, z_range), counts, strict=True)]
    x, y, z = np.meshgrid(*coordinates, indexing="ij")
    local = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    rotation = np.asarray(candidate.rotation_camera, dtype=np.float64)
    translation = np.asarray(candidate.translation_camera_m, dtype=np.float64)
    camera = local @ rotation.T + translation
    positive = camera[:, 2] > EPS
    pixels = np.full((len(camera), 2), np.nan)
    if positive.any():
        pixels[positive] = project_points(camera[positive], intrinsics)
    return pixels


def _coverage(probability: np.ndarray, pixels: np.ndarray) -> float:
    values = _sample_probability(probability, pixels)
    finite = np.isfinite(values)
    return np.nan if not finite.any() else float(np.mean(values[finite]))


def _gripper_coordinates(points_camera: np.ndarray, candidate: Candidate6D) -> np.ndarray:
    # R columns are gripper axes in camera. Row-vector equivalent is @ R.
    return (points_camera - np.asarray(candidate.translation_camera_m)) @ np.asarray(candidate.rotation_camera)


def _region_masks(local: np.ndarray, candidate: Candidate6D) -> dict[str, np.ndarray]:
    width = candidate.width_m
    height = candidate.height_m
    depth = candidate.depth_m
    vertical = np.abs(local[:, 2]) < height / 2
    finger_depth = (local[:, 0] > -PALM_DEPTH_M) & (local[:, 0] < depth)
    left = vertical & finger_depth & (local[:, 1] > width / 2) & (local[:, 1] < width / 2 + FINGER_THICKNESS_M)
    right = vertical & finger_depth & (local[:, 1] < -width / 2) & (local[:, 1] > -width / 2 - FINGER_THICKNESS_M)
    palm = vertical & (local[:, 0] > -PALM_DEPTH_M - FINGER_THICKNESS_M) & (local[:, 0] < -PALM_DEPTH_M) & (np.abs(local[:, 1]) < width / 2 + FINGER_THICKNESS_M)
    closing = vertical & finger_depth & (np.abs(local[:, 1]) <= width / 2)
    approach = vertical & (local[:, 0] >= -APPROACH_SWEEP_M) & (local[:, 0] <= -PALM_DEPTH_M) & (np.abs(local[:, 1]) <= width / 2 + FINGER_THICKNESS_M)
    return {"left": left, "right": right, "palm": palm, "closing": closing, "approach": approach}


def _covariance_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 3:
        return np.full(3, np.nan), np.eye(3)
    covariance = np.cov(points - points.mean(axis=0), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0), vectors[:, order]


def _mask_properties(probability: np.ndarray, depth: np.ndarray) -> dict[str, float]:
    binary = probability >= 0.5
    area = int(binary.sum())
    labels, components = ndimage.label(binary)
    sizes = np.bincount(labels.ravel())[1:] if components else np.zeros(0, dtype=int)
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
    eroded = ndimage.binary_erosion(binary)
    boundary = binary ^ eroded
    return {
        "mask_area_fraction": float(area / binary.size),
        "mask_entropy": float(entropy.mean()),
        "mean_mask_confidence": float(probability[binary].mean()) if area else np.nan,
        "boundary_uncertainty": float(entropy[boundary].mean()) if boundary.any() else np.nan,
        "mask_depth_valid_fraction": float(np.mean(depth[binary] > 0)) if area else np.nan,
        "number_of_mask_components": float(components),
        "largest_component_fraction": float(sizes.max() / area) if area and len(sizes) else np.nan,
    }


def extract_candidate_features(
    candidates: Iterable[Candidate6D], observation: RuntimeObservation
) -> pd.DataFrame:
    """Return the exact v1 runtime feature table in immutable candidate order."""

    values = validate_candidate_pool(candidates)
    observation.validated()
    probability = np.asarray(observation.mask_probability, dtype=np.float64)
    depth = np.asarray(observation.depth_m, dtype=np.float64)
    scene = np.asarray(observation.scene_points_camera_m, dtype=np.float64)
    target = observation.target_points_camera_m()
    target_center = np.median(target, axis=0)
    target_extent = np.ptp(target, axis=0)
    target_diagonal = float(np.linalg.norm(target_extent))
    target_eigenvalues, target_axes = _covariance_geometry(target)
    target_tree = cKDTree(target)
    binary = probability >= 0.5
    distance_inside = ndimage.distance_transform_edt(binary)
    distance_outside = ndimage.distance_transform_edt(~binary)
    signed_distance = distance_inside - distance_outside
    rows, columns = np.nonzero(binary)
    mask_centroid_uv = np.array([columns.mean(), rows.mean()]) if len(rows) else np.array([np.nan, np.nan])
    image_diagonal = float(np.hypot(*probability.shape))
    mask_features = _mask_properties(probability, depth)
    scores = np.asarray([item.native_score for item in values], dtype=np.float64)
    top_score = float(scores.max()) if len(scores) else np.nan
    mean_score = float(scores.mean()) if len(scores) else np.nan
    std_score = float(scores.std()) if len(scores) else np.nan
    score_order = np.argsort(-scores, kind="stable") if len(scores) else np.zeros(0, dtype=int)
    percentile = np.empty(len(scores), dtype=float)
    if len(scores):
        percentile[score_order] = 1.0 - np.arange(len(scores)) / max(1, len(scores) - 1)
    records: list[dict[str, float]] = []
    for index, candidate in enumerate(values):
        rotation = np.asarray(candidate.rotation_camera)
        center = np.asarray(candidate.translation_camera_m)
        center_uv = project_points(center, observation.intrinsics) if center[2] > EPS else np.array([np.nan, np.nan])
        center_probability = _sample_probability(probability, center_uv[None])[0]
        column, row = np.rint(center_uv).astype(int, casting="unsafe") if np.isfinite(center_uv).all() else (-1, -1)
        inside_image = 0 <= row < probability.shape[0] and 0 <= column < probability.shape[1]
        boundary_distance = signed_distance[row, column] / image_diagonal if inside_image else np.nan
        centroid_distance = float(np.linalg.norm(center_uv - mask_centroid_uv) / image_diagonal) if np.isfinite(center_uv).all() else np.nan
        target_local = _gripper_coordinates(target, candidate)
        target_masks = _region_masks(target_local, candidate)
        nearest_target_distance, _ = target_tree.query(scene, k=1, workers=1)
        non_target = scene[nearest_target_distance > 0.005]
        non_target_local = _gripper_coordinates(non_target, candidate)
        obstacle_masks = _region_masks(non_target_local, candidate)
        total_scene = max(len(scene), 1)
        total_target = max(len(target), 1)
        left_support = float(target_masks["left"].sum() / total_target)
        right_support = float(target_masks["right"].sum() / total_target)
        radius_mask = np.linalg.norm(scene - center, axis=1) <= 0.05
        local_points = scene[radius_mask]
        local_eigenvalues, _ = _covariance_geometry(local_points)
        contact_points = target[target_masks["closing"]]
        contact_eigenvalues, _ = _covariance_geometry(contact_points)
        surface_normal = target_axes[:, -1]
        nearest_obstacle = float(np.min(np.linalg.norm(non_target - center, axis=1))) if len(non_target) else np.nan
        closing_pixels = _project_local_samples(candidate, observation.intrinsics, (0, candidate.depth_m), (-candidate.width_m / 2, candidate.width_m / 2), (-candidate.height_m / 2, candidate.height_m / 2))
        left_pixels = _project_local_samples(candidate, observation.intrinsics, (-PALM_DEPTH_M, candidate.depth_m), (candidate.width_m / 2, candidate.width_m / 2 + FINGER_THICKNESS_M), (-candidate.height_m / 2, candidate.height_m / 2))
        right_pixels = _project_local_samples(candidate, observation.intrinsics, (-PALM_DEPTH_M, candidate.depth_m), (-candidate.width_m / 2 - FINGER_THICKNESS_M, -candidate.width_m / 2), (-candidate.height_m / 2, candidate.height_m / 2))
        approach_pixels = _project_local_samples(candidate, observation.intrinsics, (-APPROACH_SWEEP_M, -PALM_DEPTH_M), (-candidate.width_m / 2, candidate.width_m / 2), (-candidate.height_m / 2, candidate.height_m / 2))
        previous = scores[index - 1] if index else scores[index]
        following = scores[index + 1] if index + 1 < len(scores) else scores[index]
        target_center_local = (target_center - center) @ rotation
        record: dict[str, float] = {
            "native_score": float(candidate.native_score),
            "native_rank": float(candidate.native_rank),
            "score_to_top1": float(top_score - candidate.native_score),
            "score_to_previous": float(previous - candidate.native_score),
            "score_to_next": float(candidate.native_score - following),
            "score_zscore_within_group": _safe_ratio(candidate.native_score - mean_score, std_score),
            "score_percentile": float(percentile[index]),
            "center_in_mask": float(inside_image and binary[row, column]),
            "center_mask_probability": float(center_probability),
            "distance_to_mask_boundary": float(boundary_distance),
            "normalised_distance_to_mask_centroid": float(centroid_distance),
            "closing_region_mask_coverage": _coverage(probability, closing_pixels),
            "left_finger_projected_mask_coverage": _coverage(probability, left_pixels),
            "right_finger_projected_mask_coverage": _coverage(probability, right_pixels),
            "approach_region_mask_coverage": _coverage(probability, approach_pixels),
            "target_point_fraction_inside_closing_volume": float(target_masks["closing"].sum() / total_target),
            "target_points_between_fingers": float(target_masks["closing"].sum()),
            "left_contact_target_support": left_support,
            "right_contact_target_support": right_support,
            "target_support_symmetry": 1.0 - _safe_ratio(abs(left_support - right_support), left_support + right_support + EPS),
            "candidate_center_to_target_centroid": float(np.linalg.norm(center - target_center)),
            "candidate_center_relative_to_target_bbox_x": _safe_ratio(target_center_local[0], target_extent[0]),
            "candidate_center_relative_to_target_bbox_y": _safe_ratio(target_center_local[1], target_extent[1]),
            "candidate_center_relative_to_target_bbox_z": _safe_ratio(target_center_local[2], target_extent[2]),
            "gripper_width_m": float(candidate.width_m),
            "width_over_target_bbox_x": _safe_ratio(candidate.width_m, target_extent[0]),
            "width_over_target_bbox_y": _safe_ratio(candidate.width_m, target_extent[1]),
            "width_over_target_bbox_diagonal": _safe_ratio(candidate.width_m, target_diagonal),
            "estimated_closing_margin": float(candidate.width_m - target_extent[1]),
            "width_feasibility_flag": float(candidate.width_m >= min(target_extent[0], target_extent[1])),
            **{f"rotation_6d_{axis}": float(value) for axis, value in enumerate(np.concatenate((rotation[:, 0], rotation[:, 1])))},
            "approach_vs_gravity_angle": _angle(rotation[:, 0], np.asarray(observation.gravity_camera)),
            "approach_vs_table_normal_angle": _angle(rotation[:, 0], np.asarray(observation.table_normal_camera)),
            "closing_axis_vs_target_pca_axis_0": _angle(rotation[:, 1], target_axes[:, 0]),
            "closing_axis_vs_target_pca_axis_1": _angle(rotation[:, 1], target_axes[:, 1]),
            "closing_axis_vs_target_pca_axis_2": _angle(rotation[:, 1], target_axes[:, 2]),
            "approach_vs_local_surface_normal": _angle(rotation[:, 0], surface_normal),
            "roll_consistency": _angle(rotation[:, 2], target_axes[:, 2]),
            "left_finger_occupancy": float(obstacle_masks["left"].sum() / total_scene),
            "right_finger_occupancy": float(obstacle_masks["right"].sum() / total_scene),
            "palm_occupancy": float(obstacle_masks["palm"].sum() / total_scene),
            "closing_volume_obstacle_fraction": float(obstacle_masks["closing"].sum() / total_scene),
            "approach_swept_volume_occupancy": float(obstacle_masks["approach"].sum() / total_scene),
            "table_clearance_m": float(np.asarray(candidate.translation_table_m)[2] - candidate.height_m / 2),
            "nearest_obstacle_distance_m": nearest_obstacle,
            "collision_proxy_flag": float(any(obstacle_masks[name].any() for name in ("left", "right", "palm"))),
            "local_point_density": float(len(local_points) / ((4 / 3) * np.pi * 0.05**3)),
            "target_point_density": float(len(target) / max(np.prod(np.maximum(target_extent, 1e-4)), EPS)),
            "depth_variance": float(np.var(local_points[:, 2])) if len(local_points) else np.nan,
            "normal_consistency": float(1.0 - _safe_ratio(local_eigenvalues[-1], local_eigenvalues.sum())) if np.isfinite(local_eigenvalues).all() else np.nan,
            "surface_curvature": _safe_ratio(local_eigenvalues[-1], local_eigenvalues.sum()),
            "contact_region_planarity": _safe_ratio(contact_eigenvalues[-2] - contact_eigenvalues[-1], contact_eigenvalues[0]),
            "left_right_contact_balance": 1.0 - _safe_ratio(abs(left_support - right_support), left_support + right_support + EPS),
            "valid_depth_fraction_near_grasp": float(np.mean(depth[max(0, row-5):row+6, max(0, column-5):column+6] > 0)) if inside_image else np.nan,
            **mask_features,
        }
        records.append(record)
    expected = tuple(spec.name for spec in default_feature_schema())
    frame = pd.DataFrame(records, columns=expected)
    if tuple(frame.columns) != expected or len(frame) != len(values):
        raise AssertionError("feature extraction violated the v1 schema or candidate membership")
    return frame


__all__ = ["RuntimeObservation", "extract_candidate_features"]
