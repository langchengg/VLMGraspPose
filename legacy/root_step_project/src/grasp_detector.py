"""
src/grasp_detector.py — Open3D-based RGB-D Geometric Grasp Sampler
===================================================================

This module implements the project's grasp-proposal stage:

Text + RGB-D -> target point cloud -> geometric grasp candidates.

All grasp poses are in **camera frame**.
"""

import abc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ═════════════════════════════════════════════════════════════════════
#  Grasp Candidate dataclass
# ═════════════════════════════════════════════════════════════════════

@dataclass
class GraspCandidate:
    candidate_id: int
    position: List[float]        # [x, y, z] camera frame
    rotation: List[float]        # flattened 3×3 rotation [r11..r33]
    width: float                 # gripper opening (metres)
    detector_score: float        # raw quality score 0–1
    source: str = "geometric"
    approach_vector: Optional[List[float]] = None
    closing_direction: Optional[List[float]] = None
    grasp_type: str = "normal_based"

    def __post_init__(self):
        R = np.array(self.rotation, dtype=np.float32).reshape(3, 3)
        if self.approach_vector is None:
            self.approach_vector = R[:, 0].astype(float).tolist()
        if self.closing_direction is None:
            self.closing_direction = R[:, 1].astype(float).tolist()

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return 3×3 rotation matrix."""
        return np.array(self.rotation).reshape(3, 3)


# ═════════════════════════════════════════════════════════════════════
#  Base class
# ═════════════════════════════════════════════════════════════════════

class GraspDetectorBase(abc.ABC):
    @abc.abstractmethod
    def detect(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray] = None,
        top_k: int = config.GRASP_TOP_K,
    ) -> List[GraspCandidate]:
        """Generate grasp candidates from the target point cloud."""
        ...


# ═════════════════════════════════════════════════════════════════════
#  RGB-D Geometric Sampler (local default)
# ═════════════════════════════════════════════════════════════════════

class RGBDGeometricGraspSampler(GraspDetectorBase):
    """Generate 6-DoF candidates from RGB-D point-cloud geometry.

    This sampler is a local, deterministic proposal generator for machines
    grasp centers from the target RGB-D point cloud, estimates local surface
    geometry with Open3D/PCA, derives an approach axis and gripper opening from
    the local patch, and scores candidates by density, planarity, viewing
    consistency, and feasible gripper width.
    """

    def __init__(
        self,
        top_k: int = config.GRASP_TOP_K,
        num_center_samples: int = config.GEOMETRIC_NUM_CENTER_SAMPLES,
        min_width: float = config.GRASP_MIN_WIDTH,
        max_width: float = config.GRASP_MAX_WIDTH,
        local_radius: float = config.GEOMETRIC_LOCAL_RADIUS,
        min_neighbors: int = config.GEOMETRIC_MIN_NEIGHBORS,
        max_points_for_sampling: int = config.GEOMETRIC_MAX_POINTS_FOR_SAMPLING,
    ):
        self.top_k = top_k
        self.num_center_samples = num_center_samples
        self.min_width = min_width
        self.max_width = max_width
        self.local_radius = local_radius
        self.min_neighbors = min_neighbors
        self.max_points_for_sampling = max_points_for_sampling

    def _downsample_for_sampling(self, point_cloud: np.ndarray) -> np.ndarray:
        """Bound the point count before local neighborhood search."""
        if len(point_cloud) <= self.max_points_for_sampling:
            return point_cloud

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float64))
            voxel_size = max(config.VOXEL_SIZE, 1e-4)
            down = pcd.voxel_down_sample(voxel_size=voxel_size)
            down_points = np.asarray(down.points, dtype=np.float32)
            if 0 < len(down_points) <= self.max_points_for_sampling:
                return down_points
            if len(down_points) > self.max_points_for_sampling:
                point_cloud = down_points
        except Exception:
            pass

        rng = np.random.RandomState(42)
        idx = rng.choice(
            len(point_cloud),
            size=self.max_points_for_sampling,
            replace=False,
        )
        return point_cloud[np.sort(idx)]

    def _estimate_normals(self, points: np.ndarray) -> np.ndarray:
        """Estimate normals with Open3D when available, otherwise PCA fallback."""
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=config.NORMAL_RADIUS,
                    max_nn=config.NORMAL_MAX_NN,
                )
            )
            normals = np.asarray(pcd.normals, dtype=np.float32)
            if normals.shape == points.shape:
                return _orient_normals_towards_camera(points, normals)
        except Exception:
            pass

        from src.point_cloud import estimate_normals_pca

        normals = estimate_normals_pca(points, k=min(config.NORMAL_MAX_NN, len(points)))
        return _orient_normals_towards_camera(points, normals)

    def _local_patch_axes(
        self,
        center: np.ndarray,
        normal: np.ndarray,
        points: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, float, float, float]]:
        """Return approach/closing axes and quality terms for one center."""
        offsets = points - center
        dists = np.linalg.norm(offsets, axis=1)
        local = offsets[dists <= self.local_radius]

        if len(local) < self.min_neighbors:
            nearest_count = min(max(self.min_neighbors, 8), len(points))
            if nearest_count < 3:
                return None
            nearest_idx = np.argpartition(dists, nearest_count - 1)[:nearest_count]
            local = offsets[nearest_idx]

        cov = np.cov(local.T)
        if not np.all(np.isfinite(cov)):
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]

        patch_normal = eigvecs[:, 0]
        if np.linalg.norm(normal) > 1e-6 and np.dot(patch_normal, normal) < 0:
            patch_normal = -patch_normal
        patch_normal = patch_normal / (np.linalg.norm(patch_normal) + 1e-8)

        closing_axis = eigvecs[:, -1]
        closing_axis = closing_axis - np.dot(closing_axis, patch_normal) * patch_normal
        if np.linalg.norm(closing_axis) < 1e-6:
            closing_axis = eigvecs[:, 1]
        closing_axis = closing_axis / (np.linalg.norm(closing_axis) + 1e-8)

        approach = -patch_normal
        span = np.abs(local @ closing_axis)
        width = float(np.clip(2.0 * np.percentile(span, 90) + 0.01, self.min_width, self.max_width))

        eig_sum = float(eigvals.sum() + 1e-8)
        planarity = float(np.clip(1.0 - eigvals[0] / eig_sum, 0.0, 1.0))
        density = float(np.clip(len(local) / max(self.min_neighbors * 3, 1), 0.0, 1.0))

        return approach, closing_axis, width, planarity, density

    def detect(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray] = None,
        top_k: int = None,
    ) -> List[GraspCandidate]:
        if top_k is None:
            top_k = self.top_k

        if len(point_cloud) < 10:
            return []

        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        point_cloud = point_cloud[np.all(np.isfinite(point_cloud), axis=1)]
        if len(point_cloud) < 10:
            return []

        point_cloud = self._downsample_for_sampling(point_cloud)

        normals = self._estimate_normals(point_cloud)

        N = len(point_cloud)
        num_samples = min(self.num_center_samples, N)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, size=num_samples, replace=False)

        candidates = []
        for i in sample_idx:
            center = point_cloud[i]
            normal = normals[i]

            if np.linalg.norm(normal) < 1e-6:
                continue

            patch = self._local_patch_axes(center, normal, point_cloud)
            if patch is None:
                continue

            approach, closing_axis, width, planarity, density = patch
            R = _axes_to_rotation(approach, closing_axis)

            view_dir = -center / (np.linalg.norm(center) + 1e-8)
            view_score = float(np.clip(np.dot(-approach, view_dir), 0.0, 1.0))
            width_mid = 0.5 * (self.min_width + self.max_width)
            width_range = max(self.max_width - self.min_width, 1e-6)
            width_score = float(np.clip(1.0 - abs(width - width_mid) / width_range, 0.0, 1.0))
            quality = float(np.clip(
                0.35 * density + 0.30 * planarity + 0.20 * view_score + 0.15 * width_score,
                0.0,
                1.0,
            ))

            candidates.append(GraspCandidate(
                candidate_id=len(candidates),
                position=center.tolist(),
                rotation=R.flatten().tolist(),
                width=width,
                detector_score=quality,
                source="geometric",
                approach_vector=approach.astype(float).tolist(),
                closing_direction=closing_axis.astype(float).tolist(),
                grasp_type="normal_based",
            ))

        candidates = _dedup_candidates(candidates, min_dist=0.005)
        candidates.sort(key=lambda c: c.detector_score, reverse=True)
        candidates = candidates[:top_k]

        for i, c in enumerate(candidates):
            c.candidate_id = i

        return candidates


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def _orient_normals_towards_camera(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Flip normals so they point approximately toward the RGB-D camera."""
    oriented = np.asarray(normals, dtype=np.float32).copy()
    view_dirs = -np.asarray(points, dtype=np.float32)
    dots = np.sum(oriented * view_dirs, axis=1)
    oriented[dots < 0] *= -1.0
    norms = np.linalg.norm(oriented, axis=1, keepdims=True)
    return oriented / (norms + 1e-8)


def _axes_to_rotation(approach: np.ndarray, closing_axis: np.ndarray) -> np.ndarray:
    """Build an orthonormal grasp frame from approach and closing axes."""
    ax = approach / (np.linalg.norm(approach) + 1e-8)

    ay = closing_axis - np.dot(closing_axis, ax) * ax
    if np.linalg.norm(ay) < 1e-6:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(fallback, ax)) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        ay = fallback - np.dot(fallback, ax) * ax
    ay = ay / (np.linalg.norm(ay) + 1e-8)

    az = np.cross(ax, ay)
    az = az / (np.linalg.norm(az) + 1e-8)
    ay = np.cross(az, ax)
    ay = ay / (np.linalg.norm(ay) + 1e-8)

    return np.stack([ax, ay, az], axis=1)


def _approach_to_rotation(approach: np.ndarray) -> np.ndarray:
    """Convert approach vector to 3×3 rotation matrix."""
    ax = approach / (np.linalg.norm(approach) + 1e-8)

    up = np.array([0.0, 0.0, -1.0])
    if abs(np.dot(ax, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])

    az = np.cross(ax, up)
    az = az / (np.linalg.norm(az) + 1e-8)
    ay = np.cross(az, ax)
    ay = ay / (np.linalg.norm(ay) + 1e-8)

    return np.stack([ax, ay, az], axis=1)  # 3×3


def _dedup_candidates(
    candidates: List[GraspCandidate],
    min_dist: float = 0.005,
) -> List[GraspCandidate]:
    """Remove near-duplicate candidates."""
    if not candidates:
        return candidates

    kept = [candidates[0]]
    for c in candidates[1:]:
        pos = np.array(c.position)
        too_close = any(
            np.linalg.norm(pos - np.array(k.position)) < min_dist
            for k in kept
        )
        if not too_close:
            kept.append(c)
    return kept
