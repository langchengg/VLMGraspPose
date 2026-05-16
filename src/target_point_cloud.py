"""
Target point-cloud extraction and Open3D geometry processing.

This module is the contract between target grounding and grasp sampling:

Input:
  - scene point cloud from RGB-D backprojection
  - pixel coordinates for each scene point
  - target bbox or target mask from grounding

Output:
  - cleaned target point cloud
  - target center, AABB, OBB, normals, and optional table plane
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from src.point_cloud import (
    crop_point_cloud_by_bbox,
    estimate_normals_pca,
    voxel_downsample,
)


@dataclass
class PointCloudRepresentation:
    """Geometry bundle consumed by the sampler and feature extractor."""

    scene_points: np.ndarray
    scene_pixel_coords: np.ndarray
    target_points: np.ndarray
    target_pixel_coords: np.ndarray
    clean_target_points: np.ndarray
    target_center_3d: np.ndarray
    target_aabb: dict
    target_obb: dict
    surface_normals: np.ndarray
    table_plane: Optional[list]


def build_point_cloud_representation(
    scene_points: np.ndarray,
    scene_pixel_coords: np.ndarray,
    target_bbox: list,
    target_mask: Optional[np.ndarray] = None,
) -> PointCloudRepresentation:
    """Extract and process the target point cloud from scene RGB-D geometry."""
    scene_points = np.asarray(scene_points, dtype=np.float32)
    scene_pixel_coords = np.asarray(scene_pixel_coords, dtype=np.int32)

    if target_mask is not None and np.any(target_mask):
        target_points, target_pixel_coords = crop_points_and_pixels_by_binary_mask(
            scene_points,
            scene_pixel_coords,
            target_mask,
        )
    else:
        target_points, target_pixel_coords = crop_point_cloud_by_bbox(
            scene_points,
            scene_pixel_coords,
            target_bbox,
        )

    target_points = _finite_points(target_points)
    clean_target_points = clean_target_point_cloud(target_points)
    target_center_3d = compute_center(clean_target_points)
    surface_normals = estimate_surface_normals(clean_target_points)
    target_aabb = compute_aabb(clean_target_points)
    target_obb = compute_obb(clean_target_points)
    table_plane = estimate_table_plane(scene_points)

    return PointCloudRepresentation(
        scene_points=scene_points,
        scene_pixel_coords=scene_pixel_coords,
        target_points=target_points,
        target_pixel_coords=target_pixel_coords,
        clean_target_points=clean_target_points,
        target_center_3d=target_center_3d,
        target_aabb=target_aabb,
        target_obb=target_obb,
        surface_normals=surface_normals,
        table_plane=table_plane,
    )


def crop_points_and_pixels_by_binary_mask(
    points: np.ndarray,
    pixel_coords: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return points and pixels whose image coordinates fall inside mask."""
    if len(points) == 0:
        return points, pixel_coords

    u, v = pixel_coords[:, 0], pixel_coords[:, 1]
    H, W = mask.shape[:2]
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = u[valid]
    v_valid = v[valid]
    on_mask = mask[v_valid, u_valid].astype(bool)
    idx = np.where(valid)[0][on_mask]
    return points[idx], pixel_coords[idx]


def clean_target_point_cloud(points: np.ndarray) -> np.ndarray:
    """Denoise and downsample target points with Open3D when available."""
    points = _finite_points(points)
    if len(points) == 0:
        return points

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel_size=max(config.VOXEL_SIZE, 1e-4))
        if len(pcd.points) >= 20:
            pcd, keep = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        cleaned = np.asarray(pcd.points, dtype=np.float32)
        return cleaned if len(cleaned) else points
    except Exception:
        return voxel_downsample(points, config.VOXEL_SIZE).astype(np.float32)


def estimate_surface_normals(points: np.ndarray) -> np.ndarray:
    """Estimate target surface normals."""
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)

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

    normals = estimate_normals_pca(points, k=min(config.NORMAL_MAX_NN, len(points)))
    return _orient_normals_towards_camera(points, normals)


def estimate_table_plane(scene_points: np.ndarray) -> Optional[list]:
    """Estimate a dominant scene plane using Open3D RANSAC."""
    scene_points = _finite_points(scene_points)
    if len(scene_points) < 50:
        return None

    if len(scene_points) > config.TABLE_PLANE_MAX_POINTS:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(scene_points), config.TABLE_PLANE_MAX_POINTS, replace=False)
        scene_points = scene_points[idx]

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene_points.astype(np.float64))
        plane, _ = pcd.segment_plane(
            distance_threshold=config.TABLE_PLANE_DISTANCE_THRESH,
            ransac_n=3,
            num_iterations=100,
        )
        return [float(x) for x in plane]
    except Exception:
        return None


def compute_center(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32)
    return points.mean(axis=0).astype(np.float32)


def compute_aabb(points: np.ndarray) -> dict:
    if len(points) == 0:
        zeros = [0.0, 0.0, 0.0]
        return {"min": zeros, "max": zeros, "extent": zeros}
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    return {
        "min": p_min.astype(float).tolist(),
        "max": p_max.astype(float).tolist(),
        "extent": (p_max - p_min).astype(float).tolist(),
    }


def compute_obb(points: np.ndarray) -> dict:
    """Fit an oriented bounding box with Open3D or PCA fallback."""
    if len(points) < 3:
        return {
            "center": compute_center(points).astype(float).tolist(),
            "rotation": np.eye(3, dtype=float).flatten().tolist(),
            "extent": [0.0, 0.0, 0.0],
        }

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        obb = pcd.get_oriented_bounding_box()
        return {
            "center": np.asarray(obb.center, dtype=float).tolist(),
            "rotation": np.asarray(obb.R, dtype=float).flatten().tolist(),
            "extent": np.asarray(obb.extent, dtype=float).tolist(),
        }
    except Exception:
        center = compute_center(points)
        centered = points - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order]
        local = centered @ R
        extent = local.max(axis=0) - local.min(axis=0)
        return {
            "center": center.astype(float).tolist(),
            "rotation": R.astype(float).flatten().tolist(),
            "extent": extent.astype(float).tolist(),
        }


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    return points[np.all(np.isfinite(points), axis=1)]


def _orient_normals_towards_camera(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    oriented = np.asarray(normals, dtype=np.float32).copy()
    view_dirs = -np.asarray(points, dtype=np.float32)
    dots = np.sum(oriented * view_dirs, axis=1)
    oriented[dots < 0] *= -1.0
    norms = np.linalg.norm(oriented, axis=1, keepdims=True)
    return oriented / (norms + 1e-8)
