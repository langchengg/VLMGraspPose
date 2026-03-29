"""
data/point_cloud.py — Depth ↔ 3D conversions, cropping, normals
================================================================
All outputs are in **camera frame** by default.
"""

import numpy as np
from typing import Optional, Tuple


# ── Depth → Point Cloud ─────────────────────────────────────────────

def backproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map into a 3D point cloud in camera frame.

    Parameters
    ----------
    depth : (H, W) float32 in metres
    intrinsics : (3, 3) camera intrinsic matrix
    mask : (H, W) bool, optional — only back-project where True

    Returns
    -------
    points : (N, 3) float32
    pixel_coords : (N, 2) int — (u, v) for each point
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    # Valid depth
    if mask is None:
        valid = depth > 0
    else:
        valid = (depth > 0) & mask

    z = depth[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)  # (N, 3)
    pixel_coords = np.stack([u_valid.astype(int), v_valid.astype(int)], axis=-1)

    return points, pixel_coords


# ── Project 3D → Image ──────────────────────────────────────────────

def project_to_image(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project (N, 3) points in camera frame onto the image plane.

    Returns (N, 2) float array of (u, v) pixel coordinates.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z_safe = np.clip(z, 1e-6, None)

    u = x * fx / z_safe + cx
    v = y * fy / z_safe + cy

    return np.stack([u, v], axis=-1)


# ── Crop Point Cloud ─────────────────────────────────────────────────

def crop_point_cloud_by_bbox(
    points: np.ndarray,
    pixel_coords: np.ndarray,
    bbox: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only points whose image projection falls inside *bbox*.

    bbox : [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    u, v = pixel_coords[:, 0], pixel_coords[:, 1]
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return points[inside], pixel_coords[inside]


def crop_point_cloud_by_mask(
    points: np.ndarray,
    pixel_coords: np.ndarray,
    label: np.ndarray,
    instance_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only points whose pixel falls on *instance_id* in label mask."""
    u, v = pixel_coords[:, 0], pixel_coords[:, 1]
    # Bounds check
    H, W = label.shape
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid, v_valid = u[valid], v[valid]

    on_instance = label[v_valid, u_valid] == instance_id
    # Map back to original indices
    idx = np.where(valid)[0][on_instance]
    return points[idx], pixel_coords[idx]


# ── Point Cloud Down-sampling ────────────────────────────────────────

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Simple voxel grid down-sampling (no Open3D dependency)."""
    if len(points) == 0:
        return points
    # Quantise
    keys = np.floor(points / voxel_size).astype(np.int64)
    # Unique voxels
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


# ── Surface Normal Estimation ────────────────────────────────────────

def estimate_normals_pca(
    points: np.ndarray,
    k: int = 30,
    orient_towards_camera: bool = True,
) -> np.ndarray:
    """Estimate surface normals via PCA on k-nearest neighbours.

    Returns (N, 3) unit normals.
    Falls back to [0,0,-1] if neighbourhood is degenerate.
    """
    from scipy.spatial import cKDTree

    N = len(points)
    normals = np.zeros_like(points)

    tree = cKDTree(points)
    k_actual = min(k, N)

    for i in range(N):
        _, idx = tree.query(points[i], k=k_actual)
        neighbours = points[idx]
        cov = np.cov(neighbours.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]  # smallest eigenvalue
        except np.linalg.LinAlgError:
            normal = np.array([0.0, 0.0, -1.0])

        # Orient towards camera (camera is at origin in camera frame)
        if orient_towards_camera:
            if np.dot(normal, -points[i]) < 0:
                normal = -normal

        normals[i] = normal / (np.linalg.norm(normal) + 1e-8)

    return normals


def compute_target_center(
    points: np.ndarray,
) -> np.ndarray:
    """Compute the 3D centroid of a target point cloud."""
    if len(points) == 0:
        return np.zeros(3)
    return points.mean(axis=0)
