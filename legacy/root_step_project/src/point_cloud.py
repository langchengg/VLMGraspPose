"""
src/point_cloud.py — Depth ↔ 3D conversions, cropping, normals
================================================================
Migrated from data/point_cloud.py.

All outputs are in **camera frame** by default.
"""

import numpy as np
from typing import Optional, Tuple


# ═════════════════════════════════════════════════════════════════════
#  Depth → Point Cloud
# ═════════════════════════════════════════════════════════════════════

def backproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project depth map into 3D point cloud in camera frame.

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

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    if mask is None:
        valid = depth > 0
    else:
        valid = (depth > 0) & mask

    z = depth[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    pixel_coords = np.stack(
        [u_valid.astype(int), v_valid.astype(int)], axis=-1
    )
    return points, pixel_coords


# ═════════════════════════════════════════════════════════════════════
#  Project 3D → Image
# ═════════════════════════════════════════════════════════════════════

def project_to_image(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project (N, 3) points in camera frame onto image plane.

    Returns (N, 2) float array of (u, v) pixel coordinates.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z_safe = np.clip(z, 1e-6, None)

    u = x * fx / z_safe + cx
    v = y * fy / z_safe + cy
    return np.stack([u, v], axis=-1)


# ═════════════════════════════════════════════════════════════════════
#  Add RGB colours to point cloud
# ═════════════════════════════════════════════════════════════════════

def add_colors(
    rgb: np.ndarray,
    pixel_coords: np.ndarray,
) -> np.ndarray:
    """Attach RGB values to points using their pixel coordinates.

    Parameters
    ----------
    rgb : (H, W, 3) uint8
    pixel_coords : (N, 2) int — (u, v)

    Returns
    -------
    colors : (N, 3) float32 in [0, 1]
    """
    u = pixel_coords[:, 0]
    v = pixel_coords[:, 1]
    H, W = rgb.shape[:2]
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    return rgb[v, u].astype(np.float32) / 255.0


# ═════════════════════════════════════════════════════════════════════
#  Crop point cloud
# ═════════════════════════════════════════════════════════════════════

def crop_point_cloud_by_bbox(
    points: np.ndarray,
    pixel_coords: np.ndarray,
    bbox: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep points whose image projection falls inside bbox [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox
    u, v = pixel_coords[:, 0], pixel_coords[:, 1]
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return points[inside], pixel_coords[inside]


def crop_point_cloud_by_mask(
    points: np.ndarray,
    pixel_coords: np.ndarray,
    label: np.ndarray,
    mask_val: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep points whose pixel falls on mask_val in label mask."""
    u, v = pixel_coords[:, 0], pixel_coords[:, 1]
    H, W = label.shape[:2]
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = u[valid]
    v_valid = v[valid]
    on_instance = label[v_valid, u_valid] == mask_val
    idx = np.where(valid)[0][on_instance]
    return points[idx], pixel_coords[idx]


def crop_points_by_binary_mask(
    points: np.ndarray,
    pixel_coords: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Keep points whose pixel falls on a True value in a binary HxW mask.

    Unlike crop_point_cloud_by_mask (which takes a label image + mask_val),
    this works with arbitrary boolean masks (e.g. from Florence-2 segmentation).
    """
    u, v = pixel_coords[:, 0], pixel_coords[:, 1]
    H, W = mask.shape[:2]
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = u[valid]
    v_valid = v[valid]
    on_mask = mask[v_valid, u_valid]
    idx = np.where(valid)[0][on_mask]
    return points[idx]


# ═════════════════════════════════════════════════════════════════════
#  Down-sampling
# ═════════════════════════════════════════════════════════════════════

def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Voxel grid down-sampling (no Open3D dependency)."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


# ═════════════════════════════════════════════════════════════════════
#  Surface normals
# ═════════════════════════════════════════════════════════════════════

def estimate_normals_pca(
    points: np.ndarray,
    k: int = 30,
    orient_towards_camera: bool = True,
) -> np.ndarray:
    """Estimate surface normals via PCA on k-nearest neighbours.

    Returns (N, 3) unit normals.
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
            normal = eigvecs[:, 0]
        except np.linalg.LinAlgError:
            normal = np.array([0.0, 0.0, -1.0])

        if orient_towards_camera:
            if np.dot(normal, -points[i]) < 0:
                normal = -normal

        normals[i] = normal / (np.linalg.norm(normal) + 1e-8)

    return normals


def compute_target_center(points: np.ndarray) -> np.ndarray:
    """Compute 3D centroid of a point set."""
    if len(points) == 0:
        return np.zeros(3)
    return points.mean(axis=0)
