from __future__ import annotations

import numpy as np
import open3d as o3d


def filter_invalid_depth(depth: np.ndarray, max_depth: float | None = None) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth < max_depth
    return valid


def extract_target_pointcloud_from_mask(
    scene_points: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
) -> o3d.geometry.PointCloud:
    H, W = depth.shape
    ys, xs = np.where(mask.astype(bool) & filter_invalid_depth(depth))
    if len(xs) == 0:
        return o3d.geometry.PointCloud()
    z = depth[ys, xs]
    x = (xs.astype(float) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys.astype(float) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    pts = np.stack([x, y, z], axis=1)
    colors = rgb[ys, xs].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def crop_pointcloud_by_bbox(
    rgb: np.ndarray,
    depth: np.ndarray,
    bbox: list[int],
    intrinsics: np.ndarray,
) -> o3d.geometry.PointCloud:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = depth.shape
    x1, x2 = np.clip([x1, x2], 0, W - 1)
    y1, y2 = np.clip([y1, y2], 0, H - 1)
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2 + 1, x1:x2 + 1] = True
    return extract_target_pointcloud_from_mask(np.empty((0, 3)), rgb, depth, mask, intrinsics)


def pointcloud_to_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points, dtype=float)
