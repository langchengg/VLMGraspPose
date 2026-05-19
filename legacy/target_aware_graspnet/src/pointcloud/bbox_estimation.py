from __future__ import annotations

import numpy as np
import open3d as o3d


def compute_aabb(pcd: o3d.geometry.PointCloud):
    if len(pcd.points) == 0:
        return None
    return pcd.get_axis_aligned_bounding_box()


def compute_obb(pcd: o3d.geometry.PointCloud):
    if len(pcd.points) < 3:
        return None
    return pcd.get_oriented_bounding_box()


def compute_center(pcd: o3d.geometry.PointCloud) -> np.ndarray | None:
    if len(pcd.points) == 0:
        return None
    return np.asarray(pcd.points).mean(axis=0)


def get_obb_axes(obb) -> np.ndarray:
    if obb is None:
        return np.eye(3)
    return np.asarray(obb.R, dtype=float)
