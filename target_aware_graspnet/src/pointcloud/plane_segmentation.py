from __future__ import annotations

import numpy as np
import open3d as o3d


def segment_table_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
) -> tuple[np.ndarray | None, list[int]]:
    if pcd is None or len(pcd.points) < ransac_n:
        return None, []
    plane, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return np.asarray(plane, dtype=float), inliers


def remove_plane_points(
    pcd: o3d.geometry.PointCloud,
    plane_model: np.ndarray | None,
    distance_threshold: float,
) -> o3d.geometry.PointCloud:
    if plane_model is None or len(pcd.points) == 0:
        return pcd
    pts = np.asarray(pcd.points)
    dist = np.abs(pts @ plane_model[:3] + plane_model[3]) / max(np.linalg.norm(plane_model[:3]), 1e-8)
    keep = np.where(dist > distance_threshold)[0]
    return pcd.select_by_index(keep)
