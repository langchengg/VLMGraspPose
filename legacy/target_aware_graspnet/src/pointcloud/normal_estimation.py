from __future__ import annotations

import numpy as np
import open3d as o3d


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int,
) -> o3d.geometry.PointCloud:
    if pcd is None or len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd


def orient_normals_towards_camera(
    pcd: o3d.geometry.PointCloud,
    camera_location: np.ndarray = np.zeros(3),
) -> o3d.geometry.PointCloud:
    if pcd is None or len(pcd.points) == 0:
        return pcd
    pcd.orient_normals_towards_camera_location(camera_location.astype(float))
    return pcd
