from __future__ import annotations

import open3d as o3d


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if pcd is None or len(pcd.points) == 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=float(voxel_size))


def remove_statistical_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    if pcd is None or len(pcd.points) < nb_neighbors:
        return pcd
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return clean


def remove_radius_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_points: int,
    radius: float,
) -> o3d.geometry.PointCloud:
    if pcd is None or len(pcd.points) < nb_points:
        return pcd
    clean, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return clean


def preprocess_target_pcd(pcd: o3d.geometry.PointCloud, config: dict) -> o3d.geometry.PointCloud:
    pcd = voxel_downsample(pcd, config.get("voxel_size", 0.005))
    pcd = remove_statistical_outliers(
        pcd,
        config.get("outlier_nb_neighbors", 20),
        config.get("outlier_std_ratio", 2.0),
    )
    return pcd
