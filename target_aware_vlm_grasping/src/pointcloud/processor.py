from __future__ import annotations

import numpy as np

from pointcloud.bbox_estimation import compute_aabb, compute_center, compute_obb
from pointcloud.normal_estimation import estimate_normals, orient_normals_towards_camera
from pointcloud.plane_segmentation import segment_table_plane
from pointcloud.preprocessing import preprocess_target_pcd
from pointcloud.rgbd_to_pointcloud import rgbd_to_pointcloud
from pointcloud.target_extraction import crop_pointcloud_by_bbox, extract_target_pointcloud_from_mask
from utils.data_types import PointCloudRepresentation, TargetRegion


class PointCloudProcessor:
    def __init__(self, config: dict):
        self.config = config

    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        target: TargetRegion,
    ) -> PointCloudRepresentation:
        scene_pcd = rgbd_to_pointcloud(
            rgb,
            depth,
            intrinsics,
            depth_scale=1.0,
            depth_trunc=self.config.get("depth_trunc", 2.0),
        )
        if target.mask is not None:
            target_pcd = extract_target_pointcloud_from_mask(
                np.asarray(scene_pcd.points), rgb, depth, target.mask, intrinsics
            )
        elif target.bbox is not None:
            target_pcd = crop_pointcloud_by_bbox(rgb, depth, target.bbox, intrinsics)
        else:
            raise ValueError("TargetRegion must provide mask or bbox.")

        plane, _ = segment_table_plane(
            scene_pcd,
            self.config.get("table_distance_threshold", 0.01),
            self.config.get("table_ransac_n", 3),
            self.config.get("table_num_iterations", 100),
        )
        clean_target = preprocess_target_pcd(target_pcd, self.config)
        clean_target = estimate_normals(
            clean_target,
            self.config.get("normal_radius", 0.02),
            self.config.get("normal_max_nn", 30),
        )
        clean_target = orient_normals_towards_camera(clean_target)
        normals = np.asarray(clean_target.normals) if len(clean_target.points) else None
        return PointCloudRepresentation(
            scene_pcd=scene_pcd,
            target_pcd=target_pcd,
            clean_target_pcd=clean_target,
            table_plane=plane,
            target_center_3d=compute_center(clean_target),
            target_aabb=compute_aabb(clean_target),
            target_obb=compute_obb(clean_target),
            surface_normals=normals,
        )
