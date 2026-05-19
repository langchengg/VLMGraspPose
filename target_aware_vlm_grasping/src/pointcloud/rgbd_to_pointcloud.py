from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from dataset.camera_loader import load_depth as _load_depth
from dataset.camera_loader import load_intrinsics, load_rgb as _load_rgb


def load_rgb(path: Path) -> np.ndarray:
    return _load_rgb(path)


def load_depth(path: Path, depth_scale: float = 1000.0) -> np.ndarray:
    return _load_depth(path, depth_scale)


def create_intrinsic(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def create_intrinsic_from_matrix(
    K: np.ndarray,
    width: int,
    height: int,
) -> o3d.camera.PinholeCameraIntrinsic:
    return create_intrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float = 1.0,
    depth_trunc: float = 2.0,
) -> o3d.geometry.PointCloud:
    color = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_img = o3d.geometry.Image((depth * depth_scale).astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth_img,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    H, W = depth.shape
    intrinsic = create_intrinsic_from_matrix(intrinsics, W, H)
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


def save_pointcloud(path: Path, pcd: o3d.geometry.PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def load_intrinsics_with_fallback(path: Path | None, fallback: dict) -> np.ndarray:
    return load_intrinsics(path, fallback)
