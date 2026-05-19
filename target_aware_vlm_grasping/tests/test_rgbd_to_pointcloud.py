from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pointcloud.rgbd_to_pointcloud import rgbd_to_pointcloud


def test_rgbd_to_pointcloud_generates_non_empty_cloud() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    depth = np.ones((8, 8), dtype=np.float32)
    intrinsics = np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])
    pcd = rgbd_to_pointcloud(rgb, depth, intrinsics, depth_scale=1.0, depth_trunc=2.0)
    assert len(pcd.points) == 64
    assert len(pcd.colors) == 64
