"""
scripts/step05_depth_to_pcd.py — Convert depth images to point clouds
=======================================================================
Step 5: Back-project depth images to 3D point clouds using camera
intrinsics. Save per-view NPZ files.

Usage:
    python scripts/step05_depth_to_pcd.py
    python scripts/step05_depth_to_pcd.py --splits train val
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import (
    load_depth, load_rgb, load_camera_intrinsics, get_factor_depth,
)
from src.point_cloud import backproject_depth, add_colors


def depth_to_pointclouds(splits: list = None):
    """Convert depth images to point clouds for all indexed views."""
    if splits is None:
        splits = config.ALL_SPLITS

    for split in splits:
        views_path = config.SPLITS_DIR / f"{split}_views.jsonl"
        if not views_path.exists():
            print(f"  [SKIP] {views_path} not found (run step01 first)")
            continue

        config.POINTCLOUDS_DIR.mkdir(parents=True, exist_ok=True)
        total = 0

        with open(views_path) as fin:
            lines = fin.readlines()

        for line in tqdm(lines, desc=f"PointClouds [{split}]"):
            view = json.loads(line)
            scene_id = view["scene_id"]
            camera = view["camera"]
            frame_id = view["frame_id"]
            sample_id = view["sample_id"]

            out_path = config.POINTCLOUDS_DIR / f"{sample_id}.npz"
            if out_path.exists():
                total += 1
                continue

            scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"

            try:
                factor = get_factor_depth(scene_dir, camera)
                depth = load_depth(scene_dir, frame_id, camera, factor)
                rgb = load_rgb(scene_dir, frame_id, camera)
                K = load_camera_intrinsics(scene_dir, camera)
            except Exception as e:
                continue

            points, pixel_coords = backproject_depth(depth, K)

            if len(points) == 0:
                continue

            colors = add_colors(rgb, pixel_coords)

            # Create valid_mask (HxW bool)
            H, W = depth.shape
            valid_mask = depth > 0

            np.savez_compressed(
                out_path,
                points=points.astype(np.float32),
                colors=colors.astype(np.float32),
                pixel_coords=pixel_coords.astype(np.int32),
                valid_mask=valid_mask,
            )
            total += 1

        print(f"  [{split}] {total} point clouds → {config.POINTCLOUDS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Convert depth to point clouds"
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Splits to process (default: all)"
    )
    args = parser.parse_args()

    depth_to_pointclouds(splits=args.splits)


if __name__ == "__main__":
    main()
