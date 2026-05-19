"""
scripts/step01_build_index.py — Build view-level dataset index
================================================================
Step 1: Enumerate every (scene, camera, frame) view, attach split label,
record visible object IDs.

Usage:
    python scripts/step01_build_index.py
    python scripts/step01_build_index.py --camera realsense --stride 16
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import (
    discover_scenes, scene_id_from_path, split_for_scene,
    load_object_id_list, count_views, visible_object_ids,
)


def build_index(
    camera: str = config.CAMERA_TYPE,
    stride: int = config.VIEW_STRIDE,
):
    """Build JSONL view indexes for all splits."""
    scenes = discover_scenes()
    if not scenes:
        print(f"[ERROR] No scenes found in {config.SCENES_DIR}")
        print(f"        Run: python scripts/download_data.py --all")
        return

    print(f"Found {len(scenes)} scenes in {config.SCENES_DIR}")
    print(f"Camera: {camera}  |  Stride: {stride}")

    # Group by split
    split_records = {s: [] for s in config.ALL_SPLITS}

    for scene_dir in tqdm(scenes, desc="Building index"):
        scene_id = scene_id_from_path(scene_dir)

        try:
            split = split_for_scene(scene_id)
        except ValueError:
            continue

        n_views = count_views(scene_dir, camera)
        obj_ids = load_object_id_list(scene_dir)

        for frame_id in range(0, n_views, stride):
            try:
                vis_ids = visible_object_ids(scene_dir, frame_id, camera)
            except Exception:
                vis_ids = obj_ids  # fallback

            record = {
                "sample_id": f"scene_{scene_id:04d}_{camera}_{frame_id:04d}",
                "scene_id": scene_id,
                "camera": camera,
                "frame_id": frame_id,
                "rgb_path": str(scene_dir / camera / "rgb" / f"{frame_id:04d}.png"),
                "depth_path": str(scene_dir / camera / "depth" / f"{frame_id:04d}.png"),
                "label_path": str(scene_dir / camera / "label" / f"{frame_id:04d}.png"),
                "camK_path": str(scene_dir / camera / "camK.npy"),
                "camera_poses_path": str(scene_dir / camera / "camera_poses.npy"),
                "visible_object_ids": vis_ids,
                "split": split,
            }
            split_records[split].append(record)

    # Write per-split JSONL files
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split, records in split_records.items():
        if not records:
            continue
        out_path = config.SPLITS_DIR / f"{split}_views.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [{split}] {len(records)} views → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Build view-level dataset index"
    )
    parser.add_argument("--camera", type=str, default=config.CAMERA_TYPE)
    parser.add_argument("--stride", type=int, default=config.VIEW_STRIDE)
    args = parser.parse_args()

    build_index(camera=args.camera, stride=args.stride)


if __name__ == "__main__":
    main()
