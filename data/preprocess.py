"""
data/preprocess.py — Convert raw GraspNet scenes to standardised JSONL
======================================================================
Usage:
    python -m data.preprocess --split test_seen
    python -m data.preprocess --split train       (when data arrives)
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.dataset import discover_scenes, load_scene, generate_samples


def preprocess_split(split_name: str, view_stride: int = config.VIEW_STRIDE):
    """Process one split and write processed/<split>.jsonl."""
    data_dir = config.DATA_DIRS.get(split_name)
    if data_dir is None or not data_dir.exists():
        print(f"[SKIP] Data directory for '{split_name}' not found: {data_dir}")
        return

    scenes = discover_scenes(data_dir)
    print(f"[INFO] Found {len(scenes)} scenes in {data_dir}")

    output_path = config.PROCESSED_DIR / f"{split_name}.jsonl"
    total_samples = 0

    with open(output_path, "w") as fout:
        for scene_dir in tqdm(scenes, desc=f"Preprocessing {split_name}"):
            try:
                scene_meta = load_scene(scene_dir)
            except Exception as e:
                print(f"  [WARN] Skipping {scene_dir.name}: {e}")
                continue

            samples = generate_samples(scene_meta, view_stride=view_stride)

            for sample in samples:
                record = asdict(sample)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_samples += 1

    print(f"[DONE] Wrote {total_samples} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess GraspNet data")
    parser.add_argument("--split", type=str, default="test_seen",
                        help="Split name (must match a key in config.DATA_DIRS)")
    parser.add_argument("--view-stride", type=int, default=config.VIEW_STRIDE,
                        help="Use every N-th view")
    args = parser.parse_args()

    preprocess_split(args.split, view_stride=args.view_stride)


if __name__ == "__main__":
    main()
