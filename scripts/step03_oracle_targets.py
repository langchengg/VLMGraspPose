"""
scripts/step03_oracle_targets.py — Build oracle target annotations
====================================================================
Step 3: Compute GT bounding boxes and masks from GraspNet label images.
Provides the oracle-grounding track for upper-bound experiments.

Usage:
    python scripts/step03_oracle_targets.py
    python scripts/step03_oracle_targets.py --splits train val
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import load_label, bbox_from_mask, mask_pixel_count


def build_oracle_targets(splits: list = None):
    """Build GT target annotations from label images."""
    if splits is None:
        splits = config.ALL_SPLITS

    for split in splits:
        queries_path = config.QUERIES_DIR / f"{split}_queries.jsonl"
        if not queries_path.exists():
            print(f"  [SKIP] {queries_path} not found (run step02 first)")
            continue

        out_path = config.ORACLE_TARGETS_DIR / f"{split}_oracle.jsonl"
        config.ORACLE_TARGETS_DIR.mkdir(parents=True, exist_ok=True)

        total = 0
        skipped = 0

        with open(queries_path) as fin, open(out_path, "w") as fout:
            for line in tqdm(fin, desc=f"Oracle [{split}]"):
                query = json.loads(line)
                scene_id = query["scene_id"]
                camera = query["camera"]
                frame_id = query["frame_id"]
                obj_id = query["target_object_id"]
                mask_val = obj_id + 1  # GraspNet convention

                scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"

                try:
                    label = load_label(scene_dir, frame_id, camera)
                except Exception:
                    skipped += 1
                    continue

                gt_bbox = bbox_from_mask(label, mask_val)
                if gt_bbox is None:
                    skipped += 1
                    continue

                # Minimum size filter (20×20 pixels)
                bw = gt_bbox[2] - gt_bbox[0]
                bh = gt_bbox[3] - gt_bbox[1]
                if bw < 20 or bh < 20:
                    skipped += 1
                    continue

                n_pixels = mask_pixel_count(label, mask_val)

                record = {
                    "sample_id": query["sample_id"],
                    "target_object_id": obj_id,
                    "gt_bbox": gt_bbox,
                    "gt_mask_val": int(mask_val),
                    "gt_visible_pixels": n_pixels,
                    "label_path": str(
                        scene_dir / camera / "label" / f"{frame_id:04d}.png"
                    ),
                    "split": split,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

        print(f"  [{split}] {total} oracle targets → {out_path}")
        if skipped:
            print(f"           ({skipped} skipped: not visible or too small)")


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Build oracle target annotations"
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Splits to process (default: all)"
    )
    args = parser.parse_args()

    build_oracle_targets(splits=args.splits)


if __name__ == "__main__":
    main()
