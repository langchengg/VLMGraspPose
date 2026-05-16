"""
scripts/step07_build_labels.py — Build target-aware training labels
=====================================================================
Step 7: For each candidate, determine the associated object using the
scene label image.  A candidate is positive if it is on the target
object AND has a sufficiently high detector score (>= 0.3 threshold).

NOTE: GraspNet collision_labels are loaded and passed through, but
cannot be directly indexed by detector candidate_id (they are indexed
by pre-defined grasp configurations: object × angle × depth).  Until
a grasp-matching step is implemented, the detector score is used as
a quality/collision proxy.  See src/label_builder.py for details.

Usage:
    python scripts/step07_build_labels.py --splits train val
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import load_label, load_grasp_candidates
from src.label_builder import associate_grasp_to_object


def _build_view_candidate_cache(
    candidates: list,
    scene_points: np.ndarray,
    scene_pixel_coords: np.ndarray,
    label: np.ndarray,
) -> list:
    """Compute view-dependent candidate attributes once per view."""
    cached = []
    for candidate in candidates:
        associated_val = associate_grasp_to_object(
            candidate, scene_points, scene_pixel_coords, label,
        )
        cached.append({
            "candidate_id": candidate.candidate_id,
            "associated_object_val": int(associated_val) if associated_val else -1,
            "is_collision_free": int(candidate.detector_score >= 0.3),
        })
    return cached


def build_labels(splits: list = None, detector: str = config.DEFAULT_DETECTOR):
    """Build training labels for target-conditioned candidates."""
    if splits is None:
        splits = config.TRAIN_SPLITS + config.VAL_SPLITS

    for split in splits:
        queries_path = config.QUERIES_DIR / f"{split}_queries.jsonl"
        oracle_path = config.ORACLE_TARGETS_DIR / f"{split}_oracle.jsonl"

        if not queries_path.exists() or not oracle_path.exists():
            print(f"  [SKIP] {split}: missing queries or oracle (run step02/03)")
            continue

        # Load oracle targets into a dict keyed by sample_id
        oracle_map = {}
        with open(oracle_path) as f:
            for line in f:
                rec = json.loads(line)
                oracle_map[rec["sample_id"]] = rec

        config.RANK_LABELS_DIR.mkdir(parents=True, exist_ok=True)
        all_records = []
        view_cache = {}

        with open(queries_path) as fin:
            lines = fin.readlines()

        for line in tqdm(lines, desc=f"Labels [{split}]"):
            query = json.loads(line)
            sample_id = query["sample_id"]

            if sample_id not in oracle_map:
                continue

            oracle = oracle_map[sample_id]
            view_sample_id = query["view_sample_id"]
            scene_id = query["scene_id"]
            camera = query["camera"]
            frame_id = query["frame_id"]
            target_mask_val = oracle["gt_mask_val"]
            ctx = view_cache.get(view_sample_id)
            if ctx is None:
                ctx = {"skip": True}

                pcd_path = config.POINTCLOUDS_DIR / f"{view_sample_id}.npz"
                if pcd_path.exists():
                    pcd_data = np.load(str(pcd_path))
                    scene_points = pcd_data["points"]
                    scene_pixel_coords = pcd_data["pixel_coords"]

                    scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"
                    try:
                        label = load_label(scene_dir, frame_id, camera)
                    except Exception:
                        label = None

                    if label is not None:
                        ctx = {
                            "skip": False,
                            "scene_points": scene_points,
                            "scene_pixel_coords": scene_pixel_coords,
                            "label": label,
                        }

                view_cache[view_sample_id] = ctx

            if ctx["skip"]:
                continue

            candidates = load_grasp_candidates(sample_id, detector)
            if not candidates:
                continue
            scene_points = ctx["scene_points"]
            scene_pixel_coords = ctx["scene_pixel_coords"]
            label = ctx["label"]

            base_labels = _build_view_candidate_cache(
                candidates,
                scene_points,
                scene_pixel_coords,
                label,
            )
            labels = []
            for base in base_labels:
                is_on_target = base["associated_object_val"] == int(target_mask_val)
                labels.append({
                    "candidate_id": base["candidate_id"],
                    "target_mask_val": int(target_mask_val),
                    "associated_object_val": base["associated_object_val"],
                    "is_collision_free": base["is_collision_free"],
                    "label": int(is_on_target and base["is_collision_free"]),
                })

            for lbl in labels:
                lbl["sample_id"] = sample_id
                lbl["view_sample_id"] = view_sample_id
                lbl["split"] = split
                all_records.append(lbl)

        if all_records:
            df = pd.DataFrame(all_records)
            out_path = config.RANK_LABELS_DIR / f"{split}_{detector}_labels.parquet"
            df.to_parquet(out_path, index=False)

            n_pos = int(df["label"].sum())
            n_neg = len(df) - n_pos
            print(f"  [{split}] {len(df)} labels ({n_pos} pos, {n_neg} neg) → {out_path}")
        else:
            print(f"  [{split}] No labels generated")


def main():
    parser = argparse.ArgumentParser(
        description="Step 7: Build target-aware training labels"
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument(
        "--detector", type=str, default=config.DEFAULT_DETECTOR,
        choices=["geometric"],
        help="Which detector's candidates to use (default: geometric).",
    )
    args = parser.parse_args()
    build_labels(splits=args.splits, detector=args.detector)


if __name__ == "__main__":
    main()
