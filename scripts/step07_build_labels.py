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
from src.data_utils import load_label
from src.grasp_detector import GraspCandidate
from src.label_builder import generate_labels_for_sample


def _load_candidates(sample_id: str, detector: str = "antipodal"):
    """Load saved grasp candidates for a view.

    Searches detector-specific subdirectory first (new layout),
    then falls back to flat layout (legacy).
    """
    # New layout: derived/grasp_candidates/{detector}/{sample_id}.npz
    path = config.GRASP_CANDIDATES_DIR / detector / f"{sample_id}.npz"
    if not path.exists():
        # Legacy fallback: derived/grasp_candidates/{sample_id}.npz
        path = config.GRASP_CANDIDATES_DIR / f"{sample_id}.npz"
    if not path.exists():
        return []
    data = np.load(str(path), allow_pickle=True)

    candidates = []
    n = int(data.get("num_candidates", 0))
    for i in range(n):
        candidates.append(GraspCandidate(
            candidate_id=i,
            position=data["positions"][i].tolist(),
            rotation=data["rotations"][i].tolist(),
            width=float(data["widths"][i]),
            detector_score=float(data["detector_scores"][i]),
            source=str(data["sources"][i]),
        ))
    return candidates


def build_labels(splits: list = None, detector: str = "antipodal"):
    """Build training labels for all candidates."""
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

            # Load candidates for this view
            candidates = _load_candidates(view_sample_id, detector)
            if not candidates:
                continue

            # Load point cloud
            pcd_path = config.POINTCLOUDS_DIR / f"{view_sample_id}.npz"
            if not pcd_path.exists():
                continue
            pcd_data = np.load(str(pcd_path))
            scene_points = pcd_data["points"]
            scene_pixel_coords = pcd_data["pixel_coords"]

            # Load label mask
            scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"
            try:
                label = load_label(scene_dir, frame_id, camera)
            except Exception:
                continue

            # NOTE: Collision labels are NOT loaded here.
            # Official GraspNet collision labels are indexed by
            # (object × angle × depth), not by detector candidate_id.
            # See src/label_builder.py for details on the proxy used.

            # Generate labels
            labels = generate_labels_for_sample(
                candidates, target_mask_val,
                scene_points, scene_pixel_coords, label,
            )

            for lbl in labels:
                lbl["sample_id"] = sample_id
                lbl["view_sample_id"] = view_sample_id
                lbl["split"] = split
                all_records.append(lbl)

        if all_records:
            df = pd.DataFrame(all_records)
            out_path = config.RANK_LABELS_DIR / f"{split}_labels.parquet"
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
        "--detector", type=str, default="antipodal",
        choices=["antipodal", "graspnet", "precomputed"],
        help="Which detector's candidates to use (must match step06).",
    )
    args = parser.parse_args()
    build_labels(splits=args.splits, detector=args.detector)


if __name__ == "__main__":
    main()
