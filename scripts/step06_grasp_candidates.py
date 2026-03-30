"""
scripts/step06_grasp_candidates.py — Generate 6-DoF grasp candidates
======================================================================
Step 6: Run a FIXED pretrained grasp detector on FULL-SCENE point clouds.
Do NOT crop to target region — grounding is used for reranking only.

Usage:
    python scripts/step06_grasp_candidates.py
    python scripts/step06_grasp_candidates.py --splits test_seen --top-k 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.grasp_detector import GraspNetDetector


def generate_candidates(
    splits: list = None,
    top_k: int = config.GRASP_TOP_K,
):
    """Run grasp detection on all indexed views."""
    if splits is None:
        splits = config.ALL_SPLITS

    detector = GraspNetDetector()

    for split in splits:
        views_path = config.SPLITS_DIR / f"{split}_views.jsonl"
        if not views_path.exists():
            print(f"  [SKIP] {views_path} not found (run step01 first)")
            continue

        config.GRASP_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        total = 0

        with open(views_path) as fin:
            lines = fin.readlines()

        for line in tqdm(lines, desc=f"Grasps [{split}]"):
            view = json.loads(line)
            sample_id = view["sample_id"]

            out_path = config.GRASP_CANDIDATES_DIR / f"{sample_id}.npz"
            if out_path.exists():
                total += 1
                continue

            # Load pre-computed point cloud
            pcd_path = config.POINTCLOUDS_DIR / f"{sample_id}.npz"
            if not pcd_path.exists():
                continue

            pcd_data = np.load(str(pcd_path))
            points = pcd_data["points"]
            colors = pcd_data.get("colors", None)

            if len(points) < 100:
                continue

            # Run detector on FULL scene point cloud
            candidates = detector.detect(points, colors, top_k=top_k)

            if not candidates:
                continue

            # Save as NPZ
            positions = np.array([c.position for c in candidates], dtype=np.float32)
            rotations = np.array([c.rotation for c in candidates], dtype=np.float32)
            widths = np.array([c.width for c in candidates], dtype=np.float32)
            scores = np.array([c.detector_score for c in candidates], dtype=np.float32)
            sources = [c.source for c in candidates]

            np.savez_compressed(
                out_path,
                positions=positions,
                rotations=rotations,
                widths=widths,
                detector_scores=scores,
                sources=np.array(sources),
                num_candidates=len(candidates),
            )
            total += 1

        print(f"  [{split}] {total} candidate sets → {config.GRASP_CANDIDATES_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Generate grasp candidates"
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=config.GRASP_TOP_K)
    args = parser.parse_args()

    generate_candidates(splits=args.splits, top_k=args.top_k)


if __name__ == "__main__":
    main()
