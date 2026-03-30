"""
scripts/step06_grasp_candidates.py — Generate 6-DoF grasp candidates
======================================================================
Step 6: Run a FIXED pretrained grasp detector on FULL-SCENE point clouds.
Do NOT crop to target region — grounding is used for reranking only.

Usage:
    python scripts/step06_grasp_candidates.py
    python scripts/step06_grasp_candidates.py --splits test_seen --top-k 50
    python scripts/step06_grasp_candidates.py --detector antipodal
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.grasp_detector import GraspNetDetector, AntipodalSampler, PrecomputedGraspLoader


def _create_detector(detector_type: str, top_k: int):
    """Create a grasp detector, with graceful fallback.

    Detector types:
        graspnet   — official GraspNet baseline (requires checkpoint + deps)
        antipodal  — geometry-based antipodal sampler (no external deps)
        precomputed — load pre-generated .npy files
    """
    if detector_type == "antipodal":
        print("[step06] Using AntipodalSampler (no external dependencies)")
        return AntipodalSampler(top_k=top_k)

    if detector_type == "precomputed":
        print("[step06] Using PrecomputedGraspLoader")
        return PrecomputedGraspLoader()

    # Default: try GraspNetDetector, fall back to Antipodal
    try:
        det = GraspNetDetector()
        # Force lazy load to check deps now
        det._ensure_loaded()
        print("[step06] Using GraspNetDetector (official baseline)")
        return det
    except (FileNotFoundError, ImportError) as e:
        print(f"[step06] GraspNetDetector unavailable: {e}")
        print("[step06] Falling back to AntipodalSampler.")
        print("         To use the official detector, install graspnetAPI and")
        print("         clone graspnet-baseline. See README for details.")
        return AntipodalSampler(top_k=top_k)


def generate_candidates(
    splits: list = None,
    top_k: int = config.GRASP_TOP_K,
    detector_type: str = "graspnet",
):
    """Run grasp detection on all indexed views."""
    if splits is None:
        splits = config.ALL_SPLITS

    detector = _create_detector(detector_type, top_k)

    # PrecomputedGraspLoader needs special handling
    is_precomputed = isinstance(detector, PrecomputedGraspLoader)

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

            if is_precomputed:
                # PrecomputedGraspLoader reads from .npy files
                candidates = detector.load_from_file(
                    scene_id=view["scene_id"],
                    camera=view["camera"],
                    frame_id=view["frame_id"],
                    top_k=top_k,
                )
            else:
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
    parser.add_argument(
        "--detector", type=str, default="graspnet",
        choices=["graspnet", "antipodal", "precomputed"],
        help="Detector type: graspnet (try official, fall back to antipodal), "
             "antipodal (geometry-based), precomputed (load .npy files)",
    )
    args = parser.parse_args()

    generate_candidates(
        splits=args.splits,
        top_k=args.top_k,
        detector_type=args.detector,
    )


if __name__ == "__main__":
    main()

