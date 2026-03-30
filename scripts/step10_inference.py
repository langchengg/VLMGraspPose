"""
scripts/step10_inference.py — Full-chain test-time inference
==============================================================
Step 10: Run the complete pipeline on unseen test views.

    Florence-2 → depth→pcd → grasp detector → features → reranker → top-K

Usage:
    python scripts/step10_inference.py --splits test_seen --reranker rule
    python scripts/step10_inference.py --splits test_seen test_similar test_novel --reranker mlp
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import (
    load_rgb, load_depth, load_label,
    load_camera_intrinsics, get_factor_depth,
)
from src.point_cloud import (
    backproject_depth, add_colors,
    crop_point_cloud_by_mask, crop_point_cloud_by_bbox,
)
from src.grounding import get_grounder
from src.grasp_detector import GraspNetDetector, GraspCandidate
from src.feature_extractor import FeatureExtractor
from src.reranker import get_reranker
from src.label_builder import associate_grasp_to_object


def _load_candidates(view_sample_id: str):
    path = config.GRASP_CANDIDATES_DIR / f"{view_sample_id}.npz"
    if not path.exists():
        return []
    data = np.load(str(path), allow_pickle=True)
    candidates = []
    for i in range(int(data.get("num_candidates", 0))):
        candidates.append(GraspCandidate(
            candidate_id=i,
            position=data["positions"][i].tolist(),
            rotation=data["rotations"][i].tolist(),
            width=float(data["widths"][i]),
            detector_score=float(data["detector_scores"][i]),
            source=str(data["sources"][i]),
        ))
    return candidates


def run_inference(
    splits: list = None,
    grounder_name: str = "phrase",
    reranker_name: str = "rule",
    max_samples: int = None,
    use_cached_grasps: bool = True,
):
    """Run full inference chain on test splits."""
    if splits is None:
        splits = config.TEST_SPLITS

    grounder = get_grounder(grounder_name)
    reranker = get_reranker(reranker_name)
    extractor = FeatureExtractor()
    detector = None if use_cached_grasps else GraspNetDetector()

    for split in splits:
        queries_path = config.QUERIES_DIR / f"{split}_queries.jsonl"
        oracle_path = config.ORACLE_TARGETS_DIR / f"{split}_oracle.jsonl"

        if not queries_path.exists():
            print(f"  [SKIP] {split}: missing queries")
            continue

        # Load oracle for GT evaluation
        oracle_map = {}
        if oracle_path.exists():
            with open(oracle_path) as f:
                for line in f:
                    rec = json.loads(line)
                    oracle_map[rec["sample_id"]] = rec

        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        predictions = []

        with open(queries_path) as fin:
            lines = fin.readlines()

        for line in tqdm(lines, desc=f"Inference [{split}]"):
            if max_samples and len(predictions) >= max_samples:
                break

            query = json.loads(line)
            sample_id = query["sample_id"]
            view_sample_id = query["view_sample_id"]
            scene_id = query["scene_id"]
            camera = query["camera"]
            frame_id = query["frame_id"]
            text_query = query["text_query"]
            obj_id = query["target_object_id"]
            target_mask_val = obj_id + 1

            scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"

            t0 = time.time()

            try:
                # ── 1. Load data ─────────────────────────────────────
                rgb = load_rgb(scene_dir, frame_id, camera)
                factor = get_factor_depth(scene_dir, camera)
                depth = load_depth(scene_dir, frame_id, camera, factor)
                label = load_label(scene_dir, frame_id, camera)
                K = load_camera_intrinsics(scene_dir, camera)

                # ── 2. Florence-2 grounding ──────────────────────────
                if grounder_name == "gt":
                    grounding = grounder.ground(
                        rgb, text_query, label=label, mask_val=target_mask_val,
                    )
                else:
                    grounding = grounder.ground(rgb, text_query)

                if grounding is None:
                    continue

                # ── 3. Load / generate grasp candidates ──────────────
                if use_cached_grasps:
                    candidates = _load_candidates(view_sample_id)
                else:
                    points, px = backproject_depth(depth, K)
                    colors = add_colors(rgb, px)
                    candidates = detector.detect(points, colors)

                if not candidates:
                    continue

                # ── 4. Point cloud for features ──────────────────────
                pcd_path = config.POINTCLOUDS_DIR / f"{view_sample_id}.npz"
                if pcd_path.exists():
                    pcd_data = np.load(str(pcd_path))
                    scene_points = pcd_data["points"]
                    scene_pixel_coords = pcd_data["pixel_coords"]
                else:
                    scene_points, scene_pixel_coords = backproject_depth(depth, K)

                # Target points
                target_mask = (label == target_mask_val) if grounder_name == "gt" else grounding.mask
                if target_mask is not None and target_mask.any():
                    target_pts, _ = crop_point_cloud_by_mask(
                        scene_points, scene_pixel_coords, label, target_mask_val,
                    )
                else:
                    target_pts, _ = crop_point_cloud_by_bbox(
                        scene_points, scene_pixel_coords, grounding.bbox,
                    )

                # ── 5. Extract features ──────────────────────────────
                features = extractor.extract_batch(
                    candidates=candidates,
                    target_bbox=grounding.bbox,
                    target_mask=target_mask,
                    target_points=target_pts,
                    scene_points=scene_points,
                    scene_pixel_coords=scene_pixel_coords,
                    label=label,
                    target_mask_val=target_mask_val,
                    florence_conf=grounding.confidence,
                    depth=depth,
                    intrinsics=K,
                )

                # ── 6. Rerank ────────────────────────────────────────
                ranked = reranker.select_top_k(features, candidates, k=5)

                # ── 7. Evaluate on-target ────────────────────────────
                for g in ranked:
                    # Check if this grasp is on the target object
                    cid = g["candidate_id"]
                    c = candidates[cid]
                    assoc = associate_grasp_to_object(
                        c, scene_points, scene_pixel_coords, label,
                    )
                    g["is_on_target"] = (assoc == target_mask_val)

            except Exception as e:
                continue

            elapsed = time.time() - t0

            pred = {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "frame_id": frame_id,
                "target_class": query["object_name"],
                "text_query": text_query,
                "grounder": grounder_name,
                "reranker": reranker_name,
                "pred_bbox": grounding.bbox,
                "ranked_grasps": ranked,
                "best_grasp": ranked[0] if ranked else None,
                "latency": elapsed,
                "split": split,
            }
            predictions.append(pred)

        # Save predictions
        out_path = config.RESULTS_DIR / f"predictions_{split}.json"
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"  [{split}] {len(predictions)} predictions → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 10: Full-chain test-time inference"
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument(
        "--grounder", type=str, default="phrase",
        choices=["gt", "phrase", "seg"],
    )
    parser.add_argument(
        "--reranker", type=str, default="rule",
        choices=["detector", "rule", "logistic", "mlp", "pairwise"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use cached candidates, run detector live")
    args = parser.parse_args()

    run_inference(
        splits=args.splits,
        grounder_name=args.grounder,
        reranker_name=args.reranker,
        max_samples=args.max_samples,
        use_cached_grasps=not args.no_cache,
    )


if __name__ == "__main__":
    main()
