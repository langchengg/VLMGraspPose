"""
scripts/step08_extract_features.py — Extract candidate-specific features
==========================================================================
Step 8: Turn each candidate into a 9-dim semantic-geometric feature vector.

Usage:
    python scripts/step08_extract_features.py
    python scripts/step08_extract_features.py --splits train val --grounding oracle
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
from src.data_utils import (
    load_label, load_depth, load_camera_intrinsics, get_factor_depth,
    load_grasp_candidates,
)
from src.point_cloud import (
    crop_point_cloud_by_mask, crop_point_cloud_by_bbox,
    crop_points_by_binary_mask,
)


from src.feature_extractor import FeatureExtractor


def extract_features(
    splits: list = None,
    grounding: str = "predicted",
    grounding_task: str = "seg",
    detector: str = "antipodal",
):
    """Extract features for all candidates.

    IMPORTANT: When grounding='predicted', features are derived ONLY
    from the predicted bbox/mask. GT label is NOT used for features.

    Default: predicted+seg, matching the default inference grounder (seg).
    Use --grounding oracle for upper-bound experiments.

    Args:
        grounding_task: which Florence-2 task to use when grounding='predicted'.
                        Must be 'phrase' or 'seg'. This must match the task
                        used when running step04 and the grounder used at
                        inference (step10, default: seg).
    """
    if splits is None:
        splits = config.ALL_SPLITS

    extractor = FeatureExtractor()

    for split in splits:
        queries_path = config.QUERIES_DIR / f"{split}_queries.jsonl"
        oracle_path = config.ORACLE_TARGETS_DIR / f"{split}_oracle.jsonl"

        if not queries_path.exists():
            print(f"  [SKIP] {split}: missing queries")
            continue

        # Load grounding targets
        target_map = {}
        if grounding == "oracle":
            if not oracle_path.exists():
                print(f"  [SKIP] {split}: missing oracle (run step03)")
                continue
            with open(oracle_path) as f:
                for line in f:
                    rec = json.loads(line)
                    target_map[rec["sample_id"]] = {
                        "bbox": rec["gt_bbox"],
                        "mask_val": rec["gt_mask_val"],
                        "confidence": 1.0,
                        "mask_path": None,
                    }
        else:
            # Load the specific grounding task file
            # New naming: {split}_grounding_{task}.jsonl
            pred_path = config.GROUNDING_PRED_DIR / f"{split}_grounding_{grounding_task}.jsonl"
            if not pred_path.exists():
                # Backward compat: old naming
                old_path = config.GROUNDING_PRED_DIR / f"{split}_grounding_pred.jsonl"
                if old_path.exists():
                    pred_path = old_path
                    print(f"  [WARN] Using legacy grounding file: {old_path.name}")
                    print(f"         Re-run step04 with --task {grounding_task} for explicit control.")
                else:
                    print(f"  [SKIP] {split}: no grounding file for task={grounding_task}")
                    continue

            print(f"  [{split}] Loading predicted grounding: {pred_path.name}")
            with open(pred_path) as f:
                for line in f:
                    rec = json.loads(line)
                    target_map[rec["sample_id"]] = {
                        "bbox": rec["pred_bbox"],
                        "mask_val": None,
                        "confidence": rec.get("florence_confidence", 1.0),
                        "mask_path": rec.get("pred_mask_path"),
                    }

        config.RANK_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        all_records = []
        view_cache = {}

        with open(queries_path) as fin:
            lines = fin.readlines()

        for line in tqdm(lines, desc=f"Features [{split}]"):
            query = json.loads(line)
            sample_id = query["sample_id"]
            if sample_id not in target_map:
                continue

            target = target_map[sample_id]
            view_sample_id = query["view_sample_id"]
            scene_id = query["scene_id"]
            camera = query["camera"]
            frame_id = query["frame_id"]
            obj_id = query["target_object_id"]
            target_mask_val = obj_id + 1
            scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"
            ctx = view_cache.get(view_sample_id)
            if ctx is None:
                ctx = {"skip": True}

                candidates = load_grasp_candidates(view_sample_id, detector)
                if candidates:
                    pcd_path = config.POINTCLOUDS_DIR / f"{view_sample_id}.npz"
                    if pcd_path.exists():
                        pcd_data = np.load(str(pcd_path))
                        scene_points = pcd_data["points"]
                        scene_pixel_coords = pcd_data["pixel_coords"]

                        try:
                            factor = get_factor_depth(scene_dir, camera)
                            depth = load_depth(scene_dir, frame_id, camera, factor)
                            K = load_camera_intrinsics(scene_dir, camera)
                        except Exception:
                            depth = None
                            K = None

                        label = None
                        if grounding == "oracle" and depth is not None and K is not None:
                            try:
                                label = load_label(scene_dir, frame_id, camera)
                            except Exception:
                                label = None

                        if depth is not None and K is not None and (grounding != "oracle" or label is not None):
                            ctx = {
                                "skip": False,
                                "candidates": candidates,
                                "scene_points": scene_points,
                                "scene_pixel_coords": scene_pixel_coords,
                                "depth": depth,
                                "intrinsics": K,
                                "label": label,
                            }

                view_cache[view_sample_id] = ctx

            if ctx["skip"]:
                continue

            candidates = ctx["candidates"]
            scene_points = ctx["scene_points"]
            scene_pixel_coords = ctx["scene_pixel_coords"]
            depth = ctx["depth"]
            K = ctx["intrinsics"]
            label = ctx["label"]

            # ── Build target_mask from the correct source ────────────
            # Oracle: use GT label
            # Predicted: use Florence-2 mask or None (bbox fallback)
            # This is the key fix: NO GT leakage in predicted mode
            target_mask = None
            if grounding == "oracle":
                if label is None:
                    continue
                target_mask = (label == target_mask_val)
            elif target.get("mask_path"):
                mask_path = Path(target["mask_path"])
                # Resolve relative paths (new format) against project root
                if not mask_path.is_absolute():
                    mask_path = config.PROJECT_ROOT / mask_path
                if mask_path.exists():
                    target_mask = np.load(str(mask_path))

            # Get target points using the grounding-consistent mask
            if target_mask is not None and target_mask.any():
                target_pts = crop_points_by_binary_mask(
                    scene_points, scene_pixel_coords, target_mask,
                )
            else:
                target_pts, _ = crop_point_cloud_by_bbox(
                    scene_points, scene_pixel_coords, target["bbox"],
                )

            # Extract features (NO GT label passed)
            features = extractor.extract_batch(
                candidates=candidates,
                target_bbox=target["bbox"],
                target_mask=target_mask,
                target_points=target_pts,
                scene_points=scene_points,
                scene_pixel_coords=scene_pixel_coords,
                florence_conf=target["confidence"],
                depth=depth,
                intrinsics=K,
            )

            for i, c in enumerate(candidates):
                feat_dict = {
                    name: float(features[i, j])
                    for j, name in enumerate(config.FEATURE_NAMES)
                }
                all_records.append({
                    "sample_id": sample_id,
                    "view_sample_id": view_sample_id,
                    "candidate_id": c.candidate_id,
                    "grounding": grounding,
                    "split": split,
                    **feat_dict,
                })

        if all_records:
            df = pd.DataFrame(all_records)
            # Include grounding, task, AND detector in filename
            # to prevent overwriting across any configuration axis
            if grounding == "predicted":
                fname = f"{split}_predicted_{grounding_task}_{detector}_features.parquet"
            else:
                fname = f"{split}_{grounding}_{detector}_features.parquet"
            out_path = config.RANK_FEATURES_DIR / fname
            df.to_parquet(out_path, index=False)
            print(f"  [{split}] {len(df)} feature vectors → {out_path}")
        else:
            print(f"  [{split}] No features extracted")


def main():
    parser = argparse.ArgumentParser(
        description="Step 8: Extract candidate-specific features"
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument(
        "--grounding", type=str, default="predicted",
        choices=["oracle", "predicted"],
        help="Use oracle (GT) or predicted (Florence-2) grounding. "
             "Default: predicted (matches step10 default grounder=seg).",
    )
    parser.add_argument(
        "--task", type=str, default="seg",
        choices=["phrase", "seg"],
        help="Florence-2 task for predicted grounding (must match step04)."
             " Default: seg (activates all 9 features). Only used when --grounding=predicted.",
    )
    parser.add_argument(
        "--detector", type=str, default="antipodal",
        choices=["antipodal", "graspnet", "precomputed"],
        help="Which detector's candidates to use (must match step06).",
    )
    args = parser.parse_args()

    extract_features(
        splits=args.splits,
        grounding=args.grounding,
        grounding_task=args.task,
        detector=args.detector,
    )


if __name__ == "__main__":
    main()
