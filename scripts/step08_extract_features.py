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
from src.data_utils import load_label, load_depth, load_camera_intrinsics, get_factor_depth
from src.point_cloud import crop_point_cloud_by_mask, crop_point_cloud_by_bbox
from src.grasp_detector import GraspCandidate
from src.feature_extractor import FeatureExtractor


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


def _crop_points_by_binary_mask(
    points: np.ndarray,
    pixel_coords: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Crop points by a binary HxW mask."""
    u, v = pixel_coords[:, 0], pixel_coords[:, 1]
    H, W = mask.shape[:2]
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = u[valid]
    v_valid = v[valid]
    on_mask = mask[v_valid, u_valid]
    idx = np.where(valid)[0][on_mask]
    return points[idx]


def extract_features(
    splits: list = None,
    grounding: str = "oracle",
):
    """Extract features for all candidates.

    IMPORTANT: When grounding='predicted', features are derived ONLY
    from the predicted bbox/mask. GT label is NOT used for features.
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
            pred_path = config.GROUNDING_PRED_DIR / f"{split}_grounding_pred.jsonl"
            if pred_path.exists():
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

            # Load candidates
            candidates = _load_candidates(view_sample_id)
            if not candidates:
                continue

            # Load point cloud
            pcd_path = config.POINTCLOUDS_DIR / f"{view_sample_id}.npz"
            if not pcd_path.exists():
                continue
            pcd_data = np.load(str(pcd_path))
            scene_points = pcd_data["points"]
            scene_pixel_coords = pcd_data["pixel_coords"]

            # Load depth + intrinsics (needed for f8)
            scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"
            try:
                factor = get_factor_depth(scene_dir, camera)
                depth = load_depth(scene_dir, frame_id, camera, factor)
                K = load_camera_intrinsics(scene_dir, camera)
            except Exception:
                continue

            # ── Build target_mask from the correct source ────────────
            # Oracle: use GT label
            # Predicted: use Florence-2 mask or None (bbox fallback)
            # This is the key fix: NO GT leakage in predicted mode
            target_mask = None
            if grounding == "oracle":
                try:
                    label = load_label(scene_dir, frame_id, camera)
                    target_mask = (label == target_mask_val)
                except Exception:
                    continue
            elif target.get("mask_path") and Path(target["mask_path"]).exists():
                target_mask = np.load(target["mask_path"])

            # Get target points using the grounding-consistent mask
            if target_mask is not None and target_mask.any():
                target_pts = _crop_points_by_binary_mask(
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
            out_path = config.RANK_FEATURES_DIR / f"{split}_{grounding}_features.parquet"
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
        "--grounding", type=str, default="oracle",
        choices=["oracle", "predicted"],
        help="Use oracle (GT) or predicted (Florence-2) grounding",
    )
    args = parser.parse_args()

    extract_features(splits=args.splits, grounding=args.grounding)


if __name__ == "__main__":
    main()
