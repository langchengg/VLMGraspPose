"""
scripts/step10_inference.py — Full-chain test-time inference
==============================================================
Step 10: Run the complete pipeline on unseen test views.

    Florence-2-base → depth→pcd → RGB-D geometric sampler → features → MLP reranker → top-K

Usage:
    python scripts/step10_inference.py --splits test_seen
    python scripts/step10_inference.py --splits test_seen test_similar test_novel
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
    load_grasp_candidates,
)
from src.point_cloud import (
    backproject_depth,
)
from src.target_point_cloud import build_point_cloud_representation
from src.grounding import get_grounder
from src.grasp_detector import RGBDGeometricGraspSampler
from src.feature_extractor import FeatureExtractor
from src.reranker import get_reranker
from src.label_builder import associate_grasp_to_object


def group_predictions_by_view(predictions: list) -> list:
    """Group per-query predictions into one record per image/view.

    Keeps the legacy per-query prediction format intact while providing
    an additional view-level artifact with one top-1 result per target object.
    """
    grouped = {}

    for pred in predictions:
        view_sample_id = pred.get("view_sample_id")
        if not view_sample_id:
            view_sample_id = (
                f"scene_{pred['scene_id']:04d}_{pred['camera']}_{pred['frame_id']:04d}"
            )

        if view_sample_id not in grouped:
            grouped[view_sample_id] = {
                "view_sample_id": view_sample_id,
                "scene_id": pred["scene_id"],
                "camera": pred["camera"],
                "frame_id": pred["frame_id"],
                "split": pred.get("split"),
                "grounder": pred.get("grounder"),
                "reranker": pred.get("reranker"),
                "detector": pred.get("detector"),
                "objects": [],
            }

        grouped[view_sample_id]["objects"].append({
            "sample_id": pred["sample_id"],
            "target_object_id": pred["target_object_id"],
            "target_class": pred.get("target_class"),
            "text_query": pred.get("text_query"),
            "best_grasp": pred.get("best_grasp"),
            "ranked_grasps": pred.get("ranked_grasps", []),
            "failure_reason": pred.get("failure_reason"),
            "latency": pred.get("latency"),
        })

    results = []
    for view_sample_id in sorted(grouped.keys()):
        entry = grouped[view_sample_id]
        entry["objects"].sort(
            key=lambda obj: (obj["target_object_id"], obj["sample_id"])
        )
        entry["num_objects"] = len(entry["objects"])
        results.append(entry)
    return results


def _find_model_path(reranker_name: str, detector: str) -> Path:
    """Find the trained model file matching detector/grounding tags.

    Mirrors the naming convention from step09's _model_save_path().
    Searches tagged filenames first, then falls back to legacy untagged names.
    """
    if reranker_name in ("detector", "rule"):
        return None  # no model file needed

    ext = ".pt"
    base_name = {
        "mlp": "reranker_mlp",
    }.get(reranker_name, f"reranker_{reranker_name}")

    # Search order: tagged (step09 output) → legacy untagged
    candidates = [
        config.RERANKER_MLP_PATH if reranker_name == "mlp" else None,
        # Current naming: explicit grounding tags
        config.MODELS_DIR / f"{base_name}_{detector}_predicted{ext}",
        config.MODELS_DIR / f"{base_name}_{detector}_auto{ext}",
        config.MODELS_DIR / f"{base_name}_{detector}_oracle{ext}",
        # Legacy: untagged
        config.MODELS_DIR / f"{base_name}{ext}",
    ]

    for p in candidates:
        if p is not None and p.exists():
            return p

    searched = [c.name for c in candidates if c is not None]
    raise FileNotFoundError(
        f"No trained {reranker_name} model found for detector={detector}.\n"
        f"Searched: {searched}\n"
        "Run: python scripts/step09_train_reranker.py --model mlp "
        "--grounding predicted --detector geometric"
    )


def run_inference(
    splits: list = None,
    grounder_name: str = config.DEFAULT_GROUNDING,
    reranker_name: str = config.DEFAULT_RERANKER,
    max_samples: int = None,
    use_cached_grasps: bool = True,
    detector: str = config.DEFAULT_DETECTOR,
):
    """Run full inference chain on test splits."""
    if splits is None:
        splits = config.TEST_SPLITS

    grounder = get_grounder(grounder_name)

    # Find the correct model path for trained rerankers (BUG-1 fix)
    model_path = _find_model_path(reranker_name, detector)
    if model_path:
        print(f"  [Reranker] Loading {reranker_name} from {model_path.name}")
    reranker = get_reranker(reranker_name, model_path=model_path)

    extractor = FeatureExtractor()
    # Create detector for live inference (--no-cache)
    det = None
    if not use_cached_grasps:
        if detector != "geometric":
            raise ValueError(
                f"--no-cache requires detector type 'geometric', got '{detector}'."
            )
        det = RGBDGeometricGraspSampler()

    # Warm up Florence-2 model before the timing loop (ISSUE-9 fix)
    if hasattr(grounder, '_ensure_loaded'):
        print("  [Grounder] Warming up Florence-2 model...")
        grounder._ensure_loaded()

    for split in splits:
        queries_path = config.QUERIES_DIR / f"{split}_queries.jsonl"
        oracle_path = config.ORACLE_TARGETS_DIR / f"{split}_oracle.jsonl"

        if not queries_path.exists():
            print(f"  [SKIP] {split}: missing queries")
            continue

        # Load oracle map — ONLY used for evaluation (is_on_target check),
        # NOT for features or point-cloud cropping.
        oracle_map = {}
        if oracle_path.exists():
            with open(oracle_path) as f:
                for line in f:
                    rec = json.loads(line)
                    oracle_map[rec["sample_id"]] = rec

        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        predictions = []
        view_cache = {}

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
            failure_reason = None
            grounding = None
            ranked = []

            try:
                # ── 1. Load / reuse per-view context ─────────────────
                ctx = view_cache.get(view_sample_id)
                if ctx is None:
                    rgb = load_rgb(scene_dir, frame_id, camera)
                    factor = get_factor_depth(scene_dir, camera)
                    depth = load_depth(scene_dir, frame_id, camera, factor)
                    K = load_camera_intrinsics(scene_dir, camera)

                    # GT label loaded ONLY for evaluation, never for features
                    gt_label = load_label(scene_dir, frame_id, camera)

                    pcd_path = config.POINTCLOUDS_DIR / f"{view_sample_id}.npz"
                    if pcd_path.exists():
                        pcd_data = np.load(str(pcd_path))
                        scene_points = pcd_data["points"]
                        scene_pixel_coords = pcd_data["pixel_coords"]
                    else:
                        scene_points, scene_pixel_coords = backproject_depth(
                            depth, K
                        )

                    ctx = {
                        "rgb": rgb,
                        "depth": depth,
                        "K": K,
                        "gt_label": gt_label,
                        "scene_points": scene_points,
                        "scene_pixel_coords": scene_pixel_coords,
                    }
                    view_cache[view_sample_id] = ctx

                rgb = ctx["rgb"]
                depth = ctx["depth"]
                K = ctx["K"]
                gt_label = ctx["gt_label"]
                scene_points = ctx["scene_points"]
                scene_pixel_coords = ctx["scene_pixel_coords"]

                # ── 2. Grounding ─────────────────────────────────────
                if grounder_name == "gt":
                    grounding = grounder.ground(
                        rgb, text_query, label=gt_label, mask_val=target_mask_val,
                    )
                else:
                    grounding = grounder.ground(rgb, text_query)

                if grounding is None:
                    failure_reason = "grounding_failed"
                else:
                    # ── 3. Build target point cloud from GROUNDING source
                    if grounder_name == "gt":
                        target_mask = (gt_label == target_mask_val)
                    else:
                        target_mask = grounding.mask

                    pcr = build_point_cloud_representation(
                        scene_points,
                        scene_pixel_coords,
                        grounding.bbox,
                        target_mask=target_mask,
                    )
                    if len(pcr.clean_target_points) < config.TARGET_MIN_POINTS:
                        failure_reason = "target_point_cloud_too_small"
                    else:
                        # ── 4. Generate or load target-conditioned candidates
                        if use_cached_grasps:
                            candidates = load_grasp_candidates(sample_id, detector)
                        else:
                            candidates = det.detect(pcr.clean_target_points)

                        if not candidates:
                            failure_reason = "no_candidates"
                        else:
                            # ── 5. Extract target-conditioned features
                            features = extractor.extract_batch(
                                candidates=candidates,
                                target_bbox=grounding.bbox,
                                target_mask=target_mask,
                                target_points=pcr.clean_target_points,
                                scene_points=scene_points,
                                scene_pixel_coords=scene_pixel_coords,
                                grounding_score=grounding.confidence,
                                depth=depth,
                                intrinsics=K,
                            )

                            # ── 6. Re-rank and select best grasp
                            ranked = reranker.select_top_k(features, candidates, k=5)

                            # ── 7. Evaluate on-target using GT label only
                            for g in ranked:
                                cid = g["candidate_id"]
                                c = candidates[cid]
                                assoc = associate_grasp_to_object(
                                    c, scene_points, scene_pixel_coords, gt_label,
                                )
                                g["is_on_target"] = (assoc == target_mask_val)

            except Exception as e:
                failure_reason = f"exception: {type(e).__name__}: {e}"

            elapsed = time.time() - t0

            pred = {
                "sample_id": sample_id,
                "view_sample_id": view_sample_id,
                "scene_id": scene_id,
                "camera": camera,
                "frame_id": frame_id,
                "target_object_id": obj_id,
                "target_class": query["object_name"],
                "text_query": text_query,
                "grounder": grounder_name,
                "reranker": reranker_name,
                "detector": detector,
                "pred_bbox": grounding.bbox if grounding else None,
                "ranked_grasps": ranked,
                "best_grasp": ranked[0] if ranked else None,
                "latency": elapsed,
                "split": split,
                "failure_reason": failure_reason,
            }
            predictions.append(pred)

        # ── Save with grounder+reranker in filename ──────────────
        out_path = config.RESULTS_DIR / (
            f"predictions_{split}_{grounder_name}_{reranker_name}_{detector}.json"
        )
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"  [{split}] {len(predictions)} predictions → {out_path}")

        grouped = group_predictions_by_view(predictions)
        grouped_path = config.RESULTS_DIR / (
            f"top1_by_view_{split}_{grounder_name}_{reranker_name}_{detector}.json"
        )
        with open(grouped_path, "w") as f:
            json.dump(grouped, f, indent=2)

        print(f"  [{split}] {len(grouped)} grouped views → {grouped_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 10: Full-chain test-time inference"
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument(
        "--grounder", type=str, default=config.DEFAULT_GROUNDING,
        choices=["gt", "phrase", "seg"],
    )
    parser.add_argument(
        "--reranker", type=str, default=config.DEFAULT_RERANKER,
        choices=["detector", "rule", "mlp"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use cached candidates, run detector live")
    parser.add_argument(
        "--detector", type=str, default=config.DEFAULT_DETECTOR,
        choices=["geometric"],
        help="Which detector's cached candidates to use (default: geometric).",
    )
    args = parser.parse_args()

    run_inference(
        splits=args.splits,
        grounder_name=args.grounder,
        reranker_name=args.reranker,
        max_samples=args.max_samples,
        use_cached_grasps=not args.no_cache,
        detector=args.detector,
    )


if __name__ == "__main__":
    main()
