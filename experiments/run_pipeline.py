"""
experiments/run_pipeline.py — End-to-end demo pipeline
=======================================================
Runs all 5 stages on test_seen data.
Uses GT grounding + Antipodal sampler + Rule scorer.

Usage:
    python -m experiments.run_pipeline
    python -m experiments.run_pipeline --max-samples 5
    python -m experiments.run_pipeline --scorer rule
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
from data.dataset import (
    discover_scenes, load_scene, generate_samples,
    load_rgb, load_depth, load_label,
)
from data.point_cloud import (
    backproject_depth, crop_point_cloud_by_mask,
    crop_point_cloud_by_bbox,
)
from stage1.grounding import get_grounder
from stage1.postprocess_bbox import save_stage1_output
from stage2.roi_sampler import generate_target_grasps, save_stage2_output
from stage3.feature_extractor import FeatureExtractor, save_features
from stage4.rule_scorer import RuleScorer
from stage5.select_best_grasp import select_best_grasp, save_selection


def run_pipeline(
    split: str = "test_seen",
    scorer_name: str = "rule",
    max_samples: int = None,
    use_extended_features: bool = False,
    view_stride: int = config.VIEW_STRIDE,
):
    """Run the complete pipeline on a data split."""
    print(f"{'='*60}")
    print(f"VLMGraspPose Pipeline — Demo Run")
    print(f"  Split: {split}")
    print(f"  Scorer: {scorer_name}")
    print(f"  Feature dim: {'extended (9)' if use_extended_features else 'core (5)'}")
    print(f"{'='*60}")

    data_dir = config.DATA_DIRS.get(split)
    if data_dir is None or not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    # ── Initialise components ────────────────────────────────────────
    grounder = get_grounder("gt")
    feature_extractor = FeatureExtractor(use_extended=use_extended_features)

    if scorer_name == "rule":
        scorer = RuleScorer()
    elif scorer_name == "logistic":
        from stage4.logistic_scorer import LogisticScorer
        scorer = LogisticScorer(config.MODELS_DIR / "scorer_logreg.pkl")
        if not scorer.is_trained:
            print("[WARN] Logistic scorer not trained. Falling back to rule scorer.")
            scorer = RuleScorer()
            scorer_name = "rule"
    elif scorer_name == "mlp":
        from stage4.mlp_scorer import MLPScorer
        scorer = MLPScorer(model_path=config.MODELS_DIR / "scorer_mlp.pt")
        if not scorer.is_trained:
            print("[WARN] MLP scorer not trained. Falling back to rule scorer.")
            scorer = RuleScorer()
            scorer_name = "rule"
    else:
        raise ValueError(f"Unknown scorer: {scorer_name}")

    # ── Discover scenes ──────────────────────────────────────────────
    scenes = discover_scenes(data_dir)
    print(f"Found {len(scenes)} scenes")

    all_results = []
    total_time = 0
    sample_count = 0

    for scene_dir in scenes:
        try:
            scene_meta = load_scene(scene_dir)
        except Exception as e:
            print(f"  [WARN] Skipping {scene_dir.name}: {e}")
            continue

        samples = generate_samples(scene_meta, view_stride=view_stride)

        for sample in samples:
            if max_samples and sample_count >= max_samples:
                break

            t0 = time.time()

            try:
                result = _process_sample(
                    sample, scene_meta, grounder, feature_extractor,
                    scorer, scorer_name, use_extended_features,
                )
            except Exception as e:
                print(f"  [ERROR] {sample.sample_id}: {e}")
                continue

            elapsed = time.time() - t0
            total_time += elapsed
            sample_count += 1

            if result:
                result["latency"] = elapsed
                all_results.append(result)

                # Print progress
                top1 = result["selections"][0] if result["selections"] else None
                top1_score = top1["final_score"] if top1 else 0
                print(f"  [{sample_count}] {sample.sample_id} | "
                      f"candidates={result['num_candidates']} | "
                      f"top1_score={top1_score:.3f} | "
                      f"{elapsed:.2f}s")

        if max_samples and sample_count >= max_samples:
            break

    # ── Save summary ─────────────────────────────────────────────────
    summary_path = config.PROJECT_ROOT / "results" / f"pipeline_summary_{split}_{scorer_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": split,
        "scorer": scorer_name,
        "num_samples": sample_count,
        "total_time": total_time,
        "avg_time_per_sample": total_time / max(sample_count, 1),
        "results": all_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Samples processed: {sample_count}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/sample: {total_time/max(sample_count,1):.2f}s")
    print(f"  Results saved: {summary_path}")
    print(f"{'='*60}")


def _process_sample(
    sample, scene_meta, grounder, feature_extractor,
    scorer, scorer_name, use_extended,
):
    """Process one sample through all 5 stages."""

    # ── Load data ────────────────────────────────────────────────────
    rgb = load_rgb(scene_meta.scene_dir, sample.view_id, scene_meta.camera_type)
    depth = load_depth(scene_meta.scene_dir, sample.view_id,
                       scene_meta.camera_type, scene_meta.factor_depth)
    label = load_label(scene_meta.scene_dir, sample.view_id, scene_meta.camera_type)
    intrinsics = scene_meta.intrinsics

    # Instance ID in label mask = obj_id + 1
    instance_id = sample.target_obj_id + 1

    # ── Stage 1: Target Grounding ────────────────────────────────────
    grounding = grounder.ground(
        rgb, sample.text_query,
        label=label, instance_id=instance_id,
    )
    if grounding is None:
        return None

    save_stage1_output(sample.sample_id, sample.text_query, grounding)

    # ── Stage 2: Grasp Candidate Generation ──────────────────────────
    candidates = generate_target_grasps(
        depth=depth,
        intrinsics=intrinsics,
        bbox=grounding.bbox,
        label=label,
        instance_id=instance_id,
        top_k=config.GRASP_TOP_K,
    )

    if not candidates:
        return None

    save_stage2_output(sample.sample_id, candidates)

    # ── Stage 3: Feature Extraction ──────────────────────────────────
    # Get target point cloud for feature computation
    points, pixel_coords = backproject_depth(depth, intrinsics)
    target_pts, _ = crop_point_cloud_by_mask(
        points, pixel_coords, label, instance_id
    )
    if len(target_pts) < 5:
        target_pts, _ = crop_point_cloud_by_bbox(
            points, pixel_coords, grounding.bbox
        )

    features = feature_extractor.extract(
        candidates=candidates,
        target_bbox=grounding.bbox,
        target_mask=grounding.mask,
        target_points=target_pts,
        vlm_confidence=grounding.confidence,
        depth=depth,
        intrinsics=intrinsics,
        scene_points=points if use_extended else None,
    )

    save_features(
        sample.sample_id, features,
        [c.candidate_id for c in candidates],
    )

    # ── Stage 4: Scoring ─────────────────────────────────────────────
    scores = scorer.score(features)

    # ── Stage 5: Selection ───────────────────────────────────────────
    selections = select_best_grasp(candidates, scores, top_k=5)
    save_selection(sample.sample_id, selections, scorer_name)

    return {
        "sample_id": sample.sample_id,
        "scene_id": sample.scene_id,
        "view_id": sample.view_id,
        "target_class": sample.target_class,
        "text_query": sample.text_query,
        "gt_bbox": sample.gt_bbox,
        "num_candidates": len(candidates),
        "selections": selections,
    }


def main():
    parser = argparse.ArgumentParser(description="Run VLMGraspPose pipeline")
    parser.add_argument("--split", type=str, default="test_seen")
    parser.add_argument("--scorer", type=str, default="rule",
                        choices=["rule", "logistic", "mlp"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--extended", action="store_true",
                        help="Use extended 9-dim features")
    parser.add_argument("--view-stride", type=int, default=config.VIEW_STRIDE)
    args = parser.parse_args()

    run_pipeline(
        split=args.split,
        scorer_name=args.scorer,
        max_samples=args.max_samples,
        use_extended_features=args.extended,
        view_stride=args.view_stride,
    )


if __name__ == "__main__":
    main()
