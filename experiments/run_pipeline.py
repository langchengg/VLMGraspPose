"""
experiments/run_pipeline.py — VLMGraspPose Full Architecture Pipeline
========================================================================
Implements the complete architecture flow:

    ┌─ Branch A: Target Grounding (VLM / GT) ──┐
    │                                           ├──→ Association → Re-Ranking → Selection
    └─ Branch B: Grasp Proposal (Geometric)  ───┘

Blocks:
    1. INPUT          — Text query + RGB image + Depth / Point cloud
    2. BRANCH A       — Target Grounding (Florence-2 VLM or GT oracle)
    3. BRANCH B       — Grasp Proposal (Antipodal geometric sampler)
    4. ASSOCIATION    — Candidate–Target Semantic-Geometric Features
    5. RE-RANKING     — Score candidates (Rule / Logistic / MLP)
    6. SELECTION      — Pick Top-K grasp poses
    7. EVALUATION     — Metrics are computed by experiments/eval.py

Usage:
    # GT oracle + rule scorer (baseline)
    python -m experiments.run_pipeline --split test_seen --grounder gt --scorer rule

    # Florence-2 VLM + MLP scorer
    python -m experiments.run_pipeline --split test_seen --grounder vlm --scorer mlp

    # Florence-2 phrase grounding + extended features
    python -m experiments.run_pipeline --grounder phrase --scorer mlp --extended
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


# =====================================================================
#  Pipeline orchestrator
# =====================================================================

def run_pipeline(
    split: str = "test_seen",
    grounder_name: str = "gt",
    scorer_name: str = "rule",
    max_samples: int = None,
    use_extended_features: bool = False,
    view_stride: int = config.VIEW_STRIDE,
):
    """Run the complete architecture pipeline on a data split.

    Architecture:
        INPUT → [Branch A: Grounding] + [Branch B: Grasp Proposal]
              → ASSOCIATION → RE-RANKING → SELECTION
    """
    use_vlm = grounder_name != "gt"

    print(f"\n{'═'*60}")
    print(f"  VLMGraspPose — Architecture Pipeline")
    print(f"{'═'*60}")
    print(f"  Split:         {split}")
    print(f"  Branch A:      Target Grounding  [{grounder_name}]")
    print(f"  Branch B:      Grasp Proposal    [antipodal]")
    print(f"  Features:      {'extended (9-dim)' if use_extended_features else 'core (5-dim)'}")
    print(f"  Re-Ranking:    {scorer_name}")
    print(f"{'═'*60}\n")

    # ── Validate data ────────────────────────────────────────────────
    data_dir = config.DATA_DIRS.get(split)
    if data_dir is None or not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    # ── Initialise architecture components ───────────────────────────

    # Branch A: Target Grounding
    grounder = get_grounder(grounder_name)

    # Candidate–Target Association
    feature_extractor = FeatureExtractor(use_extended=use_extended_features)

    # Re-Ranking / Scoring Module
    scorer, scorer_name = _load_scorer(scorer_name)

    # ── Discover scenes and run ──────────────────────────────────────
    scenes = discover_scenes(data_dir)
    print(f"Found {len(scenes)} scenes\n")

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
                    sample, scene_meta,
                    grounder=grounder,
                    feature_extractor=feature_extractor,
                    scorer=scorer,
                    scorer_name=scorer_name,
                    use_extended=use_extended_features,
                    use_vlm=use_vlm,
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

                top1 = result["selections"][0] if result["selections"] else None
                top1_score = top1["final_score"] if top1 else 0
                print(f"  [{sample_count}] {sample.sample_id} | "
                      f"candidates={result['num_candidates']} | "
                      f"top1_score={top1_score:.3f} | "
                      f"{elapsed:.2f}s")

        if max_samples and sample_count >= max_samples:
            break

    # ── Save summary ─────────────────────────────────────────────────
    tag = f"{split}_{scorer_name}"
    if use_vlm:
        tag += f"_{grounder_name}"
    summary_path = config.PROJECT_ROOT / "results" / f"pipeline_summary_{tag}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": split,
        "grounder": grounder_name,
        "scorer": scorer_name,
        "features": "extended" if use_extended_features else "core",
        "num_samples": sample_count,
        "total_time": total_time,
        "avg_time_per_sample": total_time / max(sample_count, 1),
        "results": all_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*60}")
    print(f"  Pipeline complete!")
    print(f"  Samples processed: {sample_count}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/sample: {total_time/max(sample_count,1):.2f}s")
    print(f"  Results saved: {summary_path}")
    print(f"{'═'*60}")


def _load_scorer(scorer_name: str):
    """Instantiate the re-ranking scorer by name."""
    if scorer_name == "rule":
        return RuleScorer(), "rule"
    elif scorer_name == "logistic":
        from stage4.logistic_scorer import LogisticScorer
        scorer = LogisticScorer(config.MODELS_DIR / "scorer_logreg.pkl")
        if not scorer.is_trained:
            print("[WARN] Logistic scorer not trained. Falling back to rule scorer.")
            return RuleScorer(), "rule"
        return scorer, "logistic"
    elif scorer_name == "mlp":
        from stage4.mlp_scorer import MLPScorer
        scorer = MLPScorer(model_path=config.MODELS_DIR / "scorer_mlp.pt")
        if not scorer.is_trained:
            print("[WARN] MLP scorer not trained. Falling back to rule scorer.")
            return RuleScorer(), "rule"
        return scorer, "mlp"
    else:
        raise ValueError(f"Unknown scorer: {scorer_name}")


# =====================================================================
#  Per-sample processing — follows architecture blocks
# =====================================================================

def _process_sample(
    sample, scene_meta, grounder, feature_extractor,
    scorer, scorer_name, use_extended, use_vlm=False,
):
    """Process one sample through the full architecture.

    Architecture flow:
        ┌── Branch A: Target Grounding ────┐
        │                                   ├─→ Association → Re-Ranking → Selection
        └── Branch B: Grasp Proposal ──────┘
    """

    # ═════════════════════════════════════════════════════════════════
    #  BLOCK 1 — INPUT
    #  Text Target Description + Scene RGB Image + Depth / Point Cloud
    # ═════════════════════════════════════════════════════════════════
    rgb = load_rgb(scene_meta.scene_dir, sample.view_id, scene_meta.camera_type)
    depth = load_depth(scene_meta.scene_dir, sample.view_id,
                       scene_meta.camera_type, scene_meta.factor_depth)
    label = load_label(scene_meta.scene_dir, sample.view_id, scene_meta.camera_type)
    intrinsics = scene_meta.intrinsics
    text_query = sample.text_query

    # GT reference data (used by GT grounder; ignored by VLM grounder)
    instance_id = sample.target_obj_id + 1

    # ═════════════════════════════════════════════════════════════════
    #  BLOCK 2 — BRANCH A: Target Grounding
    #  Input:  RGB + Text query
    #  Output: target bbox, optional mask, grounding confidence
    #
    #  Options:  gt         — Oracle (GT label mask)
    #            vlm        — Florence-2 open-vocabulary detection
    #            phrase     — Florence-2 phrase grounding
    # ═════════════════════════════════════════════════════════════════
    if use_vlm:
        grounding = grounder.ground(rgb, text_query)
    else:
        grounding = grounder.ground(
            rgb, text_query,
            label=label, instance_id=instance_id,
        )

    if grounding is None:
        return None

    save_stage1_output(sample.sample_id, text_query, grounding)

    # ═════════════════════════════════════════════════════════════════
    #  BLOCK 3 — BRANCH B: Grasp Proposal
    #  Input:  Depth / Point cloud (+ optional target ROI from Branch A)
    #  Output: Candidate grasp poses (position, orientation, score)
    #
    #  Current method:  Target-region local generator (Antipodal sampler)
    #  The target ROI from Branch A guides where to sample grasps.
    # ═════════════════════════════════════════════════════════════════
    if use_vlm:
        # VLM mode: use only VLM bbox for ROI (no GT label leakage)
        candidates = generate_target_grasps(
            depth=depth,
            intrinsics=intrinsics,
            bbox=grounding.bbox,
            label=None,
            instance_id=None,
            top_k=config.GRASP_TOP_K,
        )
    else:
        # GT mode: use instance mask for precise point cloud cropping
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

    # ═════════════════════════════════════════════════════════════════
    #  BLOCK 4 — CANDIDATE–TARGET ASSOCIATION & FEATURES
    #  Merge outputs from Branch A (target info) + Branch B (grasps)
    #  Compute semantic-geometric feature vector per candidate:
    #
    #  Core (5-dim):     f1  grasp score
    #                    f2  centre in target (0/1)
    #                    f3  normalised distance to target centre
    #                    f4  IoU with target region
    #                    f5  VLM confidence
    #
    #  Extended (+4):    f6  depth consistency
    #                    f7  collision risk
    #                    f8  boundary distance
    #                    f9  normal alignment
    # ═════════════════════════════════════════════════════════════════
    points, pixel_coords = backproject_depth(depth, intrinsics)

    if use_vlm:
        # Crop target region by VLM bbox only
        target_pts, _ = crop_point_cloud_by_bbox(
            points, pixel_coords, grounding.bbox
        )
    else:
        # Crop target region by GT instance mask
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

    # ═════════════════════════════════════════════════════════════════
    #  BLOCK 5 — RE-RANKING / SCORING MODULE
    #  Input:  feature vector per candidate
    #  Output: target-aware score (higher = better)
    # ═════════════════════════════════════════════════════════════════
    scores = scorer.score(features)

    # ═════════════════════════════════════════════════════════════════
    #  BLOCK 6 — FINAL GRASP POSE SELECTION
    #  Select Top-K candidates by target-aware score
    #  Output: best grasp pose (gripper position + orientation)
    # ═════════════════════════════════════════════════════════════════
    selections = select_best_grasp(candidates, scores, top_k=5)
    save_selection(sample.sample_id, selections, scorer_name)

    return {
        "sample_id": sample.sample_id,
        "scene_id": sample.scene_id,
        "view_id": sample.view_id,
        "target_class": sample.target_class,
        "text_query": text_query,
        "grounder": "vlm" if use_vlm else "gt",
        "gt_bbox": sample.gt_bbox,
        "pred_bbox": grounding.bbox,
        "grounding_confidence": grounding.confidence,
        "num_candidates": len(candidates),
        "selections": selections,
    }


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VLMGraspPose — Full Architecture Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture Blocks:
  Block 1  INPUT           Text + RGB + Depth
  Block 2  BRANCH A        Target Grounding (--grounder)
  Block 3  BRANCH B        Grasp Proposal (antipodal sampler)
  Block 4  ASSOCIATION     Candidate–Target Features (--extended)
  Block 5  RE-RANKING      Scoring Module (--scorer)
  Block 6  SELECTION       Top-K Grasp Pose Output
  Block 7  EVALUATION      → run: python -m experiments.eval

Examples:
  python -m experiments.run_pipeline --grounder gt --scorer rule
  python -m experiments.run_pipeline --grounder vlm --scorer mlp
  python -m experiments.run_pipeline --grounder phrase --scorer mlp --extended
        """,
    )
    parser.add_argument("--split", type=str, default="test_seen")
    parser.add_argument("--grounder", type=str, default="gt",
                        choices=["gt", "vlm", "phrase"],
                        help="Branch A method: 'gt' (oracle), "
                             "'vlm' (Florence-2 open-vocab), "
                             "'phrase' (Florence-2 phrase grounding)")
    parser.add_argument("--scorer", type=str, default="rule",
                        choices=["rule", "logistic", "mlp"],
                        help="Re-ranking scorer")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--extended", action="store_true",
                        help="Use extended 9-dim features (adds f6–f9)")
    parser.add_argument("--view-stride", type=int, default=config.VIEW_STRIDE)
    args = parser.parse_args()

    run_pipeline(
        split=args.split,
        grounder_name=args.grounder,
        scorer_name=args.scorer,
        max_samples=args.max_samples,
        use_extended_features=args.extended,
        view_stride=args.view_stride,
    )


if __name__ == "__main__":
    main()
