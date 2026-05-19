"""
vis/compare_gt.py — Compare Top-1 Grasp with Ground Truth
===========================================================
Computes quantitative metrics between the predicted top-1 grasp
and the GT target object pose, then generates a comparison figure.

Metrics:
  • Position error — L2 distance between grasp centre and GT object centre
  • Angular error — rotation angle between grasp approach axis and
                    ideal approach direction (camera-to-object)
  • Target hit — whether the grasp centre projects inside GT mask/bbox
  • Width ratio — predicted width / GT object bounding diameter

Usage:
    python -m vis.compare_gt --sample <sample_id> --grounder phrase --reranker rule
    python -m vis.compare_gt --grounder phrase --reranker rule
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import (
    load_rgb, load_depth, load_label,
    load_camera_intrinsics, get_factor_depth,
    bbox_from_mask,
)
from src.point_cloud import (
    backproject_depth, crop_point_cloud_by_mask,
    crop_point_cloud_by_bbox, compute_target_center,
)
from vis.grasp_drawing import to_rotation_matrix, rank_colour


# ── Single-Sample Comparison ────────────────────────────────────────

def compare_single(
    pred: dict,
) -> Optional[Dict]:
    """Compute comparison metrics for one prediction dict.

    Expects pred to have: ranked_grasps, scene_id, frame_id, etc.
    """
    if not pred.get("ranked_grasps"):
        return None

    top1 = pred["ranked_grasps"][0]
    sample_id = pred["sample_id"]

    # Prefer explicit dict fields; fall back to sample_id parsing for legacy
    scene_id = pred.get("scene_id")
    camera = pred.get("camera")
    frame_id = pred.get("frame_id")
    if scene_id is None or camera is None or frame_id is None:
        parts = sample_id.split("_")
        scene_id = int(parts[1])
        camera = parts[2]
        frame_id = int(parts[3])

    scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"
    if not scene_dir.exists():
        return None

    try:
        factor = get_factor_depth(scene_dir, camera)
        depth = load_depth(scene_dir, frame_id, camera, factor)
        label = load_label(scene_dir, frame_id, camera)
        K = load_camera_intrinsics(scene_dir, camera)
    except Exception:
        return None

    # Back-project full scene
    points, pixel_coords = backproject_depth(depth, K)

    # ── GT target definition ──────────────────────────────────────
    # Use the GT label image to identify the real target region.
    # target_object_id should be saved by step10 in the prediction dict.
    target_mask_val = pred.get("target_object_id")
    if target_mask_val is not None:
        target_mask_val = target_mask_val + 1  # GraspNet convention: label = obj_id + 1

    gt_bbox = None
    if target_mask_val is not None:
        gt_bbox = bbox_from_mask(label, target_mask_val)
        target_pts, _ = crop_point_cloud_by_mask(
            points, pixel_coords, label, target_mask_val,
        )
    else:
        target_pts = np.zeros((0, 3))

    # Fallback: if GT mask gives too few points, use pred_bbox
    if len(target_pts) < 5:
        fallback_bbox = gt_bbox if gt_bbox is not None else pred.get("pred_bbox")
        if fallback_bbox is not None:
            target_pts, _ = crop_point_cloud_by_bbox(
                points, pixel_coords, fallback_bbox,
            )

    if len(target_pts) < 5:
        return None

    gt_center_3d = compute_target_center(target_pts)
    gt_diameter = np.ptp(target_pts, axis=0).max() if len(target_pts) > 0 else 0.05

    # ── Metrics ──────────────────────────────────────────────────────

    grasp_pos = np.array(top1["position"])
    grasp_rot = top1["rotation"]
    grasp_width = top1["width"]

    # 1. Position error (L2 distance to GT object centre)
    pos_error = float(np.linalg.norm(grasp_pos - gt_center_3d))

    # 2. Angular error
    R = to_rotation_matrix(grasp_rot)
    approach = R[:, 2]  # z-axis = approach direction
    ideal_approach = gt_center_3d / (np.linalg.norm(gt_center_3d) + 1e-8)
    cos_angle = np.clip(np.dot(approach, ideal_approach), -1.0, 1.0)
    angular_error = float(np.degrees(np.arccos(abs(cos_angle))))

    # 3. Target hit — does grasp centre project inside GT mask / bbox?
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = max(grasp_pos[2], 1e-6)
    u = int(round(grasp_pos[0] * fx / z + cx))
    v = int(round(grasp_pos[1] * fy / z + cy))

    # Mask hit (GT label)
    H, W = label.shape[:2]
    target_hit_mask = False
    if target_mask_val is not None and 0 <= u < W and 0 <= v < H:
        target_hit_mask = bool(label[v, u] == target_mask_val)

    # Bbox hit (GT bbox derived from GT label)
    bbox_hit = False
    if gt_bbox is not None:
        x1, y1, x2, y2 = gt_bbox
        bbox_hit = (x1 <= u <= x2) and (y1 <= v <= y2)

    # 4. Width ratio
    width_ratio = grasp_width / max(gt_diameter, 1e-6)

    # 5. Normalised position error
    normalised_pos_error = pos_error / max(gt_diameter, 1e-6)

    return {
        "sample_id": sample_id,
        "target_class": pred.get("target_class", "unknown"),
        "pos_error_m": round(pos_error, 5),
        "pos_error_normalised": round(normalised_pos_error, 3),
        "angular_error_deg": round(angular_error, 2),
        "target_hit_mask": target_hit_mask,
        "target_hit_bbox": bbox_hit,
        "is_on_target": top1.get("is_on_target", False),
        "width_ratio": round(width_ratio, 3),
        "rerank_score": top1.get("rerank_score", 0),
        "gt_center": gt_center_3d.tolist(),
        "grasp_center": grasp_pos.tolist(),
        "gt_diameter": round(gt_diameter, 5),
    }


# ── Batch Comparison Report ──────────────────────────────────────────

def batch_compare(
    grounder: str = "phrase",
    reranker: str = "rule",
    max_samples: int = None,
    output_dir: Path = None,
) -> Dict:
    """Run comparison on all available predictions.

    Returns aggregate statistics.
    """
    pred_files = glob.glob(
        str(config.RESULTS_DIR / f"predictions_*_{grounder}_{reranker}.json")
    )

    if not pred_files:
        print(f"[WARN] No predictions found for {grounder}+{reranker}")
        return {}

    all_preds = []
    for pf in pred_files:
        with open(pf) as f:
            all_preds.extend(json.load(f))

    all_metrics = []
    for pred in all_preds:
        metrics = compare_single(pred)
        if metrics is not None:
            all_metrics.append(metrics)
        if max_samples and len(all_metrics) >= max_samples:
            break

    if not all_metrics:
        print("[WARN] No valid samples for comparison.")
        return {}

    # Aggregate stats
    pos_errors = [m["pos_error_m"] for m in all_metrics]
    ang_errors = [m["angular_error_deg"] for m in all_metrics]
    bbox_hits = [m["target_hit_bbox"] for m in all_metrics]
    on_target = [m["is_on_target"] for m in all_metrics]
    width_ratios = [m["width_ratio"] for m in all_metrics]

    summary = {
        "num_samples": len(all_metrics),
        "grounder": grounder,
        "reranker": reranker,
        "position_error_cm": {
            "mean": round(np.mean(pos_errors) * 100, 3),
            "std": round(np.std(pos_errors) * 100, 3),
            "median": round(np.median(pos_errors) * 100, 3),
        },
        "angular_error_deg": {
            "mean": round(np.mean(ang_errors), 2),
            "std": round(np.std(ang_errors), 2),
        },
        "target_hit_rate_bbox": round(np.mean(bbox_hits), 4),
        "on_target_rate": round(np.mean(on_target), 4),
        "width_ratio": {
            "mean": round(np.mean(width_ratios), 3),
            "std": round(np.std(width_ratios), 3),
        },
        "per_sample": all_metrics,
    }

    # Save report
    if output_dir is None:
        output_dir = config.PROJECT_ROOT / "vis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"comparison_report_{grounder}_{reranker}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*55}")
    print(f" Grasp vs GT Comparison  ({grounder}+{reranker})")
    print(f"{'='*55}")
    print(f" Samples:          {summary['num_samples']}")
    print(f" Position Error:   {summary['position_error_cm']['mean']:.2f} ± "
          f"{summary['position_error_cm']['std']:.2f} cm")
    print(f" Angular Error:    {summary['angular_error_deg']['mean']:.1f} ± "
          f"{summary['angular_error_deg']['std']:.1f} deg")
    print(f" Target Hit (bbox): {summary['target_hit_rate_bbox']*100:.1f}%")
    print(f" On-Target Rate:   {summary['on_target_rate']*100:.1f}%")
    print(f" Width Ratio:      {summary['width_ratio']['mean']:.2f} ± "
          f"{summary['width_ratio']['std']:.2f}")
    print(f"{'='*55}")
    print(f" Report: {report_path}")

    return summary


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare grasp with GT")
    parser.add_argument("--sample", type=str, default=None,
                        help="Single sample ID to compare")
    parser.add_argument("--grounder", type=str, default="phrase")
    parser.add_argument("--reranker", type=str, default="rule")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.sample:
        # Single sample — find in predictions
        pred_files = glob.glob(
            str(config.RESULTS_DIR
                / f"predictions_*_{args.grounder}_{args.reranker}.json")
        )
        pred = None
        for pf in pred_files:
            with open(pf) as f:
                preds = json.load(f)
            for p in preds:
                if p.get("sample_id") == args.sample:
                    pred = p
                    break
            if pred:
                break

        if pred:
            metrics = compare_single(pred)
            if metrics:
                print(json.dumps(metrics, indent=2))
            else:
                print(f"[ERROR] Could not compute metrics for {args.sample}")
        else:
            print(f"[ERROR] Sample {args.sample} not found in predictions")
    else:
        batch_compare(
            grounder=args.grounder,
            reranker=args.reranker,
            max_samples=args.max_samples,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()
