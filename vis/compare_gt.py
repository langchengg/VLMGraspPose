"""
vis/compare_gt.py — Compare Top-1 Grasp with Ground Truth
===========================================================
Computes quantitative metrics between the predicted top-1 grasp
and the GT target object pose, then generates a comparison figure.

Metrics:
  • Position error — L2 distance between grasp centre and GT object centre
  • Angular error — rotation angle between grasp approach axis and
                    ideal approach direction (camera-to-object)
  • Target hit — whether the grasp centre projects inside GT mask
  • Width ratio — predicted width / GT object bounding diameter

Usage:
    python -m vis.compare_gt --sample scene_0100_0000_012_strawberry
    python -m vis.compare_gt --split test_seen --scorer rule
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.dataset import (
    load_scene, load_rgb, load_depth, load_label,
    bbox_from_mask, discover_scenes, generate_samples,
)
from data.point_cloud import (
    backproject_depth, crop_point_cloud_by_mask,
    crop_point_cloud_by_bbox, compute_target_center,
)
from vis.grasp_drawing import quat_to_rotation_matrix, rank_colour


# ── Grasp Output Explanation ────────────────────────────────────────
#
# The pipeline outputs a 6-DoF parallel-jaw gripper pose:
#
#   position:    [x, y, z]           — grasp centre in camera frame (metres)
#   orientation: [qx, qy, qz, qw]   — quaternion encoding gripper rotation
#                                      x-axis = closing direction
#                                      z-axis = approach direction
#   width:       float               — gripper opening width (metres)
#
# The quaternion encodes a full SO(3) rotation R:
#   R[:, 0] = x-axis = closing direction (finger-to-finger)
#   R[:, 1] = y-axis = binormal (orthogonal to fingers and approach)
#   R[:, 2] = z-axis = approach direction (wrist → object)
#


# ── Single-Sample Comparison ────────────────────────────────────────

def compare_single(
    sample_id: str,
    scorer: str = "rule",
) -> Optional[Dict]:
    """Compute comparison metrics for one sample.

    Returns a dict with metrics or None if data is missing.
    """
    # Parse sample_id
    parts = sample_id.split("_")
    scene_id = f"{parts[0]}_{parts[1]}"
    view_id = int(parts[2])
    obj_name = "_".join(parts[3:])

    scene_dir = config.DATA_DIRS["test_seen"] / scene_id
    if not scene_dir.exists():
        return None

    # Load result
    result_path = config.PROJECT_ROOT / "results" / f"{sample_id}_{scorer}.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        result = json.load(f)
    if not result.get("selections"):
        return None

    top1 = result["selections"][0]

    # Load scene data
    scene_meta = load_scene(scene_dir)
    depth = load_depth(scene_dir, view_id, scene_meta.camera_type,
                       scene_meta.factor_depth)
    label = load_label(scene_dir, view_id, scene_meta.camera_type)
    intrinsics = scene_meta.intrinsics

    # Find target object
    target_obj = None
    for obj in scene_meta.objects:
        if obj.obj_name == obj_name:
            target_obj = obj
            break
    if target_obj is None:
        return None

    instance_id = target_obj.obj_id + 1

    # GT target info
    gt_bbox = bbox_from_mask(label, instance_id)
    if gt_bbox is None:
        return None

    # Target point cloud → GT centre in camera frame
    points, pixel_coords = backproject_depth(depth, intrinsics)
    target_pts, _ = crop_point_cloud_by_mask(points, pixel_coords,
                                              label, instance_id)
    if len(target_pts) < 5:
        target_pts, _ = crop_point_cloud_by_bbox(points, pixel_coords, gt_bbox)

    gt_center_3d = compute_target_center(target_pts)
    gt_diameter = np.ptp(target_pts, axis=0).max() if len(target_pts) > 0 else 0.05

    # ── Metrics ──────────────────────────────────────────────────────

    grasp_pos = np.array(top1["position"])
    grasp_ori = top1["orientation"]
    grasp_width = top1["width"]

    # 1. Position error (L2 distance)
    pos_error = float(np.linalg.norm(grasp_pos - gt_center_3d))

    # 2. Angular error
    R = quat_to_rotation_matrix(grasp_ori)
    approach = R[:, 2]  # z-axis = approach direction
    # Ideal approach: from camera to object centre
    ideal_approach = gt_center_3d / (np.linalg.norm(gt_center_3d) + 1e-8)
    cos_angle = np.clip(np.dot(approach, ideal_approach), -1.0, 1.0)
    # We consider both approach directions valid (±180° symmetry)
    angular_error = float(np.degrees(np.arccos(abs(cos_angle))))

    # 3. Target hit — does grasp centre project inside GT mask?
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = max(grasp_pos[2], 1e-6)
    u = int(round(grasp_pos[0] * fx / z + cx))
    v = int(round(grasp_pos[1] * fy / z + cy))

    H, W = label.shape
    if 0 <= u < W and 0 <= v < H:
        target_hit = bool(label[v, u] == instance_id)
    else:
        target_hit = False

    # Also check bbox hit
    x1, y1, x2, y2 = gt_bbox
    bbox_hit = (x1 <= u <= x2) and (y1 <= v <= y2)

    # 4. Width ratio
    width_ratio = grasp_width / max(gt_diameter, 1e-6)

    # 5. Normalised position error (by object diameter)
    normalised_pos_error = pos_error / max(gt_diameter, 1e-6)

    return {
        "sample_id": sample_id,
        "target_class": target_obj.friendly_name,
        "pos_error_m": round(pos_error, 5),
        "pos_error_normalised": round(normalised_pos_error, 3),
        "angular_error_deg": round(angular_error, 2),
        "target_hit_mask": target_hit,
        "target_hit_bbox": bbox_hit,
        "width_ratio": round(width_ratio, 3),
        "final_score": top1["final_score"],
        "gt_center": gt_center_3d.tolist(),
        "grasp_center": grasp_pos.tolist(),
        "gt_diameter": round(gt_diameter, 5),
    }


# ── Comparison Figure ────────────────────────────────────────────────

def draw_comparison_figure(
    sample_id: str,
    metrics: Dict,
    scorer: str = "rule",
    output_dir: Path = None,
) -> Path:
    """Generate a side-by-side comparison image.

    Left panel: RGB with GT bbox + Top-1 grasp
    Right panel: Metrics table
    """
    from vis.vis_2d import draw_bbox, draw_grasp_2d

    # Parse and load
    parts = sample_id.split("_")
    scene_id = f"{parts[0]}_{parts[1]}"
    view_id = int(parts[2])
    obj_name = "_".join(parts[3:])

    scene_dir = config.DATA_DIRS["test_seen"] / scene_id
    scene_meta = load_scene(scene_dir)
    rgb = load_rgb(scene_dir, view_id, scene_meta.camera_type)
    label = load_label(scene_dir, view_id, scene_meta.camera_type)
    intrinsics = scene_meta.intrinsics

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Find target
    target_obj = None
    for obj in scene_meta.objects:
        if obj.obj_name == obj_name:
            target_obj = obj
            break

    # Draw GT bbox
    if target_obj:
        mask_val = target_obj.obj_id + 1
        gt_bbox = bbox_from_mask(label, mask_val)
        if gt_bbox:
            img = draw_bbox(img, gt_bbox,
                            label=f"GT: {target_obj.friendly_name}",
                            colour=(0, 220, 0), thickness=2)

    # Draw GT object center projection
    gc = metrics["gt_center"]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = max(gc[2], 1e-6)
    gt_u = int(round(gc[0] * fx / z + cx))
    gt_v = int(round(gc[1] * fy / z + cy))
    cv2.drawMarker(img, (gt_u, gt_v), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(img, "GT Centre", (gt_u + 12, gt_v - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # Load top-1 grasp and draw
    result_path = config.PROJECT_ROOT / "results" / f"{sample_id}_{scorer}.json"
    if result_path.exists():
        with open(result_path) as f:
            result = json.load(f)
        top1 = result["selections"][0]
        img = draw_grasp_2d(
            img, top1["position"], top1["orientation"], top1["width"],
            intrinsics, colour=(0, 255, 0), thickness=3,
            label=f"Top-1 ({top1['final_score']:.2f})",
        )

        # Draw line from grasp centre to GT centre
        gp = top1["position"]
        z2 = max(gp[2], 1e-6)
        g_u = int(round(gp[0] * fx / z2 + cx))
        g_v = int(round(gp[1] * fy / z2 + cy))
        cv2.arrowedLine(img, (gt_u, gt_v), (g_u, g_v),
                        (0, 180, 255), 1, cv2.LINE_AA, tipLength=0.15)

    # ── Create metrics panel ─────────────────────────────────────────
    H, W = img.shape[:2]
    panel_w = 400
    panel = np.full((H, panel_w, 3), 25, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 40
    dy = 30

    def _put(text, val, colour=(220, 220, 220)):
        nonlocal y
        cv2.putText(panel, text, (15, y), font, 0.5, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(panel, str(val), (220, y), font, 0.5, colour, 1, cv2.LINE_AA)
        y += dy

    cv2.putText(panel, "Grasp vs GT Comparison", (15, y),
                font, 0.65, (0, 220, 180), 1, cv2.LINE_AA)
    y += dy + 5

    _put("Target:", metrics["target_class"], (255, 255, 255))
    _put("Position Error:", f'{metrics["pos_error_m"]*100:.2f} cm')
    _put("Norm. Pos Error:", f'{metrics["pos_error_normalised"]:.2f}x diameter')
    _put("Angular Error:", f'{metrics["angular_error_deg"]:.1f} deg')

    hit_col = (0, 255, 0) if metrics["target_hit_mask"] else (0, 0, 255)
    _put("Target Hit (mask):", str(metrics["target_hit_mask"]), hit_col)

    hit_col2 = (0, 255, 0) if metrics["target_hit_bbox"] else (0, 0, 255)
    _put("Target Hit (bbox):", str(metrics["target_hit_bbox"]), hit_col2)

    _put("Width Ratio:", f'{metrics["width_ratio"]:.2f}')
    _put("Final Score:", f'{metrics["final_score"]:.3f}')
    _put("GT Diameter:", f'{metrics["gt_diameter"]*100:.2f} cm')

    y += 20
    cv2.putText(panel, "Output Format", (15, y),
                font, 0.6, (0, 180, 220), 1, cv2.LINE_AA)
    y += dy
    cv2.putText(panel, "position: [x,y,z] camera frame", (15, y),
                font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, "orientation: [qx,qy,qz,qw] quat", (15, y),
                font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, "  x-axis = closing direction", (15, y),
                font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, "  z-axis = approach direction", (15, y),
                font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, "width: gripper opening (m)", (15, y),
                font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    # Combine
    combined = np.hstack([img, panel])

    # Save
    if output_dir is None:
        output_dir = config.PROJECT_ROOT / "vis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sample_id}_compare.png"
    cv2.imwrite(str(out_path), combined)
    print(f"[COMPARE] Saved → {out_path}")

    return out_path


# ── Batch Comparison Report ──────────────────────────────────────────

def batch_compare(
    split: str = "test_seen",
    scorer: str = "rule",
    max_samples: int = None,
    output_dir: Path = None,
) -> Dict:
    """Run comparison on all available results in a split.

    Returns aggregate statistics.
    """
    results_dir = config.PROJECT_ROOT / "results"
    pattern = f"*_{scorer}.json"
    result_files = sorted(results_dir.glob(pattern))

    if not result_files:
        print(f"[WARN] No results found matching {pattern}")
        return {}

    all_metrics = []
    for rf in result_files:
        sample_id = rf.name.replace(f"_{scorer}.json", "")
        # Skip summary files
        if "pipeline_summary" in sample_id:
            continue

        metrics = compare_single(sample_id, scorer)
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
    mask_hits = [m["target_hit_mask"] for m in all_metrics]
    bbox_hits = [m["target_hit_bbox"] for m in all_metrics]
    width_ratios = [m["width_ratio"] for m in all_metrics]

    summary = {
        "num_samples": len(all_metrics),
        "scorer": scorer,
        "position_error_cm": {
            "mean": round(np.mean(pos_errors) * 100, 3),
            "std": round(np.std(pos_errors) * 100, 3),
            "median": round(np.median(pos_errors) * 100, 3),
            "max": round(np.max(pos_errors) * 100, 3),
        },
        "angular_error_deg": {
            "mean": round(np.mean(ang_errors), 2),
            "std": round(np.std(ang_errors), 2),
            "median": round(np.median(ang_errors), 2),
        },
        "target_hit_rate_mask": round(np.mean(mask_hits), 4),
        "target_hit_rate_bbox": round(np.mean(bbox_hits), 4),
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
    report_path = output_dir / f"comparison_report_{scorer}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*55}")
    print(f" Grasp vs GT Comparison Report  (scorer={scorer})")
    print(f"{'='*55}")
    print(f" Samples:             {summary['num_samples']}")
    print(f" Position Error:      {summary['position_error_cm']['mean']:.2f} ± "
          f"{summary['position_error_cm']['std']:.2f} cm")
    print(f" Angular Error:       {summary['angular_error_deg']['mean']:.1f} ± "
          f"{summary['angular_error_deg']['std']:.1f} deg")
    print(f" Target Hit (mask):   {summary['target_hit_rate_mask']*100:.1f}%")
    print(f" Target Hit (bbox):   {summary['target_hit_rate_bbox']*100:.1f}%")
    print(f" Width Ratio:         {summary['width_ratio']['mean']:.2f} ± "
          f"{summary['width_ratio']['std']:.2f}")
    print(f"{'='*55}")
    print(f" Report: {report_path}")

    return summary


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare grasp with GT")
    parser.add_argument("--sample", type=str, default=None,
                        help="Single sample ID to compare")
    parser.add_argument("--split", type=str, default="test_seen",
                        help="Run batch comparison on a split")
    parser.add_argument("--scorer", type=str, default="rule")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--draw", action="store_true",
                        help="Draw comparison figure(s)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.sample:
        # Single sample
        metrics = compare_single(args.sample, args.scorer)
        if metrics:
            print(json.dumps(metrics, indent=2))
            if args.draw:
                draw_comparison_figure(args.sample, metrics,
                                        args.scorer, out_dir)
        else:
            print(f"[ERROR] Could not compute metrics for {args.sample}")
    else:
        # Batch
        summary = batch_compare(args.split, args.scorer,
                                args.max_samples, out_dir)

        # Draw first few samples for visual inspection
        if args.draw and summary.get("per_sample"):
            n_draw = min(10, len(summary["per_sample"]))
            for m in summary["per_sample"][:n_draw]:
                draw_comparison_figure(
                    m["sample_id"], m, args.scorer, out_dir
                )


if __name__ == "__main__":
    main()
