"""
vis/vis_2d.py — 2D Visualisation on RGB Images
================================================
Overlay bounding boxes, grasp candidates, and GT comparison
onto RGB images from the GraspNet dataset.

Usage:
    python -m vis.vis_2d --sample <sample_id> --grounder phrase --reranker rule
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
from src.data_utils import load_rgb, load_label, load_camera_intrinsics, bbox_from_mask
from vis.grasp_drawing import (
    project_gripper_to_image, score_to_colour, rank_colour,
)


# ── Draw Bounding Box ───────────────────────────────────────────────

def draw_bbox(
    img: np.ndarray,
    bbox: List[int],
    label: str = "",
    colour: tuple = (0, 255, 0),
    thickness: int = 2,
    fill_alpha: float = 0.12,
) -> np.ndarray:
    """Draw a bounding box with semi-transparent fill and label."""
    x1, y1, x2, y2 = bbox
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    img = cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ── Draw Single Grasp Candidate ─────────────────────────────────────

def draw_grasp_2d(
    img: np.ndarray,
    position: List[float],
    rotation: List[float],
    width: float,
    intrinsics: np.ndarray,
    colour: tuple = (0, 255, 0),
    thickness: int = 2,
    label: str = "",
) -> np.ndarray:
    """Draw a gripper silhouette on the image.

    Draws:  left-finger ── palm-bar ── right-finger + approach line
    """
    pts_2d = project_gripper_to_image(position, rotation, width, intrinsics)
    pts = pts_2d.astype(np.int32)

    # Check if points are within reasonable image bounds
    H, W = img.shape[:2]
    if np.any(pts[:, 0] < -500) or np.any(pts[:, 0] > W + 500):
        return img
    if np.any(pts[:, 1] < -500) or np.any(pts[:, 1] > H + 500):
        return img

    # Left finger: 0─1
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), colour, thickness, cv2.LINE_AA)
    # Right finger: 2─3
    cv2.line(img, tuple(pts[2]), tuple(pts[3]), colour, thickness, cv2.LINE_AA)
    # Palm bar: 4─5  (= 0─3)
    cv2.line(img, tuple(pts[4]), tuple(pts[5]), colour, thickness, cv2.LINE_AA)
    # Wrist line: mid(4,5) → 6
    mid = ((pts[4] + pts[5]) // 2)
    cv2.line(img, tuple(mid), tuple(pts[6]), colour, max(1, thickness - 1), cv2.LINE_AA)

    # Draw centre dot
    centre_2d = ((pts[4] + pts[5]) // 2)
    cv2.circle(img, tuple(centre_2d), 3, colour, -1, cv2.LINE_AA)

    # Label
    if label:
        cv2.putText(img, label, (centre_2d[0] + 5, centre_2d[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

    return img


# ── Draw Multiple Candidates ────────────────────────────────────────

def draw_all_candidates(
    img: np.ndarray,
    candidates: List[Dict],
    intrinsics: np.ndarray,
    max_draw: int = 20,
    show_scores: bool = True,
) -> np.ndarray:
    """Draw all grasp candidates with colour-coded scores."""
    for i, c in enumerate(candidates[:max_draw]):
        score = c.get("rerank_score", c.get("grasp_score",
                 c.get("detector_score", 0.5)))
        bgr = score_to_colour(score)[:3]
        label = f"{score:.2f}" if show_scores else ""
        img = draw_grasp_2d(
            img, c["position"], c["rotation"], c["width"],
            intrinsics, colour=bgr, thickness=1 if i > 4 else 2,
            label=label,
        )
    return img


def draw_top_k(
    img: np.ndarray,
    selections: List[Dict],
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Draw the top-K selected grasps with rank colours."""
    for sel in selections:
        rank = sel["rank"]
        colour = rank_colour(rank)
        score = sel.get("rerank_score", sel.get("grasp_score", 0))
        label = f"#{rank} ({score:.2f})"
        img = draw_grasp_2d(
            img, sel["position"], sel["rotation"], sel["width"],
            intrinsics, colour=colour, thickness=3 if rank == 1 else 2,
            label=label,
        )
    return img


# ── Main Visualisation Function ─────────────────────────────────────

def visualise_sample(
    sample_id: str,
    grounder: str = "phrase",
    reranker: str = "rule",
    output_dir: Path = None,
    show_all_candidates: bool = False,
) -> np.ndarray:
    """Create a complete 2D visualisation for one pipeline sample.

    Reads from:
      - Raw GraspNet scene data (src.data_utils)
      - Prediction JSONs in results/ (step10 output)
    """
    # Parse sample_id to extract scene info
    # sample_id format depends on step02, e.g.:
    #   scene_0100_realsense_0012_mug
    parts = sample_id.split("_")
    scene_id = int(parts[1])
    camera = parts[2]
    frame_id = int(parts[3])

    scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    rgb = load_rgb(scene_dir, frame_id, camera)
    K = load_camera_intrinsics(scene_dir, camera)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Try to find prediction file
    pred_files = (
        glob.glob(str(config.RESULTS_DIR / f"predictions_*_{grounder}_{reranker}_*.json"))
        + glob.glob(str(config.RESULTS_DIR / f"predictions_*_{grounder}_{reranker}.json"))
    )

    pred = None
    for pf in pred_files:
        with open(pf) as f:
            preds = json.load(f)
        for p in preds:
            if p.get("sample_id") == sample_id:
                pred = p
                break
        if pred:
            break

    # Draw GT bbox if we can compute it
    gt_label = load_label(scene_dir, frame_id, camera)
    if pred and pred.get("text_query"):
        # Try to find target object from the query
        obj_id = pred.get("target_object_id")
        if obj_id is not None:
            mask_val = obj_id + 1
            gt_bbox = bbox_from_mask(gt_label, mask_val)
            if gt_bbox is not None:
                obj_name = pred.get("target_class", f"obj_{obj_id}")
                img = draw_bbox(img, gt_bbox,
                                label=f"GT: {obj_name}",
                                colour=(0, 200, 0), thickness=2)

    # Draw predicted bbox
    if pred and pred.get("pred_bbox"):
        img = draw_bbox(img, pred["pred_bbox"],
                        label=f"Pred ({grounder})",
                        colour=(255, 180, 0), thickness=2)

    # Draw grasps
    if pred and pred.get("ranked_grasps"):
        if show_all_candidates:
            img = draw_all_candidates(img, pred["ranked_grasps"], K, max_draw=30)
        else:
            img = draw_top_k(img, pred["ranked_grasps"], K)

    # Add title bar
    title = f"{sample_id}  |  {grounder}+{reranker}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(img, title, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Add legend
    y0 = 40
    for rank in range(1, 6):
        col = rank_colour(rank)
        cv2.rectangle(img, (img.shape[1] - 130, y0), (img.shape[1] - 110, y0 + 12), col, -1)
        cv2.putText(img, f"Rank {rank}", (img.shape[1] - 105, y0 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        y0 += 18

    # Save
    if output_dir is None:
        output_dir = config.PROJECT_ROOT / "vis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sample_id}_2d.png"
    cv2.imwrite(str(out_path), img)
    print(f"[VIS-2D] Saved → {out_path}")

    return img


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2D Grasp Visualisation")
    parser.add_argument("--sample", type=str, required=True,
                        help="Sample ID from predictions JSON")
    parser.add_argument("--grounder", type=str, default="phrase")
    parser.add_argument("--reranker", type=str, default="rule")
    parser.add_argument("--show-all", action="store_true",
                        help="Draw all candidates (not just top-K)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    visualise_sample(
        sample_id=args.sample,
        grounder=args.grounder,
        reranker=args.reranker,
        output_dir=out_dir,
        show_all_candidates=args.show_all,
    )


if __name__ == "__main__":
    main()
