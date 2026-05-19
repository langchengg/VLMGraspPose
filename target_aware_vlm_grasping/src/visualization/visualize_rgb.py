from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.geometry import grasp_rectangle_2d


def _candidate_from_grasp(grasp):
    return grasp.candidate if hasattr(grasp, "candidate") else grasp


def _draw_grasp_rectangle(img: np.ndarray, grasp, intrinsics: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    candidate = _candidate_from_grasp(grasp)
    rect = grasp_rectangle_2d(
        candidate.position,
        candidate.closing_direction,
        candidate.gripper_width,
        intrinsics,
    )
    pts = np.asarray(rect, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_mask_contour(img: np.ndarray, mask: np.ndarray) -> None:
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    overlay = img.copy()
    overlay[mask.astype(bool)] = [255, 170, 0]
    img[:] = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
    cv2.drawContours(img, contours, -1, (255, 170, 0), 2, lineType=cv2.LINE_AA)


def _draw_dashed_rect(img: np.ndarray, bbox: list[int], color: tuple[int, int, int], thickness: int = 2, dash: int = 10) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    for x_start in range(x1, x2, dash * 2):
        cv2.line(img, (x_start, y1), (min(x_start + dash, x2), y1), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x_start, y2), (min(x_start + dash, x2), y2), color, thickness, lineType=cv2.LINE_AA)
    for y_start in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y_start), (x1, min(y_start + dash, y2)), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y_start), (x2, min(y_start + dash, y2)), color, thickness, lineType=cv2.LINE_AA)


def _draw_legend(img: np.ndarray, target_source: str) -> None:
    items = [
        ("bbox", (0, 255, 0)),
        ("mask", (255, 170, 0)),
        ("top1 grasp", (255, 0, 0)),
        ("topK grasp", (0, 90, 255)),
    ]
    if target_source == "vlm":
        items.insert(1, ("gt bbox", (255, 255, 0)))
    x, y = 20, img.shape[0] - 16
    for label, color in items:
        cv2.line(img, (x, y - 5), (x + 24, y - 5), color, 3, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (x + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        x += 118


def save_rgb_overlay(path: Path, rgb: np.ndarray, target, best_grasp=None, intrinsics=None, top_k=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = rgb.copy()
    if target.mask is not None:
        _draw_mask_contour(img, target.mask.astype(bool))
    if target.bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in target.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    target_bbox_gt = getattr(target, "metadata", {}).get("target_bbox_gt")
    if target.target_source == "vlm" and target_bbox_gt:
        _draw_dashed_rect(img, target_bbox_gt, (255, 255, 0), thickness=2)
    if intrinsics is not None:
        grasps = list(top_k or ([] if best_grasp is None else [best_grasp]))
        for grasp in grasps[1:]:
            _draw_grasp_rectangle(img, grasp, intrinsics, color=(0, 90, 255), thickness=2)
        if grasps:
            _draw_grasp_rectangle(img, grasps[0], intrinsics, color=(255, 0, 0), thickness=3)
    if target.command:
        cv2.putText(img, target.command, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    _draw_legend(img, getattr(target, "target_source", ""))
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
