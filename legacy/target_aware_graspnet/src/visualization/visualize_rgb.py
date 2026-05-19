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


def save_rgb_overlay(path: Path, rgb: np.ndarray, target, best_grasp=None, intrinsics=None, top_k=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = rgb.copy()
    if target.mask is not None:
        overlay = img.copy()
        overlay[target.mask.astype(bool)] = [255, 64, 64]
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    if target.bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in target.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if intrinsics is not None:
        grasps = list(top_k or ([] if best_grasp is None else [best_grasp]))
        for grasp in grasps[1:]:
            _draw_grasp_rectangle(img, grasp, intrinsics, color=(0, 90, 255), thickness=2)
        if grasps:
            _draw_grasp_rectangle(img, grasps[0], intrinsics, color=(255, 0, 0), thickness=3)
    if target.command:
        cv2.putText(img, target.command, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
