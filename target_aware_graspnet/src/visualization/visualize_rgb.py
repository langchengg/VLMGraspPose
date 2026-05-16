from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_rgb_overlay(path: Path, rgb: np.ndarray, target, best_grasp=None, intrinsics=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = rgb.copy()
    if target.mask is not None:
        overlay = img.copy()
        overlay[target.mask.astype(bool)] = [255, 64, 64]
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    if target.bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in target.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if target.command:
        cv2.putText(img, target.command, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
