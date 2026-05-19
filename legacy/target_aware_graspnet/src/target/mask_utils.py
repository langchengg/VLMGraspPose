from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def bbox_to_mask(bbox: list[int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = shape
    x1, x2 = np.clip([x1, x2], 0, W - 1)
    y1, y2 = np.clip([y1, y2], 0, H - 1)
    mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def compute_mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def clean_binary_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8.astype(bool)


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(str(path)).astype(bool)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]
    return img > 0
