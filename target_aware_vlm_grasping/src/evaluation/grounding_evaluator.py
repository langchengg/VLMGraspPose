from __future__ import annotations

import numpy as np


def bbox_iou(a: list[int] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1 + 1.0), max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0)
    area_b = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    aa = np.asarray(a).astype(bool)
    bb = np.asarray(b).astype(bool)
    if aa.shape != bb.shape:
        return 0.0
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    return float(inter / union) if union else 0.0


class GroundingEvaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate_record(self, record: dict) -> dict:
        iou = bbox_iou(record.get("target_bbox_pred"), record.get("target_bbox_gt"))
        return {
            "bbox_iou": iou,
            "grounding_success": float(iou >= self.iou_threshold),
            "target_source": record.get("target_source"),
        }
