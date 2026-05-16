from __future__ import annotations

import numpy as np
import cv2


def is_proxy_valid(feature: dict, thresholds: dict) -> bool:
    return (
        feature.get("target_overlap", 0.0) > thresholds.get("target_overlap", 0.10)
        and feature.get("center_alignment", 0.0) > thresholds.get("center_alignment", 0.30)
        and feature.get("collision_penalty", 1.0) < thresholds.get("collision_penalty", 0.50)
        and feature.get("depth_stability", 0.0) > thresholds.get("depth_stability", 0.30)
        and feature.get("gripper_width_match", 0.0) > thresholds.get("gripper_width_match", 0.30)
    )


def topk_valid_rate(records: list[dict], k: int, thresholds: dict) -> float:
    if not records:
        return 0.0
    ok = 0
    for rec in records:
        candidates = [rec.get("feature_breakdown", {})] + [
            c.get("features", {}) for c in rec.get("top_k_fallback_candidates", [])[: max(k - 1, 0)]
        ]
        ok += any(is_proxy_valid(c, thresholds) for c in candidates[:k])
    return ok / len(records)


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def point_in_grasp_rectangles(point: list[float] | None, rectangles: list) -> bool:
    if point is None or not rectangles:
        return False
    pt = (float(point[0]), float(point[1]))
    for rect in rectangles:
        arr = np.asarray(rect, dtype=np.float32)
        if arr.shape == (4, 2) and cv2.pointPolygonTest(arr, pt, False) >= 0:
            return True
    return False


def topk_2d_grasp_center_hit_rate(records: list[dict], k: int) -> float:
    if not records:
        return 0.0
    hits = 0
    for rec in records:
        rectangles = rec.get("gt_grasp_rectangles", [])
        centers = rec.get("top_k_grasp_centers_2d") or [rec.get("best_grasp_center_2d")]
        hits += any(point_in_grasp_rectangles(center, rectangles) for center in centers[:k])
    return hits / len(records)
