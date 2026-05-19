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


def _polygon(rect: list) -> np.ndarray:
    arr = np.asarray(rect, dtype=np.float32)
    if arr.shape != (4, 2):
        return np.zeros((0, 2), dtype=np.float32)
    hull = cv2.convexHull(arr)
    return hull.reshape(-1, 2).astype(np.float32)


def rectangle_iou(rect_a: list, rect_b: list) -> float:
    poly_a = _polygon(rect_a)
    poly_b = _polygon(rect_b)
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    area_a = abs(float(cv2.contourArea(poly_a)))
    area_b = abs(float(cv2.contourArea(poly_b)))
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    union = area_a + area_b - float(intersection)
    if union <= 1e-6:
        return 0.0
    return float(np.clip(float(intersection) / union, 0.0, 1.0))


def rectangle_angle_deg(rect: list) -> float:
    poly = _polygon(rect)
    if len(poly) != 4:
        return 0.0
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    return float(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)


def rectangle_angle_difference_deg(rect_a: list, rect_b: list) -> float:
    diff = abs(rectangle_angle_deg(rect_a) - rectangle_angle_deg(rect_b)) % 180.0
    return float(min(diff, 180.0 - diff))


def grasp_rectangle_matches(
    pred_rect: list,
    gt_rect: list,
    iou_threshold: float = 0.25,
    angle_threshold_deg: float = 30.0,
) -> bool:
    return (
        rectangle_iou(pred_rect, gt_rect) >= iou_threshold
        and rectangle_angle_difference_deg(pred_rect, gt_rect) <= angle_threshold_deg
    )


def topk_2d_grasp_rectangle_match_rate(
    records: list[dict],
    k: int,
    iou_threshold: float = 0.25,
    angle_threshold_deg: float = 30.0,
) -> float:
    if not records:
        return 0.0
    hits = 0
    for rec in records:
        gt_rectangles = rec.get("gt_grasp_rectangles", [])
        pred_rectangles = rec.get("top_k_grasp_rectangles_2d") or [rec.get("best_grasp_rectangle_2d")]
        pred_rectangles = [rect for rect in pred_rectangles if rect is not None]
        hits += any(
            grasp_rectangle_matches(pred, gt, iou_threshold, angle_threshold_deg)
            for pred in pred_rectangles[:k]
            for gt in gt_rectangles
        )
    return hits / len(records)


def topk_2d_grasp_center_hit_rate(records: list[dict], k: int) -> float:
    if not records:
        return 0.0
    hits = 0
    for rec in records:
        rectangles = rec.get("gt_grasp_rectangles", [])
        centers = rec.get("top_k_grasp_centers_2d") or [rec.get("best_grasp_center_2d")]
        hits += any(point_in_grasp_rectangles(center, rectangles) for center in centers[:k])
    return hits / len(records)
