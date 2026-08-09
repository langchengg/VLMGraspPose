"""GT-free absolute-location and pairwise-relation candidate features."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xx):
        return (float("nan"), float("nan"))
    return (float(xx.mean()), float(yy.mean()))


def location_ranks(
    masks: Iterable[np.ndarray],
    depths: Iterable[float | None],
    image_shape: tuple[int, int],
) -> list[dict[str, float]]:
    masks = list(masks)
    depth_values = list(depths)
    if len(masks) != len(depth_values):
        raise ValueError("masks and depths must align")
    height, width = image_shape
    centroids = [mask_centroid(mask) for mask in masks]
    xs = np.asarray([value[0] for value in centroids], dtype=np.float64)
    ds = np.asarray(
        [np.nan if value is None or not np.isfinite(value) else value for value in depth_values],
        dtype=np.float64,
    )

    def ranks(values: np.ndarray, *, reverse: bool = False) -> np.ndarray:
        safe = np.where(np.isfinite(values), values, np.inf if not reverse else -np.inf)
        order = np.argsort(-safe if reverse else safe, kind="stable")
        result = np.empty(len(order), dtype=np.float64)
        result[order] = np.arange(1, len(order) + 1, dtype=np.float64)
        return result

    left = ranks(xs)
    right = ranks(xs, reverse=True)
    near = ranks(ds)
    far = ranks(ds, reverse=True)
    denominator = float(max(len(masks) - 1, 1))
    rows: list[dict[str, float]] = []
    for index, (x, y) in enumerate(centroids):
        rows.append(
            {
                "centroid_x_normalized": x / max(width - 1, 1),
                "centroid_y_normalized": y / max(height - 1, 1),
                "left_to_right_rank": float(left[index]),
                "right_to_left_rank": float(right[index]),
                "near_to_far_rank": float(near[index]),
                "far_to_near_rank": float(far[index]),
                "leftmost_score": 1.0 - (left[index] - 1.0) / denominator,
                "rightmost_score": 1.0 - (right[index] - 1.0) / denominator,
                "closest_score": 1.0 - (near[index] - 1.0) / denominator,
                "furthest_score": 1.0 - (far[index] - 1.0) / denominator,
                "scene_centre_distance": float(
                    np.hypot(x / max(width - 1, 1) - 0.5, y / max(height - 1, 1) - 0.5)
                ),
            }
        )
    return rows


def pairwise_relation_features(
    target_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    target_depth: float | None,
    reference_depth: float | None,
    image_shape: tuple[int, int],
) -> dict[str, float]:
    height, width = image_shape
    tx, ty = mask_centroid(target_mask)
    rx, ry = mask_centroid(reference_mask)
    dx = (tx - rx) / max(width - 1, 1)
    dy = (ty - ry) / max(height - 1, 1)
    depth_delta = (
        float("nan")
        if target_depth is None
        or reference_depth is None
        or not np.isfinite(target_depth)
        or not np.isfinite(reference_depth)
        else float(target_depth - reference_depth)
    )
    left = float(np.clip(-dx * 4.0, 0.0, 1.0))
    right = float(np.clip(dx * 4.0, 0.0, 1.0))
    behind_y = float(np.clip(-dy * 4.0, 0.0, 1.0))
    front_y = float(np.clip(dy * 4.0, 0.0, 1.0))
    behind_depth = 0.0 if not np.isfinite(depth_delta) else float(np.clip(depth_delta / 0.15, 0.0, 1.0))
    front_depth = 0.0 if not np.isfinite(depth_delta) else float(np.clip(-depth_delta / 0.15, 0.0, 1.0))
    behind = max(behind_y, behind_depth)
    front = max(front_y, front_depth)
    target = np.asarray(target_mask, dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)
    overlap = int(np.count_nonzero(target & reference)) / max(int(np.count_nonzero(target | reference)), 1)
    reference_area = max(int(np.count_nonzero(reference)), 1)
    containment = int(np.count_nonzero(target & reference)) / reference_area
    return {
        "target_reference_dx": float(dx),
        "target_reference_dy": float(dy),
        "target_reference_depth_difference": float(depth_delta),
        "relation_left_score": left,
        "relation_right_score": right,
        "relation_front_score": front,
        "relation_behind_score": behind,
        "relation_front_left_score": float(np.sqrt(front * left)),
        "relation_front_right_score": float(np.sqrt(front * right)),
        "relation_rear_left_score": float(np.sqrt(behind * left)),
        "relation_rear_right_score": float(np.sqrt(behind * right)),
        "relation_overlap": float(overlap),
        "relation_on_score": float(np.clip(containment * (1.0 - min(abs(dy) * 4.0, 1.0)), 0.0, 1.0)),
    }


def relation_score(features: dict[str, float], relation: str | None) -> float:
    if relation is None:
        return 0.0
    key = {
        "left": "relation_left_score",
        "right": "relation_right_score",
        "front": "relation_front_score",
        "behind": "relation_behind_score",
        "front_left": "relation_front_left_score",
        "front_right": "relation_front_right_score",
        "rear_left": "relation_rear_left_score",
        "rear_right": "relation_rear_right_score",
        "on": "relation_on_score",
    }.get(relation)
    return 0.0 if key is None else float(features[key])


__all__ = ["location_ranks", "mask_centroid", "pairwise_relation_features", "relation_score"]
