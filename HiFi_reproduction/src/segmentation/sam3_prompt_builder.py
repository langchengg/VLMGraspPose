"""Build deterministic SAM 3 visual prompts from a HiFi-CS prediction only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class VisualPrompt:
    strategy: str
    threshold: float
    tight_box_xyxy: tuple[int, int, int, int]
    expanded_box_xyxy: tuple[int, int, int, int]
    positive_points_xy: tuple[tuple[int, int], ...]
    negative_points_xy: tuple[tuple[int, int], ...]
    cleaned_mask: np.ndarray
    component_count: int
    removed_component_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "threshold": self.threshold,
            "tight_box_xyxy": list(self.tight_box_xyxy),
            "expanded_box_xyxy": list(self.expanded_box_xyxy),
            "positive_points_xy": [list(item) for item in self.positive_points_xy],
            "negative_points_xy": [list(item) for item in self.negative_points_xy],
            "component_count": self.component_count,
            "removed_component_count": self.removed_component_count,
        }


def clean_coarse_mask(
    probability: np.ndarray,
    *,
    threshold: float,
    minimum_component_area_px: int,
) -> tuple[np.ndarray, dict[str, int]]:
    probability = np.asarray(probability)
    if probability.ndim != 2 or not np.all(np.isfinite(probability)):
        raise ValueError("coarse probability must be a finite 2D array")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be within [0,1]")
    binary = probability >= float(threshold)
    labels, count = ndimage.label(binary)
    cleaned = np.zeros_like(binary, dtype=bool)
    kept = 0
    for label in range(1, count + 1):
        component = labels == label
        if int(np.count_nonzero(component)) >= int(minimum_component_area_px):
            cleaned |= component
            kept += 1
    return cleaned, {
        "original_component_count": int(count),
        "component_count": kept,
        "removed_component_count": int(count) - kept,
    }


def tight_box_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if not rows.size:
        raise ValueError("cannot build a prompt from an empty coarse mask")
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def expand_box_xyxy(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    fraction: float,
) -> tuple[int, int, int, int]:
    height, width = map(int, image_shape)
    x1, y1, x2, y2 = box
    expand_x = max(1, int(round((x2 - x1 + 1) * float(fraction))))
    expand_y = max(1, int(round((y2 - y1 + 1) * float(fraction))))
    return (
        max(0, x1 - expand_x),
        max(0, y1 - expand_y),
        min(width - 1, x2 + expand_x),
        min(height - 1, y2 + expand_y),
    )


def distance_transform_points(
    mask: np.ndarray,
    *,
    count: int,
    minimum_separation_px: float,
) -> tuple[tuple[int, int], ...]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("positive points require a non-empty mask")
    distance = ndimage.distance_transform_edt(mask)
    available = np.array(distance, copy=True)
    points: list[tuple[int, int]] = []
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    for _ in range(max(1, int(count))):
        flat = int(np.argmax(available))
        value = float(available.flat[flat])
        if value <= 0.0:
            break
        y, x = np.unravel_index(flat, available.shape)
        if not mask[y, x]:
            raise AssertionError("distance-transform maximum must be inside mask")
        points.append((int(x), int(y)))
        suppress = (xx - x) ** 2 + (yy - y) ** 2 <= float(minimum_separation_px) ** 2
        available[suppress] = 0.0
    return tuple(points)


def sample_negative_points(
    mask: np.ndarray,
    expanded_box: tuple[int, int, int, int],
    *,
    count: int,
    ring_width_px: int,
    minimum_separation_px: float,
) -> tuple[tuple[int, int], ...]:
    if count <= 0:
        return ()
    mask = np.asarray(mask, dtype=bool)
    dilation = ndimage.binary_dilation(mask, iterations=max(1, int(ring_width_px)))
    ring = dilation & ~mask
    x1, y1, x2, y2 = expanded_box
    inside_box = np.zeros_like(mask)
    inside_box[y1 : y2 + 1, x1 : x2 + 1] = True
    region = (ring | (inside_box & ~mask)) & ~mask
    # Prefer pixels far from the coarse mask and image edges, with deterministic NMS.
    distance = ndimage.distance_transform_edt(~mask) * region
    available = np.asarray(distance, dtype=np.float64)
    points: list[tuple[int, int]] = []
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    for _ in range(int(count)):
        flat = int(np.argmax(available))
        if float(available.flat[flat]) <= 0.0:
            break
        y, x = np.unravel_index(flat, available.shape)
        points.append((int(x), int(y)))
        available[(xx - x) ** 2 + (yy - y) ** 2 <= float(minimum_separation_px) ** 2] = 0.0
    return tuple(points)


def build_visual_prompt(probability: np.ndarray, config: Mapping[str, Any]) -> VisualPrompt:
    strategy = str(config["strategy"])
    if strategy not in {"box_positive_points", "box_positive_negative_points"}:
        raise ValueError(f"visual prompt builder does not support strategy {strategy!r}")
    cleaned, component_stats = clean_coarse_mask(
        probability,
        threshold=float(config["coarse_mask_threshold"]),
        minimum_component_area_px=int(config["minimum_component_area_px"]),
    )
    if not np.any(cleaned):
        raise ValueError("coarse mask is empty after conservative cleanup")
    tight = tight_box_xyxy(cleaned)
    expanded = expand_box_xyxy(tight, cleaned.shape, float(config["box_expansion_fraction"]))
    positives = distance_transform_points(
        cleaned,
        count=int(config["positive_point_count"]),
        minimum_separation_px=float(config["point_minimum_separation_px"]),
    )
    negatives: tuple[tuple[int, int], ...] = ()
    if strategy == "box_positive_negative_points":
        negatives = sample_negative_points(
            cleaned,
            expanded,
            count=int(config["negative_point_count"]),
            ring_width_px=int(config["negative_ring_width_px"]),
            minimum_separation_px=float(config["point_minimum_separation_px"]),
        )
    if not all(cleaned[y, x] for x, y in positives):
        raise AssertionError("every positive prompt point must lie inside the coarse mask")
    return VisualPrompt(
        strategy=strategy,
        threshold=float(config["coarse_mask_threshold"]),
        tight_box_xyxy=tight,
        expanded_box_xyxy=expanded,
        positive_points_xy=positives,
        negative_points_xy=negatives,
        cleaned_mask=cleaned,
        component_count=component_stats["component_count"],
        removed_component_count=component_stats["removed_component_count"],
    )
