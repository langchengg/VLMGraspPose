"""Deterministic SAM 3 prompts derived only from frozen HiFi-CS outputs.

Ground-truth paths and metrics are intentionally absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from ..sam3_prompt_builder import VisualPrompt, tight_box_xyxy


PROMPT_FAMILIES: dict[str, dict[str, Any]] = {
    "P0": {"prompt_mode": "point", "positive_point_count": 1, "negative_point_count": 0},
    "P1": {"prompt_mode": "box", "positive_point_count": 1, "negative_point_count": 0},
    "P2": {"prompt_mode": "box_point", "positive_point_count": 1, "negative_point_count": 0},
    "P3": {"prompt_mode": "box_point", "positive_point_count": 3, "negative_point_count": 0},
    "P4": {
        "prompt_mode": "box_positive_negative_points",
        "positive_point_count": 3,
        "negative_point_count": 3,
    },
}


@dataclass(frozen=True)
class SelectivePrompt:
    family: str
    prompt_mode: str
    visual_prompt: VisualPrompt
    metadata: dict[str, Any]


def _depth_metres(depth: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if depth is None:
        return None
    array = np.asarray(depth)
    if array.shape != shape:
        raise ValueError(f"depth shape {array.shape} != {shape}")
    result = array.astype(np.float32)
    if float(np.nanmax(result, initial=0.0)) > 20.0:
        result /= 1000.0
    result[~np.isfinite(result)] = 0.0
    return result


def _component_gap_px(a: np.ndarray, b: np.ndarray) -> float:
    if np.any(ndimage.binary_dilation(a, structure=np.ones((3, 3))) & b):
        return 0.0
    distance = ndimage.distance_transform_edt(~a)
    return float(np.min(distance[b])) if np.any(b) else float("inf")


def _component_depth_consistent(
    component: np.ndarray,
    main: np.ndarray,
    depth_m: np.ndarray | None,
    *,
    minimum_tolerance_m: float,
    mad_multiplier: float,
) -> tuple[bool, float | None, float | None]:
    if depth_m is None:
        return False, None, None
    main_values = depth_m[main & (depth_m > 0.0)]
    component_values = depth_m[component & (depth_m > 0.0)]
    if main_values.size < 4 or component_values.size < 4:
        return False, None, None
    main_median = float(np.median(main_values))
    component_median = float(np.median(component_values))
    mad = float(np.median(np.abs(main_values - main_median)))
    tolerance = max(float(minimum_tolerance_m), float(mad_multiplier) * mad)
    gap = abs(component_median - main_median)
    return gap <= tolerance, gap, tolerance


def clean_components(
    coarse_mask: np.ndarray,
    probability: np.ndarray,
    depth: np.ndarray | None,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep the main component plus supported secondary components.

    A secondary component is retained either when it has substantial HiFi-CS
    probability mass on its own, or when it has moderate mass and is both
    spatially close and depth-consistent with the main seed component.
    """

    mask = np.asarray(coarse_mask, dtype=bool)
    probability = np.asarray(probability, dtype=np.float32)
    if mask.shape != probability.shape or probability.ndim != 2:
        raise ValueError("coarse mask and probability must be aligned 2D arrays")
    if not np.any(mask) or not np.isfinite(probability).all():
        raise ValueError("prompt construction requires a non-empty finite prediction")
    depth_m = _depth_metres(depth, mask.shape)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    records: list[dict[str, Any]] = []
    masses: list[float] = []
    areas: list[int] = []
    for label in range(1, count + 1):
        component = labels == label
        masses.append(float(np.sum(probability[component], dtype=np.float64)))
        areas.append(int(np.count_nonzero(component)))
    main_label = int(
        max(range(1, count + 1), key=lambda item: (masses[item - 1], areas[item - 1], -item))
    )
    main = labels == main_label
    total_mass = max(float(sum(masses)), np.finfo(np.float64).eps)
    diag = float(np.hypot(mask.shape[0], mask.shape[1]))
    kept = np.zeros_like(mask)
    for label in range(1, count + 1):
        component = labels == label
        mass_fraction = masses[label - 1] / total_mass
        area = areas[label - 1]
        gap_px = 0.0 if label == main_label else _component_gap_px(main, component)
        close = gap_px <= max(
            float(config["component_max_gap_px"]),
            float(config["component_max_gap_image_fraction"]) * diag,
        )
        depth_ok, depth_gap_m, depth_tolerance_m = _component_depth_consistent(
            component,
            main,
            depth_m,
            minimum_tolerance_m=float(config["component_depth_minimum_tolerance_m"]),
            mad_multiplier=float(config["component_depth_mad_multiplier"]),
        )
        substantial = mass_fraction >= float(config["component_substantial_mass_fraction"])
        moderate = mass_fraction >= float(config["component_moderate_mass_fraction"])
        keep = label == main_label or substantial or (moderate and close and depth_ok)
        if area < int(config["minimum_component_area_px"]) and label != main_label:
            keep = False
        reason = (
            "main_probability_mass"
            if label == main_label
            else "substantial_probability_mass"
            if substantial and keep
            else "moderate_mass_close_depth_consistent"
            if keep
            else "insufficient_prediction_support"
        )
        if keep:
            kept |= component
        records.append(
            {
                "component_id": int(label),
                "area_px": area,
                "probability_mass": masses[label - 1],
                "probability_mass_fraction": mass_fraction,
                "distance_to_main_px": gap_px,
                "spatially_close": bool(close),
                "depth_consistent": bool(depth_ok),
                "depth_median_gap_m": depth_gap_m,
                "depth_tolerance_m": depth_tolerance_m,
                "kept": bool(keep),
                "decision_reason": reason,
            }
        )
    if not np.any(kept):
        raise AssertionError("component cleaning removed the mandatory main component")
    return kept, records


def high_confidence_core(
    cleaned_mask: np.ndarray,
    probability: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float, str]:
    values = np.asarray(probability, dtype=np.float32)[cleaned_mask]
    threshold = max(
        float(config["core_minimum_probability"]),
        float(np.quantile(values, float(config["core_probability_quantile"]))),
    )
    core = cleaned_mask & (probability >= threshold)
    minimum = int(config["core_minimum_area_px"])
    if int(np.count_nonzero(core)) >= minimum:
        return core, threshold, "high_probability_quantile"
    eroded = ndimage.binary_erosion(cleaned_mask, iterations=1)
    if np.any(eroded):
        return np.asarray(eroded, dtype=bool), threshold, "eroded_cleaned_mask_fallback"
    return cleaned_mask.copy(), threshold, "cleaned_mask_fallback"


def _interior_points(core: np.ndarray, count: int, separation_px: float) -> tuple[tuple[int, int], ...]:
    distance = ndimage.distance_transform_edt(core)
    available = distance.copy()
    points: list[tuple[int, int]] = []
    yy, xx = np.ogrid[: core.shape[0], : core.shape[1]]
    for _ in range(int(count)):
        flat = int(np.argmax(available))
        if float(available.flat[flat]) <= 0.0:
            break
        y, x = np.unravel_index(flat, available.shape)
        points.append((int(x), int(y)))
        available[(xx - x) ** 2 + (yy - y) ** 2 <= float(separation_px) ** 2] = 0.0
    if not points:
        y, x = np.argwhere(core)[0]
        points.append((int(x), int(y)))
    while len(points) < int(count):
        # Small cores may not support the requested NMS radius. Reusing the
        # deepest valid point is deterministic and remains inside the core.
        points.append(points[0])
    return tuple(points)


def _negative_points(
    cleaned_mask: np.ndarray,
    probability: np.ndarray,
    depth_m: np.ndarray | None,
    box: tuple[int, int, int, int],
    positives: tuple[tuple[int, int], ...],
    count: int,
    config: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    if count <= 0:
        return ()
    x1, y1, x2, y2 = box
    in_box = np.zeros_like(cleaned_mask)
    in_box[y1 : y2 + 1, x1 : x2 + 1] = True
    dilated = ndimage.binary_dilation(
        cleaned_mask, iterations=int(config["negative_dilation_px"])
    )
    eligible = (
        in_box
        & ~dilated
        & (probability <= float(config["negative_maximum_probability"]))
    )
    distance = ndimage.distance_transform_edt(~cleaned_mask)
    score = distance.astype(np.float64)
    if depth_m is not None:
        positive_depth = [depth_m[y, x] for x, y in positives if depth_m[y, x] > 0.0]
        if positive_depth:
            median = float(np.median(positive_depth))
            inconsistency = np.abs(depth_m - median)
            valid = depth_m > 0.0
            scale = float(np.quantile(inconsistency[valid], 0.9)) if np.any(valid) else 0.0
            if scale > 0.0:
                score += np.clip(inconsistency / scale, 0.0, 2.0) * float(
                    config["negative_depth_preference_weight"]
                )
    score *= eligible
    points: list[tuple[int, int]] = []
    yy, xx = np.ogrid[: cleaned_mask.shape[0], : cleaned_mask.shape[1]]
    for _ in range(int(count)):
        flat = int(np.argmax(score))
        if float(score.flat[flat]) <= 0.0:
            break
        y, x = np.unravel_index(flat, score.shape)
        points.append((int(x), int(y)))
        score[(xx - x) ** 2 + (yy - y) ** 2 <= float(config["point_separation_px"]) ** 2] = 0.0
    return tuple(points)


def _expand_box_exact(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    fraction: float,
) -> tuple[int, int, int, int]:
    """Expand by the requested fraction, including a true zero-percent case."""

    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("box expansion fraction must be within [0,1]")
    x1, y1, x2, y2 = box
    height, width = shape
    expand_x = int(round((x2 - x1 + 1) * float(fraction)))
    expand_y = int(round((y2 - y1 + 1) * float(fraction)))
    return (
        max(0, x1 - expand_x),
        max(0, y1 - expand_y),
        min(width - 1, x2 + expand_x),
        min(height - 1, y2 + expand_y),
    )


def build_selective_prompt(
    family: str,
    probability: np.ndarray,
    coarse_mask: np.ndarray,
    depth: np.ndarray | None,
    *,
    box_expansion_fraction: float,
    config: Mapping[str, Any],
) -> SelectivePrompt:
    if family not in PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt family {family!r}")
    family_config = PROMPT_FAMILIES[family]
    probability = np.asarray(probability, dtype=np.float32)
    coarse_mask = np.asarray(coarse_mask, dtype=bool)
    depth_m = _depth_metres(depth, coarse_mask.shape)
    cleaned, decisions = clean_components(coarse_mask, probability, depth_m, config)
    core, core_threshold, core_method = high_confidence_core(cleaned, probability, config)
    positives = _interior_points(
        core,
        int(family_config["positive_point_count"]),
        float(config["point_separation_px"]),
    )
    tight = tight_box_xyxy(cleaned)
    expanded = _expand_box_exact(tight, cleaned.shape, float(box_expansion_fraction))
    negatives = _negative_points(
        cleaned,
        probability,
        depth_m,
        expanded,
        positives,
        int(family_config["negative_point_count"]),
        config,
    )
    if not all(bool(core[y, x]) for x, y in positives):
        raise AssertionError("all positive points must lie inside the predicted target core")
    if any(bool(core[y, x]) for x, y in negatives):
        raise AssertionError("negative points must remain outside the predicted target core")
    prompt = VisualPrompt(
        strategy=(
            "box_positive_negative_points"
            if family == "P4"
            else "box_positive_points"
        ),
        threshold=0.5,
        tight_box_xyxy=tight,
        expanded_box_xyxy=expanded,
        positive_points_xy=positives,
        negative_points_xy=negatives,
        cleaned_mask=cleaned,
        component_count=sum(bool(item["kept"]) for item in decisions),
        removed_component_count=sum(not bool(item["kept"]) for item in decisions),
    )
    metadata = {
        "prompt_family": family,
        "prompt_mode": family_config["prompt_mode"],
        "box_expansion_fraction": float(box_expansion_fraction),
        "core_probability_threshold": core_threshold,
        "core_method": core_method,
        "core_area_px": int(np.count_nonzero(core)),
        "cleaned_area_px": int(np.count_nonzero(cleaned)),
        "positive_points_xy": [list(item) for item in positives],
        "negative_points_xy": [list(item) for item in negatives],
        "tight_box_xyxy": list(tight),
        "expanded_box_xyxy": list(expanded),
        "component_decisions": decisions,
        "uses_ground_truth": False,
    }
    return SelectivePrompt(
        family=family,
        prompt_mode=str(family_config["prompt_mode"]),
        visual_prompt=prompt,
        metadata=metadata,
    )
