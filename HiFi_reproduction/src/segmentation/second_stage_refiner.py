"""GT-free prompt construction for target-specific Stage-2 SAM 3 refinement."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from .sam3_text_proposals import TextPromptSpec
from .sam3_visual_proposals import VisualPromptSpec


def tight_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xx):
        raise ValueError("Stage-2 selected candidate is empty")
    return (float(xx.min()), float(yy.min()), float(xx.max()), float(yy.max()))


def interior_points(mask: np.ndarray, count: int, separation_px: float = 12.0) -> tuple[tuple[float, float], ...]:
    mask = np.asarray(mask, dtype=bool)
    distance = ndimage.distance_transform_edt(mask)
    score = distance.copy()
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    values: list[tuple[float, float]] = []
    for _ in range(int(count)):
        flat = int(np.argmax(score))
        if score.flat[flat] <= 0.0:
            break
        y, x = np.unravel_index(flat, score.shape)
        values.append((float(x), float(y)))
        score[(xx - x) ** 2 + (yy - y) ** 2 <= float(separation_px) ** 2] = 0.0
    while len(values) < int(count):
        values.append(values[0])
    return tuple(values)


def competitor_negative_points(
    selected: np.ndarray,
    competitors: list[np.ndarray],
    maximum: int = 3,
) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for competitor in competitors:
        exclusive = np.asarray(competitor, dtype=bool) & ~np.asarray(selected, dtype=bool)
        if not np.any(exclusive):
            continue
        distance = ndimage.distance_transform_edt(exclusive)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        points.append((float(x), float(y)))
        if len(points) >= int(maximum):
            break
    return tuple(points)


def depth_discontinuity_negative_points(
    selected: np.ndarray,
    depth_m: np.ndarray | None,
    *,
    tolerance_m: float = 0.03,
    maximum: int = 2,
) -> tuple[tuple[float, float], ...]:
    if depth_m is None:
        return ()
    selected = np.asarray(selected, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != selected.shape:
        raise ValueError("Stage-2 depth and selected mask shapes differ")
    valid_selected = selected & np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid_selected):
        return ()
    median = float(np.median(depth[valid_selected]))
    exterior = ndimage.binary_dilation(selected, iterations=12) & ~selected
    discontinuity = (
        exterior
        & np.isfinite(depth)
        & (depth > 0.0)
        & (np.abs(depth - median) >= float(tolerance_m))
    )
    return interior_points(discontinuity, min(int(maximum), 2)) if np.any(discontinuity) else ()


def low_hifi_probability_negative_points(
    selected: np.ndarray,
    hifi_probability: np.ndarray | None,
    *,
    maximum_probability: float = 0.20,
    maximum: int = 2,
) -> tuple[tuple[float, float], ...]:
    if hifi_probability is None:
        return ()
    selected = np.asarray(selected, dtype=bool)
    probability = np.asarray(hifi_probability, dtype=np.float32)
    if probability.shape != selected.shape:
        raise ValueError("Stage-2 HiFi probability and selected mask shapes differ")
    expansion = selected & np.isfinite(probability) & (
        probability <= float(maximum_probability)
    )
    return interior_points(expansion, min(int(maximum), 2)) if np.any(expansion) else ()


def build_stage2_prompts(
    selected_mask: np.ndarray,
    competitor_masks: list[np.ndarray],
    competitor_boxes: list[tuple[float, float, float, float]],
    *,
    target_text: str,
    target_attribute_text: str | None = None,
    reference_masks: list[np.ndarray] | None = None,
    reference_boxes: list[tuple[float, float, float, float]] | None = None,
    depth_m: np.ndarray | None = None,
    hifi_probability: np.ndarray | None = None,
) -> tuple[list[VisualPromptSpec], list[TextPromptSpec], dict[str, Any]]:
    selected = np.asarray(selected_mask, dtype=bool)
    box = tight_box(selected)
    point_sets = {count: interior_points(selected, count) for count in (1, 3, 5)}
    competitor_negatives = competitor_negative_points(
        selected, competitor_masks, maximum=3
    )
    reference_negatives = competitor_negative_points(
        selected, reference_masks or [], maximum=2
    )
    depth_negatives = depth_discontinuity_negative_points(selected, depth_m)
    low_probability_negatives = low_hifi_probability_negative_points(
        selected, hifi_probability
    )
    negatives = tuple(
        dict.fromkeys((*competitor_negatives, *reference_negatives))
    )[:3]
    visual = [
        VisualPromptSpec("S2A_box", "STAGE2_TRACKER", "S2A_positive_box", box_xyxy=box),
        VisualPromptSpec(
            "S2A_box_1point",
            "STAGE2_TRACKER",
            "S2A_positive_box_1point",
            box_xyxy=box,
            positive_points_xy=point_sets[1],
        ),
        VisualPromptSpec(
            "S2A_box_3points",
            "STAGE2_TRACKER",
            "S2A_positive_box_3points",
            box_xyxy=box,
            positive_points_xy=point_sets[3],
        ),
        VisualPromptSpec(
            "S2A_mask",
            "STAGE2_TRACKER",
            "S2A_mask_prompt",
            input_mask=selected.astype(np.float32),
        ),
        VisualPromptSpec(
            "S2C_mask_points",
            "STAGE2_TRACKER_CLEANUP",
            "S2C_mask_3points_negative_competitors",
            positive_points_xy=point_sets[3],
            negative_points_xy=negatives,
            input_mask=selected.astype(np.float32),
        ),
    ]
    if negatives:
        visual.append(
            VisualPromptSpec(
                "S2A_box_points_negatives",
                "STAGE2_TRACKER",
                "S2A_box_3points_negative_competitors",
                box_xyxy=box,
                positive_points_xy=point_sets[3],
                negative_points_xy=negatives,
            )
        )
    for prompt_id, variant, points in (
        (
            "S2C_mask_points_reference_negatives",
            "S2C_mask_3points_negative_references",
            reference_negatives,
        ),
        (
            "S2C_mask_points_depth_negatives",
            "S2C_mask_3points_negative_depth_discontinuities",
            depth_negatives,
        ),
        (
            "S2C_mask_points_low_hifi_negatives",
            "S2C_mask_3points_negative_low_hifi_expansion",
            low_probability_negatives,
        ),
    ):
        if points:
            visual.append(
                VisualPromptSpec(
                    prompt_id,
                    "STAGE2_TRACKER_CLEANUP",
                    variant,
                    positive_points_xy=point_sets[3],
                    negative_points_xy=points,
                    input_mask=selected.astype(np.float32),
                )
            )
    negative_boxes = [*competitor_boxes, *(reference_boxes or [])][:3]
    pcs_boxes = (box, *negative_boxes)
    pcs_labels = (1, *([0] * len(negative_boxes)))
    target_texts = list(
        dict.fromkeys(
            value.strip()
            for value in (target_text, target_attribute_text)
            if value and value.strip()
        )
    )
    text = [
        TextPromptSpec(
            prompt_id=f"S2B_text_negative_boxes_{index}",
            source_family="STAGE2_PCS",
            source_variant=(
                "S2B_short_text_positive_box_negative_context|"
                f"text_variant={index}"
            ),
            text=value,
            boxes_xyxy=tuple(pcs_boxes),
            box_labels=tuple(pcs_labels),
        )
        for index, value in enumerate(target_texts)
    ]
    metadata = {
        "selected_box_xyxy": box,
        "positive_points": {str(key): value for key, value in point_sets.items()},
        "negative_points": negatives,
        "competitor_negative_points": competitor_negatives,
        "reference_negative_points": reference_negatives,
        "depth_discontinuity_negative_points": depth_negatives,
        "low_hifi_probability_negative_points": low_probability_negatives,
        "competitor_boxes": competitor_boxes[:3],
        "reference_boxes": (reference_boxes or [])[:3],
        "combined_negative_boxes": negative_boxes,
        "target_texts": target_texts,
        "uses_ground_truth": False,
    }
    return visual, text, metadata


__all__ = [
    "build_stage2_prompts",
    "competitor_negative_points",
    "depth_discontinuity_negative_points",
    "interior_points",
    "low_hifi_probability_negative_points",
    "tight_box",
]
