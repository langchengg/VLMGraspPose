"""Inference-only SAM 3 hypothesis scoring and explicit safe fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .sam3_prompt_builder import VisualPrompt


@dataclass(frozen=True)
class SelectionResult:
    selected_mask_source: str
    selected_hypothesis_id: str | None
    selected_mask: np.ndarray
    selected_probability: np.ndarray
    refinement_score: float | None
    fallback_reason: str | None
    candidate_metrics: tuple[dict[str, Any], ...]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(numerator / denominator)


def _point_support(mask: np.ndarray, points: Sequence[Sequence[int]]) -> float:
    if not points:
        return 1.0
    height, width = mask.shape
    hits = 0
    for x, y in points:
        if 0 <= int(x) < width and 0 <= int(y) < height and bool(mask[int(y), int(x)]):
            hits += 1
    return hits / len(points)


def _largest_component_ratio(mask: np.ndarray) -> tuple[int, float]:
    labels, count = ndimage.label(mask)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0, 0.0
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return int(count), float(np.max(sizes) / area)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.nonzero(mask)
    if not rows.size:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def _box_overlap_fraction(mask: np.ndarray, box: Sequence[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(int, box)
    inside = np.zeros_like(mask, dtype=bool)
    inside[max(0, y1) : min(mask.shape[0], y2 + 1), max(0, x1) : min(mask.shape[1], x2 + 1)] = True
    area = int(np.count_nonzero(mask))
    inside_count = int(np.count_nonzero(mask & inside))
    return _safe_ratio(inside_count, area), _safe_ratio(area - inside_count, area)


def depth_consistency_score(
    candidate_mask: np.ndarray,
    coarse_mask: np.ndarray,
    depth_m: np.ndarray | None,
    config: Mapping[str, Any],
) -> float:
    """Measure candidate support near the robust coarse-mask depth surface."""

    if depth_m is None or not bool(config.get("enabled", False)):
        return 0.0
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.shape != candidate_mask.shape or not np.all(np.isfinite(depth)):
        raise ValueError("depth consistency requires a finite aligned depth map")
    valid = depth > 0.0
    reference_values = depth[np.asarray(coarse_mask, dtype=bool) & valid]
    candidate_values = depth[np.asarray(candidate_mask, dtype=bool) & valid]
    if not reference_values.size or not candidate_values.size:
        return 0.0
    median = float(np.median(reference_values))
    mad = float(np.median(np.abs(reference_values - median)))
    tolerance = max(float(config["minimum_tolerance_m"]), float(config["mad_multiplier"]) * mad)
    return float(np.mean(np.abs(candidate_values - median) <= tolerance))


def area_ratio_penalty(area_ratio: float, config: Mapping[str, Any]) -> float:
    lower = float(config["preferred_min"])
    upper = float(config["preferred_max"])
    if lower <= area_ratio <= upper:
        return 0.0
    if area_ratio < lower:
        return float(np.clip((lower - area_ratio) / max(lower, 1e-12), 0.0, 1.0))
    return float(np.clip((area_ratio - upper) / max(upper, 1e-12), 0.0, 1.0))


def score_candidate_mask(
    candidate_id: str,
    mask: np.ndarray,
    probability: np.ndarray,
    sam_quality: float | None,
    *,
    coarse_mask: np.ndarray,
    coarse_probability: np.ndarray | None = None,
    prompt: VisualPrompt,
    depth_m: np.ndarray | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    raw_mask = np.asarray(mask)
    if raw_mask.ndim != 2 or not np.all(np.isfinite(raw_mask)):
        raise ValueError("SAM candidate mask must be a finite 2D array")
    if sam_quality is not None and not math.isfinite(float(sam_quality)):
        raise ValueError("SAM candidate quality must be finite")
    mask = raw_mask.astype(bool)
    probability = np.asarray(probability, dtype=np.float32)
    coarse = np.asarray(coarse_mask, dtype=bool)
    if mask.shape != coarse.shape or probability.shape != coarse.shape:
        raise ValueError("SAM candidate mask/probability must match coarse-mask resolution")
    if not np.all(np.isfinite(probability)):
        raise ValueError("SAM candidate probability contains non-finite values")
    area = int(np.count_nonzero(mask))
    coarse_area = int(np.count_nonzero(coarse))
    intersection = int(np.count_nonzero(mask & coarse))
    union = int(np.count_nonzero(mask | coarse))
    recall = _safe_ratio(intersection, coarse_area)
    precision = _safe_ratio(intersection, area)
    coarse_iou = _safe_ratio(intersection, union)
    if coarse_probability is None:
        probability_mass_recall = recall
    else:
        coarse_probability = np.asarray(coarse_probability, dtype=np.float64)
        if coarse_probability.shape != coarse.shape or not np.all(np.isfinite(coarse_probability)):
            raise ValueError("coarse probability must be finite and aligned")
        probability_mass_recall = _safe_ratio(
            float(np.sum(coarse_probability[mask])),
            float(np.sum(coarse_probability)),
        )
    area_ratio = _safe_ratio(area, coarse_area)
    component_count, largest_ratio = _largest_component_ratio(mask)
    positive_support = _point_support(mask, prompt.positive_points_xy)
    main_positive = bool(prompt.positive_points_xy and positive_support > 0.0)
    if prompt.positive_points_xy:
        x, y = prompt.positive_points_xy[0]
        main_positive = bool(mask[y, x])
    negative_violation = 1.0 - _point_support(~mask, prompt.negative_points_xy)
    box_overlap, outside_box = _box_overlap_fraction(mask, prompt.expanded_box_xyxy)
    depth_score = depth_consistency_score(mask, coarse, depth_m, config["depth_consistency"])
    area_penalty = area_ratio_penalty(area_ratio, config["area_ratio_penalty"])
    fragmentation_penalty = 1.0 - largest_ratio
    weights = config["weights"]
    quality_weight = float(weights.get("sam_quality", 0.0))
    probability_weight = float(weights.get("probability_mass_recall", 0.0))
    quality_term = (
        quality_weight * float(np.clip(sam_quality, 0.0, 1.0))
        if sam_quality is not None
        else 0.0
    )
    score = (
        quality_term
        + float(weights["coarse_recall"]) * recall
        + float(weights["coarse_precision"]) * precision
        + float(weights["coarse_iou"]) * coarse_iou
        + probability_weight * probability_mass_recall
        + float(weights["positive_points"]) * positive_support
        + float(weights["largest_component"]) * largest_ratio
        + float(weights["depth_consistency"]) * depth_score
        + float(weights["prompt_box_overlap"]) * box_overlap
        - float(weights["negative_violation"]) * negative_violation
        - float(weights["area_ratio_penalty"]) * area_penalty
        - float(weights["fragmentation_penalty"]) * fragmentation_penalty
    )
    # Keep the configured score scale when the official output has no quality
    # field. This is a deterministic renormalization, not an invented SAM score.
    if sam_quality is None and quality_weight > 0.0:
        active = sum(abs(float(value)) for key, value in weights.items() if key != "sam_quality")
        configured = active + quality_weight
        if active > 0.0:
            score *= configured / active
    return {
        "candidate_id": str(candidate_id),
        "sam_quality": None if sam_quality is None else float(sam_quality),
        "sam_quality_available": sam_quality is not None,
        "area_px": area,
        "coarse_area_px": coarse_area,
        "coarse_recall": recall,
        "coarse_precision": precision,
        "coarse_iou": coarse_iou,
        "probability_mass_recall": probability_mass_recall,
        "main_positive_point_included": main_positive,
        "positive_point_inclusion_ratio": positive_support,
        "negative_point_violation_ratio": negative_violation,
        "area_ratio": area_ratio,
        "component_count": component_count,
        "largest_component_ratio": largest_ratio,
        "candidate_box_xyxy": _bbox(mask),
        "expanded_prompt_box_overlap": box_overlap,
        "outside_expanded_box_fraction": outside_box,
        "depth_consistency": depth_score,
        "area_ratio_penalty": area_penalty,
        "fragmentation_penalty": fragmentation_penalty,
        "refinement_score": float(score),
        "valid_hypothesis": True,
        "invalid_reason": None,
    }


def _invalid_candidate_metrics(
    candidate_id: str,
    sam_quality: float | None,
    coarse_area: int,
    reason: str,
) -> dict[str, Any]:
    """Retain an invalid hypothesis in audit tables while making it unselectable."""

    return {
        "candidate_id": candidate_id,
        "sam_quality": (
            float(sam_quality)
            if sam_quality is not None and math.isfinite(float(sam_quality))
            else None
        ),
        "sam_quality_available": sam_quality is not None,
        "area_px": 0,
        "coarse_area_px": coarse_area,
        "coarse_recall": 0.0,
        "coarse_precision": 0.0,
        "coarse_iou": 0.0,
        "probability_mass_recall": 0.0,
        "main_positive_point_included": False,
        "positive_point_inclusion_ratio": 0.0,
        "negative_point_violation_ratio": 0.0,
        "area_ratio": 0.0,
        "component_count": 0,
        "largest_component_ratio": 0.0,
        "candidate_box_xyxy": None,
        "expanded_prompt_box_overlap": 0.0,
        "outside_expanded_box_fraction": 0.0,
        "depth_consistency": 0.0,
        "area_ratio_penalty": 1.0,
        "fragmentation_penalty": 1.0,
        "refinement_score": -1.0e9,
        "valid_hypothesis": False,
        "invalid_reason": reason,
    }


def fallback_reasons(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    fallback = config["fallback"]
    reasons: list[str] = []
    if not bool(metrics.get("valid_hypothesis", True)):
        reasons.append(f"invalid_sam_hypothesis:{metrics.get('invalid_reason')}")
    if int(metrics["area_px"]) <= 0:
        reasons.append("empty_sam_mask")
    if not bool(metrics["main_positive_point_included"]):
        reasons.append("main_positive_point_missing")
    if float(metrics["area_ratio"]) < float(fallback["minimum_area_ratio"]):
        reasons.append("area_ratio_below_minimum")
    if float(metrics["area_ratio"]) > float(fallback["maximum_area_ratio"]):
        reasons.append("area_ratio_above_maximum")
    if float(metrics["coarse_recall"]) < float(fallback["minimum_coarse_recall"]):
        reasons.append("coarse_recall_below_minimum")
    if float(metrics["outside_expanded_box_fraction"]) > float(fallback["maximum_outside_box_fraction"]):
        reasons.append("extends_beyond_expanded_prompt_box")
    if float(metrics["refinement_score"]) < float(fallback["minimum_refinement_score"]):
        reasons.append("refinement_score_below_minimum")
    return reasons


def select_refined_mask(
    masks: Sequence[np.ndarray],
    probabilities: Sequence[np.ndarray],
    sam_qualities: Sequence[float | None],
    *,
    coarse_mask: np.ndarray,
    coarse_probability: np.ndarray,
    prompt: VisualPrompt,
    depth_m: np.ndarray | None,
    config: Mapping[str, Any],
    accepted_source: str = "sam3",
) -> SelectionResult:
    coarse_mask = np.asarray(coarse_mask, dtype=bool)
    coarse_probability = np.asarray(coarse_probability, dtype=np.float32)
    if coarse_mask.shape != coarse_probability.shape or not np.any(coarse_mask):
        raise ValueError("fallback requires a non-empty aligned coarse mask and probability")
    if not (len(masks) == len(probabilities) == len(sam_qualities)):
        raise ValueError("SAM hypotheses, probabilities, and qualities must have equal length")
    if not masks:
        return SelectionResult(
            "hifics_fallback", None, coarse_mask.copy(), coarse_probability.copy(), None,
            "no_sam_masks_returned", (),
        )
    metric_rows: list[dict[str, Any]] = []
    for index, (mask, probability, quality) in enumerate(
        zip(masks, probabilities, sam_qualities, strict=True)
    ):
        candidate_id = f"sam3_{index:03d}"
        try:
            metric_rows.append(
                score_candidate_mask(
                    candidate_id,
                    mask,
                    probability,
                    quality,
                    coarse_mask=coarse_mask,
                    coarse_probability=coarse_probability,
                    prompt=prompt,
                    depth_m=depth_m,
                    config=config,
                )
            )
        except (TypeError, ValueError) as error:
            metric_rows.append(
                _invalid_candidate_metrics(
                    candidate_id,
                    quality,
                    int(np.count_nonzero(coarse_mask)),
                    f"{type(error).__name__}: {error}",
                )
            )
    metrics = tuple(metric_rows)
    # Score descending, then deterministic candidate ID ascending.
    selected = sorted(metrics, key=lambda item: (-item["refinement_score"], item["candidate_id"]))[0]
    index = int(str(selected["candidate_id"]).rsplit("_", 1)[1])
    reasons = fallback_reasons(selected, config)
    if reasons:
        return SelectionResult(
            "hifics_fallback", selected["candidate_id"], coarse_mask.copy(), coarse_probability.copy(),
            float(selected["refinement_score"]), "|".join(reasons), metrics,
        )
    probability = np.asarray(probabilities[index], dtype=np.float32)
    if not np.all(np.isfinite(probability)):
        return SelectionResult(
            "hifics_fallback", selected["candidate_id"], coarse_mask.copy(), coarse_probability.copy(),
            float(selected["refinement_score"]), "selected_probability_nonfinite", metrics,
        )
    return SelectionResult(
        accepted_source,
        selected["candidate_id"],
        np.asarray(masks[index], dtype=bool).copy(),
        probability.copy(),
        float(selected["refinement_score"]), None, metrics,
    )
