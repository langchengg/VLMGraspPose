"""Ground-truth-free trigger and candidate feature extraction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from ..sam3_prompt_builder import VisualPrompt


EPS = np.finfo(np.float64).eps


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _components(mask: np.ndarray) -> tuple[int, float]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    area = int(np.count_nonzero(mask))
    if not area:
        return 0, 0.0
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return int(count), float(np.max(sizes) / area)


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def _perimeter(mask: np.ndarray) -> float:
    return float(np.count_nonzero(_boundary(mask)))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask)
    if not yy.size:
        return None
    return int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())


def _entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability.astype(np.float64), 1.0e-7, 1.0 - 1.0e-7)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _gray_gradient(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    return np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))


def _edge_alignment(mask: np.ndarray, gradient: np.ndarray) -> float:
    boundary = _boundary(mask)
    if not np.any(boundary):
        return 0.0
    scale = float(np.quantile(gradient, 0.95))
    return float(np.mean(np.clip(gradient[boundary] / max(scale, EPS), 0.0, 1.0)))


def _query_type(query: str) -> str:
    text = " " + query.lower().strip() + " "
    relation_words = (" left of ", " right of ", " behind ", " in front of ", " next to ", " between ")
    location_words = (" on the left ", " on the right ", " top ", " bottom ", " corner ", " center ")
    attribute_words = (" red ", " blue ", " green ", " yellow ", " white ", " black ", " small ", " large ", " big ")
    flags = [any(word in text for word in words) for words in (relation_words, location_words, attribute_words)]
    if sum(flags) > 1:
        return "mixed"
    if flags[0]:
        return "relation"
    if flags[1]:
        return "location"
    if flags[2]:
        return "attribute"
    return "name"


def pre_sam_features(
    coarse_mask: np.ndarray,
    probability: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray | None,
    query: str,
) -> dict[str, Any]:
    mask = np.asarray(coarse_mask, dtype=bool)
    probability = np.asarray(probability, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if mask.shape != probability.shape or rgb.shape[:2] != mask.shape:
        raise ValueError("pre-SAM features require aligned prediction inputs")
    area = int(np.count_nonzero(mask))
    image_area = int(mask.size)
    box = _bbox(mask)
    if box is None:
        x1 = y1 = x2 = y2 = 0
        box_area = 0
    else:
        x1, y1, x2, y2 = box
        box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    component_count, largest_ratio = _components(mask)
    perimeter = _perimeter(mask)
    compactness = float(4.0 * np.pi * area / max(perimeter * perimeter, EPS))
    inside = probability[mask]
    ring = ndimage.binary_dilation(mask, iterations=3) ^ ndimage.binary_erosion(mask, iterations=3)
    distance = ndimage.distance_transform_edt(mask)
    gradient = _gray_gradient(rgb)
    features: dict[str, Any] = {
        "coarse_area_fraction": area / image_area,
        "bbox_area_fraction": box_area / image_area,
        "bbox_fill_ratio": _ratio(area, box_area),
        "connected_component_count": component_count,
        "largest_component_ratio": largest_ratio,
        "perimeter_to_area_ratio": _ratio(perimeter, area),
        "compactness": compactness if area else 0.0,
        "probability_mean_inside": float(np.mean(inside)) if inside.size else 0.0,
        "probability_median_inside": float(np.median(inside)) if inside.size else 0.0,
        "probability_q10_inside": float(np.quantile(inside, 0.10)) if inside.size else 0.0,
        "probability_q25_inside": float(np.quantile(inside, 0.25)) if inside.size else 0.0,
        "probability_q75_inside": float(np.quantile(inside, 0.75)) if inside.size else 0.0,
        "probability_q90_inside": float(np.quantile(inside, 0.90)) if inside.size else 0.0,
        "total_target_probability_mass": float(np.sum(probability, dtype=np.float64) / image_area),
        "uncertain_pixel_fraction": float(np.mean((probability >= 0.1) & (probability <= 0.9))),
        "boundary_ring_binary_entropy": float(np.mean(_entropy(probability)[ring])) if np.any(ring) else 0.0,
        "maximum_distance_transform_radius": float(np.max(distance)),
        "rgb_gradient_alignment_boundary": _edge_alignment(mask, gradient),
        "distance_to_image_edge_fraction": (
            float(min(x1, y1, mask.shape[1] - 1 - x2, mask.shape[0] - 1 - y2) / max(mask.shape))
            if box is not None
            else 0.0
        ),
        "query_type": _query_type(query),
    }
    if depth is None:
        depth_m = np.zeros(mask.shape, dtype=np.float32)
    else:
        depth_m = np.asarray(depth, dtype=np.float32)
        if float(np.nanmax(depth_m, initial=0.0)) > 20.0:
            depth_m /= 1000.0
        depth_m[~np.isfinite(depth_m)] = 0.0
    valid = depth_m > 0.0
    values = depth_m[mask & valid]
    outside_ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
    inside_ring = mask & ~ndimage.binary_erosion(mask, iterations=3)
    inside_depth = depth_m[inside_ring & valid]
    outside_depth = depth_m[outside_ring & valid]
    features.update(
        {
            "valid_depth_fraction_inside": _ratio(float(values.size), float(area)),
            "depth_median": float(np.median(values)) if values.size else 0.0,
            "depth_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) if values.size else 0.0,
            "depth_standard_deviation": float(np.std(values)) if values.size else 0.0,
            "depth_discontinuity_across_boundary": abs(float(np.median(inside_depth)) - float(np.median(outside_depth))) if inside_depth.size and outside_depth.size else 0.0,
        }
    )
    return features


def _point_fraction(mask: np.ndarray, points: Sequence[Sequence[int]], expected: bool) -> float:
    if not points:
        return 1.0 if expected else 0.0
    hits = [bool(mask[int(y), int(x)]) == expected for x, y in points]
    return float(np.mean(hits))


def post_sam_features(
    candidate_mask: np.ndarray,
    candidate_probability: np.ndarray,
    sam_quality: float | None,
    coarse_mask: np.ndarray,
    hifi_probability: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray | None,
    prompt: VisualPrompt,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_mask, dtype=bool)
    coarse = np.asarray(coarse_mask, dtype=bool)
    probability = np.asarray(candidate_probability, dtype=np.float32)
    hifi = np.asarray(hifi_probability, dtype=np.float32)
    if not (candidate.shape == coarse.shape == probability.shape == hifi.shape):
        raise ValueError("candidate feature arrays are not aligned")
    area = int(np.count_nonzero(candidate))
    coarse_area = int(np.count_nonzero(coarse))
    intersection = int(np.count_nonzero(candidate & coarse))
    union = int(np.count_nonzero(candidate | coarse))
    candidate_mass = float(np.sum(hifi[candidate], dtype=np.float64))
    coarse_mass = float(np.sum(hifi[coarse], dtype=np.float64))
    total_candidate_capacity = float(area)
    components, largest = _components(candidate)
    perimeter = _perimeter(candidate)
    prompt_box = np.zeros_like(candidate)
    x1, y1, x2, y2 = prompt.expanded_box_xyxy
    prompt_box[y1 : y2 + 1, x1 : x2 + 1] = True
    candidate_boundary = _boundary(candidate)
    coarse_boundary = _boundary(coarse)
    if np.any(candidate_boundary) and np.any(coarse_boundary):
        distance_to_coarse = ndimage.distance_transform_edt(~coarse_boundary)
        distance_to_candidate = ndimage.distance_transform_edt(~candidate_boundary)
        displacement = 0.5 * (
            float(np.mean(distance_to_coarse[candidate_boundary]))
            + float(np.mean(distance_to_candidate[coarse_boundary]))
        )
    else:
        displacement = float(max(candidate.shape))
    gradient = _gray_gradient(rgb)
    depth_score = 0.0
    contamination = 0.0
    if depth is not None:
        depth_m = np.asarray(depth, dtype=np.float32)
        if float(np.nanmax(depth_m, initial=0.0)) > 20.0:
            depth_m /= 1000.0
        depth_m[~np.isfinite(depth_m)] = 0.0
        seed_values = [depth_m[y, x] for x, y in prompt.positive_points_xy if depth_m[y, x] > 0.0]
        if seed_values:
            median = float(np.median(seed_values))
            reference = depth_m[coarse & (depth_m > 0.0)]
            mad = float(np.median(np.abs(reference - np.median(reference)))) if reference.size else 0.0
            tolerance = max(0.02, 3.0 * mad)
            valid_candidate = candidate & (depth_m > 0.0)
            if np.any(valid_candidate):
                consistent = np.abs(depth_m - median) <= tolerance
                depth_score = float(np.mean(consistent[valid_candidate]))
                expanded = candidate & ~coarse & (depth_m > 0.0)
                contamination = float(np.mean(~consistent[expanded])) if np.any(expanded) else 0.0
    expansion = candidate & ~coarse
    low_probability_expansion = _ratio(
        float(np.count_nonzero(expansion & (hifi < 0.1))),
        float(np.count_nonzero(expansion)),
    )
    return {
        "candidate_id": str(candidate_id),
        "sam_quality": 0.0 if sam_quality is None else float(sam_quality),
        "sam_quality_available": bool(sam_quality is not None),
        "coarse_sam_iou": _ratio(intersection, union),
        "coarse_recall": _ratio(intersection, coarse_area),
        "coarse_precision": _ratio(intersection, area),
        "sam_to_coarse_area_ratio": _ratio(area, coarse_area),
        "hifi_probability_mass_recall": _ratio(candidate_mass, coarse_mass),
        "hifi_probability_mass_precision": _ratio(candidate_mass, total_candidate_capacity),
        "positive_point_inclusion_ratio": _point_fraction(candidate, prompt.positive_points_xy, True),
        "negative_point_violation_ratio": 1.0 - _point_fraction(candidate, prompt.negative_points_xy, False),
        "connected_component_count": components,
        "largest_component_ratio": largest,
        "area_px": area,
        "bbox_xyxy": _bbox(candidate),
        "perimeter": perimeter,
        "compactness": float(4.0 * np.pi * area / max(perimeter * perimeter, EPS)) if area else 0.0,
        "boundary_displacement_px": displacement,
        "prompt_box_support": _ratio(float(np.count_nonzero(candidate & prompt_box)), float(area)),
        "rgb_edge_alignment": _edge_alignment(candidate, gradient),
        "depth_consistency_with_positive_region": depth_score,
        "depth_contamination_outside_coarse": contamination,
        "fragmentation_penalty": 1.0 - largest,
        "low_hifi_probability_expansion_fraction": low_probability_expansion,
    }
