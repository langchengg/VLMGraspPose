from __future__ import annotations

import cv2
import numpy as np


def refine_bbox_mask_with_depth(
    bbox: list[int],
    depth: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, dict]:
    """Refine a coarse bbox mask using the nearest valid depth component.

    This is a CPU-only fallback for VLM backends that return a bbox but no
    segmentation mask. It assumes the target object is generally closer to the
    camera than the supporting table/floor pixels inside the bbox.
    """
    h, w = depth.shape
    x1, y1, x2, y2 = _expand_and_clip_bbox(bbox, w, h, config)
    bbox_mask = np.zeros((h, w), dtype=bool)
    bbox_mask[y1:y2 + 1, x1:x2 + 1] = True
    valid = np.isfinite(depth) & (depth > 0) & bbox_mask
    values = depth[valid]
    if values.size == 0:
        return bbox_mask, {"status": "fallback_bbox", "reason": "no_valid_depth"}

    percentile = float(config.get("foreground_percentile", 30.0))
    band = float(config.get("depth_band_m", 0.08))
    seed_depth = float(np.percentile(values, percentile))
    candidate = valid & (depth <= seed_depth + band)

    open_kernel = int(config.get("open_kernel", 3))
    close_kernel = int(config.get("close_kernel", 5))
    candidate = _morph(candidate, open_kernel, cv2.MORPH_OPEN)
    candidate = _morph(candidate, close_kernel, cv2.MORPH_CLOSE)
    component = _select_component(candidate, (0.5 * (x1 + x2), 0.5 * (y1 + y2)), config)
    if component is None:
        return bbox_mask, {
            "status": "fallback_bbox",
            "reason": "no_component",
            "seed_depth": seed_depth,
            "depth_band_m": band,
        }

    dilate = int(config.get("dilate_pixels", 3))
    if dilate > 0:
        component = _morph(component, dilate, cv2.MORPH_DILATE)
        component &= bbox_mask

    bbox_area = max(int(bbox_mask.sum()), 1)
    area = int(component.sum())
    min_area = max(int(config.get("min_area_pixels", 20)), int(float(config.get("min_area_ratio", 0.03)) * bbox_area))
    if area < min_area:
        return bbox_mask, {
            "status": "fallback_bbox",
            "reason": "component_too_small",
            "area": area,
            "min_area": min_area,
            "seed_depth": seed_depth,
            "depth_band_m": band,
        }

    return component.astype(bool), {
        "status": "depth_refined",
        "bbox": [x1, y1, x2, y2],
        "area": area,
        "bbox_area": bbox_area,
        "area_ratio": float(area / bbox_area),
        "seed_depth": seed_depth,
        "depth_band_m": band,
        "foreground_percentile": percentile,
    }


def should_refine_target_mask(target_source: str, metadata: dict, config: dict) -> bool:
    if not bool(config.get("enabled", True)):
        return False
    sources = config.get("apply_to_sources", ["vlm"])
    if sources != "all" and target_source not in set(sources):
        return False
    mask_source = metadata.get("target_mask_source")
    return mask_source in {None, "vlm_bbox", "bbox", "bbox_fallback"}


def _clip_bbox(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = sorted(np.clip([x1, x2], 0, width - 1).astype(int).tolist())
    y1, y2 = sorted(np.clip([y1, y2], 0, height - 1).astype(int).tolist())
    return x1, y1, x2, y2


def _expand_and_clip_bbox(bbox: list[int], width: int, height: int, config: dict) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(x2 - x1 + 1, 1)
    bh = max(y2 - y1 + 1, 1)
    ratio = float(config.get("bbox_expansion_ratio", 0.25))
    pixels = int(config.get("bbox_expansion_pixels", 0))
    dx = int(round(max(pixels, ratio * bw)))
    dy = int(round(max(pixels, ratio * bh)))
    bottom_extra = int(round(float(config.get("bbox_bottom_expansion_ratio", ratio)) * bh))
    expanded = [x1 - dx, y1 - dy, x2 + dx, y2 + max(dy, bottom_extra)]
    return _clip_bbox(expanded, width, height)


def _morph(mask: np.ndarray, kernel_size: int, op: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), op, kernel).astype(bool)


def _select_component(mask: np.ndarray, center_xy: tuple[float, float], config: dict) -> np.ndarray | None:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None
    min_area = int(config.get("component_min_area_pixels", 10))
    best_label = None
    best_score = -float("inf")
    cx, cy = center_xy
    for label in range(1, num):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_cx, comp_cy = centroids[label]
        dist = float(np.hypot(comp_cx - cx, comp_cy - cy))
        score = area - float(config.get("center_distance_weight", 1.0)) * dist
        if score > best_score:
            best_score = score
            best_label = label
    if best_label is None:
        return None
    return labels == best_label
