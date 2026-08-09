from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy import ndimage

from failure_analysis.failure_utils import rle_to_mask
from failure_analysis.reranking.feature_extraction import load_depth_m
from failure_analysis.reranking.geometry import (
    candidate_contact_bands,
    frozen_candidate_signature,
    local_coordinates,
    rasterize_candidate,
)


FEATURE_SCHEMA_VERSION = "pairwise_deterministic_v1"
COORDINATE_CONVENTION = {
    "image_size": [640, 480],
    "origin": "top-left",
    "x_axis": "column, positive right",
    "y_axis": "row, positive down",
    "coordinates": "original image pixels",
    "angle_unit": "degrees",
    "angle_definition": "jaw closing/opening axis; passed negated to OpenCV image rotation",
    "angle_symmetry": "theta and theta+180 degrees are equivalent",
    "width_definition": "jaw opening along closing axis in pixels",
    "rectangle_height": "finger-contact extent perpendicular to closing axis in pixels",
}


def ordered_candidates(feature: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = list(feature["candidates"])
    if len(candidates) != 5:
        raise ValueError("frozen candidate pool must contain exactly five candidates")
    frozen_candidate_signature(candidates)
    ordered = sorted(candidates, key=lambda row: (-float(row["q_raw"]), str(row["candidate_id"])))
    for rank, row in enumerate(ordered):
        if int(row["q_rank"]) != rank:
            raise AssertionError("stored q rank differs from deterministic frozen order")
    return ordered


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _stored(candidate: Mapping[str, Any], name: str) -> tuple[float | None, float, str | None]:
    row = candidate.get("features", {}).get(name, {})
    if not isinstance(row, Mapping):
        return None, 0.0, "feature_absent"
    return _finite(row.get("value")), float(row.get("reliability", 0.0) or 0.0), row.get("missing_reason")


def _median(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    return float(np.median(values)) if values.size else None


def _raw_candidate_features(
    candidate: Mapping[str, Any],
    *,
    rank: int,
    baseline_q: float,
    mask: np.ndarray,
    depth_m: np.ndarray | None,
) -> dict[str, Any]:
    shape = mask.shape
    visible, full_count, _ = rasterize_candidate(candidate, shape)
    contact_left, contact_right = candidate_contact_bands(candidate, shape)
    u, v = local_coordinates(candidate, shape)
    axis_band = (
        (np.abs(u) <= max(float(candidate["width_px"]) / 2.0, 1.0))
        & (np.abs(v) <= 1.5)
    )
    row, col = int(candidate["row"]), int(candidate["col"])
    centre_in = bool(0 <= row < shape[0] and 0 <= col < shape[1] and mask[row, col])
    mask_count = int(mask.sum())
    if mask_count:
        rr, cc = np.nonzero(mask)
        centroid_x, centroid_y = float(np.mean(cc)), float(np.mean(rr))
        distance_centroid = float(math.hypot(float(candidate["cx"]) - centroid_x, float(candidate["cy"]) - centroid_y))
        signed = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
        distance_boundary = float(signed[row, col]) if 0 <= row < shape[0] and 0 <= col < shape[1] else None
    else:
        distance_centroid = distance_boundary = None
    rectangle_coverage = float(np.mean(mask[visible])) if visible.any() else 0.0
    image_support = float(visible.sum() / max(full_count, 1))
    axis_support = float(np.mean(mask[axis_band])) if axis_band.any() else 0.0
    left_support = float(np.mean(mask[contact_left])) if contact_left.any() else 0.0
    right_support = float(np.mean(mask[contact_right])) if contact_right.any() else 0.0

    depth_available = depth_m is not None and depth_m.shape == shape
    valid_fraction = 0.0
    centre_depth = left_depth = right_depth = depth_difference = variance = None
    if depth_available:
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        valid_fraction = float(np.mean(valid[visible])) if visible.any() else 0.0
        centre_patch = depth_m[max(0, row - 2):min(shape[0], row + 3), max(0, col - 2):min(shape[1], col + 3)]
        centre_depth = _median(centre_patch)
        left_depth = _median(depth_m[contact_left])
        right_depth = _median(depth_m[contact_right])
        if left_depth is not None and right_depth is not None:
            depth_difference = abs(left_depth - right_depth)
        local = np.asarray(depth_m[visible], dtype=np.float64)
        local = local[np.isfinite(local) & (local > 0.0)]
        variance = float(np.var(local)) if local.size else None

    width_compatibility, width_rel, width_missing = _stored(candidate, "width_compatibility")
    clearance, clearance_rel, clearance_missing = _stored(candidate, "clearance")
    collision, collision_rel, collision_missing = _stored(candidate, "collision_proxy")
    q_prominence, q_prominence_rel, q_prominence_missing = _stored(candidate, "q_prominence")
    diagnostics = candidate.get("diagnostics", {})
    object_width_px = _finite(diagnostics.get("object_width_px"))
    depth_reliable = bool(depth_available and valid_fraction >= 0.5)
    hard_valid = bool(
        image_support >= 0.95
        and float(candidate["width_px"]) > 0
        and float(candidate["height_px"]) > 0
        and 0 <= float(candidate["cx"]) < shape[1]
        and 0 <= float(candidate["cy"]) < shape[0]
    )
    reliability_values = [image_support, min(rectangle_coverage * 2.0, 1.0), float(width_rel), float(clearance_rel), float(collision_rel)]
    if depth_available:
        reliability_values.append(valid_fraction)
    aggregate_reliability = float(np.clip(np.mean(reliability_values), 0.0, 1.0))
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_checksum": str(candidate["candidate_checksum"]),
        "original_rank": int(rank),
        "q": float(candidate["q_raw"]),
        "delta_q_to_baseline": float(candidate["q_raw"]) - float(baseline_q),
        "q_prominence": q_prominence,
        "centre_x": float(candidate["cx"]),
        "centre_y": float(candidate["cy"]),
        "angle_deg": float(candidate["angle_deg"]),
        "width_px": float(candidate["width_px"]),
        "height_px": float(candidate["height_px"]),
        "centre_in_predicted_mask": centre_in,
        "mask_rectangle_coverage": rectangle_coverage,
        "mask_axis_support": axis_support,
        "mask_contact_support_left": left_support,
        "mask_contact_support_right": right_support,
        "distance_to_mask_centroid_px": distance_centroid,
        "signed_distance_to_mask_boundary_px": distance_boundary,
        "valid_depth_fraction": valid_fraction,
        "centre_depth_m": centre_depth,
        "left_contact_median_depth_m": left_depth,
        "right_contact_median_depth_m": right_depth,
        "absolute_contact_depth_difference_m": depth_difference,
        "local_depth_variance_m2": variance,
        "estimated_object_width_px": object_width_px,
        "estimated_object_width_m": None,
        "grasp_width_compatibility": width_compatibility,
        "clearance_proxy": clearance,
        "collision_proxy": collision,
        "collision_proxy_name": "relative_2p5d_obstacle_proxy",
        "metric_3d_collision_available": False,
        "depth_missing": not depth_available,
        "depth_reliable": depth_reliable,
        "hard_valid": hard_valid,
        "aggregate_reliability": aggregate_reliability,
        "missingness": {
            "q_prominence": q_prominence_missing,
            "width_compatibility": width_missing,
            "clearance": clearance_missing,
            "collision": collision_missing,
            "depth": None if depth_available else "depth_unavailable",
        },
        "reliability": {
            "q_prominence": q_prominence_rel,
            "width_compatibility": width_rel,
            "clearance": clearance_rel,
            "collision": collision_rel,
            "depth_valid_fraction": valid_fraction,
        },
    }


def build_pair_evidence(
    feature: Mapping[str, Any], challenger_id: str, *, load_depth: bool = True
) -> dict[str, Any]:
    candidates = ordered_candidates(feature)
    by_id = {str(row["candidate_id"]): row for row in candidates}
    baseline = candidates[0]
    if challenger_id not in by_id or challenger_id == str(baseline["candidate_id"]):
        raise ValueError("challenger must be a different member of the frozen Top-5")
    mask = np.asarray(rle_to_mask(feature.get("predicted_mask_rle")), dtype=bool)
    if mask.shape != (480, 640):
        raise ValueError(f"unexpected authoritative mask shape: {mask.shape}")
    depth = None
    if load_depth and feature.get("depth_path"):
        depth_path = Path(str(feature["depth_path"]))
        if depth_path.is_file():
            depth, _ = load_depth_m(depth_path, expected_shape=mask.shape)
    ranks = {str(row["candidate_id"]): index for index, row in enumerate(candidates)}
    baseline_row = _raw_candidate_features(
        baseline, rank=0, baseline_q=float(baseline["q_raw"]), mask=mask, depth_m=depth
    )
    challenger = by_id[challenger_id]
    challenger_row = _raw_candidate_features(
        challenger,
        rank=ranks[challenger_id],
        baseline_q=float(baseline["q_raw"]),
        mask=mask,
        depth_m=depth,
    )
    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sample_id": str(feature.get("sample_id", feature.get("sample_index"))),
        "language_instruction": str(feature["language_instruction"]),
        "coordinate_convention": COORDINATE_CONVENTION,
        "baseline": baseline_row,
        "challenger": challenger_row,
    }
    payload["evidence_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return payload


def local_feature_vector(pair: Mapping[str, Any], *, include_q: bool = True) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[float] = []
    fields = [
        "mask_rectangle_coverage", "mask_axis_support", "mask_contact_support_left",
        "mask_contact_support_right", "distance_to_mask_centroid_px",
        "signed_distance_to_mask_boundary_px", "valid_depth_fraction", "centre_depth_m",
        "absolute_contact_depth_difference_m", "local_depth_variance_m2",
        "estimated_object_width_px", "grasp_width_compatibility", "clearance_proxy",
        "collision_proxy", "aggregate_reliability",
    ]
    if include_q:
        fields = ["q", "delta_q_to_baseline", "q_prominence"] + fields
    for side in ("baseline", "challenger"):
        row = pair[side]
        for field in fields:
            name = f"{side}_{field}"
            value = _finite(row.get(field))
            names.extend((name, f"{name}_missing"))
            values.extend((0.0 if value is None else value, 1.0 if value is None else 0.0))
    return names, np.asarray(values, dtype=np.float64)


def select_challengers(feature: Mapping[str, Any], *, maximum: int = 2) -> list[str]:
    """Frozen preselector: q rank plus deterministic evidence only, never labels."""

    if not 1 <= maximum <= 4:
        raise ValueError("maximum challengers must be between one and four")
    candidates = ordered_candidates(feature)
    baseline_q = float(candidates[0]["q_raw"])
    scored = []
    for candidate in candidates[1:]:
        mask_consistency, mask_rel, _ = _stored(candidate, "mask_consistency")
        safety, safety_rel, _ = _stored(candidate, "safety")
        width, width_rel, _ = _stored(candidate, "width_compatibility")
        q_gap = max(0.0, baseline_q - float(candidate["q_raw"]))
        local = (
            0.55 * float(candidate["q_raw"])
            + 0.20 * (mask_consistency if mask_consistency is not None else 0.5) * mask_rel
            + 0.15 * (safety if safety is not None else 0.5) * safety_rel
            + 0.10 * (width if width is not None else 0.5) * width_rel
            - 0.10 * q_gap
        )
        scored.append((-local, int(candidate["q_rank"]), str(candidate["candidate_id"])))
    return [candidate_id for _, _, candidate_id in sorted(scored)[:maximum]]


def stored_local_feature_vector(
    feature: Mapping[str, Any], challenger_id: str, *, include_q: bool = True
) -> tuple[list[str], np.ndarray]:
    """Fast frozen V2 feature vector for full calibration/validation replay."""

    candidates = ordered_candidates(feature)
    by_id = {str(row["candidate_id"]): row for row in candidates}
    if challenger_id not in by_id or challenger_id == str(candidates[0]["candidate_id"]):
        raise ValueError("challenger identity is not a frozen non-baseline candidate")
    fields = [
        "center_prob", "soft_coverage", "binary_coverage", "center_margin",
        "mask_consistency", "angle_consistency", "depth_mad_m",
        "contact_depth_difference_m", "width_ratio", "width_symmetry",
        "width_compatibility", "clearance", "collision_proxy", "safety",
    ]
    if include_q:
        fields = ["q", "q_patch_mean", "q_prominence"] + fields
    names: list[str] = []
    values: list[float] = []
    baseline_q = float(candidates[0]["q_raw"])
    for side, candidate in (("baseline", candidates[0]), ("challenger", by_id[challenger_id])):
        names.extend((f"{side}_q_delta_to_baseline", f"{side}_q_rank"))
        values.extend((float(candidate["q_raw"]) - baseline_q, float(candidate["q_rank"])))
        for field in fields:
            value, reliability, _ = _stored(candidate, field)
            names.extend((f"{side}_{field}", f"{side}_{field}_reliability", f"{side}_{field}_missing"))
            values.extend((0.0 if value is None else value, reliability, 1.0 if value is None else 0.0))
        diagnostics = candidate.get("diagnostics", {})
        for field in ("object_width_px", "valid_scanline_fraction", "nearest_obstacle_distance_px"):
            value = _finite(diagnostics.get(field))
            names.extend((f"{side}_{field}", f"{side}_{field}_missing"))
            values.extend((0.0 if value is None else value, 1.0 if value is None else 0.0))
        names.append(f"{side}_depth_available")
        values.append(float(bool(diagnostics.get("depth_available", False))))
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise AssertionError("stored local feature vector contains non-finite values")
    return names, vector
