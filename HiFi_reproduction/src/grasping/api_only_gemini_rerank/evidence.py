"""Deterministic deployment-only evidence for Gemini; never a local score."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from src.grasping.common.geometry import rectangle_corners
from src.grasping.common.sample_io import CompactSampleLoader

from .contracts import assert_no_gt_payload, validate_candidate_manifest


EVIDENCE_SCHEMA_VERSION = "api_only_gemini_evidence_v1"


def shared_evidence_context(arrays: Any) -> dict[str, Any]:
    mask_y, mask_x = np.nonzero(arrays.binary_mask)
    valid_depth = np.isfinite(arrays.depth_m) & (arrays.depth_m > 0)
    foreground = _depth_median(arrays.depth_m[valid_depth & arrays.binary_mask])
    background = _depth_median(arrays.depth_m[valid_depth & ~arrays.binary_mask])
    return {
        "signed_distance": ndimage.distance_transform_edt(arrays.binary_mask) - ndimage.distance_transform_edt(~arrays.binary_mask),
        "mask_centroid": np.asarray([mask_x.mean(), mask_y.mean()]) if len(mask_x) else np.asarray([arrays.rgb.shape[1]/2, arrays.rgb.shape[0]/2]),
        "valid_depth": valid_depth,
        "foreground_background_depth_gap_m": None if foreground is None or background is None else float(background - foreground),
    }


def closing_axis(angle_deg: float) -> np.ndarray:
    radians = math.radians(float(angle_deg))
    return np.asarray([math.cos(radians), math.sin(radians)], dtype=float)


def contact_points(candidate: Mapping[str, Any]) -> np.ndarray:
    centre = np.asarray([candidate["center_x"], candidate["center_y"]], dtype=float)
    axis = closing_axis(float(candidate["angle_deg"]))
    return np.stack(
        [centre - 0.5 * float(candidate["width_px"]) * axis,
         centre + 0.5 * float(candidate["width_px"]) * axis]
    )


def _bilinear(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=float)
    height, width = data.shape
    x = np.clip(points[:, 0], 0, width - 1)
    y = np.clip(points[:, 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    return (
        data[y0, x0] * (1 - dx) * (1 - dy)
        + data[y0, x1] * dx * (1 - dy)
        + data[y1, x0] * (1 - dx) * dy
        + data[y1, x1] * dx * dy
    )


def _patch(image: np.ndarray, point: np.ndarray, radius: int) -> np.ndarray:
    height, width = image.shape
    x, y = np.rint(point).astype(int)
    return image[max(0, y-radius):min(height, y+radius+1), max(0, x-radius):min(width, x+radius+1)]


def _polygon_values(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape
    polygon = np.rint(corners).astype(np.int32)
    clipped = polygon.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    region = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(region, clipped, 1)
    intended = max(abs(float(cv2.contourArea(polygon.astype(np.float32)))), 1.0)
    return image[region.astype(bool)], float(np.clip(region.sum() / intended, 0.0, 1.0))


def _depth_median(values: np.ndarray) -> float | None:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    return None if not len(valid) else float(np.median(valid))


def _mask_width(mask: np.ndarray, centre: np.ndarray, axis: np.ndarray) -> float:
    span = int(math.ceil(math.hypot(*mask.shape)))
    offsets = np.arange(-span, span + 1, dtype=float)
    samples = _bilinear(mask.astype(float), centre[None, :] + offsets[:, None] * axis[None, :]) >= 0.5
    middle = span
    if not samples[middle]:
        return 0.0
    left = middle
    while left > 0 and samples[left - 1]:
        left -= 1
    right = middle
    while right + 1 < len(samples) and samples[right + 1]:
        right += 1
    return float(right - left + 1)


def candidate_evidence(
    candidate: Mapping[str, Any],
    *,
    top1_score: float,
    probability: np.ndarray,
    mask: np.ndarray,
    depth_m: np.ndarray,
    shared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    height, width = mask.shape
    centre = np.asarray([candidate["center_x"], candidate["center_y"]], dtype=float)
    axis = closing_axis(float(candidate["angle_deg"]))
    contacts = contact_points(candidate)
    line_offsets = np.linspace(-0.5, 0.5, max(int(round(float(candidate["width_px"]))) + 1, 3))
    line = centre[None, :] + line_offsets[:, None] * float(candidate["width_px"]) * axis[None, :]
    corners = rectangle_corners(
        float(candidate["center_x"]), float(candidate["center_y"]),
        float(candidate["width_px"]), float(candidate["height_px"]),
        float(candidate["angle_deg"]),
    )
    rectangle_mask, inside_fraction = _polygon_values(mask.astype(float), corners)
    contact_probability = [_bilinear(probability, point[None, :])[0] for point in contacts]
    centre_p = float(_bilinear(probability, centre[None, :])[0])
    shared = dict(shared or {})
    signed_distance = shared.get("signed_distance")
    if signed_distance is None:
        signed_distance = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
    x = int(np.clip(round(centre[0]), 0, width - 1))
    y = int(np.clip(round(centre[1]), 0, height - 1))
    centroid = shared.get("mask_centroid")
    if centroid is None:
        mask_y, mask_x = np.nonzero(mask)
        centroid = np.asarray([mask_x.mean(), mask_y.mean()]) if len(mask_x) else np.asarray([width/2, height/2])
    local_depth = _patch(depth_m, centre, 6).astype(float).ravel()
    local_valid = np.isfinite(local_depth) & (local_depth > 0)
    contact_depth = [_depth_median(_patch(depth_m, point, 3)) for point in contacts]
    centre_depth = _depth_median(_patch(depth_m, centre, 1))
    contact_difference = None if None in contact_depth else abs(float(contact_depth[0]) - float(contact_depth[1]))
    valid_depth = shared.get("valid_depth")
    if valid_depth is None:
        valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    foreground_background = shared.get("foreground_background_depth_gap_m")
    if "foreground_background_depth_gap_m" not in shared:
        foreground = _depth_median(depth_m[valid_depth & mask])
        background = _depth_median(depth_m[valid_depth & ~mask])
        foreground_background = None if foreground is None or background is None else float(background - foreground)
    sweep_corners = rectangle_corners(
        float(candidate["center_x"]), float(candidate["center_y"]),
        1.25 * float(candidate["width_px"]), 2.5 * float(candidate["height_px"]),
        float(candidate["angle_deg"]),
    )
    sweep_mask, _ = _polygon_values(mask.astype(float), sweep_corners)
    sweep_depth, _ = _polygon_values(depth_m.astype(float), sweep_corners)
    sweep_valid_depth = sweep_depth[np.isfinite(sweep_depth) & (sweep_depth > 0)]
    mask_width = _mask_width(mask, centre, axis)
    score = float(candidate["original_score"])
    top = float(top1_score)
    payload = {
        "candidate_id": str(candidate["candidate_id"]),
        "original_rank": int(candidate["original_rank"]),
        "original_score": score,
        "score_to_top1_ratio": None if abs(top) <= 1e-12 else score / top,
        "score_gap_to_original_top1": top - score,
        "center_x": float(candidate["center_x"]),
        "center_y": float(candidate["center_y"]),
        "angle_deg": float(candidate["angle_deg"]),
        "width_px": float(candidate["width_px"]),
        "height_px": float(candidate["height_px"]),
        "contact_points_xy": contacts.tolist(),
        "center_probability": centre_p,
        "centre_in_predicted_mask": bool(mask[y, x]),
        "rectangle_mask_coverage": float(rectangle_mask.mean()) if len(rectangle_mask) else 0.0,
        "grasp_axis_mask_support": float((_bilinear(mask.astype(float), line) >= 0.5).mean()),
        "left_contact_mask_support": float(contact_probability[0]),
        "right_contact_mask_support": float(contact_probability[1]),
        "minimum_contact_mask_support": float(min(contact_probability)),
        "distance_to_mask_centroid_px": float(np.linalg.norm(centre - centroid)),
        "distance_to_mask_boundary_px": float(signed_distance[y, x]),
        "mask_width_along_closing_axis_px": mask_width,
        "candidate_width_to_mask_width_ratio": None if mask_width <= 0 else float(candidate["width_px"]) / mask_width,
        "center_depth_m": centre_depth,
        "center_depth_valid": centre_depth is not None,
        "local_valid_depth_fraction": float(local_valid.mean()) if len(local_valid) else 0.0,
        "left_contact_median_depth_m": contact_depth[0],
        "right_contact_median_depth_m": contact_depth[1],
        "absolute_contact_depth_difference_m": contact_difference,
        "local_depth_std_m": float(local_depth[local_valid].std()) if local_valid.any() else None,
        "depth_gradient_along_closing_axis": float(
            _bilinear(depth_m, (centre + axis)[None, :])[0] - _bilinear(depth_m, (centre - axis)[None, :])[0]
        ) if centre_depth is not None else None,
        "foreground_background_depth_gap_m": foreground_background,
        "missing_depth_fraction": float(1.0 - valid_depth.mean()),
        "contact_points_inside_image": bool(np.all((contacts[:, 0] >= 0) & (contacts[:, 0] < width) & (contacts[:, 1] >= 0) & (contacts[:, 1] < height))),
        "gripper_rectangle_inside_image_fraction": inside_fraction,
        "predicted_mask_support_at_both_contacts": bool(min(contact_probability) >= 0.5),
        "sweep_region_mask_fraction": float(sweep_mask.mean()) if len(sweep_mask) else 0.0,
        "sweep_region_depth_discontinuity": float(sweep_valid_depth.std()) if len(sweep_valid_depth) else None,
        "border_distance_px": float(min(centre[0], centre[1], width-1-centre[0], height-1-centre[1])),
    }
    assert_no_gt_payload(payload)
    return payload


def extract_sample_evidence(
    candidates: pd.DataFrame,
    deployment: Mapping[str, Any],
    loader: CompactSampleLoader,
    arrays: Any | None = None,
    shared: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if candidates.empty:
        return []
    validate_candidate_manifest(candidates)
    arrays = arrays if arrays is not None else loader.load(deployment, mask_source="predicted", load_intrinsics=False)
    shared = dict(shared or shared_evidence_context(arrays))
    ordered = candidates.sort_values(["original_rank", "candidate_id"], kind="mergesort")
    top = float(ordered.iloc[0]["original_score"])
    return [
        {
            "backend": str(row["backend"]),
            "split": str(row["split"]),
            "sample_id": str(row["sample_id"]),
            "scene_id": str(row["scene_id"]),
            "language": arrays.language,
            "image_width": int(arrays.rgb.shape[1]),
            "image_height": int(arrays.rgb.shape[0]),
            "source_rgb_path": str(deployment["source_rgb_path"]),
            "source_rgb_sha256": str(deployment["source_rgb_sha256"]),
            "source_depth_path": str(deployment["source_depth_path"]),
            "source_depth_sha256": str(deployment["source_depth_sha256"]),
            "predicted_mask_path": str(deployment["predicted_mask_path"]),
            "predicted_mask_sha256": str(deployment["predicted_mask_sha256"]),
            "predicted_probability_path": str(deployment["predicted_probability_path"]),
            "predicted_probability_sha256": str(deployment["predicted_probability_sha256"]),
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            **candidate_evidence(
                row,
                top1_score=top,
                probability=arrays.probability,
                mask=arrays.binary_mask,
                depth_m=arrays.depth_m,
                shared=shared,
            ),
        }
        for row in ordered.to_dict(orient="records")
    ]


def extract_evidence_table(
    manifest: pd.DataFrame,
    deployment_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    by_sample = {str(sample_id): group.copy() for sample_id, group in manifest.groupby("sample_id", sort=False)}
    loader = CompactSampleLoader()
    rows: list[dict[str, Any]] = []
    for deployment in deployment_rows:
        rows.extend(extract_sample_evidence(by_sample.get(str(deployment["sample_id"]), manifest.iloc[:0]), deployment, loader))
    output = pd.DataFrame(rows)
    assert_no_gt_payload({"columns": output.columns.tolist()})
    return output


def evidence_schema() -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "role": "Gemini input evidence only; forbidden for local ranking or learned gating",
        "coordinate_convention": "640x480; origin top-left; x=column right; y=row down; angle is closing/width axis; 180-degree periodic",
        "physical_claim": False,
        "missing_value_rule": "null plus explicit validity/fraction fields",
    }
