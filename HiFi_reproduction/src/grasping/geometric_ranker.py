"""Transparent target-aware ranking of frozen post-NMS planar grasps.

This module deliberately has no dependency on TensorFlow or GQ-CNN model
scoring.  It treats ``candidates.npz`` as the numeric source of truth and uses
the adjacent JSON only for fields that the legacy NPZ schema did not retain
(candidate IDs, antipodal contacts, and contact normals).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .camera_geometry import (
    T_CAMERA_GRASP_FIXED_APPROACH_KEY,
    CameraIntrinsicsData,
    width_pixels_to_meters,
)
from .mask_processing import binary_dilate, valid_depth_mask


REQUIRED_NPZ_KEYS = (
    "center_uv",
    "center_depth_m",
    "center_camera_xyz_m",
    "angle_rad",
    "width_m",
    "width_px",
    "endpoints_uv",
    "mask_support",
    "boundary_distance_px",
    "gqcnn_q_value",
    "valid",
    T_CAMERA_GRASP_FIXED_APPROACH_KEY,
)

RAW_FEATURE_SPECS: Mapping[str, Tuple[bool, bool]] = {
    # name: (larger_is_better, naturally_bounded_0_1)
    "contact_endpoint_1_support_raw": (True, True),
    "contact_endpoint_2_support_raw": (True, True),
    "contact_support_raw": (True, True),
    "jaw_endpoint_1_support_raw": (True, True),
    "jaw_endpoint_2_support_raw": (True, True),
    "axis_mask_support_raw": (True, True),
    "boundary_distance_px_raw": (True, False),
    "centroid_distance_px_raw": (False, False),
    "valid_depth_support_raw": (True, True),
    "closing_depth_variance_m2_raw": (False, False),
    "closing_depth_std_m_raw": (False, False),
    "local_antipodal_score_raw": (True, True),
    "normal_1_axis_alignment_raw": (True, True),
    "normal_2_axis_alignment_raw": (True, True),
    "normal_pair_opposition_raw": (True, True),
    "contact_span_px_raw": (True, False),
    "contact_span_m_raw": (True, False),
    "width_score_raw": (True, True),
    "image_boundary_margin_px_raw": (True, False),
    "image_edge_penalty_raw": (False, True),
    "interference_penalty_raw": (False, True),
}

COMPONENT_NAMES = (
    "contact_support",
    "axis_mask_support",
    "boundary_margin",
    "local_antipodal_score",
    "depth_support",
    "width_score",
    "target_centrality",
    "interference_penalty",
    "image_edge_penalty",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_intrinsics(path: Path | str) -> CameraIntrinsicsData:
    """Load the JSON-formatted ``.intr`` emitted by autolab-core."""

    values = json.loads(Path(path).read_text(encoding="utf-8"))
    mapped = {
        "frame": values.get("frame", values.get("_frame")),
        "fx": values.get("fx", values.get("_fx")),
        "fy": values.get("fy", values.get("_fy")),
        "cx": values.get("cx", values.get("_cx")),
        "cy": values.get("cy", values.get("_cy")),
        "skew": values.get("skew", values.get("_skew", 0.0)),
        "height": values.get("height", values.get("_height")),
        "width": values.get("width", values.get("_width")),
    }
    return CameraIntrinsicsData.from_mapping(mapped)


def _candidate_records(json_path: Path) -> Tuple[list[dict[str, Any]], Mapping[str, Any]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError(f"{json_path} must contain a candidate list")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{json_path} metadata must be an object")
    return payload["candidates"], metadata


def _json_array(
    records: Sequence[Mapping[str, Any]],
    field: str,
    *,
    dtype: Any,
    shape: Tuple[int, ...],
) -> np.ndarray:
    try:
        array = np.asarray([record[field] for record in records], dtype=dtype)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"candidate JSON has invalid {field}") from error
    expected = (len(records),) + shape
    if array.shape != expected:
        raise ValueError(f"candidate JSON {field} has shape {array.shape}, expected {expected}")
    return array


def _assert_same(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"NPZ/JSON mismatch for {name}: {actual.shape}/{actual.dtype} versus "
            f"{expected.shape}/{expected.dtype}"
        )
    if not np.array_equal(actual, expected, equal_nan=True):
        raise ValueError(f"NPZ/JSON numeric mismatch for {name}")


def load_frozen_candidates(
    npz_path: Path | str,
    json_path: Path | str | None = None,
) -> Tuple[list[dict[str, Any]], Mapping[str, Any], Dict[str, str]]:
    """Load frozen candidates, failing closed if the sidecar disagrees.

    IDs cannot be reconstructed from post-NMS row positions because their
    sampler IDs are intentionally non-contiguous.
    """

    npz_path = Path(npz_path)
    json_path = Path(json_path) if json_path is not None else npz_path.with_suffix(".json")
    records, metadata = _candidate_records(json_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_NPZ_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"candidate NPZ is missing keys: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in REQUIRED_NPZ_KEYS}

    count = arrays["center_uv"].shape[0]
    if len(records) != count:
        raise ValueError(f"NPZ/JSON candidate count mismatch: {count} versus {len(records)}")
    ids = [record.get("candidate_id") for record in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("every frozen candidate must retain a non-empty candidate_id")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")

    scalar_fields = {
        "center_depth_m": "center_depth_m",
        "angle_rad": "angle_rad",
        "width_m": "width_m",
        "width_px": "width_px",
        "mask_support": "grasp_axis_mask_support",
        "boundary_distance_px": "centre_boundary_distance_px",
        "gqcnn_q_value": "gqcnn_q_value",
    }
    _assert_same(
        "center_uv",
        arrays["center_uv"],
        np.asarray(
            [[record["center_u_px"], record["center_v_px"]] for record in records],
            dtype=arrays["center_uv"].dtype,
        ),
    )
    _assert_same(
        "center_camera_xyz_m",
        arrays["center_camera_xyz_m"],
        _json_array(
            records,
            "center_camera_xyz_m",
            dtype=arrays["center_camera_xyz_m"].dtype,
            shape=(3,),
        ),
    )
    _assert_same(
        "endpoints_uv",
        arrays["endpoints_uv"],
        np.asarray(
            [[record["endpoint_1_uv"], record["endpoint_2_uv"]] for record in records],
            dtype=arrays["endpoints_uv"].dtype,
        ),
    )
    _assert_same(
        T_CAMERA_GRASP_FIXED_APPROACH_KEY,
        arrays[T_CAMERA_GRASP_FIXED_APPROACH_KEY],
        _json_array(
            records,
            T_CAMERA_GRASP_FIXED_APPROACH_KEY,
            dtype=arrays[T_CAMERA_GRASP_FIXED_APPROACH_KEY].dtype,
            shape=(4, 4),
        ),
    )
    for array_name, field_name in scalar_fields.items():
        values = []
        for record in records:
            value = record.get(field_name)
            values.append(np.nan if value is None else float(value))
        _assert_same(
            array_name,
            arrays[array_name],
            np.asarray(values, dtype=arrays[array_name].dtype),
        )

    expected_valid = np.asarray(
        [
            record.get("rejection_reason") in (None, "")
            and record.get("centre_inside_mask") is not False
            for record in records
        ],
        dtype=arrays["valid"].dtype,
    )
    _assert_same("valid", arrays["valid"], expected_valid)
    if not np.all(arrays["valid"]):
        raise ValueError("ranker accepts only the frozen valid post-NMS candidate set")

    output: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        record = deepcopy(dict(source))
        contacts = np.asarray(record.get("contact_points_uv"), dtype=np.float64)
        normals = np.asarray(record.get("contact_normals"), dtype=np.float64)
        if contacts.shape != (2, 2) or not np.all(np.isfinite(contacts)):
            raise ValueError(f"candidate {ids[index]} has invalid antipodal contacts")
        if normals.shape != (2, 2) or not np.all(np.isfinite(normals)):
            raise ValueError(f"candidate {ids[index]} has invalid contact normals")
        record.update(
            {
                "center_uv": arrays["center_uv"][index].astype(float).tolist(),
                "center_u_px": float(arrays["center_uv"][index, 0]),
                "center_v_px": float(arrays["center_uv"][index, 1]),
                "center_depth_m": float(arrays["center_depth_m"][index]),
                "center_camera_xyz_m": arrays["center_camera_xyz_m"][index].astype(float).tolist(),
                "angle_rad": float(arrays["angle_rad"][index]),
                "angle_deg": float(np.degrees(arrays["angle_rad"][index])),
                "width_m": float(arrays["width_m"][index]),
                "width_px": float(arrays["width_px"][index]),
                "endpoints_uv": arrays["endpoints_uv"][index].astype(float).tolist(),
                "endpoint_1_uv": arrays["endpoints_uv"][index, 0].astype(float).tolist(),
                "endpoint_2_uv": arrays["endpoints_uv"][index, 1].astype(float).tolist(),
                T_CAMERA_GRASP_FIXED_APPROACH_KEY: arrays[
                    T_CAMERA_GRASP_FIXED_APPROACH_KEY
                ][index].astype(float).tolist(),
                "gqcnn_q_value": None,
                "gqcnn_rank": None,
                "source_candidate_valid": True,
                "nms_status": "kept_post_nms",
                "is_duplicate": False,
            }
        )
        output.append(record)
    return output, metadata, {
        "candidates_npz_sha256": sha256_file(npz_path),
        "candidates_json_sha256": sha256_file(json_path),
    }


def _disk_fraction(mask: np.ndarray, point_uv: Sequence[float], radius: int) -> float:
    u, v = np.rint(np.asarray(point_uv, dtype=np.float64)).astype(int)
    height, width = mask.shape
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - u) ** 2 + (yy - v) ** 2 <= radius**2
    count = int(np.count_nonzero(disk))
    return 0.0 if count == 0 else float(np.count_nonzero(mask[y0:y1, x0:x1] & disk) / count)


def _line_indices(
    first_uv: Sequence[float],
    second_uv: Sequence[float],
    shape: Tuple[int, int],
    spacing_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first_uv, dtype=np.float64)
    second = np.asarray(second_uv, dtype=np.float64)
    distance = float(np.linalg.norm(second - first))
    count = max(2, int(math.ceil(distance / max(float(spacing_px), 1e-6))) + 1)
    points = np.linspace(first, second, count)
    uu = np.rint(points[:, 0]).astype(int)
    vv = np.rint(points[:, 1]).astype(int)
    keep = (uu >= 0) & (uu < shape[1]) & (vv >= 0) & (vv < shape[0])
    if not np.any(keep):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    pairs = np.unique(np.stack((vv[keep], uu[keep]), axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _oriented_region_indices(
    first_uv: Sequence[float],
    second_uv: Sequence[float],
    half_height_px: float,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first_uv, dtype=np.float64)
    second = np.asarray(second_uv, dtype=np.float64)
    delta = second - first
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    axis = delta / length
    normal = np.array([-axis[1], axis[0]])
    center = (first + second) * 0.5
    half_length = length * 0.5
    corners = np.stack(
        [
            center + sx * half_length * axis + sy * half_height_px * normal
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        ]
    )
    x0 = max(0, int(math.floor(np.min(corners[:, 0]))))
    x1 = min(shape[1] - 1, int(math.ceil(np.max(corners[:, 0]))))
    y0 = max(0, int(math.floor(np.min(corners[:, 1]))))
    y1 = min(shape[0] - 1, int(math.ceil(np.max(corners[:, 1]))))
    if x0 > x1 or y0 > y1:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    vv, uu = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    rel_u = uu.astype(np.float64) - center[0]
    rel_v = vv.astype(np.float64) - center[1]
    along = rel_u * axis[0] + rel_v * axis[1]
    across = rel_u * normal[0] + rel_v * normal[1]
    keep = (np.abs(along) <= half_length + 0.5) & (
        np.abs(across) <= float(half_height_px) + 0.5
    )
    return vv[keep], uu[keep]


def _trapezoid_score(value: float, config: Mapping[str, Any]) -> float:
    hard_min = float(config["hard_min_m"])
    preferred_min = float(config["preferred_min_m"])
    preferred_max = float(config["preferred_max_m"])
    hard_max = float(config["hard_max_m"])
    if not hard_min <= preferred_min <= preferred_max <= hard_max:
        raise ValueError("width plausibility thresholds must be monotonic")
    if value <= hard_min or value >= hard_max:
        return 0.0
    if preferred_min <= value <= preferred_max:
        return 1.0
    if value < preferred_min:
        return float((value - hard_min) / max(preferred_min - hard_min, 1e-12))
    return float((hard_max - value) / max(hard_max - preferred_max, 1e-12))


def _robust_normalize(
    values: np.ndarray,
    *,
    bounded: bool,
    lower_quantile: float,
    upper_quantile: float,
    epsilon: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("normalization values must be a finite vector")
    low = float(np.quantile(values, lower_quantile))
    high = float(np.quantile(values, upper_quantile))
    degenerate = bool(high - low <= epsilon)
    if degenerate:
        if bounded:
            normalized = np.clip(values, 0.0, 1.0)
            fallback = "preserve_raw_bounded_0_1"
        else:
            normalized = np.full_like(values, 0.5)
            fallback = "neutral_0.5"
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        fallback = None
    stats = {
        "minimum": float(np.min(values)),
        "lower_quantile": low,
        "median": float(np.median(values)),
        "upper_quantile": high,
        "maximum": float(np.max(values)),
        "degenerate": degenerate,
        "fallback": fallback,
    }
    return normalized, stats


def compute_raw_features(
    record: Mapping[str, Any],
    *,
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    dilated_mask: np.ndarray,
    boundary_distance: np.ndarray,
    centroid_uv: np.ndarray,
    intrinsics: CameraIntrinsicsData,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    geometry = config["geometry"]
    contacts = np.asarray(record["contact_points_uv"], dtype=np.float64)
    normals = np.asarray(record["contact_normals"], dtype=np.float64)
    jaws = np.asarray(record["endpoints_uv"], dtype=np.float64)
    center = np.asarray(record["center_uv"], dtype=np.float64)
    valid_depth = valid_depth_mask(depth_m)
    support_radius = int(geometry["contact_support_radius_px"])
    jaw_radius = int(geometry["jaw_endpoint_support_radius_px"])

    c1 = _disk_fraction(dilated_mask, contacts[0], support_radius)
    c2 = _disk_fraction(dilated_mask, contacts[1], support_radius)
    j1 = _disk_fraction(dilated_mask, jaws[0], jaw_radius)
    j2 = _disk_fraction(dilated_mask, jaws[1], jaw_radius)
    line_v, line_u = _line_indices(
        contacts[0], contacts[1], target_mask.shape, float(geometry["line_sample_spacing_px"])
    )
    axis_support = float(np.mean(dilated_mask[line_v, line_u])) if line_u.size else 0.0
    depth_support = float(np.mean(valid_depth[line_v, line_u])) if line_u.size else 0.0

    center_u = int(np.clip(np.rint(center[0]), 0, target_mask.shape[1] - 1))
    center_v = int(np.clip(np.rint(center[1]), 0, target_mask.shape[0] - 1))
    boundary = float(boundary_distance[center_v, center_u])
    centroid_distance = float(np.linalg.norm(center - centroid_uv))

    close_v, close_u = _oriented_region_indices(
        contacts[0],
        contacts[1],
        float(geometry["closing_region_half_height_px"]),
        target_mask.shape,
    )
    close_keep = target_mask[close_v, close_u] & valid_depth[close_v, close_u]
    close_values = depth_m[close_v[close_keep], close_u[close_keep]].astype(np.float64)
    if close_values.size:
        lower = float(np.quantile(close_values, float(geometry["depth_winsor_lower_quantile"])))
        upper = float(np.quantile(close_values, float(geometry["depth_winsor_upper_quantile"])))
        winsorized = np.clip(close_values, lower, upper)
        depth_variance = float(np.var(winsorized))
        depth_std = float(np.std(winsorized))
        reference_depth = float(np.median(winsorized))
    else:
        depth_variance = 0.0
        depth_std = 0.0
        reference_depth = float(record["center_depth_m"])

    first_to_second = contacts[1] - contacts[0]
    contact_span_px = float(np.linalg.norm(first_to_second))
    if contact_span_px <= 1e-9:
        alignment_1 = alignment_2 = opposition = antipodal = 0.0
    else:
        inward_for_first = (contacts[0] - contacts[1]) / contact_span_px
        n1 = normals[0] / max(float(np.linalg.norm(normals[0])), 1e-12)
        n2 = normals[1] / max(float(np.linalg.norm(normals[1])), 1e-12)
        alignment_1 = float(np.clip(np.dot(n1, inward_for_first), 0.0, 1.0))
        alignment_2 = float(np.clip(np.dot(n2, -inward_for_first), 0.0, 1.0))
        opposition = float(np.clip(-np.dot(n1, n2), 0.0, 1.0))
        antipodal = float(min(alignment_1, alignment_2) * opposition)
    contact_angle = float(math.atan2(first_to_second[1], first_to_second[0]))
    contact_span_m = width_pixels_to_meters(
        contact_span_px,
        float(record["center_depth_m"]),
        intrinsics,
        contact_angle,
    )
    width_score = _trapezoid_score(contact_span_m, config["width_plausibility"])

    height, width = target_mask.shape
    margins = np.concatenate(
        (
            jaws[:, 0],
            jaws[:, 1],
            (width - 1) - jaws[:, 0],
            (height - 1) - jaws[:, 1],
        )
    )
    image_margin = float(np.min(margins))
    safe_margin = float(geometry["image_safe_margin_px"])
    edge_penalty = float(np.clip((safe_margin - image_margin) / max(safe_margin, 1e-12), 0.0, 1.0))

    sweep_v, sweep_u = _oriented_region_indices(
        jaws[0], jaws[1], float(geometry["sweep_region_half_height_px"]), target_mask.shape
    )
    exclusion = binary_dilate(target_mask, int(geometry["interference_target_clearance_px"]))
    hazard_keep = valid_depth[sweep_v, sweep_u] & ~exclusion[sweep_v, sweep_u]
    hazard_depth = depth_m[sweep_v[hazard_keep], sweep_u[hazard_keep]].astype(np.float64)
    if hazard_depth.size:
        clearance = float(geometry["interference_depth_clearance_m"])
        band = float(geometry["interference_depth_band_m"])
        hazard = np.clip((reference_depth - clearance - hazard_depth) / max(band, 1e-12), 0.0, 1.0)
        interference = float(np.mean(hazard))
    else:
        interference = 0.0

    return {
        "contact_endpoint_1_support_raw": c1,
        "contact_endpoint_2_support_raw": c2,
        "contact_support_raw": min(c1, c2),
        "jaw_endpoint_1_support_raw": j1,
        "jaw_endpoint_2_support_raw": j2,
        "axis_mask_support_raw": axis_support,
        "boundary_distance_px_raw": boundary,
        "centroid_distance_px_raw": centroid_distance,
        "valid_depth_support_raw": depth_support,
        "closing_depth_variance_m2_raw": depth_variance,
        "closing_depth_std_m_raw": depth_std,
        "closing_valid_pixel_count": int(close_values.size),
        "local_antipodal_score_raw": antipodal,
        "normal_1_axis_alignment_raw": alignment_1,
        "normal_2_axis_alignment_raw": alignment_2,
        "normal_pair_opposition_raw": opposition,
        "contact_span_px_raw": contact_span_px,
        "contact_span_m_raw": float(contact_span_m),
        "width_score_raw": width_score,
        "image_boundary_margin_px_raw": image_margin,
        "image_edge_penalty_raw": edge_penalty,
        "interference_penalty_raw": interference,
        "interference_valid_pixel_count": int(hazard_depth.size),
        "closing_reference_depth_m": reference_depth,
        "contact_geometry_source": "official_antipodal_contact_points",
        "jaw_geometry_source": "frozen_npz_maximum_opening_endpoints",
    }


def rank_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: CameraIntrinsicsData,
    config: Mapping[str, Any],
) -> Tuple[list[dict[str, Any]], Dict[str, Any]]:
    """Compute all features, robust normalization, and deterministic rank."""

    depth_m = np.asarray(depth_m, dtype=np.float32)
    target_mask = np.asarray(target_mask, dtype=bool)
    if depth_m.ndim != 2 or target_mask.shape != depth_m.shape:
        raise ValueError("depth and processed target mask must be aligned HxW arrays")
    if not np.any(target_mask):
        raise ValueError("processed target mask is empty")
    if (intrinsics.height, intrinsics.width) != target_mask.shape:
        raise ValueError("camera intrinsics and image dimensions disagree")
    if not records:
        raise ValueError("cannot rank an empty candidate set")

    geometry = config["geometry"]
    dilated = binary_dilate(target_mask, int(geometry["target_mask_dilation_px"]))
    boundary_distance = ndimage.distance_transform_edt(target_mask)
    mask_v, mask_u = np.nonzero(target_mask)
    centroid_uv = np.array([np.mean(mask_u), np.mean(mask_v)], dtype=np.float64)

    ranked = [deepcopy(dict(record)) for record in records]
    for record in ranked:
        record.update(
            compute_raw_features(
                record,
                depth_m=depth_m,
                target_mask=target_mask,
                dilated_mask=dilated,
                boundary_distance=boundary_distance,
                centroid_uv=centroid_uv,
                intrinsics=intrinsics,
                config=config,
            )
        )

    normalization = config["normalization"]
    stats: Dict[str, Any] = {}
    for field, (larger_is_better, bounded) in RAW_FEATURE_SPECS.items():
        values = np.asarray([record[field] for record in ranked], dtype=np.float64)
        normalized, feature_stats = _robust_normalize(
            values,
            bounded=bounded,
            lower_quantile=float(normalization["lower_quantile"]),
            upper_quantile=float(normalization["upper_quantile"]),
            epsilon=float(normalization["epsilon"]),
        )
        normalized_name = field.removesuffix("_raw") + "_normalized"
        for index, value in enumerate(normalized):
            ranked[index][normalized_name] = float(value)
        feature_stats["larger_is_better"] = larger_is_better
        feature_stats["bounded_0_1"] = bounded
        stats[field] = feature_stats

    depth_config = config["depth_component"]
    alpha_valid = float(depth_config["valid_support_fraction"])
    alpha_variance = float(depth_config["low_variance_fraction"])
    if not np.isclose(alpha_valid + alpha_variance, 1.0):
        raise ValueError("depth component fractions must sum to one")
    weights = config["weights"]
    for record in ranked:
        components = {
            "contact_support": record["contact_support_normalized"],
            "axis_mask_support": record["axis_mask_support_normalized"],
            "boundary_margin": record["boundary_distance_px_normalized"],
            "local_antipodal_score": record["local_antipodal_score_normalized"],
            "depth_support": alpha_valid * record["valid_depth_support_normalized"]
            + alpha_variance * (1.0 - record["closing_depth_variance_m2_normalized"]),
            "width_score": record["width_score_normalized"],
            "target_centrality": 1.0 - record["centroid_distance_px_normalized"],
            "interference_penalty": record["interference_penalty_normalized"],
            "image_edge_penalty": record["image_edge_penalty_normalized"],
        }
        record["component_scores"] = {name: float(components[name]) for name in COMPONENT_NAMES}
        contributions = {
            "contact": float(weights["contact"]) * components["contact_support"],
            "axis": float(weights["axis"]) * components["axis_mask_support"],
            "boundary": float(weights["boundary"]) * components["boundary_margin"],
            "antipodal": float(weights["antipodal"]) * components["local_antipodal_score"],
            "depth": float(weights["depth"]) * components["depth_support"],
            "width": float(weights["width"]) * components["width_score"],
            "target": float(weights["target"]) * components["target_centrality"],
            "collision": -float(weights["collision"]) * components["interference_penalty"],
            "edge": -float(weights["edge"]) * components["image_edge_penalty"],
        }
        score = float(sum(contributions.values()))
        record["score_contributions"] = contributions
        record["geometric_score"] = score
        record["geometric_score_for_sort"] = round(
            score, int(config["ranking"]["score_round_decimals_for_sort"])
        )
        record["geometric_score_is_calibrated_success_probability"] = False
        record["geometric_method"] = str(config["method"])

    ranked.sort(key=lambda item: (-item["geometric_score_for_sort"], item["candidate_id"]))
    for rank, record in enumerate(ranked, start=1):
        record["geometric_rank"] = rank
    breakdown = {
        "method": config["method"],
        "score_semantics": "transparent heuristic; not a GQ-CNN score or calibrated success probability",
        "candidate_count": len(ranked),
        "target_centroid_uv": centroid_uv.tolist(),
        "normalization": {
            "scope": "per_sample",
            "lower_quantile": normalization["lower_quantile"],
            "upper_quantile": normalization["upper_quantile"],
            "statistics": stats,
        },
        "formula": (
            "w_contact*contact_support + w_axis*axis_mask_support + "
            "w_boundary*boundary_margin + w_antipodal*local_antipodal_score + "
            "w_depth*depth_support + w_width*width_score + "
            "w_target*target_centrality - w_collision*interference_penalty - "
            "w_edge*image_edge_penalty"
        ),
        "weights": dict(weights),
        "thresholds": deepcopy(dict(config)),
        "tie_break": ["geometric_score rounded for sort descending", "candidate_id ascending"],
        "candidate_order": [record["candidate_id"] for record in ranked],
    }
    return ranked, breakdown


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_strict_json(path: Path | str, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _strict_json_value(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
        indent=2,
    ) + "\n"
    destination.write_text(text, encoding="utf-8", newline="\n")
    return destination


def _csv_cell(value: Any) -> Any:
    value = _strict_json_value(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return "" if value is None else value


def save_ranked_csv(path: Path | str, records: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "geometric_rank",
        "candidate_id",
        "sample_id",
        "query",
        "geometric_score",
        "center_u_px",
        "center_v_px",
        "center_depth_m",
        "angle_rad",
        "angle_deg",
        "width_m",
        "width_px",
        "gqcnn_q_value",
    ]
    fields = preferred + sorted({key for record in records for key in record} - set(preferred))
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: _csv_cell(record.get(field)) for field in fields})
    return destination


def _ranking_arrays(records: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    count = len(records)
    arrays: Dict[str, np.ndarray] = {
        "candidate_id": np.asarray([record["candidate_id"] for record in records], dtype="<U64"),
        "geometric_rank": np.asarray([record["geometric_rank"] for record in records], dtype=np.int32),
        "geometric_score": np.asarray([record["geometric_score"] for record in records], dtype=np.float64),
        "center_uv": np.asarray([record["center_uv"] for record in records], dtype=np.float32).reshape(count, 2),
        "center_depth_m": np.asarray([record["center_depth_m"] for record in records], dtype=np.float32),
        "center_camera_xyz_m": np.asarray([record["center_camera_xyz_m"] for record in records], dtype=np.float32).reshape(count, 3),
        "angle_rad": np.asarray([record["angle_rad"] for record in records], dtype=np.float32),
        "width_m": np.asarray([record["width_m"] for record in records], dtype=np.float32),
        "width_px": np.asarray([record["width_px"] for record in records], dtype=np.float32),
        "endpoints_uv": np.asarray([record["endpoints_uv"] for record in records], dtype=np.float32).reshape(count, 2, 2),
        "contact_points_uv": np.asarray([record["contact_points_uv"] for record in records], dtype=np.float32).reshape(count, 2, 2),
        "contact_normals": np.asarray([record["contact_normals"] for record in records], dtype=np.float32).reshape(count, 2, 2),
        "gqcnn_q_value": np.full(count, np.nan, dtype=np.float32),
        T_CAMERA_GRASP_FIXED_APPROACH_KEY: np.asarray(
            [record[T_CAMERA_GRASP_FIXED_APPROACH_KEY] for record in records], dtype=np.float64
        ).reshape(count, 4, 4),
    }
    scalar_names = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if (key.endswith("_raw") or key.endswith("_normalized"))
            and isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    for name in scalar_names:
        arrays[name] = np.asarray([record[name] for record in records], dtype=np.float64)
    for component in COMPONENT_NAMES:
        arrays[f"component_{component}"] = np.asarray(
            [record["component_scores"][component] for record in records], dtype=np.float64
        )
    return arrays


def save_deterministic_npz(path: Path | str, arrays: Mapping[str, np.ndarray]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return destination


def save_ranked_npz(path: Path | str, records: Sequence[Mapping[str, Any]]) -> Path:
    return save_deterministic_npz(path, _ranking_arrays(records))


def load_processed_mask(path: Path | str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def _angle_difference_rad(first: float, second: float) -> float:
    return float(abs((first - second + math.pi / 2.0) % math.pi - math.pi / 2.0))


def _rectangle(center: np.ndarray, width: float, height: float, angle_rad: float) -> np.ndarray:
    axis = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    return np.stack(
        [
            center + sx * width * 0.5 * axis + sy * height * 0.5 * normal
            for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))
        ]
    ).astype(np.float32)


def _polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    import cv2

    area_first = float(abs(cv2.contourArea(first)))
    area_second = float(abs(cv2.contourArea(second)))
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area_first + area_second - float(intersection)
    return 0.0 if union <= 0.0 else float(intersection / union)


def make_ocid_vlg_evaluation_rectangles(
    grasp_rectangles: Sequence[Sequence[Sequence[float]]],
    evaluation_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the exact normalized ground-truth rectangles used by the metric."""

    gt = np.asarray(grasp_rectangles, dtype=np.float64)
    if gt.ndim != 3 or gt.shape[1:] != (4, 2) or not np.all(np.isfinite(gt)):
        raise ValueError("OCID-VLG grasps must have shape (N,4,2)")
    gt_height = float(evaluation_config["ground_truth_rectangle_height_px"])
    gt_width_clip = float(evaluation_config["ground_truth_width_clip_px"])
    rectangles = []
    for corners in gt:
        center = (corners[0] + corners[2]) * 0.5
        opening = corners[3] - corners[0]
        width = min(float(np.linalg.norm(opening)), gt_width_clip)
        angle = float(math.atan2(opening[1], opening[0]))
        rectangles.append(
            {
                "center_uv": center,
                "width_px": width,
                "height_px": gt_height,
                "angle_rad": angle,
                "polygon": _rectangle(center, width, gt_height, angle),
            }
        )
    return rectangles


def make_candidate_evaluation_rectangle(
    record: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact contact-span rectangle used for one predicted grasp."""

    center = np.asarray(record["center_uv"], dtype=np.float64)
    contacts = np.asarray(record["contact_points_uv"], dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("candidate center_uv must be a finite length-2 vector")
    if contacts.shape != (2, 2) or not np.all(np.isfinite(contacts)):
        raise ValueError("candidate contact_points_uv must have shape (2,2)")
    opening = contacts[1] - contacts[0]
    width = float(np.linalg.norm(opening))
    if width <= 0.0:
        raise ValueError("candidate contact span must be positive")
    angle = float(math.atan2(opening[1], opening[0]))
    height = float(evaluation_config["predicted_rectangle_height_px"])
    return {
        "center_uv": center,
        "width_px": width,
        "height_px": height,
        "angle_rad": angle,
        "polygon": _rectangle(center, width, height, angle),
    }


def evaluate_planar_annotation_consistency(
    ranked: Sequence[Mapping[str, Any]],
    grasp_rectangles: Sequence[Sequence[Sequence[float]]],
    evaluation_config: Mapping[str, Any],
    *,
    rank_field: str = "geometric_rank",
) -> Dict[str, Any]:
    """Evaluate 2D rectangle consistency without feeding labels to ranking."""

    angle_threshold = float(evaluation_config["angle_threshold_deg"])
    iou_threshold = float(evaluation_config["iou_threshold"])
    gt_geometry = make_ocid_vlg_evaluation_rectangles(grasp_rectangles, evaluation_config)

    per_candidate = []
    for expected_rank, record in enumerate(ranked, start=1):
        try:
            candidate_rank = int(record[rank_field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"candidate has no valid {rank_field}") from error
        if candidate_rank != expected_rank:
            raise ValueError(
                f"{rank_field} must be one-based and contiguous in list order: "
                f"expected {expected_rank}, got {candidate_rank}"
            )
        predicted = make_candidate_evaluation_rectangle(record, evaluation_config)
        centers = [
            float(np.linalg.norm(predicted["center_uv"] - item["center_uv"]))
            for item in gt_geometry
        ]
        angles = [
            math.degrees(_angle_difference_rad(predicted["angle_rad"], item["angle_rad"]))
            for item in gt_geometry
        ]
        ious = [
            _polygon_iou(predicted["polygon"], item["polygon"])
            for item in gt_geometry
        ]
        eligible_ious = [iou if diff <= angle_threshold else 0.0 for iou, diff in zip(ious, angles)]
        max_iou = max(eligible_ious, default=0.0)
        per_candidate.append(
            {
                rank_field: candidate_rank,
                "candidate_id": record["candidate_id"],
                "minimum_center_distance_px": min(centers, default=None),
                "minimum_angle_difference_deg_modulo_pi": min(angles, default=None),
                "maximum_rectangle_iou_with_angle_gate": max_iou,
                "rectangle_match": bool(max_iou >= iou_threshold),
            }
        )
    first_match = next((item[rank_field] for item in per_candidate if item["rectangle_match"]), None)
    top_k = int(evaluation_config.get("top_k", 5))
    return {
        "label": str(evaluation_config["label"]),
        "is_physical_grasp_success": False,
        "annotation_count": len(gt_geometry),
        "rank_field": rank_field,
        "angle_threshold_deg": angle_threshold,
        "iou_threshold": iou_threshold,
        "top1_rectangle_accuracy": bool(per_candidate and per_candidate[0]["rectangle_match"]),
        "topk": top_k,
        "topk_recall": bool(any(item["rectangle_match"] for item in per_candidate[:top_k])),
        "first_matching_rank": first_match,
        "per_candidate": per_candidate,
    }


def load_ocid_vlg_annotation_record(
    annotation_path: Path | str,
    question_index: int,
) -> dict[str, Any]:
    """Load one expression by its field value, never by list position alone."""

    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise ValueError("OCID-VLG annotation file has no data list")
    if 0 <= question_index < len(data) and int(data[question_index].get("question_index", -1)) == question_index:
        return dict(data[question_index])
    matches = [item for item in data if int(item.get("question_index", -1)) == question_index]
    if len(matches) != 1:
        raise ValueError(f"question_index {question_index} has {len(matches)} annotation matches")
    return dict(matches[0])


def load_ocid_vlg_grasps(annotation_path: Path | str, question_index: int) -> list[Any]:
    return load_ocid_vlg_annotation_record(annotation_path, question_index)["grasps"]


def make_final_grasp(record: Mapping[str, Any], *, camera_frame: str) -> Dict[str, Any]:
    return {
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "candidate_id": record["candidate_id"],
        "sample_id": record.get("sample_id"),
        "query": record.get("query"),
        "center_pixel_uv": record["center_uv"],
        "center_depth_m": record["center_depth_m"],
        "center_camera_xyz_m": record["center_camera_xyz_m"],
        "angle_rad": record["angle_rad"],
        "angle_deg": record["angle_deg"],
        "width_m": record["width_m"],
        "width_px": record["width_px"],
        "contact_span_m": record["contact_span_m_raw"],
        "contact_span_px": record["contact_span_px_raw"],
        "geometric_score": record["geometric_score"],
        "geometric_rank": record["geometric_rank"],
        "component_scores": record["component_scores"],
        "score_contributions": record["score_contributions"],
        "gqcnn_q_value": None,
        "camera_frame": camera_frame,
        "calibration_warning": (
            "Camera intrinsics were derived from the organized OCID point cloud, not a "
            "factory calibration. The geometric score is a transparent heuristic and is "
            "not a calibrated physical grasp-success probability."
        ),
        T_CAMERA_GRASP_FIXED_APPROACH_KEY: record[T_CAMERA_GRASP_FIXED_APPROACH_KEY],
    }
