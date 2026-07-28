import math
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .geometry import (
    candidate_contact_bands,
    candidate_jaw_bands,
    geometry_checksum,
    neutral_expectation,
    rasterize_candidate,
)


DEFAULT_FEATURE_CONFIG = {
    "mask_threshold": 0.35,
    "candidate_generation": {"k": 5, "peak_threshold": 0.4, "min_distance": 2},
    "q_patch_radius_px": 2,
    "q_ring_inner_radius_px": 3,
    "q_ring_outer_radius_px": 7,
    "width_scan_offsets_px": [-2, -1, 0, 1, 2],
    "width_max_hole_px": 2,
    "boundary_tolerance_px": 2,
    "depth_tolerance_mm": 10.0,
    "jaw_band_thickness_ratio": 0.15,
    "minimum_angle_pixels": 3,
    "minimum_depth_pixels": 3,
}

DEFAULT_GRIPPER_CONFIG = {
    "metric_3d_enabled": False,
    "coordinate_frame": None,
    "approach_direction": None,
    "finger_length_m": None,
    "finger_thickness_m": None,
    "finger_height_m": None,
    "palm_dimensions_m": None,
    "maximum_opening_m": None,
    "approach_distance_m": None,
    "safe_clearance_m": None,
    "voxel_size_m": None,
}


def _feature(value, reliability=1.0, missing_reason=None, *, clip=False):
    reliability = float(np.clip(reliability, 0.0, 1.0))
    if value is None or not np.isfinite(value):
        return {"value": None, "reliability": 0.0, "missing_reason": missing_reason or "non_finite"}
    value = float(value)
    if clip:
        value = float(np.clip(value, 0.0, 1.0))
    if reliability <= 0.0 and missing_reason is None:
        missing_reason = "unavailable"
    return {"value": value, "reliability": reliability, "missing_reason": missing_reason}


def _finite_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return float(np.median(values))


def _patch(array, row, col, radius):
    height, width = array.shape
    r0, r1 = max(0, row - radius), min(height, row + radius + 1)
    c0, c1 = max(0, col - radius), min(width, col + radius + 1)
    return np.asarray(array[r0:r1, c0:c1])


def _q_features(candidate, quality, config):
    row, col = candidate["row"], candidate["col"]
    radius = int(config["q_patch_radius_px"])
    q_patch = _patch(quality, row, col, radius)
    q_patch_mean = float(np.mean(q_patch)) if q_patch.size else None

    outer = int(config["q_ring_outer_radius_px"])
    inner = int(config["q_ring_inner_radius_px"])
    r0, r1 = max(0, row - outer), min(quality.shape[0], row + outer + 1)
    c0, c1 = max(0, col - outer), min(quality.shape[1], col + outer + 1)
    rows, cols = np.indices((r1 - r0, c1 - c0))
    distance = np.hypot(rows + r0 - row, cols + c0 - col)
    ring_values = quality[r0:r1, c0:c1][(distance >= inner) & (distance <= outer)]
    prominence = None if not ring_values.size else float(candidate["q_raw"] - np.median(ring_values))
    return {
        "q": _feature(np.clip(candidate["q_raw"], 0.0, 1.0), clip=True),
        "q_patch_mean": _feature(
            q_patch_mean, missing_reason=None if q_patch_mean is not None else "empty_q_patch", clip=True
        ),
        "q_prominence": _feature(
            prominence, missing_reason=None if prominence is not None else "empty_q_ring"
        ),
    }


def _center_probability(mask_probability, row, col):
    height, width = mask_probability.shape
    if not (0 <= row < height and 0 <= col < width):
        return None, 0.0, "centre_outside_image"
    weights = np.asarray([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
    total = weighted = 0.0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr, cc = row + dr, col + dc
            if 0 <= rr < height and 0 <= cc < width:
                weight = weights[dr + 1, dc + 1]
                weighted += weight * float(mask_probability[rr, cc])
                total += weight
    if total <= 0:
        return None, 0.0, "empty_centre_patch"
    return weighted / total, total / float(weights.sum()), None


def _mask_features(
    candidate,
    mask_probability,
    binary_mask,
    visible,
    full_count,
    signed_mask_distance=None,
):
    visible_count = int(visible.sum())
    image_support = visible_count / float(full_count)
    row, col = candidate["row"], candidate["col"]
    center_prob, center_reliability, center_reason = _center_probability(mask_probability, row, col)

    if visible_count:
        soft_coverage = float(np.mean(mask_probability[visible]))
        binary_coverage = float(np.mean(binary_mask[visible]))
        coverage_reliability = image_support
        coverage_reason = None if image_support >= 1.0 else "rectangle_partially_outside_image"
    else:
        soft_coverage = binary_coverage = None
        coverage_reliability = 0.0
        coverage_reason = "rectangle_outside_image"

    if binary_mask.any() and 0 <= row < binary_mask.shape[0] and 0 <= col < binary_mask.shape[1]:
        if signed_mask_distance is None:
            signed_mask_distance = ndimage.distance_transform_edt(
                binary_mask
            ) - ndimage.distance_transform_edt(~binary_mask)
        r0 = max(2.0, 0.25 * float(candidate["height_px"]))
        center_margin = float(
            np.clip(0.5 + signed_mask_distance[row, col] / (2.0 * r0), 0.0, 1.0)
        )
        margin_reliability, margin_reason = 1.0, None
    else:
        center_margin, margin_reliability, margin_reason = None, 0.0, "empty_predicted_mask"

    center_feature = _feature(center_prob, center_reliability, center_reason, clip=True)
    soft_feature = _feature(soft_coverage, coverage_reliability, coverage_reason, clip=True)
    binary_feature = _feature(binary_coverage, coverage_reliability, coverage_reason, clip=True)
    margin_feature = _feature(center_margin, margin_reliability, margin_reason, clip=True)
    mask_consistency = (
        0.30 * neutral_expectation(center_feature["value"], center_feature["reliability"])
        + 0.45 * neutral_expectation(soft_feature["value"], soft_feature["reliability"])
        + 0.25 * neutral_expectation(margin_feature["value"], margin_feature["reliability"])
    )
    return {
        "center_prob": center_feature,
        "soft_coverage": soft_feature,
        "binary_coverage": binary_feature,
        "center_margin": margin_feature,
        "image_support": _feature(image_support, clip=True),
        "mask_consistency": _feature(mask_consistency, clip=True),
    }


def _fill_small_holes(values, maximum_hole):
    values = np.asarray(values, dtype=bool).copy()
    index = 0
    while index < len(values):
        if values[index]:
            index += 1
            continue
        end = index
        while end < len(values) and not values[end]:
            end += 1
        if index > 0 and end < len(values) and end - index <= maximum_hole:
            values[index:end] = True
        index = end
    return values


def mask_span_width(candidate, binary_mask, config):
    if not binary_mask.any():
        return None, "empty_predicted_mask"
    height, width = binary_mask.shape
    theta = math.radians(float(candidate["angle_deg"]))
    opening = np.asarray([math.cos(theta), -math.sin(theta)], dtype=np.float64)
    perpendicular = np.asarray([-opening[1], opening[0]], dtype=np.float64)
    extent = int(math.ceil(math.hypot(height, width))) + 2
    offsets = config["width_scan_offsets_px"]
    spans = []
    sample_t = np.arange(-extent, extent + 1, dtype=np.float64)
    centre_index = extent
    for offset in offsets:
        base = np.asarray([candidate["cx"], candidate["cy"]]) + float(offset) * perpendicular
        cols = np.rint(base[0] + sample_t * opening[0]).astype(np.intp)
        rows = np.rint(base[1] + sample_t * opening[1]).astype(np.intp)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        valid_values = np.zeros(sample_t.shape, dtype=bool)
        valid_values[inside] = binary_mask[rows[inside], cols[inside]]
        valid_values = _fill_small_holes(valid_values, int(config["width_max_hole_px"]))
        if not valid_values[centre_index]:
            continue
        left_false = np.flatnonzero(~valid_values[:centre_index])
        left = int(left_false[-1] + 1) if left_false.size else 0
        right_false = np.flatnonzero(~valid_values[centre_index + 1 :])
        right = (
            int(centre_index + right_false[0])
            if right_false.size
            else len(valid_values) - 1
        )
        # A zero-valued, in-image sample must delimit both sides.
        if left == 0 or right == len(valid_values) - 1:
            continue
        if not inside[left - 1] or not inside[right + 1]:
            continue
        d_minus = float(-sample_t[left])
        d_plus = float(sample_t[right])
        object_width = d_minus + d_plus
        if object_width >= 3.0:
            spans.append((d_minus, d_plus, object_width))
    if not spans:
        return None, "no_complete_mask_span_through_centre"
    values = np.asarray(spans, dtype=np.float64)
    return {
        "d_minus_px": float(np.median(values[:, 0])),
        "d_plus_px": float(np.median(values[:, 1])),
        "object_width_px": float(np.median(values[:, 2])),
        "valid_scanline_fraction": float(len(spans) / max(1, len(offsets))),
    }, None


def _width_features(candidate, binary_mask, config, calibration):
    span, reason = mask_span_width(candidate, binary_mask, config)
    if span is None:
        missing = _feature(None, 0.0, reason)
        return {
            "width_compatibility": missing,
            "width_compatibility_calibrated": missing.copy(),
            "width_ratio": missing.copy(),
            "width_symmetry": missing.copy(),
        }, {}
    predicted = float(candidate["width_px"])
    observed = span["object_width_px"]
    epsilon = 1e-6
    reliability = span["valid_scanline_fraction"]
    compatibility = min(predicted, observed) / (max(predicted, observed) + epsilon)
    ratio = predicted / (observed + epsilon)
    symmetry = 1.0 - abs(span["d_minus_px"] - span["d_plus_px"]) / (
        span["d_minus_px"] + span["d_plus_px"] + epsilon
    )
    calibrated = None
    calibrated_reason = "training_width_calibration_unavailable"
    if calibration and calibration.get("width_mu_rho") is not None:
        rho = math.log((predicted + epsilon) / (observed + epsilon))
        sigma = max(float(calibration["width_sigma_rho"]), 0.10)
        calibrated = math.exp(-0.5 * ((rho - float(calibration["width_mu_rho"])) / sigma) ** 2)
        calibrated_reason = None
    return {
        "width_compatibility": _feature(compatibility, reliability, clip=True),
        "width_compatibility_calibrated": _feature(
            calibrated, reliability if calibrated is not None else 0.0, calibrated_reason, clip=True
        ),
        "width_ratio": _feature(ratio, reliability),
        "width_symmetry": _feature(symmetry, reliability, clip=True),
    }, span


def _angle_features(candidate, mask_probability, quality, sin_map, cos_map, visible, config):
    norm = np.hypot(cos_map, sin_map)
    valid = visible & np.isfinite(norm) & (norm > 1e-8)
    weights = np.clip(mask_probability, 0.0, 1.0) * np.clip(quality, 0.0, 1.0)
    valid &= np.isfinite(weights) & (weights > 0)
    count = int(valid.sum())
    total_weight = float(weights[valid].sum()) if count else 0.0
    if count < int(config["minimum_angle_pixels"]) or total_weight <= 1e-8:
        return {"angle_consistency": _feature(None, 0.0, "insufficient_weighted_angle_pixels")}, {}
    unit_cos = cos_map[valid] / norm[valid]
    unit_sin = sin_map[valid] / norm[valid]
    mean_cos = float(np.sum(weights[valid] * unit_cos) / total_weight)
    mean_sin = float(np.sum(weights[valid] * unit_sin) / total_weight)
    concentration = float(np.clip(math.hypot(mean_cos, mean_sin), 0.0, 1.0))
    if concentration <= 1e-8:
        alignment = 0.5
    else:
        mean_cos /= concentration
        mean_sin /= concentration
        centre = np.asarray(
            [math.cos(2.0 * candidate["angle_rad"]), math.sin(2.0 * candidate["angle_rad"])],
            dtype=np.float64,
        )
        alignment = float(np.clip((1.0 + centre[0] * mean_cos + centre[1] * mean_sin) / 2.0, 0.0, 1.0))
    consistency = 0.5 * concentration + 0.5 * alignment
    return {"angle_consistency": _feature(consistency, clip=True)}, {
        "angle_concentration": concentration,
        "center_alignment": alignment,
        "weighted_angle_pixels": count,
    }


def _valid_depth(depth):
    return np.isfinite(depth) & (depth > 0)


def _depth_features(candidate, depth_m, binary_mask, config, calibration):
    missing = _feature(None, 0.0, "depth_unavailable")
    if depth_m is None or depth_m.shape != binary_mask.shape:
        return {
            "depth_geometry": missing,
            "depth_mad_m": missing.copy(),
            "contact_depth_difference_m": missing.copy(),
        }, {"depth_available": False}
    valid = _valid_depth(depth_m)
    row, col = candidate["row"], candidate["col"]
    radius = 2
    r0, r1 = max(0, row - radius), min(depth_m.shape[0], row + radius + 1)
    c0, c1 = max(0, col - radius), min(depth_m.shape[1], col + radius + 1)
    centre_values = depth_m[r0:r1, c0:c1]
    centre_keep = valid[r0:r1, c0:c1] & binary_mask[r0:r1, c0:c1]
    centre_values = centre_values[centre_keep]
    centre_depth = _finite_median(centre_values)
    depth_mad = None
    if centre_values.size:
        median = float(np.median(centre_values))
        depth_mad = 1.4826 * float(np.median(np.abs(centre_values - median)))

    left, right = candidate_contact_bands(
        candidate, depth_m.shape, config["jaw_band_thickness_ratio"]
    )
    left_depth = _finite_median(depth_m[left & valid & binary_mask])
    right_depth = _finite_median(depth_m[right & valid & binary_mask])
    contact_difference = None
    if left_depth is not None and right_depth is not None:
        contact_difference = abs(left_depth - right_depth)

    minimum = int(config["minimum_depth_pixels"])
    raw_reliability = min(1.0, float(centre_values.size) / max(1, minimum))
    mad_feature = _feature(
        depth_mad,
        raw_reliability,
        None if depth_mad is not None else "insufficient_masked_centre_depth",
    )
    contact_reliability = 1.0 if contact_difference is not None else 0.0
    contact_feature = _feature(
        contact_difference,
        contact_reliability,
        None if contact_difference is not None else "missing_contact_band_depth",
    )

    geometry = None
    geometry_reliability = 0.0
    geometry_reason = "training_depth_calibration_unavailable"
    if calibration and calibration.get("tau_variance_m") is not None and calibration.get("tau_balance_m") is not None:
        surface = None if depth_mad is None else math.exp(-depth_mad / float(calibration["tau_variance_m"]))
        balance = None if contact_difference is None else math.exp(
            -contact_difference / float(calibration["tau_balance_m"])
        )
        geometry = 0.5 * neutral_expectation(surface, raw_reliability) + 0.5 * neutral_expectation(
            balance, contact_reliability
        )
        geometry_reliability = min(raw_reliability, contact_reliability)
        geometry_reason = None
    return {
        "depth_geometry": _feature(geometry, geometry_reliability, geometry_reason, clip=True),
        "depth_mad_m": mad_feature,
        "contact_depth_difference_m": contact_feature,
    }, {
        "depth_available": True,
        "center_depth_m": centre_depth,
        "local_depth_median_m": centre_depth,
        "left_contact_depth_m": left_depth,
        "right_contact_depth_m": right_depth,
    }


def _safety_features(candidate, depth_m, binary_mask, config, dilated_mask=None):
    missing = _feature(None, 0.0, "relative_2p5d_obstacle_proxy_unavailable")
    if depth_m is None or depth_m.shape != binary_mask.shape or not binary_mask.any():
        return {
            "safety": missing,
            "collision_proxy": missing.copy(),
            "clearance": missing.copy(),
        }, {"metric_3d_collision_available": False}
    valid = _valid_depth(depth_m)
    visible, _, _ = rasterize_candidate(candidate, depth_m.shape)
    target_depth = depth_m[visible & binary_mask & valid]
    if not target_depth.size:
        return {
            "safety": missing,
            "collision_proxy": missing.copy(),
            "clearance": missing.copy(),
        }, {"metric_3d_collision_available": False}
    z_ref = float(np.median(target_depth))
    if dilated_mask is None:
        iterations = int(config["boundary_tolerance_px"])
        dilated_mask = (
            ndimage.binary_dilation(binary_mask, iterations=iterations)
            if iterations
            else binary_mask
        )
    obstacle = valid & ~dilated_mask & (
        depth_m <= z_ref + float(config["depth_tolerance_mm"]) / 1000.0
    )
    left, right = candidate_jaw_bands(
        candidate, depth_m.shape, config["jaw_band_thickness_ratio"]
    )
    jaws = left | right
    valid_jaws = jaws & valid
    denominator = int(valid_jaws.sum())
    if denominator == 0:
        return {
            "safety": missing,
            "collision_proxy": missing.copy(),
            "clearance": missing.copy(),
        }, {"metric_3d_collision_available": False}
    collision = float(np.sum(obstacle & jaws) / denominator)
    if obstacle.any() and jaws.any():
        distance = ndimage.distance_transform_edt(~obstacle)
        nearest = float(np.min(distance[jaws]))
        clearance = float(np.clip(nearest / max(float(candidate["height_px"]) / 2.0, 1.0), 0.0, 1.0))
    else:
        nearest, clearance = math.inf, 1.0
    reliability = float(np.mean(valid_jaws[jaws])) if jaws.any() else 0.0
    safety = 0.70 * neutral_expectation(1.0 - collision, reliability) + 0.30 * neutral_expectation(
        clearance, reliability
    )
    return {
        "safety": _feature(safety, reliability, clip=True),
        "collision_proxy": _feature(collision, reliability, clip=True),
        "clearance": _feature(clearance, reliability, clip=True),
    }, {
        "metric_3d_collision_available": False,
        "collision_proxy_name": "relative_2p5d_obstacle_proxy",
        "z_ref_m": z_ref,
        "nearest_obstacle_distance_px": None if not np.isfinite(nearest) else nearest,
    }


def _prepare_feature_context(
    mask_probability,
    quality,
    sin_map,
    cos_map,
    depth_m,
    feature_config,
):
    """Prepare expression-level arrays once for all frozen candidates."""
    config = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}
    mask_probability = np.clip(
        np.asarray(mask_probability, dtype=np.float64), 0.0, 1.0
    )
    quality = np.asarray(quality, dtype=np.float64)
    sin_map = np.asarray(sin_map, dtype=np.float64)
    cos_map = np.asarray(cos_map, dtype=np.float64)
    shape = quality.shape
    if any(array.shape != shape for array in (mask_probability, sin_map, cos_map)):
        raise ValueError("inference map shapes do not match")

    binary_mask = mask_probability > float(config["mask_threshold"])
    dilated_mask = None
    if depth_m is not None and depth_m.shape == binary_mask.shape and binary_mask.any():
        iterations = int(config["boundary_tolerance_px"])
        dilated_mask = (
            ndimage.binary_dilation(binary_mask, iterations=iterations)
            if iterations
            else binary_mask
        )
    return {
        "config": config,
        "mask_probability": mask_probability,
        "quality": quality,
        "sin_map": sin_map,
        "cos_map": cos_map,
        "binary_mask": binary_mask,
        # The distance transform is intentionally candidate-local: five SciPy
        # calls execute concurrently and benchmark faster than one serial
        # expression-level transform on the supported Mac runtime.
        "signed_mask_distance": None,
        "dilated_mask": dilated_mask,
        "shape": shape,
    }


def extract_candidate_features(
    candidate,
    *,
    mask_probability,
    quality,
    sin_map,
    cos_map,
    depth_m=None,
    feature_config=None,
    calibration=None,
    _prepared_context=None,
):
    context = _prepared_context or _prepare_feature_context(
        mask_probability,
        quality,
        sin_map,
        cos_map,
        depth_m,
        feature_config,
    )
    config = context["config"]
    mask_probability = context["mask_probability"]
    quality = context["quality"]
    sin_map = context["sin_map"]
    cos_map = context["cos_map"]
    binary_mask = context["binary_mask"]
    shape = context["shape"]
    if geometry_checksum(candidate) != candidate["candidate_checksum"]:
        raise AssertionError("candidate geometry checksum mismatch before feature extraction")
    visible, full_count, polygon = rasterize_candidate(candidate, shape)
    result = deepcopy(candidate)
    result["polygon"] = polygon.astype(float).tolist()
    if geometry_checksum(result) != candidate["candidate_checksum"]:
        raise AssertionError("feature extraction changed frozen candidate geometry")

    features = {}
    diagnostics = {}
    features.update(_q_features(candidate, quality, config))
    features.update(
        _mask_features(
            candidate,
            mask_probability,
            binary_mask,
            visible,
            full_count,
            context["signed_mask_distance"],
        )
    )
    width_features, width_diagnostics = _width_features(candidate, binary_mask, config, calibration)
    angle_features, angle_diagnostics = _angle_features(
        candidate, mask_probability, quality, sin_map, cos_map, visible, config
    )
    depth_features, depth_diagnostics = _depth_features(
        candidate, depth_m, binary_mask, config, calibration
    )
    safety_features, safety_diagnostics = _safety_features(
        candidate,
        depth_m,
        binary_mask,
        config,
        context["dilated_mask"],
    )
    features.update(width_features)
    features.update(angle_features)
    features.update(depth_features)
    features.update(safety_features)
    diagnostics.update(width_diagnostics)
    diagnostics.update(angle_diagnostics)
    diagnostics.update(depth_diagnostics)
    diagnostics.update(safety_diagnostics)
    diagnostics["visible_rectangle_pixels"] = int(visible.sum())
    diagnostics["full_rectangle_pixels"] = int(full_count)
    result["features"] = features
    result["diagnostics"] = diagnostics

    for name, feature in features.items():
        value = feature["value"]
        reliability = feature["reliability"]
        if value is not None and not np.isfinite(value):
            raise AssertionError(f"non-finite feature {name}")
        if not np.isfinite(reliability):
            raise AssertionError(f"non-finite feature reliability {name}")
    return result


def extract_features_for_candidates(
    candidates,
    *,
    mask_probability,
    quality,
    sin_map,
    cos_map,
    depth_m=None,
    feature_config=None,
    calibration=None,
):
    candidates = list(candidates)
    context = _prepare_feature_context(
        mask_probability,
        quality,
        sin_map,
        cos_map,
        depth_m,
        feature_config,
    )
    if len(candidates) <= 1:
        return [
            extract_candidate_features(
                candidate,
                mask_probability=mask_probability,
                quality=quality,
                sin_map=sin_map,
                cos_map=cos_map,
                depth_m=depth_m,
                feature_config=feature_config,
                calibration=calibration,
                _prepared_context=context,
            )
            for candidate in candidates
        ]

    def extract(candidate):
        return extract_candidate_features(
            candidate,
            mask_probability=mask_probability,
            quality=quality,
            sin_map=sin_map,
            cos_map=cos_map,
            depth_m=depth_m,
            feature_config=feature_config,
            calibration=calibration,
            _prepared_context=context,
        )

    # Candidate computations are independent and NumPy/SciPy release the GIL.
    # executor.map preserves legacy candidate order deterministically.
    with ThreadPoolExecutor(max_workers=min(5, len(candidates))) as executor:
        return list(executor.map(extract, candidates))


def load_depth_m(path, expected_shape=None):
    if path is None:
        return None, "depth_path_missing"
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None, "depth_file_missing_or_unreadable"
    if depth.ndim != 2:
        return None, "depth_not_single_channel"
    depth = depth.astype(np.float32) / 1000.0
    if expected_shape is not None and depth.shape != tuple(expected_shape):
        return None, "depth_image_shape_mismatch"
    return depth, None


def pcd_path_from_image(image_path):
    path = Path(image_path)
    parts = list(path.parts)
    try:
        parts[parts.index("rgb")] = "pcd"
    except ValueError:
        return None
    return Path(*parts).with_suffix(".pcd")


def load_pcd_xyz(path):
    """Read only XYZ from an organized binary OCID PCD; never expose label."""
    path = Path(path)
    if not path.exists():
        return None, "pcd_file_missing"
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                return None, "invalid_pcd_header"
            header_lines.append(line.decode("ascii", errors="strict").strip())
            if line.startswith(b"DATA "):
                break
        header = {}
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key] = values
        if header.get("DATA") != ["binary"]:
            return None, "unsupported_pcd_encoding"
        if header.get("FIELDS") != ["x", "y", "z", "rgba", "label"]:
            return None, "unexpected_pcd_fields"
        width, height = int(header["WIDTH"][0]), int(header["HEIGHT"][0])
        count = int(header["POINTS"][0])
        dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgba", "<u4"), ("label", "<u4")]
        )
        records = np.fromfile(handle, dtype=dtype, count=count)
    if records.size != width * height:
        return None, "pcd_point_count_mismatch"
    # Copy only XYZ into a new array so the GT label field is physically absent.
    xyz = np.stack([records["x"], records["y"], records["z"]], axis=-1).reshape(height, width, 3)
    return xyz, None
