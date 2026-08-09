from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .aligned_crops import CROP_CHANNELS
from .coordinate_mapping import axial_difference_deg


CROG_WIDTH_FACTOR_PX = 100.0
CORRECTED_WIDTH_FEATURE_NAMES = (
    "g4_candidate_predicted_width_difference_fraction",
    "g4_normalized_width_error",
    "g4_local_mask_thickness",
    "g4_candidate_width_over_mask_thickness",
    "g4_obviously_too_wide",
    "g4_obviously_too_narrow",
    "g5_five_heads_joint_support",
)


def _regions(size: int) -> dict[str, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, int(size))
    vv, uu = np.meshgrid(axis, axis, indexing="ij")
    return {
        "roi": np.ones((size, size), dtype=bool),
        "center": (uu**2 + vv**2) <= 0.12**2,
        "centerline": np.abs(vv) <= 0.12,
        "axis": np.abs(vv) <= 0.22,
        "contact": (np.abs(uu) >= 0.35) & (np.abs(uu) <= 0.75) & (np.abs(vv) <= 0.24),
        "left_contact": (uu <= -0.35) & (uu >= -0.75) & (np.abs(vv) <= 0.24),
        "right_contact": (uu >= 0.35) & (uu <= 0.75) & (np.abs(vv) <= 0.24),
        "finger": (np.abs(uu) >= 0.48) & (np.abs(vv) <= 0.75),
        "outside": (np.abs(uu) >= 0.82) | (np.abs(vv) >= 0.82),
    }


def _safe_stats(prefix: str, values: np.ndarray) -> tuple[list[str], list[float]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    names = [f"{prefix}_{name}" for name in ("mean", "max", "min", "std", "median", "q10", "q25", "q75", "q90")]
    if not len(values):
        return names, [0.0] * len(names)
    return names, [
        float(values.mean()), float(values.max()), float(values.min()), float(values.std()),
        float(np.median(values)), float(np.quantile(values, 0.1)), float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.75)), float(np.quantile(values, 0.9)),
    ]


def _add(names: list[str], values: list[float], name: str, value: float | bool) -> None:
    names.append(name); values.append(float(value) if np.isfinite(value) else 0.0)


def _channel(crop: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(crop[CROP_CHANNELS.index(name)], dtype=np.float64)


def g0_features(candidate: dict[str, Any], candidates: list[dict[str, Any]], *, image_shape: tuple[int, int]) -> tuple[list[str], list[float]]:
    height, width = image_shape
    q = np.asarray([float(item["q_raw"]) for item in candidates], dtype=np.float64)
    probs = np.clip(q, 1e-8, None); probs /= probs.sum()
    entropy = -float(np.sum(probs * np.log(probs))) / math.log(len(probs))
    index = int(candidate["q_rank"]); top = q[0]
    theta = math.radians(float(candidate["angle_deg"]))
    area = float(candidate["width_px"]) * float(candidate["height_px"])
    edge = min(float(candidate["cx"]), width - 1 - float(candidate["cx"]), float(candidate["cy"]), height - 1 - float(candidate["cy"]))
    names = [
        "g0_q_raw", "g0_q_probability", "g0_original_q_rank", "g0_q1_q2_margin",
        "g0_qi_minus_q1", "g0_qi_over_q1", "g0_set_q_mean", "g0_set_q_std",
        "g0_set_q_entropy", "g0_peak_prominence", "g0_x_fraction", "g0_y_fraction",
        "g0_width_fraction", "g0_height_fraction", "g0_sin_theta", "g0_cos_theta",
        "g0_sin_2theta", "g0_cos_2theta", "g0_rectangle_area_fraction",
        "g0_aspect_ratio", "g0_edge_distance_fraction",
    ]
    values = [
        q[index], probs[index], index / 4.0, q[0] - q[1], q[index] - top,
        q[index] / max(abs(top), 1e-8), q.mean(), q.std(), entropy,
        float(candidate.get("features", {}).get("q_prominence", {}).get("value") or 0.0),
        float(candidate["cx"]) / width, float(candidate["cy"]) / height,
        float(candidate["width_px"]) / width, float(candidate["height_px"]) / height,
        math.sin(theta), math.cos(theta), math.sin(2 * theta), math.cos(2 * theta),
        area / (width * height), float(candidate["width_px"]) / max(float(candidate["height_px"]), 1e-8),
        edge / max(height, width),
    ]
    return names, list(map(float, values))


def extract_head_features(
    crop: np.ndarray,
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    image_shape: tuple[int, int],
) -> tuple[list[str], np.ndarray]:
    if crop.shape[0] != len(CROP_CHANNELS):
        raise ValueError("full-chain crop schema mismatch")
    regions = _regions(crop.shape[-1])
    names, values = g0_features(candidate, candidates, image_shape=image_shape)
    mask_raw = _channel(crop, "mask_raw"); mask = _channel(crop, "mask_probability")
    q_raw = _channel(crop, "quality_raw"); quality = _channel(crop, "quality_probability")
    sin2 = _channel(crop, "sin_2theta"); cos2 = _channel(crop, "cos_2theta")
    width_raw = _channel(crop, "width_raw"); width = _channel(crop, "width_probability")
    center = crop.shape[-1] // 2

    # G1: native Q local shape / peak evidence.
    for map_name, field in (("g1_q_raw", q_raw), ("g1_q_probability", quality)):
        stat_names, stat_values = _safe_stats(map_name, field[regions["roi"]]); names += stat_names; values += stat_values
        _add(names, values, f"{map_name}_center", field[center, center])
    for region_name in ("centerline", "contact", "finger", "outside"):
        _add(names, values, f"g1_q_{region_name}_mean", quality[regions[region_name]].mean())
    grad_y, grad_x = np.gradient(quality)
    lap = cv2.Laplacian(quality.astype(np.float32), cv2.CV_32F)
    dxx = np.gradient(grad_x, axis=1); dyy = np.gradient(grad_y, axis=0); dxy = np.gradient(grad_x, axis=0)
    hessian = np.asarray([[dxx[center, center], dxy[center, center]], [dxy[center, center], dyy[center, center]]])
    eig = np.linalg.eigvalsh(hessian)
    _add(names, values, "g1_q_gradient_center", math.hypot(grad_x[center, center], grad_y[center, center]))
    _add(names, values, "g1_q_gradient_mean", np.hypot(grad_x, grad_y).mean())
    _add(names, values, "g1_q_laplacian_center", lap[center, center])
    _add(names, values, "g1_q_hessian_eigen_low", eig[0]); _add(names, values, "g1_q_hessian_eigen_high", eig[1])
    _add(names, values, "g1_q_peak_curvature", -lap[center, center])
    _add(names, values, "g1_q_local_anisotropy", abs(eig[1] - eig[0]) / (abs(eig).sum() + 1e-8))
    neighbourhood = quality[regions["axis"]]
    _add(names, values, "g1_q_peak_neighbour_margin", quality[center, center] - np.median(neighbourhood))
    _add(names, values, "g1_center_is_local_maximum", quality[center, center] >= quality.max() - 1e-7)

    # G2: predicted-mask support, including both contacts and local topology.
    binary = mask >= 0.5
    for region_name in ("center", "roi", "centerline", "axis", "left_contact", "right_contact", "contact", "finger"):
        region = regions[region_name]
        _add(names, values, f"g2_mask_{region_name}_soft", mask[region].mean())
        _add(names, values, f"g2_mask_{region_name}_binary", binary[region].mean())
    left_support = mask[regions["left_contact"]].mean(); right_support = mask[regions["right_contact"]].mean()
    _add(names, values, "g2_contact_support_imbalance", abs(left_support - right_support))
    _add(names, values, "g2_finger_outside_penalty", 1.0 - mask[regions["finger"]].mean())
    mask_u8 = binary.astype(np.uint8)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 3)
    signed = inside - outside
    _add(names, values, "g2_signed_boundary_distance_center", signed[center, center] / crop.shape[-1])
    _add(names, values, "g2_left_endpoint_boundary_distance", signed[center, max(0, int(crop.shape[-1] * .25))] / crop.shape[-1])
    _add(names, values, "g2_right_endpoint_boundary_distance", signed[center, min(crop.shape[-1]-1, int(crop.shape[-1] * .75))] / crop.shape[-1])
    components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    left_label = labels[center, max(0, int(crop.shape[-1] * .25))]; right_label = labels[center, min(crop.shape[-1]-1, int(crop.shape[-1] * .75))]
    _add(names, values, "g2_endpoints_same_component", left_label > 0 and left_label == right_label)
    entropy_map = -(mask * np.log(np.clip(mask, 1e-8, 1)) + (1-mask)*np.log(np.clip(1-mask, 1e-8, 1)))
    _add(names, values, "g2_local_mask_entropy", entropy_map.mean())
    boundary = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    _add(names, values, "g2_boundary_density", boundary.mean())
    center_label = labels[center, center]
    component_fraction = stats[center_label, cv2.CC_STAT_AREA] / mask_u8.size if center_label else 0.0
    _add(names, values, "g2_component_size_fraction", component_fraction)
    if center_label:
        cx, cy = centroids[center_label]
        _add(names, values, "g2_component_centroid_distance", math.hypot(cx-center, cy-center)/crop.shape[-1])
    else:
        _add(names, values, "g2_component_centroid_distance", 1.0)
    holes = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
    hole_count = 0 if holes is None else int(np.sum(holes[0, :, 3] >= 0))
    _add(names, values, "g2_crosses_mask_hole", hole_count > 0 and binary[regions["axis"]].mean() < 0.8)
    soft_intersection = np.minimum(mask, regions["axis"].astype(float)).sum()
    soft_union = np.maximum(mask, regions["axis"].astype(float)).sum()
    _add(names, values, "g2_soft_overlap_proxy", soft_intersection / max(soft_union, 1e-8))

    # G3: axial circular confidence and consistency.
    rho = np.hypot(sin2, cos2)
    theta_field = 0.5 * np.arctan2(sin2, cos2)
    for region_name in ("center", "roi", "contact", "left_contact", "right_contact"):
        region = regions[region_name]
        _add(names, values, f"g3_rho_{region_name}_mean", rho[region].mean())
        _add(names, values, f"g3_rho_{region_name}_std", rho[region].std())
    weights = np.clip(rho[regions["roi"]], 1e-8, None)
    axial_sin = np.sum(weights * np.sin(2*theta_field[regions["roi"]])); axial_cos = np.sum(weights * np.cos(2*theta_field[regions["roi"]]))
    mean_theta = 0.5 * math.atan2(axial_sin, axial_cos)
    concentration = math.hypot(axial_sin, axial_cos) / weights.sum()
    candidate_theta = math.radians(float(candidate["angle_deg"]))
    _add(names, values, "g3_axial_circular_mean_sin", math.sin(2*mean_theta)); _add(names, values, "g3_axial_circular_mean_cos", math.cos(2*mean_theta))
    _add(names, values, "g3_axial_circular_variance", 1.0-concentration)
    _add(names, values, "g3_candidate_mean_angle_difference", axial_difference_deg(math.degrees(candidate_theta), math.degrees(mean_theta))/90.0)
    left_s = sin2[regions["left_contact"]].mean(); left_c = cos2[regions["left_contact"]].mean(); right_s = sin2[regions["right_contact"]].mean(); right_c = cos2[regions["right_contact"]].mean()
    _add(names, values, "g3_contact_angle_consistency", 1.0-axial_difference_deg(0.5*math.degrees(math.atan2(left_s,left_c)),0.5*math.degrees(math.atan2(right_s,right_c)))/90.0)
    angle_grad = np.hypot(*np.gradient(theta_field))
    _add(names, values, "g3_angle_field_gradient", angle_grad.mean())
    _add(names, values, "g3_angle_discontinuity", np.quantile(angle_grad, .9))
    _add(names, values, "g3_angle_confidence_times_q", rho[center,center]*quality[center,center])
    _add(names, values, "g3_mask_weighted_angle_confidence", np.sum(mask*rho)/max(mask.sum(),1e-8))
    _add(names, values, "g3_q_weighted_angle_confidence", np.sum(quality*rho)/max(quality.sum(),1e-8))

    # G4: width head and local thickness.
    for map_name, field in (("g4_width_raw", width_raw), ("g4_width_probability", width)):
        stat_names, stat_values = _safe_stats(map_name, field[regions["roi"]]); names += stat_names; values += stat_values
        _add(names, values, f"{map_name}_center", field[center,center])
    # Candidate generation in utils/grasp_eval.py uses sigmoid(W) * 100,
    # independent of the original image width.
    predicted_width_px = width[center,center] * CROG_WIDTH_FACTOR_PX
    _add(names, values, "g4_candidate_predicted_width_difference_fraction", (float(candidate["width_px"])-predicted_width_px)/CROG_WIDTH_FACTOR_PX)
    _add(names, values, "g4_normalized_width_error", abs(float(candidate["width_px"])-predicted_width_px)/max(float(candidate["width_px"]),1e-8))
    _add(names, values, "g4_width_field_gradient", np.hypot(*np.gradient(width)).mean())
    _add(names, values, "g4_width_uncertainty", width[regions["roi"]].std())
    # The crop's horizontal axis is the gripper opening axis and spans 1.5x
    # candidate width. Convert mask occupancy back to pixels before comparing.
    opening_occupancy = float(binary[center, :].mean())
    thickness_px = opening_occupancy * float(candidate["width_px"]) * 1.5
    _add(names, values, "g4_local_mask_thickness", thickness_px/CROG_WIDTH_FACTOR_PX)
    _add(names, values, "g4_candidate_width_over_mask_thickness", float(candidate["width_px"])/max(thickness_px,1e-3))
    _add(names, values, "g4_contact_distance_consistency", min(left_support,right_support))
    _add(names, values, "g4_obviously_too_wide", float(candidate["width_px"]) > thickness_px*1.5)
    _add(names, values, "g4_obviously_too_narrow", float(candidate["width_px"]) < thickness_px*.5)

    # G5: explicit cross-head evidence.
    mq = mask * quality; mqr = mq * rho
    for region_name in ("center", "axis", "contact"):
        region = regions[region_name]
        _add(names, values, f"g5_mq_{region_name}", mq[region].mean())
        _add(names, values, f"g5_mq_rho_{region_name}", mqr[region].mean())
    _add(names, values, "g5_mask_weighted_q", np.sum(mask*quality)/max(mask.sum(),1e-8))
    _add(names, values, "g5_q_weighted_width_consistency", np.sum(quality*(1-np.abs(width-width[center,center])))/max(quality.sum(),1e-8))
    for pair_name, left_field, right_field in (("q_mask",quality,mask),("q_angle",quality,rho),("mask_width",mask,width)):
        corr = (
            np.corrcoef(left_field.ravel(), right_field.ravel())[0,1]
            if left_field.std() > 1e-12 and right_field.std() > 1e-12
            else 0.0
        )
        _add(names, values, f"g5_{pair_name}_correlation", 0.0 if not np.isfinite(corr) else corr)
    joint_contacts = min(mqr[regions["left_contact"]].mean(), mqr[regions["right_contact"]].mean())
    _add(names, values, "g5_left_right_joint_contact_support", joint_contacts)
    supports = np.asarray([mqr[regions[name]].mean() for name in ("center","axis","contact")])
    _add(names, values, "g5_center_axis_contact_consistency", 1.0-supports.std())
    _add(names, values, "g5_five_heads_joint_support", np.mean([mask[center,center],quality[center,center],rho[center,center],1-abs(width[center,center]-float(candidate["width_px"])/CROG_WIDTH_FACTOR_PX)]))
    _add(names, values, "g5_five_head_disagreement", np.std([mask[center,center],quality[center,center],np.clip(rho[center,center],0,1),width[center,center]]))

    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite output-map feature")
    return names, result
