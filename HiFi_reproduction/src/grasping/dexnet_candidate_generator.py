"""Official antipodal sampling plus explicit target-aware post-filtering."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .camera_geometry import (
    T_CAMERA_GRASP_FIXED_APPROACH_KEY,
    backproject_pixel,
)
from .dexnet_adapter import (
    make_camera_intrinsics,
    make_rgbd_and_segmask,
    sample_antipodal_grasps,
    sampling_config_hash,
)
from .mask_processing import binary_dilate
from .ocid_vlg_grasp_adapter import OcidVlgGraspSample


REPRESENTATION = "planar_parallel_jaw_4dof"
APPROACH_CONSTRAINT = "fixed_camera_optical_axis"


@dataclass
class CandidateGenerationResult:
    sample: OcidVlgGraspSample
    official_grasps: list[Any]
    raw_candidates: list[dict[str, Any]]
    mask_validated_candidates: list[dict[str, Any]]
    deduplicated_candidates: list[dict[str, Any]]
    topk_candidates: list[dict[str, Any]]
    rejected_candidates: list[dict[str, Any]]
    rejection_summary: dict[str, int]
    requested_candidate_count: int
    generation_time_ms: float
    scoring_time_ms: float = 0.0


def _pixel(mask: np.ndarray, uv: Sequence[float]) -> bool:
    u, v = np.rint(uv).astype(int)
    return bool(0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u])


def _local_support(mask: np.ndarray, uv: Sequence[float], radius: int) -> float:
    u, v = np.rint(uv).astype(int)
    radius = max(int(radius), 0)
    u0, u1 = max(u - radius, 0), min(u + radius + 1, mask.shape[1])
    v0, v1 = max(v - radius, 0), min(v + radius + 1, mask.shape[0])
    if u0 >= u1 or v0 >= v1:
        return 0.0
    return float(np.any(mask[v0:v1, u0:u1]))


def _line_pixels(first_uv: Sequence[float], second_uv: Sequence[float]) -> np.ndarray:
    first = np.asarray(first_uv, dtype=np.float64)
    second = np.asarray(second_uv, dtype=np.float64)
    count = max(2, int(np.ceil(np.linalg.norm(second - first))) + 1)
    return np.linspace(first, second, count)


def _line_support(mask: np.ndarray, points_uv: np.ndarray) -> float:
    uu = np.rint(points_uv[:, 0]).astype(int)
    vv = np.rint(points_uv[:, 1]).astype(int)
    in_bounds = (uu >= 0) & (uu < mask.shape[1]) & (vv >= 0) & (vv < mask.shape[0])
    supported = np.zeros(points_uv.shape[0], dtype=bool)
    supported[in_bounds] = mask[vv[in_bounds], uu[in_bounds]]
    return float(np.mean(supported)) if supported.size else 0.0


def _image_boundary_distance(endpoints_uv: np.ndarray, shape: tuple[int, int]) -> float:
    height, width = shape
    u = endpoints_uv[:, 0]
    v = endpoints_uv[:, 1]
    return float(np.min(np.r_[u, v, width - 1 - u, height - 1 - v]))


def _canonical_angle_distance(first: float, second: float) -> float:
    delta = abs((first - second) % math.pi)
    return min(delta, math.pi - delta)


def _official_pose_matrix(grasp: Any) -> np.ndarray:
    transform = grasp.pose()
    matrix = np.asarray(transform.matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Official Grasp2D.pose() returned an invalid transform")
    return matrix


def grasp_to_candidate(
    grasp: Any,
    sample: OcidVlgGraspSample,
    *,
    sampler_rank: int,
    seed: int,
    config_hash: str,
) -> dict[str, Any]:
    center = np.asarray(grasp.center.data, dtype=np.float64)
    endpoints = np.asarray(grasp.endpoints, dtype=np.float64)
    if center.shape != (2,) or endpoints.shape != (2, 2):
        raise ValueError("Official Grasp2D has malformed center/endpoints")
    center_depth = float(grasp.depth)
    xyz = backproject_pixel(center[0], center[1], center_depth, sample.intrinsics)
    contacts_uv = None
    contact_normals_uv = None
    if grasp.contact_points is not None:
        contacts_vu = np.asarray(grasp.contact_points, dtype=np.float64)
        if contacts_vu.shape == (2, 2):
            contacts_uv = contacts_vu[:, ::-1]
    if grasp.contact_normals is not None:
        normals_vu = np.asarray(grasp.contact_normals, dtype=np.float64)
        if normals_vu.shape == (2, 2):
            contact_normals_uv = normals_vu[:, ::-1]
    return {
        "candidate_id": f"g{sampler_rank - 1:04d}",
        "sample_id": sample.sample_id,
        "query": sample.query,
        "representation": REPRESENTATION,
        "approach_constraint": APPROACH_CONSTRAINT,
        "center_u_px": float(center[0]),
        "center_v_px": float(center[1]),
        "center_uv": center.tolist(),
        "center_depth_m": center_depth,
        "center_camera_xyz_m": xyz.tolist(),
        "angle_rad": float(grasp.angle),
        "angle_deg": float(np.degrees(grasp.angle)),
        "width_m": float(grasp.width),
        "width_px": float(grasp.width_px),
        "endpoint_1_uv": endpoints[0].tolist(),
        "endpoint_2_uv": endpoints[1].tolist(),
        "endpoints_uv": endpoints.tolist(),
        "contact_points_uv": None if contacts_uv is None else contacts_uv.tolist(),
        "contact_normals": (
            None if contact_normals_uv is None else contact_normals_uv.tolist()
        ),
        "sampler_rank": sampler_rank,
        "gqcnn_q_value": None,
        "gqcnn_rank": None,
        "rejection_reason": None,
        "rejection_reasons": [],
        "camera_frame": sample.intrinsics.frame,
        "seed": int(seed),
        "sampler_configuration_hash": config_hash,
        "model_name": None,
        T_CAMERA_GRASP_FIXED_APPROACH_KEY: _official_pose_matrix(grasp).tolist(),
        "pose_source": "official_gqcnn_Grasp2D.pose",
        "pose_transform_convention": "from grasp frame to camera frame",
        "fixed_approach_direction_camera": [0.0, 0.0, 1.0],
        "width_semantics": "configured_maximum_jaw_opening",
    }


def validate_candidate(
    candidate: dict[str, Any],
    sample: OcidVlgGraspSample,
    filtering: Mapping[str, Any],
) -> dict[str, Any]:
    mask = sample.target_mask_processed
    depth_valid = sample.valid_depth_mask
    center = np.asarray(candidate["center_uv"], dtype=np.float64)
    endpoints = np.asarray(candidate["endpoints_uv"], dtype=np.float64)
    contacts = candidate.get("contact_points_uv")
    support_endpoints = (
        np.asarray(contacts, dtype=np.float64)
        if contacts is not None
        else endpoints
    )
    contact_mask = binary_dilate(
        mask, int(filtering.get("contact_mask_dilation_px", 0))
    )
    distance = ndimage.distance_transform_edt(mask)
    u, v = np.rint(center).astype(int)
    center_in_bounds = 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]
    center_inside = bool(center_in_bounds and mask[v, u])
    center_valid_depth = bool(center_in_bounds and depth_valid[v, u])
    boundary_distance = float(distance[v, u]) if center_in_bounds else 0.0
    radius = int(filtering.get("endpoint_support_radius_px", 0))
    endpoint_support = [
        _local_support(contact_mask, support_endpoints[index], radius)
        for index in range(2)
    ]
    line_points = _line_pixels(support_endpoints[0], support_endpoints[1])
    axis_mask_support = _line_support(contact_mask, line_points)
    valid_depth_support = _line_support(depth_valid, line_points)
    image_boundary_distance = _image_boundary_distance(endpoints, mask.shape)

    candidate.update(
        {
            "centre_inside_mask": center_inside,
            "centre_boundary_distance_px": boundary_distance,
            "valid_depth_at_center": center_valid_depth,
            "endpoint_1_mask_support": endpoint_support[0],
            "endpoint_2_mask_support": endpoint_support[1],
            "grasp_axis_mask_support": axis_mask_support,
            "valid_depth_support": valid_depth_support,
            "image_boundary_distance_px": image_boundary_distance,
            "support_endpoint_source": (
                "official_antipodal_contacts" if contacts is not None else "jaw_endpoints"
            ),
        }
    )
    reasons: list[str] = []
    if filtering.get("require_center_inside_mask", True) and not center_inside:
        reasons.append("center_outside_target_mask")
    if not center_valid_depth:
        reasons.append("invalid_center_depth")
    if boundary_distance < float(filtering.get("min_center_boundary_distance_px", 0.0)):
        reasons.append("center_too_close_to_target_boundary")
    if endpoint_support[0] <= 0:
        reasons.append("endpoint_1_without_target_support")
    if endpoint_support[1] <= 0:
        reasons.append("endpoint_2_without_target_support")
    if axis_mask_support < float(filtering.get("min_grasp_axis_mask_support", 0.0)):
        reasons.append("insufficient_grasp_axis_mask_support")
    if valid_depth_support < float(filtering.get("min_valid_depth_support", 0.0)):
        reasons.append("insufficient_valid_depth_support")
    width = float(candidate["width_m"])
    if width < float(filtering.get("min_gripper_width_m", 0.0)):
        reasons.append("gripper_width_below_minimum")
    if width > float(filtering.get("max_gripper_width_m", math.inf)):
        reasons.append("gripper_width_above_maximum")
    if image_boundary_distance < float(filtering.get("image_boundary_margin_px", 0.0)):
        reasons.append("grasp_too_close_to_image_boundary")
    candidate["rejection_reasons"] = reasons
    candidate["rejection_reason"] = reasons[0] if reasons else None
    return candidate


def deduplicate_candidates(
    candidates: Sequence[dict[str, Any]],
    filtering: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    center_threshold = float(filtering.get("nms_center_distance_px", 0.0))
    angle_threshold = math.radians(
        float(filtering.get("nms_angle_distance_deg", 0.0))
    )
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        center = np.asarray(candidate["center_uv"], dtype=np.float64)
        duplicate = any(
            np.linalg.norm(center - np.asarray(other["center_uv"], dtype=np.float64))
            < center_threshold
            and _canonical_angle_distance(
                float(candidate["angle_rad"]), float(other["angle_rad"])
            ) < angle_threshold
            for other in kept
        )
        if duplicate:
            duplicate_candidate = dict(candidate)
            duplicate_candidate["rejection_reason"] = "duplicate_nms"
            duplicate_candidate["rejection_reasons"] = ["duplicate_nms"]
            rejected.append(duplicate_candidate)
        else:
            kept.append(candidate)
    return kept, rejected


def generate_candidates(
    sample: OcidVlgGraspSample,
    sampling_config: Mapping[str, Any],
    filtering_config: Mapping[str, Any],
    *,
    num_samples: int,
    top_k: int,
    seed: int,
    visualize_sampler: bool = False,
) -> CandidateGenerationResult:
    start = time.perf_counter()
    official_intrinsics = make_camera_intrinsics(
        {
            "fx": sample.intrinsics.fx,
            "fy": sample.intrinsics.fy,
            "cx": sample.intrinsics.cx,
            "cy": sample.intrinsics.cy,
            "skew": sample.intrinsics.skew,
            "width": sample.intrinsics.width,
            "height": sample.intrinsics.height,
        },
        frame=sample.intrinsics.frame,
    )
    rgbd_image, target_segmask = make_rgbd_and_segmask(
        sample.rgb,
        sample.depth_m,
        sample.target_mask_processed,
        frame=sample.intrinsics.frame,
    )
    official_grasps = sample_antipodal_grasps(
        rgbd_image,
        official_intrinsics,
        target_segmask,
        sampling_config,
        num_samples=num_samples,
        seed=seed,
        visualize=visualize_sampler,
    )
    config_hash = sampling_config_hash(sampling_config)
    raw = [
        validate_candidate(
            grasp_to_candidate(
                grasp,
                sample,
                sampler_rank=index + 1,
                seed=seed,
                config_hash=config_hash,
            ),
            sample,
            filtering_config,
        )
        for index, grasp in enumerate(official_grasps)
    ]
    valid = [candidate for candidate in raw if candidate["rejection_reason"] is None]
    invalid = [candidate for candidate in raw if candidate["rejection_reason"] is not None]
    deduplicated, duplicate_rejections = deduplicate_candidates(valid, filtering_config)
    rejected = invalid + duplicate_rejections
    summary = Counter(
        reason
        for candidate in rejected
        for reason in candidate.get("rejection_reasons", [candidate["rejection_reason"]])
    )
    top = list(deduplicated[: max(0, int(top_k))])
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return CandidateGenerationResult(
        sample=sample,
        official_grasps=official_grasps,
        raw_candidates=raw,
        mask_validated_candidates=valid,
        deduplicated_candidates=deduplicated,
        topk_candidates=top,
        rejected_candidates=rejected,
        rejection_summary=dict(sorted(summary.items())),
        requested_candidate_count=int(num_samples),
        generation_time_ms=elapsed_ms,
    )
