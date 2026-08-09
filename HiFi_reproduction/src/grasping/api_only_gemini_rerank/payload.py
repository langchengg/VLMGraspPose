"""Whitelisted, GT-free request text and exact semantic request identity."""

from __future__ import annotations

import math
import numbers
from typing import Any, Mapping, Sequence

from .constants import EVIDENCE_VARIANTS, EXACT_MODEL_IDS, PROTOCOLS
from .contracts import assert_no_gt_payload, validate_display_mapping
from .io import sha256_json


REQUEST_HASH_VERSION = "api_only_gemini_request_v6_revision_bound"

BASE_FIELDS = (
    "center_x", "center_y", "angle_deg", "width_px", "height_px",
)
MASK_FIELDS = (
    "center_probability", "centre_in_predicted_mask", "rectangle_mask_coverage",
    "grasp_axis_mask_support", "left_contact_mask_support", "right_contact_mask_support",
    "minimum_contact_mask_support", "distance_to_mask_centroid_px",
    "distance_to_mask_boundary_px", "mask_width_along_closing_axis_px",
    "candidate_width_to_mask_width_ratio",
)
DEPTH_GEOMETRY_FIELDS = (
    "contact_points_xy", "center_depth_m", "center_depth_valid",
    "local_valid_depth_fraction", "left_contact_median_depth_m",
    "right_contact_median_depth_m", "absolute_contact_depth_difference_m",
    "local_depth_std_m", "depth_gradient_along_closing_axis",
    "foreground_background_depth_gap_m", "missing_depth_fraction",
    "contact_points_inside_image", "gripper_rectangle_inside_image_fraction",
    "predicted_mask_support_at_both_contacts", "sweep_region_mask_fraction",
    "sweep_region_depth_discontinuity", "border_distance_px",
)
SCORE_FIELDS = (
    "original_score", "original_rank", "score_to_top1_ratio",
    "score_gap_to_original_top1",
)


def _json_safe(value: Any) -> Any:
    """Map unavailable optional evidence to JSON null without inventing data."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def evidence_fields(variant: str) -> tuple[str, ...]:
    if variant not in EVIDENCE_VARIANTS:
        raise ValueError(f"unknown evidence variant: {variant}")
    fields = BASE_FIELDS
    if variant != "E0_RGB_ONLY":
        fields += MASK_FIELDS
    if variant in {"E2_RGBD_GEOMETRY_SCORE_BLIND", "E3_RGBD_GEOMETRY_SCORE_AWARE"}:
        fields += DEPTH_GEOMETRY_FIELDS
    if variant == "E3_RGBD_GEOMETRY_SCORE_AWARE":
        fields += SCORE_FIELDS
    return fields


def build_text_payload(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    language: str,
    variant: str,
    protocol: str,
    display_mapping: Mapping[str, str],
    image_width: int = 640,
    image_height: int = 480,
) -> dict[str, Any]:
    if backend not in {"G1", "C1"}:
        raise ValueError("backend must be G1 or C1")
    if protocol not in PROTOCOLS[:2]:
        raise ValueError("only P1/P2 are provider protocols; P3-P5 are deterministic derivations")
    candidate_ids = [str(row["candidate_id"]) for row in evidence_rows]
    validate_display_mapping(display_mapping, candidate_ids)
    fields = evidence_fields(variant)
    candidates = []
    for row in evidence_rows:
        candidate = {"display_id": str(display_mapping[str(row["candidate_id"])])}
        candidate.update({field: _json_safe(row.get(field)) for field in fields})
        candidates.append(candidate)
    candidates.sort(key=lambda row: row["display_id"])
    payload = {
        # Keep the request contract name distinct from the response field
        # ``schema_version``.  Both exact models otherwise copied the request
        # version into an otherwise schema-valid response.
        "request_payload_version": "api_only_gemini_text_payload_v1",
        "scene_instruction": str(language),
        "backend": backend,
        "evidence_variant": variant,
        "protocol": protocol,
        "image_coordinates": {
            "origin": "top-left", "x": "column, increases rightward",
            "y": "row, increases downward", "width_px": int(image_width),
            "height_px": int(image_height),
            "angle": "degrees in [-90,90); 180-degree periodic; rectangle width/parallel-jaw closing axis; visually positive clockwise because y increases downward",
            "contact_points": "center +/- width/2 * (cos(angle), sin(angle))",
            "rectangle_height_px": 20,
        },
        "display_identity_note": "A-E are randomized display identities and do not encode original rank.",
        "candidates": candidates,
    }
    if protocol == "P2_API_BASELINE_AWARE":
        baseline_internal = str(evidence_rows[0]["candidate_id"])
        payload["original_top1_display_id"] = str(display_mapping[baseline_internal])
    assert_no_gt_payload(payload)
    return payload


def generation_config(model_id: str) -> dict[str, Any]:
    if model_id not in EXACT_MODEL_IDS:
        raise ValueError("exact model substitution is forbidden")
    config: dict[str, Any] = {"thinking_level": "medium", "max_output_tokens": 4096}
    if model_id == "gemini-robotics-er-2-preview":
        config["temperature"] = 0.0
    else:
        config["temperature_policy"] = "omitted: deprecated for Gemini 3.6 Flash"
    return config


def logical_request_hash(bindings: Mapping[str, Any]) -> str:
    required = {
        "backend", "sample_id", "model_id", "protocol", "evidence_variant",
        "prompt_version", "prompt_hash", "response_schema_hash", "renderer_hash",
        "scene_board_hash", "candidate_board_hash", "candidate_set_hash",
        "display_mapping", "generation_config", "endpoint_type",
        "text_payload_hash", "response_parser_version", "model_revision",
        "sdk_version",
    }
    missing = sorted(required - set(bindings))
    if missing:
        raise ValueError(f"request bindings missing: {missing}")
    if bindings["model_id"] not in EXACT_MODEL_IDS:
        raise ValueError("exact model substitution is forbidden")
    assert_no_gt_payload(bindings)
    return sha256_json({"hash_version": REQUEST_HASH_VERSION, **dict(bindings)})
