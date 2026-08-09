from __future__ import annotations

import hashlib
import json
import math
import re
from html import escape
from typing import Any


FORBIDDEN_INFERENCE_FIELDS = (
    "program",
    "answer",
    "target",
    "target_idx",
    "box",
    "grasps",
    "objid",
    "gt_mask",
    "gt_grasp",
    "gt_qua",
    "gt_sin",
    "gt_cos",
    "gt_wid",
    "candidate_positive",
    "candidate_correct",
    "candidate_iou",
    "angle_error",
    "correct_candidate_id",
    "first_valid_rank",
    "recovered",
    "harmful",
    "oracle",
    "j@1",
    "j@any",
    "critic_score",
    "latent_residual",
    "setrank_score",
    "gate_score",
)

EVALUATION_ONLY_FIELDS = (
    "program",
    "answer",
    "target",
    "target_idx",
    "box",
    "grasps",
    "objID",
    "gt_mask",
    "gt_qua",
    "gt_sin",
    "gt_cos",
    "gt_wid",
    "candidate_iou",
    "candidate_angle_error",
    "candidate_correctness",
    "first_valid_rank",
    "j1_label",
    "jany_label",
    "recovered",
    "harmful",
    "oracle",
)

INFERENCE_INPUT_ALLOWLIST = {
    "raw_inputs": ["rgb", "referring_expression"],
    "crog_predictions": [
        "m_logit",
        "m_probability",
        "m_binary_mask",
        "q_raw",
        "q_probability",
        "sin_2theta_raw",
        "cos_2theta_raw",
        "decoded_angle",
        "angle_vector_magnitude",
        "w_raw",
        "w_probability",
        "decoded_width",
        "frozen_top5_geometry",
        "frozen_q_value",
        "frozen_original_rank",
    ],
    "derived_without_ground_truth": [
        "q_local_statistics",
        "predicted_mask_support",
        "angle_consistency",
        "width_consistency",
        "candidate_relations",
        "candidate_overlap",
        "candidate_uncertainty",
    ],
}

CANDIDATE_IDENTITY_FIELDS = (
    "candidate_id",
    "candidate_checksum",
    "cx",
    "cy",
    "row",
    "col",
    "angle_deg",
    "width_px",
    "height_px",
    "polygon",
    "q_raw",
)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9@]+", "_", value.lower()).strip("_")


def assert_no_gt_leak(value: Any, *, path: str = "request") -> None:
    """Reject forbidden keys and explicit evaluator tokens before API calls."""
    normalized_forbidden = {_normalize_key(item) for item in FORBIDDEN_INFERENCE_FIELDS}
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalize_key(str(key))
            explicitly_allowed = normalized in {
                "target_alignment_score",
                "target_alignment",
                "target_misalignment",
                "weak_mask_support",
            }
            hit = None
            if not explicitly_allowed:
                hit = next(
                    (
                        item
                        for item in normalized_forbidden
                        if normalized == item
                        or normalized.startswith(item + "_")
                        or normalized.endswith("_" + item)
                    ),
                    None,
                )
            if hit is not None:
                raise ValueError(f"forbidden inference field at {path}.{key}: {hit}")
            assert_no_gt_leak(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_gt_leak(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and path.endswith((".prompt", ".system_instruction")):
        for token in ("candidate_correct", "correct_candidate_id", "first_valid_rank", "J@Any"):
            if token.lower() in value.lower():
                raise ValueError(f"forbidden evaluator token in {path}: {token}")


def assert_no_gt_values_in_provider_text(
    provider_text: str,
    *,
    correct_candidate_ids: Any,
    gt_grasps: Any,
) -> None:
    """Reject exact GT-only identifiers or grasp tuples in provider-visible text.

    Scalar predicted values can legitimately coincide with a GT scalar, so this
    audit intentionally checks identities and multi-value grasp tuples while
    source/layer provenance proves that all displayed geometry came from CROG
    predictions rather than evaluator data.
    """

    text = str(provider_text)
    for candidate_id in {str(value) for value in correct_candidate_ids}:
        if candidate_id and re.search(rf"(?<![A-Za-z0-9_]){re.escape(candidate_id)}(?![A-Za-z0-9_])", text):
            raise ValueError("provider-visible text contains a GT-only candidate identity")
    for grasp in gt_grasps or []:
        if not isinstance(grasp, (list, tuple)) or len(grasp) < 4:
            continue
        signature = r"\s*[,;|]\s*".join(
            re.escape(f"{float(value):.6f}") for value in grasp[:4]
        )
        if re.search(signature, text):
            raise ValueError("provider-visible text contains a GT grasp-coordinate tuple")


def candidate_identity_payload(candidates: Any) -> list[dict[str, Any]]:
    """Return a strict, canonical identity view of one frozen Top-5 set."""

    if not isinstance(candidates, (list, tuple)) or len(candidates) != 5:
        raise ValueError("candidate identity requires exactly five candidates")
    payload: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index} is not an object")
        missing = [name for name in CANDIDATE_IDENTITY_FIELDS if name not in candidate]
        if missing:
            raise ValueError(f"candidate {index} is missing identity fields: {missing}")
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in identifiers:
            raise ValueError(f"duplicate candidate identity: {candidate_id}")
        identifiers.add(candidate_id)
        polygon = candidate["polygon"]
        if not isinstance(polygon, (list, tuple)) or len(polygon) != 4:
            raise ValueError(f"candidate {candidate_id} polygon must contain four corners")
        corners = []
        for corner in polygon:
            if not isinstance(corner, (list, tuple)) or len(corner) != 2:
                raise ValueError(f"candidate {candidate_id} has an invalid polygon corner")
            pair = [float(corner[0]), float(corner[1])]
            if not all(math.isfinite(value) for value in pair):
                raise ValueError(f"candidate {candidate_id} polygon is not finite")
            corners.append(pair)
        numeric = {
            name: float(candidate[name])
            for name in ("cx", "cy", "row", "col", "angle_deg", "width_px", "height_px", "q_raw")
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError(f"candidate {candidate_id} identity is not finite")
        payload.append(
            {
                "candidate_id": candidate_id,
                "candidate_checksum": str(candidate["candidate_checksum"]),
                **numeric,
                "polygon": corners,
            }
        )
    return payload


def assert_candidate_identity(
    candidates: Any,
    *,
    frozen_candidates: Any | None = None,
    expected_sha256: str | None = None,
) -> str:
    """Fail closed if frozen geometry, q values, or ordering changed."""

    payload = candidate_identity_payload(candidates)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if frozen_candidates is not None and payload != candidate_identity_payload(frozen_candidates):
        raise ValueError("candidate identity changed from the frozen Top-5 payload")
    if expected_sha256 is not None and digest != str(expected_sha256):
        raise ValueError("candidate identity SHA-256 changed")
    return digest


def assert_display_mapping(mapping: Any, *, candidate_ids: Any | None = None) -> None:
    if not isinstance(mapping, dict):
        raise ValueError("candidate mapping must be an object")
    display_to_candidate = mapping.get("display_to_candidate")
    candidate_to_display = mapping.get("candidate_to_display")
    displays = {"A", "B", "C", "D", "E"}
    if not isinstance(display_to_candidate, dict) or set(display_to_candidate) != displays:
        raise ValueError("display mapping must contain A-E exactly once")
    values = [str(display_to_candidate[name]) for name in sorted(displays)]
    if len(set(values)) != 5:
        raise ValueError("display mapping candidate IDs must be unique")
    expected_inverse = {candidate_id: display for display, candidate_id in display_to_candidate.items()}
    if candidate_to_display != expected_inverse:
        raise ValueError("candidate_to_display is not the exact inverse mapping")
    if candidate_ids is not None and set(values) != {str(value) for value in candidate_ids}:
        raise ValueError("display mapping does not match frozen candidate IDs")


def wrap_untrusted_referring_expression(text: str) -> str:
    safe = escape(str(text), quote=False)
    return f"<referring_expression>\n{safe}\n</referring_expression>"


def redact_sensitive(value: Any, *, api_key: str | None = None) -> Any:
    """Return a recursively redacted copy suitable for logs/artifacts."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"x-goog-api-key", "authorization", "api_key", "apikey"}:
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_sensitive(child, api_key=api_key)
        return result
    if isinstance(value, list):
        return [redact_sensitive(item, api_key=api_key) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item, api_key=api_key) for item in value)
    if isinstance(value, str):
        text = value
        if api_key:
            text = text.replace(api_key, "[REDACTED]")
            if len(api_key) >= 8:
                for fragment in (api_key[:4], api_key[-4:]):
                    text = text.replace(fragment, "[REDACTED]")
        text = re.sub(
            r"(?i)(x-goog-api-key|authorization)\s*[:=]\s*[^\s,}]+",
            r"\1=[REDACTED]",
            text,
        )
        return text
    return value


def canonical_request_json(value: dict[str, Any]) -> str:
    assert_no_gt_leak(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
