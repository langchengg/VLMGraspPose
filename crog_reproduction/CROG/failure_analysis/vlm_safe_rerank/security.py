from __future__ import annotations

import json
import re
from typing import Any, Mapping


FORBIDDEN_KEYS = {
    "program", "answer", "target", "target_idx", "box", "grasps", "objid",
    "gt_mask", "gt_grasp", "gt_qua", "gt_sin", "gt_cos", "gt_wid",
    "candidate_positive", "candidate_correct", "candidate_iou", "angle_error",
    "correct_candidate_id", "first_valid_rank", "recovered", "harmful", "oracle",
    "j@1", "j@any", "legacy_correct", "corrected_correct", "evaluation_only",
}

FORBIDDEN_KEY_TOKENS = {
    "answer", "answers", "correct", "corrected", "correctness", "eval",
    "evaluation", "evaluator", "groundtruth", "gt", "label", "labels",
    "legacy", "oracle", "targetidx",
}
SAFE_PRESENTATION_LABEL_KEYS = {"baseline_label", "challenger_label", "display_label"}
FORBIDDEN_PATH_TOKENS = {
    "corrected", "eval", "evaluation", "evaluator", "groundtruth", "gt",
    "label", "labels", "legacy", "oracle",
}


def _tokens(value: str) -> set[str]:
    parts = re.findall(r"[a-z0-9]+", value.strip().lower())
    return set(parts) | ({"".join(parts)} if parts else set())


def _walk(value: Any, path: str = "$", parent_key: str | None = None) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            key_tokens = _tokens(normalized)
            if (
                normalized in FORBIDDEN_KEYS
                or normalized.startswith("gt_")
                or (
                    key_tokens.intersection(FORBIDDEN_KEY_TOKENS)
                    and normalized not in SAFE_PRESENTATION_LABEL_KEYS
                )
            ):
                raise AssertionError(f"forbidden evaluation field at {path}.{key}")
            _walk(item, f"{path}.{key}", normalized)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, f"{path}[{index}]", parent_key)
    elif isinstance(value, str) and parent_key is not None:
        # File paths are never needed by the provider.  Reject evaluator/label
        # path smuggling while allowing ordinary referring-expression text.
        parent_tokens = _tokens(parent_key)
        looks_like_path = bool(
            parent_tokens.intersection({"path", "file", "filepath", "source", "uri"})
        ) or value.lower().endswith((".json", ".jsonl", ".parquet", ".csv"))
        if looks_like_path and _tokens(value).intersection(FORBIDDEN_PATH_TOKENS):
            raise AssertionError(f"forbidden evaluation path at {path}")


def assert_no_ground_truth(payload: Mapping[str, Any]) -> None:
    _walk(payload)
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    forbidden_tokens = ("ground truth", "oracle@", "candidate correctness", "corrected evaluator", "legacy evaluator")
    lowered = serialized.lower()
    for token in forbidden_tokens:
        if token in lowered:
            raise AssertionError(f"forbidden evaluation concept in request: {token}")


def assert_frozen_pair(
    payload: Mapping[str, Any], *, baseline_id: str, challenger_id: str,
    baseline_expected: Mapping[str, Any] | None = None,
    challenger_expected: Mapping[str, Any] | None = None,
) -> None:
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise AssertionError("request missing deterministic evidence")
    if evidence.get("baseline", {}).get("candidate_id") != baseline_id:
        raise AssertionError("baseline identity drift")
    if evidence.get("challenger", {}).get("candidate_id") != challenger_id:
        raise AssertionError("challenger identity drift")
    for side, expected in (
        ("baseline", baseline_expected), ("challenger", challenger_expected)
    ):
        if expected is None:
            continue
        actual = evidence.get(side)
        if not isinstance(actual, Mapping):
            raise AssertionError(f"{side} evidence is missing")
        comparisons = {
            "candidate_checksum": "candidate_checksum",
            "q": "q_raw",
            "centre_x": "cx",
            "centre_y": "cy",
            "angle_deg": "angle_deg",
            "width_px": "width_px",
            "height_px": "height_px",
        }
        for actual_key, expected_key in comparisons.items():
            if actual_key not in actual or expected_key not in expected:
                raise AssertionError(f"{side} frozen identity lacks {actual_key}")
            actual_value = actual[actual_key]
            expected_value = expected[expected_key]
            if actual_key == "candidate_checksum":
                equal = str(actual_value) == str(expected_value)
            else:
                try:
                    equal = abs(float(actual_value) - float(expected_value)) <= 1e-12
                except (TypeError, ValueError):
                    equal = False
            if not equal:
                raise AssertionError(f"{side} frozen identity drift in {actual_key}")
    forbidden_geometry = {"new_coordinates", "proposed_grasp", "generated_grasp"}
    if forbidden_geometry.intersection({str(key).lower() for key in payload}):
        raise AssertionError("request asks provider to generate candidate geometry")


def secret_pattern_hits(text: str) -> list[str]:
    patterns = [r"AIza[0-9A-Za-z_-]{20,}", r"AQ\.[0-9A-Za-z_-]{20,}", r"X-goog-api-key\s*:"]
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]
