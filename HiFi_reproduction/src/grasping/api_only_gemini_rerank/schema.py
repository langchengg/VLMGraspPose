"""Strict response validation without natural-language repair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .contracts import map_display_selection, validate_display_mapping


RESPONSE_PARSER_VERSION = "strict_json_no_repair_v1"


@dataclass(frozen=True)
class ParsedResponse:
    decision: str
    selected_internal_candidate_id: str
    ordered_internal_candidate_ids: tuple[str, ...]
    evidence_reliability: str
    switch_confidence: int
    reason_codes: tuple[str, ...]
    parsed: Mapping[str, Any]
    warnings: tuple[str, ...]


def load_response_schema(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("response schema must be an object")
    return value


def parse_response(
    raw_text: str,
    *,
    schema: Mapping[str, Any],
    display_mapping: Mapping[str, str],
    candidate_ids: Sequence[str],
) -> ParsedResponse:
    validate_display_mapping(display_mapping, candidate_ids)
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("response is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    jsonschema.Draft202012Validator(dict(schema)).validate(value)
    expected_display = set(display_mapping.values())
    ordered = list(map(str, value["ordered_candidate_ids"]))
    if len(ordered) != len(expected_display) or set(ordered) != expected_display:
        raise ValueError("ordered_candidate_ids is not the exact total permutation")
    assessments = [str(row["candidate_id"]) for row in value["candidate_assessments"]]
    if len(assessments) != len(expected_display) or set(assessments) != expected_display:
        raise ValueError("candidate_assessments does not cover each candidate exactly once")
    selected = map_display_selection(str(value["selected_candidate_id"]), display_mapping)
    ordered_internal = tuple(map_display_selection(item, display_mapping) for item in ordered)
    warnings: list[str] = []
    if len(str(value["brief_rationale"]).split()) > 50:
        warnings.append("RATIONALE_OVER_50_WORDS")
    return ParsedResponse(
        decision=str(value["decision"]),
        selected_internal_candidate_id=selected,
        ordered_internal_candidate_ids=ordered_internal,
        evidence_reliability=str(value["evidence_reliability"]),
        switch_confidence=int(value["switch_confidence"]),
        reason_codes=tuple(map(str, value["reason_codes"])),
        parsed=value,
        warnings=tuple(warnings),
    )


def fallback_selection(
    baseline_candidate_id: str,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "selected_candidate_id": str(baseline_candidate_id),
        "decision": "KEEP_ORIGINAL_TOP1",
        "fallback": True,
        "fallback_reason": str(reason),
    }
