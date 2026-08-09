from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator


class PairwiseDecision(str, Enum):
    KEEP_BASELINE = "KEEP_BASELINE"
    PREFER_CHALLENGER = "PREFER_CHALLENGER"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ReasonCode(str, Enum):
    BASELINE_OFF_TARGET = "BASELINE_OFF_TARGET"
    CHALLENGER_OFF_TARGET = "CHALLENGER_OFF_TARGET"
    WEAK_MASK_SUPPORT = "WEAK_MASK_SUPPORT"
    BETTER_MASK_SUPPORT = "BETTER_MASK_SUPPORT"
    CONTACT_DEPTH_INCONSISTENT = "CONTACT_DEPTH_INCONSISTENT"
    WIDTH_MISMATCH = "WIDTH_MISMATCH"
    COLLISION_RISK = "COLLISION_RISK"
    INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"
    AMBIGUOUS_VISUAL_EVIDENCE = "AMBIGUOUS_VISUAL_EVIDENCE"
    NO_CLEAR_ADVANTAGE = "NO_CLEAR_ADVANTAGE"


class PairwiseCriticResponse(BaseModel):
    """Strict provider response. Scores are observations, not probabilities."""

    model_config = ConfigDict(extra="forbid")

    decision: PairwiseDecision
    evidence_reliable: StrictBool
    baseline_target_alignment: float = Field(ge=0.0, le=1.0, strict=True)
    challenger_target_alignment: float = Field(ge=0.0, le=1.0, strict=True)
    baseline_contact_geometry: float = Field(ge=0.0, le=1.0, strict=True)
    challenger_contact_geometry: float = Field(ge=0.0, le=1.0, strict=True)
    baseline_collision_risk: float = Field(ge=0.0, le=1.0, strict=True)
    challenger_collision_risk: float = Field(ge=0.0, le=1.0, strict=True)
    baseline_width_compatibility: float = Field(ge=0.0, le=1.0, strict=True)
    challenger_width_compatibility: float = Field(ge=0.0, le=1.0, strict=True)
    reason_codes: list[ReasonCode] = Field(min_length=1, max_length=10)
    brief_rationale: str = Field(min_length=1, max_length=320)

    @field_validator("brief_rationale")
    @classmethod
    def rationale_has_at_most_40_words(cls, value: str) -> str:
        if len(re.findall(r"\b\w+\b", value, flags=re.UNICODE)) > 40:
            raise ValueError("brief_rationale must contain at most 40 words")
        return value


def parse_pairwise_response(raw: str) -> PairwiseCriticResponse:
    """Accept exactly one bare JSON object; never repair prose or code fences."""

    if not isinstance(raw, str):
        raise TypeError("provider response must be text")
    if raw != raw.strip():
        raise ValueError("response must not contain leading or trailing text")
    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(raw)
    if end != len(raw) or not isinstance(value, dict):
        raise ValueError("response must be one bare JSON object")
    return PairwiseCriticResponse.model_validate(value)


def pairwise_response_schema() -> dict[str, Any]:
    return PairwiseCriticResponse.model_json_schema()


class FullListDecision(str, Enum):
    KEEP_BASELINE = "KEEP_BASELINE"
    PREFER_CANDIDATE = "PREFER_CANDIDATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class FullListKeepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: FullListDecision
    selected_candidate_id: str = Field(pattern=r"^candidate_[0-4]$")
    evidence_reliable: StrictBool
    reason_codes: list[str] = Field(min_length=1, max_length=8)
    brief_rationale: str = Field(min_length=1, max_length=320)

    @field_validator("brief_rationale")
    @classmethod
    def full_list_rationale_has_at_most_40_words(cls, value: str) -> str:
        if len(re.findall(r"\b\w+\b", value, flags=re.UNICODE)) > 40:
            raise ValueError("brief_rationale must contain at most 40 words")
        return value

    @field_validator("selected_candidate_id")
    @classmethod
    def keep_maps_to_baseline(cls, value: str, info: Any) -> str:
        decision = info.data.get("decision")
        if decision in (FullListDecision.KEEP_BASELINE, FullListDecision.INSUFFICIENT_EVIDENCE) and value != "candidate_0":
            raise ValueError("KEEP/INSUFFICIENT must select frozen baseline candidate_0")
        return value


def parse_full_list_keep_response(raw: str) -> FullListKeepResponse:
    if raw != raw.strip():
        raise ValueError("response must not contain leading or trailing text")
    value, end = json.JSONDecoder().raw_decode(raw)
    if end != len(raw) or not isinstance(value, dict):
        raise ValueError("response must be one bare JSON object")
    return FullListKeepResponse.model_validate(value)
