from __future__ import annotations

import json
import math
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CandidateId = Literal["A", "B", "C", "D", "E"]
BoundedScore = Annotated[float, Field(ge=0.0, le=1.0)]


class AllowedReasonCode(str, Enum):
    target_alignment = "target_alignment"
    target_misalignment = "target_misalignment"
    mask_support = "mask_support"
    weak_mask_support = "weak_mask_support"
    both_jaws_supported = "both_jaws_supported"
    one_jaw_unsupported = "one_jaw_unsupported"
    quality_peak_support = "quality_peak_support"
    isolated_quality_spike = "isolated_quality_spike"
    angle_consistent = "angle_consistent"
    angle_inconsistent = "angle_inconsistent"
    width_consistent = "width_consistent"
    width_inconsistent = "width_inconsistent"
    crosses_background = "crosses_background"
    crosses_distractor = "crosses_distractor"
    edge_risk = "edge_risk"
    candidate_redundancy = "candidate_redundancy"
    q_prior_support = "q_prior_support"
    q_prior_conflict = "q_prior_conflict"
    ambiguous_target = "ambiguous_target"
    insufficient_evidence = "insufficient_evidence"
    keep_original_due_uncertainty = "keep_original_due_uncertainty"


class StrictModel(BaseModel):
    # JSON enum members arrive as their string values.  Extra fields remain
    # forbidden; numeric range/finite checks below provide the required score
    # strictness without rejecting valid JSON enum strings.
    model_config = ConfigDict(extra="forbid")


class CandidateAssessment(StrictModel):
    candidate_id: CandidateId
    target_alignment_score: BoundedScore
    mask_support_score: BoundedScore
    quality_evidence_score: BoundedScore
    angle_consistency_score: BoundedScore
    width_consistency_score: BoundedScore
    edge_safety_score: BoundedScore
    overall_score: BoundedScore
    reason_codes: list[AllowedReasonCode]

    @model_validator(mode="after")
    def validate_values(self) -> "CandidateAssessment":
        values = (
            self.target_alignment_score,
            self.mask_support_score,
            self.quality_evidence_score,
            self.angle_consistency_score,
            self.width_consistency_score,
            self.edge_safety_score,
            self.overall_score,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate scores must be finite")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("reason_codes must be unique")
        return self


class GeminiRankingResponse(StrictModel):
    selected_candidate_id: CandidateId
    ranking: Annotated[list[CandidateAssessment], Field(min_length=5, max_length=5)]
    confidence: BoundedScore
    score_margin_top1_top2: BoundedScore
    decision: Literal["keep_original", "switch", "abstain"]
    global_reason_codes: list[AllowedReasonCode]

    @model_validator(mode="after")
    def validate_ranking(self) -> "GeminiRankingResponse":
        identifiers = [item.candidate_id for item in self.ranking]
        if set(identifiers) != {"A", "B", "C", "D", "E"}:
            raise ValueError("ranking must contain A-E exactly once")
        if self.selected_candidate_id != identifiers[0]:
            raise ValueError("selected_candidate_id must equal ranking[0]")
        scores = [item.overall_score for item in self.ranking]
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise ValueError("ranking overall_score must be non-increasing")
        if not math.isfinite(self.confidence) or not math.isfinite(
            self.score_margin_top1_top2
        ):
            raise ValueError("response scores must be finite")
        if len(self.global_reason_codes) != len(set(self.global_reason_codes)):
            raise ValueError("global_reason_codes must be unique")
        return self


def parse_ranking_response(raw: str) -> GeminiRankingResponse:
    """Parse only a bare JSON object; fenced/prose responses are invalid."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return GeminiRankingResponse.model_validate(value)


def response_json_schema() -> dict:
    schema = GeminiRankingResponse.model_json_schema(mode="validation")
    schema["additionalProperties"] = False
    return schema
