from __future__ import annotations

from scoring.score_normalization import clamp_score
from scoring.scorer_interface import ScorerInterface
from utils.data_types import CandidateFeatureVector, GraspCandidate, ScoredGrasp


DEFAULT_WEIGHTS = {
    "initial_geometric_score": 0.20,
    "target_overlap": 0.25,
    "center_alignment": 0.15,
    "gripper_width_match": 0.10,
    "depth_stability": 0.10,
    "approach_direction_score": 0.10,
    "collision_penalty": -0.07,
    "boundary_penalty": -0.03,
}


class RuleBasedScorer(ScorerInterface):
    def __init__(self, weights: dict | None = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def score(
        self,
        candidates: list[GraspCandidate],
        features: list[CandidateFeatureVector],
    ) -> list[ScoredGrasp]:
        scored = []
        for candidate, feat in zip(candidates, features):
            raw = (
                self.weights.get("initial_geometric_score", 0.20) * feat.initial_geometric_score
                + self.weights.get("target_overlap", 0.25) * feat.target_overlap
                + self.weights.get("center_alignment", 0.15) * feat.center_alignment
                + self.weights.get("gripper_width_match", 0.10) * feat.gripper_width_match
                + self.weights.get("depth_stability", 0.10) * feat.depth_stability
                + self.weights.get("approach_direction_score", 0.10) * feat.approach_direction_score
                + self.weights.get("collision_penalty", -0.07) * feat.collision_penalty
                + self.weights.get("boundary_penalty", -0.03) * feat.boundary_penalty
            )
            metadata = {
                "score_breakdown": self._breakdown(feat),
                "scoring_weights": dict(self.weights),
                "scoring_formula": (
                    "0.20*initial_geometric_score + 0.25*target_overlap + "
                    "0.15*center_alignment + 0.10*gripper_width_match + "
                    "0.10*depth_stability + 0.10*approach_direction_score - "
                    "0.07*collision_penalty - 0.03*boundary_penalty"
                ),
            }
            scored.append(ScoredGrasp(candidate, feat, clamp_score(raw), rank=0, scorer_type="rule_based", metadata=metadata))
        scored.sort(key=lambda x: x.final_score, reverse=True)
        for i, sg in enumerate(scored, start=1):
            sg.rank = i
        return scored

    def _breakdown(self, feat: CandidateFeatureVector) -> dict:
        return {
            key: self.weights.get(key, 0.0) * getattr(feat, key)
            for key in DEFAULT_WEIGHTS
        }
