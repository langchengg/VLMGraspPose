from __future__ import annotations

import abc

from utils.data_types import CandidateFeatureVector, GraspCandidate, ScoredGrasp


class ScorerInterface(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        candidates: list[GraspCandidate],
        features: list[CandidateFeatureVector],
    ) -> list[ScoredGrasp]:
        ...

    def top_k(
        self,
        candidates: list[GraspCandidate],
        features: list[CandidateFeatureVector],
        k: int,
    ) -> list[ScoredGrasp]:
        return self.score(candidates, features)[:k]
