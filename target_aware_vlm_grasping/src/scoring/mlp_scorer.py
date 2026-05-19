from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scoring.rule_based_scorer import DEFAULT_WEIGHTS
from scoring.score_normalization import clamp_score
from scoring.scorer_interface import ScorerInterface
from utils.data_types import CandidateFeatureVector, GraspCandidate, ScoredGrasp


FEATURE_NAMES = [
    "initial_geometric_score",
    "target_overlap",
    "center_alignment",
    "gripper_width_match",
    "depth_stability",
    "approach_direction_score",
    "collision_penalty",
    "boundary_penalty",
]


class MLPScorer(ScorerInterface):
    """Small CPU-only MLP scoring head for semantic-geometric grasp ranking."""

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: float,
        feature_names: list[str] | None = None,
        source: str = "configured",
    ):
        self.feature_names = feature_names or FEATURE_NAMES
        self.w1 = np.asarray(w1, dtype=float)
        self.b1 = np.asarray(b1, dtype=float)
        self.w2 = np.asarray(w2, dtype=float).reshape(-1)
        self.b2 = float(b2)
        self.source = source
        if self.w1.shape[1] != len(self.feature_names):
            raise ValueError("MLP W1 input dimension must match feature_names.")
        if self.w1.shape[0] != self.b1.shape[0] or self.w1.shape[0] != self.w2.shape[0]:
            raise ValueError("MLP hidden dimensions are inconsistent.")

    @classmethod
    def from_config(cls, config: dict | None = None) -> "MLPScorer":
        config = config or {}
        checkpoint = config.get("checkpoint_path")
        if checkpoint:
            return cls.from_checkpoint(Path(checkpoint))
        return cls.rule_based_initialization(config.get("feature_names") or FEATURE_NAMES)

    @classmethod
    def from_checkpoint(cls, path: Path) -> "MLPScorer":
        if path.suffix.lower() == ".npz":
            data = np.load(str(path), allow_pickle=True)
            feature_names = list(data["feature_names"]) if "feature_names" in data else FEATURE_NAMES
            return cls(data["w1"], data["b1"], data["w2"], float(data["b2"]), feature_names, source=str(path))
        with open(path) as f:
            data = json.load(f)
        return cls(
            np.asarray(data["w1"], dtype=float),
            np.asarray(data["b1"], dtype=float),
            np.asarray(data["w2"], dtype=float),
            float(data.get("b2", 0.0)),
            list(data.get("feature_names", FEATURE_NAMES)),
            source=str(path),
        )

    @classmethod
    def rule_based_initialization(cls, feature_names: list[str]) -> "MLPScorer":
        n = len(feature_names)
        w1 = np.eye(n, dtype=float)
        b1 = np.zeros(n, dtype=float)
        w2 = np.array([DEFAULT_WEIGHTS.get(name, 0.0) for name in feature_names], dtype=float)
        return cls(w1, b1, w2, 0.0, feature_names, source="rule_based_initialization")

    def score(
        self,
        candidates: list[GraspCandidate],
        features: list[CandidateFeatureVector],
    ) -> list[ScoredGrasp]:
        scored = []
        for candidate, feat in zip(candidates, features):
            x = self._feature_vector(feat)
            hidden = np.maximum(x @ self.w1.T + self.b1, 0.0)
            raw = float(hidden @ self.w2 + self.b2)
            metadata = {
                "scorer": "mlp",
                "mlp_source": self.source,
                "mlp_feature_names": self.feature_names,
                "mlp_feature_vector": x.tolist(),
                "mlp_hidden": hidden.tolist(),
                "raw_score": raw,
                "score_breakdown": self._linearized_breakdown(feat),
            }
            scored.append(ScoredGrasp(candidate, feat, clamp_score(raw), rank=0, scorer_type="mlp", metadata=metadata))
        scored.sort(key=lambda item: item.final_score, reverse=True)
        for rank, sg in enumerate(scored, start=1):
            sg.rank = rank
        return scored

    def _feature_vector(self, feat: CandidateFeatureVector) -> np.ndarray:
        return np.array([float(getattr(feat, name)) for name in self.feature_names], dtype=float)

    def _linearized_breakdown(self, feat: CandidateFeatureVector) -> dict:
        if self.source != "rule_based_initialization":
            return {}
        return {
            name: DEFAULT_WEIGHTS.get(name, 0.0) * float(getattr(feat, name))
            for name in self.feature_names
        }
