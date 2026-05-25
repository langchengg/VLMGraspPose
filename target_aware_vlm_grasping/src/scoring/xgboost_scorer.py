from __future__ import annotations

from pathlib import Path

import numpy as np

from scoring.mlp_scorer import FEATURE_NAMES
from scoring.rule_based_scorer import RuleBasedScorer
from scoring.score_normalization import clamp_score
from scoring.scorer_interface import ScorerInterface
from utils.data_types import CandidateFeatureVector, GraspCandidate, ScoredGrasp


class XGBoostScorer(ScorerInterface):
    """Optional XGBoost re-ranker.

    This is intentionally optional. The core Mac-compatible pipeline must not
    require xgboost or a trained checkpoint. Use `from_config`; it returns a
    rule-based scorer if no usable model is available.
    """

    def __init__(self, booster, feature_names: list[str] | None = None, source: str = "xgboost"):
        self.booster = booster
        self.feature_names = feature_names or FEATURE_NAMES
        self.source = source

    @classmethod
    def from_config(cls, config: dict | None = None) -> ScorerInterface:
        config = config or {}
        checkpoint = config.get("checkpoint_path")
        if not checkpoint:
            return RuleBasedScorer()
        path = Path(checkpoint)
        if not path.exists():
            return RuleBasedScorer()
        try:
            import xgboost as xgb
        except ImportError:
            return RuleBasedScorer()
        booster = xgb.Booster()
        booster.load_model(str(path))
        return cls(booster, config.get("feature_names") or FEATURE_NAMES, source=str(path))

    def score(
        self,
        candidates: list[GraspCandidate],
        features: list[CandidateFeatureVector],
    ) -> list[ScoredGrasp]:
        import xgboost as xgb

        if not candidates:
            return []
        x = np.asarray([self._feature_vector(feat) for feat in features], dtype=float)
        pred = self.booster.predict(xgb.DMatrix(x, feature_names=self.feature_names))
        scored = []
        for candidate, feat, raw in zip(candidates, features, pred):
            metadata = {
                "scorer": "xgboost",
                "xgboost_source": self.source,
                "xgboost_feature_names": self.feature_names,
                "xgboost_feature_vector": self._feature_vector(feat).tolist(),
                "raw_score": float(raw),
            }
            scored.append(ScoredGrasp(
                candidate,
                feat,
                clamp_score(float(raw)),
                rank=0,
                scorer_type="xgboost",
                metadata=metadata,
            ))
        scored.sort(key=lambda item: item.final_score, reverse=True)
        for rank, sg in enumerate(scored, start=1):
            sg.rank = rank
        return scored

    def _feature_vector(self, feat: CandidateFeatureVector) -> np.ndarray:
        return np.array([float(getattr(feat, name)) for name in self.feature_names], dtype=float)
