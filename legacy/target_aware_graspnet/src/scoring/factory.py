from __future__ import annotations

from scoring.mlp_scorer import MLPScorer
from scoring.rule_based_scorer import RuleBasedScorer
from scoring.scorer_interface import ScorerInterface


def build_scorer(config: dict | None = None) -> ScorerInterface:
    config = config or {}
    method = str(config.get("method", "rule_based")).lower()
    if method in {"mlp", "mlp_scoring_head"}:
        return MLPScorer.from_config(config.get("mlp", {}))
    if method in {"rule", "rule_based", "weighted"}:
        return RuleBasedScorer(config.get("weights"))
    raise ValueError(f"Unknown scoring method: {method}. Use 'rule_based' or 'mlp'.")

