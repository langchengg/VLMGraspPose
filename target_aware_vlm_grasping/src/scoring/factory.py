from __future__ import annotations

from scoring.mlp_scorer import MLPScorer
from scoring.rule_based_scorer import RuleBasedScorer
from scoring.scorer_interface import ScorerInterface
from scoring.xgboost_scorer import XGBoostScorer


def build_scorer(config: dict | None = None) -> ScorerInterface:
    config = config or {}
    method = str(config.get("method", "rule_based")).lower()
    if method in {"mlp", "mlp_scoring_head"}:
        mlp_cfg = config.get("mlp", {})
        checkpoint = mlp_cfg.get("checkpoint_path") if isinstance(mlp_cfg, dict) else None
        if checkpoint:
            from pathlib import Path
            if not Path(checkpoint).exists():
                return RuleBasedScorer(config.get("weights"))
        return MLPScorer.from_config(mlp_cfg)
    if method in {"xgboost", "xgb"}:
        xgb_cfg = config.get("xgboost", {})
        scorer = XGBoostScorer.from_config(xgb_cfg if isinstance(xgb_cfg, dict) else {})
        if isinstance(scorer, RuleBasedScorer):
            return RuleBasedScorer(config.get("weights"))
        return scorer
    if method in {"rule", "rule_based", "weighted"}:
        return RuleBasedScorer(config.get("weights"))
    raise ValueError(f"Unknown scoring method: {method}. Use 'rule_based', 'mlp', or 'xgboost'.")
