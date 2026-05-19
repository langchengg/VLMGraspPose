from __future__ import annotations

from pathlib import Path

import numpy as np

from scoring.mlp_scorer import FEATURE_NAMES, MLPScorer


def save_rule_initialized_checkpoint(path: Path, feature_names: list[str] | None = None) -> Path:
    """Save a CPU-only MLP checkpoint initialized to mimic rule-based scoring.

    This is intentionally lightweight. Real training can replace this file with
    learned weights without changing the inference runner.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scorer = MLPScorer.rule_based_initialization(feature_names or FEATURE_NAMES)
    np.savez(
        str(path),
        w1=scorer.w1,
        b1=scorer.b1,
        w2=scorer.w2,
        b2=np.asarray(scorer.b2),
        feature_names=np.asarray(scorer.feature_names),
    )
    return path
