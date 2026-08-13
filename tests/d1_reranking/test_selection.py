from __future__ import annotations

import pandas as pd
import pytest

from d1_reranking.selection import ensemble_seed_scores, select_validation_winner


def _predictions(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "native_rank": [1, 2],
            "candidate_identity_sha256": ["i-a", "i-b"],
            "candidate_geometry_sha256": ["g-a", "g-b"],
            "score": [0.2 + offset, 0.8 + offset],
        }
    )


def test_three_seed_ensemble_is_exact_and_deterministic() -> None:
    value = ensemble_seed_scores(
        {42: _predictions(0.0), 123: _predictions(0.3), 2026: _predictions(0.6)}
    )
    assert value["ensemble_score"].tolist() == pytest.approx([0.5, 1.1])
    mutated = _predictions(0.6).copy()
    mutated.loc[1, "candidate_geometry_sha256"] = "drift"
    with pytest.raises(ValueError, match="universes differ"):
        ensemble_seed_scores(
            {42: _predictions(0.0), 123: _predictions(0.3), 2026: mutated}
        )


def test_validation_selection_uses_declared_metric_and_stable_ties() -> None:
    rows = [
        {
            "method": "R6",
            "validation_j_at_1": 0.5,
            "validation_mrr_at_5": 0.7,
            "validation_ndcg_at_5": 0.8,
        },
        {
            "method": "R3",
            "validation_j_at_1": 0.5,
            "validation_mrr_at_5": 0.7,
            "validation_ndcg_at_5": 0.8,
        },
    ]
    assert select_validation_winner(rows, final_tie_column="method")["method"] == "R3"
