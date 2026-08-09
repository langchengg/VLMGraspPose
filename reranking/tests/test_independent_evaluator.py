from __future__ import annotations

import pandas as pd
import pytest

from reranking.independent_evaluator import (
    IndependentEvaluationError,
    independent_j_at_1,
)


def test_independent_evaluator_counts_empty_queries_and_uses_id_tiebreak() -> None:
    reference = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "candidate_id": ["b", "a", "a"],
            "label": [0, 1, 1],
        }
    )
    predictions = pd.DataFrame(
        {
            "query_id": ["q2", "q1", "q1"],
            "candidate_id": ["a", "b", "a"],
            "score": [0.1, 0.5, 0.5],
        }
    )
    universe = pd.DataFrame({"query_id": ["q1", "q2", "empty"]})
    result = independent_j_at_1(reference, predictions, universe)
    assert result["j_at_1_count"] == 2
    assert result["oracle_count"] == 2
    assert result["empty_query_count"] == 1
    assert result["query_count"] == 3


def test_independent_evaluator_rejects_missing_candidate_prediction() -> None:
    reference = pd.DataFrame(
        {"query_id": ["q"], "candidate_id": ["a"], "label": [1]}
    )
    predictions = pd.DataFrame(
        {"query_id": ["q"], "candidate_id": ["b"], "score": [1.0]}
    )
    with pytest.raises(IndependentEvaluationError, match="key sets differ"):
        independent_j_at_1(reference, predictions, pd.DataFrame({"query_id": ["q"]}))
