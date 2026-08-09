import math

import pandas as pd
import pytest

from unified_reranking.metrics import (
    compare_selections,
    evaluate_order_only,
    select_order_only,
)


def test_no_output_is_retained_and_order_only_oracle_is_constant():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "candidate_id": ["a0", "a1", "b0", "b1"],
            "native_rank": [1, 2, 1, 2],
            "candidate_success": [0, 1, 1, 0],
            "native": [2.0, 1.0, 2.0, 1.0],
            "challenger": [1.0, 2.0, 2.0, 1.0],
        }
    )
    native_metrics, native = evaluate_order_only(["a", "b", "c"], frame, score_column="native")
    challenger_metrics, challenger = evaluate_order_only(
        ["a", "b", "c"], frame, score_column="challenger"
    )
    assert native_metrics["j_at_1_numerator"] == 1
    assert challenger_metrics["j_at_1_numerator"] == 2
    assert native_metrics["j_at_5"] == challenger_metrics["j_at_5"] == 2 / 3
    comparison = compare_selections(native, challenger, oracle_at_5=2 / 3)
    assert comparison["recovered"] == 1
    assert comparison["harmful"] == 0
    assert math.isclose(comparison["headroom_recovery_at_5"], 1.0)


def test_score_ties_preserve_native_rank():
    frame = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "candidate_id": ["later", "first"],
            "native_rank": [2, 1],
            "candidate_success": [1, 0],
            "score": [0.5, 0.5],
        }
    )
    _, decisions = evaluate_order_only(["s"], frame, score_column="score")
    assert decisions.iloc[0]["selected_candidate_id"] == "first"


def test_label_free_selection_preserves_no_output_and_stable_ties():
    frame = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "candidate_id": ["later", "first"],
            "native_rank": [2, 1],
            "score": [0.5, 0.5],
        }
    )
    result = select_order_only(["s", "empty"], frame, score_column="score")
    assert result.loc[0, "selected_candidate_id"] == "first"
    assert result.loc[0, "candidate_count"] == 2
    assert pd.isna(result.loc[1, "selected_candidate_id"])
    assert result.loc[1, "candidate_count"] == 0


def test_ndcg_counts_all_positive_candidates() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s"] * 3,
            "candidate_id": ["negative", "positive-1", "positive-2"],
            "native_rank": [1, 2, 3],
            "candidate_success": [0, 1, 1],
            "score": [3.0, 2.0, 1.0],
        }
    )
    metrics, _ = evaluate_order_only(["s"], frame, score_column="score")
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert metrics["ndcg_at_1"] == 0.0
    assert metrics["ndcg_at_5"] == pytest.approx(dcg / idcg)
