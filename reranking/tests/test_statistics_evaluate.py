from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from reranking.evaluate import (
    EvaluationError,
    compare_rankings,
    evaluate_rankings,
    join_predictions_with_labels,
    rank_candidates,
    validate_oracle_invariance,
    validate_rank_permutations,
)
from reranking.statistics import (
    candidate_classification_metrics,
    cluster_bootstrap_difference,
    holm_adjust,
    mcnemar_exact,
)


def test_strict_ranking_uses_candidate_id_for_exact_ties() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["q2", "q1", "q1", "q1"],
            "candidate_id": ["z", "b", "a", "c"],
            "score": [0.1, 0.8, 0.8, 0.9],
        }
    )
    ranked = rank_candidates(candidates)
    q1 = ranked.loc[ranked["query_id"].eq("q1")]
    assert q1["candidate_id"].tolist() == ["c", "a", "b"]
    assert q1["rank"].tolist() == [1, 2, 3]
    assert validate_rank_permutations(ranked)


def test_empty_and_no_positive_queries_count_as_zero() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["no_positive", "no_positive"],
            "candidate_id": ["a", "b"],
            "label": [0, 0],
            "score": [0.8, 0.2],
            "scene_id": ["scene", "scene"],
            "frame_id": ["frame", "frame"],
        }
    )
    universe = pd.DataFrame(
        {
            "query_id": ["no_positive", "empty"],
            "scene_id": ["scene", "empty_scene"],
            "frame_id": ["frame", "empty_frame"],
        }
    )
    result = evaluate_rankings(candidates, query_universe=universe)
    assert result["query_count"] == 2
    assert result["empty_query_count"] == 1
    assert result["no_positive_query_count"] == 2
    assert result["nonempty_no_positive_query_count"] == 1
    for metric in ("j_at_1", "j_at_5", "oracle", "mrr", "map", "ndcg"):
        assert result[metric] == 0.0
    empty = result["per_query"].set_index("query_id").loc["empty"]
    assert bool(empty["empty_query"])
    assert empty["candidate_count"] == 0


def test_ranking_and_candidate_metrics_are_computed_from_full_id_join() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["q", "q", "q"],
            "candidate_id": ["a", "b", "c"],
            "label": [0, 1, 1],
            "scene_id": ["s", "s", "s"],
            "frame_id": ["f", "f", "f"],
        }
    )
    predictions = pd.DataFrame(
        {
            "query_id": ["q", "q", "q"],
            "candidate_id": ["c", "a", "b"],
            "score": [0.7, 0.9, 0.8],
            "probability": [0.7, 0.1, 0.9],
        }
    )
    result = evaluate_rankings(
        candidates, predictions, probability_col="probability"
    )
    assert result["j_at_1"] == 0.0
    assert result["j_at_5"] == 1.0
    assert result["oracle"] == 1.0
    assert result["mrr"] == 0.5
    assert result["map"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert result["headroom"] == 1.0
    assert result["candidate_metrics"]["roc_auc"] == 1.0
    assert result["candidate_metrics"]["pr_auc"] == 1.0
    assert result["candidate_metrics"]["brier"] == pytest.approx(
        (0.3**2 + 0.1**2 + 0.1**2) / 3
    )


def test_candidate_calibration_handles_binary_probabilities() -> None:
    result = candidate_classification_metrics(
        [0, 1], [0.1, 0.9], probabilities=[0.1, 0.9], bins=2
    )
    assert result["roc_auc"] == 1.0
    assert result["pr_auc"] == 1.0
    assert result["brier"] == pytest.approx(0.01)
    assert result["ece"] == pytest.approx(0.1)


def test_scene_bootstrap_resamples_whole_scene_clusters() -> None:
    result = cluster_bootstrap_difference(
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        ["scene_a", "scene_a", "scene_b", "scene_b"],
        iterations=250,
        seed=7,
        return_distribution=True,
    )
    assert result["group_count"] == 2
    assert result["point_estimate"] == 0.0
    assert set(np.unique(result["distribution"])).issubset({-1.0, 0.0, 1.0})


def test_exact_mcnemar_zero_discordant_and_matched_odds_ratio() -> None:
    result = mcnemar_exact([0, 1, 1, 0], [0, 1, 1, 0])
    assert result["discordant"] == 0
    assert result["pvalue"] == 1.0
    assert result["matched_odds_ratio"] == 1.0
    assert result["cross_check_passed"] is True

    one_sided = mcnemar_exact([0, 0], [1, 1])
    assert math.isinf(one_sided["matched_odds_ratio"])


def test_holm_adjustment_is_monotonic_in_sorted_raw_p_values() -> None:
    raw = np.asarray([0.04, 0.01, 0.03, 0.4])
    adjusted = holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0.0)
    assert np.all(adjusted >= raw)
    mapping = holm_adjust({"a": 0.04, "b": 0.01})
    assert mapping["b"] <= mapping["a"]


def test_id_pool_rank_and_oracle_mismatches_are_rejected() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["q", "q"],
            "candidate_id": ["a", "b"],
            "candidate_identity_sha256": ["hash-a", "hash-b"],
            "label": [0, 1],
        }
    )
    missing = pd.DataFrame(
        {"query_id": ["q"], "candidate_id": ["a"], "score": [0.5]}
    )
    with pytest.raises(EvaluationError, match="candidate pool changed"):
        join_predictions_with_labels(candidates, missing)

    wrong_identity = pd.DataFrame(
        {
            "query_id": ["q", "q"],
            "candidate_id": ["a", "b"],
            "candidate_identity_sha256": ["wrong", "hash-b"],
            "score": [0.8, 0.2],
        }
    )
    with pytest.raises(EvaluationError, match="candidate identity changed"):
        join_predictions_with_labels(candidates, wrong_identity)

    invalid_ranks = pd.DataFrame(
        {
            "query_id": ["q", "q"],
            "candidate_id": ["a", "b"],
            "score": [0.8, 0.2],
            "rank": [1, 1],
        }
    )
    with pytest.raises(EvaluationError, match="1..N permutation"):
        validate_rank_permutations(invalid_ranks)

    before = pd.DataFrame({"query_id": ["q"], "oracle": [1]})
    after = pd.DataFrame({"query_id": ["q"], "oracle": [0]})
    with pytest.raises(EvaluationError, match="changed the frozen-pool oracle"):
        validate_oracle_invariance(before, after)


def test_compare_rankings_reports_switches_and_cluster_statistics() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
            "candidate_id": ["a", "b", "a", "b"],
            "label": [0, 1, 1, 0],
            "scene_id": ["s", "s", "s", "s"],
            "frame_id": ["f1", "f1", "f2", "f2"],
        }
    )
    reference = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
            "candidate_id": ["a", "b", "a", "b"],
            "score": [0.9, 0.1, 0.9, 0.1],
        }
    )
    challenger = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
            "candidate_id": ["a", "b", "a", "b"],
            "score": [0.1, 0.9, 0.1, 0.9],
        }
    )
    result = compare_rankings(
        candidates,
        reference,
        challenger,
        bootstrap_iterations=100,
        bootstrap_seed=3,
    )
    switch = result["switch_metrics"]
    assert switch["switch_count"] == 2
    assert switch["recovered"] == 1
    assert switch["harmful"] == 1
    assert switch["switch_precision"] == 0.5
    assert result["statistics"]["scene_bootstrap"]["group_count"] == 1
    assert result["statistics"]["frame_bootstrap"]["group_count"] == 2
    assert result["oracle_invariant"] is True
