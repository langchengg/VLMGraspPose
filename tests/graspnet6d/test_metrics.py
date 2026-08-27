"""Deterministic raw-row unit fixtures only; not formal experiment results."""

from __future__ import annotations

import pandas as pd
import pytest

from graspnet6d.metrics import (
    assert_metrics_match_raw_predictions,
    derive_graded_relevance,
    evaluate_target_rankings,
    mcnemar_exact,
    paired_intervention_outcomes,
    paired_metric_deltas,
    scene_cluster_bootstrap,
)


def _raw_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = pd.DataFrame(
        {
            "group_id": ["g1", "g2", "g3"],
            "scene_id": ["scene-a", "scene-a", "scene-b"],
        }
    )
    rows = pd.DataFrame(
        {
            "group_id": ["g1", "g1", "g1", "g2", "g2"],
            "scene_id": ["scene-a"] * 3 + ["scene-a"] * 2,
            "candidate_id": ["wrong", "good", "hard", "collision", "good-2"],
            "native_rank": [1, 2, 3, 1, 2],
            "native_score": [0.9, 0.8, 0.7, 0.9, 0.8],
            "rerank_score": [0.1, 0.95, 0.7, 0.8, 0.9],
            "target_object_id": [1, 1, 1, 2, 2],
            "associated_object_id": [9, 1, 1, 2, 2],
            "collision": [False, False, False, True, False],
            "pose_valid": [True, True, True, True, True],
            "friction_required": [0.2, 0.4, 1.2, 0.2, 0.8],
        }
    )
    return rows, universe


def test_wrong_target_is_relevance_zero() -> None:
    rows, _ = _raw_fixture()
    relevance = derive_graded_relevance(rows)
    # Excellent friction on the distractor and a colliding target both remain zero.
    assert relevance.tolist() == [0, 5, 1, 0, 3]


def test_target_metrics_are_recomputed_from_raw_rows_with_empty_denominator() -> None:
    rows, universe = _raw_fixture()
    metrics, per_group = evaluate_target_rankings(
        rows, universe, score_column="rerank_score", max_k=3
    )
    assert metrics["group_count"] == 3
    assert metrics["non_empty_pool_rate"] == pytest.approx(2 / 3)
    assert metrics["target_p_at_1_mu_0.4"] == pytest.approx(1 / 3)
    assert metrics["target_p_at_1_mu_1.2"] == pytest.approx(2 / 3)
    # g1 precision@2=1, g2=.5, g3=0, averaged over the explicit universe.
    assert metrics["target_precision_at_2_mu_1.2"] == pytest.approx(0.5)
    # At max_k=3: AP(g1)=8/9, AP(g2)=11/18, AP(empty)=0 -> 1/2.
    assert metrics["target_graspnet_style_ap_mu_1.2"] == pytest.approx(0.5)
    assert metrics["oracle_at_1"] == 0.0
    assert metrics["oracle_at_5"] == pytest.approx(2 / 3)
    assert metrics["candidate_absence_rate"] == pytest.approx(1 / 3)
    assert per_group.loc[per_group.group_id.eq("g3"), "candidate_count"].item() == 0


def test_report_metrics_match_raw_predictions() -> None:
    rows, universe = _raw_fixture()
    metrics, _ = evaluate_target_rankings(
        rows, universe, score_column="rerank_score", max_k=3
    )
    recomputed = assert_metrics_match_raw_predictions(
        metrics,
        rows,
        universe,
        score_column="rerank_score",
        max_k=3,
    )
    assert recomputed == metrics
    corrupted = dict(metrics)
    corrupted["target_p_at_1_mu_1.2"] = 0.0
    with pytest.raises(ValueError, match="recomputed"):
        assert_metrics_match_raw_predictions(
            corrupted,
            rows,
            universe,
            score_column="rerank_score",
            max_k=3,
        )


def test_paired_outcomes_exact_mcnemar_and_scene_bootstrap() -> None:
    rows, universe = _raw_fixture()
    _, native = evaluate_target_rankings(
        rows, universe, score_column="native_score", max_k=3
    )
    _, reranked = evaluate_target_rankings(
        rows, universe, score_column="rerank_score", max_k=3
    )
    summary, paired = paired_intervention_outcomes(native, reranked)
    assert summary["recovered"] == 2
    assert summary["harmful"] == 0
    assert summary["unchanged_failure"] == 1
    assert summary["net_recovered"] == 2
    assert paired["net_recovered"].sum() == 2

    exact = mcnemar_exact([0, 0, 0, 1], [1, 1, 1, 1])
    assert exact["recovered"] == 3
    assert exact["harmful"] == 0
    assert exact["pvalue"] == pytest.approx(0.25)

    deltas = paired_metric_deltas(native, reranked)
    bootstrap_a = scene_cluster_bootstrap(
        deltas,
        delta_columns=["delta_p_at_1", "delta_ap", "delta_mrr", "net_recovered_rate"],
        iterations=200,
        seed=17,
    )
    bootstrap_b = scene_cluster_bootstrap(
        deltas,
        delta_columns=["delta_p_at_1", "delta_ap", "delta_mrr", "net_recovered_rate"],
        iterations=200,
        seed=17,
    )
    pd.testing.assert_frame_equal(bootstrap_a, bootstrap_b)
    assert bootstrap_a["scene_count"].eq(2).all()
    assert bootstrap_a["resampling_unit"].eq("scene").all()

