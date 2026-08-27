"""Deterministic unit fixtures only; no test metric is a paper result."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graspnet6d.ranker import (
    FORMAL_SEEDS,
    LABEL_GAIN,
    GradedLightGBMLambdaRank,
    ValidationData,
    contiguous_group_sizes,
    fit_ranker,
    graded_relevance,
    rank_candidate_table,
    predict_scores,
    validate_group_sizes,
)
from unified_reranking.models.lightgbm_ranker import LightGBMLambdaRank


def _training_fixture() -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(24, 4))
    relevance = np.tile(np.asarray([6, 4, 2, 0], dtype=int), 6)
    # Give the tiny unit model a real, learnable signal.
    features[:, 0] = relevance + rng.normal(scale=0.05, size=len(relevance))
    return features, relevance, [4] * 6


def test_group_contract_rejects_noncontiguous_queries_and_bad_sizes() -> None:
    assert contiguous_group_sizes(["q1", "q1", "q2", "q2", "q2"], length=5) == [2, 3]
    with pytest.raises(ValueError, match="non-contiguous"):
        contiguous_group_sizes(["q1", "q2", "q1"], length=3)
    assert validate_group_sizes([2, 3], length=5) == [2, 3]
    with pytest.raises(ValueError, match="sum"):
        validate_group_sizes([2, 2], length=5)


def test_graded_relevance_contract_is_exactly_zero_through_six() -> None:
    np.testing.assert_array_equal(graded_relevance([0, 1, 6], length=3), [0, 1, 6])
    for bad in ([-1, 0], [0, 7], [0, 1.5], [0, np.nan]):
        with pytest.raises(ValueError):
            graded_relevance(bad, length=2)


@pytest.mark.parametrize("seed", FORMAL_SEEDS)
def test_formal_seeds_lock_cpu_determinism_and_graded_gain(seed: int) -> None:
    model = GradedLightGBMLambdaRank(seed=seed, label_gain=[0, 1], device_type="gpu")
    assert model.parameters["label_gain"] == list(LABEL_GAIN)
    assert model.parameters["device_type"] == "cpu"
    assert model.parameters["deterministic"] is True
    assert model.parameters["n_jobs"] == 1
    assert model.parameters["random_state"] == seed


def test_lightgbm_fit_uses_validation_early_stopping_and_preserves_membership() -> None:
    features, relevance, groups = _training_fixture()
    model = GradedLightGBMLambdaRank(
        seed=FORMAL_SEEDS[0],
        n_estimators=30,
        num_leaves=7,
        min_child_samples=1,
        early_stopping_rounds=5,
    ).fit_grouped(
        features,
        relevance,
        groups,
        validation=ValidationData(features, relevance, groups),
    )
    candidate_ids = [f"candidate-{index:03d}" for index in range(len(features))]
    prediction = model.predict_table(features, candidate_ids)
    assert prediction["candidate_id"].tolist() == candidate_ids
    assert len(prediction) == len(features)
    assert np.isfinite(prediction["rerank_score"]).all()
    artifact = model.artifact()
    assert artifact["label_contract"] == "integer relevance 0..6"
    assert artifact["row_reordering"] is False
    assert artifact["best_iteration"] is not None

    frame = pd.DataFrame(features, columns=[f"f{index}" for index in range(4)])
    frame.insert(0, "candidate_id", candidate_ids)
    ranked = rank_candidate_table(
        model, frame, feature_columns=[f"f{index}" for index in range(4)]
    )
    assert ranked["candidate_id"].tolist() == candidate_ids
    assert len(ranked) == len(frame)
    with pytest.raises(ValueError, match="unique"):
        model.predict(features[:2], candidate_ids=["same", "same"])


def test_old_4d_binary_adapter_contract_is_untouched() -> None:
    old = LightGBMLambdaRank(seed=42)
    assert old.parameters["label_gain"] == [0, 1]
    with pytest.raises(ValueError, match="binary"):
        old.fit(np.ones((2, 1)), [0, 2], ["q", "q"])


def test_functional_ranker_entry_points_validate_complete_groups() -> None:
    features, relevance, groups = _training_fixture()
    model = fit_ranker(
        features,
        relevance,
        groups,
        ValidationData(features, relevance, groups),
        {
            "seed": FORMAL_SEEDS[0],
            "n_estimators": 5,
            "min_child_samples": 1,
            "early_stopping_rounds": 2,
        },
    )
    ids = [f"id-{index}" for index in range(len(features))]
    scores = predict_scores(model, features, groups, candidate_ids=ids)
    assert scores.shape == (len(features),)
    with pytest.raises(ValueError, match="sum"):
        predict_scores(model, features, [len(features) - 1], candidate_ids=ids)
