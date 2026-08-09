from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reranking.models.gates import (
    ConservativeGate,
    GateError,
    LearnedLogisticGate,
    MarginGate,
    attach_switch_outcomes,
    build_switch_proposals,
)
from reranking.splits import SplitError, build_stratified_group_folds


def _candidate_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # Every scene/frame group contains one positive and one no-positive query,
    # making the grouped stratification target feasible in all five folds.
    for group in range(10):
        for query_kind in ("positive", "no_positive"):
            query_id = f"g{group:02d}-{query_kind}"
            for candidate in range(2):
                rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": f"c{candidate}",
                        "scene_id": f"scene-{group:02d}",
                        "frame_id": f"frame-{group:02d}",
                        "label": int(query_kind == "positive" and candidate == 1),
                    }
                )
    return pd.DataFrame(rows)


def test_five_fold_query_split_expands_to_candidates_without_leakage() -> None:
    candidates = _candidate_rows()
    plan = build_stratified_group_folds(candidates, random_state=17)

    assert len(plan.folds) == 5
    assert len(plan.query_assignments) == candidates["query_id"].nunique()
    assert len(plan.candidate_assignments) == len(candidates)
    assert plan.audit["all_folds_leakage_free"] is True
    assert plan.audit["candidate_validation_assignment_min"] == 1
    assert plan.audit["candidate_validation_assignment_max"] == 1
    held_rows = np.zeros(len(candidates), dtype=int)
    for fold, fold_audit in zip(plan.folds, plan.audit["folds"]):
        held_rows[fold.validation_indices] += 1
        assert not set(fold.train_query_ids) & set(fold.validation_query_ids)
        assert not set(fold.train_group_hashes) & set(fold.validation_group_hashes)
        assert not set(fold.train_candidate_keys) & set(fold.validation_candidate_keys)
        assert fold_audit["intersections"] == {
            "query_ids": [],
            "group_hashes": [],
            "candidate_keys": [],
        }
        # Candidate rows of one query inherit exactly its query-level fold.
        local = plan.candidate_assignments.loc[
            plan.candidate_assignments["query_id"].isin(fold.validation_query_ids)
        ]
        assert set(local["fold"]) == {fold.fold}
    assert np.array_equal(held_rows, np.ones(len(candidates), dtype=int))


def test_split_is_deterministic_under_candidate_row_permutation() -> None:
    candidates = _candidate_rows()
    first = build_stratified_group_folds(candidates, random_state=9)
    permuted = candidates.sample(frac=1.0, random_state=3).reset_index(drop=True)
    second = build_stratified_group_folds(permuted, random_state=9)
    first_map = first.query_assignments.set_index("query_id")["fold"].to_dict()
    second_map = second.query_assignments.set_index("query_id")["fold"].to_dict()
    assert first_map == second_map


def test_split_rejects_duplicate_candidates_and_queries_spanning_groups() -> None:
    candidates = _candidate_rows()
    duplicate = pd.concat([candidates, candidates.iloc[[0]]], ignore_index=True)
    with pytest.raises(SplitError, match="duplicate"):
        build_stratified_group_folds(duplicate)
    inconsistent = candidates.copy()
    same_query = inconsistent["query_id"].eq(inconsistent.iloc[0]["query_id"])
    inconsistent.loc[inconsistent.index[same_query][0], "frame_id"] = "different"
    with pytest.raises(SplitError, match="multiple frame_id"):
        build_stratified_group_folds(inconsistent)


def _candidate_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["q", "q", "q"],
            "candidate_id": ["c2", "c0", "c1"],
            "baseline_score": [0.8, 0.8, 0.2],
            "reranker_score": [0.3, 0.7, 0.7],
            "risk": [0.0, 0.0, 0.0],
            "reliability": [0.5, 0.8, 0.9],
            "label": [0, 0, 1],
        }
    )


def test_switch_proposal_ties_are_deterministic_and_labels_join_later() -> None:
    candidates = _candidate_scores()
    proposals = build_switch_proposals(candidates.drop(columns="label"))
    assert proposals.iloc[0]["baseline_candidate_id"] == "c0"
    assert proposals.iloc[0]["challenger_candidate_id"] == "c0"
    assert proposals.iloc[0]["challenger_margin"] == 0.0
    assert "switch_outcome" not in proposals.columns
    labelled = attach_switch_outcomes(proposals, candidates)
    assert labelled.iloc[0]["switch_outcome"] == 0


def _gate_examples() -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    outcomes: list[int] = []
    for index in range(20):
        beneficial = index >= 10
        rows.append(
            {
                "query_id": f"validation-{index:02d}",
                "split": "validation",
                "baseline_candidate_id": "old",
                "challenger_candidate_id": "new",
                "challenger_margin": 0.8 if beneficial else 0.05,
                "baseline_score_delta": 0.2 if beneficial else -0.5,
                "baseline_rank_delta": -1.0 if beneficial else -3.0,
                "mask_support_delta": 0.4 if beneficial else -0.2,
                "width_compatibility_delta": 0.3 if beneficial else -0.1,
                "depth_contact_delta": 0.2 if beneficial else -0.2,
                "collision_proxy_delta": 0.3 if beneficial else -0.3,
                "challenger_risk": 0.0,
                "challenger_reliability": 0.9 if beneficial else 0.2,
                "model_disagreement": 1.0,
                "seed_agreement": 1.0 if beneficial else 1.0 / 3.0,
                "score_perturbation_stability": float(beneficial),
                "baseline_score": 0.7,
            }
        )
        outcomes.append(1 if beneficial else (-1 if index < 5 else 0))
    return pd.DataFrame(rows), np.asarray(outcomes)


def test_margin_gate_fits_validation_only_and_ties_at_threshold_are_stable() -> None:
    examples = pd.DataFrame(
        {
            "query_id": ["a", "b", "c"],
            "split": ["validation"] * 3,
            "baseline_candidate_id": ["old"] * 3,
            "challenger_candidate_id": ["new"] * 3,
            "challenger_margin": [0.1, 0.2, 0.3],
        }
    )
    gate = MarginGate(harmful_rate_limit=0.0).fit(
        examples, [-1, 1, 1], fit_scope="validation"
    )
    assert gate.metadata["threshold"] == pytest.approx(0.2)
    assert gate.predict_switch(examples).tolist() == [False, True, True]
    assert gate.metadata["fit_query_ids"] == ["a", "b", "c"]

    forbidden = examples.assign(split="test")
    with pytest.raises(GateError, match="test/holdout"):
        MarginGate().fit(forbidden, [-1, 1, 1], fit_scope="validation")


def test_learned_logistic_gate_has_validation_only_provenance() -> None:
    examples, outcomes = _gate_examples()
    gate = LearnedLogisticGate(harmful_rate_limit=0.0).fit(
        examples, outcomes, fit_scope="validation"
    )
    probability = gate.predict_gain_probability(examples)
    assert np.isfinite(probability).all()
    assert probability[10:].min() > probability[:10].max()
    assert gate.metadata["fit_scope"] == "validation"
    assert gate.metadata["fit_query_ids"] == examples["query_id"].tolist()
    assert gate.metadata["neutral_ignored_count"] == 5
    assert gate.metadata["decisive_fit_count"] == 15

    oof = examples.drop(columns="split")
    with pytest.raises(GateError, match="oof_fold"):
        LearnedLogisticGate().fit(oof, outcomes, fit_scope="oof")
    with pytest.raises(GateError, match="validation.*oof"):
        LearnedLogisticGate().fit(examples, outcomes, fit_scope="test")


def test_conservative_gate_switches_only_for_benefit_and_all_risk_constraints() -> None:
    examples, outcomes = _gate_examples()
    feature_columns = (
        "challenger_margin",
        "baseline_score_delta",
        "challenger_reliability",
    )
    gate = ConservativeGate(
        feature_columns,
        min_margin=0.2,
        max_risk=0.1,
        min_reliability=0.7,
        min_baseline_score=0.5,
        minimum_gain_probability=0.5,
        harmful_rate_limit=0.0,
    ).fit(examples, outcomes, fit_scope="validation")

    deployment = pd.DataFrame(
        {
            "query_id": ["safe", "risky", "not_beneficial", "same"],
            "baseline_candidate_id": ["old", "old", "old", "old"],
            "challenger_candidate_id": ["new", "new", "new", "old"],
            "challenger_margin": [0.8, 0.8, 0.05, 0.8],
            "baseline_score_delta": [0.2, 0.2, -0.5, 0.2],
            "challenger_risk": [0.0, 0.9, 0.0, 0.0],
            "challenger_reliability": [0.9, 0.9, 0.2, 0.9],
            "baseline_score": [0.7, 0.7, 0.7, 0.7],
        }
    )
    decided = gate.apply(deployment)
    assert decided["switch_applied"].tolist() == [True, False, False, False]
    assert decided["selected_candidate_id"].tolist() == ["new", "old", "old", "old"]
    assert decided["fallback_reason"].tolist() == [
        "switch",
        "risk_above_limit",
        "benefit_below_threshold",
        "same_candidate",
    ]
    assert gate.metadata["selection_rule"].startswith("benefit AND")
