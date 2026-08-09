from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from unified_reranking.gate import (
    ConservativeTransitionModel,
    GateEvidence,
    GateOperatingPoint,
    OOFTransitionData,
    candidate_immutability_flags,
    gate_switch_mask,
    select_gate_operating_point,
)


def _oof_data(*, source: str = "train_oof") -> OOFTransitionData:
    # Each fixed fold contains recovery, harm, unchanged-wrong, and unchanged-correct.
    native_pattern = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    challenger_pattern = np.asarray([1, 0, 0, 1, 1, 0, 0, 1], dtype=int)
    native = np.tile(native_pattern, 3)
    challenger = np.tile(challenger_pattern, 3)
    index = np.arange(len(native), dtype=float)
    features = np.column_stack(
        [
            challenger - native,
            np.sin(index),
            np.full(len(index), 0.8),
        ]
    )
    folds = np.repeat(["fold0", "fold1", "fold2"], 8)
    scenes = np.asarray([f"scene-{fold}-{row}" for fold in range(3) for row in range(8)])
    return OOFTransitionData(
        features=features,
        feature_names=("ranker_margin", "perturbation_signal", "candidate_exists"),
        native_correct=native,
        challenger_correct=challenger,
        scene_ids=scenes,
        oof_fold_ids=folds,
        prediction_source=source,  # type: ignore[arg-type]
    )


def _all_pass_evidence(length: int) -> GateEvidence:
    return GateEvidence(
        score_margin=np.ones(length),
        challenger_reliability=np.ones(length),
        perturbation_stability=np.ones(length),
        seed_challenger_votes=np.full(length, 3),
        candidate_id_unchanged=np.ones(length, dtype=bool),
        geometry_hash_unchanged=np.ones(length, dtype=bool),
        challenger_exists=np.ones(length, dtype=bool),
    )


def test_oof_contract_rejects_in_sample_source_and_scene_leakage() -> None:
    with pytest.raises(ValueError, match="Train OOF"):
        _oof_data(source="train_in_sample").validated()
    data = _oof_data()
    leaking_folds = np.asarray(data.oof_fold_ids).copy()
    scenes = np.asarray(data.scene_ids).copy()
    scenes[8] = scenes[0]
    leaking = OOFTransitionData(
        features=data.features,
        feature_names=data.feature_names,
        native_correct=data.native_correct,
        challenger_correct=data.challenger_correct,
        scene_ids=scenes,
        oof_fold_ids=leaking_folds,
    )
    with pytest.raises(ValueError, match="crosses OOF folds"):
        leaking.validated()


def test_oof_contract_rejects_supervision_feature() -> None:
    data = _oof_data()
    invalid = OOFTransitionData(
        features=data.features,
        feature_names=("ranker_margin", "candidate_success", "candidate_exists"),
        native_correct=data.native_correct,
        challenger_correct=data.challenger_correct,
        scene_ids=data.scene_ids,
        oof_fold_ids=data.oof_fold_ids,
    )
    with pytest.raises(ValueError, match="supervision"):
        invalid.validated()


def test_transition_model_fits_two_fold_safe_calibrated_probabilities() -> None:
    data = _oof_data()
    model = ConservativeTransitionModel(seed=42).fit(data)
    recover, harm = model.predict_probabilities(data.features)
    assert recover.shape == harm.shape == (24,)
    assert np.all((recover >= 0.0) & (recover <= 1.0))
    assert np.all((harm >= 0.0) & (harm <= 1.0))
    artifact = model.artifact()
    assert artifact["prediction_source"] == "train_oof"
    assert artifact["fold_count"] == 3
    assert artifact["targets"] == ["recovered", "harmful"]


def test_gate_requires_every_deterministic_eligibility_condition() -> None:
    length = 9
    recover = np.full(length, 0.9)
    harm = np.full(length, 0.1)
    recover[1] = 0.39  # utility is below the strict threshold
    margin = np.ones(length)
    margin[2] = 0.2
    reliability = np.ones(length)
    reliability[3] = 0.7
    stability = np.ones(length)
    stability[4] = 0.79
    votes = np.full(length, 3)
    votes[5] = 1
    candidate_same = np.ones(length, dtype=bool)
    candidate_same[6] = False
    geometry_same = np.ones(length, dtype=bool)
    geometry_same[7] = False
    exists = np.ones(length, dtype=bool)
    exists[8] = False
    evidence = GateEvidence(
        score_margin=margin,
        challenger_reliability=reliability,
        perturbation_stability=stability,
        seed_challenger_votes=votes,
        candidate_id_unchanged=candidate_same,
        geometry_hash_unchanged=geometry_same,
        challenger_exists=exists,
    )
    point = GateOperatingPoint(1, 0.3, 0.2, 0.7, 0.8)
    np.testing.assert_array_equal(
        gate_switch_mask(recover, harm, evidence, point),
        [True, False, False, False, False, False, False, False, False],
    )


def test_candidate_and_geometry_hash_comparison() -> None:
    candidate_same, geometry_same = candidate_immutability_flags(
        ["a", "b"], ["a", "x"], ["h1", "h2"], ["h1", "h3"]
    )
    np.testing.assert_array_equal(candidate_same, [True, False])
    np.testing.assert_array_equal(geometry_same, [True, False])


def test_validation_selects_positive_lower_bound_or_native_no_go() -> None:
    length = 40
    recover = np.full(length, 0.95)
    harm = np.full(length, 0.01)
    evidence = _all_pass_evidence(length)
    native = np.zeros(length, dtype=int)
    challenger = np.ones(length, dtype=int)
    scenes = [f"scene-{index}" for index in range(length)]
    go_point = GateOperatingPoint(1, 0.2, 0.0, 0.5, 0.5)
    abstain_point = GateOperatingPoint(4, 0.99, 0.0, 0.5, 0.5)
    result = select_gate_operating_point(
        recover,
        harm,
        evidence,
        native,
        challenger,
        scenes,
        [abstain_point, go_point],
        bootstrap_iterations=500,
    )
    assert result.status == "GO"
    assert result.selected_operating_point == go_point
    assert result.artifact()["selected_operating_point"] == asdict(go_point)

    no_go = select_gate_operating_point(
        recover,
        harm,
        evidence,
        np.ones(length, dtype=int),
        np.zeros(length, dtype=int),
        scenes,
        [go_point],
        bootstrap_iterations=500,
    )
    assert no_go.status == "NO_GO_NATIVE"
    assert no_go.selected_operating_point is None
