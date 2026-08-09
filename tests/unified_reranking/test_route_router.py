from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from unified_reranking.gate import OOFTransitionData
from unified_reranking.route_router import (
    CROGDefaultTransitionRouter,
    RouterEvidence,
    RouterOperatingPoint,
    route_decisions,
    select_router_operating_point,
)


def _router_training_data(challenger_offset: int = 0) -> OOFTransitionData:
    crog_pattern = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    alternative_patterns = [
        np.asarray([1, 0, 0, 1, 1, 0, 0, 1], dtype=int),
        np.asarray([1, 0, 1, 1, 0, 0, 0, 1], dtype=int),
    ]
    crog = np.tile(crog_pattern, 3)
    alternative = np.tile(alternative_patterns[challenger_offset], 3)
    index = np.arange(len(crog), dtype=float)
    features = np.column_stack(
        [alternative - crog, np.cos(index), np.ones(len(index))]
    )
    return OOFTransitionData(
        features=features,
        feature_names=("route_margin", "agreement", "candidate_exists"),
        native_correct=crog,
        challenger_correct=alternative,
        scene_ids=[f"scene-{fold}-{row}" for fold in range(3) for row in range(8)],
        oof_fold_ids=np.repeat(["f0", "f1", "f2"], 8),
    )


def _evidence(exists: np.ndarray) -> RouterEvidence:
    length = len(exists)
    return RouterEvidence(
        route_margin=np.ones(length),
        reliability=np.ones(length),
        perturbation_stability=np.ones(length),
        candidate_exists=exists,
    )


def test_router_trains_paired_oof_models_and_persists_contract() -> None:
    g1 = _router_training_data(0)
    c1 = _router_training_data(1)
    router = CROGDefaultTransitionRouter(seed=42).fit(g1, c1)
    probabilities = router.predict_probabilities(
        {"G1": g1.features, "C1": c1.features}
    )
    assert set(probabilities) == {"G1", "C1"}
    assert probabilities["G1"][0].shape == (24,)
    artifact = router.artifact()
    assert artifact["default_route"] == "CROG"
    assert artifact["prediction_source"] == "paired_train_oof"
    assert artifact["tie_break"] == ["G1", "C1"]


def test_router_requires_candidate_existence_feature() -> None:
    data = _router_training_data(0)
    missing = OOFTransitionData(
        features=data.features,
        feature_names=("route_margin", "agreement", "reliability"),
        native_correct=data.native_correct,
        challenger_correct=data.challenger_correct,
        scene_ids=data.scene_ids,
        oof_fold_ids=data.oof_fold_ids,
    )
    with pytest.raises(ValueError, match="candidate-existence"):
        CROGDefaultTransitionRouter().fit(missing, _router_training_data(1))


def test_route_decisions_are_crog_default_with_deterministic_tie_break() -> None:
    # row0 exact utility tie -> G1; row1 G1 absent -> C1; row2 C1 better;
    # row3 neither clears the utility threshold -> CROG.
    probabilities = {
        "G1": (np.asarray([0.8, 0.8, 0.6, 0.2]), np.asarray([0.1] * 4)),
        "C1": (np.asarray([0.8, 0.7, 0.9, 0.2]), np.asarray([0.1] * 4)),
    }
    evidence = {
        "G1": _evidence(np.asarray([1, 0, 1, 1], dtype=bool)),
        "C1": _evidence(np.ones(4, dtype=bool)),
    }
    point = RouterOperatingPoint(1, 0.3, 0.0, 0.5, 0.5)
    np.testing.assert_array_equal(
        route_decisions(probabilities, evidence, point),
        ["G1", "C1", "C1", "CROG"],
    )
    reverse = route_decisions(
        probabilities, evidence, point, tie_break=("C1", "G1")
    )
    assert reverse[0] == "C1"


def test_router_validation_selects_positive_bound_or_crog_no_go() -> None:
    length = 40
    probabilities = {
        "G1": (np.full(length, 0.95), np.full(length, 0.01)),
        "C1": (np.full(length, 0.2), np.full(length, 0.1)),
    }
    evidence = {
        "G1": _evidence(np.ones(length, dtype=bool)),
        "C1": _evidence(np.ones(length, dtype=bool)),
    }
    scenes = [f"scene-{index}" for index in range(length)]
    go_point = RouterOperatingPoint(1, 0.3, 0.0, 0.5, 0.5)
    abstain_point = RouterOperatingPoint(4, 0.99, 0.0, 0.5, 0.5)
    result = select_router_operating_point(
        probabilities,
        evidence,
        {
            "CROG": np.zeros(length, dtype=int),
            "G1": np.ones(length, dtype=int),
            "C1": np.zeros(length, dtype=int),
        },
        scenes,
        [abstain_point, go_point],
        bootstrap_iterations=500,
    )
    assert result.status == "GO"
    assert result.selected_operating_point == go_point
    assert result.artifact()["selected_operating_point"] == asdict(go_point)

    no_go = select_router_operating_point(
        probabilities,
        evidence,
        {
            "CROG": np.ones(length, dtype=int),
            "G1": np.zeros(length, dtype=int),
            "C1": np.zeros(length, dtype=int),
        },
        scenes,
        [go_point],
        bootstrap_iterations=500,
    )
    assert no_go.status == "NO_GO_CROG"
    assert no_go.selected_operating_point is None
