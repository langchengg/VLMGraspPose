from __future__ import annotations

import pandas as pd
import pytest

from d1_reranking.gate_inputs import (
    build_gate_input_frame,
    build_label_free_test_gate_input_frame,
)
from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS


def test_gate_input_builder_preserves_denominator_and_safe_schema() -> None:
    paired = pd.DataFrame({"sample_id": ["s0", "s1"], "scene_id": ["a", "b"]})
    native_predictions = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "native_rank": [1, 2],
            "candidate_geometry_sha256": ["ga", "gb"],
            "candidate_identity_sha256": ["ia", "ib"],
        }
    )
    native_decisions = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "selected_candidate_id": ["a", None],
            "selected_correct": [False, False],
        }
    )
    challenger_predictions = native_predictions.copy()
    challenger_predictions["ensemble_score"] = [0.1, 0.9]
    for seed in (42, 123, 2026):
        challenger_predictions[f"score_seed_{seed}"] = [0.1, 0.9]
    challenger_decisions = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "selected_candidate_id": ["b", None],
            "selected_correct": [True, False],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "calibrated_native_probability": [0.4, 0.7],
            "native_score_raw": [0.8, 0.2],
            "overall_feature_reliability": [0.8, 0.9],
            "peak_retention_rate": [0.7, 0.8],
            "perturbed_valid_fraction": [0.6, 0.9],
            "mask_reliability": [0.75, 0.85],
        }
    )
    output = build_gate_input_frame(
        paired=paired,
        native_predictions=native_predictions,
        native_decisions=native_decisions,
        challenger_predictions=challenger_predictions,
        challenger_decisions=challenger_decisions,
        candidate_features=features,
        candidates=native_predictions[
            [
                "sample_id",
                "candidate_id",
                "native_rank",
                "candidate_identity_sha256",
                "candidate_geometry_sha256",
            ]
        ],
        prediction_source="train_oof",
        folds=pd.DataFrame({"sample_id": ["s0", "s1"], "fold": [0, 1]}),
    )
    assert len(output) == 2
    assert output.loc[0, "challenger_candidate_id"] == "b"
    assert output.loc[0, "seed_challenger_votes"] == 3
    assert output.loc[1, "challenger_exists"] == 0
    assert set(SAFE_GATE_FEATURE_COLUMNS).issubset(output.columns)


def test_gate_input_builder_normalizes_native_rank_integer_width() -> None:
    paired = pd.DataFrame({"sample_id": ["s0"], "scene_id": ["a"]})
    predictions = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "candidate_id": ["a"],
            "native_rank": pd.Series([1], dtype="int64"),
            "candidate_geometry_sha256": ["ga"],
            "candidate_identity_sha256": ["ia"],
            "ensemble_score": [0.1],
            "score_seed_42": [0.1],
            "score_seed_123": [0.1],
            "score_seed_2026": [0.1],
        }
    )
    decisions = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "selected_candidate_id": ["a"],
            "selected_correct": [True],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "candidate_id": ["a"],
            "calibrated_native_probability": [0.4],
            "native_score_raw": [0.8],
            "overall_feature_reliability": [0.8],
            "peak_retention_rate": [0.7],
            "perturbed_valid_fraction": [0.6],
            "mask_reliability": [0.75],
        }
    )
    candidates = predictions[
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ]
    ].copy()
    candidates["native_rank"] = candidates["native_rank"].astype("int32")

    output = build_gate_input_frame(
        paired=paired,
        native_predictions=predictions,
        native_decisions=decisions,
        challenger_predictions=predictions,
        challenger_decisions=decisions,
        candidate_features=features,
        candidates=candidates,
        prediction_source="validation",
    )

    assert output.loc[0, "native_candidate_id"] == "a"


def test_label_free_test_gate_builder_preserves_no_output_and_geometry() -> None:
    paired = pd.DataFrame({"sample_id": ["s0", "s1"], "scene_id": ["a", "b"]})
    decisions = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "native_candidate_id": ["a", None],
            "native_identity_sha256": ["ia", None],
            "native_geometry_sha256": ["ga", None],
            "selected_candidate_id": ["b", None],
            "selected_identity_sha256": ["ib", None],
            "selected_geometry_sha256": ["gb", None],
            "ensemble_score_margin": [0.4, 0.0],
            "seed_challenger_votes": [3, 0],
            "candidate_count": [2, 0],
            "challenger_exists": [True, False],
        }
    )
    candidates = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "native_rank": [1, 2],
            "candidate_identity_sha256": ["ia", "ib"],
            "candidate_geometry_sha256": ["ga", "gb"],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "calibrated_native_probability": [0.4, 0.7],
            "native_score_raw": [0.8, 0.2],
            "overall_feature_reliability": [0.8, 0.9],
            "peak_retention_rate": [0.7, 0.8],
            "perturbed_valid_fraction": [0.6, 0.9],
            "mask_reliability": [0.75, 0.85],
        }
    )
    output = build_label_free_test_gate_input_frame(
        paired=paired,
        ranker_decisions=decisions,
        candidate_features=features,
        candidates=candidates,
    )
    assert len(output) == 2
    assert output.loc[0, "challenger_candidate_id"] == "b"
    assert bool(output.loc[0, "geometry_hash_unchanged"])
    no_output = output.loc[1]
    assert no_output["candidate_count"] == 0
    assert no_output["native_candidate_id"] == ""
    assert no_output["challenger_candidate_id"] == ""
    assert not bool(no_output["challenger_exists"])
    assert set(SAFE_GATE_FEATURE_COLUMNS).issubset(output.columns)


def test_label_free_test_gate_builder_rejects_outcomes() -> None:
    decisions = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "native_candidate_id": ["a"],
            "native_identity_sha256": ["ia"],
            "native_geometry_sha256": ["ga"],
            "selected_candidate_id": ["b"],
            "selected_identity_sha256": ["ib"],
            "selected_geometry_sha256": ["gb"],
            "ensemble_score_margin": [0.4],
            "seed_challenger_votes": [3],
            "candidate_count": [2],
            "challenger_exists": [True],
            "selected_correct": [True],
        }
    )
    with pytest.raises(PermissionError, match="contain outcomes"):
        build_label_free_test_gate_input_frame(
            paired=pd.DataFrame({"sample_id": ["s0"]}),
            ranker_decisions=decisions,
            candidate_features=pd.DataFrame(),
            candidates=pd.DataFrame(),
        )


def test_label_free_test_gate_builder_rejects_missing_feature_row() -> None:
    decisions = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "native_candidate_id": ["a"],
            "native_identity_sha256": ["ia"],
            "native_geometry_sha256": ["ga"],
            "selected_candidate_id": ["b"],
            "selected_identity_sha256": ["ib"],
            "selected_geometry_sha256": ["gb"],
            "ensemble_score_margin": [0.4],
            "seed_challenger_votes": [3],
            "candidate_count": [2],
            "challenger_exists": [True],
        }
    )
    candidates = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "native_rank": [1, 2],
            "candidate_identity_sha256": ["ia", "ib"],
            "candidate_geometry_sha256": ["ga", "gb"],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "candidate_id": ["a"],
            "calibrated_native_probability": [0.4],
            "native_score_raw": [0.8],
            "overall_feature_reliability": [0.8],
            "peak_retention_rate": [0.7],
            "perturbed_valid_fraction": [0.6],
            "mask_reliability": [0.75],
        }
    )
    with pytest.raises(RuntimeError, match="feature/candidate membership differs"):
        build_label_free_test_gate_input_frame(
            paired=pd.DataFrame({"sample_id": ["s0"]}),
            ranker_decisions=decisions,
            candidate_features=features,
            candidates=candidates,
        )
