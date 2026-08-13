from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from d1_reranking.contracts import forbidden_test_schema_columns
from tools.d1_reranking.apply_locked_test_gate import (
    _grid_points,
    _normalize_no_output_seed_votes,
    _validate_locked_selection,
)
from tools.d1_reranking.apply_selected_test_ranker import _decisions
from tools.d1_reranking.apply_selected_test_ranker import (
    _normalized_candidate_rank_contract,
)
from tools.d1_reranking.apply_selected_test_ranker import _raw_feature_columns
from tools.d1_reranking.apply_selected_test_ranker import _same_artifact_record


def _grid() -> dict[str, object]:
    return {
        "minimum_seed_votes": 2,
        "lambda_harm": [1.0, 2.0, 4.0],
        "utility_thresholds": [0.0, 0.05, 0.1],
        "score_margin_thresholds": [0.0, 0.05, 0.1],
        "reliability_thresholds": [0.5, 0.75],
        "stability_thresholds": [0.5, 0.75],
    }


def _gate() -> dict[str, object]:
    points = _grid_points(_grid())
    trials = [
        {
            "operating_point": point,
            "bootstrap_lower_bound": -0.1,
            "mean_delta": 0.0,
            "recovered": 0,
            "harmful": 0,
            "switch_count": 0,
            "switch_rate": 0.0,
        }
        for point in points
    ]
    trials[7]["bootstrap_lower_bound"] = 0.01
    trials[7]["mean_delta"] = 0.02
    return {
        "decision": "GO",
        "selection": {
            "status": "GO",
            "selected_operating_point": points[7],
            "trials": trials,
        },
    }


def test_gate_selection_replays_exact_grid_and_tie_break() -> None:
    gate = _gate()
    selected = _validate_locked_selection(gate, _grid())
    assert selected is not None
    assert asdict(selected) == gate["selection"]["selected_operating_point"]

    gate["selection"]["selected_operating_point"] = _grid_points(_grid())[8]
    with pytest.raises(RuntimeError, match="tie-break"):
        _validate_locked_selection(gate, _grid())


def test_test_schema_guard_covers_diagnostics_and_scene_semantics() -> None:
    forbidden = forbidden_test_schema_columns(
        (
            "safe_feature",
            "j_at_1",
            "target_object_name",
            "object_category",
            "category_label",
            "scene_graph_relation",
            "best-gt-index",
        )
    )
    assert forbidden == (
        "best-gt-index",
        "category_label",
        "j_at_1",
        "object_category",
        "scene_graph_relation",
        "target_object_name",
    )


def test_ranker_decisions_preserve_all_no_output_denominator() -> None:
    denominator = pd.DataFrame({"sample_id": ["s0", "s1"]})
    candidates = pd.DataFrame(
        columns=[
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ]
    )
    ensemble = pd.DataFrame(
        columns=[
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
            "score_seed_42",
            "score_seed_123",
            "score_seed_2026",
            "ensemble_score",
        ]
    )
    decisions = _decisions(denominator, candidates, ensemble)
    assert decisions["candidate_count"].tolist() == [0, 0]
    assert not decisions["challenger_exists"].any()
    assert decisions["selected_candidate_id"].isna().all()


def test_ranker_application_compares_normalized_artifact_identity() -> None:
    expected = {"path": "/tmp/candidates.parquet", "sha256": "a" * 64}
    _same_artifact_record(
        {**expected, "bytes": 123, "rows": 10},
        expected,
        name="candidate contract",
    )

    with pytest.raises(RuntimeError, match="does not bind"):
        _same_artifact_record(
            {**expected, "sha256": "b" * 64},
            expected,
            name="candidate contract",
        )


def test_ranker_application_reads_identity_only_from_candidate_contract() -> None:
    columns = _raw_feature_columns(
        ("native_score_raw", "p_center", "base_logit"),
        ("sample_id", "candidate_id", "native_rank", "native_score_raw", "p_center"),
    )
    assert columns == (
        "sample_id",
        "candidate_id",
        "native_rank",
        "native_score_raw",
        "p_center",
    )
    assert "candidate_identity_sha256" not in columns
    assert "candidate_geometry_sha256" not in columns


def test_test_candidate_projection_retains_identity_precision_contract() -> None:
    source = (
        Path(__file__).parents[2]
        / "tools"
        / "d1_reranking"
        / "apply_selected_test_ranker.py"
    ).read_text(encoding="utf-8")
    formal_source = (
        Path(__file__).parents[2]
        / "tools"
        / "d1_reranking"
        / "build_d1_formal_inputs.py"
    ).read_text(encoding="utf-8")
    assert '"candidate_identity_precision_contract"' in source
    assert '"candidate_identity_precision_contract"' in formal_source


def test_ranker_candidate_contract_normalizes_integer_width_only() -> None:
    values = {
        "sample_id": ["s0", "s0"],
        "candidate_id": ["b", "a"],
        "native_rank": [2, 1],
    }
    raw = pd.DataFrame(values).astype({"native_rank": "int64"})
    candidates = pd.DataFrame(values).astype({"native_rank": "int32"})
    assert _normalized_candidate_rank_contract(raw).equals(
        _normalized_candidate_rank_contract(candidates)
    )


def test_ranker_application_telemetry_uses_calibrated_features() -> None:
    source = (
        Path(__file__).parents[2]
        / "tools"
        / "d1_reranking"
        / "apply_selected_test_ranker.py"
    ).read_text(encoding="utf-8")
    assert "missing_feature_rate(\n        calibrated, model_columns" in source
    assert "missing_feature_rate(raw_features, model_columns)" not in source


def test_gate_normalizes_seed_votes_only_for_no_output_rows() -> None:
    frame = pd.DataFrame(
        {
            "candidate_count": [0, 5],
            "seed_challenger_votes": [float("nan"), 3.0],
        }
    )
    result = _normalize_no_output_seed_votes(frame)
    assert result["seed_challenger_votes"].tolist() == [0, 3]

    frame.loc[1, "seed_challenger_votes"] = float("nan")
    with pytest.raises(RuntimeError, match="candidate-bearing"):
        _normalize_no_output_seed_votes(frame)

    frame.loc[:, "seed_challenger_votes"] = [1.0, 3.0]
    with pytest.raises(RuntimeError, match="no-output"):
        _normalize_no_output_seed_votes(frame)

    source = (
        Path(__file__).parents[2]
        / "tools"
        / "d1_reranking"
        / "apply_locked_test_gate.py"
    ).read_text(encoding="utf-8")
    assert "inputs = _normalize_no_output_seed_votes(inputs)" in source
