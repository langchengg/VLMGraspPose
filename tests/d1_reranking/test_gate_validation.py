from __future__ import annotations

from dataclasses import asdict
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import d1_reranking.gate_validation as gate_validation
from d1_reranking.execution import artifact_record
from d1_reranking.io import atomic_pickle
from tools.d1_reranking import apply_locked_test_gate
from unified_reranking.gate import (
    ConservativeTransitionModel,
    OOFTransitionData,
    SAFE_GATE_FEATURE_COLUMNS,
    select_gate_operating_point,
)
from unified_reranking.hashing import atomic_json, canonical_sha256


def _features(rows: int, *, offset: float) -> dict[str, np.ndarray]:
    base = np.linspace(0.05 + offset, 0.75 + offset, rows)
    return {
        column: base + index * 0.003
        for index, column in enumerate(SAFE_GATE_FEATURE_COLUMNS)
    }


def _development_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = np.asarray([(0, 1), (1, 0), (0, 0), (1, 1)] * 2, dtype=int)
    train = pd.DataFrame(_features(len(outcomes), offset=0.0))
    train["sample_id"] = [f"train-{index}" for index in range(len(train))]
    train["scene_id"] = [f"train-scene-{index}" for index in range(len(train))]
    train["prediction_source"] = "train_oof"
    train["oof_fold"] = [0] * 4 + [1] * 4
    train["native_correct"] = outcomes[:, 0].astype(bool)
    train["challenger_correct"] = outcomes[:, 1].astype(bool)

    validation = pd.DataFrame(_features(len(outcomes), offset=0.02))
    validation["sample_id"] = [
        f"validation-{index}" for index in range(len(validation))
    ]
    validation["scene_id"] = [
        f"validation-scene-{index}" for index in range(len(validation))
    ]
    validation["prediction_source"] = "validation"
    validation["native_correct"] = outcomes[:, 0].astype(bool)
    validation["challenger_correct"] = outcomes[:, 1].astype(bool)
    validation["native_candidate_id"] = [
        f"native-{index}" for index in range(len(validation))
    ]
    validation["challenger_candidate_id"] = [
        f"challenger-{index}" for index in range(len(validation))
    ]
    validation["score_margin"] = 0.2
    validation["challenger_reliability"] = 0.9
    validation["perturbation_stability"] = 0.9
    validation["seed_challenger_votes"] = 3
    validation["candidate_id_unchanged"] = True
    validation["geometry_hash_unchanged"] = True
    validation["challenger_exists"] = True
    return train, validation


def _write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(gate_validation, "BOOTSTRAP_ITERATIONS", 24)
    train, validation = _development_frames()
    train_path = _write_parquet(tmp_path / "train_oof.parquet", train)
    validation_path = _write_parquet(tmp_path / "validation.parquet", validation)
    input_configuration = {
        "feature_columns": list(SAFE_GATE_FEATURE_COLUMNS),
        "train_prediction_source": "train_oof",
        "validation_prediction_source": "validation",
        "candidate_test_labels_read": False,
    }
    input_artifacts = {
        "train_oof": artifact_record(train_path),
        "validation": artifact_record(validation_path),
    }
    input_manifest: dict[str, object] = {
        "status": "COMPLETE",
        "configuration": input_configuration,
        "candidate_test_labels_read": False,
        "sources": {},
        "artifacts": input_artifacts,
    }
    input_manifest["content_sha256"] = canonical_sha256(input_manifest)
    input_manifest_path = tmp_path / "gate_inputs.json"
    atomic_json(input_manifest_path, input_manifest)
    grid: dict[str, object] = {
        "schema_version": 1,
        "status": "PLANNED",
        "lambda_harm": [1.0, 2.0, 4.0],
        "utility_thresholds": [0.0, 0.05, 0.1],
        "score_margin_thresholds": [0.0, 0.05, 0.1],
        "reliability_thresholds": [0.5, 0.75],
        "stability_thresholds": [0.5, 0.75],
        "minimum_seed_votes": 2,
        "bootstrap_iterations": 24,
        "bootstrap_seed": gate_validation.BOOTSTRAP_SEED,
        "candidate_test_labels_read": False,
        "sources": {"test_builder": artifact_record(Path(__file__))},
    }
    grid["content_sha256"] = canonical_sha256(grid)
    grid_path = tmp_path / "d1_gate_grid.json"
    atomic_json(grid_path, grid)

    features = tuple(SAFE_GATE_FEATURE_COLUMNS)
    model = ConservativeTransitionModel(seed=gate_validation.BOOTSTRAP_SEED).fit(
        OOFTransitionData(
            features=train.loc[:, features].to_numpy(float),
            feature_names=features,
            native_correct=train["native_correct"].to_numpy(),
            challenger_correct=train["challenger_correct"].to_numpy(),
            scene_ids=train["scene_id"].to_numpy(),
            oof_fold_ids=train["oof_fold"].to_numpy(),
            prediction_source="train_oof",
        )
    )
    probability_recover, probability_harm = model.predict_probabilities(
        validation.loc[:, features].to_numpy(float)
    )
    points = gate_validation._grid_points(grid)
    selection = select_gate_operating_point(
        probability_recover,
        probability_harm,
        gate_validation._evidence(validation),
        validation["native_correct"].to_numpy(),
        validation["challenger_correct"].to_numpy(),
        validation["scene_id"].to_numpy(),
        points,
        bootstrap_iterations=24,
        bootstrap_seed=gate_validation.BOOTSTRAP_SEED,
    )
    decisions, switches, native, challenger = gate_validation._validation_decisions(
        validation,
        probability_recover,
        probability_harm,
        selection.selected_operating_point,
    )
    trials = pd.DataFrame(
        [
            {
                **asdict(trial.operating_point),
                "bootstrap_lower_bound": trial.bootstrap_lower_bound,
                "mean_delta": trial.mean_delta,
                "recovered": trial.recovered,
                "harmful": trial.harmful,
                "switch_count": trial.switch_count,
                "switch_rate": trial.switch_rate,
            }
            for trial in selection.trials
        ]
    )
    metrics = gate_validation._validation_metrics(switches, native, challenger)
    model_path = atomic_pickle(model, tmp_path / "transition_model.pkl")
    decision_path = _write_parquet(tmp_path / "validation_decisions.parquet", decisions)
    trial_path = _write_parquet(tmp_path / "validation_trials.parquet", trials)
    metrics_path = tmp_path / "gate_operating_point.csv"
    metrics_path.write_text(
        pd.DataFrame([metrics]).to_csv(index=False), encoding="utf-8"
    )
    configuration = {
        "schema_version": 1,
        "route": "D1",
        "method": "R7",
        "feature_columns": list(features),
        "model_seed": gate_validation.BOOTSTRAP_SEED,
        "operating_point_count": 108,
        "candidate_test_labels_read": False,
    }
    sources = {
        "input_manifest": artifact_record(input_manifest_path),
        "train_oof": artifact_record(train_path),
        "validation": artifact_record(validation_path),
        "grid": artifact_record(grid_path),
    }
    gate: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "decision": selection.status,
        "source_signature_sha256": canonical_sha256(
            {"configuration": configuration, "sources": sources}
        ),
        "configuration": configuration,
        "transition_model": model.artifact(),
        "selection": selection.artifact(),
        "validation_metrics": metrics,
        "candidate_test_labels_read": False,
        "sources": sources,
        "artifacts": {
            "transition_model": artifact_record(model_path),
            "validation_decisions": artifact_record(decision_path),
            "validation_trials": artifact_record(trial_path),
            "gate_operating_point_table": artifact_record(metrics_path),
        },
    }
    gate["content_sha256"] = canonical_sha256(gate)
    return gate


def _refresh_source_signature(gate: dict[str, object]) -> None:
    gate["source_signature_sha256"] = canonical_sha256(
        {"configuration": gate["configuration"], "sources": gate["sources"]}
    )


def test_gate_validator_replays_complete_development_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _bundle(tmp_path, monkeypatch)

    replay = gate_validation.validate_gate_selection_semantics(gate)

    assert replay.model.artifact() == gate["transition_model"]
    selected = gate["selection"]["selected_operating_point"]  # type: ignore[index]
    assert (
        None
        if replay.selected_operating_point is None
        else asdict(replay.selected_operating_point)
    ) == selected


def test_gate_validator_rejects_tampered_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _bundle(tmp_path, monkeypatch)
    trial = gate["selection"]["trials"][0]  # type: ignore[index]
    trial["mean_delta"] = float(trial["mean_delta"]) + 0.25

    with pytest.raises(RuntimeError, match="bootstrap trials/selection"):
        gate_validation.validate_gate_selection_semantics(gate)


def test_gate_validator_rejects_tampered_bootstrap_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _bundle(tmp_path, monkeypatch)
    gate["selection"]["bootstrap_seed"] = -1  # type: ignore[index]

    with pytest.raises(RuntimeError, match="bootstrap trials/selection"):
        gate_validation.validate_gate_selection_semantics(gate)


def test_gate_validator_rejects_tampered_validation_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _bundle(tmp_path, monkeypatch)
    path = Path(gate["artifacts"]["validation_decisions"]["path"])  # type: ignore[index]
    decisions = pd.read_parquet(path)
    decisions.loc[0, "switch"] = not bool(decisions.loc[0, "switch"])
    decisions.to_parquet(path, index=False)
    gate["artifacts"]["validation_decisions"] = artifact_record(path)  # type: ignore[index]

    with pytest.raises(RuntimeError, match="Validation decisions differs"):
        gate_validation.validate_gate_selection_semantics(gate)


def test_gate_validator_rejects_tampered_top_level_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _bundle(tmp_path, monkeypatch)
    gate["decision"] = "GO" if gate["decision"] == "NO_GO_NATIVE" else "NO_GO_NATIVE"

    with pytest.raises(RuntimeError, match="decision differs"):
        gate_validation.validate_gate_selection_semantics(gate)


def test_gate_validator_rejects_tampered_transition_model_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _bundle(tmp_path, monkeypatch)
    path = Path(gate["artifacts"]["transition_model"]["path"])  # type: ignore[index]
    with path.open("rb") as stream:
        model = pickle.load(stream)
    model.seed += 1
    with path.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    gate["artifacts"]["transition_model"] = artifact_record(path)  # type: ignore[index]

    with pytest.raises(RuntimeError, match="artifact differs"):
        gate_validation.validate_gate_selection_semantics(gate)


def test_gate_validator_rejects_hash_rebound_validation_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _bundle(tmp_path, monkeypatch)
    path = Path(gate["sources"]["validation"]["path"])  # type: ignore[index]
    validation = pd.read_parquet(path)
    validation.loc[0, SAFE_GATE_FEATURE_COLUMNS[0]] += 5.0
    validation.to_parquet(path, index=False)
    gate["sources"]["validation"] = artifact_record(path)  # type: ignore[index]
    _refresh_source_signature(gate)

    with pytest.raises(RuntimeError, match="do not bind the input manifest"):
        gate_validation.validate_gate_selection_semantics(gate)


def test_test_application_replays_gate_before_opening_any_test_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []

    def load_manifest(path: Path, **_kwargs: object) -> dict[str, object]:
        opened.append(path)
        return {"status": "COMPLETE"}

    def stop_after_replay(_gate: object) -> None:
        raise RuntimeError("semantic replay sentinel")

    monkeypatch.setattr(apply_locked_test_gate, "load_content_manifest", load_manifest)
    monkeypatch.setattr(
        apply_locked_test_gate,
        "validate_gate_selection_semantics",
        stop_after_replay,
    )

    with pytest.raises(RuntimeError, match="semantic replay sentinel"):
        apply_locked_test_gate.run(tmp_path, resume=False)

    assert opened == [
        tmp_path / "07_validation" / "gate" / "d1" / "gate_selection.json"
    ]
