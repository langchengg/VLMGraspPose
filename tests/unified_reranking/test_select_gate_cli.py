from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.unified_reranking.select_gate import (
    GateColumns,
    execute_gate_selection,
    load_gate_grid,
    run_gate_selection,
)
from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS


FEATURES = SAFE_GATE_FEATURE_COLUMNS


def _gate_features(values: np.ndarray | float) -> dict[str, np.ndarray | float]:
    return {name: values for name in FEATURES}


def _write_gate_inputs(root: Path, *, validation_native: int, validation_challenger: int) -> tuple[Path, Path, Path]:
    native_pattern = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    challenger_pattern = np.asarray([1, 0, 0, 1, 1, 0, 0, 1], dtype=int)
    native = np.tile(native_pattern, 3)
    challenger = np.tile(challenger_pattern, 3)
    index = np.arange(len(native), dtype=float)
    oof = pd.DataFrame(
        {
            "sample_id": [f"train-{i}" for i in range(len(native))],
            "scene_id": [f"scene-{fold}-{row}" for fold in range(3) for row in range(8)],
            "prediction_source": "train_oof",
            "oof_fold": np.repeat(["f0", "f1", "f2"], 8),
            "native_correct": native,
            "challenger_correct": challenger,
            **_gate_features(np.sin(index)),
        }
    )
    length = 40
    validation = pd.DataFrame(
        {
            "sample_id": [f"validation-{i}" for i in range(length)],
            "scene_id": [f"validation-scene-{i}" for i in range(length)],
            "prediction_source": "validation",
            "native_correct": validation_native,
            "challenger_correct": validation_challenger,
            "native_candidate_id": [f"native-{i}" for i in range(length)],
            "challenger_candidate_id": [f"challenger-{i}" for i in range(length)],
            "score_margin": 1.0,
            "challenger_reliability": 1.0,
            "perturbation_stability": 1.0,
            "seed_challenger_votes": 3,
            "candidate_id_unchanged": True,
            "geometry_hash_unchanged": True,
            "challenger_exists": True,
            **_gate_features(0.5),
        }
    )
    oof_path = root / "gate_oof.parquet"
    validation_path = root / "gate_validation.parquet"
    grid_path = root / "gate_grid.json"
    oof.to_parquet(oof_path, index=False)
    validation.to_parquet(validation_path, index=False)
    grid_path.write_text(
        json.dumps(
            {
                "operating_points": [
                    {
                        "lambda_harm": 1,
                        "utility_threshold": -10,
                        "score_margin_threshold": 0,
                        "reliability_threshold": 0.5,
                        "stability_threshold": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return oof_path, validation_path, grid_path


def test_gate_cli_executes_resumes_and_records_ledger(tmp_path: Path) -> None:
    oof, validation, grid = _write_gate_inputs(
        tmp_path, validation_native=0, validation_challenger=1
    )
    run_dir = tmp_path / "run"
    output = run_dir / "08_lock" / "gates" / "crog"
    kwargs = {
        "oof_path": oof,
        "validation_path": validation,
        "grid_path": grid,
        "output_dir": output,
        "route": "crog",
        "feature_columns": FEATURES,
        "columns": GateColumns(),
        "model_seed": 42,
    }
    first = execute_gate_selection(run_dir=run_dir, command="synthetic-gate", **kwargs)
    assert first["decision"] == "GO"
    assert first["configuration"]["bootstrap_seed"] == 20260808
    assert first["configuration"]["bootstrap_iterations"] == 10_000
    assert first["test_access"] == "NONE"
    decisions = pd.read_parquet(output / "gate_validation_decisions.parquet")
    assert decisions["switch"].all()
    assert decisions["selected_correct"].all()
    model_path = output / "gate_transition_model.pkl"
    original_mtime = model_path.stat().st_mtime_ns
    second = execute_gate_selection(run_dir=run_dir, command="synthetic-resume", **kwargs)
    assert second["signature_sha256"] == first["signature_sha256"]
    assert model_path.stat().st_mtime_ns == original_mtime
    with sqlite3.connect(run_dir / "run_ledger.sqlite") as connection:
        row = connection.execute(
            "SELECT status, artifact_sha256 FROM stages WHERE stage='P8'"
        ).fetchone()
    assert row is not None and row[0] == "COMPLETE" and len(row[1]) == 64
    marker = output / "gate_selection.json"
    tampered = json.loads(marker.read_text(encoding="utf-8"))
    tampered["decision"] = "TAMPERED"
    marker.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="content hash"):
        execute_gate_selection(
            run_dir=run_dir, command="synthetic-tampered-resume", **kwargs
        )


def test_gate_cli_no_go_and_immutable_signature(tmp_path: Path) -> None:
    oof, validation, grid = _write_gate_inputs(
        tmp_path, validation_native=1, validation_challenger=0
    )
    output = tmp_path / "gate-output"
    kwargs = {
        "oof_path": oof,
        "validation_path": validation,
        "grid_path": grid,
        "output_dir": output,
        "route": "g1",
        "feature_columns": FEATURES,
        "columns": GateColumns(),
        "model_seed": 42,
    }
    result = run_gate_selection(**kwargs)
    assert result["decision"] == "NO_GO_NATIVE"
    decisions = pd.read_parquet(output / "gate_validation_decisions.parquet")
    assert not decisions["switch"].any()
    assert decisions["selected_correct"].all()
    grid.write_text(
        json.dumps(
            {
                "operating_points": [
                    {
                        "lambda_harm": 2,
                        "utility_threshold": -9,
                        "score_margin_threshold": 0,
                        "reliability_threshold": 0.5,
                        "stability_threshold": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="different signature"):
        run_gate_selection(**kwargs)


def test_gate_grid_cartesian_and_provenance_guard(tmp_path: Path) -> None:
    grid = tmp_path / "grid.json"
    grid.write_text(
        json.dumps(
            {
                "lambda_harm": [1, 2],
                "utility_thresholds": [0.0],
                "score_margin_thresholds": [0.0],
                "reliability_thresholds": [0.5],
                "stability_thresholds": [0.5, 0.8],
            }
        ),
        encoding="utf-8",
    )
    assert len(load_gate_grid(grid)) == 4
    assert GateColumns.from_mapping({"sample_id": "query_id"}).sample_id == "query_id"
    oof, validation, explicit_grid = _write_gate_inputs(
        tmp_path, validation_native=0, validation_challenger=1
    )
    frame = pd.read_parquet(validation)
    frame["prediction_source"] = "test"
    frame.to_parquet(validation, index=False)
    with pytest.raises(ValueError, match="provenance"):
        run_gate_selection(
            oof_path=oof,
            validation_path=validation,
            grid_path=explicit_grid,
            output_dir=tmp_path / "invalid",
            route="c1",
            feature_columns=FEATURES,
        )


@pytest.mark.parametrize(
    "features",
    [
        ("challenger_correct",),
        (*FEATURES, "challenger_correct"),
        ("ranker_score_margin",),
    ],
)
def test_gate_selector_rejects_noncanonical_or_supervised_features(
    tmp_path: Path, features: tuple[str, ...]
) -> None:
    oof, validation, grid = _write_gate_inputs(
        tmp_path, validation_native=0, validation_challenger=1
    )
    with pytest.raises(ValueError, match="feature|forbidden|supervision"):
        run_gate_selection(
            oof_path=oof,
            validation_path=validation,
            grid_path=grid,
            output_dir=tmp_path / "leaked-gate",
            route="g1",
            feature_columns=features,
        )
