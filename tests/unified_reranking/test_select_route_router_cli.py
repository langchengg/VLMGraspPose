from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.unified_reranking.select_route_router import (
    RouterColumns,
    execute_route_router_selection,
    load_router_grid,
    run_route_router_selection,
)
from unified_reranking.cross_route_inputs import router_feature_columns


G1_FEATURES = router_feature_columns("g1")
C1_FEATURES = router_feature_columns("c1")


def _router_features(
    names: tuple[str, ...], values: np.ndarray | float
) -> dict[str, np.ndarray | float]:
    return {name: values for name in names}


def _write_router_inputs(root: Path, *, crog_correct: int, g1_correct: int, c1_correct: int) -> tuple[Path, Path, Path]:
    crog_pattern = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    g1_pattern = np.asarray([1, 0, 0, 1, 1, 0, 0, 1], dtype=int)
    c1_pattern = np.asarray([1, 0, 1, 1, 0, 0, 0, 1], dtype=int)
    crog = np.tile(crog_pattern, 3)
    g1 = np.tile(g1_pattern, 3)
    c1 = np.tile(c1_pattern, 3)
    index = np.arange(len(crog), dtype=float)
    oof = pd.DataFrame(
        {
            "sample_id": [f"train-{i}" for i in range(len(crog))],
            "scene_id": [f"scene-{fold}-{row}" for fold in range(3) for row in range(8)],
            "prediction_source": "train_oof",
            "oof_fold": np.repeat(["f0", "f1", "f2"], 8),
            "crog_correct": crog,
            "g1_correct": g1,
            "c1_correct": c1,
            **_router_features(G1_FEATURES, np.sin(index)),
            **_router_features(C1_FEATURES, np.cos(index)),
        }
    )
    length = 40
    validation = pd.DataFrame(
        {
            "sample_id": [f"validation-{i}" for i in range(length)],
            "scene_id": [f"validation-scene-{i}" for i in range(length)],
            "prediction_source": "validation",
            "crog_correct": crog_correct,
            "g1_correct": g1_correct,
            "c1_correct": c1_correct,
            "crog_candidate_id": [f"crog-{i}" for i in range(length)],
            "g1_candidate_id": [f"g1-{i}" for i in range(length)],
            "c1_candidate_id": [f"c1-{i}" for i in range(length)],
            "g1_margin": 1.0,
            "c1_margin": 1.0,
            "g1_reliability": 1.0,
            "c1_reliability": 1.0,
            "g1_stability": 1.0,
            "c1_stability": 1.0,
            "g1_candidate_exists": True,
            "c1_candidate_exists": False,
            **_router_features(G1_FEATURES, 0.5),
            **_router_features(C1_FEATURES, 0.5),
        }
    )
    oof_path = root / "router_oof.parquet"
    validation_path = root / "router_validation.parquet"
    grid_path = root / "router_grid.json"
    oof.to_parquet(oof_path, index=False)
    validation.to_parquet(validation_path, index=False)
    grid_path.write_text(
        json.dumps(
            {
                "operating_points": [
                    {
                        "lambda_router": 1,
                        "utility_threshold": -10,
                        "margin_threshold": 0,
                        "reliability_threshold": 0.5,
                        "stability_threshold": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return oof_path, validation_path, grid_path


def test_router_cli_go_resumes_and_records_crog_first_audit(tmp_path: Path) -> None:
    oof, validation, grid = _write_router_inputs(
        tmp_path, crog_correct=0, g1_correct=1, c1_correct=0
    )
    run_dir = tmp_path / "run"
    output = run_dir / "08_lock" / "route_router"
    kwargs = {
        "oof_path": oof,
        "validation_path": validation,
        "grid_path": grid,
        "output_dir": output,
        "g1_feature_columns": G1_FEATURES,
        "c1_feature_columns": C1_FEATURES,
        "columns": RouterColumns(),
        "model_seed": 42,
    }
    first = execute_route_router_selection(
        run_dir=run_dir, command="synthetic-router", **kwargs
    )
    assert first["decision"] == "GO"
    assert first["configuration"]["default_route"] == "CROG"
    assert first["configuration"]["tie_break"] == ["G1", "C1"]
    assert first["configuration"]["bootstrap_seed"] == 20260808
    assert first["configuration"]["bootstrap_iterations"] == 10_000
    decisions = pd.read_parquet(output / "route_router_validation_decisions.parquet")
    assert set(decisions["selected_route"]) == {"G1"}
    assert decisions["selected_correct"].all()
    model_path = output / "route_transition_models.pkl"
    mtime = model_path.stat().st_mtime_ns
    second = execute_route_router_selection(
        run_dir=run_dir, command="synthetic-router-resume", **kwargs
    )
    assert second["signature_sha256"] == first["signature_sha256"]
    assert model_path.stat().st_mtime_ns == mtime
    with sqlite3.connect(run_dir / "run_ledger.sqlite") as connection:
        row = connection.execute(
            "SELECT status, artifact_sha256 FROM stages WHERE stage='P9'"
        ).fetchone()
    assert row is not None and row[0] == "COMPLETE" and len(row[1]) == 64
    marker = output / "route_router_selection.json"
    tampered = json.loads(marker.read_text(encoding="utf-8"))
    tampered["decision"] = "TAMPERED"
    marker.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="content hash"):
        execute_route_router_selection(
            run_dir=run_dir, command="synthetic-router-tampered-resume", **kwargs
        )


def test_router_cli_no_go_defaults_every_row_to_crog(tmp_path: Path) -> None:
    oof, validation, grid = _write_router_inputs(
        tmp_path, crog_correct=1, g1_correct=0, c1_correct=0
    )
    output = tmp_path / "router-output"
    result = run_route_router_selection(
        oof_path=oof,
        validation_path=validation,
        grid_path=grid,
        output_dir=output,
        g1_feature_columns=G1_FEATURES,
        c1_feature_columns=C1_FEATURES,
        model_seed=42,
    )
    assert result["decision"] == "NO_GO_CROG"
    decisions = pd.read_parquet(output / "route_router_validation_decisions.parquet")
    assert set(decisions["selected_route"]) == {"CROG"}
    assert decisions["selected_correct"].all()


def test_router_grid_and_provenance_are_fail_closed(tmp_path: Path) -> None:
    grid = tmp_path / "cartesian.json"
    grid.write_text(
        json.dumps(
            {
                "lambda_router": [1, 2],
                "utility_thresholds": [0],
                "margin_thresholds": [0],
                "reliability_thresholds": [0.5],
                "stability_thresholds": [0.5, 0.8],
            }
        ),
        encoding="utf-8",
    )
    assert len(load_router_grid(grid)) == 4
    assert RouterColumns.from_mapping({"sample_id": "query_id"}).sample_id == "query_id"
    oof, validation, explicit_grid = _write_router_inputs(
        tmp_path, crog_correct=0, g1_correct=1, c1_correct=0
    )
    frame = pd.read_parquet(oof)
    frame["prediction_source"] = "train_in_sample"
    frame.to_parquet(oof, index=False)
    with pytest.raises(ValueError, match="provenance"):
        run_route_router_selection(
            oof_path=oof,
            validation_path=validation,
            grid_path=explicit_grid,
            output_dir=tmp_path / "invalid",
            g1_feature_columns=G1_FEATURES,
            c1_feature_columns=C1_FEATURES,
        )


@pytest.mark.parametrize(
    ("g1_features", "c1_features"),
    [
        (("g1_correct",), C1_FEATURES),
        ((*G1_FEATURES, "g1_correct"), C1_FEATURES),
        ((G1_FEATURES[0],), C1_FEATURES),
        (G1_FEATURES, ("c1_correct",)),
    ],
)
def test_router_selector_rejects_noncanonical_or_supervised_features(
    tmp_path: Path,
    g1_features: tuple[str, ...],
    c1_features: tuple[str, ...],
) -> None:
    oof, validation, grid = _write_router_inputs(
        tmp_path, crog_correct=0, g1_correct=1, c1_correct=0
    )
    with pytest.raises(ValueError, match="feature|identity|supervision"):
        run_route_router_selection(
            oof_path=oof,
            validation_path=validation,
            grid_path=grid,
            output_dir=tmp_path / "leaked-router",
            g1_feature_columns=g1_features,
            c1_feature_columns=c1_features,
        )
