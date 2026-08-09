from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.unified_reranking.apply_locked_test_gates import run
from tools.unified_reranking.prepare_gate_inputs import FEATURE_COLUMNS
from unified_reranking.gate import ConservativeTransitionModel, OOFTransitionData
from unified_reranking.hashing import sha256_file


def _fitted_model() -> ConservativeTransitionModel:
    pattern_native = np.asarray([0, 0, 1, 1] * 2, dtype=int)
    pattern_challenger = np.asarray([0, 1, 0, 1] * 2, dtype=int)
    native = np.tile(pattern_native, 3)
    challenger = np.tile(pattern_challenger, 3)
    columns = [challenger - native]
    columns.extend(
        np.sin(np.arange(len(native), dtype=float) + offset)
        for offset in range(1, len(FEATURE_COLUMNS))
    )
    matrix = np.column_stack(columns)
    return ConservativeTransitionModel(seed=42).fit(
        OOFTransitionData(
            features=matrix,
            feature_names=tuple(FEATURE_COLUMNS),
            native_correct=native,
            challenger_correct=challenger,
            scene_ids=[f"scene-{i}" for i in range(len(native))],
            oof_fold_ids=np.repeat(["f0", "f1", "f2"], 8),
        )
    )


def _write_run(root: Path) -> Path:
    run_dir = root / "run"
    (run_dir / "01_manifests").mkdir(parents=True)
    (run_dir / "02_candidates").mkdir(parents=True)
    feature_dir = (
        run_dir
        / "03_features"
        / "tracks"
        / "T2_matched_common"
        / "crog_test"
    )
    feature_dir.mkdir(parents=True)
    ranker_dir = run_dir / "08_lock" / "label_free_test_rankers" / "crog"
    ranker_dir.mkdir(parents=True)
    gate_dir = run_dir / "08_lock" / "gates" / "crog"
    gate_dir.mkdir(parents=True)

    pd.DataFrame({"sample_id": ["s1", "s2"]}).to_parquet(
        run_dir / "01_manifests" / "paired_test.parquet", index=False
    )
    candidates = pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "candidate_id": ["native", "challenger"],
            "candidate_geometry_sha256": ["g-native", "g-challenger"],
        }
    )
    candidates.to_parquet(
        run_dir / "02_candidates" / "crog_test_top5.parquet", index=False
    )
    features = candidates[["sample_id", "candidate_id"]].copy()
    features["calibrated_native_probability"] = [0.4, 0.8]
    features["native_score_raw"] = [0.3, 0.9]
    features["overall_feature_reliability"] = [0.7, 0.9]
    features["peak_retention_rate"] = [0.8, 0.9]
    features["perturbed_valid_fraction"] = [0.8, 0.9]
    features["mask_reliability"] = [0.6, 0.9]
    features.to_parquet(feature_dir / "candidate_features.parquet", index=False)
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "selected_candidate_id": ["challenger", None],
            "candidate_count": [2, 0],
            "native_candidate_id": ["native", None],
            "native_geometry_sha256": ["g-native", None],
            "selected_geometry_sha256": ["g-challenger", None],
            "ensemble_score_margin": [0.5, 0.0],
            "seed_challenger_votes": [3, 0],
            "challenger_exists": [True, False],
            "prediction_source": ["test_label_free", "test_label_free"],
        }
    ).to_parquet(ranker_dir / "per_sample_decisions.parquet", index=False)

    model_path = gate_dir / "gate_transition_model.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(_fitted_model(), stream, protocol=pickle.HIGHEST_PROTOCOL)
    gate = {
        "status": "COMPLETE",
        "decision": "GO",
        "test_access": "NONE",
        "configuration": {"feature_columns": list(FEATURE_COLUMNS)},
        "selection": {
            "selected_operating_point": {
                "lambda_harm": 1.0,
                "utility_threshold": -10.0,
                "score_margin_threshold": 0.0,
                "reliability_threshold": 0.0,
                "stability_threshold": 0.0,
                "minimum_seed_votes": 2,
            }
        },
        "artifacts": {
            "transition_model": {
                "path": str(model_path.resolve()),
                "sha256": sha256_file(model_path),
            }
        },
    }
    (gate_dir / "gate_selection.json").write_text(json.dumps(gate), encoding="utf-8")
    return run_dir


def test_applies_gate_without_test_outcomes_and_resumes(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    first = run(run_dir, "crog")
    assert first["candidate_test_labels_read"] is False
    decisions = pd.read_parquet(first["artifacts"]["decisions"]["path"])
    assert decisions["prediction_source"].eq("test_label_free").all()
    assert decisions.loc[decisions["sample_id"].eq("s1"), "switch"].item()
    no_output = decisions.loc[decisions["sample_id"].eq("s2")].iloc[0]
    assert not no_output["switch"]
    assert no_output["selected_candidate_id"] == ""
    mtime = Path(first["artifacts"]["decisions"]["path"]).stat().st_mtime_ns
    second = run(run_dir, "crog")
    assert second["signature_sha256"] == first["signature_sha256"]
    assert Path(second["artifacts"]["decisions"]["path"]).stat().st_mtime_ns == mtime


def test_rejects_outcome_columns_in_label_free_ranker_input(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    path = (
        run_dir
        / "08_lock"
        / "label_free_test_rankers"
        / "crog"
        / "per_sample_decisions.parquet"
    )
    frame = pd.read_parquet(path)
    frame["selected_correct"] = True
    frame.to_parquet(path, index=False)
    with pytest.raises(PermissionError, match="contain outcomes"):
        run(run_dir, "crog")
