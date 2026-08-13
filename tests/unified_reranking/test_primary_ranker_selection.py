from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from unified_reranking.hashing import sha256_file
from unified_reranking.prelock import _selected_ranker_contract


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "unified_reranking"
    / "select_primary_rankers.py"
)
SPEC = importlib.util.spec_from_file_location("select_primary_rankers", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_configuration_match_is_exact_for_seed_mode_fold_and_parameters() -> None:
    source_identity = {
        "train_features_sha256": "a" * 64,
        "train_feature_manifest_sha256": "d" * 64,
        "train_feature_columns": ["native_score_raw"],
        "train_feature_schema_sha256": "e" * 64,
        "train_labels_sha256": "b" * 64,
        "folds_sha256": "c" * 64,
        "selected_feature_columns": ["native_score_raw"],
        "selected_feature_schema_sha256": "f" * 64,
        "tool_sha256": "1" * 64,
        "training_code_sha256": "2" * 64,
    }
    configuration = {
        "encoder": "mlp",
        "loss": "ranknet",
        "seed": 42,
        "mode": "oof",
        "held_fold": 3,
        "alpha": 0.5,
        "learning_rate": 0.0003,
        "source_identity": source_identity,
    }
    choice = {
        "encoder": "mlp",
        "loss": "ranknet",
        "parameters": {"alpha": 0.5, "learning_rate": 0.0003},
        "screen_source_identity": source_identity,
    }
    assert MODULE._configuration_matches(
        configuration, choice, seed=42, mode="oof", fold=3
    )
    assert not MODULE._configuration_matches(
        configuration, choice, seed=123, mode="oof", fold=3
    )
    assert not MODULE._configuration_matches(
        configuration, {**choice, "parameters": {"alpha": 1.0}}, seed=42, mode="oof", fold=3
    )
    assert not MODULE._configuration_matches(
        {**configuration, "source_identity": {**source_identity, "folds_sha256": "d" * 64}},
        choice,
        seed=42,
        mode="oof",
        fold=3,
    )


def test_t3_configuration_match_requires_same_benchmark_hash() -> None:
    source_identity = {
        "train_features_sha256": "a" * 64,
        "train_feature_manifest_sha256": "b" * 64,
        "train_feature_columns": ["native_score_raw"],
        "train_feature_schema_sha256": "c" * 64,
        "train_labels_sha256": "d" * 64,
        "folds_sha256": "e" * 64,
        "selected_feature_columns": ["native_score_raw"],
        "selected_feature_schema_sha256": "f" * 64,
        "tool_sha256": "1" * 64,
        "training_code_sha256": "2" * 64,
        "train_feature_extraction_benchmark_sha256": "3" * 64,
        "validation_feature_extraction_benchmark_sha256": "3" * 64,
    }
    configuration = {
        "track": "T3_tri_backend",
        "encoder": "mlp",
        "loss": "ranknet",
        "seed": 42,
        "mode": "validation",
        "held_fold": None,
        "source_identity": source_identity,
    }
    choice = {
        "encoder": "mlp",
        "loss": "ranknet",
        "parameters": {},
        "screen_source_identity": source_identity,
    }
    assert MODULE._configuration_matches(
        configuration, choice, seed=42, mode="validation", fold=None
    )
    assert not MODULE._configuration_matches(
        {
            **configuration,
            "source_identity": {
                **source_identity,
                "validation_feature_extraction_benchmark_sha256": "4" * 64,
            },
        },
        choice,
        seed=42,
        mode="validation",
        fold=None,
    )


def test_decision_augmentation_uses_stable_vote_and_margin_contract() -> None:
    candidates = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "candidate_id": ["native", "challenger"],
            "native_rank": [1, 2],
            "candidate_geometry_sha256": ["g1", "g2"],
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "candidate_id": ["native", "challenger"],
            "native_rank": [1, 2],
            "score_seed_42": [0.0, 1.0],
            "score_seed_123": [0.0, 2.0],
            "score_seed_2026": [2.0, 0.0],
            "ensemble_score": [2 / 3, 1.0],
        }
    )
    decisions = pd.DataFrame(
        {
            "sample_id": ["s"],
            "selected_candidate_id": ["challenger"],
            "selected_correct": [True],
        }
    )
    result = MODULE._augment_decisions(candidates, predictions, decisions)
    assert result.loc[0, "seed_challenger_votes"] == 2
    assert result.loc[0, "ensemble_score_margin"] == pytest.approx(1 / 3)
    assert bool(result.loc[0, "challenger_exists"])
    assert result.loc[0, "selected_geometry_sha256"] == "g2"


def test_margin_is_challenger_minus_native_not_top1_minus_runner_up() -> None:
    candidates = pd.DataFrame(
        {
            "sample_id": ["s"] * 3,
            "candidate_id": ["native", "runner_up", "challenger"],
            "native_rank": [1, 2, 3],
            "candidate_geometry_sha256": ["g1", "g2", "g3"],
        }
    )
    predictions = candidates[["sample_id", "candidate_id", "native_rank"]].copy()
    for seed in (42, 123, 2026):
        predictions[f"score_seed_{seed}"] = [0.1, 0.8, 1.0]
    predictions["ensemble_score"] = [0.1, 0.8, 1.0]
    result = MODULE._augment_decisions(
        candidates,
        predictions,
        pd.DataFrame(
            {
                "sample_id": ["s"],
                "selected_candidate_id": ["challenger"],
                "selected_correct": [True],
            }
        ),
    )
    assert result.loc[0, "ensemble_score_margin"] == pytest.approx(0.9)


def test_producer_hash_bindings_are_accepted_by_prelock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    screen_manifest = run / "07_validation" / "matrix_cells" / "screen.json"
    screen_manifest.parent.mkdir(parents=True, exist_ok=True)
    screen_manifest.write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    source_identity = {
        "train_features_sha256": "a" * 64,
        "train_labels_sha256": "b" * 64,
        "folds_sha256": "c" * 64,
    }
    selections = {
        f"{route}/{track}": [
            {
                "method_code": "R4_mlp_ranknet",
                "encoder": "mlp",
                "loss": "ranknet",
                "parameters": {},
                "screen_source_identity": source_identity,
                "screen_manifest": str(screen_manifest.resolve()),
                "screen_manifest_sha256": sha256_file(screen_manifest),
            }
        ]
        for route in MODULE.ROUTES
        for track in MODULE.TRACKS
    }
    finalist = run / "05_models/screen_finalists.json"
    finalist.parent.mkdir(parents=True, exist_ok=True)
    finalist.write_text(
        json.dumps(
            {"status": "VALIDATION_SCREEN_LOCKED", "selections": selections}
        ),
        encoding="utf-8",
    )
    selected_execution = (
        run / "05_models" / "matrix_plans" / "selected_latest_execution.json"
    )
    selected_execution.parent.mkdir(parents=True, exist_ok=True)
    selected_execution.write_text(
        json.dumps({"status": "COMPLETE", "phase": "selected"}),
        encoding="utf-8",
    )

    def fake_ensemble(
        run_dir: Path,
        _cells,
        *,
        route: str,
        track: str,
        choice: dict,
        split: str,
    ) -> dict:
        identity = {
            "route": route,
            "track": track,
            "method_code": choice["method_code"],
            "encoder": choice["encoder"],
            "loss": choice["loss"],
            "seeds": [42, 123, 2026],
        }
        ensemble_id = f"{route}-{track}"
        directory = run_dir / ("06_oof" if split == "train" else "07_validation") / "ensembles" / ensemble_id
        directory.mkdir(parents=True, exist_ok=True)
        decisions = directory / "per_sample_decisions.parquet"
        predictions = directory / "per_candidate_scores.parquet"
        pd.DataFrame(
            {
                "sample_id": ["s"],
                "selected_candidate_id": ["a"],
                "selected_correct": [True],
            }
        ).to_parquet(decisions, index=False)
        pd.DataFrame(
            {"sample_id": ["s"], "candidate_id": ["a"], "ensemble_score": [1.0]}
        ).to_parquet(predictions, index=False)
        matrix_records = []
        if split == "validation" and track == MODULE.PRIMARY_TRACK:
            for seed in MODULE.FORMAL_SEEDS:
                cell_path = directory / f"cell-{seed}.json"
                cell_path.write_text(
                    json.dumps(
                        {
                            "status": "COMPLETE",
                            "configuration": {
                                "route": route,
                                "track": track,
                                "encoder": "mlp",
                                "loss": "ranknet",
                                "seed": seed,
                                "mode": "validation",
                                "held_fold": None,
                            },
                            "feature_columns": [
                                "native_score_raw",
                                "overall_feature_reliability",
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                matrix_records.append(
                    {"path": str(cell_path.resolve()), "sha256": sha256_file(cell_path)}
                )
        manifest = {
            "status": "COMPLETE",
            "split": split,
            "identity": identity,
            "ensemble_id": ensemble_id,
            "metrics": {"j_at_1": 1.0, "mrr_at_5": 1.0},
            "sources": {"matrix_manifests": matrix_records},
            "artifacts": {
                "predictions": {"path": str(predictions), "sha256": sha256_file(predictions)},
                "decisions": {"path": str(decisions), "sha256": sha256_file(decisions)},
            },
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr(MODULE, "_scan_cells", lambda _run: [])
    monkeypatch.setattr(MODULE, "_build_ensemble", fake_ensemble)
    monkeypatch.setattr(
        MODULE,
        "_native_decisions",
        lambda _run, _route, _split: (
            {"oracle_at_5": 1.0},
            pd.DataFrame(
                {
                    "sample_id": ["s"],
                    "selected_candidate_id": ["a"],
                    "selected_correct": [True],
                }
            ),
        ),
    )
    MODULE.run(run)
    selection_path = run / "07_validation/selected_primary_ungated.json"
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    for route in MODULE.ROUTES:
        record = selected["selections"][route]
        assert record["validation_manifest_sha256"] == sha256_file(
            Path(record["validation_manifest"])
        )
        assert record["oof_manifest_sha256"] == sha256_file(
            Path(record["oof_manifest"])
        )
        contract = _selected_ranker_contract(run, route, record)
        assert contract["identity"]["method_code"] == "R4_mlp_ranknet"
