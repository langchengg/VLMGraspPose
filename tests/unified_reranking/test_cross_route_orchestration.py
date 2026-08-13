from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.unified_reranking.analyze_union_headroom import run as run_union_headroom
from tools.unified_reranking.apply_locked_route_router import run as apply_router
from tools.unified_reranking.apply_locked_union_ranker import run as apply_union_ranker
from tools.unified_reranking.prepare_label_free_test_router_inputs import (
    build_label_free_gate_input,
)
from tools.unified_reranking.prepare_route_router_inputs import enrich_gated_route
from tools.unified_reranking.prepare_union_features import (
    build_union_split,
    run_split as prepare_union_split,
)
from tools.unified_reranking.train_union_rankers import (
    UnionBudget,
    formal_plan,
    run_orchestrator,
    train_union_cell,
)
from unified_reranking.cross_route_inputs import (
    add_router_features,
    assert_router_feature_names,
    cross_fitted_gate_probabilities,
    geometry_agreement,
    router_feature_columns,
)
from unified_reranking.hashing import sha256_file


class _FakeRouter:
    def predict_probabilities(self, features_by_route):
        length = len(features_by_route["G1"])
        assert length == 3
        return {
            "G1": (np.asarray([0.8, 0.8, 0.2]), np.asarray([0.1, 0.1, 0.1])),
            "C1": (np.asarray([0.8, 0.7, 0.2]), np.asarray([0.1, 0.1, 0.1])),
        }


class _FakeUnionModel:
    def predict(self, features):
        return np.asarray(features, dtype=float)[:, 0]


def _gated_route(route: str) -> pd.DataFrame:
    offsets = {"crog": 0.0, "g1": 1.0, "c1": 2.0}
    offset = offsets[route]
    return pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "scene_id": ["a", "b"],
            "prediction_source": "validation",
            "gated_candidate_id": [f"{route}-0", f"{route}-1"],
            "gated_correct": [True, False],
            "gate_switch": [False, True],
            "gate_probability_recover": [0.2, 0.8],
            "gate_probability_harm": [0.1, 0.1],
            "gate_utility": [0.1, 0.7],
            "route_specific_margin": [0.2, 0.4],
            "selected_calibrated_probability": [0.7, 0.8],
            "selected_reliability": [0.9, 0.8],
            "selected_stability": [0.9, 0.8],
            "selected_mask_reliability": [0.9, 0.8],
            "selected_candidate_exists": [True, True],
            "selected_candidate_geometry_sha256": [f"h-{route}-0", f"h-{route}-1"],
            "selected_cx_px": [10.0 + offset, 20.0 + offset],
            "selected_cy_px": [10.0, 20.0],
            "selected_theta_deg": [0.0, 10.0],
            "selected_width_px": [20.0, 20.0],
            "selected_height_px": [10.0, 10.0],
        }
    )


def test_router_features_are_identity_free_and_geometry_is_periodic() -> None:
    same = geometry_agreement(
        {"cx_px": 0, "cy_px": 0, "theta_deg": 0, "width_px": 20, "height_px": 10},
        {"cx_px": 0, "cy_px": 0, "theta_deg": 180, "width_px": 20, "height_px": 10},
    )
    assert same == {"center": 1.0, "angle": 1.0, "width": 1.0, "geometry": 1.0}
    frame = add_router_features({route: _gated_route(route) for route in ("crog", "g1", "c1")})
    assert frame.columns.is_unique
    for route in ("g1", "c1"):
        columns = router_feature_columns(route)
        assert_router_feature_names(columns)
        assert set(columns).issubset(frame.columns)
        assert np.isfinite(frame.loc[:, columns].to_numpy(float)).all()
    with pytest.raises(ValueError, match="identity/supervision"):
        assert_router_feature_names(("sample_id", "g1_margin"))


def test_gate_probabilities_are_outer_cross_fitted() -> None:
    native_pattern = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    challenger_pattern = np.asarray([1, 0, 0, 1, 1, 0, 0, 1])
    native = np.tile(native_pattern, 5)
    challenger = np.tile(challenger_pattern, 5)
    index = np.arange(len(native), dtype=float)
    frame = pd.DataFrame(
        {
            "sample_id": [f"s-{i}" for i in range(len(native))],
            "scene_id": [f"scene-{fold}-{row}" for fold in range(5) for row in range(8)],
            "oof_fold": np.repeat([f"f{i}" for i in range(5)], 8),
            "prediction_source": "train_oof",
            "native_correct": native,
            "challenger_correct": challenger,
            "margin": challenger - native,
            "stability": np.sin(index),
            "candidate_exists": 1.0,
        }
    )
    recover, harm, audit = cross_fitted_gate_probabilities(
        frame, ("margin", "stability", "candidate_exists"), seed=42
    )
    assert recover.shape == harm.shape == (40,)
    assert np.isfinite(recover).all() and np.isfinite(harm).all()
    assert len(audit) == 5
    assert all(row["training_fold_count"] == 4 for row in audit)


def test_router_reconstructs_native_reliability_from_locked_gate_deltas() -> None:
    gated = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "scene_id": ["scene"],
            "prediction_source": ["validation"],
            "native_candidate_id": ["a"],
            "challenger_candidate_id": ["b"],
            "gated_candidate_id": ["a"],
            "gate_switch": [False],
            "gate_probability_recover": [0.2],
            "gate_probability_harm": [0.1],
            "gate_utility": [0.1],
            "score_margin": [0.4],
            "native_calibrated_probability": [0.7],
            "challenger_calibrated_probability": [0.8],
            "challenger_overall_reliability": [0.9],
            "overall_reliability_delta": [0.2],
            "challenger_perturbation_stability": [0.8],
            "perturbation_stability_delta": [0.3],
            "challenger_mask_reliability": [0.85],
            "mask_reliability_delta": [0.25],
        }
    )
    catalog = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "candidate_id": ["a"],
            "cx_px": [1.0],
            "cy_px": [2.0],
            "theta_deg": [3.0],
            "width_px": [4.0],
            "height_px": [2.0],
            "candidate_geometry_sha256": ["geometry"],
        }
    )
    enriched = enrich_gated_route(gated, catalog)
    assert enriched.loc[0, "selected_reliability"] == pytest.approx(0.7)
    assert enriched.loc[0, "selected_stability"] == pytest.approx(0.5)
    assert enriched.loc[0, "selected_mask_reliability"] == pytest.approx(0.6)


def test_label_free_gate_input_rejects_supervision_and_builds_fixed_features() -> None:
    paired = pd.DataFrame({"sample_id": ["s0"], "scene_id": ["scene"]})
    decisions = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "native_candidate_id": ["n"],
            "selected_candidate_id": ["c"],
            "selected_geometry_sha256": ["hc"],
            "ensemble_score_margin": [0.3],
            "seed_challenger_votes": [3],
            "challenger_exists": [True],
            "native_native_score": [0.1],
            "challenger_native_score": [0.4],
        }
    )
    catalog = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["n", "c"],
            "calibrated_native_probability": [0.2, 0.8],
            "overall_feature_reliability": [0.9, 0.8],
            "stability": [0.9, 0.7],
            "mask_reliability": [0.9, 0.7],
            "candidate_geometry_sha256": ["hn", "hc"],
        }
    )
    frame = build_label_free_gate_input(paired, decisions, catalog)
    assert set(frame["prediction_source"]) == {"test_label_free"}
    assert frame.loc[0, "native_score_delta"] == pytest.approx(0.3)
    assert not any("correct" in column for column in frame.columns)
    invalid = decisions.assign(selected_correct=True)
    with pytest.raises(PermissionError, match="supervision"):
        build_label_free_gate_input(paired, invalid, catalog)


def _write_union_run(root: Path, *, complementary: bool) -> Path:
    run = root / ("complementary" if complementary else "no-headroom")
    for split, length in (("train", 8), ("validation", 4)):
        manifest = pd.DataFrame(
            {
                "sample_id": [f"{split}-{i}" for i in range(length)],
                "scene_id": [f"scene-{split}-{i}" for i in range(length)],
            }
        )
        path = run / "01_manifests" / f"paired_{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_parquet(path, index=False)
        for route_index, route in enumerate(("crog", "g1", "c1")):
            candidates = pd.DataFrame(
                {
                    "sample_id": manifest["sample_id"],
                    "candidate_id": [f"{route}-0"] * length,
                    "native_rank": 1,
                }
            )
            if complementary:
                success = np.asarray([(i % 3) == route_index for i in range(length)])
            else:
                success = np.asarray([(i % 4) == 0 for i in range(length)])
            labels = candidates[["sample_id", "candidate_id"]].assign(candidate_success=success)
            candidate_path = run / "02_candidates" / f"{route}_{split}_top5.parquet"
            label_path = run / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            candidates.to_parquet(candidate_path, index=False)
            labels.to_parquet(label_path, index=False)
    return run


def test_union_headroom_emits_exact_no_go_or_eligible_states(tmp_path: Path) -> None:
    no_headroom = _write_union_run(tmp_path, complementary=False)
    result = run_union_headroom(no_headroom)
    assert result["decision"] == "NO_UNION_HEADROOM"
    assert result["complex_union_training"] == "NOT_APPLICABLE_NO_HEADROOM"
    assert result["deterministic_union_nms_secondary"] == "NOT_APPLICABLE_NO_HEADROOM"
    assert result["test_access"] == "NONE"
    marker = no_headroom / "07_validation" / "union_headroom" / "manifest.json"
    mtime = marker.stat().st_mtime_ns
    assert run_union_headroom(no_headroom)["signature_sha256"] == result["signature_sha256"]
    assert marker.stat().st_mtime_ns == mtime

    complementary = _write_union_run(tmp_path, complementary=True)
    go = run_union_headroom(complementary)
    assert go["decision"] == "UNION_HEADROOM_AVAILABLE"
    assert go["summaries"]["validation"]["maximum_candidates_per_sample"] == 3


def test_locked_router_application_is_label_free_crog_default_and_g1_first(tmp_path: Path) -> None:
    run = tmp_path / "run"
    model_path = tmp_path / "router.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(_FakeRouter(), stream)
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    selection_path = selection_dir / "route_router_selection.json"
    selection = {
        "status": "COMPLETE",
        "test_access": "NONE",
        "decision": "GO",
        "configuration": {
            "default_route": "CROG",
            "tie_break": ["G1", "C1"],
            "feature_columns": {"G1": ["g1_feature"], "C1": ["c1_feature"]},
        },
        "selection": {
            "selected_operating_point": {
                "lambda_router": 1,
                "utility_threshold": 0.3,
                "margin_threshold": 0.0,
                "reliability_threshold": 0.5,
                "stability_threshold": 0.5,
            }
        },
        "artifacts": {
            "transition_models": {"path": str(model_path), "sha256": sha256_file(model_path)}
        },
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    input_path = input_dir / "test_label_free.parquet"
    frame = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "scene_id": ["a", "b", "c"],
            "prediction_source": "test_label_free",
            "g1_feature": [1.0, 1.0, 0.0],
            "c1_feature": [1.0, 1.0, 0.0],
            "g1_margin": 1.0,
            "c1_margin": 1.0,
            "g1_reliability": 1.0,
            "c1_reliability": 1.0,
            "g1_stability": 1.0,
            "c1_stability": 1.0,
            "g1_candidate_exists": [True, False, True],
            "c1_candidate_exists": True,
            "crog_candidate_id": ["crog0", "crog1", "crog2"],
            "g1_candidate_id": ["g10", "", "g12"],
            "c1_candidate_id": ["c10", "c11", "c12"],
            "crog_selected_candidate_geometry_sha256": ["hc0", "hc1", "hc2"],
            "g1_selected_candidate_geometry_sha256": ["hg0", "", "hg2"],
            "c1_selected_candidate_geometry_sha256": ["h10", "h11", "h12"],
        }
    )
    frame.to_parquet(input_path, index=False)
    input_manifest = {
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "artifacts": {
            "test_label_free": {"path": str(input_path), "sha256": sha256_file(input_path)}
        },
    }
    (input_dir / "manifest.json").write_text(json.dumps(input_manifest), encoding="utf-8")
    result = apply_router(
        run,
        router_selection_path=selection_path,
        test_input_path=input_path,
        output_dir=tmp_path / "application",
    )
    decisions = pd.read_parquet(result["artifacts"]["decisions"]["path"])
    assert decisions["selected_route"].tolist() == ["G1", "C1", "CROG"]
    assert result["candidate_test_labels_read"] is False
    assert not any("correct" in column for column in decisions.columns)


def _write_union_feature_sources(root: Path) -> Path:
    run = root / "union-model-run"
    model_columns = ["base_logit", "calibrated_native_probability", "signal"]
    route_bias = {"crog": 0.0, "g1": 0.1, "c1": 0.2}
    for split, length in (("train", 10), ("validation", 5), ("test", 3)):
        sample_ids = [f"{split}-{index}" for index in range(length)]
        manifest = pd.DataFrame(
            {"sample_id": sample_ids, "scene_id": [f"scene-{value}" for value in sample_ids]}
        )
        manifest_path = run / "01_manifests" / f"paired_{split}.parquet"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_parquet(manifest_path, index=False)
        for route_index, route in enumerate(("crog", "g1", "c1")):
            rows = []
            labels = []
            candidates = []
            for sample_index, sample_id in enumerate(sample_ids):
                for candidate_index in range(2):
                    candidate_id = f"candidate_{candidate_index}"
                    probability = 0.25 + 0.45 * (candidate_index == 0) + route_bias[route]
                    probability = min(probability, 0.95)
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "candidate_id": candidate_id,
                            "route": route.upper(),
                            "native_rank": candidate_index + 1,
                            "base_logit": float(np.log(probability / (1.0 - probability))),
                            "calibrated_native_probability": probability,
                            "signal": float(route_index - candidate_index),
                        }
                    )
                    candidates.append(
                        {
                            "sample_id": sample_id,
                            "candidate_id": candidate_id,
                            "native_rank": candidate_index + 1,
                            # Intentionally identical across routes: no-dedup must retain it.
                            "candidate_geometry_sha256": f"geometry-{sample_index}-{candidate_index}",
                        }
                    )
                    if split != "test":
                        success = candidate_index == 0 and route_index == sample_index % 3
                        labels.append(
                            {
                                "sample_id": sample_id,
                                "candidate_id": candidate_id,
                                "candidate_success": success,
                                "jacquard_margin": 1.0 if success else -1.0,
                            }
                        )
            feature_dir = run / "03_features" / "tracks" / "T2_matched_common" / f"{route}_{split}"
            feature_dir.mkdir(parents=True, exist_ok=True)
            feature_path = feature_dir / "candidate_features.parquet"
            pd.DataFrame(rows).to_parquet(feature_path, index=False)
            (feature_dir / "feature_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "model_feature_columns": model_columns,
                        "artifact": {
                            "path": str(feature_path),
                            "sha256": sha256_file(feature_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_path = run / "02_candidates" / f"{route}_{split}_top5.parquet"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(candidates).to_parquet(candidate_path, index=False)
            if split != "test":
                label_path = run / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(labels).to_parquet(label_path, index=False)
    folds = pd.DataFrame(
        {
            "sample_id": [f"train-{index}" for index in range(10)],
            "fold": np.repeat(np.arange(5), 2),
        }
    )
    fold_path = run / "04_splits" / "fold_assignments.parquet"
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(fold_path, index=False)
    return run


def test_union_features_preserve_route_identity_calibration_and_no_dedup(tmp_path: Path) -> None:
    run = _write_union_feature_sources(tmp_path)
    features, labels, columns, audit = build_union_split(run, "train")
    assert labels is not None
    assert features.groupby("sample_id").size().eq(6).all()
    assert features["candidate_id"].str.match(r"^(CROG|G1|C1):candidate_[01]$").all()
    assert set(("union_route_crog", "union_route_g1", "union_route_c1")).issubset(columns)
    assert audit["primary_union_deduplication"] == "NONE"
    assert audit["geometry_duplicate_rows_retained"] == len(features)
    assert len(labels) == len(features)
    with pytest.raises(PermissionError, match="Test labels"):
        # Development-only oracle code has an explicit split guard.
        from tools.unified_reranking.analyze_union_headroom import analyze_split

        analyze_split(run, "test")


def test_union_formal_plan_has_equal_fixed_budget_and_cells_are_resumable(tmp_path: Path) -> None:
    plan = formal_plan()
    assert len(plan) == 36
    assert {json.dumps(cell["budget"], sort_keys=True) for cell in plan} == {
        json.dumps(plan[0]["budget"], sort_keys=True)
    }
    assert {(cell["encoder"], cell["seed"]) for cell in plan} == {
        (encoder, seed) for encoder in ("lambdamart", "deepsets") for seed in (42, 123, 2026)
    }

    run = _write_union_feature_sources(tmp_path)
    prepare_union_split(run, "train")
    prepare_union_split(run, "validation")
    smoke_budget = UnionBudget(
        lambdamart_n_estimators=5,
        deepsets_epochs=2,
        deepsets_patience=1,
        deepsets_batch_size=8,
    )
    cell = train_union_cell(
        run,
        encoder="deepsets",
        seed=42,
        mode="oof",
        held_fold=0,
        budget=smoke_budget,
        output_root=tmp_path / "smoke-cells",
    )
    assert cell["status"] == "COMPLETE"
    assert cell["configuration"]["deterministic_cpu"] is True
    model_path = Path(cell["artifacts"]["model"]["path"])
    mtime = model_path.stat().st_mtime_ns
    resumed = train_union_cell(
        run,
        encoder="deepsets",
        seed=42,
        mode="oof",
        held_fold=0,
        budget=smoke_budget,
        output_root=tmp_path / "smoke-cells",
    )
    assert resumed["cell_id"] == cell["cell_id"]
    assert model_path.stat().st_mtime_ns == mtime

    headroom_dir = run / "07_validation" / "union_headroom"
    headroom_dir.mkdir(parents=True, exist_ok=True)
    (headroom_dir / "manifest.json").write_text(
        json.dumps({"status": "COMPLETE", "decision": "UNION_HEADROOM_AVAILABLE"}),
        encoding="utf-8",
    )
    planned = run_orchestrator(run, execute=False)
    assert planned["status"] == "PLANNED"
    assert planned["cell_count"] == 36
    assert planned["execution_authorized"] is False


def test_locked_union_test_application_uses_three_models_without_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.unified_reranking.apply_locked_union_ranker as application_module

    run = _write_union_feature_sources(tmp_path)
    test_manifest = prepare_union_split(run, "test")
    columns = tuple(test_manifest["model_feature_columns"])
    cell_records = []
    for seed in (42, 123, 2026):
        model_path = tmp_path / f"union-{seed}.pkl"
        with model_path.open("wb") as stream:
            pickle.dump(_FakeUnionModel(), stream)
        cell_dir = tmp_path / f"cell-{seed}"
        cell_dir.mkdir()
        cell_path = cell_dir / "manifest.json"
        cell = {
            "status": "COMPLETE",
            "configuration": {"encoder": "lambdamart", "seed": seed, "mode": "validation"},
            "feature_columns": list(columns),
            "preprocessor": {
                "columns": list(columns),
                "medians": [0.0] * len(columns),
                "means": [0.0] * len(columns),
                "scales": [1.0] * len(columns),
            },
            "artifacts": {
                "model": {"path": str(model_path), "sha256": sha256_file(model_path)}
            },
        }
        cell_path.write_text(json.dumps(cell), encoding="utf-8")
        cell_records.append({"path": str(cell_path), "sha256": sha256_file(cell_path)})
    loaded_models: list[Path] = []

    def safe_scores(path: Path, features: np.ndarray) -> np.ndarray:
        loaded_models.append(path)
        return _FakeUnionModel().predict(features)

    monkeypatch.setattr(
        application_module, "_native_lightgbm_scores", safe_scores
    )
    validation_ensemble = {
        "status": "COMPLETE",
        "identity": {"encoder": "lambdamart", "split": "validation"},
        "sources": {"cells": cell_records},
    }
    validation_ensemble_path = tmp_path / "selected_union_validation.json"
    validation_ensemble_path.write_text(
        json.dumps(validation_ensemble), encoding="utf-8"
    )
    selection_path = tmp_path / "selected_union_ranker.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "VALIDATION_LOCKED",
                "selected_encoder": "lambdamart",
                "test_access": "NONE",
                "selected_validation_manifest": {
                    "path": str(validation_ensemble_path),
                    "sha256": sha256_file(validation_ensemble_path),
                },
                "selected_validation_ensemble": validation_ensemble,
            }
        ),
        encoding="utf-8",
    )
    result = apply_union_ranker(
        run,
        selection_path=selection_path,
        output_dir=tmp_path / "union-test-application",
    )
    decisions = pd.read_parquet(result["artifacts"]["decisions"]["path"])
    assert len(decisions) == 3
    assert decisions["selected_candidate_id"].str.contains(":").all()
    assert result["candidate_test_labels_read"] is False
    assert not any("correct" in column for column in decisions.columns)
    assert loaded_models == [tmp_path / f"union-{seed}.pkl" for seed in (42, 123, 2026)]
