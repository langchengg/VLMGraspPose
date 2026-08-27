"""Fixture-scoped contract tests; no value here is a formal experiment result."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import graspnet6d.experiment_analysis as experiment_analysis
from graspnet6d.experiment_analysis import (
    FIXTURE_SCOPE,
    INPUT_SCHEMA,
    AnalysisConfig,
    AnalysisInputError,
    FormalReportRefused,
    FrozenPoolError,
    SplitLeakageError,
    assert_frozen_prediction_pool,
    hard_target_support_scores,
    load_analysis_input_manifest,
    run_post_feature_analysis,
    validate_split_disjointness,
    write_formal_analysis_report,
)
from graspnet6d.io import sha256_file


_SCHEMA = {
    "schema_version": "fixture_6d_v1",
    "features": [
        {"name": "native_score", "group": "native_quality"},
        {"name": "native_rank", "group": "native_quality"},
        {"name": "score_to_top1", "group": "native_quality"},
        {"name": "center_mask_probability", "group": "semantic_mask"},
        {
            "name": "target_point_fraction_inside_closing_volume",
            "group": "semantic_mask",
        },
        {"name": "target_points_between_fingers", "group": "semantic_mask"},
        {"name": "gripper_width_m", "group": "width"},
        {"name": "rotation_6d_0", "group": "orientation"},
        {"name": "approach_vs_gravity_angle", "group": "orientation"},
        {"name": "collision_proxy_flag", "group": "collision_risk"},
        {"name": "local_point_density", "group": "local_geometry"},
    ],
}


def _partition_rows(
    partition: str, scene_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    for scene_index in range(scene_count):
        scene_id = f"{partition}-scene-{scene_index:02d}"
        group_id = f"{partition}-group-{scene_index:02d}"
        groups.append(
            {
                "partition": partition,
                "scene_id": scene_id,
                "group_id": group_id,
                "is_fixture": True,
            }
        )
        # Alternating native success/failure supplies a real paired fixture,
        # while every group retains exactly one officially derivable target grasp.
        good_index = 0 if scene_index % 2 else 1
        for candidate_index in range(4):
            candidate_id = f"{group_id}-candidate-{candidate_index}"
            native_score = 0.9 - 0.1 * candidate_index
            is_good = candidate_index == good_index
            records.append(
                {
                    "partition": partition,
                    "scene_id": scene_id,
                    "group_id": group_id,
                    "candidate_id": candidate_id,
                    "geometry_sha256": __import__("hashlib")
                    .sha256(candidate_id.encode("utf-8"))
                    .hexdigest(),
                    "native_rank": candidate_index + 1,
                    "native_score": native_score,
                    "collision": False,
                    "pose_valid": True,
                    "friction_required": 0.4 if is_good else 0.2,
                    "target_object_id": 1,
                    "associated_object_id": 1 if is_good else 9,
                    "relevance": 5 if is_good else 0,
                    "score_to_top1": 0.9 - native_score,
                    "center_mask_probability": 0.9 if is_good else 0.1,
                    "target_point_fraction_inside_closing_volume": 0.2
                    if is_good
                    else 0.0,
                    "target_points_between_fingers": 12.0 if is_good else 0.0,
                    "gripper_width_m": 0.05 + 0.001 * candidate_index,
                    "rotation_6d_0": float(candidate_index == 0),
                    "approach_vs_gravity_angle": 0.1 * candidate_index,
                    "collision_proxy_flag": float(candidate_index == 3),
                    "local_point_density": 100.0 + 10.0 * candidate_index,
                    "is_fixture": True,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(groups)


def _write_inputs(
    root: Path,
    *,
    status: str = "COMPLETE",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    schema_path = root / "feature_schema.json"
    schema_path.write_text(json.dumps(_SCHEMA), encoding="utf-8")
    partition_entries: dict[str, object] = {}
    for partition, count in (("train", 6), ("validation", 3), ("test", 3)):
        rows, universe = _partition_rows(partition, count)
        if partition == "test":
            universe = pd.concat(
                [
                    universe,
                    pd.DataFrame(
                        [
                            {
                                "partition": "test",
                                "scene_id": "test-scene-00",
                                "group_id": "test-empty-pool",
                                "is_fixture": True,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        rows_path = root / f"{partition}_rows.csv"
        universe_path = root / f"{partition}_universe.csv"
        rows.to_csv(rows_path, index=False)
        universe.to_csv(universe_path, index=False)
        partition_entries[partition] = {
            "rows": {"path": rows_path.name, "sha256": sha256_file(rows_path)},
            "group_universe": {
                "path": universe_path.name,
                "sha256": sha256_file(universe_path),
            },
        }
    manifest = {
        "schema_version": INPUT_SCHEMA,
        "run_id": "explicit-fixture-run",
        "status": status,
        "scope": FIXTURE_SCOPE,
        "fixture_only": True,
        "feature_schema": {
            "path": schema_path.name,
            "sha256": sha256_file(schema_path),
        },
        "partitions": partition_entries,
        "provenance": {
            "candidate_manifest_sha256": "a" * 64,
            "official_label_manifest_sha256": "b" * 64,
            "feature_extraction_manifest_sha256": "c" * 64,
        },
    }
    path = root / "input_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _fixture_config(*, attempt_gate_fit: bool = True) -> AnalysisConfig:
    return AnalysisConfig(
        config_grid=(
            {
                "num_leaves": 7,
                "learning_rate": 0.1,
                "n_estimators": 8,
                "min_child_samples": 1,
                "feature_fraction": 1.0,
            },
        ),
        early_stopping_rounds=2,
        bootstrap_iterations=50,
        attempt_gate_fit=attempt_gate_fit,
    )


def test_failure_taxonomy_distinguishes_grounding_terminal_from_vgn_empty_pool() -> (
    None
):
    group_ids = ["terminal-empty", "vgn-empty"]
    universe = pd.DataFrame(
        [
            {
                "partition": "test",
                "scene_id": "scene-terminal",
                "group_id": group_ids[0],
                "grounding_condition": "hifics_zero_shot_mask",
                "generation_status": "skipped_grounding_failure",
                "grounding_failure_reason": "empty_predicted_mask",
            },
            {
                "partition": "test",
                "scene_id": "scene-vgn",
                "group_id": group_ids[1],
                "grounding_condition": "hifics_zero_shot_mask",
                "generation_status": "completed_vgn_inference",
                "grounding_failure_reason": None,
            },
        ]
    )
    outcomes = pd.DataFrame(
        {
            "group_id": group_ids,
            "top1_success_mu_1.2": [False, False],
            "oracle_at_50_mu_1.2": [False, False],
        }
    )
    raw = pd.DataFrame(columns=["group_id", "collision"])
    table, _ = experiment_analysis._failure_taxonomy(
        raw,
        universe,
        outcomes,
        outcomes,
        scope=FIXTURE_SCOPE,
        config=_fixture_config(),
    )
    categories = table.set_index("group_id")["category"].to_dict()
    assert categories == {
        "terminal-empty": "F1_GROUNDING_FAILURE",
        "vgn-empty": "F3_CANDIDATE_GENERATION_FAILURE",
    }


def test_strict_manifest_checks_unknown_fields_status_and_content_hash(
    tmp_path: Path,
) -> None:
    path = _write_inputs(tmp_path / "inputs")
    loaded = load_analysis_input_manifest(path)
    assert loaded.fixture_only is True
    assert loaded.status == "COMPLETE"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(AnalysisInputError, match="unknown"):
        load_analysis_input_manifest(path)

    path = _write_inputs(tmp_path / "stale")
    rows_path = path.parent / "train_rows.csv"
    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(AnalysisInputError, match="stale train rows"):
        load_analysis_input_manifest(path)


def test_split_guards_reject_test_rows_in_train_and_all_overlap_levels() -> None:
    train, train_universe = _partition_rows("train", 2)
    validation, validation_universe = _partition_rows("validation", 2)
    test, test_universe = _partition_rows("test", 2)
    test.loc[0, "scene_id"] = train.loc[0, "scene_id"]
    with pytest.raises(SplitLeakageError, match="scene_id overlap"):
        validate_split_disjointness(
            {"train": train, "validation": validation, "test": test},
            {
                "train": train_universe,
                "validation": validation_universe,
                "test": test_universe,
            },
        )


def test_declared_test_partition_is_rejected_from_train_file(tmp_path: Path) -> None:
    manifest_path = _write_inputs(tmp_path / "inputs")
    train_path = manifest_path.parent / "train_rows.csv"
    train = pd.read_csv(train_path)
    train.loc[0, "partition"] = "test"
    train.to_csv(train_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partitions"]["train"]["rows"]["sha256"] = sha256_file(train_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SplitLeakageError, match="test rows may never enter train"):
        run_post_feature_analysis(
            manifest_path,
            tmp_path / "analysis",
            config=_fixture_config(attempt_gate_fit=False),
        )


def test_b1_is_label_free_deterministic_and_frozen_pool_guard_is_exact() -> None:
    rows, _ = _partition_rows("test", 1)
    scored = hard_target_support_scores(rows, _fixture_config(attempt_gate_fit=False))
    relabelled = rows.copy()
    relabelled["relevance"] = relabelled["relevance"].iloc[::-1].to_numpy()
    relabelled["associated_object_id"] = 777
    rescored = hard_target_support_scores(
        relabelled, _fixture_config(attempt_gate_fit=False)
    )
    pd.testing.assert_series_equal(scored["b1_eligible"], rescored["b1_eligible"])
    pd.testing.assert_series_equal(scored["b1_score"], rescored["b1_score"])
    assert scored.loc[scored["relevance"].eq(5), "b1_eligible"].all()
    assert not scored.loc[scored["relevance"].eq(0), "b1_eligible"].any()
    assert_frozen_prediction_pool(rows, scored, system="B1")
    reordered = scored.sample(frac=1.0, random_state=4)
    assert_frozen_prediction_pool(rows, reordered, system="reordered")
    with pytest.raises(FrozenPoolError, match="membership"):
        assert_frozen_prediction_pool(rows, scored.iloc[:-1], system="truncated")
    changed = scored.copy()
    changed.loc[0, "geometry_sha256"] = "f" * 64
    with pytest.raises(FrozenPoolError, match="geometry"):
        assert_frozen_prediction_pool(rows, changed, system="changed")


def test_fixture_end_to_end_outputs_are_hashed_resumable_and_not_formal(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path / "inputs")
    output = tmp_path / "analysis"
    result = run_post_feature_analysis(
        inputs, output, config=_fixture_config(), resume=False
    )
    assert result.resumed is False
    manifest = json.loads(
        (output / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "COMPLETE"
    assert manifest["fixture_only"] is True
    assert manifest["formal_report_eligible"] is False
    assert manifest["split_guards"]["test_used_for_training_or_selection"] is False
    pool_audit = json.loads((output / "frozen_pool_audit.json").read_text())
    assert pool_audit["status"] == "PASS"
    assert pool_audit["group_count"] == 4
    assert pool_audit["empty_group_count"] == 1

    native = pd.read_csv(output / "native_predictions.csv")
    reranked = pd.read_csv(output / "reranked_predictions.csv")
    assert set(native["candidate_id"]) == set(reranked["candidate_id"])
    assert len(reranked) == len(native)
    for seed in (20260815, 20260816, 20260817):
        assert f"raw_rerank_score_seed_{seed}" in reranked
    selection = json.loads((output / "model_selection.json").read_text())
    assert selection["selection_partition"] == "validation_only"
    assert selection["test_access_during_selection"] is False
    assert selection["seeds"] == [20260815, 20260816, 20260817]
    gate = json.loads((output / "metrics.json").read_text())["gate"]
    assert gate["old_4d_models_or_thresholds_reused"] is False
    assert gate["transition_training_scope"] == "train_oof_only"
    assert gate["operating_point_selection_scope"] == "validation_only"
    assert gate["test_outcomes_used_for_fit_or_selection"] is False
    assert gate["status"] in {"GO", "FAIL_CLOSED_NATIVE"}
    if gate["status"] == "FAIL_CLOSED_NATIVE":
        assert gate["reason_code"]
        assert gate["reason"]

    ablations = pd.read_csv(output / "ablation_results.csv")
    assert set(f"A{index}" for index in range(11)).issubset(set(ablations["ablation"]))
    a9_five = ablations.loc[
        ablations["ablation"].eq("A9") & ablations["setting"].eq("five_view")
    ]
    assert a9_five["status"].eq("UNEXECUTED_RESOURCE_DEPENDENT").all()
    assert a9_five["metric_name"].isna().all()
    assert a9_five["metric_value"].isna().all()
    taxonomy = pd.read_csv(output / "failure_taxonomy.csv")
    assert len(taxonomy) == 4
    assert taxonomy["group_id"].nunique() == 4
    assert (
        taxonomy.loc[taxonomy["group_id"].eq("test-empty-pool"), "category"].item()
        == "F3_CANDIDATE_GENERATION_FAILURE"
    )
    assert taxonomy["category"].notna().all()
    with pytest.raises(FormalReportRefused, match="formal_real_data"):
        write_formal_analysis_report(result)
    assert not (output / "RESULTS.md").exists()

    resumed = run_post_feature_analysis(
        inputs, output, config=_fixture_config(), resume=True
    )
    assert resumed.resumed is True
    assert resumed.analysis_fingerprint == result.analysis_fingerprint
    metrics_path = output / "metrics.csv"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(AnalysisInputError, match="missing or stale"):
        run_post_feature_analysis(inputs, output, config=_fixture_config(), resume=True)


def test_noncomplete_input_is_rejected_before_any_model_or_result(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path / "inputs", status="BLOCKED")
    output = tmp_path / "analysis"
    with pytest.raises(AnalysisInputError, match="requires upstream status COMPLETE"):
        run_post_feature_analysis(inputs, output, config=_fixture_config())
    assert not (output / "metrics.csv").exists()


def test_locked_seed_and_formal_bootstrap_guards() -> None:
    with pytest.raises(AnalysisInputError, match="training seeds are locked"):
        AnalysisConfig(seeds=(1, 2, 3)).validated(scope=FIXTURE_SCOPE)
    trial = dict(_fixture_config().config_grid[0])
    second = {**trial, "num_leaves": 9}
    with pytest.raises(AnalysisInputError, match="10,000"):
        AnalysisConfig(config_grid=(trial, second), bootstrap_iterations=50).validated(
            scope="formal_real_data"
        )
