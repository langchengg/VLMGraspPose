from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from scripts.run_vgn_sim_benchmark import (
    build_blocked_aggregate,
    official_retention_success,
)
from scripts.summarize_real_robot_trials import summarize_real_robot_records
from src.experiments.bootstrap import (
    bootstrap_experiment,
    cluster_bootstrap_interval,
    select_scene_cluster_bootstrap,
    validate_truthfulness_metadata,
)
from src.experiments.experiment_store import ExperimentStore, ManifestCountMismatch
from src.experiments.failure_taxonomy import classify_status, is_retryable, is_terminal
from src.experiments.metrics import aggregate_metrics, export_metrics, wilson_interval


def _truth_metadata() -> dict[str, object]:
    return {
        "repository_url": "https://github.com/ethz-asl/vgn",
        "repository_branch": "corl2020",
        "repository_commit": "d7af0622433f52ae88ebe81533f12b46b33e951a",
        "checkpoint_sha256": (
            "ba3391d0805e9c9b178cd18106866313cee808ff2b654f689663e92a814cec4b"
        ),
        "tsdf_mode": "single_view_adaptation",
        "score_source": "official_vgn_processed_quality",
        "custom_reranking": False,
        "limitations": [
            "single-view TSDF adaptation",
            "no 6-DoF ground truth in OCID-VLG",
            "no robot execution validation",
        ],
    }


def _manifest(count: int = 3) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{index}",
            "dataset_index": index,
            "scene_id": f"scene-{index // 2}",
            "instruction": f"pick object {index}",
            "view": "top",
        }
        for index in range(count)
    ]


def _metric_rows() -> list[dict[str, object]]:
    truth = {
        "score_source": "official_vgn_processed_quality",
        "custom_reranking": False,
        "tsdf_mode": "single_view_adaptation",
    }
    return [
        {
            "sample_id": "a",
            "scene_id": "scene-1",
            "status": "ok",
            "official_candidate_count": 7,
            "target_candidate_count": 2,
            "top1_vgn_quality": 0.97,
            "top1_width_m": 0.04,
            "support_plane_residual": 0.003,
            **truth,
        },
        {
            "sample_id": "b",
            "scene_id": "scene-1",
            "status": "no_target_grasp",
            "official_candidate_count": 3,
            "target_candidate_count": 0,
            "support_plane_residual": 0.004,
            **truth,
        },
        {
            "sample_id": "c",
            "scene_id": "scene-2",
            "status": "no_official_grasp",
            "official_candidate_count": 0,
            "target_candidate_count": 0,
            "support_plane_residual": 0.005,
            **truth,
        },
        {
            "sample_id": "d",
            "scene_id": "scene-3",
            "status": "missing_camera_intrinsics",
            "failure_reason": "calibration unavailable",
        },
    ]


def test_sqlite_wal_atomic_claim_and_terminal_state(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    first = ExperimentStore(database, "run-a")
    first.initialize_run(_truth_metadata(), 3)
    first.register_samples(_manifest(), expected_count=3)
    second = ExperimentStore(database, "run-a")
    try:
        assert first.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        claim_a = first.claim_next("worker-a", lease_seconds=10, now=100)
        claim_b = second.claim_next("worker-b", lease_seconds=10, now=100)
        assert claim_a is not None and claim_b is not None
        assert claim_a["sample_id"] == "sample-0"
        assert claim_b["sample_id"] == "sample-1"

        first.complete_sample(
            "sample-0",
            "no_target_grasp",
            result={
                "official_candidate_count": 4,
                "target_candidate_count": 0,
                "score_source": "official_vgn_processed_quality",
                "custom_reranking": False,
                "tsdf_mode": "single_view_adaptation",
            },
            worker_id="worker-a",
            now=101,
        )
        completed = first.get_sample("sample-0")
        assert completed is not None
        assert completed["state"] == "terminal"
        assert completed["status"] == "no_target_grasp"
        with pytest.raises(RuntimeError, match="already terminal"):
            first.complete_sample("sample-0", "ok")

        assert second.recover_expired_claims(now=111) == 1
        recovered = second.get_sample("sample-1")
        assert recovered is not None
        assert recovered["state"] == "pending"
        assert recovered["failure_reason"].startswith("worker_lost")
    finally:
        second.close()
        first.close()


def test_manifest_count_guard_is_strict(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "state.sqlite3", "run-count") as store:
        store.initialize_run(_truth_metadata(), 3)
        with pytest.raises(ManifestCountMismatch, match="exactly 3"):
            store.register_samples(_manifest(2), expected_count=3)
        with pytest.raises(ManifestCountMismatch, match="expects 3"):
            store.initialize_run(_truth_metadata(), 4)


def test_failure_taxonomy_terminal_and_retryable_semantics() -> None:
    assert is_terminal("ok")
    assert is_terminal("no_official_grasp")
    assert is_terminal("missing_camera_intrinsics")
    assert not is_retryable("support_plane_failed")
    assert is_retryable("vgn_inference_failed")
    unexpected = classify_status("new_driver_failure")
    assert unexpected.category == "unexpected_failure"
    assert unexpected.retryable and not unexpected.terminal


def test_wilson_and_denominators_are_explicit() -> None:
    interval = wilson_interval(2, 3)
    assert interval["numerator"] == 2
    assert interval["denominator"] == 3
    assert interval["ci_lower"] < interval["estimate"] < interval["ci_upper"]

    aggregate = aggregate_metrics(
        _metric_rows(), manifest_count=4, bootstrap_replicates=40, seed=17
    )
    proportions = aggregate["proportions"]
    assert proportions["manifest_processing_coverage"]["denominator"] == 4
    assert proportions["manifest_processing_coverage"]["numerator"] == 4
    assert proportions["official_candidate_availability"]["denominator"] == 3
    assert proportions["official_candidate_availability"]["numerator"] == 2
    assert proportions["target_candidate_availability"]["denominator"] == 3
    assert proportions["target_candidate_availability"]["numerator"] == 1
    assert proportions["target_given_official_availability"]["denominator"] == 2
    assert proportions["target_given_official_availability"]["numerator"] == 1


def test_scene_cluster_bootstrap_is_deterministic_and_clustered() -> None:
    rows = [
        {"sample_id": f"{scene}-{item}", "scene_id": scene, "value": value}
        for scene, value in (("a", 1.0), ("b", 3.0), ("c", 5.0), ("d", 7.0))
        for item in range(2)
    ]
    selected_a, clusters_a = select_scene_cluster_bootstrap(
        rows, seed=42, replicate_index=5
    )
    selected_b, clusters_b = select_scene_cluster_bootstrap(
        rows, seed=42, replicate_index=5
    )
    assert clusters_a == clusters_b
    assert selected_a == selected_b
    for scene in set(clusters_a):
        expected_repetitions = clusters_a.count(scene) * 2
        assert sum(row["scene_id"] == scene for row in selected_a) == expected_repetitions

    interval_a = cluster_bootstrap_interval(rows, "value", replicates=100, seed=7)
    interval_b = cluster_bootstrap_interval(rows, "value", replicates=100, seed=7)
    assert interval_a == interval_b
    assert interval_a["cluster_count"] == 4


def test_bootstrap_preserves_truthfulness_fields(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "state.sqlite3", "truth-run") as store:
        result = bootstrap_experiment(store, _manifest(), _truth_metadata())
        assert result["manifest_count"] == 3
        assert result["scene_cluster_count"] == 2
        metadata = store.get_run()["metadata"]
        assert metadata["score_source"] == "official_vgn_processed_quality"
        assert metadata["custom_reranking"] is False
        assert metadata["tsdf_mode"] == "single_view_adaptation"
        assert len(metadata["manifest_identity_sha256"]) == 64

    invalid = _truth_metadata()
    invalid["custom_reranking"] = True
    with pytest.raises(ValueError, match="exactly false"):
        validate_truthfulness_metadata(invalid)


def test_per_sample_and_aggregate_exports_are_truthful(tmp_path: Path) -> None:
    exported = export_metrics(
        _metric_rows(),
        tmp_path,
        manifest_count=4,
        bootstrap_replicates=25,
        seed=11,
        require_parquet=importlib.util.find_spec("pyarrow") is not None,
    )
    assert exported["csv_path"].is_file()
    assert exported["aggregate_path"].is_file()
    serialized = exported["aggregate_path"].read_text(encoding="utf-8")
    assert "success" + "_rate" not in serialized.lower()
    aggregate = json.loads(serialized)
    truth = aggregate["truthfulness"]
    assert truth["all_scores_from_official_processed_quality"] is True
    assert truth["any_custom_reranking"] is False
    assert truth["all_candidate_outcomes_disable_custom_reranking"] is True
    assert truth["all_candidate_outcomes_disclose_single_view_adaptation"] is True

    parquet_path = exported["parquet_path"]
    if importlib.util.find_spec("pyarrow") is not None:
        assert parquet_path is not None and parquet_path.is_file()
        assert parquet_path.read_bytes()[:4] == b"PAR1"
    else:
        assert parquet_path is None
        assert exported["parquet_error"]


def test_metric_rejects_impossible_target_candidate_count() -> None:
    row = _metric_rows()[0]
    row["official_candidate_count"] = 1
    row["target_candidate_count"] = 2
    with pytest.raises(ValueError, match="more target candidates"):
        aggregate_metrics([row], manifest_count=1, bootstrap_replicates=0)


def test_sim_success_requires_retained_object() -> None:
    assert not official_retention_success(0, 0.04)
    assert not official_retention_success(1, 0.008)
    assert official_retention_success(1, 0.0081)


def test_simulation_and_real_robot_metrics_are_separate() -> None:
    aggregate = build_blocked_aggregate(
        {"blockers": [{"code": "missing_pybullet"}]}, {"seed": 42}
    )
    assert aggregate["simulated_grasp_success_rate"] is None
    assert aggregate["real_robot_metrics_not_computed_here"] is True
    assert "real_robot_grasp_success_rate" not in aggregate


def test_no_physical_logs_returns_real_robot_null() -> None:
    summary = summarize_real_robot_records([], planned_trial_count=100)
    assert summary["real_robot_grasp_success_rate"] is None
    assert summary["end_to_end_real_success_rate"] is None
    assert summary["reason"] == "no physical robot execution logs"


def test_simulation_result_cannot_substitute_for_physical_log() -> None:
    summary = summarize_real_robot_records(
        [
            {
                "physical_execution_attempted": False,
                "simulated_grasp_success_rate": 1.0,
                "physical_success": True,
            }
        ]
    )
    assert summary["real_robot_grasp_success_rate"] is None
    assert summary["simulation_substitution"] is False
