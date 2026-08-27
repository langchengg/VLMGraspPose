from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from robustness_completion.common import sha256_file
from robustness_completion.runtime_6d import validate_source_contract


REPO = Path(__file__).resolve().parents[2]
RUN_DIR = (
    REPO
    / "artifacts"
    / "robustness_completion"
    / "20260822_194405_remaining_robustness"
)
RUNTIME_DIR = RUN_DIR / "runtime_full"

COMPLETE_VARIANTS = {("6D_ORACLE", "native"), ("6D_ADAPTED", "native")}
FAILED_4D_ROUTES = {"CROG", "G1", "C1"}

EXPECTED_STAGES = {
    "CROG": {
        "input_io",
        "preprocess",
        "crog_model_inference",
        "candidate_decode",
        "nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    },
    "G1": {
        "input_io",
        "preprocess",
        "hifics_inference",
        "mask_postprocess",
        "grasp_backend_inference",
        "candidate_decode",
        "nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    },
    "C1": {
        "input_io",
        "preprocess",
        "hifics_inference",
        "mask_postprocess",
        "grasp_backend_inference",
        "candidate_decode",
        "nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    },
    "D1": {
        "input_io",
        "preprocess",
        "hifics_inference",
        "mask_postprocess",
        "grasp_backend_inference",
        "candidate_decode",
        "nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    },
    "6D_ORACLE": {
        "input_io",
        "gt_mask_load",
        "depth_backprojection",
        "workspace_construction",
        "tsdf_integration",
        "vgn_inference",
        "candidate_decode",
        "pose_nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    },
    "6D_ADAPTED": {
        "input_io",
        "hifics_inference",
        "mask_postprocess",
        "depth_backprojection",
        "workspace_construction",
        "tsdf_integration",
        "vgn_inference",
        "candidate_decode",
        "pose_nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def parity() -> pd.DataFrame:
    return pd.read_csv(RUNTIME_DIR / "route_parity.csv")


@pytest.fixture(scope="module")
def cold() -> pd.DataFrame:
    return pd.read_csv(RUNTIME_DIR / "cold_start_raw.csv")


@pytest.fixture(scope="module")
def disk() -> pd.DataFrame:
    return pd.read_parquet(RUNTIME_DIR / "warm_disk_raw.parquet")


@pytest.fixture(scope="module")
def preloaded() -> pd.DataFrame:
    return pd.read_parquet(RUNTIME_DIR / "warm_preloaded_raw.parquet")


@pytest.fixture(scope="module")
def stages() -> pd.DataFrame:
    return pd.read_parquet(RUNTIME_DIR / "stage_timings.parquet")


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(RUNTIME_DIR / "runtime_summary.csv")


def _complete(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["complete_deployment"].eq(True)].copy()


def _blocked_timing_rows(
    frames: tuple[pd.DataFrame, ...], route: str, method: str
) -> None:
    for frame in frames:
        selected = frame[(frame["route"] == route) & (frame["method"] == method)]
        assert len(selected), f"missing fail-closed row for {route}/{method}"
        assert not selected["complete_deployment"].eq(True).any()
        if "whole_elapsed_ns" in selected:
            assert selected["whole_elapsed_ns"].isna().all()
        if "process_spawn_to_output_ns" in selected:
            assert selected["process_spawn_to_output_ns"].isna().all()


def _combined_members(value: Any) -> set[str]:
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return set()
    parsed = json.loads(value) if isinstance(value, str) else value
    return {str(item) for item in parsed}


def test_route_adapter_uses_real_checkpoint() -> None:
    # The live 4D parity workers loaded the immutable formal checkpoints before
    # failing numerical parity.  Re-hash every unique model checkpoint named by
    # their preflight evidence; merely checking a filename would be insufficient.
    checkpoint_records: dict[Path, str] = {}
    for route in ("crog", "g1", "c1"):
        payload = _read_json(
            RUNTIME_DIR / "workers" / "4d" / route / "parity.json"
        )
        preflight = payload["asset_preflight"]
        assert preflight["status"] == "PASS"
        assert int(payload["startup_model_load_ns"]) > 0
        route_checkpoints = [
            record
            for record in preflight["records"]
            if str(record["label"]).endswith("_checkpoint")
        ]
        assert route_checkpoints
        for record in route_checkpoints:
            path = Path(record["path"])
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size == int(record["bytes"])
            checkpoint_records[path] = str(record["sha256"])

    contract = validate_source_contract(REPO)
    six_d_paths = {
        contract.vgn_checkpoint: contract.hashes["vgn_checkpoint"],
        contract.base_hifi_checkpoint: contract.hashes["base_hifi_checkpoint"],
        contract.adapted_checkpoint: contract.hashes["adapted_checkpoint"],
        contract.clip_checkpoint: contract.hashes["clip_checkpoint"],
    }
    checkpoint_records.update(six_d_paths)
    for path, expected_hash in checkpoint_records.items():
        assert sha256_file(path) == expected_hash

    for route in ("oracle", "adapted"):
        status = _read_json(
            RUNTIME_DIR / "workers" / "6d" / route / "native" / "route_status.json"
        )
        assert status["complete_deployment"] is True
        assert int(status["model_load_ns"]) > 0
        assert status["checkpoint_hashes"]
        assert int(status["checkpoint_bytes"]) > 0


def test_route_adapter_does_not_load_candidate_cache_in_full_mode(
    cold: pd.DataFrame, disk: pd.DataFrame, preloaded: pd.DataFrame
) -> None:
    for route in ("oracle", "adapted"):
        worker_dir = RUNTIME_DIR / "workers" / "6d" / route / "native"
        result_paths = [
            *(worker_dir.glob("cold_result_*.json")),
            worker_dir / "warm_disk_result.json",
            worker_dir / "warm_preloaded_result.json",
        ]
        assert len(result_paths) == 7
        for path in result_paths:
            payload = _read_json(path)
            assert payload["complete_deployment"] is True
            assert payload["cache_disabled"] is True
            assert payload["candidate_cache_used_for_timing"] is False
            assert payload["score_cache_used_for_timing"] is False

    # 4D caches were allowed only after live execution for parity comparison;
    # failed routes contain no deployment timings that could be mislabeled.
    for route in ("crog", "g1", "c1"):
        payload = _read_json(
            RUNTIME_DIR / "workers" / "4d" / route / "parity.json"
        )
        assert payload["cache_use"] == "parity_comparison_only"
        assert payload["asset_preflight"]["candidate_feature_score_caches_read"] is False
        assert payload["timings"] == []
        assert payload["complete_deployment"] is False
    for route in FAILED_4D_ROUTES:
        for method in ("native", "raw", "gated"):
            _blocked_timing_rows((cold, disk, preloaded), route, method)


def test_route_adapter_does_not_load_feature_cache_in_full_mode(
    cold: pd.DataFrame, disk: pd.DataFrame, preloaded: pd.DataFrame
) -> None:
    for route in ("oracle", "adapted"):
        worker_dir = RUNTIME_DIR / "workers" / "6d" / route / "native"
        result_paths = [
            *(worker_dir.glob("cold_result_*.json")),
            worker_dir / "warm_disk_result.json",
            worker_dir / "warm_preloaded_result.json",
        ]
        for path in result_paths:
            payload = _read_json(path)
            assert payload["feature_cache_used_for_timing"] is False
    for route in ("6D_ORACLE", "6D_ADAPTED"):
        _blocked_timing_rows((cold, disk, preloaded), route, "raw")


def test_all_required_stages_present(stages: pd.DataFrame) -> None:
    contracts = _read_json(RUNTIME_DIR / "adapter_preflight.json")[
        "route_contracts"
    ]
    for route in ("CROG", "G1", "C1", "D1"):
        assert set(contracts["4d"][route]["required_stages"]) == EXPECTED_STAGES[route]
    for source_route, reported_route in (
        ("oracle", "6D_ORACLE"),
        ("adapted", "6D_ADAPTED"),
    ):
        assert (
            set(contracts["6d"][source_route]["required_stages"])
            == EXPECTED_STAGES[reported_route]
        )

    # Only parity-passing variants must have measured stage records.  Combined
    # formal APIs explicitly cover their non-separable member stages, while
    # native-only reranker/gate/feature stages may be present as null N/A rows.
    for route, method in COMPLETE_VARIANTS:
        selected = stages[(stages["route"] == route) & (stages["method"] == method)]
        for (_mode, _sample_id), group in selected.groupby(["mode", "sample_id"]):
            covered = set(group["stage"].astype(str))
            for members in group["combined_members"]:
                covered.update(_combined_members(members))
            assert EXPECTED_STAGES[route] <= covered

    blocked = stages[
        ~stages.set_index(["route", "method"]).index.isin(COMPLETE_VARIANTS)
    ]
    assert set(blocked["stage"]) == {"not_measured"}
    assert blocked["elapsed_ns"].isna().all()


def test_stage_timers_positive(
    stages: pd.DataFrame, disk: pd.DataFrame, preloaded: pd.DataFrame
) -> None:
    complete_stages = stages[
        stages.set_index(["route", "method"]).index.isin(COMPLETE_VARIANTS)
    ]
    measured = complete_stages[
        complete_stages["stage_status"].isin({"measured", "combined_formal_api"})
    ]
    assert len(measured)
    assert measured["elapsed_ns"].notna().all()
    assert measured["elapsed_ns"].gt(0).all()

    not_timed = complete_stages[
        ~complete_stages["stage_status"].isin({"measured", "combined_formal_api"})
    ]
    assert set(not_timed["stage_status"]) <= {
        "not_applicable_native",
        "preloaded_outside_timed_region",
    }
    assert not_timed["elapsed_ns"].isna().all()
    for frame in (disk, preloaded):
        actual = _complete(frame)
        assert set(zip(actual["route"], actual["method"], strict=True)) == COMPLETE_VARIANTS
        assert actual["whole_elapsed_ns"].gt(0).all()


def test_cold_start_includes_checkpoint_load(cold: pd.DataFrame) -> None:
    measured = _complete(cold)
    assert len(measured) == 10
    assert set(zip(measured["route"], measured["method"], strict=True)) == COMPLETE_VARIANTS
    for column in (
        "process_spawn_to_output_ns",
        "model_load_ns",
        "first_complete_output_ns",
        "cold_inner_ns",
    ):
        assert measured[column].notna().all()
        assert measured[column].gt(0).all()
    assert np.allclose(
        measured["cold_inner_ns"],
        measured["model_load_ns"] + measured["first_complete_output_ns"],
        rtol=0.0,
        atol=1.0,
    )
    assert (
        measured["process_spawn_to_output_ns"] >= measured["cold_inner_ns"]
    ).all()

    blocked = cold[~cold["complete_deployment"].eq(True)]
    assert blocked["process_spawn_to_output_ns"].isna().all()
    assert blocked["model_load_ns"].isna().all()


def test_offline_evaluator_excluded_from_deployment(
    stages: pd.DataFrame, summary: pd.DataFrame
) -> None:
    assert summary["offline_evaluator_excluded"].eq(True).all()
    forbidden = {
        "offline_evaluator",
        "official_evaluator",
        "ground_truth_evaluator",
        "metric_aggregation",
        "bootstrap",
    }
    assert forbidden.isdisjoint(set(stages["stage"].astype(str)))


def test_candidate_membership_matches_formal_route(
    parity: pd.DataFrame,
    cold: pd.DataFrame,
    disk: pd.DataFrame,
    preloaded: pd.DataFrame,
) -> None:
    for route_name, route_path in (
        ("6D_ORACLE", "oracle"),
        ("6D_ADAPTED", "adapted"),
    ):
        details = pd.read_parquet(
            RUNTIME_DIR
            / "workers"
            / "6d"
            / route_path
            / "native"
            / "parity_results.parquet"
        )
        assert len(details) == 20
        assert details["candidate_ids_equal"].eq(True).all()
        assert (
            details["candidate_count_actual"] == details["candidate_count_frozen"]
        ).all()
        row = parity[(parity["route"] == route_name) & (parity["method"] == "native")]
        assert len(row) == 1 and bool(row.iloc[0]["candidate_parity"])

    # The live batch-1 4D candidates did not match the locked route.  This is a
    # tested blocker, not a parity pass or a reason to time a different route.
    for route in FAILED_4D_ROUTES:
        row = parity[(parity["route"] == route) & (parity["method"] == "native")]
        assert len(row) == 1
        assert row.iloc[0]["status"] == "FAILED_PARITY"
        assert not bool(row.iloc[0]["candidate_parity"])
        _blocked_timing_rows((cold, disk, preloaded), route, "native")


def test_native_top1_parity(
    parity: pd.DataFrame,
    cold: pd.DataFrame,
    disk: pd.DataFrame,
    preloaded: pd.DataFrame,
) -> None:
    passed = parity[
        (parity["method"] == "native") & (parity["status"] == "PASS")
    ]
    assert set(passed["route"]) == {"6D_ORACLE", "6D_ADAPTED"}
    assert passed["parity_n"].eq(20).all()
    assert passed["native_top1_parity"].eq(True).all()
    assert passed["final_top1_parity"].eq(True).all()

    for route in FAILED_4D_ROUTES:
        row = parity[(parity["route"] == route) & (parity["method"] == "native")]
        assert row["native_top1_parity"].eq(False).all()
        _blocked_timing_rows((cold, disk, preloaded), route, "native")


def test_reranked_top1_parity(
    parity: pd.DataFrame,
    cold: pd.DataFrame,
    disk: pd.DataFrame,
    preloaded: pd.DataFrame,
) -> None:
    raw = parity[parity["method"] == "raw"]
    assert len(raw) == 6
    assert not raw["status"].eq("PASS").any()
    assert set(raw.loc[raw["route"].isin(FAILED_4D_ROUTES), "status"]) == {
        "FAILED_PARITY"
    }
    assert raw.loc[
        raw["route"].isin(FAILED_4D_ROUTES), "reranked_top1_parity"
    ].eq(False).all()
    assert set(
        raw.loc[raw["route"].str.startswith("6D_"), "blocker_code"]
    ) == {"MISSING_FORMAL_CHECKPOINT"}
    for route in raw["route"]:
        _blocked_timing_rows((cold, disk, preloaded), str(route), "raw")


def test_gate_decision_parity(
    parity: pd.DataFrame,
    cold: pd.DataFrame,
    disk: pd.DataFrame,
    preloaded: pd.DataFrame,
) -> None:
    gated = parity[parity["method"] == "gated"]
    assert set(gated["route"]) == {"CROG", "G1", "C1", "D1"}
    assert not gated["status"].eq("PASS").any()
    four_d = gated[gated["route"].isin(FAILED_4D_ROUTES)]
    assert four_d["gate_parity"].eq(False).all()
    assert four_d["final_top1_parity"].eq(False).all()
    d1 = gated[gated["route"] == "D1"].iloc[0]
    assert d1["status"] == "NOT_EXECUTED_IMPLEMENTATION_INCOMPLETE"
    assert pd.isna(d1["gate_parity"])
    for route in gated["route"]:
        _blocked_timing_rows((cold, disk, preloaded), str(route), "gated")


def test_pose_parity_within_tolerance(
    cold: pd.DataFrame, disk: pd.DataFrame, preloaded: pd.DataFrame
) -> None:
    for route in ("oracle", "adapted"):
        details = pd.read_parquet(
            RUNTIME_DIR
            / "workers"
            / "6d"
            / route
            / "native"
            / "parity_results.parquet"
        )
        assert details["max_translation_error_m"].le(1e-5).all()
        assert details["max_rotation_error_rad"].le(1e-4).all()
        assert details["max_width_error_m"].le(1e-5).all()
        assert details["passed"].eq(True).all()

    for route in ("crog", "g1", "c1"):
        payload = _read_json(
            RUNTIME_DIR / "workers" / "4d" / route / "parity.json"
        )
        maxima, tolerances = payload["maxima"], payload["tolerances"]
        assert payload["candidate_parity"] is False
        assert any(
            float(maxima[name]) > float(tolerances[name])
            for name in ("center_px", "angle_rad", "width_px", "height_px")
        )
        assert payload["timings"] == []
        _blocked_timing_rows((cold, disk, preloaded), route.upper(), "native")


def test_runtime_summary_matches_raw_timings(
    summary: pd.DataFrame,
    cold: pd.DataFrame,
    disk: pd.DataFrame,
    preloaded: pd.DataFrame,
) -> None:
    for route, method in COMPLETE_VARIANTS:
        row = summary[
            (summary["route"] == route) & (summary["method"] == method)
        ].iloc[0]
        cold_values = _complete(cold[
            (cold["route"] == route) & (cold["method"] == method)
        ])["process_spawn_to_output_ns"].to_numpy(float) / 1e6
        disk_values = _complete(disk[
            (disk["route"] == route) & (disk["method"] == method)
        ])["whole_elapsed_ns"].to_numpy(float) / 1e6
        preloaded_values = _complete(preloaded[
            (preloaded["route"] == route) & (preloaded["method"] == method)
        ])["whole_elapsed_ns"].to_numpy(float) / 1e6
        assert int(row["cold_n"]) == len(cold_values) == 5
        assert int(row["warm_disk_n"]) == len(disk_values) == 98
        assert int(row["warm_preloaded_n"]) == len(preloaded_values) == 98
        assert row["cold_start_median_ms"] == pytest.approx(np.median(cold_values))
        assert row["cold_start_min_ms"] == pytest.approx(np.min(cold_values))
        assert row["cold_start_max_ms"] == pytest.approx(np.max(cold_values))
        assert row["cold_start_p95_ms"] == pytest.approx(np.percentile(cold_values, 95))
        assert row["warm_disk_median_ms"] == pytest.approx(np.median(disk_values))
        assert row["warm_disk_p95_ms"] == pytest.approx(np.percentile(disk_values, 95))
        assert row["warm_preloaded_median_ms"] == pytest.approx(
            np.median(preloaded_values)
        )
        assert row["warm_preloaded_p95_ms"] == pytest.approx(
            np.percentile(preloaded_values, 95)
        )

    blocked = summary[~summary["complete_deployment"].eq(True)]
    for column in (
        "cold_start_median_ms",
        "warm_disk_median_ms",
        "warm_preloaded_median_ms",
        "reranker_gate_overhead_ms",
    ):
        assert blocked[column].isna().all()


def test_instrumentation_overhead_reported(
    stages: pd.DataFrame, summary: pd.DataFrame
) -> None:
    for route, method in COMPLETE_VARIANTS:
        selected = stages[
            (stages["route"] == route) & (stages["method"] == method)
        ]
        for mode in ("warm-disk", "warm-preloaded"):
            mode_rows = selected[selected["mode"] == mode]
            overheads: list[float] = []
            for _sample_id, group in mode_rows.groupby("sample_id"):
                instrumented = group["instrumented_wall_ns"].dropna().unique()
                whole = group["whole_elapsed_ns"].dropna().unique()
                overhead = group["instrumentation_overhead_ns"].dropna().unique()
                assert len(instrumented) == len(whole) == len(overhead) == 1
                assert int(group["elapsed_ns"].dropna().sum()) == int(instrumented[0])
                assert int(overhead[0]) == int(instrumented[0]) - int(whole[0])
                overheads.append(float(overhead[0]))
            assert len(overheads) == 98

            if mode == "warm-disk":
                row = summary[
                    (summary["route"] == route) & (summary["method"] == method)
                ].iloc[0]
                assert row["instrumentation_overhead_median_ns"] == pytest.approx(
                    np.median(overheads)
                )
                assert row["instrumentation_overhead_p95_ns"] == pytest.approx(
                    np.percentile(overheads, 95)
                )

    blocked = summary[~summary["complete_deployment"].eq(True)]
    assert blocked["instrumentation_overhead_median_ns"].isna().all()
    assert blocked["instrumentation_overhead_p95_ns"].isna().all()
