from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from robustness_completion import runtime_aggregation as aggregation
from robustness_completion.reporting import _canonical_parity


REPO = Path(__file__).resolve().parents[2]
RUN = (
    REPO
    / "artifacts/robustness_completion/20260822_194405_remaining_robustness"
)
RUNTIME = RUN / "runtime_full"


@pytest.fixture(scope="module")
def aggregate_frames() -> dict[str, object]:
    subset = aggregation._read_json(RUNTIME / "profile_subset_manifest.json")
    parity, parity_status = aggregation._parity_table(RUNTIME, REPO, subset)
    monitors = aggregation._monitor_index(RUNTIME, REPO)
    cold = aggregation._cold_table(
        RUNTIME, REPO, parity, parity_status, monitors
    )
    disk, preloaded, stages = aggregation._warm_tables(
        RUNTIME, REPO, subset, parity, parity_status
    )
    memory, _ = aggregation._memory_samples(
        RUNTIME, REPO, parity, parity_status, cold, disk, preloaded
    )
    memory_summary = aggregation._memory_summary(
        RUNTIME,
        memory,
        cold,
        disk,
        preloaded,
        parity,
        parity_status,
    )
    summary = aggregation._runtime_summary_table(
        cold,
        disk,
        preloaded,
        stages,
        parity,
        memory_summary,
        parity_status,
    )
    return {
        "subset": subset,
        "parity": parity,
        "cold": cold,
        "disk": disk,
        "preloaded": preloaded,
        "stages": stages,
        "memory": memory,
        "memory_summary": memory_summary,
        "summary": summary,
    }


def test_route_parity_schema_is_reporter_compatible(
    aggregate_frames: dict[str, object],
) -> None:
    parity = aggregate_frames["parity"]
    assert isinstance(parity, pd.DataFrame)
    assert len(parity) == 16
    assert parity["n"].equals(parity["parity_n"])
    canonical = _canonical_parity(parity)
    native_6d = canonical[
        (canonical.route == "6D_ORACLE") & (canonical.method == "native")
    ].iloc[0]
    assert native_6d["n"] == 20
    assert native_6d["status"]


def test_failed_4d_parity_has_no_formal_timing(
    aggregate_frames: dict[str, object],
) -> None:
    parity = aggregate_frames["parity"]
    cold = aggregate_frames["cold"]
    disk = aggregate_frames["disk"]
    preloaded = aggregate_frames["preloaded"]
    assert isinstance(parity, pd.DataFrame)
    for route in ("CROG", "G1", "C1"):
        assert set(parity.loc[parity.route == route, "status"]) == {
            "FAILED_PARITY"
        }
        for frame in (cold, disk, preloaded):
            selected = frame[frame.route == route]
            assert not selected.complete_deployment.eq(True).any()
            assert selected["whole_elapsed_ns"].isna().all() if "whole_elapsed_ns" in selected else True
            if "process_spawn_to_output_ns" in selected:
                assert selected["process_spawn_to_output_ns"].isna().all()


def test_native_6d_exact_profile_counts(
    aggregate_frames: dict[str, object],
) -> None:
    cold = aggregate_frames["cold"]
    disk = aggregate_frames["disk"]
    preloaded = aggregate_frames["preloaded"]
    subset = aggregate_frames["subset"]
    expected = set(subset["profile_6d"]["group_ids"])
    for route in ("6D_ORACLE", "6D_ADAPTED"):
        assert len(
            cold[
                (cold.route == route)
                & (cold.method == "native")
                & cold.complete_deployment.eq(True)
            ]
        ) == 5
        for frame in (disk, preloaded):
            selected = frame[
                (frame.route == route)
                & (frame.method == "native")
                & frame.complete_deployment.eq(True)
            ]
            assert len(selected) == 98
            assert set(selected.sample_id) == expected
            assert selected.whole_elapsed_ns.gt(0).all()
            assert selected.feature_bytes.eq(0).all()


def test_native_6d_does_not_time_reranking_features(
    aggregate_frames: dict[str, object],
) -> None:
    stages = aggregate_frames["stages"]
    assert isinstance(stages, pd.DataFrame)
    for route in ("6D_ORACLE", "6D_ADAPTED"):
        selected = stages[
            (stages.route == route)
            & (stages.method == "native")
            & stages["mode"].isin(["warm-disk", "warm-preloaded"])
        ]
        feature = selected[
            selected.stage.astype(str) == "runtime_feature_extraction"
        ]
        assert len(feature) == 196
        assert feature.elapsed_ns.isna().all()
        assert feature.stage_status.eq("not_applicable_native").all()


def test_blocked_variants_remain_null_not_zero(
    aggregate_frames: dict[str, object],
) -> None:
    summary = aggregate_frames["summary"]
    blocked = summary[
        ((summary.route == "D1") | (summary.method == "raw"))
        & ~summary.complete_deployment.eq(True)
    ]
    assert len(blocked) >= 5
    for column in (
        "cold_start_median_ms",
        "warm_disk_median_ms",
        "warm_preloaded_median_ms",
        "peak_rss_mib",
    ):
        assert blocked[column].isna().all()


def test_stage_sum_integrity_and_signed_instrumentation_delta(
    aggregate_frames: dict[str, object],
) -> None:
    stages = aggregate_frames["stages"]
    selected = stages[
        (stages.route == "6D_ORACLE")
        & (stages.method == "native")
        & (stages["mode"] == "warm-disk")
    ]
    for _, group in selected.groupby("sample_id"):
        assert int(group.elapsed_ns.dropna().sum()) == int(
            group.instrumented_wall_ns.dropna().iloc[0]
        )
        expected = int(group.instrumented_wall_ns.dropna().iloc[0]) - int(
            group.whole_elapsed_ns.dropna().iloc[0]
        )
        assert set(group.instrumentation_overhead_ns.dropna().astype(int)) == {
            expected
        }
    # Negative values are retained, proving the paired delta is not clamped.
    assert (selected.instrumentation_overhead_ns < 0).any()


def test_memory_eligibility_separates_failed_parity(
    aggregate_frames: dict[str, object],
) -> None:
    memory = aggregate_frames["memory"]
    assert memory[
        memory.route.isin(["CROG", "G1", "C1"])
    ].eligible_deployment_memory.eq(False).all()
    valid = memory[
        memory.route.isin(["6D_ORACLE", "6D_ADAPTED"])
        & (memory.method == "native")
        & memory["mode"].isin(["cold", "warm-disk", "warm-preloaded"])
    ]
    assert valid.eligible_deployment_memory.eq(True).all()
    assert valid.rss_total_bytes.ge(0).all()
    assert valid.rss_total_bytes.gt(0).any()
    for reported_route, worker_route in (
        ("6D_ORACLE", "oracle"),
        ("6D_ADAPTED", "adapted"),
    ):
        worker_root = RUNTIME / "workers/6d" / worker_route / "native"
        for mode in ("warm-disk", "warm-preloaded"):
            result = aggregation._read_json(
                worker_root / f"{mode.replace('-', '_')}_result.json"
            )
            selected = valid[
                (valid.route == reported_route) & (valid["mode"] == mode)
            ]
            assert set(selected.child_pid.astype(int)) == {int(result["pid"])}


def test_input_signature_excludes_aggregate_outputs() -> None:
    inventory = aggregation._source_inventory(RUNTIME, REPO)
    assert inventory
    assert not any(Path(name).name in aggregation.OUTPUT_NAMES for name in inventory)
    assert aggregation._input_signature(inventory) == aggregation._input_signature(
        dict(reversed(list(inventory.items())))
    )


def test_warm_subset_rejects_duplicate_padding() -> None:
    variant = aggregation._variant("6D_ORACLE", "native")
    expected = [f"g{index}" for index in range(98)]
    padded = pd.DataFrame({"sample_id": [*expected[:-1], expected[0]]})
    with pytest.raises(aggregation.RuntimeAggregationError, match="exactly"):
        aggregation._validate_warm_ids(
            padded, expected, variant, "warm-disk"
        )


def test_resume_hash_validation_detects_output_change(tmp_path: Path) -> None:
    outputs = {}
    for name in aggregation.OUTPUT_NAMES:
        output = tmp_path / name
        output.write_text(f"locked {name}\n", encoding="utf-8")
        outputs[name] = aggregation.sha256_file(output)
    completion = {
        "input_signature": "signature",
        "output_sha256": outputs,
    }
    assert aggregation._validate_resume(completion, tmp_path, "signature")
    output = tmp_path / "route_parity.csv"
    output.write_text("changed\n", encoding="utf-8")
    with pytest.raises(aggregation.RuntimeAggregationError, match="changed"):
        aggregation._validate_resume(completion, tmp_path, "signature")


def test_reranker_gate_overhead_is_median_of_paired_sums() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b", "c", "c"],
            "stage": ["reranker", "gate"] * 3,
            "elapsed_ns": [1, 100, 100, 1, 100, 100],
            "stage_status": ["measured"] * 6,
        }
    )
    assert aggregation._stage_component_sum_median(
        frame, {"reranker", "gate"}
    ) == 101.0
