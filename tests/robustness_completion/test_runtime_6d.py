from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from robustness_completion.common import PREREGISTRATION_SHA256, sha256_file
from robustness_completion.runtime_6d import (
    CPU_THREADS,
    EXPECTED_PROFILE_GROUPS,
    MISSING_FORMAL_CHECKPOINT,
    PARITY_GROUPS,
    WARMUP_GROUPS,
    Formal6DBackend,
    MissingFormalCheckpointError,
    _rotation_distance_radians,
    _selection_manifest,
    _stage_contract,
    _stage_row,
    assert_formal_reranker_available,
    build_parser,
    process_memory_snapshot,
    select_profile_groups,
    validate_source_contract,
)


REPO = Path(__file__).resolve().parents[2]
RUN_DIR = (
    REPO
    / "artifacts"
    / "robustness_completion"
    / "20260822_194405_remaining_robustness"
)


@pytest.fixture(scope="module")
def source_contract():
    return validate_source_contract(REPO)


@pytest.fixture(scope="module")
def selection(source_contract):
    return select_profile_groups(source_contract)


def test_preregistration_hash_is_locked_before_inference() -> None:
    path = RUN_DIR / "PRE_REGISTRATION.md"
    assert sha256_file(path) == PREREGISTRATION_SHA256
    assert (RUN_DIR / "PRE_REGISTRATION.sha256").read_text().split()[0] == (
        PREREGISTRATION_SHA256
    )


def test_6d_subset_is_14_per_scene_stable_hash(selection) -> None:
    assert len(selection.groups) == EXPECTED_PROFILE_GROUPS == 98
    counts = Counter(str(group.target["scene_id"]) for group in selection.groups)
    assert len(counts) == 7
    assert set(counts.values()) == {14}
    for scene in counts:
        actual = [
            group.group_id
            for group in selection.groups
            if str(group.target["scene_id"]) == scene
        ]
        assert set(actual) == set(
            sorted(
                actual,
                key=lambda value: (
                    __import__("hashlib").sha256(value.encode()).hexdigest(),
                    value,
                ),
            )
        )


def test_parity_ids_are_first_20_of_declared_order(selection) -> None:
    assert len(selection.parity_groups) == PARITY_GROUPS
    assert selection.parity_groups == selection.groups[:PARITY_GROUPS]
    manifest = _selection_manifest(selection)
    assert manifest["parity_group_ids"] == manifest["ordered_group_ids"][:20]
    assert manifest["ordering_sha256"] == selection.ordering_sha256


def test_five_warmups_are_distinct(selection) -> None:
    identifiers = [group.group_id for group in selection.warmup_groups]
    assert len(identifiers) == WARMUP_GROUPS
    assert len(set(identifiers)) == WARMUP_GROUPS
    assert tuple(selection.warmup_groups) == tuple(selection.groups[:5])


def test_missing_formal_ranker_fails_closed(source_contract) -> None:
    with pytest.raises(MissingFormalCheckpointError, match=MISSING_FORMAL_CHECKPOINT):
        assert_formal_reranker_available(source_contract)


def test_worker_never_names_cached_reranked_predictions() -> None:
    source = (REPO / "src" / "robustness_completion" / "runtime_6d.py").read_text()
    forbidden = "reranked" + "_predictions.parquet"
    assert forbidden not in source


def test_stage_contract_reports_unsplittable_formal_apis() -> None:
    oracle = _stage_contract("oracle", "native")
    adapted = _stage_contract("adapted", "native")
    assert oracle["stage_profile_complete"] is False
    assert oracle["measured_combined_stages"]["depth_workspace_tsdf_combined"] == [
        "depth_backprojection",
        "workspace_construction",
        "tsdf_integration",
    ]
    assert adapted["measured_combined_stages"][
        "hifics_inference_and_mask_postprocess"
    ] == ["hifics_inference", "mask_postprocess"]
    assert oracle["reranker_stage"] == "not_applicable_native"
    assert oracle["runtime_feature_extraction_stage"] == "not_applicable_native"


def test_skipped_stage_is_null_not_fake_zero() -> None:
    record = _stage_row("reranker", None, status="not_applicable_native")
    assert record["elapsed_ns"] is None
    assert record["stage_status"] == "not_applicable_native"


def test_rotation_parity_tolerance_is_in_radians() -> None:
    import numpy as np

    angle = 5e-5
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert _rotation_distance_radians(np.eye(3), rotation) == pytest.approx(angle)


def test_cpu_and_eight_thread_contract(monkeypatch) -> None:
    assert CPU_THREADS == 8
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert __import__("os").environ[variable] == "8"


def test_memory_hook_snapshot_is_nonnegative() -> None:
    snapshot = process_memory_snapshot()
    assert snapshot["device"] == "cpu"
    assert snapshot["rss_bytes"] > 0
    assert snapshot["peak_rss_bytes"] > 0
    assert snapshot["perf_counter_ns"] >= 0
    assert snapshot["unified_memory_peak"] == "not directly measurable"


def test_adapted_raw_loader_does_not_open_ground_truth(source_contract, selection) -> None:
    loaded = Formal6DBackend(source_contract).load_group(
        selection.groups[0], "adapted"
    )
    assert loaded.rgb is not None
    assert loaded.instance_label is None
    assert "instance_label" not in loaded.source_hashes
    assert "rgb" in loaded.source_hashes


def test_oracle_raw_loader_uses_label_not_rgb(source_contract, selection) -> None:
    loaded = Formal6DBackend(source_contract).load_group(
        selection.groups[0], "oracle"
    )
    assert loaded.instance_label is not None
    assert loaded.rgb is None
    assert "instance_label" in loaded.source_hashes
    assert "rgb" not in loaded.source_hashes


def test_cli_exposes_orchestrator_modes() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--repo",
            str(REPO),
            "--run-dir",
            str(RUN_DIR),
            "--route",
            "oracle",
            "--method",
            "native",
            "--mode",
            "warm-preloaded",
            "--output",
            str(RUN_DIR / "runtime_full/workers/6d/oracle/native/result.json"),
            "--resume",
        ]
    )
    assert args.mode == "warm-preloaded"
    assert args.route == "oracle"
    assert args.method == "native"
    assert args.resume is True


def test_source_contract_records_selected_adapted_condition(source_contract) -> None:
    payload = json.loads(source_contract.condition_selection.read_text())
    assert payload["selected_condition"] == "hifics_adapted_mask"
