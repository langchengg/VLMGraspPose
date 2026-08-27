from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from robustness_suite.cli import _extended_source_paths, _source_hash_contract
from robustness_suite.io import sha256_file
from robustness_suite.runtime_profile import (
    _runtime_completion_valid,
    _sha256_file,
    _sha256_json,
    _gate_result_for_row,
    _load_formal_gate_bundle,
    _load_formal_ranker_bundle,
    _ranker_scores_for_group,
    _route_sources,
    build_subset_manifest,
    deployment_stage_total_ns,
    stage_sum_matches_total,
    synchronize_device,
    timed_call_ns,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (
    REPO_ROOT
    / "artifacts/robustness_suite/20260822_085125_robustness_suite"
)


def test_source_artifacts_are_read_only() -> None:
    for relative, expected in _source_hash_contract(REPO_ROOT).items():
        path = REPO_ROOT / relative
        assert path.is_file()
        assert sha256_file(path) == expected


def test_extended_source_inventory_hashes_are_current() -> None:
    inventory = json.loads((RUN_DIR / "source_artifact_inventory.json").read_text())
    expected = {str(item["path"]): str(item["sha256"]) for item in inventory["files"]}
    observed = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in _extended_source_paths(REPO_ROOT)
    }
    assert observed == expected


def test_runtime_subset_deterministic(tmp_path: Path) -> None:
    first = build_subset_manifest(REPO_ROOT, tmp_path / "first")
    second = build_subset_manifest(REPO_ROOT, tmp_path / "second")
    assert first["four_d"]["sample_ids"] == second["four_d"]["sample_ids"]
    assert first["six_d"]["group_ids"] == second["six_d"]["group_ids"]
    assert first["four_d"]["count"] >= 100
    assert first["six_d"]["count"] >= 100


def test_runtime_timer_nonnegative() -> None:
    elapsed, value = timed_call_ns(lambda: 7)
    assert elapsed >= 0
    assert value == 7


def test_runtime_resume_signature_rejects_tamper(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run = repo / "artifacts/robustness_suite/example"
    output = run / "runtime_profile"
    output.mkdir(parents=True)
    source = repo / "source.bin"
    source.write_bytes(b"source")
    (run / "PRE_REGISTRATION.md").write_text("locked\n")
    (run / "source_artifact_inventory.json").write_text("{}\n")
    (output / "summary.csv").write_text("route\nCROG\n")
    payload = {
        "schema_version": 1,
        "status": "PARTIAL_RUNTIME_PROFILE",
        "run_id": "example",
        "runtime_code_sha256": _sha256_file(
            Path(__import__("robustness_suite.runtime_profile", fromlist=["x"]).__file__)
        ),
        "source_hashes": {"source.bin": _sha256_file(source)},
        "run_local_hashes": {
            "PRE_REGISTRATION.md": _sha256_file(run / "PRE_REGISTRATION.md"),
            "source_artifact_inventory.json": _sha256_file(
                run / "source_artifact_inventory.json"
            ),
        },
        "output_hashes": {"summary.csv": _sha256_file(output / "summary.csv")},
        "row_counts": {
            "measured_rows": 1,
            "startup_rows": 0,
            "offline_evaluator_rows": 0,
        },
        "subset_protocol_deviations": [],
    }
    payload["completion_signature_sha256"] = _sha256_json(payload)
    (output / "completion.json").write_text(json.dumps(payload))
    assert _runtime_completion_valid(repo, run) is not None
    (output / "summary.csv").write_text("route\nD1\n")
    assert _runtime_completion_valid(repo, run) is None


@pytest.mark.parametrize("route", ["CROG", "G1", "C1", "D1"])
def test_formal_4d_ranker_executes_and_matches_frozen_scores(route: str) -> None:
    source = next(item for item in _route_sources(REPO_ROOT) if item.route == route)
    assert source.feature_path is not None
    assert source.prediction_path is not None
    features = pd.read_parquet(source.feature_path)
    frozen = pd.read_parquet(source.prediction_path)
    sample_id = str(frozen.iloc[0]["sample_id"])
    group = features.loc[features["sample_id"].astype(str).eq(sample_id)]
    locked = frozen.loc[frozen["sample_id"].astype(str).eq(sample_id)].sort_values(
        ["native_rank", "candidate_id"], kind="mergesort"
    )
    bundle = _load_formal_ranker_bundle(REPO_ROOT, source)
    seed_scores, ensemble = _ranker_scores_for_group(bundle, group)
    assert bundle.seeds == (42, 123, 2026)
    for index, seed in enumerate(bundle.seeds):
        assert seed_scores[:, index] == pytest.approx(locked[f"score_seed_{seed}"])
    assert ensemble == pytest.approx(locked["ensemble_score"])


@pytest.mark.parametrize("route", ["CROG", "G1", "C1", "D1"])
def test_formal_4d_gate_executes_and_matches_frozen_decision(route: str) -> None:
    source = next(item for item in _route_sources(REPO_ROOT) if item.route == route)
    bundle = _load_formal_gate_bundle(REPO_ROOT, source)
    row = bundle.inputs.head(1)
    recovered, harmful, switch = _gate_result_for_row(bundle, row)
    expected = bundle.decisions.loc[
        bundle.decisions["sample_id"].astype(str).eq(str(row.iloc[0]["sample_id"]))
    ].iloc[0]
    assert recovered == pytest.approx(expected["probability_recover"])
    assert harmful == pytest.approx(expected["probability_harm"])
    assert switch == bool(expected["switch"])


def test_runtime_stage_sum_matches_total_within_tolerance() -> None:
    rows = pd.DataFrame(
        {
            "stage": ["input_io", "reranker", "offline_evaluator"],
            "elapsed_ns": [100, 50, 1_000],
            "deployment_stage": [True, True, False],
        }
    )
    assert deployment_stage_total_ns(rows) == 150
    assert stage_sum_matches_total(rows, 151, relative_tolerance=0.01)
    assert not stage_sum_matches_total(rows, 200, relative_tolerance=0.01)


def test_mps_synchronization_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(synchronize=lambda: calls.append("mps")),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    synchronize_device("mps")
    assert calls == ["mps"]


def test_offline_evaluator_excluded_from_deployment_total() -> None:
    rows = pd.DataFrame(
        {
            "elapsed_ns": [11, 13, 10_000],
            "deployment_stage": [True, True, False],
        }
    )
    assert deployment_stage_total_ns(rows) == 24


def test_duplicate_map_hash_locked() -> None:
    source = json.loads((RUN_DIR / "source_run_manifest.json").read_text())
    duplicate = source["sources"]["duplicate_map"]
    assert duplicate["status"] == "FAIL_CLOSED_MISSING_DUPLICATE_MAP"
    assert duplicate["path"] is None
    assert duplicate["sha256"] is None


def test_report_values_match_raw_results() -> None:
    summary_path = RUN_DIR / "runtime_profile/summary.csv"
    timing_path = RUN_DIR / "runtime_profile/raw_stage_timings.parquet"
    if not summary_path.is_file() or not timing_path.is_file():
        pytest.skip("runtime profile has not run")
    summary = pd.read_csv(summary_path)
    timings = pd.read_parquet(timing_path)
    assert summary["partial_measured_component_median_ms"].isna().all()
    assert summary["deployment_median_ms"].isna().all()
    for route, row in summary.set_index("route").iterrows():
        raw = timings.loc[
            timings["route"].eq(route)
            & timings["stage"].eq("reranker")
            & ~timings["sample_id"].astype(str).str.startswith("__")
        ]
        expected = raw["elapsed_ms"].median()
        if pd.isna(expected):
            assert pd.isna(row["reranker_overhead_ms"])
        else:
            assert row["reranker_overhead_ms"] == pytest.approx(expected)
