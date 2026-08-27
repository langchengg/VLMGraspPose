from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import robustness_completion.reporting as reporting
from robustness_completion.common import PREREGISTRATION_SHA256, sha256_file
from robustness_completion.runtime_aggregation import (
    OUTPUT_NAMES,
    _input_signature,
    _source_inventory,
)

from robustness_completion.reporting import (
    METHOD_ORDER,
    PRIMARY_FOUR_D_ROUTES,
    SIX_D_ROUTES,
    _canonical_parity,
    _canonical_stages,
    _canonical_timing,
    _completion_source_paths,
    _duplicate_paragraphs,
    _plot_p50_p95,
    _plot_stages,
    _parity_passes,
    _preferred_variant,
    _runtime_latex,
    _runtime_summary,
    _stage_bottlenecks,
    _verify_runtime_completion,
    assess_completion,
)


REPO = Path(__file__).resolve().parents[2]
RUN = (
    REPO
    / "artifacts/robustness_completion/20260822_194405_remaining_robustness"
)


def test_reporting_accepts_worker_timing_aliases() -> None:
    source = pd.DataFrame(
        {
            "route_id": ["oracle_gt_mask", "crog"],
            "variant": ["raw_lambdamart", "expected_gain_gated"],
            "cache_policy": ["warm_disk_input", "warm_preloaded"],
            "group_id": ["g0", "t0"],
            "formal_device": ["cpu", "mps"],
            "uninstrumented_total_ns": [2_500_000, 4_000_000],
            "status": ["MEASURED_FULL_DEPLOYMENT"] * 2,
            "complete_deployment": [True] * 2,
        }
    )
    result = _canonical_timing(source, default_mode="warm-disk")
    assert result["route"].tolist() == ["6D_ORACLE", "CROG"]
    assert result["method"].tolist() == ["raw", "gated"]
    assert result["mode"].tolist() == ["warm-disk", "warm-preloaded"]
    assert result["elapsed_ms"].tolist() == [2.5, 4.0]
    assert result["complete_deployment"].tolist() == [True, True]


def test_missing_runtime_values_are_not_zero_filled() -> None:
    summary = _runtime_summary(
        pd.DataFrame(),
        pd.DataFrame(
            {
                "route": ["G1"],
                "method": ["native"],
                "sample_id": ["one"],
                "whole_elapsed_ns": [10_000_000],
                "device": ["mps"],
                "status": ["MEASURED_FULL_DEPLOYMENT"],
                "complete_deployment": [True],
            }
        ),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    row = summary.iloc[0]
    assert row.warm_disk_median_ms == 10.0
    assert np.isnan(row.cold_median_ms)
    assert np.isnan(row.warm_preloaded_median_ms)
    assert np.isnan(row.peak_rss_mib)
    assert "not estimable" in _runtime_latex(summary)


def test_empty_runtime_is_partial_not_complete() -> None:
    assessment = assess_completion(
        RUN,
        pd.DataFrame(),
        pd.DataFrame(),
        d1_required=True,
    )
    assert assessment.duplicate_complete
    assert not assessment.runtime_complete
    assert not assessment.emitted
    assert assessment.status == "PARTIAL_REMAINING_ROBUSTNESS_EXPERIMENTS"
    assert any("6D_ORACLE/raw" in item for item in assessment.blockers)
    assert any("D1/gated" in item for item in assessment.blockers)


def test_formal_missing_checkpoint_cannot_be_masked_by_fabricated_rows() -> None:
    pairs = {
        (route, method)
        for route in PRIMARY_FOUR_D_ROUTES
        for method in METHOD_ORDER
    }
    pairs.update(
        (route, method)
        for route in SIX_D_ROUTES
        for method in ("native", "raw")
    )
    pairs.update(("D1", method) for method in METHOD_ORDER)
    rows = []
    parity_rows = []
    for route, method in pairs:
        rows.append(
            {
                "route": route,
                "method": method,
                "cold_n": 5,
                "warm_disk_n": 100 if not route.startswith("6D_") else 98,
                "warm_preloaded_n": 100
                if not route.startswith("6D_")
                else 98,
                "peak_rss_mib": 512.0,
            }
        )
        parity_rows.append(
            {
                "route": route,
                "method": method,
                "n": 20,
                "status": True,
                "candidate_parity": True,
                "native_top1_parity": True,
                "reranked_top1_parity": True,
                "gate_parity": True,
                "final_top1_parity": True,
            }
        )
    assessment = assess_completion(
        RUN,
        pd.DataFrame(rows),
        pd.DataFrame(parity_rows),
        d1_required=True,
    )
    assert not assessment.emitted
    assert not assessment.runtime_complete
    assert assessment.status == "PARTIAL_REMAINING_ROBUSTNESS_EXPERIMENTS"
    assert any("formal LambdaMART Booster" in item for item in assessment.blockers)


def test_zero_match_duplicate_audit_is_worded_as_noninformative() -> None:
    interpretation, legacy, _, lock = _duplicate_paragraphs(RUN)
    assert lock["counts"]["strict_pairs"] == 0
    assert "non-informative" in interpretation
    assert "does not demonstrate duplicate robustness" in interpretation
    assert "previous approximate count could not be reproduced" in legacy


def test_partial_runtime_figure_has_no_fabricated_points(tmp_path: Path) -> None:
    hashes = _plot_p50_p95(pd.DataFrame(), tmp_path)
    assert set(hashes) == {"pdf", "svg", "png"}
    for extension, digest in hashes.items():
        output = tmp_path / f"full_runtime_p50_p95.{extension}"
        assert output.is_file()
        assert len(digest) == 64


def test_parity_aliases_preserve_explicit_failure() -> None:
    source = pd.DataFrame(
        {
            "route": ["crog", "crog"],
            "method": ["gated", "gated"],
            "status": ["PASS", "FAILED_PARITY"],
            "parity_count": [20, 20],
            "candidate_parity": [True, False],
            "ranker_parity": [True, True],
            "gate_parity": [True, True],
        }
    )
    result = _canonical_parity(source).iloc[0]
    assert result["n"] == 20
    assert not result["status"]
    assert not result["candidate_parity"]


def test_parity_gate_requires_one_exact_method_row_and_exact_count() -> None:
    row = {
        "route": "G1",
        "method": "native",
        "n": 20,
        "status": True,
        "candidate_parity": True,
        "native_top1_parity": True,
        "reranked_top1_parity": True,
        "gate_parity": True,
        "final_top1_parity": True,
    }
    assert _parity_passes(pd.DataFrame([row]), "G1", "native", 20)

    gated = {**row, "method": "gated"}
    assert not _parity_passes(pd.DataFrame([gated]), "G1", "native", 20)

    wrong_count = {**row, "n": 21}
    assert not _parity_passes(
        pd.DataFrame([wrong_count]), "G1", "native", 20
    )
    fractional_count = {**row, "n": 20.5}
    assert not _parity_passes(
        pd.DataFrame([fractional_count]), "G1", "native", 20
    )
    assert not _parity_passes(
        pd.DataFrame([row, row]), "G1", "native", 20
    )

    duplicate_source = _canonical_parity(
        pd.DataFrame(
            [
                {**row, "parity_count": 20, "status": "PASS"},
                {**row, "parity_count": 20, "status": "PASS"},
            ]
        )
    )
    assert duplicate_source.iloc[0].source_rows == 2
    assert not _parity_passes(duplicate_source, "G1", "native", 20)


def test_runtime_summary_ignores_finite_ineligible_rows() -> None:
    disk = pd.DataFrame(
        [
            {
                "route": "G1",
                "method": "native",
                "sample_id": "eligible",
                "whole_elapsed_ns": 10_000_000,
                "device": "mps",
                "status": "MEASURED_FULL_DEPLOYMENT",
                "complete_deployment": True,
            },
            {
                "route": "G1",
                "method": "native",
                "sample_id": "blocked",
                "whole_elapsed_ns": 1_000,
                "device": "mps",
                "status": "FAILED_PARITY",
                "complete_deployment": False,
            },
            {
                "route": "G1",
                "method": "raw",
                "sample_id": "blocked-only",
                "whole_elapsed_ns": 2_000,
                "device": "mps",
                "status": "MISSING_FORMAL_CHECKPOINT",
                "complete_deployment": False,
            },
        ]
    )
    memory = pd.DataFrame(
        [
            {
                "route": "G1",
                "method": "native",
                "device": "mps",
                "status": "COMPLETE_DEPLOYMENT_MEMORY",
                "complete_deployment": True,
                "peak_rss_mib": 512.0,
            },
            {
                "route": "G1",
                "method": "native",
                "device": "mps",
                "status": "FAILED_PARITY",
                "complete_deployment": False,
                "peak_rss_mib": 8192.0,
            },
        ]
    )
    summary = _runtime_summary(
        pd.DataFrame(), disk, pd.DataFrame(), pd.DataFrame(), memory
    ).set_index(["route", "method"])
    native = summary.loc[("G1", "native")]
    raw = summary.loc[("G1", "raw")]
    assert native.warm_disk_n == 1
    assert native.warm_disk_median_ms == 10.0
    assert native.peak_rss_mib == 512.0
    assert raw.warm_disk_n == 0
    assert np.isnan(raw.warm_disk_median_ms)


def test_stage_reporting_uses_measured_stage_values_and_falls_back_to_native(
    tmp_path: Path,
) -> None:
    source = pd.read_parquet(RUN / "runtime_full/stage_timings.parquet")
    measured_source = source[
        (source["status"] == "measured") & source["elapsed_ns"].notna()
    ]
    assert not measured_source.empty

    canonical = _canonical_stages(source)
    source_row = measured_source.iloc[0]
    canonical_row = canonical.loc[source_row.name]
    assert canonical_row.elapsed_ms == source_row.elapsed_ns / 1e6
    assert canonical_row.elapsed_ms != source_row.whole_elapsed_ns / 1e6

    preferred = _preferred_variant(canonical, "warm-disk")
    assert set(zip(preferred["route"], preferred["method"])) == {
        ("6D_ORACLE", "native"),
        ("6D_ADAPTED", "native"),
    }

    bottlenecks = _stage_bottlenecks(source)
    assert "not estimable" not in bottlenecks
    assert "6D ORACLE — native" in bottlenecks
    assert "6D ADAPTED — native" in bottlenecks
    assert "runtime feature extraction" not in bottlenecks
    assert "depth workspace tsdf combined" in bottlenecks
    assert "hifics inference and mask postprocess" in bottlenecks

    hashes = _plot_stages(
        source,
        tmp_path,
        mode="warm-disk",
        stem="actual_aggregate_stages",
    )
    assert set(hashes) == {"pdf", "svg", "png"}
    for extension, digest in hashes.items():
        output = tmp_path / f"actual_aggregate_stages.{extension}"
        assert output.is_file()
        assert output.stat().st_size > 0
        assert len(digest) == 64


def _write_synthetic_runtime_completion(runtime_root: Path) -> dict[str, object]:
    runtime_root.mkdir()
    output_hashes = {}
    for name in OUTPUT_NAMES:
        output = runtime_root / name
        output.write_bytes(f"locked synthetic output: {name}\n".encode())
        output_hashes[name] = sha256_file(output)
    payload: dict[str, object] = {
        "schema_version": 2,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "input_signature": _input_signature(
            _source_inventory(runtime_root, runtime_root.parent)
        ),
        "output_sha256": output_hashes,
    }
    (runtime_root / "runtime_completion.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return payload


def test_runtime_completion_binds_exact_current_aggregate_outputs(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime_full"
    expected = _write_synthetic_runtime_completion(runtime_root)
    assert _verify_runtime_completion(runtime_root, tmp_path) == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("old_schema", "schema must be >= 2"),
        ("wrong_preregistration", "preregistration SHA-256"),
        ("missing_output_key", "exact output set"),
        ("extra_output_key", "exact output set"),
        ("changed_output", "output SHA-256 mismatch"),
        ("changed_input", "input signature mismatch"),
    ),
)
def test_runtime_completion_integrity_failures_are_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    runtime_root = tmp_path / "runtime_full"
    payload = _write_synthetic_runtime_completion(runtime_root)
    hashes = payload["output_sha256"]
    assert isinstance(hashes, dict)
    if mutation == "old_schema":
        payload["schema_version"] = 1
    elif mutation == "wrong_preregistration":
        payload["preregistration_sha256"] = "0" * 64
    elif mutation == "missing_output_key":
        hashes.pop(OUTPUT_NAMES[0])
    elif mutation == "extra_output_key":
        hashes["unexpected.csv"] = "0" * 64
    elif mutation == "changed_output":
        (runtime_root / OUTPUT_NAMES[0]).write_text("changed", encoding="utf-8")
    else:
        workers = runtime_root / "workers"
        workers.mkdir()
        (workers / "new_source.json").write_text("{}", encoding="utf-8")
    (runtime_root / "runtime_completion.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match=message):
        _verify_runtime_completion(runtime_root, tmp_path)


def test_report_verifies_runtime_lock_before_reading_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_runtime(
        _runtime_root: Path, _repo: Path
    ) -> dict[str, object]:
        raise RuntimeError("synthetic runtime integrity failure")

    def reject_read(_path: Path) -> pd.DataFrame:
        raise AssertionError("aggregate reader ran before integrity verification")

    monkeypatch.setattr(reporting, "_verify_runtime_completion", reject_runtime)
    monkeypatch.setattr(reporting, "_read_csv", reject_read)
    monkeypatch.setattr(reporting, "_read_parquet", reject_read)
    with pytest.raises(RuntimeError, match="synthetic runtime integrity failure"):
        reporting.generate_completion_report(REPO, RUN)


def test_completion_manifest_sources_include_runtime_integrity_evidence() -> None:
    runtime = RUN / "runtime_full"
    parent = REPO / reporting.PARENT_RELATIVE
    sources = set(_completion_source_paths(RUN, parent, runtime))
    assert runtime / "runtime_completion.json" in sources
    assert runtime / "4D_PARITY_FAILURE_AUDIT.md" in sources
    assert RUN / "source_integrity_final.json" in sources
    assert RUN / "COMMAND_LOG.md" in sources
    assert RUN / "test_report.txt" in sources
