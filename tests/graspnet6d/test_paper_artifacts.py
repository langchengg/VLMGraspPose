"""Fail-closed tests for formal paper artifact publication.

These tests intentionally stop at the provenance boundary.  Fixture, blocked,
missing, and stale runs must never be turned into paper-looking placeholder
outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import graspnet6d.paper_artifacts as paper_artifacts
from graspnet6d.paper_artifacts import (
    FIGURE_FAMILIES,
    PaperArtifactsRefused,
    generate_formal_paper_artifacts,
    validate_formal_paper_inputs,
)
from graspnet6d.provenance import RUN_MANIFEST_SCHEMA_VERSION


_FORMAL_DOCUMENTS = (
    "METHODS.md",
    "RESULTS.md",
    "CONCLUSIONS.md",
    "LIMITATIONS.md",
    "REPRODUCE.md",
    "dataset_summary.json",
    "conclusion_evidence.json",
)


def _write_manifest(
    run_dir: Path,
    *,
    status: str,
    profile: str = "paper-lite",
    resolved_config_sha256: str = "0" * 64,
) -> None:
    run_dir.mkdir(parents=True)
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "status": status,
        "profile": profile,
        "immutable_identity": {
            "profile": profile,
            "resolved_config": {
                "profile": profile,
                "formal_results": True,
            },
        },
        "resolved_config_sha256": resolved_config_sha256,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _assert_no_new_paper_outputs(run_dir: Path) -> None:
    for filename in _FORMAL_DOCUMENTS:
        assert not (run_dir / filename).exists()
    assert not (run_dir / "paper_artifact_manifest.json").exists()
    assert not (run_dir / "paper").exists()
    assert list(run_dir.rglob("*.pdf")) == []
    assert list(run_dir.rglob("*.png")) == []


def test_quantitative_contract_names_exactly_fifteen_unique_families() -> None:
    identifiers = [identifier for identifier, _ in FIGURE_FAMILIES]

    assert len(identifiers) == 15
    assert len(set(identifiers)) == 15


def test_smoke_contract_reserves_check_twelve_for_legacy_regression() -> None:
    assert {
        number: paper_artifacts._expected_smoke_kind(number) for number in range(1, 13)
    } == {
        **{number: "formal_smoke_gate" for number in range(1, 12)},
        12: "data_independent_regression",
    }


def test_report_owned_run_state_does_not_stale_successful_resume() -> None:
    upstream = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "status": "IN_PROGRESS",
        "sample_counts": {"target_groups": 12},
        "formal_results_emitted": False,
    }
    before = paper_artifacts._paper_run_manifest_hash(upstream)
    terminal = {
        **upstream,
        "status": "COMPLETE",
        "ended_at_utc": "2026-08-16T00:00:00+00:00",
        "formal_results_emitted": True,
        "paper_artifact_manifest_sha256": "a" * 64,
    }

    assert paper_artifacts._paper_run_manifest_hash(terminal) == before
    terminal["sample_counts"] = {"target_groups": 13}
    assert paper_artifacts._paper_run_manifest_hash(terminal) != before


def test_report_runtime_row_is_outside_immutable_figure_fifteen_input(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "formal_runtime"
    run_dir.mkdir()
    rows = [
        {
            "stage": stage,
            "status": "COMPLETE",
            "wall_time_s": float(index + 1),
            "process_rss_bytes_at_end": 1024 + index,
            "resume_requested": False,
        }
        for index, stage in enumerate(paper_artifacts._REQUIRED_RUNTIME_STAGES)
    ]
    path = run_dir / "runtime.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    before_frame, before_sources = paper_artifacts._validate_runtime(run_dir)

    pd.DataFrame(
        [
            *rows,
            {
                "stage": "report",
                "status": "COMPLETE",
                "wall_time_s": 9.5,
                "process_rss_bytes_at_end": 2048,
                "resume_requested": False,
            },
        ]
    ).to_csv(path, index=False)
    after_frame, after_sources = paper_artifacts._validate_runtime(run_dir)

    pd.testing.assert_frame_equal(after_frame, before_frame)
    assert after_sources == before_sources


def test_missing_run_refuses_without_creating_a_directory(tmp_path: Path) -> None:
    missing = tmp_path / "20260815_missing_formal_run"

    with pytest.raises(PaperArtifactsRefused, match="directory is invalid"):
        validate_formal_paper_inputs(missing)

    assert not missing.exists()


def test_fixture_scope_refuses_all_paper_outputs_without_placeholders(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "fixture_formal_run"
    _write_manifest(run_dir, status="IN_PROGRESS")

    with pytest.raises(PaperArtifactsRefused, match="fixture/diagnostic"):
        generate_formal_paper_artifacts(run_dir)

    _assert_no_new_paper_outputs(run_dir)
    assert not (run_dir / "FINAL_STATUS.md").exists()


def test_blocked_run_preserves_blocked_status_and_emits_no_results(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "20260815_blocked_formal_run"
    _write_manifest(run_dir, status="BLOCKED")
    blocked_status = "# BLOCKED\n\nDisk preflight did not pass.\n"
    (run_dir / "FINAL_STATUS.md").write_text(blocked_status, encoding="utf-8")

    with pytest.raises(PaperArtifactsRefused, match="paper outputs are forbidden"):
        generate_formal_paper_artifacts(run_dir)

    assert (run_dir / "FINAL_STATUS.md").read_text(encoding="utf-8") == blocked_status
    _assert_no_new_paper_outputs(run_dir)


def test_stale_resolved_configuration_refuses_before_rendering(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "20260815_stale_formal_run"
    _write_manifest(run_dir, status="IN_PROGRESS", resolved_config_sha256="0" * 64)
    (run_dir / "resolved_config.yaml").write_text(
        "formal_results: true\nprofile: paper-lite\n", encoding="utf-8"
    )

    with pytest.raises(PaperArtifactsRefused, match="configuration is stale"):
        generate_formal_paper_artifacts(run_dir)

    _assert_no_new_paper_outputs(run_dir)
    assert not (run_dir / "FINAL_STATUS.md").exists()
