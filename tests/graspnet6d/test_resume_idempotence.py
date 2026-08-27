from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from graspnet6d import cli, paper_artifacts, provenance
from graspnet6d.io import sha256_file
from graspnet6d.paper_artifacts import (
    FormalPaperInputs,
    PaperArtifactsRefused,
    generate_formal_paper_artifacts,
)


def test_unchanged_complete_stage_record_is_byte_stable(tmp_path: Path) -> None:
    provenance.record_stage(
        tmp_path,
        "features",
        "COMPLETE",
        summary={
            "total_groups": 2,
            "completed_groups": 2,
            "resumed_groups": 0,
            "output_paths": ["a.parquet", "b.parquet"],
        },
        resumed=False,
    )
    path = tmp_path / "stages/features.json"
    original = path.read_bytes()

    provenance.record_stage(
        tmp_path,
        "features",
        "COMPLETE",
        summary={
            "total_groups": 2,
            "completed_groups": 0,
            "resumed_groups": 2,
            "output_paths": ["a.parquet", "b.parquet"],
        },
        resumed=True,
    )

    assert path.read_bytes() == original

    provenance.record_stage(
        tmp_path,
        "features",
        "COMPLETE",
        summary={
            "total_groups": 2,
            "completed_groups": 1,
            "resumed_groups": 1,
            "output_paths": ["a.parquet", "changed.parquet"],
        },
        resumed=True,
    )
    assert path.read_bytes() != original


def test_first_successful_runtime_measurement_is_immutable(tmp_path: Path) -> None:
    cli._record_runtime(
        tmp_path,
        "features",
        started=time.perf_counter() - 2.0,
        status="COMPLETE",
        resumed=False,
    )
    path = tmp_path / "runtime.csv"
    original = path.read_bytes()

    cli._record_runtime(
        tmp_path,
        "features",
        started=time.perf_counter() - 200.0,
        status="COMPLETE",
        resumed=True,
    )
    cli._record_runtime(
        tmp_path,
        "features",
        started=time.perf_counter() - 1.0,
        status="FAILED",
        resumed=True,
    )

    assert path.read_bytes() == original


def test_operational_resume_metadata_is_outside_paper_fingerprint() -> None:
    scientific = {
        "profile": "paper-lite",
        "sample_counts": {"target_groups": 20},
        "candidate_cache_hash": "a" * 64,
    }
    before = paper_artifacts._paper_run_manifest_hash(scientific)
    resumed = {
        **scientific,
        "status": "IN_PROGRESS",
        "commands_executed": [["graspnet6d", "all", "--resume"]],
        "last_resumed_at_utc": "2026-08-16T12:00:00+00:00",
        "status_history": [
            {
                "status": "COMPLETE",
                "ended_at_utc": "2026-08-16T11:00:00+00:00",
                "resumed_at_utc": "2026-08-16T12:00:00+00:00",
            }
        ],
        "formal_results_emitted": False,
    }

    assert paper_artifacts._paper_run_manifest_hash(resumed) == before
    resumed["candidate_cache_hash"] = "b" * 64
    assert paper_artifacts._paper_run_manifest_hash(resumed) != before


def _resumable_inputs(run_dir: Path, fingerprint: str) -> FormalPaperInputs:
    return FormalPaperInputs(
        run_dir=run_dir,
        run_manifest=json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        ),
        config={},
        target_rows=(),
        language_rows=(),
        grounding=pd.DataFrame(),
        candidate_summary=pd.DataFrame(),
        runtime=pd.DataFrame(),
        selected_condition="hifics_zero_shot_mask",
        analyses={},
        source_hashes={},
        input_fingerprint=fingerprint,
    )


def test_completed_paper_resume_revalidates_without_rewriting_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "formal-run"
    run_dir.mkdir()
    completed_at = "2026-08-16T11:00:00+00:00"
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "IN_PROGRESS",
                "formal_results_emitted": False,
                "status_history": [
                    {"status": "COMPLETE", "ended_at_utc": completed_at}
                ],
            }
        ),
        encoding="utf-8",
    )
    output = run_dir / "RESULTS.md"
    output.write_text("immutable result\n", encoding="utf-8")
    fingerprint = "1" * 64
    paper_manifest = run_dir / "paper_artifact_manifest.json"
    paper_manifest.write_text(
        json.dumps(
            {
                "schema_version": paper_artifacts.PAPER_ARTIFACT_SCHEMA,
                "status": "COMPLETE",
                "scope": paper_artifacts.FORMAL_SCOPE,
                "run_id": run_dir.name,
                "input_fingerprint": fingerprint,
                "selected_predicted_condition": "hifics_zero_shot_mask",
                "outputs": {"RESULTS.md": sha256_file(output)},
                "figures": {"executed": ["01"], "unexecuted": []},
            }
        ),
        encoding="utf-8",
    )
    inputs = _resumable_inputs(run_dir, fingerprint)
    monkeypatch.setattr(
        paper_artifacts, "validate_formal_paper_inputs", lambda path: inputs
    )
    original_manifest = paper_manifest.read_bytes()
    original_output = output.read_bytes()

    result = generate_formal_paper_artifacts(run_dir, resume=True)

    assert result.resumed is True
    assert paper_manifest.read_bytes() == original_manifest
    assert output.read_bytes() == original_output
    terminal = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "COMPLETE"
    assert terminal["formal_results_emitted"] is True
    assert terminal["ended_at_utc"] == completed_at

    stale = _resumable_inputs(run_dir, "2" * 64)
    monkeypatch.setattr(
        paper_artifacts, "validate_formal_paper_inputs", lambda path: stale
    )
    with pytest.raises(PaperArtifactsRefused, match="manifest is stale"):
        generate_formal_paper_artifacts(run_dir, resume=True)
    assert paper_manifest.read_bytes() == original_manifest
    assert output.read_bytes() == original_output
