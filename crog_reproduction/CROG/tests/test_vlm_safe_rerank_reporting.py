from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import sqlite3
from pathlib import Path

import pytest

import failure_analysis.vlm_safe_rerank.reporting as reporting_module
from failure_analysis.vlm_safe_rerank.reporting import ReportingError, generate_reports
from failure_analysis.vlm_safe_rerank.security import secret_pattern_hits


REQUIRED = {
    "SUMMARY.md",
    "GO_NO_GO.md",
    "RESULTS.json",
    "METRICS.csv",
    "API_LEDGER_SUMMARY.json",
    "COST_REPORT.md",
    "LATENCY_REPORT.md",
    "FAILURE_ANALYSIS.md",
    "REPRODUCE.md",
    "MANIFEST.json",
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _metric(*, total: int, baseline: int, selected: int, recovered: int, harmful: int, switches: int) -> dict:
    return {
        "total": total,
        "baseline_successes": baseline,
        "baseline_j1": baseline / total,
        "final_successes": selected,
        "final_j1": selected / total,
        "delta_pp": 100.0 * (selected - baseline) / total,
        "recovered": recovered,
        "harmful": harmful,
        "net": recovered - harmful,
        "switches": switches,
        "switch_rate": switches / total,
        "harm_rate": harmful / baseline if baseline else None,
        "outcome_changing_precision": recovered / (recovered + harmful)
        if recovered + harmful
        else 0.0,
    }


def _create_ledger(path: Path, *, duplicate_success: bool = False) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE requests (
          request_hash TEXT PRIMARY KEY,
          status TEXT NOT NULL
        );
        CREATE TABLE attempts (
          attempt_id INTEGER PRIMARY KEY,
          request_hash TEXT NOT NULL,
          status TEXT NOT NULL,
          latency_seconds REAL,
          estimated_cost_usd REAL NOT NULL,
          error_class TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO requests VALUES (?,?)",
        [("request-a", "SUCCEEDED"), ("request-b", "PERMANENT_FAILED")],
    )
    attempts = [
        (1, "request-a", "SUCCEEDED", 2.0, 0.02, None),
        (2, "request-b", "RETRYABLE_FAILED", 4.0, 0.0, "RateLimitError"),
        (3, "request-b", "PERMANENT_FAILED", 6.0, 0.0, "TransportError"),
    ]
    if duplicate_success:
        attempts.append((4, "request-a", "SUCCEEDED", 3.0, 0.02, None))
    connection.executemany("INSERT INTO attempts VALUES (?,?,?,?,?,?)", attempts)
    connection.commit()
    connection.close()


def _run_fixture(
    root: Path,
    *,
    validation_status: str = "NO_GO",
    primary: str = "q_only",
    formal_flag: bool = False,
    duplicate_success: bool = False,
) -> Path:
    root.mkdir(parents=True)
    _write_json(
        root / "DATA_MANIFEST.json",
        {
            "run_id": "safe-fixture",
            "expected_denominator": 4,
            "metadata": {"corrected_evaluator_primary": True},
        },
    )
    _write_json(
        root / "INFERENCE_MANIFEST.json",
        {
            "run_id": "safe-fixture",
            "expected_denominator": 4,
            "inference_inputs": {
                "default_output": "q_only_c0",
                "fallback_policy": "keep q-only",
                "formal_requests_authorized": formal_flag,
            },
        },
    )
    _write_json(
        root / "p1_direct_diagnostic" / "P1_DIAGNOSTIC_FREEZE.json",
        {
            "decision_count": 8,
            "diagnostic_NO_GO": True,
            "eligible_for_primary": False,
            "ledger": {"attempts": 9, "attempt_estimated_charge_usd": 0.2},
            "models": {
                "gemini-3.6-flash": {
                    "decisions": 4,
                    "schema_valid": 3,
                    "fallback": 1,
                    "terminal_failures": 1,
                    "latency_p50_seconds": 10.0,
                    "latency_p95_seconds": 20.0,
                    "legacy": {
                        "total": 4, "q_only_j1": 0.5, "selected_j1": 0.25,
                        "recovered": 0, "harmful": 1, "net": -1,
                        "switches": 1, "switch_rate": 0.25,
                    },
                    "corrected": {
                        "total": 4, "q_only_j1": 0.75, "selected_j1": 0.5,
                        "recovered": 0, "harmful": 1, "net": -1,
                        "switches": 1, "switch_rate": 0.25,
                    },
                },
                "gemini-robotics-er-2-preview": {
                    "decisions": 4,
                    "schema_valid": 4,
                    "fallback": 0,
                    "terminal_failures": 0,
                    "latency_p50_seconds": 8.0,
                    "latency_p95_seconds": 12.0,
                    "legacy": {
                        "total": 4, "q_only_j1": 0.5, "selected_j1": 0.5,
                        "recovered": 1, "harmful": 1, "net": 0,
                        "switches": 2, "switch_rate": 0.5,
                    },
                    "corrected": {
                        "total": 4, "q_only_j1": 0.75, "selected_j1": 0.75,
                        "recovered": 0, "harmful": 0, "net": 0,
                        "switches": 1, "switch_rate": 0.25,
                    },
                },
            },
        },
    )
    _write_json(
        root / "smoke" / "API_SUMMARY.json",
        {
            "model_pair_variants": 4,
            "successful": 3,
            "fallback": 1,
            "schema_valid_rate": 0.75,
            "ledger": {"attempts": 5, "attempt_estimated_cost_usd": 0.03},
        },
    )
    _write_json(
        root / "smoke" / "DIAGNOSTIC_RESULTS.json",
        {
            "models": {
                "gemini-3.6-flash": {
                    "status_counts": {"PERMANENT_FAILED": 1, "SUCCEEDED": 1},
                    "corrected_hard_rule": _metric(
                        total=2, baseline=1, selected=1, recovered=0, harmful=0, switches=0
                    ),
                },
                "gemini-robotics-er-2-preview": {
                    "status_counts": {"SUCCEEDED": 2},
                    "corrected_hard_rule": _metric(
                        total=2, baseline=1, selected=2, recovered=1, harmful=0, switches=1
                    ),
                },
            }
        },
    )
    _write_json(
        root / "diagnostic" / "API_PROGRESS.json",
        {
            "completed_model_pair_variants": 4,
            "planned_model_pair_variants": 8,
            "ledger": {"attempts": 6},
        },
    )
    _write_json(
        root / "diagnostic" / "DIAGNOSTIC_RESULTS.json",
        {
            "models": {
                "gemini-3.6-flash": {
                    "status_counts": {"PERMANENT_FAILED": 1, "SUCCEEDED": 2}
                },
                "gemini-robotics-er-2-preview": {
                    "status_counts": {"SUCCEEDED": 3}
                },
            }
        },
    )
    _write_json(
        root / "calibration" / "CALIBRATION_RESULTS.json",
        {
            "method": "P6_local_only_safe_gate",
            "samples": 3,
            "pairs": 6,
            "threshold_state": "no_beneficial_switch",
            "selected_thresholds": {
                "tau": 1.0,
                "eta": 0.0,
                "recovered": 0,
                "harmful": 0,
                "net": 0,
                "switches": 0,
            },
        },
    )
    corrected = _metric(
        total=2,
        baseline=1,
        selected=1 if validation_status == "NO_GO" else 2,
        recovered=0 if validation_status == "NO_GO" else 1,
        harmful=0,
        switches=0 if validation_status == "NO_GO" else 1,
    )
    _write_json(
        root / "validation" / "VALIDATION_RESULTS.json",
        {
            "schema_version": "1.0.0",
            "validation_status": validation_status,
            "primary_method": primary,
            "expected_denominator": 2,
            "corrected": corrected,
            "exact_mcnemar_p": 1.0,
            "scene_sequence_bootstrap_delta_j1": {
                "draws": 100,
                "lower": 0.0,
                "median": 0.0,
                "upper": 0.5 if validation_status == "GO" else 0.0,
            },
            "no_validation_tuning": True,
        },
    )
    _create_ledger(root / "pairwise_cache.sqlite", duplicate_success=duplicate_success)
    return root


def _read_all_text(output: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.iterdir()
        if path.is_file()
    )


def test_no_go_generates_complete_secret_safe_reports_without_formal_claims(
    tmp_path: Path,
) -> None:
    root = _run_fixture(tmp_path / "run")
    output = tmp_path / "reports"
    result = generate_reports(root, output_dir=output)

    assert result["report_status"] == "NO_GO"
    assert result["corrected_primary_method"] == "q_only"
    assert result["formal_run_allowed"] is False
    assert REQUIRED.issubset({path.name for path in output.iterdir()})
    assert not (output / "LOCKED_MANIFEST.json").exists()
    assert not (output / "FORMAL_TEST_REPORT.md").exists()

    results = json.loads((output / "RESULTS.json").read_text())
    assert results["primary_evaluator"] == "corrected"
    assert results["default_action"] == "q_only"
    assert results["q_only_is_default_and_fallback"] is True
    assert results["phase_availability"]["diagnostic"] is True
    assert results["api_phase_progress"]["diagnostic"] == {
        "completed_model_pair_variants": 4,
        "planned_model_pair_variants": 8,
    }
    assert results["denominators"] == {
        "calibration_pairs": 6,
        "calibration_samples": 3,
        "formal_expected": 4,
        "formal_observed": None,
        "p1_decisions": 8,
        "pairwise_attempts": 3,
        "pairwise_requests": 2,
        "validation_expected": 2,
        "validation_observed": 2,
    }
    summary = (output / "SUMMARY.md").read_text()
    assert "corrected evaluator is primary" in summary
    assert "q-only" in summary
    assert "Full formal denominator: **4**" in summary
    failures = (output / "FAILURE_ANALYSIS.md").read_text()
    assert "gemini-3.6-flash" in failures and "gemini-robotics-er-2-preview" in failures
    assert "Fallback is not a successful model decision" in failures

    rows = list(csv.DictReader(io.StringIO((output / "METRICS.csv").read_text())))
    assert {row["stage"] for row in rows} == {
        "p1_diagnostic", "smoke", "calibration", "validation"
    }
    ledger = json.loads((output / "API_LEDGER_SUMMARY.json").read_text())
    assert ledger["requests"] == 2 and ledger["attempts"] == 3
    assert ledger["duplicate_successful_request_hashes"] == 0
    assert ledger["phase_progress"]["diagnostic"]["completed_model_pair_variants"] == 4
    assert ledger["model_failures_from_phase_artifacts"]["gemini-3.6-flash"][
        "diagnostic_status_counts"
    ] == {"PERMANENT_FAILED": 1, "SUCCEEDED": 2}

    manifest = json.loads((output / "MANIFEST.json").read_text())
    for name, identity in manifest["generated_files"].items():
        content = (output / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == identity["sha256"]
        assert len(content) == identity["size_bytes"]
    assert not secret_pattern_hits(_read_all_text(output))


def test_missing_phases_are_partial_and_never_formal(tmp_path: Path) -> None:
    root = _run_fixture(tmp_path / "run")
    (root / "calibration" / "CALIBRATION_RESULTS.json").unlink()
    (root / "validation" / "VALIDATION_RESULTS.json").unlink()
    output = tmp_path / "reports"

    result = generate_reports(root, output_dir=output)
    results = json.loads((output / "RESULTS.json").read_text())

    assert result["report_status"] == "PARTIAL"
    assert results["corrected_primary_method"] == "q_only"
    assert results["formal_complete"] is False
    assert set(results["missing_core_evidence"]) >= {"calibration", "validation"}
    assert not (output / "LOCKED_MANIFEST.json").exists()
    assert not (output / "FORMAL_TEST_REPORT.md").exists()
    assert "partial result is never represented as formal" in (
        output / "SUMMARY.md"
    ).read_text()


def test_inconclusive_completed_validation_is_operational_no_go(tmp_path: Path) -> None:
    root = _run_fixture(
        tmp_path / "run",
        validation_status="INCONCLUSIVE",
        primary="q_only",
        formal_flag=True,
    )
    output = tmp_path / "reports"

    result = generate_reports(root, output_dir=output)
    results = json.loads((output / "RESULTS.json").read_text())

    assert result["report_status"] == "NO_GO"
    assert results["validation"]["status"] == "INCONCLUSIVE"
    assert results["corrected_primary_method"] == "q_only"
    assert result["formal_run_allowed"] is False
    assert not (output / "LOCKED_MANIFEST.json").exists()
    assert not (output / "FORMAL_TEST_REPORT.md").exists()


def test_go_creates_reporting_lock_and_only_complete_formal_creates_report(
    tmp_path: Path,
) -> None:
    root = _run_fixture(
        tmp_path / "run",
        validation_status="GO",
        primary="flash_safe",
        formal_flag=True,
    )
    output = tmp_path / "reports"
    go = generate_reports(root, output_dir=output)

    assert go["report_status"] == "GO"
    assert go["formal_run_allowed"] is True
    assert (output / "REPORTING_LOCK_SNAPSHOT.json").exists()
    assert not (output / "FORMAL_TEST_REPORT.md").exists()
    lock = json.loads((output / "REPORTING_LOCK_SNAPSHOT.json").read_text())
    assert lock["validation_status"] == "GO"
    assert lock["primary_method"] == "flash_safe"
    assert lock["reporting_lock_only"] is True

    _write_json(
        root / "formal_test" / "FORMAL_TEST_RESULTS.json",
        {
            "status": "COMPLETE",
            "primary_method": "flash_safe",
            "total": 4,
            "corrected": _metric(
                total=4, baseline=3, selected=4, recovered=1, harmful=0, switches=1
            ),
        },
    )
    formal = generate_reports(root, output_dir=output)
    assert formal["report_status"] == "FORMAL_COMPLETE"
    assert formal["formal_complete"] is True
    assert (output / "FORMAL_TEST_REPORT.md").exists()
    assert "Observed/expected denominator: **4/4**" in (
        output / "FORMAL_TEST_REPORT.md"
    ).read_text()


def test_duplicate_success_blocks_go_lock_and_formal(tmp_path: Path) -> None:
    root = _run_fixture(
        tmp_path / "run",
        validation_status="GO",
        primary="flash_safe",
        formal_flag=True,
        duplicate_success=True,
    )
    output = tmp_path / "reports"
    result = generate_reports(root, output_dir=output)
    results = json.loads((output / "RESULTS.json").read_text())

    assert result["report_status"] == "PARTIAL"
    assert result["formal_run_allowed"] is False
    assert any("duplicate successful" in item for item in results["blockers"])
    assert not (output / "LOCKED_MANIFEST.json").exists()
    assert not (output / "FORMAL_TEST_REPORT.md").exists()


def test_partial_formal_artifact_never_creates_formal_report(tmp_path: Path) -> None:
    root = _run_fixture(
        tmp_path / "run",
        validation_status="GO",
        primary="flash_safe",
        formal_flag=True,
    )
    _write_json(
        root / "formal_test" / "FORMAL_TEST_RESULTS.json",
        {
            "status": "COMPLETE",
            "primary_method": "flash_safe",
            "total": 3,
            "corrected": _metric(
                total=3, baseline=2, selected=3, recovered=1, harmful=0, switches=1
            ),
        },
    )
    output = tmp_path / "reports"

    result = generate_reports(root, output_dir=output)
    results = json.loads((output / "RESULTS.json").read_text())

    assert result["report_status"] == "GO"
    assert result["formal_complete"] is False
    assert results["denominators"]["formal_expected"] == 4
    assert results["denominators"]["formal_observed"] == 3
    assert any("not a complete full-denominator" in item for item in results["blockers"])
    assert (output / "REPORTING_LOCK_SNAPSHOT.json").exists()
    assert not (output / "FORMAL_TEST_REPORT.md").exists()


def test_secret_scan_fails_before_any_report_is_written(tmp_path: Path) -> None:
    secret = "AQ." + "x" * 24
    root = _run_fixture(
        tmp_path / "run",
        validation_status="GO",
        primary=secret,
        formal_flag=True,
    )
    output = tmp_path / "reports"

    with pytest.raises(ReportingError, match="secret scan failed"):
        generate_reports(root, output_dir=output)
    assert not output.exists()


def test_reporting_module_is_offline_and_dry_run_writes_nothing(tmp_path: Path) -> None:
    tree = ast.parse(Path(reporting_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name.startswith(("google", "requests", "httpx", "aiohttp"))
        or name.endswith(".api")
        for name in imported
    )

    root = _run_fixture(tmp_path / "run")
    output = tmp_path / "reports"
    result = generate_reports(root, output_dir=output, dry_run=True)
    assert result["written"] is False
    assert set(result["files"]) == REQUIRED
    assert not output.exists()
