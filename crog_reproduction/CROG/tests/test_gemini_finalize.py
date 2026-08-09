from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from failure_analysis.gemini_crog_evidence_v1.finalize import (
    ER2_COST_DISCLAIMER,
    FinalizationError,
    aggregate_evidence,
    append_command_log,
    audit_cache,
    audit_manifest_request_counts,
    build_logical_request_manifest,
    cache_derived_artifacts,
    publish_highest_complete_phase,
    secret_scan,
    write_docs,
    write_environment,
)
from failure_analysis.gemini_crog_evidence_v1.protocol import file_identity


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _cache(path: Path, *, duplicate_success: bool = False) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE responses (
          request_hash TEXT PRIMARY KEY, sample_id TEXT, model_id TEXT, usage_json TEXT,
          valid INTEGER, latency_seconds REAL, estimated_charge_usd REAL,
          fallback_reason TEXT, parsed_output_json TEXT, mapping_json TEXT,
          protocol_id TEXT, replicate_id TEXT, namespace TEXT
        );
        CREATE TABLE request_states (request_hash TEXT PRIMARY KEY, status TEXT);
        CREATE TABLE request_attempts (
          attempt_id INTEGER PRIMARY KEY, request_hash TEXT, attempt_number INTEGER,
          status TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "h1", "sample-1", "gemini-3.6-flash",
            json.dumps({"total_input_tokens": 10, "total_output_tokens": 4, "total_thought_tokens": 3}),
            1, 2.0, 0.01, None,
            json.dumps({"ranking": [{"candidate_id": "A", "overall_score": 0.9, "reason_codes": ["ok"]}]}),
            json.dumps({"display_to_candidate": {"A": "candidate_0"}}), None, None, None,
        ),
    )
    connection.execute("INSERT INTO request_states VALUES ('h1','SUCCEEDED')")
    connection.execute("INSERT INTO request_attempts VALUES (1,'h1',1,'SUCCEEDED')")
    if duplicate_success:
        connection.execute("INSERT INTO request_attempts VALUES (2,'h1',2,'SUCCEEDED')")
    connection.commit()
    connection.close()


def _plan(root: Path) -> None:
    phases = []
    slot_sum = 0
    for phase in ("pilot", "stability", "ablation", "calibration", "validation", "formal_test"):
        protocols = ["a", "b", "c", "d"] if phase == "ablation" else ["p1"]
        replicates = [1, 2, 3] if phase == "stability" else [0]
        slots = len(protocols) * len(replicates) * 2
        slot_sum += slots
        _json(root / f"{phase}_manifest.json", {"rows": [{"sample_id": f"{phase}-1"}]})
        phases.append(
            {
                "phase": phase,
                "sample_count": 1,
                "protocols": protocols,
                "replicates": replicates,
                "model_ids": ["er2", "flash"],
                "planned_request_slots": slots,
                "new_requests": slots,
                "cache_hits": 0,
            }
        )
    _json(
        root / "full_run_plan.json",
        {
            "phases": phases,
            "totals": {
                "planned_request_slots": slot_sum,
                "planned_unique_requests": slot_sum,
                "new_requests": slot_sum,
                "existing_cache_hits": 0,
                "cross_phase_cache_hits": 0,
            },
        },
    )


def _evidence_dir(path: Path, sample: str, stable: str) -> None:
    path.mkdir(parents=True)
    _json(path / "export_summary.json", {"status": "complete"})
    pq.write_table(
        pa.Table.from_pylist([{"sample_id": sample, "stable_candidate_id": stable, "value": 1.0}]),
        path / "candidate_evidence.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"sample_id": sample, "mapping_json": "{}"}]),
        path / "candidate_mapping.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"sample_id": sample, "board_sha256": "board"}]),
        path / "request_manifest.parquet",
    )


def test_cache_audit_rejects_duplicate_success(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    _cache(path, duplicate_success=True)
    with pytest.raises(FinalizationError, match="duplicate successful"):
        audit_cache(path)


def test_cache_audit_reports_unique_responses(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    _cache(path)
    result = audit_cache(path)
    assert result["status"] == "passed"
    assert result["valid_response_count"] == 1
    assert result["duplicate_successful_request_hash_count"] == 0


def test_manifest_request_arithmetic_and_mismatch(tmp_path: Path) -> None:
    _plan(tmp_path)
    result = audit_manifest_request_counts(tmp_path, enforce_canonical_counts=False)
    assert result["planned_request_slots"] == 22
    plan = json.loads((tmp_path / "full_run_plan.json").read_text())
    plan["phases"][0]["planned_request_slots"] += 1
    _json(tmp_path / "full_run_plan.json", plan)
    with pytest.raises(FinalizationError, match="request slots"):
        audit_manifest_request_counts(tmp_path, enforce_canonical_counts=False)


def test_logical_request_manifest_has_one_row_per_preregistered_slot(tmp_path: Path) -> None:
    _plan(tmp_path)
    result = build_logical_request_manifest(tmp_path)
    assert result["planned_slots"] == 22
    rows = pq.read_table(tmp_path / "request_manifest.parquet").to_pylist()
    assert len(rows) == len({row["logical_request_key"] for row in rows}) == 22
    assert all(row["lifecycle_status"] == "PLANNED" for row in rows)


def test_secret_scan_fails_without_disclosing_secret(tmp_path: Path) -> None:
    secret = "AQ." + "x" * 28
    (tmp_path / "artifact.json").write_text(json.dumps({"bad": secret}))
    with pytest.raises(FinalizationError) as error:
        secret_scan([tmp_path], secret_values=[secret])
    assert secret not in str(error.value)
    assert "28" not in str(error.value)


def test_secret_scan_ignores_secure_env_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GEMINI_API_KEY=AQ." + "x" * 28)
    assert secret_scan([tmp_path])["credential_match_file_count"] == 0


def test_evidence_aggregation_deduplicates_identical_rows(tmp_path: Path) -> None:
    _evidence_dir(tmp_path / "smoke_evidence_final", "s1", "s1/c0")
    _evidence_dir(tmp_path / "pilot" / "evidence", "s1", "s1/c0")
    _evidence_dir(tmp_path / "ablation" / "evidence", "s2", "s2/c0")
    result = aggregate_evidence(tmp_path)
    assert result["candidate_evidence"]["row_count"] == 2
    assert result["candidate_evidence"]["deduplicated_identical_row_count"] == 1
    assert pq.read_table(tmp_path / "per_candidate_evidence.parquet").num_rows == 2


def test_superseded_evidence_is_not_aggregated(tmp_path: Path) -> None:
    source = tmp_path / "smoke_evidence"
    _evidence_dir(source, "bad", "bad/c0")
    _json(source / "SUPERSEDED.json", {"status": "superseded"})
    _evidence_dir(tmp_path / "smoke_evidence_final", "good", "good/c0")
    result = aggregate_evidence(tmp_path)
    assert result["candidate_evidence"]["row_count"] == 1


def test_cache_derived_artifacts_preserve_er2_disclaimer(tmp_path: Path) -> None:
    _cache(tmp_path / "gemini_cache.sqlite")
    _json(tmp_path / "full_run_plan.json", {"totals": {"planned_unique_requests": 1, "expected_cost": {}}})
    _json(tmp_path / "full_run_preflight.json", {"environment": {"gemini_er2_cost_cap_per_request_usd": 1.0}})
    result = cache_derived_artifacts(tmp_path)
    actual = json.loads((tmp_path / "actual_cost_estimate.json").read_text())
    assert actual["er2_cost_note"] == ER2_COST_DISCLAIMER
    assert result["score_rows"] == 1
    scores = pq.read_table(tmp_path / "per_candidate_gemini_scores.parquet").to_pylist()
    assert scores[0]["stable_candidate_id"] == "sample-1/candidate_0"


def test_partial_publication_does_not_claim_formal(tmp_path: Path) -> None:
    result = publish_highest_complete_phase(tmp_path)
    assert result["publication_scope"] == "partial"
    assert result["formal_complete"] is False


def test_formal_marker_requires_lock_and_recompute(tmp_path: Path) -> None:
    artifact = tmp_path / "formal_test" / "per_sample_predictions.parquet"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"bound formal artifact")
    _json(
        tmp_path / "formal_test" / "PHASE_COMPLETE.json",
        {
            "phase": "formal_test",
            "status": "complete",
            "run_id": "formal-run",
            "artifact_identities": [file_identity(artifact)],
        },
    )
    with pytest.raises(FinalizationError, match="valid lock and recomputation"):
        publish_highest_complete_phase(tmp_path)


def test_environment_contains_status_not_secret(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    secret = "AQ." + "z" * 28
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    payload = write_environment(tmp_path)
    serialized = (tmp_path / "environment.json").read_text()
    assert payload["environment_variable_status"]["GEMINI_API_KEY"] == "SET"
    assert secret not in serialized


def test_command_log_is_idempotent(tmp_path: Path) -> None:
    append_command_log(tmp_path)
    append_command_log(tmp_path)
    lines = (tmp_path / "commands.log").read_text().splitlines()
    assert len(lines) == 1


def test_docs_are_upserted_and_partial_is_explicit(tmp_path: Path) -> None:
    bundle = {
        "generated_at_utc": "2026-08-01T00:00:00Z",
        "status": "blocked_credentials",
        "publication_scope": "partial",
        "execution_status": {"pilot": "pending"},
        "baseline_integrity": {"status": "passed", "baseline": {}},
        "metrics": [],
        "formal_results_available": False,
        "formal_lock": {"valid": False},
        "secret_scan": {"status": "passed"},
        "cache_integrity": {"duplicate_successful_request_hash_count": 0},
        "manifest_integrity": {"status": "passed"},
    }
    write_docs(bundle, tmp_path)
    write_docs(bundle, tmp_path)
    text = (tmp_path / "CROG_GEMINI_EVIDENCE_V1_RESULTS.md").read_text()
    assert text.count("GEMINI-OFFLINE-FINALIZER:START") == 1
    assert "No incomplete phase is presented as a formal result" in text
    assert ER2_COST_DISCLAIMER in text
