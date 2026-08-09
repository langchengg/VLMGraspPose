from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from failure_analysis.gemini_crog_evidence_v1.cli import build_parser
from failure_analysis.gemini_crog_evidence_v1.protocol import (
    build_lock_payload,
    candidate_identity_stream_sha256,
    canonical_json,
    claim_formal_test_once,
    file_identity,
    lock_experiment,
    verify_experiment_lock,
    verify_lock_payload,
)
from failure_analysis.gemini_crog_evidence_v1.schema import response_json_schema


def _write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8")
    return path


def _json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _valid_payload(tmp_path: Path, *, run_id: str = "formal-run", primary: str = "flash_safe"):
    checkpoint = _write(tmp_path / "checkpoint.pt", "checkpoint")
    config = _write(tmp_path / "config.yaml", "batch_size: 16\n")
    candidates = _write(
        tmp_path / "features.jsonl",
        json.dumps(
            {
                "split": "test",
                "sample_id": 0,
                "candidates": [
                    {"candidate_id": f"candidate_{index}", "q_raw": 1.0 - index / 10, "candidate_checksum": f"sum-{index}"}
                    for index in range(5)
                ],
            }
        )
        + "\n",
    )
    split = _json(tmp_path / "split.json", {"split": "frozen"})
    protocol_lock = _json(tmp_path / "development_protocol_lock.json", {"protocol": "A2"})
    renderer = _write(tmp_path / "renderer.py", "def render(): pass\n")
    prompt = _write(tmp_path / "prompt.txt", "rank frozen candidates\n")
    schema_source = _write(tmp_path / "schema.py", "SCHEMA = 'frozen'\n")
    evidence_schema = _json(tmp_path / "evidence_schema.json", {"version": 1})
    thresholds_path = _json(tmp_path / "thresholds.json", {"flash": {"confidence": 0.8}})
    primary_path = _json(tmp_path / "primary.json", {"locked_primary": primary})
    validation_path = _json(tmp_path / "validation.json", {"legacy_net": 3})
    run_plan = _json(tmp_path / "full_run_plan.json", {"status": "ready"})
    cohort = _json(tmp_path / "formal_test_manifest.json", {"sample_count": 17_749})
    calibration_grid = _json(tmp_path / "calibration_grid.json", {"confidence": [0.5, 0.55]})
    validation_stats = _json(tmp_path / "statistical_tests.json", {"status": "complete"})
    legacy_labels = _write(tmp_path / "legacy_labels.jsonl", "{}\n")
    corrected_labels = _write(tmp_path / "corrected_labels.jsonl", "{}\n")
    raw_predictions = _write(tmp_path / "predictions.jsonl", "{}\n")
    evaluator = _write(tmp_path / "evaluator.py", "def evaluate(): pass\n")
    source = _write(tmp_path / "full_run.py", "def run(): pass\n")
    thresholds = json.loads(thresholds_path.read_text())
    primary_values = json.loads(primary_path.read_text())
    validation = json.loads(validation_path.read_text())
    schema_sha = hashlib.sha256(canonical_json(response_json_schema()).encode()).hexdigest()
    return build_lock_payload(
        experiment_id="gemini-crog-v1",
        run_id=run_id,
        locked_at_utc="2026-08-01T12:00:00Z",
        source_code=[file_identity(source)],
        full_run_plan=file_identity(run_plan),
        cohort_manifests={"formal_test": file_identity(cohort)},
        git_commit="deadbeef",
        git_diff_sha256="diff",
        checkpoint=file_identity(checkpoint),
        config=file_identity(config),
        baseline_candidates={
            "features": file_identity(candidates),
            "candidate_identity_stream_sha256": candidate_identity_stream_sha256(candidates),
        },
        split_manifest=file_identity(split),
        development_protocol_lock=file_identity(protocol_lock),
        renderer=file_identity(renderer),
        prompt=file_identity(prompt),
        response_schema={"sha256": schema_sha, "source": file_identity(schema_source)},
        evidence_schema=file_identity(evidence_schema),
        candidate_permutation={
            **file_identity(renderer),
            "algorithm": "deterministic_candidate_mapping(seed=47)",
        },
        model_ids=["gemini-robotics-er-2-preview", "gemini-3.6-flash"],
        sdk="google-genai==2.16.0",
        endpoint="https://generativelanguage.googleapis.com/v1beta/interactions",
        store=False,
        background=False,
        stream=False,
        tools_enabled=False,
        previous_interaction=None,
        thinking_level="medium",
        temperature_policy="model_default",
        image_resolution="high",
        max_output_tokens=4096,
        safe_thresholds={"values": thresholds, "source": file_identity(thresholds_path)},
        calibration_grid=file_identity(calibration_grid),
        harmful_cap=0.01,
        primary_selection_rule="validation-only",
        primary_method=primary,
        primary_selection={"values": primary_values, "source": file_identity(primary_path)},
        secondary_methods=["er2_safe", "consensus_safe"],
        request_hash_algorithm="sha256 canonical JSON v1",
        cache_schema="responses v1 + request_state v1",
        budget={"max_spend_usd": 100.0, "er2_cost_cap_per_request_usd": 0.1},
        retry_policy={"max_retries": 5},
        concurrency=2,
        transport="standard_interactions",
        validation_metrics={"values": validation, "source": file_identity(validation_path)},
        validation_artifacts={"statistical_tests.json": file_identity(validation_stats)},
        ground_truth_inputs={
            "legacy_labels": file_identity(legacy_labels),
            "corrected_labels": file_identity(corrected_labels),
            "raw_predictions_with_gt": file_identity(raw_predictions),
        },
        formal_test_expected_sample_count=17_749,
        formal_test_expected_request_count=35_498,
        evaluator={
            "legacy": {"definition": "legacy", "source": file_identity(evaluator)},
            "corrected": {"definition": "corrected", "source": file_identity(evaluator)},
        },
    )


def test_verify_payload_recomputes_lock_and_external_artifacts(tmp_path: Path):
    result = verify_lock_payload(_valid_payload(tmp_path))
    assert result["status"] == "verified"
    assert result["run_id"] == "formal-run"
    assert result["primary_method"] == "flash_safe"
    assert result["verified_artifact_count"] >= 14


def test_verify_experiment_lock_reports_file_and_payload_hashes(tmp_path: Path):
    lock = tmp_path / "lock.json"
    payload = _valid_payload(tmp_path)
    lock_experiment(lock, payload, dry_run=False)
    result = verify_experiment_lock(lock, expected_run_id="formal-run")
    assert result["lock_sha256"] == payload["lock_sha256"]
    assert len(result["file_sha256"]) == 64


def test_payload_tampering_is_detected_before_semantic_use(tmp_path: Path):
    payload = _valid_payload(tmp_path)
    payload["primary_method"] = "er2_safe"
    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_lock_payload(payload)


def test_external_threshold_mutation_invalidates_lock(tmp_path: Path):
    payload = _valid_payload(tmp_path)
    lock = tmp_path / "lock.json"
    lock_experiment(lock, payload, dry_run=False)
    Path(payload["safe_thresholds"]["source"]["path"]).write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_experiment_lock(lock)


def test_primary_artifact_must_agree_with_embedded_primary(tmp_path: Path):
    payload = _valid_payload(tmp_path, primary="flash_safe")
    payload["primary_selection"]["values"]["locked_primary"] = "er2_safe"
    unsigned = dict(payload)
    unsigned.pop("lock_sha256")
    payload["lock_sha256"] = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    with pytest.raises(ValueError, match="disagrees"):
        verify_lock_payload(payload)


def test_response_schema_runtime_identity_is_recomputed(tmp_path: Path):
    payload = _valid_payload(tmp_path)
    payload["response_schema"]["sha256"] = "wrong"
    unsigned = dict(payload)
    unsigned.pop("lock_sha256")
    payload["lock_sha256"] = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    with pytest.raises(ValueError, match="response schema identity changed"):
        verify_lock_payload(payload)


def test_candidate_identity_stream_is_recomputed(tmp_path: Path):
    payload = _valid_payload(tmp_path)
    payload["baseline_candidates"]["candidate_identity_stream_sha256"] = "wrong"
    unsigned = dict(payload)
    unsigned.pop("lock_sha256")
    payload["lock_sha256"] = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    with pytest.raises(ValueError, match="candidate identity stream changed"):
        verify_lock_payload(payload)


def test_threshold_values_must_match_external_json(tmp_path: Path):
    payload = _valid_payload(tmp_path)
    payload["safe_thresholds"]["values"] = {"different": True}
    unsigned = dict(payload)
    unsigned.pop("lock_sha256")
    payload["lock_sha256"] = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    with pytest.raises(ValueError, match="values disagree"):
        verify_lock_payload(payload)


def test_dry_run_is_non_mutating_and_returns_proposed_hash(tmp_path: Path):
    lock = tmp_path / "lock.json"
    payload = _valid_payload(tmp_path)
    result = lock_experiment(lock, payload, dry_run=True)
    assert result["status"] == "dry_run"
    assert result["lock_sha256"] == payload["lock_sha256"]
    assert not lock.exists()


def test_formal_lock_is_open_x_immutable(tmp_path: Path):
    lock = tmp_path / "lock.json"
    payload = _valid_payload(tmp_path)
    lock_experiment(lock, payload, dry_run=False)
    original = lock.read_bytes()
    with pytest.raises(FileExistsError):
        lock_experiment(lock, payload, dry_run=False)
    assert lock.read_bytes() == original


def test_formal_claim_resumes_same_lock_and_run_id(tmp_path: Path):
    lock, claim = tmp_path / "lock.json", tmp_path / "claim.json"
    lock_experiment(lock, _valid_payload(tmp_path), dry_run=False)
    first = claim_formal_test_once(claim, lock, run_id="formal-run")
    original = claim.read_bytes()
    second = claim_formal_test_once(claim, lock, run_id="formal-run")
    assert first["status"] == "claimed"
    assert second["status"] == "resumed"
    assert claim.read_bytes() == original


def test_formal_claim_rejects_a_second_run_id(tmp_path: Path):
    lock, claim = tmp_path / "lock.json", tmp_path / "claim.json"
    lock_experiment(lock, _valid_payload(tmp_path), dry_run=False)
    claim_formal_test_once(claim, lock, run_id="formal-run")
    with pytest.raises(ValueError, match="run_id mismatch"):
        claim_formal_test_once(claim, lock, run_id="second-run")


def test_cli_exposes_verify_and_claim_commands():
    parser = build_parser()
    verify = parser.parse_args(["verify-experiment-lock", "--lock", "lock.json"])
    claim = parser.parse_args(
        ["claim-formal-test", "--lock", "lock.json", "--claim", "claim.json", "--run-id", "run"]
    )
    assert verify.command == "verify-experiment-lock"
    assert claim.command == "claim-formal-test"
