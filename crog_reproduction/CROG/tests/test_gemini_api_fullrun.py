from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from failure_analysis.gemini_crog_evidence_v1.api import (
    BudgetGuard,
    GeminiCache,
    GoogleInteractionsRunner,
    compare_model_metadata,
    estimate_usage_cost_usd,
    model_metadata_fingerprint,
    request_hash,
    response_metadata,
    sha256_json,
)


FLASH = "gemini-3.6-flash"
ER2 = "gemini-robotics-er-2-preview"
ROOT = Path(__file__).resolve().parents[1]


def _mapping() -> dict:
    forward = {letter: f"candidate_{index}" for index, letter in enumerate("ABCDE")}
    return {
        "display_to_candidate": forward,
        "candidate_to_display": {value: key for key, value in forward.items()},
    }


def _valid_response(*, decision: str = "switch") -> dict:
    return {
        "selected_candidate_id": "B",
        "ranking": [
            {
                "candidate_id": candidate_id,
                "target_alignment_score": 0.9 - index * 0.1,
                "mask_support_score": 0.9 - index * 0.1,
                "quality_evidence_score": 0.9 - index * 0.1,
                "angle_consistency_score": 0.9 - index * 0.1,
                "width_consistency_score": 0.9 - index * 0.1,
                "edge_safety_score": 0.9 - index * 0.1,
                "overall_score": 0.9 - index * 0.1,
                "reason_codes": ["target_alignment"],
            }
            for index, candidate_id in enumerate("BACDE")
        ],
        "confidence": 0.9,
        "score_margin_top1_top2": 0.1,
        "decision": decision,
        "global_reason_codes": ["target_alignment"],
    }


class _Interactions:
    def __init__(self, effects):
        self.effects = list(effects)
        self.calls = 0
        self.kwargs = []

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        effect = self.effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return SimpleNamespace(
            output_text=json.dumps(effect.get("output", _valid_response())),
            usage=effect.get(
                "usage",
                {
                    "total_input_tokens": 1000,
                    "total_output_tokens": 100,
                    "total_thought_tokens": 50,
                    "total_cached_tokens": 0,
                },
            ),
            id=effect.get("id", "interaction-1"),
            model=effect.get("model", FLASH),
            service_tier=effect.get("service_tier", "standard"),
            created=effect.get("created", "2026-08-01T12:00:00Z"),
            updated=effect.get("updated", "2026-08-01T12:00:01Z"),
            status=effect.get("status", "completed"),
        )


class _Client:
    def __init__(self, effects):
        self.interactions = _Interactions(effects)


def _runner_kwargs(tmp_path) -> dict:
    image = tmp_path / "board.png"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
    return {
        "sample_id": "sample-1",
        "frame_id": "frame-1",
        "model_id": FLASH,
        "image_path": image,
        "system_instruction": "Rank the frozen candidates.",
        "metadata_prompt": "Evidence only.",
        "mapping": _mapping(),
        "prompt_hash": "prompt",
        "schema_hash": "schema",
        "renderer_hash": "renderer",
        "evidence_schema_hash": "evidence",
        "request_upper_bound_usd": 0.10,
    }


def _cache_record(digest: str, *, raw_output: str = "first") -> dict:
    return {
        "request_hash": digest,
        "sample_id": "s",
        "frame_id": "f",
        "model_id": FLASH,
        "image_hash": "i",
        "mapping": _mapping(),
        "created_at": "2026-08-01T00:00:00Z",
        "request_id": "r",
        "http_status": 200,
        "latency_seconds": 1.0,
        "usage": {"total_input_tokens": 1},
        "retry_count": 0,
        "raw_output": raw_output,
        "parsed_output": _valid_response(),
        "valid": True,
        "abstain": False,
        "fallback_reason": None,
        "model_metadata": {"requested_model_id": FLASH},
        "prompt_hash": "p",
        "schema_hash": "s",
        "renderer_hash": "r",
        "evidence_schema_hash": "e",
        "generation_config": {"max_output_tokens": 4096},
        "estimated_charge_usd": 0.0,
    }


def _legacy_hash_kwargs() -> dict:
    return {
        "model_id": FLASH,
        "model_metadata": {"api_version": "v1beta"},
        "prompt_hash": "p",
        "schema_hash": "s",
        "renderer_hash": "r",
        "evidence_schema_hash": "e",
        "image_sha256": "image",
        "candidate_mapping": {"A": "candidate_0"},
        "serialized_metadata": "metadata",
        "generation_config": {"max_output_tokens": 4096},
    }


def test_legacy_hash_is_byte_compatible_and_dimensions_isolate_namespaces():
    kwargs = _legacy_hash_kwargs()
    legacy_payload = {
        "model_id": kwargs["model_id"],
        "model_metadata": kwargs["model_metadata"],
        "prompt_hash": kwargs["prompt_hash"],
        "schema_hash": kwargs["schema_hash"],
        "renderer_hash": kwargs["renderer_hash"],
        "evidence_schema_hash": kwargs["evidence_schema_hash"],
        "image_sha256": kwargs["image_sha256"],
        "candidate_mapping_sha256": sha256_json(kwargs["candidate_mapping"]),
        "serialized_metadata_sha256": hashlib.sha256(b"metadata").hexdigest(),
        "generation_config": kwargs["generation_config"],
    }
    legacy = request_hash(**kwargs)
    assert legacy == sha256_json(legacy_payload)
    assert request_hash(**kwargs, protocol_id="A2") != legacy
    assert request_hash(**kwargs, replicate_id=1) != request_hash(
        **kwargs, replicate_id=2
    )
    assert request_hash(**kwargs, replicate_id=1) == request_hash(
        **kwargs, replicate_id="1"
    )
    assert request_hash(**kwargs, namespace="stability") != legacy


def test_v1_26_row_cache_migrates_without_replacing_responses(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE responses (
            request_hash TEXT PRIMARY KEY, sample_id TEXT NOT NULL,
            frame_id TEXT NOT NULL, model_id TEXT NOT NULL,
            image_hash TEXT NOT NULL, mapping_json TEXT NOT NULL,
            created_at TEXT NOT NULL, request_id TEXT, http_status INTEGER,
            latency_seconds REAL NOT NULL, usage_json TEXT NOT NULL,
            retry_count INTEGER NOT NULL, raw_output TEXT,
            parsed_output_json TEXT, valid INTEGER NOT NULL,
            abstain INTEGER NOT NULL, fallback_reason TEXT,
            model_metadata_json TEXT NOT NULL, prompt_hash TEXT NOT NULL,
            schema_hash TEXT NOT NULL, renderer_hash TEXT NOT NULL,
            evidence_schema_hash TEXT NOT NULL,
            generation_config_json TEXT NOT NULL,
            estimated_charge_usd REAL NOT NULL
        )
        """
    )
    for index in range(26):
        valid = index < 20
        connection.execute(
            "INSERT INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"hash-{index}", f"sample-{index}", "frame", FLASH, "image",
                json.dumps(_mapping()), "2026-08-01T00:00:00Z", f"r-{index}",
                200, 1.0, "{}", 0, f"raw-{index}",
                json.dumps(_valid_response()) if valid else None,
                int(valid), 0,
                None if valid else "invalid_structured_response:JSONDecodeError",
                "{}", "p", "s", "r", "e", "{}", 0.1,
            ),
        )
    connection.commit()
    before = connection.execute(
        "SELECT request_hash, raw_output FROM responses ORDER BY request_hash"
    ).fetchall()
    connection.close()

    cache = GeminiCache(path)
    after = cache.connection.execute(
        "SELECT request_hash, raw_output FROM responses ORDER BY request_hash"
    ).fetchall()
    statuses = dict(
        cache.connection.execute(
            "SELECT status, COUNT(*) FROM request_states GROUP BY status"
        ).fetchall()
    )
    assert [tuple(row) for row in after] == before
    assert cache.connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 26
    assert statuses == {"SUCCEEDED": 20, "TECHNICAL_FALLBACK": 6}
    assert cache.get("hash-0")["lifecycle_status"] == "SUCCEEDED"
    assert cache.get("hash-25")["lifecycle_status"] == "TECHNICAL_FALLBACK"
    cache.close()


def test_empty_root_cache_auto_imports_all_26_smoke_responses_with_provenance(
    tmp_path,
):
    root = tmp_path / "experiment"
    source_path = root / "access_smoke_2x1" / "gemini_cache.sqlite"
    source = GeminiCache(source_path)
    for index in range(26):
        record = _cache_record(f"smoke-{index}", raw_output=f"raw-{index}")
        if index >= 20:
            record.update(
                valid=False,
                parsed_output=None,
                raw_output="truncated",
                fallback_reason="invalid_structured_response:JSONDecodeError",
            )
        assert source.put(record)
    source.close()

    root_cache = GeminiCache(root / "gemini_cache.sqlite")
    assert root_cache.connection.execute(
        "SELECT COUNT(*) FROM responses"
    ).fetchone()[0] == 26
    assert tuple(
        root_cache.connection.execute(
            "SELECT SUM(valid), COUNT(*) - SUM(valid) FROM responses"
        ).fetchone()
    ) == (20, 6)
    assert root_cache.connection.execute(
        "SELECT COUNT(*) FROM response_import_provenance"
    ).fetchone()[0] == 26
    audit = root_cache.connection.execute("SELECT * FROM cache_imports").fetchone()
    assert audit["imported_response_count"] == 26
    assert audit["valid_response_count"] == 20
    assert audit["invalid_response_count"] == 6
    imported = root_cache.get("smoke-0")
    assert imported["import_provenance"]["source_request_hash"] == "smoke-0"
    root_cache.close()


def test_response_alias_is_explicit_and_cannot_use_invalid_or_overwrite(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    valid = _cache_record("source")
    valid["generation_config"] = {
        "thinking_level": "medium",
        "max_output_tokens": 3072,
        "image_resolution": "high",
    }
    valid["usage"] = {
        "total_input_tokens": 1000,
        "total_output_tokens": 100,
        "total_thought_tokens": 50,
    }
    assert cache.put(valid)
    target_config = {
        "thinking_level": "medium",
        "max_output_tokens": 4096,
        "image_resolution": "high",
    }
    assert cache.add_response_alias(
        target_request_hash="target",
        source_request_hash="source",
        target_generation_config=target_config,
    )
    assert not cache.add_response_alias(
        target_request_hash="target",
        source_request_hash="source",
        target_generation_config=target_config,
    )
    resolved = cache.get("target")
    assert resolved["cache_hit"]
    assert resolved["request_hash"] == "target"
    assert resolved["source_request_hash"] == "source"
    assert resolved["reuse_reason"] == (
        "paid_valid_response_reused_for_higher_output_token_ceiling"
    )
    assert resolved["source_generation_config"]["max_output_tokens"] == 3072
    assert resolved["target_generation_config"]["max_output_tokens"] == 4096

    invalid = _cache_record("invalid")
    invalid.update(
        valid=False,
        parsed_output=None,
        fallback_reason="invalid_structured_response:JSONDecodeError",
    )
    invalid["generation_config"] = valid["generation_config"]
    assert cache.put(invalid)
    with pytest.raises(ValueError, match="invalid or truncated"):
        cache.add_response_alias(
            target_request_hash="bad-target",
            source_request_hash="invalid",
            target_generation_config=target_config,
        )
    other = _cache_record("other")
    other["generation_config"] = valid["generation_config"]
    assert cache.put(other)
    with pytest.raises(RuntimeError, match="cannot be overwritten"):
        cache.add_response_alias(
            target_request_hash="target",
            source_request_hash="other",
            target_generation_config=target_config,
        )
    missing_id = _cache_record("ordinary-missing-id")
    missing_id["request_id"] = ""
    missing_id["generation_config"] = valid["generation_config"]
    assert cache.put(missing_id)
    with pytest.raises(ValueError, match="paid-response provenance"):
        cache.add_response_alias(
            target_request_hash="ordinary-missing-id-target",
            source_request_hash="ordinary-missing-id",
            target_generation_config=target_config,
        )
    cache.close()


def test_real_smoke_cache_missing_request_ids_reuse_only_via_import_audit(tmp_path):
    experiment = ROOT / (
        "failure_analysis/reranking_outputs/"
        "gemini_crog_evidence_v1_20260801T111025+0100"
    )
    real_source = experiment / "access_smoke_2x1/gemini_cache.sqlite"
    real_manifest = experiment / "smoke_evidence_final/request_manifest.jsonl"
    if not real_source.exists() or not real_manifest.exists():
        pytest.skip("real Phase-B smoke fixture is unavailable")

    copied_source = tmp_path / "experiment/access_smoke_2x1/gemini_cache.sqlite"
    copied_source.parent.mkdir(parents=True)
    source_connection = sqlite3.connect(real_source)
    copied_connection = sqlite3.connect(copied_source)
    try:
        source_connection.backup(copied_connection)
    finally:
        copied_connection.close()
        source_connection.close()

    cache = GeminiCache(tmp_path / "experiment/gemini_cache.sqlite")
    assert tuple(
        cache.connection.execute(
            "SELECT COUNT(*), SUM(valid), SUM(request_id IS NULL OR request_id = '') "
            "FROM responses"
        ).fetchone()
    ) == (26, 20, 26)
    source = cache.connection.execute(
        """
        SELECT * FROM responses
        WHERE valid = 1
          AND json_extract(generation_config_json, '$.max_output_tokens') = 3072
        ORDER BY request_hash LIMIT 1
        """
    ).fetchone()
    candidates = cache.lower_ceiling_candidates(
        sample_id=source["sample_id"],
        model_id=source["model_id"],
        target_max_output_tokens=4096,
    )
    selected = next(
        item for item in candidates if item["request_hash"] == source["request_hash"]
    )
    assert selected["legacy_import_evidence"]["imported_response_count"] == 26

    manifest = {
        item["sample_id"]: item
        for item in map(json.loads, real_manifest.read_text().splitlines())
    }
    request_row = manifest[source["sample_id"]]
    stored_metadata = json.loads(source["model_metadata_json"])
    request_metadata = {
        key: value
        for key, value in stored_metadata.items()
        if key
        not in {
            "response_model",
            "response_id",
            "service_tier",
            "response_created_at",
            "response_updated_at",
            "response_status",
        }
    }
    reconstructed = request_hash(
        model_id=source["model_id"],
        model_metadata=request_metadata,
        prompt_hash=source["prompt_hash"],
        schema_hash=source["schema_hash"],
        renderer_hash=source["renderer_hash"],
        evidence_schema_hash=source["evidence_schema_hash"],
        image_sha256=source["image_hash"],
        candidate_mapping=json.loads(source["mapping_json"]),
        serialized_metadata=request_row["metadata"],
        generation_config=json.loads(source["generation_config_json"]),
    )
    assert reconstructed == source["request_hash"]

    target_config = json.loads(source["generation_config_json"])
    target_config["max_output_tokens"] = 4096
    assert cache.add_response_alias(
        target_request_hash="verified-real-target",
        source_request_hash=source["request_hash"],
        target_generation_config=target_config,
    )
    resolved = cache.get("verified-real-target")
    assert resolved["alias_validation"]["provider_request_id_missing_legacy"] is True
    evidence = resolved["alias_validation"]["legacy_import_evidence"]
    assert evidence["import_reason"] == "immutable_phase_b_smoke_cache_import"
    assert (
        evidence["imported_response_count"],
        evidence["valid_response_count"],
        evidence["invalid_response_count"],
    ) == (26, 20, 6)
    cache.close()


def test_runner_safely_reuses_verified_lower_ceiling_paid_response(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    kwargs = _runner_kwargs(tmp_path)
    image_hash = hashlib.sha256(kwargs["image_path"].read_bytes()).hexdigest()
    source_generation = {
        "thinking_level": "medium",
        "max_output_tokens": 3072,
        "image_resolution": "high",
    }
    model_metadata = {"requested_model_id": FLASH}
    source_digest = request_hash(
        model_id=FLASH,
        model_metadata=model_metadata,
        prompt_hash=kwargs["prompt_hash"],
        schema_hash=kwargs["schema_hash"],
        renderer_hash=kwargs["renderer_hash"],
        evidence_schema_hash=kwargs["evidence_schema_hash"],
        image_sha256=image_hash,
        candidate_mapping=kwargs["mapping"],
        serialized_metadata=kwargs["metadata_prompt"],
        generation_config=source_generation,
    )
    record = _cache_record(source_digest)
    record.update(
        sample_id=kwargs["sample_id"],
        frame_id=kwargs["frame_id"],
        image_hash=image_hash,
        mapping=kwargs["mapping"],
        prompt_hash=kwargs["prompt_hash"],
        schema_hash=kwargs["schema_hash"],
        renderer_hash=kwargs["renderer_hash"],
        evidence_schema_hash=kwargs["evidence_schema_hash"],
        model_metadata=model_metadata,
        generation_config=source_generation,
        usage={
            "total_input_tokens": 1000,
            "total_output_tokens": 100,
            "total_thought_tokens": 50,
        },
    )
    assert cache.put(record)
    client = _Client([])
    result = GoogleInteractionsRunner(
        cache=cache,
        budget=BudgetGuard(1.0),
        api_key="test-only",
        client=client,
    ).run(**kwargs, max_output_tokens=4096)
    assert result["cache_hit"] and result["source_request_hash"] == source_digest
    assert result["request_hash"] != source_digest
    assert client.interactions.calls == 0
    cache.close()


def test_single_owner_claim_and_stale_lease_recovery_are_transactional(tmp_path):
    path = tmp_path / "cache.sqlite"
    first = GeminiCache(path)
    second = GeminiCache(path)
    first.plan_request(request_hash="h", sample_id="s", model_id=FLASH)
    claim_one = first.claim_request(
        "h", owner_id="owner-1", lease_seconds=10, now_epoch=100
    )
    claim_two = second.claim_request(
        "h", owner_id="owner-2", lease_seconds=10, now_epoch=105
    )
    recovered = second.claim_request(
        "h", owner_id="owner-2", lease_seconds=10, now_epoch=111
    )
    assert claim_one.acquired
    assert not claim_two.acquired and claim_two.reason == "active_lease"
    assert recovered.acquired and recovered.stale_lease_recovered
    assert second.request_state("h")["owner_id"] == "owner-2"
    attempts = second.attempts("h")
    assert len(attempts) == 1
    assert attempts[0]["status"] == "RETRYABLE_FAILED"
    assert attempts[0]["error_message"] == "stale_lease_recovered"
    first.close()
    second.close()


def test_claim_budget_check_includes_existing_outstanding_reserve(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    for digest in ("one", "two"):
        cache.plan_request(request_hash=digest, sample_id=digest, model_id=FLASH)
    assert cache.claim_request(
        "one", owner_id="a", lease_seconds=100, reserved_usd=0.6,
        max_spend_usd=1.0, already_estimated_usd=0.0,
    ).acquired
    blocked = cache.claim_request(
        "two", owner_id="b", lease_seconds=100, reserved_usd=0.5,
        max_spend_usd=1.0, already_estimated_usd=0.0,
    )
    assert not blocked.acquired and blocked.reason == "budget_would_be_exceeded"
    assert cache.request_state("two")["status"] == "PLANNED"
    cache.close()


def test_attempt_rows_and_final_responses_are_append_only(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    cache.plan_request(request_hash="h", sample_id="s", model_id=FLASH)
    cache.claim_request("h", owner_id="owner", lease_seconds=10)
    cache.append_attempt(
        "h", owner_id="owner", started_at="a", completed_at="b",
        status="SUCCEEDED", latency_seconds=1.0,
    )
    cache.append_attempt(
        "h", owner_id="owner", started_at="c", completed_at="d",
        status="RETRYABLE_FAILED", latency_seconds=2.0,
    )
    assert [row["attempt_number"] for row in cache.attempts("h")] == [1, 2]
    assert cache.put(_cache_record("h", raw_output="first"))
    assert not cache.put(_cache_record("h", raw_output="must-not-overwrite"))
    assert cache.get("h")["raw_output"] == "first"
    schema = cache.connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'responses'"
    ).fetchone()[0]
    assert "PRIMARY KEY" in schema
    cache.close()


def test_cached_response_repairs_crash_window_state_and_lease(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    cache.plan_request(request_hash="h", sample_id="s", model_id=FLASH)
    cache.claim_request("h", owner_id="crashed", lease_seconds=100)
    assert cache.put(_cache_record("h"))
    assert cache.request_state("h")["status"] == "IN_FLIGHT"
    cached = cache.get("h")
    assert cached["lifecycle_status"] == "SUCCEEDED"
    assert cache.request_state("h")["status"] == "SUCCEEDED"
    assert cache.request_state("h")["owner_id"] is None
    assert cache.connection.execute(
        "SELECT COUNT(*) FROM request_leases WHERE request_hash = 'h'"
    ).fetchone()[0] == 0
    cache.close()


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (ER2, 1000 * 2.0e-6 + 150 * 10.0e-6),
        (FLASH, 1000 * 1.5e-6 + 150 * 7.5e-6),
    ],
)
def test_official_standard_token_pricing(model_id, expected):
    usage = {
        "total_input_tokens": 1000,
        "total_output_tokens": 100,
        "total_thought_tokens": 50,
    }
    assert estimate_usage_cost_usd(model_id, usage) == pytest.approx(expected)


def test_cached_input_uses_cache_rate():
    usage = {
        "total_input_tokens": 1000,
        "total_cached_tokens": 400,
        "total_output_tokens": 0,
        "total_thought_tokens": 0,
    }
    assert estimate_usage_cost_usd(FLASH, usage) == pytest.approx(
        (600 * 1.5 + 400 * 0.15) / 1_000_000
    )


def test_budget_guard_accounts_for_settled_outstanding_next_and_er2_cap():
    guard = BudgetGuard(
        1.0,
        already_estimated_usd=0.4,
        er2_cost_cap_per_request_usd=0.05,
    )
    allowed = guard.check(FLASH, 0.2, outstanding_request_reserve_usd=0.3)
    blocked = guard.check(FLASH, 0.31, outstanding_request_reserve_usd=0.3)
    assert allowed.allowed and allowed.projected_spend_usd == pytest.approx(0.9)
    assert not blocked.allowed and blocked.projected_spend_usd == pytest.approx(1.01)
    assert guard.check(ER2, 0.01).next_request_reserve_usd == pytest.approx(0.05)

    reserved = guard.reserve(FLASH, 0.2)
    assert guard.outstanding_request_reserve_usd == pytest.approx(0.2)
    actual = guard.settle(
        FLASH,
        reserved,
        {"total_input_tokens": 1000, "total_output_tokens": 100},
    )
    assert guard.outstanding_request_reserve_usd == 0
    assert guard.estimated_spend_usd == pytest.approx(0.4 + actual)

    unknown_reserve = guard.reserve(FLASH, 0.1)
    assert guard.settle_reserved_upper_bound(unknown_reserve) == pytest.approx(0.1)
    assert guard.estimated_spend_usd == pytest.approx(0.5 + actual)


def test_budget_guard_reads_optional_er2_reserve_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_SPEND_USD", "2")
    monkeypatch.setenv("GEMINI_ER2_COST_CAP_PER_REQUEST_USD", "0.07")
    guard = BudgetGuard.from_environment()
    assert guard.check(ER2, 0.01).next_request_reserve_usd == pytest.approx(0.07)


def test_runner_persists_provider_metadata_actual_usage_and_standard_tier(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    client = _Client([{"output": _valid_response()}])
    budget = BudgetGuard(1.0)
    runner = GoogleInteractionsRunner(
        cache=cache, budget=budget, api_key="test-only", client=client
    )
    result = runner.run(
        **_runner_kwargs(tmp_path),
        protocol_id="A2",
        replicate_id=3,
        namespace="stability",
    )
    assert result["lifecycle_status"] == "SUCCEEDED"
    assert result["request_id"] == "interaction-1"
    assert result["response_model"] == FLASH
    assert result["service_tier"] == "standard"
    assert result["response_status"] == "completed"
    assert result["estimated_charge_usd"] == pytest.approx(
        estimate_usage_cost_usd(FLASH, result["usage"])
    )
    assert budget.outstanding_request_reserve_usd == 0
    assert client.interactions.kwargs[0]["service_tier"] == "standard"
    assert cache.request_state(result["request_hash"])["status"] == "SUCCEEDED"
    attempt = cache.attempts(result["request_hash"])[0]
    assert attempt["response_id"] == "interaction-1"
    assert attempt["status"] == "SUCCEEDED"
    cache.close()


def test_retry_exhaustion_remains_resumable_in_same_cache(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    budget = BudgetGuard(1.0)
    kwargs = _runner_kwargs(tmp_path)
    first_client = _Client([TimeoutError("timeout")])
    first = GoogleInteractionsRunner(
        cache=cache, budget=budget, api_key="test-only", client=first_client,
        max_retries=0, sleep=lambda _: None,
    ).run(**kwargs)
    assert first["lifecycle_status"] == "RETRYABLE_FAILED"
    assert cache.get(first["request_hash"]) is None
    assert cache.request_state(first["request_hash"])["status"] == "RETRYABLE_FAILED"

    second_client = _Client([{"output": _valid_response()}])
    second = GoogleInteractionsRunner(
        cache=cache, budget=budget, api_key="test-only", client=second_client,
        max_retries=0,
    ).run(**kwargs)
    assert second["lifecycle_status"] == "SUCCEEDED"
    assert len(cache.attempts(second["request_hash"])) == 2
    assert first_client.interactions.calls == second_client.interactions.calls == 1
    cache.close()


def test_cached_success_never_calls_provider_twice(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    client = _Client([{"output": _valid_response()}])
    runner = GoogleInteractionsRunner(
        cache=cache, budget=BudgetGuard(1.0), api_key="test-only", client=client
    )
    kwargs = _runner_kwargs(tmp_path)
    first = runner.run(**kwargs)
    second = runner.run(**kwargs)
    assert not first["cache_hit"] and second["cache_hit"]
    assert client.interactions.calls == 1
    cache.close()


def test_response_metadata_and_visible_model_drift_hooks():
    interaction = SimpleNamespace(
        id="id", model=FLASH, service_tier="standard",
        created=datetime(2026, 8, 1, tzinfo=timezone.utc),
        updated="later", status=SimpleNamespace(value="completed"),
    )
    extracted = response_metadata(interaction)
    assert extracted["response_id"] == "id"
    assert extracted["response_status"] == "completed"
    expected = {"name": FLASH, "version": "3.6", "outputTokenLimit": 65536}
    observed = {"name": FLASH, "version": "3.7", "outputTokenLimit": 65536}
    comparison = compare_model_metadata(expected, observed)
    assert comparison["drift_detected"]
    assert comparison["changes"] == {
        "version": {"expected": "3.6", "observed": "3.7"}
    }
    assert model_metadata_fingerprint(expected) == comparison["expected_sha256"]
    assert comparison["limitation"] == "provider_hidden_build_changes_are_not_observable"


def test_invalid_display_mapping_fails_before_cache_or_provider(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    client = _Client([{"output": _valid_response()}])
    runner = GoogleInteractionsRunner(
        cache=cache, budget=BudgetGuard(1.0), api_key="test-only", client=client
    )
    kwargs = _runner_kwargs(tmp_path)
    kwargs["mapping"]["candidate_to_display"]["candidate_0"] = "B"
    with pytest.raises(ValueError, match="inverse"):
        runner.run(**kwargs)
    assert client.interactions.calls == 0
    assert cache.connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
    cache.close()
