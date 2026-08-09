from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from failure_analysis.vlm_safe_rerank.api import (
    BudgetHardStop,
    PairwiseInteractionsRunner,
    ReplayOnlyCacheMiss,
)
from failure_analysis.vlm_safe_rerank.ledger import PairwiseLedger


def _response(decision: str = "KEEP_BASELINE") -> str:
    return json.dumps({
        "decision": decision, "evidence_reliable": True,
        "baseline_target_alignment": .8, "challenger_target_alignment": .7,
        "baseline_contact_geometry": .8, "challenger_contact_geometry": .7,
        "baseline_collision_risk": .1, "challenger_collision_risk": .2,
        "baseline_width_compatibility": .8, "challenger_width_compatibility": .7,
        "reason_codes": ["NO_CLEAR_ADVANTAGE"], "brief_rationale": "No clear reliable advantage."
    }, separators=(",", ":"))


class FakeInteractions:
    def __init__(self, outputs: list[object]):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(output_text=output, id="request-1", model="fixed-model", usage=None)


def _kwargs() -> dict:
    evidence = {
        "evidence_hash": "e" * 64,
        "baseline": {"candidate_id": "c0"},
        "challenger": {"candidate_id": "c1"},
    }
    return dict(
        sample_id="sample", model_id="gemini-3.6-flash", protocol="P4",
        baseline_candidate_id="c0", challenger_candidate_id="c1", evidence=evidence,
        board_png=b"\x89PNG\r\n", system_prompt="safe", prompt_hash="p",
        schema_hash="s", renderer_hash="r", perturbation_variant="original",
    )


def test_schema_retry_once_then_success_and_cache(tmp_path: Path) -> None:
    interactions = FakeInteractions(["not-json", _response()])
    client = SimpleNamespace(interactions=interactions)
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = PairwiseInteractionsRunner(
            ledger=ledger, max_spend_usd=10, er2_request_reserve_usd=.1,
            api_key="in-memory-only", client=client, sleep=lambda _: None,
        )
        result = runner.run(**_kwargs())
        assert result.status == "SUCCEEDED" and result.api_attempts == 2
        replay = runner.run(**_kwargs())
        assert replay.cache_hit and len(interactions.calls) == 2
        assert ledger.summary()["attempt_status_counts"] == {"SCHEMA_FAILED": 1, "SUCCEEDED": 1}


def test_network_failure_becomes_terminal_fallback(tmp_path: Path) -> None:
    interactions = FakeInteractions([RuntimeError("503 unavailable"), RuntimeError("503 unavailable")])
    client = SimpleNamespace(interactions=interactions)
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = PairwiseInteractionsRunner(
            ledger=ledger, max_spend_usd=10, er2_request_reserve_usd=.1,
            api_key="in-memory-only", client=client, sleep=lambda _: None, max_transport_retries=1,
        )
        result = runner.run(**_kwargs())
        assert result.status == "PERMANENT_FAILED" and result.parsed is None
        assert runner.run(**_kwargs()).cache_hit
        assert len(interactions.calls) == 2


def test_request_payload_is_stateless_and_contains_no_evaluator_fields(tmp_path: Path) -> None:
    interactions = FakeInteractions([_response()])
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = PairwiseInteractionsRunner(
            ledger=ledger, max_spend_usd=10, er2_request_reserve_usd=.1,
            api_key="in-memory-only", client=SimpleNamespace(interactions=interactions), sleep=lambda _: None,
        )
        runner.run(**_kwargs())
    call = interactions.calls[0]
    assert call["store"] is False and call["stream"] is False and call["background"] is False
    serialized = json.dumps(call)
    assert "candidate_correct" not in serialized and "ground_truth" not in serialized


def test_budget_checked_before_provider_call(tmp_path: Path) -> None:
    interactions = FakeInteractions([_response()])
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = PairwiseInteractionsRunner(
            ledger=ledger, max_spend_usd=.01, er2_request_reserve_usd=.1,
            flash_request_reserve_usd=.1, api_key="in-memory-only",
            client=SimpleNamespace(interactions=interactions), sleep=lambda _: None,
        )
        with pytest.raises(BudgetHardStop):
            runner.run(**_kwargs())
        assert not interactions.calls


def test_open_circuit_records_terminal_q_only_fallback_without_api(tmp_path: Path) -> None:
    interactions = FakeInteractions([_response()])
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = PairwiseInteractionsRunner(
            ledger=ledger, max_spend_usd=10, er2_request_reserve_usd=.1,
            api_key="in-memory-only", client=SimpleNamespace(interactions=interactions), sleep=lambda _: None,
        )
        kwargs = _kwargs()
        for key in ("board_png", "system_prompt"):
            kwargs.pop(key)
        kwargs["board_sha256"] = __import__("hashlib").sha256(b"\x89PNG\r\n").hexdigest()
        result = runner.record_circuit_breaker_fallback(**kwargs)
        assert result.status == "TECHNICAL_FALLBACK" and result.api_attempts == 0
        assert not interactions.calls
        assert ledger.summary()["request_status_counts"] == {"PLANNED": 1}


def test_schema_retry_rechecks_budget_and_releases_lease(tmp_path: Path) -> None:
    interactions = FakeInteractions(["not-json", _response()])
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = PairwiseInteractionsRunner(
            ledger=ledger, max_spend_usd=.1, er2_request_reserve_usd=.1,
            flash_request_reserve_usd=.1, api_key="in-memory-only",
            client=SimpleNamespace(interactions=interactions), sleep=lambda _: None,
        )
        with pytest.raises(BudgetHardStop):
            runner.run(**_kwargs())
        assert len(interactions.calls) == 1
        assert ledger.summary()["request_status_counts"] == {"RETRYABLE_FAILED": 1}


def test_transport_attempts_use_conservative_reserve(tmp_path: Path) -> None:
    interactions = FakeInteractions([RuntimeError("503 unavailable"), RuntimeError("503 unavailable")])
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = PairwiseInteractionsRunner(
            ledger=ledger, max_spend_usd=1, er2_request_reserve_usd=.1,
            flash_request_reserve_usd=.1, api_key="in-memory-only",
            client=SimpleNamespace(interactions=interactions), sleep=lambda _: None,
            max_transport_retries=1,
        )
        runner.run(**_kwargs())
        assert ledger.summary()["attempt_estimated_cost_usd"] == pytest.approx(.2)


def test_replay_only_cache_miss_never_constructs_client(tmp_path: Path) -> None:
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = PairwiseInteractionsRunner(
            ledger=ledger, max_spend_usd=1, er2_request_reserve_usd=.1,
            api_key="in-memory-only", replay_only=True,
        )
        with pytest.raises(ReplayOnlyCacheMiss):
            runner.run(**_kwargs())
        assert runner.client is None


def test_er2_attempts_always_use_configured_conservative_cap(tmp_path: Path) -> None:
    interactions = FakeInteractions([_response(), _response()])
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = PairwiseInteractionsRunner(
            ledger=ledger, max_spend_usd=2, er2_request_reserve_usd=.4,
            api_key="in-memory-only", client=SimpleNamespace(interactions=interactions),
            sleep=lambda _: None,
        )
        kwargs = _kwargs()
        kwargs["model_id"] = "gemini-robotics-er-2-preview"
        runner.run(**kwargs)
        runner.run(**{**kwargs, "sample_id": "sample-2"})
        assert ledger.summary()["attempt_estimated_cost_usd"] == pytest.approx(.8)
