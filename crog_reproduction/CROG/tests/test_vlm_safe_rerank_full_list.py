from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from failure_analysis.vlm_safe_rerank.full_list_api import (
    FULL_LIST_PROTOCOL,
    FullListInteractionsRunner,
    full_list_request_hash,
    load_full_list_contract,
)
from failure_analysis.vlm_safe_rerank.full_list_renderer import (
    full_list_renderer_hash,
    render_full_list_board,
)
from failure_analysis.vlm_safe_rerank.ledger import PairwiseLedger
from failure_analysis.vlm_safe_rerank.schema import parse_full_list_keep_response


REPO = Path(__file__).resolve().parents[1]
FEATURES = (
    REPO
    / "failure_analysis/reranking_outputs/v2_20260727T174412+0100/base_train/features.jsonl"
)
FLASH = "gemini-3.6-flash"


@pytest.fixture(scope="module")
def frozen_feature() -> dict:
    with FEATURES.open(encoding="utf-8") as handle:
        return json.loads(next(handle))


@pytest.fixture(scope="module")
def rendered(frozen_feature: dict) -> tuple[bytes, dict, dict]:
    return render_full_list_board(frozen_feature)


def _response(
    decision: str = "KEEP_BASELINE",
    selected_candidate_id: str = "candidate_0",
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "selected_candidate_id": selected_candidate_id,
            "evidence_reliable": True,
            "reason_codes": ["NO_CLEAR_ADVANTAGE"],
            "brief_rationale": "No concrete defect justifies replacing the baseline.",
        },
        separators=(",", ":"),
    )


class FakeInteractions:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(
            output_text=output,
            id=f"request-{len(self.calls)}",
            model=FLASH,
            usage=None,
        )


def _runner(
    ledger: PairwiseLedger,
    interactions: FakeInteractions,
    **changes: object,
) -> FullListInteractionsRunner:
    kwargs = {
        "ledger": ledger,
        "max_spend_usd": 10.0,
        "er2_request_reserve_usd": 0.10,
        "flash_request_reserve_usd": 0.10,
        "api_key": "in-memory-test-only",
        "client": SimpleNamespace(interactions=interactions),
        "sleep": lambda _seconds: None,
    }
    kwargs.update(changes)
    return FullListInteractionsRunner(**kwargs)


def _run_kwargs(rendered: tuple[bytes, dict, dict]) -> dict:
    board_png, evidence, metadata = rendered
    contract = load_full_list_contract()
    return {
        "sample_id": str(evidence["sample_id"]),
        "model_id": FLASH,
        "protocol": FULL_LIST_PROTOCOL,
        "evidence": evidence,
        "board_png": board_png,
        "board_metadata": metadata,
        "system_prompt": contract.system_prompt,
        "response_schema": contract.response_schema,
        "prompt_hash": contract.prompt_hash,
        "schema_hash": contract.schema_hash,
        "renderer_hash": metadata["renderer_contract_hash"],
    }


def test_renderer_shows_frozen_top5_and_explicit_c0_prior(
    frozen_feature: dict,
    rendered: tuple[bytes, dict, dict],
) -> None:
    before = [
        (row["candidate_id"], row["candidate_checksum"], row["q_raw"])
        for row in frozen_feature["candidates"]
    ]
    board_png, evidence, metadata = rendered
    after = [
        (row["candidate_id"], row["candidate_checksum"], row["q_raw"])
        for row in frozen_feature["candidates"]
    ]

    assert board_png.startswith(b"\x89PNG")
    assert before == after
    assert evidence["baseline_candidate_id"] == "candidate_0"
    assert [row["candidate_id"] for row in evidence["candidates"]] == [
        f"candidate_{index}" for index in range(5)
    ]
    assert evidence["candidates"][0]["original_rank"] == 0
    assert all("mask_rectangle_coverage" in row for row in evidence["candidates"])
    assert all("aggregate_reliability" in row for row in evidence["candidates"])
    assert metadata["visible_candidate_ids"] == [
        f"candidate_{index}" for index in range(5)
    ]
    assert metadata["baseline_label"] == "candidate_0 (FROZEN BASELINE / KEEP PRIOR)"
    assert metadata["input_panels"] == [
        "rgb",
        "predicted_mask",
        "metric_depth",
        "numeric_evidence",
    ]
    assert metadata["renderer_contract_hash"] == full_list_renderer_hash()


def test_contract_loads_registered_prompt_and_schema() -> None:
    contract = load_full_list_contract()
    assert "candidate_0" in contract.system_prompt
    assert "Default to KEEP_BASELINE" in contract.system_prompt
    assert contract.response_schema["additionalProperties"] is False
    assert set(contract.response_schema["required"]) == {
        "decision",
        "selected_candidate_id",
        "evidence_reliable",
        "reason_codes",
        "brief_rationale",
    }
    assert len(contract.prompt_hash) == len(contract.schema_hash) == 64


def test_full_list_hash_binds_prompt_schema_renderer_evidence_and_generation() -> None:
    base = {
        "sample_id": "sample",
        "model_id": FLASH,
        "protocol": FULL_LIST_PROTOCOL,
        "candidate_ids": [f"candidate_{index}" for index in range(5)],
        "prompt_hash": "p1",
        "schema_hash": "s1",
        "renderer_hash": "r1",
        "evidence_hash": "e1",
        "board_sha256": "b1",
        "generation": {"temperature": 0.0, "thinking_level": "low"},
    }
    digest = full_list_request_hash(**base)
    assert digest == full_list_request_hash(**base)
    for field, replacement in (
        ("prompt_hash", "p2"),
        ("schema_hash", "s2"),
        ("renderer_hash", "r2"),
        ("evidence_hash", "e2"),
        ("generation", {"temperature": 0.1, "thinking_level": "low"}),
    ):
        assert digest != full_list_request_hash(**{**base, field: replacement})


@pytest.mark.parametrize("decision", ["KEEP_BASELINE", "INSUFFICIENT_EVIDENCE"])
def test_keep_and_insufficient_must_select_candidate_zero(decision: str) -> None:
    parsed = parse_full_list_keep_response(_response(decision, "candidate_0"))
    assert parsed.selected_candidate_id == "candidate_0"
    with pytest.raises(ValueError):
        parse_full_list_keep_response(_response(decision, "candidate_1"))
    with pytest.raises(ValueError):
        parse_full_list_keep_response("```json\n" + _response() + "\n```")


def test_success_uses_stateless_interactions_and_exact_cache(
    tmp_path: Path,
    rendered: tuple[bytes, dict, dict],
) -> None:
    interactions = FakeInteractions([_response("PREFER_CANDIDATE", "candidate_2")])
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = _runner(ledger, interactions)
        result = runner.run(**_run_kwargs(rendered))
        replay = runner.run(**_run_kwargs(rendered))

        assert result.status == "SUCCEEDED"
        assert result.selected_candidate_id == "candidate_2"
        assert not result.cache_hit
        assert replay.cache_hit and replay.selected_candidate_id == "candidate_2"
        assert result.request_hash == replay.request_hash
        assert ledger.summary()["attempt_status_counts"] == {"SUCCEEDED": 1}

    assert len(interactions.calls) == 1
    call = interactions.calls[0]
    assert call["store"] is False
    assert call["stream"] is False
    assert call["background"] is False
    assert "tools" not in call and "previous_interaction_id" not in call
    assert call["response_format"]["schema"]["additionalProperties"] is False
    request_text = call["input"][0]["text"]
    assert "candidate_0" in request_text and "frozen baseline" in request_text.lower()
    assert "candidate_correct" not in json.dumps(call).lower()


def test_schema_retry_once_then_success(
    tmp_path: Path,
    rendered: tuple[bytes, dict, dict],
) -> None:
    interactions = FakeInteractions(
        ["not-json", _response("PREFER_CANDIDATE", "candidate_1")]
    )
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        result = _runner(
            ledger,
            interactions,
            max_schema_retries=1,
            max_transport_retries=0,
        ).run(**_run_kwargs(rendered))
        assert result.status == "SUCCEEDED"
        assert result.selected_candidate_id == "candidate_1"
        assert result.api_attempts == 2
        assert ledger.summary()["attempt_status_counts"] == {
            "SCHEMA_FAILED": 1,
            "SUCCEEDED": 1,
        }


def test_schema_exhaustion_is_cached_c0_fallback(
    tmp_path: Path,
    rendered: tuple[bytes, dict, dict],
) -> None:
    interactions = FakeInteractions(["not-json", "also-not-json"])
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = _runner(ledger, interactions, max_schema_retries=1)
        result = runner.run(**_run_kwargs(rendered))
        replay = runner.run(**_run_kwargs(rendered))
        assert result.status == "SCHEMA_FAILED"
        assert result.selected_candidate_id == "candidate_0"
        assert result.fallback_reason == "schema_invalid"
        assert replay.cache_hit and replay.selected_candidate_id == "candidate_0"
        assert len(interactions.calls) == 2


def test_transport_retry_is_finite_and_falls_back_to_c0(
    tmp_path: Path,
    rendered: tuple[bytes, dict, dict],
) -> None:
    interactions = FakeInteractions(
        [RuntimeError("503 unavailable"), RuntimeError("503 unavailable")]
    )
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        runner = _runner(
            ledger,
            interactions,
            max_transport_retries=1,
            max_schema_retries=0,
        )
        result = runner.run(**_run_kwargs(rendered))
        replay = runner.run(**_run_kwargs(rendered))
        assert result.status == "TECHNICAL_FALLBACK"
        assert result.selected_candidate_id == "candidate_0"
        assert result.api_attempts == 2
        assert replay.cache_hit and replay.selected_candidate_id == "candidate_0"
        assert len(interactions.calls) == 2


def test_no_gt_rejected_before_cache_or_client(
    tmp_path: Path,
    rendered: tuple[bytes, dict, dict],
) -> None:
    interactions = FakeInteractions([_response()])
    kwargs = _run_kwargs(rendered)
    contaminated = copy.deepcopy(kwargs["evidence"])
    contaminated["candidates"][1]["candidate_correct"] = True
    kwargs["evidence"] = contaminated

    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        with pytest.raises(AssertionError, match="forbidden evaluation field"):
            _runner(ledger, interactions).run(**kwargs)
        assert ledger.summary()["attempts"] == 0
        assert ledger.summary()["request_status_counts"] == {}
    assert interactions.calls == []


def test_unregistered_prompt_is_rejected_before_cache_or_client(
    tmp_path: Path,
    rendered: tuple[bytes, dict, dict],
) -> None:
    interactions = FakeInteractions([_response()])
    kwargs = _run_kwargs(rendered)
    kwargs["system_prompt"] += "\nChanged semantics."
    kwargs["prompt_hash"] = hashlib.sha256(
        kwargs["system_prompt"].encode("utf-8")
    ).hexdigest()

    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        with pytest.raises(ValueError, match="prompt hash mismatch"):
            _runner(ledger, interactions).run(**kwargs)
        assert ledger.summary()["request_status_counts"] == {}
    assert interactions.calls == []
