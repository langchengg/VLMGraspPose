from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from failure_analysis.gemini_crog_evidence_v1.api import (
    BudgetGuard,
    GeminiCache,
    GoogleInteractionsRunner,
)
from failure_analysis.gemini_crog_evidence_v1.full_run import (
    FullRunConfig,
    FullRunOrchestrator,
    build_parser,
)


FLASH = "gemini-3.6-flash"


class _TimeoutInteractions:
    def __init__(self, effects: list[Exception]) -> None:
        self.effects = list(effects)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        raise self.effects.pop(0)


def _client(*effects: Exception) -> SimpleNamespace:
    return SimpleNamespace(interactions=_TimeoutInteractions(list(effects)))


def _mapping() -> dict:
    display_to_candidate = {
        letter: f"candidate_{index}" for index, letter in enumerate("ABCDE")
    }
    return {
        "display_to_candidate": display_to_candidate,
        "candidate_to_display": {
            candidate: display for display, candidate in display_to_candidate.items()
        },
    }


def _request_kwargs(tmp_path: Path) -> dict:
    board = tmp_path / "board.png"
    board.write_bytes(b"not-decoded-by-transport-tests")
    return {
        "sample_id": "sample-1",
        "frame_id": "frame-1",
        "model_id": FLASH,
        "image_path": board,
        "system_instruction": "Rank the frozen candidates.",
        "metadata_prompt": "Evidence only.",
        "mapping": _mapping(),
        "prompt_hash": "prompt",
        "schema_hash": "schema",
        "renderer_hash": "renderer",
        "evidence_schema_hash": "evidence",
        "request_upper_bound_usd": 0.10,
    }


def _frozen_candidates() -> list[dict]:
    candidates = []
    for index in range(5):
        candidates.append(
            {
                "candidate_id": f"candidate_{index}",
                "candidate_checksum": f"checksum-{index}",
                "cx": float(index + 1),
                "cy": float(index + 2),
                "row": float(index + 2),
                "col": float(index + 1),
                "angle_deg": 0.0,
                "width_px": 10.0,
                "height_px": 5.0,
                "q_raw": 0.9 - index * 0.1,
                "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            }
        )
    return candidates


def _runner(
    cache: GeminiCache,
    client: SimpleNamespace,
    *,
    cumulative_cap: int | None,
    max_retries: int = 0,
    sleep=None,
) -> GoogleInteractionsRunner:
    kwargs = {
        "cache": cache,
        "budget": BudgetGuard(10.0),
        "api_key": "test-only",
        "client": client,
        "max_retries": max_retries,
        "sleep": (lambda _seconds: None) if sleep is None else sleep,
    }
    if cumulative_cap is not None:
        kwargs["max_cumulative_retryable_attempts"] = cumulative_cap
    return GoogleInteractionsRunner(**kwargs)


def test_cumulative_retryable_cap_survives_resume_and_terminally_caches_c0_fallback(
    tmp_path,
):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    request = _request_kwargs(tmp_path)

    first_client = _client(TimeoutError("timeout"))
    first = _runner(cache, first_client, cumulative_cap=2).run(**request)
    assert first["lifecycle_status"] == "RETRYABLE_FAILED"

    second_client = _client(TimeoutError("timeout"))
    second = _runner(cache, second_client, cumulative_cap=2).run(**request)
    assert second["request_hash"] == first["request_hash"]
    assert second["lifecycle_status"] == "TECHNICAL_FALLBACK"
    assert second["fallback_reason"] == "cumulative_retryable_attempt_cap_reached"
    assert second["retry_count"] == 2
    assert cache.request_state(second["request_hash"])["status"] == "TECHNICAL_FALLBACK"
    assert [row["status"] for row in cache.attempts(second["request_hash"])] == [
        "RETRYABLE_FAILED",
        "RETRYABLE_FAILED",
    ]

    never_called = _client(AssertionError("provider must not be called after terminal fallback"))
    cached = _runner(cache, never_called, cumulative_cap=2).run(**request)
    assert cached["cache_hit"] is True
    assert cached["lifecycle_status"] == "TECHNICAL_FALLBACK"
    assert never_called.interactions.calls == 0
    assert first_client.interactions.calls == second_client.interactions.calls == 1

    orchestrator = FullRunOrchestrator(
        FullRunConfig(
            run_root=tmp_path / "run",
            verify_environment=False,
            verify_repository_contract=False,
            enforce_canonical_counts=False,
        )
    )
    normalized = orchestrator._normalize_decision(
        phase="pilot",
        protocol="p1_full_crog_evidence",
        replicate_id=0,
        model_id=FLASH,
        feature={"candidates": _frozen_candidates()},
        request={
            "sample_id": "sample-1",
            "frame_id": "frame-1",
            "mapping": _mapping(),
            "board_sha256": "board",
        },
        result=second,
    )
    assert normalized["selected_candidate_id"] == "candidate_0"
    assert normalized["technical_fallback"] is True
    orchestrator._check_hard_stop_result("pilot", normalized)
    cache.close()


def test_cumulative_cap_stops_retries_inside_one_runner_call(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    client = _client(
        TimeoutError("timeout-1"),
        TimeoutError("timeout-2"),
        AssertionError("third provider call must be capped"),
    )
    sleeps: list[float] = []
    result = _runner(
        cache,
        client,
        cumulative_cap=2,
        max_retries=5,
        sleep=sleeps.append,
    ).run(**_request_kwargs(tmp_path))

    assert result["lifecycle_status"] == "TECHNICAL_FALLBACK"
    assert result["retry_count"] == 2
    assert client.interactions.calls == 2
    assert len(sleeps) == 1
    cache.close()


def test_retry_cap_can_come_from_environment_and_cli_keeps_stop_after_pilot(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GEMINI_MAX_CUMULATIVE_RETRYABLE_ATTEMPTS", "3")
    cache = GeminiCache(tmp_path / "cache.sqlite")
    runner = _runner(cache, _client(TimeoutError("unused")), cumulative_cap=None)
    assert runner.max_cumulative_retryable_attempts == 3

    args = build_parser().parse_args(
        [
            "--run-root",
            str(tmp_path / "run"),
            "--stop-after",
            "pilot",
            "--max-cumulative-retryable-attempts",
            "4",
        ]
    )
    assert args.stop_after == "pilot"
    assert args.max_cumulative_retryable_attempts == 4
    cache.close()


class _ProgressState:
    def __init__(self, root: Path) -> None:
        self.paths = SimpleNamespace(api=root / "api.json", runtime=root / "runtime.json")
        self.progress: dict[str, dict] = {}

    def update_phase(self, *_args, **_kwargs) -> None:
        return None

    def update_progress(self, name: str, payload: dict) -> None:
        self.progress[name] = dict(payload)


def test_progress_recomputes_cumulative_retries_from_sqlite_attempt_ledger(tmp_path):
    cache = GeminiCache(tmp_path / "cache.sqlite")
    cache.plan_request(request_hash="hash-1", sample_id="sample-1", model_id=FLASH)
    for status in ("RETRYABLE_FAILED", "RETRYABLE_FAILED", "SUCCEEDED"):
        cache.append_attempt(
            "hash-1",
            owner_id="test-owner",
            started_at="2026-08-03T00:00:00Z",
            completed_at="2026-08-03T00:00:01Z",
            status=status,
            latency_seconds=1.0,
        )

    config = FullRunConfig(
        run_root=tmp_path / "run",
        verify_environment=False,
        verify_repository_contract=False,
        enforce_canonical_counts=False,
    )
    orchestrator = FullRunOrchestrator(config)
    state = _ProgressState(config.run_root)
    orchestrator._cache = cache
    orchestrator._state = state
    decision_root = config.run_root / "pilot" / "decisions"
    decision_root.mkdir(parents=True)
    (decision_root / "one.json").write_text(
        """{
          "logical_request_key": "pilot|p1|gemini-3.6-flash|sample-1|0",
          "model_id": "gemini-3.6-flash",
          "request_hash": "hash-1",
          "retry_count": 0,
          "cache_hit": false,
          "api_attempted": true,
          "latency_seconds": 1.0,
          "lifecycle_status": "SUCCEEDED",
          "completed_at_utc": "2026-08-03T00:00:01Z"
        }""",
        encoding="utf-8",
    )

    orchestrator._update_request_progress("pilot", 1)

    assert state.progress["api"]["retries"] == 2
    assert state.progress["api"]["new_api_attempts"] == 3
    cache.close()


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_invalid_retry_cap_environment_is_rejected_before_transport(
    tmp_path, monkeypatch, value
):
    monkeypatch.setenv("GEMINI_MAX_CUMULATIVE_RETRYABLE_ATTEMPTS", value)
    cache = GeminiCache(tmp_path / "cache.sqlite")
    with pytest.raises(ValueError, match="cumulative retryable"):
        _runner(cache, _client(TimeoutError("unused")), cumulative_cap=None)
    cache.close()
