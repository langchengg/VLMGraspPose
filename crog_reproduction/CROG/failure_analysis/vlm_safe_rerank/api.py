from __future__ import annotations

import base64
import hashlib
import email.utils
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from failure_analysis.gemini_crog_evidence_v1.api import estimate_usage_cost_usd

from .ledger import PairwiseLedger, legacy_pairwise_request_hash, pairwise_request_hash
from .schema import PairwiseCriticResponse, parse_pairwise_response, pairwise_response_schema
from .security import assert_frozen_pair, assert_no_ground_truth


MODEL_IDS = ("gemini-robotics-er-2-preview", "gemini-3.6-flash")
RETRYABLE = {429, 500, 502, 503, 504}
PERMANENT = {400, 401, 403, 404}


class AuthenticationHardStop(RuntimeError):
    pass


class BudgetHardStop(RuntimeError):
    pass


class ReplayOnlyCacheMiss(RuntimeError):
    pass


def _status_code(error: Exception) -> int | None:
    for name in ("status_code", "code"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
        if callable(value):
            try:
                result = value()
                if isinstance(result, int):
                    return result
            except Exception:
                pass
    import re
    match = re.search(r"\b(400|401|403|404|429|500|502|503|504)\b", str(error))
    return int(match.group(1)) if match else None


def _retry_after(error: Exception) -> float | None:
    for source in (getattr(error, "response", None), error):
        headers = getattr(source, "headers", None)
        if headers is None:
            continue
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                parsed = email.utils.parsedate_to_datetime(str(value))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
            except Exception:
                return None
    return None


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    if isinstance(usage, Mapping):
        return dict(usage)
    return {
        name: value for name in (
            "total_input_tokens", "total_output_tokens", "total_thought_tokens", "total_tokens",
            "total_cached_tokens", "input_tokens_by_modality", "output_tokens_by_modality",
        ) if (value := getattr(usage, name, None)) is not None
    }


@dataclass(frozen=True)
class PairwiseAPIResult:
    request_hash: str
    status: str
    parsed: PairwiseCriticResponse | None
    raw_response: str | None
    cache_hit: bool
    api_attempts: int
    fallback_reason: str | None
    usage: dict[str, Any]
    latency_seconds: float
    estimated_cost_usd: float
    response_model: str | None
    api_request_id: str | None


class PairwiseInteractionsRunner:
    """Strict label-free Interactions transport with exact cache replay."""

    def __init__(
        self,
        *,
        ledger: PairwiseLedger,
        max_spend_usd: float,
        er2_request_reserve_usd: float,
        flash_request_reserve_usd: float = 0.10,
        api_key: str | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_transport_retries: int = 3,
        max_schema_retries: int = 1,
        owner_id: str | None = None,
        replay_only: bool = False,
    ) -> None:
        self.ledger = ledger
        self.max_spend_usd = float(max_spend_usd)
        self.reserves = {
            "gemini-robotics-er-2-preview": float(er2_request_reserve_usd),
            "gemini-3.6-flash": float(flash_request_reserve_usd),
        }
        if self.max_spend_usd < 0 or any(value <= 0 for value in self.reserves.values()):
            raise ValueError("budget and request reserves must be positive")
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self.client = client
        self.sleep = sleep
        self.max_transport_retries = int(max_transport_retries)
        self.max_schema_retries = int(max_schema_retries)
        self.owner_id = owner_id or f"pairwise-{os.getpid()}-{uuid.uuid4().hex}"
        self.replay_only = bool(replay_only)

    def _client(self) -> Any:
        if not self.api_key:
            raise AuthenticationHardStop("Gemini API credential is unavailable")
        if self.client is None:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(api_version="v1beta"))
        return self.client

    def _check_budget(self, model_id: str) -> None:
        spent = float(self.ledger.summary()["attempt_estimated_cost_usd"])
        if spent + self.reserves[model_id] > self.max_spend_usd + 1e-12:
            raise BudgetHardStop("next request would exceed max API spend")

    def _attempt_charge(self, model_id: str, usage: Mapping[str, Any] | None = None) -> float:
        """Return the experiment's conservative charge for one provider attempt."""

        reserve = self.reserves[model_id]
        if model_id == "gemini-robotics-er-2-preview":
            return reserve
        if usage:
            try:
                return float(estimate_usage_cost_usd(model_id, usage))
            except Exception:
                pass
        return reserve

    def _request_hashes(
        self,
        *,
        board_sha256: str,
        sample_id: str,
        baseline_candidate_id: str,
        challenger_candidate_id: str,
        model_id: str,
        protocol: str,
        prompt_hash: str,
        schema_hash: str,
        renderer_hash: str,
        evidence_hash: str,
        generation: Mapping[str, Any],
        perturbation_variant: str,
    ) -> tuple[str, str]:
        common = {
            "sample_id": str(sample_id),
            "baseline_candidate_id": baseline_candidate_id,
            "challenger_candidate_id": challenger_candidate_id,
            "model_id": model_id,
            "protocol": protocol,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "renderer_hash": renderer_hash,
            "evidence_hash": evidence_hash,
            "generation": generation,
            "perturbation_variant": perturbation_variant,
        }
        return (
            pairwise_request_hash(**common, board_sha256=board_sha256),
            legacy_pairwise_request_hash(**common),
        )

    def record_circuit_breaker_fallback(
        self,
        *,
        sample_id: str,
        model_id: str,
        protocol: str,
        baseline_candidate_id: str,
        challenger_candidate_id: str,
        evidence: Mapping[str, Any],
        prompt_hash: str,
        schema_hash: str,
        renderer_hash: str,
        board_sha256: str,
        perturbation_variant: str,
        temperature: float = 0.0,
        thinking_level: str = "low",
        max_output_tokens: int = 1024,
    ) -> PairwiseAPIResult:
        """Durably keep c0 without calling a model whose phase circuit is open."""

        safe_payload = {"evidence": dict(evidence), "baseline_candidate_id": baseline_candidate_id, "challenger_candidate_id": challenger_candidate_id}
        assert_no_ground_truth(safe_payload)
        assert_frozen_pair(safe_payload, baseline_id=baseline_candidate_id, challenger_id=challenger_candidate_id)
        generation = {"temperature": float(temperature), "thinking_level": thinking_level, "max_output_tokens": int(max_output_tokens)}
        digest, legacy_digest = self._request_hashes(
            board_sha256=board_sha256, sample_id=str(sample_id),
            baseline_candidate_id=baseline_candidate_id,
            challenger_candidate_id=challenger_candidate_id, model_id=model_id,
            protocol=protocol, prompt_hash=prompt_hash, schema_hash=schema_hash,
            renderer_hash=renderer_hash, evidence_hash=str(evidence["evidence_hash"]),
            generation=generation, perturbation_variant=perturbation_variant,
        )
        self.ledger.migrate_terminal_hash(
            legacy_digest, digest, board_sha256=board_sha256
        )
        cached = self.ledger.cached(digest)
        if cached is not None:
            parsed = PairwiseCriticResponse.model_validate(cached.parsed_json) if cached.parsed_json else None
            return PairwiseAPIResult(
                digest, cached.status, parsed, cached.raw_response, True, 0,
                None if cached.status == "SUCCEEDED" else "model_phase_circuit_breaker_open",
                {}, float(cached.latency_seconds or 0.0), cached.estimated_cost_usd,
                cached.response_model, cached.api_request_id,
            )
        if self.replay_only:
            raise ReplayOnlyCacheMiss("replay-only mode forbids an uncached circuit fallback")
        if self.ledger.acquire(digest, self.owner_id) != "ACQUIRED":
            raise RuntimeError("circuit-breaker fallback request could not be acquired")
        # An open circuit is a phase-level pause, not a provider response.  Do
        # not poison exact request cache; a later resume may legitimately try it.
        self.ledger.release(digest, status="PLANNED")
        return PairwiseAPIResult(
            digest, "TECHNICAL_FALLBACK", None, None, False, 0,
            "model_phase_circuit_breaker_open", {}, 0.0, 0.0, None, None,
        )

    def run(
        self,
        *,
        sample_id: str,
        model_id: str,
        protocol: str,
        baseline_candidate_id: str,
        challenger_candidate_id: str,
        evidence: Mapping[str, Any],
        board_png: bytes,
        system_prompt: str,
        prompt_hash: str,
        schema_hash: str,
        renderer_hash: str,
        perturbation_variant: str,
        temperature: float = 0.0,
        thinking_level: str = "low",
        max_output_tokens: int = 1024,
        baseline_expected: Mapping[str, Any] | None = None,
        challenger_expected: Mapping[str, Any] | None = None,
    ) -> PairwiseAPIResult:
        if model_id not in MODEL_IDS:
            raise ValueError("model substitution is forbidden")
        metadata = {
            "sample_id": str(sample_id), "protocol": protocol,
            "baseline_candidate_id": baseline_candidate_id,
            "challenger_candidate_id": challenger_candidate_id,
            "evidence": dict(evidence),
            "instruction": "Compare only the frozen BASELINE and CHALLENGER; default KEEP_BASELINE.",
        }
        assert_no_ground_truth({
            "metadata": metadata,
            "system_instruction": system_prompt,
            "response_schema": pairwise_response_schema(),
        })
        assert_frozen_pair(
            metadata, baseline_id=baseline_candidate_id,
            challenger_id=challenger_candidate_id,
            baseline_expected=baseline_expected,
            challenger_expected=challenger_expected,
        )
        generation = {
            "temperature": float(temperature), "thinking_level": thinking_level,
            "max_output_tokens": int(max_output_tokens),
        }
        board_sha256 = hashlib.sha256(board_png).hexdigest()
        digest, legacy_digest = self._request_hashes(
            board_sha256=board_sha256, sample_id=str(sample_id),
            baseline_candidate_id=baseline_candidate_id,
            challenger_candidate_id=challenger_candidate_id, model_id=model_id,
            protocol=protocol, prompt_hash=prompt_hash, schema_hash=schema_hash,
            renderer_hash=renderer_hash, evidence_hash=str(evidence["evidence_hash"]),
            generation=generation, perturbation_variant=perturbation_variant,
        )
        self.ledger.migrate_terminal_hash(
            legacy_digest, digest, board_sha256=board_sha256
        )
        cached = self.ledger.cached(digest)
        if cached is not None:
            parsed = PairwiseCriticResponse.model_validate(cached.parsed_json) if cached.parsed_json else None
            return PairwiseAPIResult(
                digest, cached.status, parsed, cached.raw_response, True, 0,
                None if cached.status == "SUCCEEDED" else cached.status.lower(), {},
                float(cached.latency_seconds or 0.0), cached.estimated_cost_usd,
                cached.response_model, cached.api_request_id,
            )
        if self.replay_only:
            raise ReplayOnlyCacheMiss("replay-only mode forbids a provider call on cache miss")
        self._check_budget(model_id)
        claim = self.ledger.acquire(digest, self.owner_id)
        if claim != "ACQUIRED":
            raise RuntimeError(f"request not acquired: {claim}")
        started = time.perf_counter()
        attempts = 0; schema_failures = 0; last_raw = None; total_cost = 0.0; usage: dict[str, Any] = {}
        while attempts <= self.max_transport_retries:
            try:
                self._check_budget(model_id)
            except BudgetHardStop:
                self.ledger.release(digest)
                raise
            attempts += 1
            attempt_start = time.perf_counter()
            try:
                response = self._client().interactions.create(
                    model=model_id,
                    input=[
                        {"type": "text", "text": json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)},
                        {"type": "image", "data": base64.b64encode(board_png).decode("ascii"), "mime_type": "image/png", "resolution": "high"},
                    ],
                    stream=False, store=False, background=False, service_tier="standard",
                    system_instruction=system_prompt,
                    generation_config=generation,
                    response_format={"type": "text", "mime_type": "application/json", "schema": pairwise_response_schema()},
                )
                last_raw = str(getattr(response, "output_text", "") or "")
                usage = _usage(response)
                charge = self._attempt_charge(model_id, usage)
                total_cost += float(charge)
                try:
                    parsed = parse_pairwise_response(last_raw)
                except Exception as error:
                    schema_failures += 1
                    self.ledger.record_attempt(digest, "SCHEMA_FAILED", latency_seconds=time.perf_counter() - attempt_start, estimated_cost_usd=charge, error_class=type(error).__name__)
                    if schema_failures <= self.max_schema_retries:
                        continue
                    latency = time.perf_counter() - started
                    self.ledger.finish(
                        digest, "SCHEMA_FAILED", raw_response=last_raw,
                        latency_seconds=latency, estimated_cost_usd=total_cost,
                        requested_model=model_id,
                        response_model=getattr(response, "model", None),
                    )
                    return PairwiseAPIResult(digest, "SCHEMA_FAILED", None, last_raw, False, attempts, "schema_invalid", usage, latency, total_cost, getattr(response, "model", None), getattr(response, "id", None))
                latency = time.perf_counter() - started
                response_id = getattr(response, "id", None)
                response_model = getattr(response, "model", None)
                self.ledger.record_attempt(digest, "SUCCEEDED", latency_seconds=time.perf_counter() - attempt_start, estimated_cost_usd=charge)
                self.ledger.finish(
                    digest, "SUCCEEDED", raw_response=last_raw,
                    parsed_json=parsed.model_dump(mode="json"), api_request_id=response_id,
                    latency_seconds=latency, estimated_cost_usd=total_cost,
                    requested_model=model_id, response_model=response_model,
                )
                return PairwiseAPIResult(digest, "SUCCEEDED", parsed, last_raw, False, attempts, None, usage, latency, total_cost, response_model, response_id)
            except AuthenticationHardStop:
                raise
            except BudgetHardStop:
                raise
            except Exception as error:
                status = _status_code(error)
                retryable = status in RETRYABLE or status is None
                charge = self._attempt_charge(model_id)
                total_cost += charge
                self.ledger.record_attempt(
                    digest, "RETRYABLE_FAILED" if retryable else "PERMANENT_FAILED",
                    latency_seconds=time.perf_counter() - attempt_start,
                    estimated_cost_usd=charge, error_class=type(error).__name__,
                )
                if status in (401, 403):
                    self.ledger.finish(
                        digest, "PERMANENT_FAILED",
                        latency_seconds=time.perf_counter() - started,
                        estimated_cost_usd=total_cost, requested_model=model_id,
                    )
                    raise AuthenticationHardStop("Gemini API credential was rejected") from error
                if not retryable or attempts > self.max_transport_retries:
                    latency = time.perf_counter() - started
                    self.ledger.finish(
                        digest, "PERMANENT_FAILED", raw_response=last_raw,
                        latency_seconds=latency, estimated_cost_usd=total_cost,
                        requested_model=model_id,
                    )
                    return PairwiseAPIResult(digest, "PERMANENT_FAILED", None, last_raw, False, attempts, "transport_failure", usage, latency, total_cost, None, None)
                delay = _retry_after(error)
                if delay is None:
                    delay = min(30.0, 1.5 * (2 ** (attempts - 1))) + random.random() * 0.25
                self.sleep(delay)
        raise AssertionError("unreachable")
