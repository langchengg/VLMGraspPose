"""Finite, budgeted, resumable provider runner for exact Gemini model IDs."""

from __future__ import annotations

import base64
import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .constants import EXACT_MODEL_IDS
from .ledger import ApiLedger
from .payload import logical_request_hash
from .contracts import assert_no_gt_payload
from .io import sha256_json
from .schema import ParsedResponse, parse_response


RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504}


class AuthenticationHardStop(RuntimeError):
    pass


class BudgetHardStop(RuntimeError):
    pass


class ModelUnavailableHardStop(RuntimeError):
    pass


class ProviderQuotaHardStop(RuntimeError):
    """Persistent provider quota exhaustion; pause the stage without consuming its queue."""


class ModelRateLimiter:
    """Thread-safe, per-model start-rate limiter with provider deferrals."""

    def __init__(
        self, requests_per_minute: Mapping[str, float], *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.intervals = {
            str(model): 60.0 / float(rate)
            for model, rate in requests_per_minute.items()
            if float(rate) > 0
        }
        self.clock, self.sleep = clock, sleep
        self.next_allowed: dict[str, float] = {}
        self.lock = threading.Lock()

    def wait(self, model_id: str) -> None:
        interval = self.intervals.get(model_id, 0.0)
        while True:
            with self.lock:
                now = self.clock()
                allowed = self.next_allowed.get(model_id, now)
                if now >= allowed:
                    self.next_allowed[model_id] = now + interval
                    return
                delay = allowed - now
            self.sleep(delay)

    def defer(self, model_id: str, seconds: float) -> None:
        with self.lock:
            target = self.clock() + max(0.0, float(seconds))
            self.next_allowed[model_id] = max(self.next_allowed.get(model_id, target), target)


@dataclass(frozen=True)
class ProviderResult:
    request_hash: str
    status: str
    selected_candidate_id: str
    parsed: ParsedResponse | None
    raw_response: str | None
    cache_hit: bool
    attempts: int
    fallback_reason: str | None
    usage: Mapping[str, Any]
    latency_seconds: float
    estimated_cost_usd: float | None
    response_model: str | None
    request_id: str | None
    endpoint_used: str | None = None


def _status_code(error: BaseException) -> int | None:
    for value in (getattr(error, "status_code", None), getattr(error, "code", None)):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    text = str(error)
    lowered = text.lower()
    if any(token in lowered for token in ("quota exceeded", "rate limit", "resource exhausted")):
        return 429
    message_text = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith(("File ", "return ", "raise "))
    )
    for code in (400, 401, 403, 404, 408, 409, 429, 500, 502, 503, 504):
        if re.search(rf"(?<!\d){code}(?!\d)", message_text):
            return code
    return None


def _explicit_interactions_unsupported(error: BaseException) -> bool:
    text = str(error).lower()
    return "interaction" in text and any(token in text for token in ("not supported", "unsupported", "not available for"))


def _retry_after_seconds(error: BaseException) -> float | None:
    text = str(error)
    match = re.search(r"(?:please\s+)?retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s?", text, re.I)
    return None if match is None else float(match.group(1))


def estimate_flash_cost(usage: Mapping[str, Any]) -> float | None:
    """2026-08-05 standard pricing: $1.50/M input, $7.50/M output incl. thinking."""
    def number(*names: str) -> float:
        for name in names:
            if usage.get(name) is not None:
                return float(usage[name])
        return 0.0
    input_tokens = number("total_input_tokens", "input_tokens", "prompt_token_count", "input_token_count")
    output_tokens = number("total_output_tokens", "output_tokens", "candidates_token_count", "output_token_count")
    thought_tokens = number("total_thought_tokens", "thoughts_token_count", "thinking_tokens", "thought_tokens")
    if not usage:
        return None
    billed_output = output_tokens + thought_tokens
    return input_tokens * 1.50 / 1_000_000 + billed_output * 7.50 / 1_000_000


def subprocess_transport(worker_python: Path, worker_script: Path) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    def invoke(request: Mapping[str, Any]) -> Mapping[str, Any]:
        with tempfile.TemporaryDirectory(prefix="gemini-api-only-") as directory:
            request_path = Path(directory) / "request.json"
            response_path = Path(directory) / "response.json"
            request_path.write_text(json.dumps(dict(request), sort_keys=True), encoding="utf-8")
            completed = subprocess.run(
                [str(worker_python), str(worker_script), "invoke", "--request", str(request_path), "--response", str(response_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300, check=False,
                env=os.environ.copy(),
            )
            if completed.returncode != 0:
                message = completed.stderr.strip() or completed.stdout.strip() or f"worker exit {completed.returncode}"
                error = RuntimeError(message[:2000])
                setattr(error, "status_code", None)
                raise error
            return json.loads(response_path.read_text(encoding="utf-8"))
    return invoke


class GeminiProviderRunner:
    def __init__(
        self, *, ledger: ApiLedger, max_api_cost_usd: float,
        max_provider_requests: int, er2_reserve_per_request_usd: float | None,
        transport: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        api_key: str | None = None, sleep: Callable[[float], None] = time.sleep,
        max_transport_retries: int = 2, max_schema_retries: int = 1,
        owner: str | None = None, rate_limiter: ModelRateLimiter | None = None,
        max_rate_limit_retries: int = 4,
    ) -> None:
        if max_api_cost_usd <= 0 or max_provider_requests <= 0:
            raise ValueError("positive API cost and request caps are required")
        self.ledger = ledger
        self.max_api_cost_usd = float(max_api_cost_usd)
        self.max_provider_requests = int(max_provider_requests)
        self.er2_reserve = er2_reserve_per_request_usd
        self.transport = transport
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.sleep = sleep
        self.max_transport_retries = int(max_transport_retries)
        self.max_schema_retries = int(max_schema_retries)
        self.max_rate_limit_retries = int(max_rate_limit_retries)
        self.owner = owner or f"api-only-{os.getpid()}-{uuid.uuid4().hex}"
        self.rate_limiter = rate_limiter

    def _reserve(self, model_id: str) -> float:
        if model_id == "gemini-robotics-er-2-preview":
            if self.er2_reserve is None or self.er2_reserve <= 0:
                raise BudgetHardStop("ER2 public price is unverified; set a positive GEMINI_ER2_COST_CAP_PER_REQUEST_USD")
            return float(self.er2_reserve)
        return 0.10

    def _preflight(self, model_id: str) -> None:
        if not self.api_key:
            raise AuthenticationHardStop("GEMINI_API_KEY or GOOGLE_API_KEY is unavailable")
        self._reserve(model_id)

    def run(
        self, *, request_hash: str, model_id: str, protocol: str,
        text_payload: Mapping[str, Any], overview_png: bytes, grid_png: bytes,
        system_prompt: str, response_schema: Mapping[str, Any],
        generation: Mapping[str, Any], display_mapping: Mapping[str, str],
        candidate_ids: list[str], metadata: Mapping[str, Any], endpoint_type: str = "interactions",
    ) -> ProviderResult:
        if model_id not in EXACT_MODEL_IDS:
            raise ValueError("model substitution is forbidden")
        if endpoint_type not in {"interactions", "generateContent"}:
            raise ValueError("unknown provider endpoint")
        assert_no_gt_payload({
            "text_payload": dict(text_payload), "system_prompt": system_prompt,
            "response_schema": dict(response_schema), "generation": dict(generation),
            "metadata": dict(metadata),
        })
        binding_keys = {
            "backend", "sample_id", "model_id", "protocol", "evidence_variant",
            "prompt_version", "prompt_hash", "response_schema_hash", "renderer_hash",
            "scene_board_hash", "candidate_board_hash", "candidate_set_hash",
            "display_mapping", "generation_config", "endpoint_type", "replicate_id",
            "text_payload_hash", "response_parser_version", "model_revision",
            "sdk_version",
        }
        bindings = {key: value for key, value in metadata.items() if key in binding_keys}
        if bindings.get("text_payload_hash") != sha256_json(text_payload):
            raise ValueError("request text payload hash mismatch")
        if logical_request_hash(bindings) != request_hash:
            raise ValueError("supplied request hash does not match final semantic request")
        baseline = str(candidate_ids[0])
        cached = self.ledger.cached(request_hash)
        if cached is not None:
            self.ledger.mark_cache_hit(request_hash)
            selected = baseline if cached.parsed_response is None else str(cached.parsed_response["selected_internal_candidate_id"])
            parsed = None
            if cached.parsed_response is not None:
                value = cached.parsed_response
                parsed = ParsedResponse(
                    decision=str(value["decision"]),
                    selected_internal_candidate_id=selected,
                    ordered_internal_candidate_ids=tuple(map(str, value["ordered_internal_candidate_ids"])),
                    evidence_reliability=str(value["evidence_reliability"]),
                    switch_confidence=int(value["switch_confidence"]),
                    reason_codes=tuple(map(str, value.get("reason_codes", []))),
                    parsed=dict(value.get("provider_json", {})),
                    warnings=tuple(map(str, value.get("warnings", []))),
                )
            return ProviderResult(request_hash, cached.status, selected, parsed, cached.raw_response, True, 0,
                                  None if cached.status == "SUCCEEDED" else cached.status.lower(), {},
                                  float(cached.latency_seconds or 0), cached.estimated_cost_usd,
                                  cached.response_model, cached.request_id)
        self._preflight(model_id)
        claim = self.ledger.acquire(request_hash, self.owner, requested_model=model_id,
                                    endpoint_type=endpoint_type, metadata=metadata)
        if claim == "CACHED":
            return self.run(request_hash=request_hash, model_id=model_id, protocol=protocol,
                            text_payload=text_payload, overview_png=overview_png, grid_png=grid_png,
                            system_prompt=system_prompt, response_schema=response_schema,
                            generation=generation, display_mapping=display_mapping,
                            candidate_ids=candidate_ids, metadata=metadata, endpoint_type=endpoint_type)
        if claim != "ACQUIRED":
            raise RuntimeError(f"request is owned by another live worker: {claim}")
        started = time.perf_counter()
        # Retry limits are logical-request limits, not per-process limits.  A
        # resumed RETRYABLE_FAILED row must continue its existing attempt
        # budget instead of receiving a fresh retry allowance.
        attempts = self.ledger.attempt_count(request_hash)
        transport_failures = self.ledger.attempt_count(
            request_hash, status="RETRYABLE_FAILED"
        )
        schema_failures = self.ledger.attempt_count(request_hash, status="SCHEMA_FAILED")
        total_cost = 0.0
        last_raw = None
        last_usage: Mapping[str, Any] = {}
        try:
            while True:
                self._preflight(model_id)
                if self.rate_limiter is not None:
                    self.rate_limiter.wait(model_id)
                self.ledger.renew(request_hash, self.owner, lease_seconds=900)
                try:
                    reservation_id = self.ledger.reserve_attempt(
                        request_hash, self.owner, reserve_cost_usd=self._reserve(model_id),
                        max_cost_usd=self.max_api_cost_usd,
                        max_attempts=self.max_provider_requests,
                    )
                except RuntimeError as error:
                    if "MAX_" in str(error):
                        raise BudgetHardStop(str(error)) from error
                    raise
                attempts += 1
                attempt_started = time.perf_counter()
                request = {
                    "model_id": model_id, "endpoint_type": endpoint_type,
                    "protocol": protocol, "text_payload": dict(text_payload),
                    "overview_png_base64": base64.b64encode(overview_png).decode("ascii"),
                    "grid_png_base64": base64.b64encode(grid_png).decode("ascii"),
                    "system_prompt": system_prompt, "response_schema": dict(response_schema),
                    "generation_config": dict(generation), "store": False,
                    "background": False, "stream": False, "tools": [],
                }
                try:
                    response = self.transport(request)
                    last_raw = str(response.get("raw_text") or "")
                    last_usage = dict(response.get("usage") or {})
                    charge = self._reserve(model_id) if model_id == EXACT_MODEL_IDS[0] else estimate_flash_cost(last_usage)
                    charged = self._reserve(model_id) if charge is None else float(charge)
                    total_cost += charged
                    response_model = response.get("response_model")
                    request_id = response.get("request_id")
                    if response_model is not None:
                        normalized_response_model = str(response_model).split("/")[-1]
                        if normalized_response_model != model_id:
                            drift = ModelUnavailableHardStop(
                                f"provider response model drift: requested {model_id}, received {normalized_response_model}"
                            )
                            self.ledger.record_attempt_and_finalize(
                                request_hash, attempt_status="PERMANENT_FAILED", final_status="PERMANENT_FAILED",
                                latency_seconds=time.perf_counter()-attempt_started,
                                total_latency_seconds=time.perf_counter()-started,
                                error=drift, usage=last_usage, attempt_cost_usd=charged,
                                total_cost_usd=total_cost, raw_response=last_raw, parsed_response=None,
                                response_model=str(response_model), request_id=request_id,
                                reservation_id=reservation_id, owner=self.owner,
                            )
                            raise drift
                    try:
                        parsed = parse_response(last_raw, schema=response_schema,
                                                display_mapping=display_mapping, candidate_ids=candidate_ids)
                    except Exception as error:
                        schema_failures += 1
                        if schema_failures <= self.max_schema_retries:
                            self.ledger.record_attempt(request_hash, status="SCHEMA_FAILED",
                                                       latency_seconds=time.perf_counter()-attempt_started,
                                                       error=error, usage=last_usage,
                                                       estimated_cost_usd=charged,
                                                       response_model=response_model, request_id=request_id,
                                                       reservation_id=reservation_id)
                            continue
                        elapsed = time.perf_counter() - started
                        self.ledger.record_attempt_and_finalize(
                            request_hash, attempt_status="SCHEMA_FAILED", final_status="SCHEMA_FAILED",
                            latency_seconds=time.perf_counter()-attempt_started,
                            total_latency_seconds=elapsed, error=error, usage=last_usage,
                            attempt_cost_usd=charged, total_cost_usd=total_cost,
                            raw_response=last_raw, parsed_response=None,
                            response_model=response_model, request_id=request_id,
                            reservation_id=reservation_id, owner=self.owner,
                        )
                        return ProviderResult(request_hash, "SCHEMA_FAILED", baseline, None, last_raw,
                                              False, attempts, "schema_invalid", last_usage, elapsed,
                                              total_cost, response_model, request_id, response.get("endpoint_used"))
                    normalized = {
                        "decision": parsed.decision,
                        "selected_internal_candidate_id": parsed.selected_internal_candidate_id,
                        "ordered_internal_candidate_ids": list(parsed.ordered_internal_candidate_ids),
                        "evidence_reliability": parsed.evidence_reliability,
                        "switch_confidence": parsed.switch_confidence,
                        "reason_codes": list(parsed.reason_codes),
                        "warnings": list(parsed.warnings),
                        "provider_json": dict(parsed.parsed),
                    }
                    elapsed = time.perf_counter() - started
                    self.ledger.record_attempt_and_finalize(
                        request_hash, attempt_status="SUCCEEDED", final_status="SUCCEEDED",
                        latency_seconds=time.perf_counter()-attempt_started,
                        total_latency_seconds=elapsed, usage=last_usage,
                        attempt_cost_usd=charged, total_cost_usd=total_cost,
                        raw_response=last_raw, parsed_response=normalized,
                        response_model=response_model, request_id=request_id,
                        reservation_id=reservation_id, owner=self.owner,
                    )
                    return ProviderResult(request_hash, "SUCCEEDED", parsed.selected_internal_candidate_id,
                                          parsed, last_raw, False, attempts, None, last_usage,
                                          elapsed, total_cost, response_model, request_id, response.get("endpoint_used"))
                except KeyboardInterrupt as error:
                    # A local interruption can occur after the provider accepted
                    # the request.  Preserve the reservation as a conservative
                    # attempt and make this exact hash terminal so resume cannot
                    # issue an untracked duplicate request.
                    charged = self._reserve(model_id)
                    total_cost += charged
                    interrupted = RuntimeError("local runner interrupted; provider completion state unknown")
                    self.ledger.record_attempt_and_finalize(
                        request_hash, attempt_status="PERMANENT_FAILED", final_status="TECHNICAL_FALLBACK",
                        latency_seconds=time.perf_counter()-attempt_started,
                        total_latency_seconds=time.perf_counter()-started,
                        error=interrupted, attempt_cost_usd=charged, total_cost_usd=total_cost,
                        raw_response=last_raw, parsed_response=None, response_model=None,
                        request_id=None, reservation_id=reservation_id, owner=self.owner,
                    )
                    raise error
                except (AuthenticationHardStop, BudgetHardStop, ModelUnavailableHardStop):
                    raise
                except Exception as error:
                    code = _status_code(error)
                    if endpoint_type == "interactions" and _explicit_interactions_unsupported(error):
                        charged = self._reserve(model_id)
                        total_cost += charged
                        self.ledger.record_attempt_and_finalize(
                            request_hash, attempt_status="PERMANENT_FAILED", final_status="PERMANENT_FAILED",
                            latency_seconds=time.perf_counter()-attempt_started,
                            total_latency_seconds=time.perf_counter()-started,
                            error=error, attempt_cost_usd=charged, total_cost_usd=total_cost,
                            raw_response=None, parsed_response=None, response_model=None,
                            request_id=None, reservation_id=reservation_id, owner=self.owner,
                        )
                        fallback_metadata = dict(metadata)
                        fallback_bindings = {
                            key: value for key, value in fallback_metadata.items()
                            if key in {
                                "backend", "sample_id", "model_id", "protocol", "evidence_variant",
                                "prompt_version", "prompt_hash", "response_schema_hash", "renderer_hash",
                                "scene_board_hash", "candidate_board_hash", "candidate_set_hash",
                                "display_mapping", "generation_config", "endpoint_type", "replicate_id",
                                "text_payload_hash", "response_parser_version", "model_revision",
                                "sdk_version",
                            }
                        }
                        fallback_bindings["endpoint_type"] = "generateContent"
                        fallback_hash = logical_request_hash(fallback_bindings)
                        fallback_metadata["endpoint_type"] = "generateContent"
                        fallback_metadata["fallback_from_request_hash"] = request_hash
                        fallback_metadata["request_hash"] = fallback_hash
                        fallback_result = self.run(
                            request_hash=fallback_hash, model_id=model_id, protocol=protocol,
                            text_payload=text_payload, overview_png=overview_png, grid_png=grid_png,
                            system_prompt=system_prompt, response_schema=response_schema,
                            generation=generation, display_mapping=display_mapping,
                            candidate_ids=candidate_ids, metadata=fallback_metadata,
                            endpoint_type="generateContent",
                        )
                        self.ledger.set_alias(
                            request_hash, fallback_result.request_hash,
                            reason="explicit_interactions_unsupported_generateContent_fallback",
                        )
                        return fallback_result
                    unknown_remote_state = isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) or code is None
                    retryable = code in RETRYABLE_HTTP and not unknown_remote_state
                    charged = 0.0 if code == 429 else self._reserve(model_id)
                    total_cost += charged
                    if code in {401, 403}:
                        self.ledger.record_attempt_and_finalize(
                            request_hash, attempt_status="PERMANENT_FAILED", final_status="PERMANENT_FAILED",
                            latency_seconds=time.perf_counter()-attempt_started,
                            total_latency_seconds=time.perf_counter()-started,
                            error=error, attempt_cost_usd=charged, total_cost_usd=total_cost,
                            raw_response=last_raw, parsed_response=None, response_model=None,
                            request_id=None, reservation_id=reservation_id, owner=self.owner,
                        )
                        raise AuthenticationHardStop("Gemini credential was rejected") from error
                    if code == 404:
                        self.ledger.record_attempt_and_finalize(
                            request_hash, attempt_status="PERMANENT_FAILED", final_status="PERMANENT_FAILED",
                            latency_seconds=time.perf_counter()-attempt_started,
                            total_latency_seconds=time.perf_counter()-started,
                            error=error, attempt_cost_usd=charged, total_cost_usd=total_cost,
                            raw_response=last_raw, parsed_response=None, response_model=None,
                            request_id=None, reservation_id=reservation_id, owner=self.owner,
                        )
                        raise ModelUnavailableHardStop(f"exact model unavailable: {model_id}") from error
                    transport_failures += int(retryable)
                    retry_limit = self.max_rate_limit_retries if code == 429 else self.max_transport_retries
                    if retryable and transport_failures <= retry_limit:
                        self.ledger.record_attempt(
                            request_hash, status="RETRYABLE_FAILED",
                            latency_seconds=time.perf_counter()-attempt_started,
                            error=error, estimated_cost_usd=charged,
                            reservation_id=reservation_id,
                        )
                        retry_after = _retry_after_seconds(error) if code == 429 else None
                        delay = retry_after if retry_after is not None else min(30.0, 1.5 * 2 ** (transport_failures - 1))
                        delay += random.Random(f"{request_hash}:{transport_failures}").random() * 0.25
                        if self.rate_limiter is not None:
                            self.rate_limiter.defer(model_id, delay)
                        else:
                            self.sleep(delay)
                        continue
                    status = "TECHNICAL_FALLBACK" if retryable or unknown_remote_state else "PERMANENT_FAILED"
                    elapsed = time.perf_counter() - started
                    self.ledger.record_attempt_and_finalize(
                        request_hash,
                        attempt_status="RETRYABLE_FAILED" if retryable else "PERMANENT_FAILED",
                        final_status=status, latency_seconds=time.perf_counter()-attempt_started,
                        total_latency_seconds=elapsed, error=error,
                        attempt_cost_usd=charged, total_cost_usd=total_cost,
                        raw_response=last_raw, parsed_response=None, response_model=None,
                        request_id=None, reservation_id=reservation_id, owner=self.owner,
                    )
                    if code == 429:
                        raise ProviderQuotaHardStop(
                            f"persistent provider quota exhausted for {model_id} after {attempts} attempts"
                        ) from error
                    return ProviderResult(request_hash, status, baseline, None, last_raw, False,
                                          attempts, "unknown_remote_timeout_no_retry" if unknown_remote_state else ("transport_retries_exhausted" if retryable else f"http_{code}"),
                                          last_usage, elapsed, total_cost, None, None)
        except Exception:
            self.ledger.release(request_hash, self.owner)
            raise
