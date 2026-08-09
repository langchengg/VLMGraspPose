from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from failure_analysis.gemini_crog_evidence_v1.api import estimate_usage_cost_usd

from .api import (
    MODEL_IDS,
    AuthenticationHardStop,
    BudgetHardStop,
    _retry_after,
    _status_code,
    _usage,
)
from .full_list_renderer import (
    EXPECTED_CANDIDATE_IDS,
    full_list_renderer_hash,
    validate_full_list_evidence,
)
from .ledger import PairwiseLedger
from .schema import FullListKeepResponse, parse_full_list_keep_response
from .security import assert_no_ground_truth


FULL_LIST_PROTOCOL = "P2_FULL_LIST_KEEP_PRIOR"
FULL_LIST_REQUEST_HASH_VERSION = "full_list_keep_prior_request_hash_v1"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = REPO_ROOT / "prompts/full_list_keep_prior_v1.txt"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "prompts/full_list_keep_prior_v1.schema.json"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FullListContract:
    system_prompt: str
    response_schema: dict[str, Any]
    prompt_hash: str
    schema_hash: str


def load_full_list_contract(
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> FullListContract:
    """Load and hash the registered P2 prompt and strict response schema."""

    system_prompt = Path(prompt_path).read_text(encoding="utf-8")
    response_schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if not isinstance(response_schema, dict):
        raise ValueError("P2 response schema must be a JSON object")
    if response_schema.get("additionalProperties") is not False:
        raise ValueError("P2 response schema must forbid additional properties")
    if response_schema.get("type") != "object":
        raise ValueError("P2 response schema must describe one object")
    return FullListContract(
        system_prompt=system_prompt,
        response_schema=response_schema,
        prompt_hash=_sha256_text(system_prompt),
        schema_hash=_sha256_json(response_schema),
    )


def full_list_request_hash(
    *,
    sample_id: str,
    model_id: str,
    protocol: str,
    candidate_ids: Sequence[str],
    prompt_hash: str,
    schema_hash: str,
    renderer_hash: str,
    evidence_hash: str,
    board_sha256: str,
    generation: Mapping[str, Any],
) -> str:
    """Hash every semantic and rendered input to one P2 model request."""

    normalized_ids = tuple(str(value) for value in candidate_ids)
    if normalized_ids != EXPECTED_CANDIDATE_IDS:
        raise ValueError("P2 request requires frozen candidate_0..candidate_4")
    payload = {
        "hash_version": FULL_LIST_REQUEST_HASH_VERSION,
        "sample_id": str(sample_id),
        "model_id": str(model_id),
        "protocol": str(protocol),
        "baseline_candidate_id": "candidate_0",
        "candidate_ids": list(normalized_ids),
        "prompt_hash": str(prompt_hash),
        "schema_hash": str(schema_hash),
        "renderer_hash": str(renderer_hash),
        "evidence_hash": str(evidence_hash),
        "board_sha256": str(board_sha256),
        "generation": dict(generation),
    }
    return _sha256_json(payload)


@dataclass(frozen=True)
class FullListAPIResult:
    request_hash: str
    status: str
    selected_candidate_id: str
    parsed: FullListKeepResponse | None
    raw_response: str | None
    cache_hit: bool
    api_attempts: int
    fallback_reason: str | None
    usage: dict[str, Any]
    latency_seconds: float
    estimated_cost_usd: float
    response_model: str | None
    api_request_id: str | None


class FullListInteractionsRunner:
    """Finite, stateless P2 Interactions transport with exact ledger replay."""

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
        max_transport_retries: int = 2,
        max_schema_retries: int = 1,
        owner_id: str | None = None,
    ) -> None:
        self.ledger = ledger
        self.max_spend_usd = float(max_spend_usd)
        self.reserves = {
            "gemini-robotics-er-2-preview": float(er2_request_reserve_usd),
            "gemini-3.6-flash": float(flash_request_reserve_usd),
        }
        if self.max_spend_usd < 0 or any(
            reserve <= 0 for reserve in self.reserves.values()
        ):
            raise ValueError("budget and request reserves must be positive")
        self.max_transport_retries = int(max_transport_retries)
        self.max_schema_retries = int(max_schema_retries)
        if self.max_transport_retries < 0 or self.max_schema_retries < 0:
            raise ValueError("transport and schema retry limits must be non-negative")
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self.client = client
        self.sleep = sleep
        self.owner_id = owner_id or f"full-list-{os.getpid()}-{uuid.uuid4().hex}"

    def _client(self) -> Any:
        if not self.api_key:
            raise AuthenticationHardStop("Gemini API credential is unavailable")
        if self.client is None:
            from google import genai
            from google.genai import types

            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )
        return self.client

    def _check_budget(self, model_id: str) -> None:
        spent = float(self.ledger.summary()["attempt_estimated_cost_usd"])
        if spent + self.reserves[model_id] > self.max_spend_usd + 1e-12:
            raise BudgetHardStop("next request would exceed max API spend")

    def _attempt_charge(self, model_id: str, usage: Mapping[str, Any] | None = None) -> float:
        reserve = self.reserves[model_id]
        if model_id == "gemini-robotics-er-2-preview":
            return reserve
        if usage:
            try:
                return float(estimate_usage_cost_usd(model_id, usage))
            except Exception:
                pass
        return reserve

    @staticmethod
    def _cached_result(request_hash: str, cached: Any) -> FullListAPIResult:
        parsed = (
            FullListKeepResponse.model_validate(cached.parsed_json)
            if cached.parsed_json
            else None
        )
        selected = parsed.selected_candidate_id if parsed is not None else "candidate_0"
        return FullListAPIResult(
            request_hash=request_hash,
            status=str(cached.status),
            selected_candidate_id=selected,
            parsed=parsed,
            raw_response=cached.raw_response,
            cache_hit=True,
            api_attempts=0,
            fallback_reason=(
                None if cached.status == "SUCCEEDED" else str(cached.status).lower()
            ),
            usage={},
            latency_seconds=float(cached.latency_seconds or 0.0),
            estimated_cost_usd=float(cached.estimated_cost_usd),
            response_model=cached.response_model,
            api_request_id=cached.api_request_id,
        )

    def run(
        self,
        *,
        sample_id: str,
        model_id: str,
        protocol: str,
        evidence: Mapping[str, Any],
        board_png: bytes,
        board_metadata: Mapping[str, Any],
        system_prompt: str,
        response_schema: Mapping[str, Any],
        prompt_hash: str,
        schema_hash: str,
        renderer_hash: str,
        temperature: float = 0.0,
        thinking_level: str = "low",
        max_output_tokens: int = 1024,
    ) -> FullListAPIResult:
        if model_id not in MODEL_IDS:
            raise ValueError("model substitution is forbidden")
        if protocol != FULL_LIST_PROTOCOL:
            raise ValueError("P2 full-list protocol identity changed")
        if not isinstance(board_png, bytes) or not board_png.startswith(b"\x89PNG"):
            raise ValueError("P2 board must be PNG bytes")

        # Validate all semantic bindings before cache lookup or ledger mutation.
        assert_no_ground_truth(
            {
                "system_instruction": system_prompt,
                "response_schema": dict(response_schema),
                "evidence": dict(evidence),
                "board_metadata": dict(board_metadata),
            }
        )
        candidate_ids = validate_full_list_evidence(evidence)
        if str(evidence.get("sample_id")) != str(sample_id):
            raise ValueError("P2 sample identity differs from evidence")
        registered_contract = load_full_list_contract()
        if (
            system_prompt != registered_contract.system_prompt
            or str(prompt_hash) != registered_contract.prompt_hash
            or _sha256_text(system_prompt) != str(prompt_hash)
        ):
            raise ValueError("P2 prompt hash mismatch")
        response_schema = dict(response_schema)
        if (
            response_schema != registered_contract.response_schema
            or str(schema_hash) != registered_contract.schema_hash
            or _sha256_json(response_schema) != str(schema_hash)
        ):
            raise ValueError("P2 schema hash mismatch")
        if str(renderer_hash) != full_list_renderer_hash():
            raise ValueError("P2 renderer hash mismatch")
        board_sha256 = hashlib.sha256(board_png).hexdigest()
        expected_board_metadata = {
            "image_sha256": board_sha256,
            "evidence_hash": str(evidence["evidence_hash"]),
            "renderer_contract_hash": str(renderer_hash),
            "baseline_candidate_id": "candidate_0",
            "visible_candidate_ids": list(candidate_ids),
        }
        for key, expected in expected_board_metadata.items():
            if board_metadata.get(key) != expected:
                raise ValueError(f"P2 board binding mismatch: {key}")

        generation = {
            "temperature": float(temperature),
            "thinking_level": str(thinking_level),
            "max_output_tokens": int(max_output_tokens),
        }
        request_hash = full_list_request_hash(
            sample_id=str(sample_id),
            model_id=model_id,
            protocol=protocol,
            candidate_ids=candidate_ids,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            renderer_hash=renderer_hash,
            evidence_hash=str(evidence["evidence_hash"]),
            board_sha256=board_sha256,
            generation=generation,
        )
        metadata = {
            "sample_id": str(sample_id),
            "protocol": protocol,
            "baseline_candidate_id": "candidate_0",
            "candidate_ids": list(candidate_ids),
            "instruction": (
                "Review only the frozen Top-5. candidate_0 is the frozen baseline "
                "and explicit KEEP prior; ambiguous evidence must not replace it."
            ),
            "evidence": dict(evidence),
        }
        assert_no_ground_truth(metadata)

        cached = self.ledger.cached(request_hash)
        if cached is not None:
            return self._cached_result(request_hash, cached)
        if not self.api_key:
            raise AuthenticationHardStop("Gemini API credential is unavailable")
        self._check_budget(model_id)
        claim = self.ledger.acquire(request_hash, self.owner_id)
        if claim != "ACQUIRED":
            raise RuntimeError(f"request not acquired: {claim}")

        started = time.perf_counter()
        api_attempts = 0
        transport_failures = 0
        schema_failures = 0
        total_cost = 0.0
        last_raw: str | None = None
        last_usage: dict[str, Any] = {}
        while True:
            try:
                self._check_budget(model_id)
            except BudgetHardStop:
                self.ledger.release(request_hash)
                raise

            api_attempts += 1
            attempt_started = time.perf_counter()
            try:
                response = self._client().interactions.create(
                    model=model_id,
                    input=[
                        {"type": "text", "text": _canonical_json(metadata)},
                        {
                            "type": "image",
                            "data": base64.b64encode(board_png).decode("ascii"),
                            "mime_type": "image/png",
                            "resolution": "high",
                        },
                    ],
                    stream=False,
                    store=False,
                    background=False,
                    service_tier="standard",
                    system_instruction=system_prompt,
                    generation_config=generation,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": response_schema,
                    },
                )
                last_raw = str(getattr(response, "output_text", "") or "")
                last_usage = _usage(response)
                charge = self._attempt_charge(model_id, last_usage)
                total_cost += float(charge)
                try:
                    parsed = parse_full_list_keep_response(last_raw)
                except Exception as error:
                    schema_failures += 1
                    self.ledger.record_attempt(
                        request_hash,
                        "SCHEMA_FAILED",
                        latency_seconds=time.perf_counter() - attempt_started,
                        estimated_cost_usd=charge,
                        error_class=type(error).__name__,
                    )
                    if schema_failures <= self.max_schema_retries:
                        continue
                    latency = time.perf_counter() - started
                    self.ledger.finish(
                        request_hash,
                        "SCHEMA_FAILED",
                        raw_response=last_raw,
                        latency_seconds=latency,
                        estimated_cost_usd=total_cost,
                        requested_model=model_id,
                        response_model=getattr(response, "model", None),
                    )
                    return FullListAPIResult(
                        request_hash, "SCHEMA_FAILED", "candidate_0", None,
                        last_raw, False, api_attempts, "schema_invalid", last_usage,
                        latency, total_cost, getattr(response, "model", None),
                        getattr(response, "id", None),
                    )

                latency = time.perf_counter() - started
                response_model = getattr(response, "model", None)
                response_id = getattr(response, "id", None)
                self.ledger.record_attempt(
                    request_hash,
                    "SUCCEEDED",
                    latency_seconds=time.perf_counter() - attempt_started,
                    estimated_cost_usd=charge,
                )
                self.ledger.finish(
                    request_hash,
                    "SUCCEEDED",
                    raw_response=last_raw,
                    parsed_json=parsed.model_dump(mode="json"),
                    api_request_id=response_id,
                    latency_seconds=latency,
                    estimated_cost_usd=total_cost,
                    requested_model=model_id,
                    response_model=response_model,
                )
                return FullListAPIResult(
                    request_hash, "SUCCEEDED", parsed.selected_candidate_id,
                    parsed, last_raw, False, api_attempts, None, last_usage,
                    latency, total_cost, response_model, response_id,
                )
            except AuthenticationHardStop:
                raise
            except BudgetHardStop:
                raise
            except Exception as error:
                status_code = _status_code(error)
                retryable = status_code in RETRYABLE_STATUS_CODES or status_code is None
                attempt_status = (
                    "RETRYABLE_FAILED" if retryable else "PERMANENT_FAILED"
                )
                charge = self._attempt_charge(model_id)
                total_cost += charge
                self.ledger.record_attempt(
                    request_hash,
                    attempt_status,
                    latency_seconds=time.perf_counter() - attempt_started,
                    estimated_cost_usd=charge,
                    error_class=type(error).__name__,
                )
                if status_code in (401, 403):
                    self.ledger.finish(
                        request_hash,
                        "PERMANENT_FAILED",
                        latency_seconds=time.perf_counter() - started,
                        estimated_cost_usd=total_cost,
                        requested_model=model_id,
                    )
                    raise AuthenticationHardStop(
                        "Gemini API credential was rejected"
                    ) from error
                if retryable:
                    transport_failures += 1
                if retryable and transport_failures <= self.max_transport_retries:
                    delay = _retry_after(error)
                    if delay is None:
                        delay = min(30.0, 1.5 * (2 ** (transport_failures - 1)))
                        delay += random.Random(
                            f"{request_hash}:{transport_failures}"
                        ).random() * 0.25
                    self.sleep(delay)
                    continue

                latency = time.perf_counter() - started
                terminal_status = (
                    "TECHNICAL_FALLBACK" if retryable else "PERMANENT_FAILED"
                )
                fallback_reason = (
                    "transport_retries_exhausted"
                    if retryable
                    else f"http_{status_code or 'permanent_error'}"
                )
                self.ledger.finish(
                    request_hash,
                    terminal_status,
                    raw_response=last_raw,
                    latency_seconds=latency,
                    estimated_cost_usd=total_cost,
                    requested_model=model_id,
                )
                return FullListAPIResult(
                    request_hash, terminal_status, "candidate_0", None, last_raw,
                    False, api_attempts, fallback_reason, last_usage, latency,
                    total_cost, None, None,
                )
