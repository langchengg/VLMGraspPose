"""Deterministic, local-only VLM re-ranking with strict audit and fallback.

The input ``candidate_ids`` order is the frozen full-precision GQ-CNN order.
Consequently, falling back to q-only is always the deterministic operation of
preserving that order and selecting its first element.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import numbers
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SYSTEM_PROMPT = """You are a robotic parallel-jaw grasp candidate evaluator.
You must rank only the candidate IDs provided.
Do not invent, move, rotate, resize, or merge candidates.
Use the natural-language instruction to identify the referred target.
Then judge whether both jaw regions contact the predicted target,
whether the grasp width is compatible,
whether depth is locally continuous,
whether the candidate crosses a depth boundary or nearby object,
and whether visible clearance appears adequate.
The images and metadata may be uncertain.
When evidence is insufficient, preserve the original GQ-CNN Top-1
or abstain.
Never claim physical grasp success.
Return only the required JSON object."""

REASON_CODES = (
    "target_support",
    "jaw_contact",
    "width_fit",
    "depth_continuity",
    "clearance",
    "candidate_consensus",
    "uncertain",
    "occlusion",
)

ALLOWED_METADATA_FIELDS = (
    "q_percentile",
    "soft_mask_support",
    "width_compatibility",
    "jaw_depth_difference",
    "clearance_proxy",
    "candidate_cluster_size",
)

FORBIDDEN_GT_KEY_FRAGMENTS = (
    "candidate_positive",
    "candidate_correct",
    "best_gt",
    "gt_grasp",
    "gt_mask",
    "ground_truth",
    "first_valid_rank",
    "center_error",
    "angle_error",
    "iou_with_gt",
    "correct_candidate",
    "target_object_id",
    "j@1",
    "j@any",
)

_CANDIDATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def ranking_json_schema() -> dict[str, Any]:
    """Return the full JSON schema sent through Ollama's ``format`` field."""

    ranked_item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_id": {"type": "string"},
            "score": {"type": "number"},
            "reason_codes": {
                "type": "array",
                "items": {"type": "string", "enum": list(REASON_CODES)},
                "uniqueItems": True,
            },
        },
        "required": ["candidate_id", "score", "reason_codes"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_candidate_id": {"type": "string"},
            "ranking": {"type": "array", "items": ranked_item},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "abstain": {"type": "boolean"},
            "switch_from_original_top1": {"type": "boolean"},
        },
        "required": [
            "selected_candidate_id",
            "ranking",
            "confidence",
            "abstain",
            "switch_from_original_top1",
        ],
    }


@dataclass(frozen=True)
class VLMGenerationOptions:
    """Frozen deterministic generation settings."""

    temperature: float = 0.0
    seed: int = 20260728
    max_output_tokens: int = 768
    stream: bool = False
    think: bool = False

    def __post_init__(self) -> None:
        if self.temperature != 0.0:
            raise ValueError("VLM re-ranking requires temperature=0")
        if self.stream:
            raise ValueError("VLM re-ranking requires stream=false")
        if self.think:
            raise ValueError("VLM re-ranking requires think=false")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.max_output_tokens,
            "stream": self.stream,
            "think": self.think,
        }


@dataclass(frozen=True)
class VLMRankingRequest:
    """One independent sample-level request; no conversation history is kept."""

    sample_id: str
    instruction: str
    candidate_ids: tuple[str, ...]
    original_top1_candidate_id: str
    image_paths: tuple[Path, Path]
    visualization_manifest_path: Path
    candidate_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    include_metadata: bool = False
    input_recipe_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        object.__setattr__(
            self, "image_paths", tuple(Path(path) for path in self.image_paths)
        )
        object.__setattr__(
            self,
            "visualization_manifest_path",
            Path(self.visualization_manifest_path),
        )
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not self.instruction.strip():
            raise ValueError("instruction must be non-empty")
        if not self.candidate_ids:
            raise ValueError("candidate_ids must be non-empty")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        invalid = [
            candidate_id
            for candidate_id in self.candidate_ids
            if not _CANDIDATE_ID_PATTERN.fullmatch(candidate_id)
        ]
        if invalid:
            raise ValueError(f"invalid candidate ID syntax: {invalid[:3]}")
        if self.original_top1_candidate_id != self.candidate_ids[0]:
            raise ValueError(
                "candidate_ids must use q-only order with original Top-1 first"
            )
        if len(self.image_paths) != 2:
            raise ValueError("exactly two VLM input images are required")
        if self.image_paths[0].resolve() == self.image_paths[1].resolve():
            raise ValueError("the two VLM input images must be distinct")
        if (
            self.input_recipe_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.input_recipe_sha256) is None
        ):
            raise ValueError(
                "input_recipe_sha256 must be a lowercase SHA-256"
            )


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    score: float
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class VLMRankingResponse:
    """Effective ranking plus a complete inference/cache audit record."""

    sample_id: str
    selected_candidate_id: str
    ranking: tuple[RankedCandidate, ...]
    confidence: float
    abstain: bool
    switch_from_original_top1: bool
    fallback: bool
    fallback_reason: str | None
    request_hash: str
    cache_hit: bool
    latency_seconds: float
    prompt_eval_count: int | None
    eval_count: int | None
    total_duration_ns: int | None
    raw_response: Mapping[str, Any] | None
    parsed_model_response: Mapping[str, Any] | None
    parser_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "selected_candidate_id": self.selected_candidate_id,
            "ranking": [item.to_dict() for item in self.ranking],
            "confidence": self.confidence,
            "abstain": self.abstain,
            "switch_from_original_top1": self.switch_from_original_top1,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "request_hash": self.request_hash,
            "cache_hit": self.cache_hit,
            "latency_seconds": self.latency_seconds,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "total_duration_ns": self.total_duration_ns,
            "raw_response": self.raw_response,
            "parsed_model_response": self.parsed_model_response,
            "parser_error": self.parser_error,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, cache_hit: bool
    ) -> "VLMRankingResponse":
        return cls(
            sample_id=str(value["sample_id"]),
            selected_candidate_id=str(value["selected_candidate_id"]),
            ranking=tuple(
                RankedCandidate(
                    candidate_id=str(item["candidate_id"]),
                    score=float(item["score"]),
                    reason_codes=tuple(str(code) for code in item["reason_codes"]),
                )
                for item in value["ranking"]
            ),
            confidence=float(value["confidence"]),
            abstain=bool(value["abstain"]),
            switch_from_original_top1=bool(
                value["switch_from_original_top1"]
            ),
            fallback=bool(value["fallback"]),
            fallback_reason=value.get("fallback_reason"),
            request_hash=str(value["request_hash"]),
            cache_hit=cache_hit,
            latency_seconds=float(value["latency_seconds"]),
            prompt_eval_count=_optional_int(value.get("prompt_eval_count")),
            eval_count=_optional_int(value.get("eval_count")),
            total_duration_ns=_optional_int(value.get("total_duration_ns")),
            raw_response=value.get("raw_response"),
            parsed_model_response=value.get("parsed_model_response"),
            parser_error=value.get("parser_error"),
        )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


SESSION_AUDIT_POLICY_VERSION = 1


def stable_session_contract(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Project a live Ollama audit onto immutable runtime/model identity."""

    ollama = audit.get("ollama", {})
    model = audit.get("model", {})
    machine = audit.get("machine", {})
    layers = model.get("layers", []) if isinstance(model, Mapping) else []
    if (
        not isinstance(ollama, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(machine, Mapping)
        or not isinstance(layers, Sequence)
        or isinstance(layers, (str, bytes, bytearray))
    ):
        raise ValueError("local Ollama audit has no stable session contract")
    normalized_layers = sorted(
        [
            {
                "digest": str(layer["digest"]),
                "media_type": str(layer["mediaType"]),
                "declared_size": int(layer["size"]),
                "local_size": int(layer["local_size"]),
                "local_sha256": str(layer["local_sha256"]),
            }
            for layer in layers
            if isinstance(layer, Mapping)
        ],
        key=lambda item: (
            item["digest"],
            item["media_type"],
            item["declared_size"],
        ),
    )
    if len(normalized_layers) != len(layers):
        raise ValueError("local Ollama audit contains malformed model layers")
    contract = {
        "policy_version": SESSION_AUDIT_POLICY_VERSION,
        "machine_architecture": str(machine["architecture"]),
        "ollama": {
            "cli_version": str(ollama["cli_version"]),
            "api_version": dict(ollama["api_version"]),
            "executable_path": str(ollama["executable_path"]),
            "executable_sha256": str(ollama["executable_sha256"]),
            "endpoint": str(ollama["endpoint"]),
            "server_config_path": str(ollama["server_config_path"]),
            "server_config_sha256": str(ollama["server_config_sha256"]),
            "disable_ollama_cloud": bool(
                ollama.get("server_config", {}).get(
                    "disable_ollama_cloud"
                )
            ),
        },
        "model": {
            "exact_name": str(model["exact_name"]),
            "manifest_sha256": str(model["manifest_sha256"]),
            "model_layer_digest": str(model["model_layer_digest"]),
            "model_layer_bytes": int(model["model_layer_bytes"]),
            "quantization": str(model["quantization"]),
            "layers": normalized_layers,
        },
    }
    if (
        contract["ollama"]["endpoint"] != "http://127.0.0.1:11434"
        or contract["ollama"]["disable_ollama_cloud"] is not True
    ):
        raise ValueError("stable Ollama contract is not loopback/local-only")
    return contract


def stable_session_contract_sha256(audit: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(stable_session_contract(audit))
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_keys(value: Any, path: str = "request") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in FORBIDDEN_GT_KEY_FRAGMENTS):
                raise ValueError(f"forbidden GT-derived request field: {path}.{key}")
            _audit_keys(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            _audit_keys(nested, f"{path}[{index}]")


def audit_request_gt_free(request: VLMRankingRequest) -> dict[str, Any]:
    """Reject GT-derived fields and verify the visualization provenance sidecar."""

    _audit_keys(request.candidate_metadata)
    if not request.include_metadata and request.candidate_metadata:
        raise ValueError("visual-only requests must not carry candidate metadata")
    if request.include_metadata:
        if set(request.candidate_metadata) != set(request.candidate_ids):
            raise ValueError("metadata candidate IDs must exactly match candidate_ids")
        for candidate_id, metadata in request.candidate_metadata.items():
            extra = set(metadata) - set(ALLOWED_METADATA_FIELDS)
            missing = set(ALLOWED_METADATA_FIELDS) - set(metadata)
            if extra or missing:
                raise ValueError(
                    f"{candidate_id} metadata schema mismatch: "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            for key, value in metadata.items():
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise ValueError(f"{candidate_id}.{key} must be numeric")
                if not math.isfinite(float(value)):
                    raise ValueError(f"{candidate_id}.{key} must be finite")

    for image_path in request.image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    manifest_path = request.visualization_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _audit_keys(manifest.get("inputs", {}))
    if manifest.get("sample_id") != request.sample_id:
        raise ValueError("visualization manifest sample_id does not match request")
    if manifest.get("gt_free") is not True:
        raise ValueError("visualization manifest does not assert gt_free=true")
    if manifest.get("candidate_ids") != list(request.candidate_ids):
        raise ValueError("visualization candidate IDs do not match request IDs")
    if (
        request.input_recipe_sha256 is not None
        and manifest.get("source_recipe_sha256")
        != request.input_recipe_sha256
    ):
        raise ValueError("visualization sidecar does not match input recipe")
    expected_images = {
        str(Path(item["path"]).resolve()): str(item["sha256"])
        for item in manifest.get("output_images", [])
    }
    actual_images = {
        str(path.resolve()): _sha256_file(path) for path in request.image_paths
    }
    if expected_images != actual_images:
        raise ValueError("visualization image hashes do not match sidecar")
    return {
        "gt_free": True,
        "candidate_ids_complete_unique": True,
        "visualization_manifest_sha256": _sha256_file(manifest_path),
        "image_sha256": actual_images,
    }


def build_user_prompt(request: VLMRankingRequest) -> str:
    """Build the deterministic candidate-complete prompt without GT fields."""

    lines = [
        f"Instruction: {request.instruction.strip()}",
        "Candidate IDs in original full-precision GQ-CNN order "
        f"(first is original Top-1): {json.dumps(list(request.candidate_ids))}",
        "Rank every candidate ID exactly once. The selected_candidate_id must "
        "equal the first candidate in ranking.",
    ]
    if request.include_metadata:
        metadata = _normalized_metadata(request)
        lines.append(
            "Metadata semantics: q_percentile, soft_mask_support, and "
            "width_compatibility are dimensionless values in [0,1] where higher "
            "is generally better; jaw_depth_difference is in metres where lower "
            "is generally better; clearance_proxy is a dimensionless visible-"
            "surface score in [0,1] where higher is generally better, not a "
            "physical collision guarantee; candidate_cluster_size is a count."
        )
        lines.append(
            "Inference-allowed candidate summaries: "
            + json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
    lines.append(
        "Required JSON Schema: "
        + json.dumps(ranking_json_schema(), sort_keys=True, separators=(",", ":"))
    )
    prompt = "\n".join(lines)
    for fragment in FORBIDDEN_GT_KEY_FRAGMENTS:
        if fragment in prompt.lower():
            raise AssertionError(f"GT field leaked into prompt: {fragment}")
    return prompt


def _normalized_metadata(
    request: VLMRankingRequest,
) -> dict[str, dict[str, float]]:
    if not request.include_metadata:
        return {}
    return {
        candidate_id: {
            key: float(request.candidate_metadata[candidate_id][key])
            for key in ALLOWED_METADATA_FIELDS
        }
        for candidate_id in request.candidate_ids
    }


class LocalVLMBackend(ABC):
    """Shared backend contract for local visual candidate rankers."""

    @abstractmethod
    def rank_candidates(self, request: VLMRankingRequest) -> VLMRankingResponse:
        """Rank one frozen candidate set without changing any candidate pose."""


Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any) -> None:
        return None


def _local_http_transport(
    endpoint: str, payload: Mapping[str, Any], timeout_seconds: float
) -> Mapping[str, Any]:
    body = _canonical_json(payload)
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect()
    )
    with opener.open(request, timeout=timeout_seconds) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("Ollama response exceeds 4 MiB limit")
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Ollama response body must be a JSON object")
    return value


def _validate_loopback_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 11434
        or parsed.path != "/api/chat"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "Ollama endpoint must be exactly the loopback URL "
            "http://127.0.0.1:11434/api/chat"
        )
    return endpoint


class ResponseValidationError(ValueError):
    """Model JSON was syntactically valid but violated the ranking contract."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def validate_ranking_response(
    value: Any, request: VLMRankingRequest
) -> tuple[RankedCandidate, ...]:
    """Strictly validate schema plus candidate-set semantic constraints."""

    if not isinstance(value, Mapping):
        raise ResponseValidationError("invalid_json_schema", "response is not an object")
    required = {
        "selected_candidate_id",
        "ranking",
        "confidence",
        "abstain",
        "switch_from_original_top1",
    }
    if set(value) != required:
        raise ResponseValidationError(
            "invalid_json_schema",
            f"response keys mismatch: expected={sorted(required)}, got={sorted(value)}",
        )
    selected = value["selected_candidate_id"]
    if not isinstance(selected, str) or selected not in request.candidate_ids:
        raise ResponseValidationError(
            "invalid_candidate_id", f"invalid selected_candidate_id={selected!r}"
        )
    ranking_value = value["ranking"]
    if not isinstance(ranking_value, list):
        raise ResponseValidationError("invalid_json_schema", "ranking must be a list")
    parsed: list[RankedCandidate] = []
    for index, item in enumerate(ranking_value):
        if not isinstance(item, Mapping) or set(item) != {
            "candidate_id",
            "score",
            "reason_codes",
        }:
            raise ResponseValidationError(
                "invalid_json_schema", f"ranking[{index}] schema mismatch"
            )
        candidate_id = item["candidate_id"]
        if not isinstance(candidate_id, str) or candidate_id not in request.candidate_ids:
            raise ResponseValidationError(
                "invalid_candidate_id",
                f"ranking[{index}] has invalid candidate_id={candidate_id!r}",
            )
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ResponseValidationError(
                "non_finite_score", f"ranking[{index}] score is not numeric"
            )
        if not math.isfinite(float(score)):
            raise ResponseValidationError(
                "non_finite_score", f"ranking[{index}] score is not finite"
            )
        codes = item["reason_codes"]
        if (
            not isinstance(codes, list)
            or len(codes) != len(set(codes))
            or any(not isinstance(code, str) or code not in REASON_CODES for code in codes)
        ):
            raise ResponseValidationError(
                "invalid_reason_code", f"ranking[{index}] has invalid reason_codes"
            )
        parsed.append(
            RankedCandidate(candidate_id, float(score), tuple(codes))
        )
    parsed_ids = [item.candidate_id for item in parsed]
    if len(parsed_ids) != len(set(parsed_ids)):
        raise ResponseValidationError(
            "duplicate_candidate_id", "ranking candidate IDs are not unique"
        )
    missing = set(request.candidate_ids) - set(parsed_ids)
    extra = set(parsed_ids) - set(request.candidate_ids)
    if missing or extra or len(parsed_ids) != len(request.candidate_ids):
        raise ResponseValidationError(
            "missing_candidate",
            f"ranking candidate set mismatch: missing={sorted(missing)}, extra={sorted(extra)}",
        )
    if not parsed or parsed[0].candidate_id != selected:
        raise ResponseValidationError(
            "selected_ranking_mismatch",
            "selected_candidate_id must be the first ranked candidate",
        )
    if any(
        parsed[index].score < parsed[index + 1].score
        for index in range(len(parsed) - 1)
    ):
        raise ResponseValidationError(
            "invalid_score_order", "ranking scores must be non-increasing"
        )
    confidence = value["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ResponseValidationError(
            "invalid_confidence", "confidence must be finite and in [0,1]"
        )
    if not isinstance(value["abstain"], bool) or not isinstance(
        value["switch_from_original_top1"], bool
    ):
        raise ResponseValidationError(
            "invalid_json_schema", "abstain and switch flag must be booleans"
        )
    expected_switch = selected != request.original_top1_candidate_id
    if value["switch_from_original_top1"] != expected_switch:
        raise ResponseValidationError(
            "invalid_switch_flag", "switch flag is inconsistent with selected candidate"
        )
    return tuple(parsed)


def validate_effective_result_record(
    value: Mapping[str, Any],
    *,
    candidate_ids: Sequence[str],
    original_top1_candidate_id: str,
) -> tuple[RankedCandidate, ...]:
    """Revalidate a persisted backend result before a downstream decision."""

    required = {
        "selected_candidate_id",
        "ranking",
        "confidence",
        "abstain",
        "switch_from_original_top1",
        "fallback",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ResponseValidationError(
            "invalid_json_schema", f"stored result missing fields: {missing}"
        )

    class _CandidateContract:
        pass

    request = _CandidateContract()
    request.candidate_ids = tuple(map(str, candidate_ids))
    request.original_top1_candidate_id = str(original_top1_candidate_id)
    effective = {
        key: value[key]
        for key in (
            "selected_candidate_id",
            "ranking",
            "confidence",
            "abstain",
            "switch_from_original_top1",
        )
    }
    parsed = validate_ranking_response(effective, request)  # type: ignore[arg-type]
    if bool(value["fallback"]):
        observed_ids = [item.candidate_id for item in parsed]
        if (
            observed_ids != list(request.candidate_ids)
            or str(value["selected_candidate_id"])
            != request.original_top1_candidate_id
            or bool(value["switch_from_original_top1"])
            or float(value["confidence"]) != 0.0
        ):
            raise ResponseValidationError(
                "invalid_fallback", "fallback must preserve the exact q-only ranking"
            )
    else:
        model_response = value.get("parsed_model_response")
        if not isinstance(model_response, Mapping):
            raise ResponseValidationError(
                "invalid_json_schema",
                "successful stored result requires parsed_model_response",
            )
        validate_ranking_response(model_response, request)  # type: ignore[arg-type]
        if _canonical_json(model_response) != _canonical_json(effective):
            raise ResponseValidationError(
                "stored_response_mismatch",
                "effective result differs from parsed model response",
            )
    return parsed


class OllamaLocalVLMBackend(LocalVLMBackend):
    """Ollama REST backend constrained to the local loopback interface."""

    def __init__(
        self,
        *,
        model_name: str,
        model_digest: str,
        backend_version: str,
        cache_dir: str | Path,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout_seconds: float = 180.0,
        options: VLMGenerationOptions | None = None,
        runtime_contract_sha256: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        if not model_name.strip() or not model_digest.strip() or not backend_version.strip():
            raise ValueError("model name, digest, and backend version are required")
        self.model_name = model_name
        self.model_digest = model_digest
        self.backend_version = backend_version
        self.runtime_contract_sha256 = (
            str(runtime_contract_sha256)
            if runtime_contract_sha256 is not None
            else None
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_database_path = self.cache_dir / "responses.sqlite3"
        self._initialize_cache()
        self.endpoint = _validate_loopback_endpoint(endpoint)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.options = options or VLMGenerationOptions()
        self._transport = transport or _local_http_transport

    def _payload_and_hash(
        self, request: VLMRankingRequest
    ) -> tuple[dict[str, Any], str]:
        audit = audit_request_gt_free(request)
        user_prompt = build_user_prompt(request)
        images = [
            base64.b64encode(path.read_bytes()).decode("ascii")
            for path in request.image_paths
        ]
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt, "images": images},
            ],
            "format": ranking_json_schema(),
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "seed": self.options.seed,
                "num_predict": self.options.max_output_tokens,
            },
        }
        hash_record = {
            "sample_id": request.sample_id,
            "request_variant": (
                "visual_metadata" if request.include_metadata else "visual"
            ),
            "model_name": self.model_name,
            "model_digest": self.model_digest,
            "backend_version": self.backend_version,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "input_recipe_sha256": request.input_recipe_sha256,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "candidate_metadata": _normalized_metadata(request),
            "generation_options": self.options.to_dict(),
            "image_sha256": [
                _sha256_file(path) for path in request.image_paths
            ],
            "request_audit": {
                "gt_free": audit["gt_free"],
                "candidate_ids_complete_unique": audit[
                    "candidate_ids_complete_unique"
                ],
            },
        }
        return payload, hashlib.sha256(_canonical_json(hash_record)).hexdigest()

    def _fallback(
        self,
        *,
        request: VLMRankingRequest,
        request_hash: str,
        reason: str,
        latency_seconds: float,
        raw_response: Mapping[str, Any] | None,
        parsed_response: Mapping[str, Any] | None,
        parser_error: str | None,
    ) -> VLMRankingResponse:
        count = len(request.candidate_ids)
        ranking = tuple(
            RankedCandidate(
                candidate_id=candidate_id,
                score=float(count - index) / float(count),
                reason_codes=("uncertain",),
            )
            for index, candidate_id in enumerate(request.candidate_ids)
        )
        return VLMRankingResponse(
            sample_id=request.sample_id,
            selected_candidate_id=request.original_top1_candidate_id,
            ranking=ranking,
            confidence=0.0,
            abstain=reason == "abstain",
            switch_from_original_top1=False,
            fallback=True,
            fallback_reason=reason,
            request_hash=request_hash,
            cache_hit=False,
            latency_seconds=latency_seconds,
            prompt_eval_count=_response_int(raw_response, "prompt_eval_count"),
            eval_count=_response_int(raw_response, "eval_count"),
            total_duration_ns=_response_int(raw_response, "total_duration"),
            raw_response=raw_response,
            parsed_model_response=parsed_response,
            parser_error=parser_error,
        )

    def _cache_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.cache_database_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize_cache(self) -> None:
        with self._cache_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    request_hash TEXT PRIMARY KEY,
                    response_json BLOB NOT NULL,
                    response_sha256 TEXT NOT NULL,
                    created_at_unix_ns INTEGER NOT NULL
                ) WITHOUT ROWID
                """
            )

    def _load_cache(self, request_hash: str) -> dict[str, Any] | None:
        with self._cache_connection() as connection:
            row = connection.execute(
                """
                SELECT response_json, response_sha256
                FROM responses
                WHERE request_hash = ?
                """,
                (request_hash,),
            ).fetchone()
        if row is None:
            return None
        encoded = bytes(row[0])
        if hashlib.sha256(encoded).hexdigest() != str(row[1]):
            raise ValueError(
                f"SQLite VLM cache payload hash mismatch: {request_hash}"
            )
        cached = json.loads(encoded)
        if (
            not isinstance(cached, dict)
            or cached.get("request_hash") != request_hash
        ):
            raise ValueError(
                f"SQLite VLM cache request hash mismatch: {request_hash}"
            )
        return cached

    def _save_cache(self, response: VLMRankingResponse) -> None:
        encoded = _canonical_json(response.to_dict())
        digest = hashlib.sha256(encoded).hexdigest()
        with self._cache_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT response_json, response_sha256
                FROM responses
                WHERE request_hash = ?
                """,
                (response.request_hash,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO responses (
                        request_hash,
                        response_json,
                        response_sha256,
                        created_at_unix_ns
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        response.request_hash,
                        sqlite3.Binary(encoded),
                        digest,
                        time.time_ns(),
                    ),
                )
            elif (
                bytes(existing[0]) != encoded
                or str(existing[1]) != digest
            ):
                connection.execute("ROLLBACK")
                raise ValueError(
                    "refusing to overwrite a different cached VLM response"
                )
            connection.execute("COMMIT")

    def rank_candidates(self, request: VLMRankingRequest) -> VLMRankingResponse:
        payload, request_hash = self._payload_and_hash(request)
        cached = self._load_cache(request_hash)
        if cached is not None:
            response = VLMRankingResponse.from_dict(cached, cache_hit=True)
            self._validate_cached_response(response, request)
            return response

        started = time.monotonic()
        raw: Mapping[str, Any] | None = None
        parsed: Mapping[str, Any] | None = None
        parser_error: str | None = None
        try:
            raw = self._transport(self.endpoint, payload, self.timeout_seconds)
            if not isinstance(raw, Mapping):
                raise ResponseValidationError(
                    "invalid_backend_response", "Ollama response is not an object"
                )
            if raw.get("model") != self.model_name:
                raise ResponseValidationError(
                    "model_identity_mismatch",
                    f"Ollama returned model={raw.get('model')!r}, "
                    f"expected={self.model_name!r}",
                )
            if raw.get("done") is not True or raw.get("done_reason") != "stop":
                raise ResponseValidationError(
                    "incomplete_backend_response",
                    f"Ollama completion status is done={raw.get('done')!r}, "
                    f"done_reason={raw.get('done_reason')!r}",
                )
            message = raw.get("message")
            if not isinstance(message, Mapping):
                raise ResponseValidationError(
                    "invalid_backend_response", "missing Ollama message object"
                )
            content = message.get("content")
            if not isinstance(content, str):
                raise ResponseValidationError(
                    "invalid_json_schema", "missing message.content string"
                )
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError as error:
                raise ResponseValidationError("parser_failure", str(error)) from error
            if not isinstance(decoded, Mapping):
                raise ResponseValidationError(
                    "invalid_json_schema", "model content must decode to an object"
                )
            parsed = decoded
            ranking = validate_ranking_response(parsed, request)
            latency = time.monotonic() - started
            if parsed["abstain"]:
                response = self._fallback(
                    request=request,
                    request_hash=request_hash,
                    reason="abstain",
                    latency_seconds=latency,
                    raw_response=raw,
                    parsed_response=parsed,
                    parser_error=None,
                )
            else:
                response = VLMRankingResponse(
                    sample_id=request.sample_id,
                    selected_candidate_id=str(parsed["selected_candidate_id"]),
                    ranking=ranking,
                    confidence=float(parsed["confidence"]),
                    abstain=False,
                    switch_from_original_top1=bool(
                        parsed["switch_from_original_top1"]
                    ),
                    fallback=False,
                    fallback_reason=None,
                    request_hash=request_hash,
                    cache_hit=False,
                    latency_seconds=latency,
                    prompt_eval_count=_response_int(raw, "prompt_eval_count"),
                    eval_count=_response_int(raw, "eval_count"),
                    total_duration_ns=_response_int(raw, "total_duration"),
                    raw_response=raw,
                    parsed_model_response=parsed,
                    parser_error=None,
                )
        except ResponseValidationError as error:
            parser_error = str(error)
            response = self._fallback(
                request=request,
                request_hash=request_hash,
                reason=error.reason,
                latency_seconds=time.monotonic() - started,
                raw_response=raw,
                parsed_response=parsed,
                parser_error=parser_error,
            )
        except (TimeoutError, socket.timeout) as error:
            response = self._fallback(
                request=request,
                request_hash=request_hash,
                reason="timeout",
                latency_seconds=time.monotonic() - started,
                raw_response=raw,
                parsed_response=parsed,
                parser_error=f"{type(error).__name__}: {error}",
            )
        except urllib.error.URLError as error:
            reason = "timeout" if isinstance(error.reason, TimeoutError) else "transport_error"
            response = self._fallback(
                request=request,
                request_hash=request_hash,
                reason=reason,
                latency_seconds=time.monotonic() - started,
                raw_response=raw,
                parsed_response=parsed,
                parser_error=f"{type(error).__name__}: {error}",
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            response = self._fallback(
                request=request,
                request_hash=request_hash,
                reason="parser_failure",
                latency_seconds=time.monotonic() - started,
                raw_response=raw,
                parsed_response=parsed,
                parser_error=f"{type(error).__name__}: {error}",
            )
        except OSError as error:
            response = self._fallback(
                request=request,
                request_hash=request_hash,
                reason="transport_error",
                latency_seconds=time.monotonic() - started,
                raw_response=raw,
                parsed_response=parsed,
                parser_error=f"{type(error).__name__}: {error}",
            )
        self._save_cache(response)
        return response

    def _validate_cached_response(
        self, response: VLMRankingResponse, request: VLMRankingRequest
    ) -> None:
        if response.sample_id != request.sample_id:
            raise ValueError("cached response sample_id does not match request")
        if response.request_hash == "":
            raise ValueError("cached response has an empty request hash")
        expected_ids = list(request.candidate_ids)
        observed_ids = [item.candidate_id for item in response.ranking]
        if observed_ids != expected_ids and response.fallback:
            raise ValueError("cached fallback is not the exact q-only ranking")
        if response.fallback:
            if (
                response.selected_candidate_id
                != request.original_top1_candidate_id
                or response.switch_from_original_top1
            ):
                raise ValueError("cached fallback does not preserve q-only Top-1")
            return
        parsed = response.parsed_model_response
        if not isinstance(parsed, Mapping):
            raise ValueError("cached successful response has no parsed model response")
        ranking = validate_ranking_response(parsed, request)
        if tuple(response.ranking) != ranking:
            raise ValueError("cached effective ranking differs from parsed response")
        if response.selected_candidate_id != str(parsed["selected_candidate_id"]):
            raise ValueError("cached selected candidate differs from parsed response")


def _response_int(response: Mapping[str, Any] | None, key: str) -> int | None:
    if response is None:
        return None
    value = response.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
