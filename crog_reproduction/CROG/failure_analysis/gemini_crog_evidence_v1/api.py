from __future__ import annotations

import base64
import email.utils
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import MODEL_IDS
from .schema import GeminiRankingResponse, parse_ranking_response, response_json_schema
from .security import (
    assert_display_mapping,
    canonical_request_json,
    redact_sensitive,
)


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}
DEFAULT_CONCURRENCY = 2
# The strict five-candidate JSON is about 500 visible tokens, while medium thinking used
# about 490 tokens in the first access smoke; a ceiling of 512 therefore
# truncated after four visible tokens. A second 1536-token smoke also truncated
# because medium thinking varied up to 2428 tokens. Two Flash samples still
# truncated at 3072; 4096 completed both with 3315/3460 generation tokens.
# Provider usage meters thinking separately, so the budget guard does not treat
# 4096 as a proven upper bound on output-plus-thinking cost.
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_THINKING_LEVEL = "medium"
DEFAULT_IMAGE_RESOLUTION = "high"
# A retryable provider failure is durable and survives process restarts.  Keep
# the cross-resume ceiling separate from max_retries, which only limits one
# runner invocation.  Twelve permits two complete default retry windows while
# preventing a rate-limit outage from becoming an unbounded attempt storm.
DEFAULT_MAX_CUMULATIVE_RETRYABLE_ATTEMPTS = 12
MAX_CUMULATIVE_RETRYABLE_ATTEMPTS_ENV = (
    "GEMINI_MAX_CUMULATIVE_RETRYABLE_ATTEMPTS"
)
LOWER_CEILING_REUSE_REASON = (
    "paid_valid_response_reused_for_higher_output_token_ceiling"
)
REQUEST_STATUSES = frozenset(
    {
        "PLANNED",
        "IN_FLIGHT",
        "SUCCEEDED",
        "RETRYABLE_FAILED",
        "PERMANENT_FAILED",
        "ABSTAIN",
        "TECHNICAL_FALLBACK",
    }
)
TERMINAL_REQUEST_STATUSES = frozenset(
    {"SUCCEEDED", "PERMANENT_FAILED", "ABSTAIN", "TECHNICAL_FALLBACK"}
)

# Google Gemini API Standard-service prices verified against the official
# public pricing table on 2026-08-01. Output pricing includes thinking tokens.
# Cached input is included for completeness even though this experiment sends
# inline images without explicit context caching.
MODEL_PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    "gemini-robotics-er-2-preview": {
        "input": 2.0,
        "cached_input": 0.20,
        "output_and_thought": 10.0,
    },
    "gemini-3.6-flash": {
        "input": 1.5,
        "cached_input": 0.15,
        "output_and_thought": 7.5,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_json_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json", exclude_none=True))
    result: dict[str, Any] = {}
    for name in (
        "name",
        "base_model_id",
        "baseModelId",
        "version",
        "display_name",
        "displayName",
        "input_token_limit",
        "inputTokenLimit",
        "output_token_limit",
        "outputTokenLimit",
        "supported_generation_methods",
        "supportedGenerationMethods",
        "thinking",
    ):
        item = getattr(value, name, None)
        if item is not None:
            result[name] = item
    return result


def model_metadata_fingerprint(metadata: Any) -> str:
    """Hash provider-visible model metadata for best-effort drift checks.

    The Models API does not expose an immutable weights/build identifier, so
    this hook detects visible metadata drift only.
    """
    return sha256_json(_as_json_mapping(metadata))


def compare_model_metadata(expected: Any, observed: Any) -> dict[str, Any]:
    """Return a stable, machine-readable comparison of model metadata."""
    left = _as_json_mapping(expected)
    right = _as_json_mapping(observed)
    keys = sorted(set(left) | set(right))
    changes = {
        key: {"expected": left.get(key), "observed": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }
    return {
        "drift_detected": bool(changes),
        "expected_sha256": sha256_json(left),
        "observed_sha256": sha256_json(right),
        "changes": changes,
        "limitation": "provider_hidden_build_changes_are_not_observable",
    }


def response_metadata(interaction: Any) -> dict[str, Any]:
    """Extract documented Interactions response provenance without secrets."""
    result: dict[str, Any] = {}
    for source_name, target_name in (
        ("id", "response_id"),
        ("model", "response_model"),
        ("service_tier", "service_tier"),
        ("created", "response_created_at"),
        ("updated", "response_updated_at"),
        ("status", "response_status"),
    ):
        value = getattr(interaction, source_name, None)
        if value is not None:
            if isinstance(value, datetime):
                value = value.isoformat()
            elif hasattr(value, "value"):
                value = value.value
            result[target_name] = str(value)
    return result


def estimate_usage_cost_usd(model_id: str, usage: dict[str, Any] | None) -> float:
    """Estimate Standard-service cost from provider-reported token usage."""
    if model_id not in MODEL_PRICING_USD_PER_MILLION:
        raise ValueError(f"model pricing is unavailable: {model_id}")
    usage = dict(usage or {})

    def token_count(name: str) -> int:
        raw = usage.get(name, 0) or 0
        value = int(raw)
        if value < 0:
            raise ValueError(f"negative token usage: {name}")
        return value

    total_input = token_count("total_input_tokens")
    cached_input = min(total_input, token_count("total_cached_tokens"))
    uncached_input = total_input - cached_input
    output_and_thought = token_count("total_output_tokens") + token_count(
        "total_thought_tokens"
    )
    rates = MODEL_PRICING_USD_PER_MILLION[model_id]
    return (
        uncached_input * rates["input"]
        + cached_input * rates["cached_input"]
        + output_and_thought * rates["output_and_thought"]
    ) / 1_000_000.0


def configured_concurrency(*, smoke: bool = False) -> int:
    """Read the concurrency contract without ever reading or logging a secret."""
    raw = os.environ.get("GEMINI_MAX_CONCURRENCY", str(DEFAULT_CONCURRENCY)).strip()
    try:
        requested = int(raw)
    except ValueError as exc:
        raise ValueError("GEMINI_MAX_CONCURRENCY must be a positive integer") from exc
    if requested < 1:
        raise ValueError("GEMINI_MAX_CONCURRENCY must be a positive integer")
    # Phase B is intentionally serial even when a later-phase limit is higher.
    return 1 if smoke else requested


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_hash(
    *,
    model_id: str,
    model_metadata: dict[str, Any],
    prompt_hash: str,
    schema_hash: str,
    renderer_hash: str,
    evidence_schema_hash: str,
    image_sha256: str,
    candidate_mapping: dict[str, Any],
    serialized_metadata: str,
    generation_config: dict[str, Any],
    protocol_id: str | None = None,
    replicate_id: int | str | None = None,
    namespace: str | None = None,
) -> str:
    payload = {
        "model_id": model_id,
        "model_metadata": model_metadata,
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "renderer_hash": renderer_hash,
        "evidence_schema_hash": evidence_schema_hash,
        "image_sha256": image_sha256,
        "candidate_mapping_sha256": sha256_json(candidate_mapping),
        "serialized_metadata_sha256": hashlib.sha256(
            serialized_metadata.encode("utf-8")
        ).hexdigest(),
        "generation_config": generation_config,
    }
    # Omitting all three dimensions deliberately produces the exact v1 hash,
    # preserving every existing smoke-cache row. Explicit values isolate
    # ablations and stability replicates from normal inference.
    if protocol_id is not None:
        payload["protocol_id"] = str(protocol_id)
    if replicate_id is not None:
        payload["replicate_id"] = str(replicate_id)
    if namespace is not None:
        payload["namespace"] = str(namespace)
    return sha256_json(payload)


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str | None
    remaining_usd: float | None
    projected_spend_usd: float | None = None
    next_request_reserve_usd: float = 0.0


@dataclass(frozen=True)
class ClaimDecision:
    acquired: bool
    status: str
    reason: str | None
    owner_id: str | None
    stale_lease_recovered: bool = False
    recovery_count: int = 0


class BudgetGuard:
    def __init__(
        self,
        max_spend_usd: float | None,
        *,
        smoke_limit_per_model: int = 10,
        already_estimated_usd: float = 0.0,
        request_counts: dict[str, int] | None = None,
        er2_cost_cap_per_request_usd: float | None = None,
    ) -> None:
        if er2_cost_cap_per_request_usd is None:
            er2_raw = os.environ.get(
                "GEMINI_ER2_COST_CAP_PER_REQUEST_USD", ""
            ).strip()
            if er2_raw:
                er2_cost_cap_per_request_usd = float(er2_raw)
        if max_spend_usd is not None and max_spend_usd < 0:
            raise ValueError("GEMINI_MAX_SPEND_USD must be non-negative")
        if er2_cost_cap_per_request_usd is not None and er2_cost_cap_per_request_usd < 0:
            raise ValueError(
                "GEMINI_ER2_COST_CAP_PER_REQUEST_USD must be non-negative"
            )
        self.max_spend_usd = max_spend_usd
        self.smoke_limit_per_model = int(smoke_limit_per_model)
        self.estimated_spend_usd = float(already_estimated_usd)
        self.request_counts = dict(request_counts or {})
        self.er2_cost_cap_per_request_usd = er2_cost_cap_per_request_usd
        self.outstanding_request_reserve_usd = 0.0
        self._lock = threading.RLock()

    @classmethod
    def from_environment(cls) -> "BudgetGuard":
        raw = os.environ.get("GEMINI_MAX_SPEND_USD", "").strip()
        er2_raw = os.environ.get(
            "GEMINI_ER2_COST_CAP_PER_REQUEST_USD", ""
        ).strip()
        return cls(
            None if not raw else float(raw),
            er2_cost_cap_per_request_usd=None if not er2_raw else float(er2_raw),
        )

    def effective_reserve(self, model_id: str, requested_upper_bound_usd: float) -> float:
        requested = float(requested_upper_bound_usd)
        if requested < 0:
            raise ValueError("request upper bound must be non-negative")
        if (
            model_id == "gemini-robotics-er-2-preview"
            and self.er2_cost_cap_per_request_usd is not None
        ):
            # The environment cap is an extra conservative floor on the
            # reservation, not a replacement for public token-rate costing.
            return max(requested, float(self.er2_cost_cap_per_request_usd))
        return requested

    def check(
        self,
        model_id: str,
        next_request_upper_bound_usd: float,
        *,
        outstanding_request_reserve_usd: float | None = None,
    ) -> BudgetDecision:
        if model_id not in MODEL_IDS:
            return BudgetDecision(False, "model_not_allowlisted", None)
        reserve = self.effective_reserve(model_id, next_request_upper_bound_usd)
        with self._lock:
            if self.max_spend_usd is None:
                count = int(self.request_counts.get(model_id, 0))
                if count >= self.smoke_limit_per_model:
                    return BudgetDecision(
                        False,
                        "missing_budget_smoke_limit",
                        None,
                        None,
                        reserve,
                    )
                return BudgetDecision(True, None, None, None, reserve)
            outstanding = (
                self.outstanding_request_reserve_usd
                if outstanding_request_reserve_usd is None
                else float(outstanding_request_reserve_usd)
            )
            if outstanding < 0:
                raise ValueError("outstanding request reserve must be non-negative")
            remaining = self.max_spend_usd - self.estimated_spend_usd - outstanding
            projected = self.estimated_spend_usd + outstanding + reserve
            if projected > self.max_spend_usd:
                return BudgetDecision(
                    False,
                    "budget_would_be_exceeded",
                    remaining,
                    projected,
                    reserve,
                )
            return BudgetDecision(True, None, remaining, projected, reserve)

    def reserve(self, model_id: str, upper_bound_usd: float) -> float:
        with self._lock:
            decision = self.check(model_id, upper_bound_usd)
            if not decision.allowed:
                raise RuntimeError(decision.reason)
            reserve = decision.next_request_reserve_usd
            self.request_counts[model_id] = int(self.request_counts.get(model_id, 0)) + 1
            self.outstanding_request_reserve_usd += reserve
            return reserve

    def settle(
        self,
        model_id: str,
        reserved_usd: float,
        usage: dict[str, Any] | None,
    ) -> float:
        """Replace an upper-bound reservation with actual token-rate estimate."""
        actual = estimate_usage_cost_usd(model_id, usage)
        with self._lock:
            self.outstanding_request_reserve_usd = max(
                0.0, self.outstanding_request_reserve_usd - float(reserved_usd)
            )
            self.estimated_spend_usd += actual
        return actual

    def release(self, reserved_usd: float) -> None:
        with self._lock:
            self.outstanding_request_reserve_usd = max(
                0.0, self.outstanding_request_reserve_usd - float(reserved_usd)
            )

    def settle_reserved_upper_bound(self, reserved_usd: float) -> float:
        """Conservatively settle a provider response that omitted token usage."""
        reserved = float(reserved_usd)
        with self._lock:
            self.outstanding_request_reserve_usd = max(
                0.0, self.outstanding_request_reserve_usd - reserved
            )
            self.estimated_spend_usd += reserved
        return reserved


class GeminiCache:
    """SQLite cache containing request provenance but no headers, keys, or labels."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                request_hash TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL,
                frame_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                mapping_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                request_id TEXT,
                http_status INTEGER,
                latency_seconds REAL NOT NULL,
                usage_json TEXT NOT NULL,
                retry_count INTEGER NOT NULL,
                raw_output TEXT,
                parsed_output_json TEXT,
                valid INTEGER NOT NULL,
                abstain INTEGER NOT NULL,
                fallback_reason TEXT,
                model_metadata_json TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                renderer_hash TEXT NOT NULL,
                evidence_schema_hash TEXT NOT NULL,
                generation_config_json TEXT NOT NULL,
                estimated_charge_usd REAL NOT NULL,
                response_model TEXT,
                service_tier TEXT,
                response_created_at TEXT,
                response_updated_at TEXT,
                response_status TEXT,
                protocol_id TEXT,
                replicate_id TEXT,
                namespace TEXT
            )
            """
        )
        self._migrate_response_columns()
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS request_states (
                request_hash TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                protocol_id TEXT,
                replicate_id TEXT,
                namespace TEXT,
                status TEXT NOT NULL CHECK(status IN (
                    'PLANNED', 'IN_FLIGHT', 'SUCCEEDED', 'RETRYABLE_FAILED',
                    'PERMANENT_FAILED', 'ABSTAIN', 'TECHNICAL_FALLBACK'
                )),
                owner_id TEXT,
                reserved_usd REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT,
                recovery_count INTEGER NOT NULL DEFAULT 0,
                response_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_request_states_status
                ON request_states(status);
            CREATE TABLE IF NOT EXISTS request_leases (
                request_hash TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                expires_at REAL NOT NULL,
                recovery_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_request_leases_expiry
                ON request_leases(expires_at);
            CREATE TABLE IF NOT EXISTS request_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_hash TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                owner_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'PLANNED', 'IN_FLIGHT', 'SUCCEEDED', 'RETRYABLE_FAILED',
                    'PERMANENT_FAILED', 'ABSTAIN', 'TECHNICAL_FALLBACK'
                )),
                http_status INTEGER,
                error_type TEXT,
                error_message TEXT,
                latency_seconds REAL NOT NULL,
                response_id TEXT,
                response_model TEXT,
                service_tier TEXT,
                response_created_at TEXT,
                response_updated_at TEXT,
                response_status TEXT,
                usage_json TEXT NOT NULL,
                estimated_charge_usd REAL NOT NULL,
                UNIQUE(request_hash, attempt_number)
            );
            CREATE INDEX IF NOT EXISTS idx_request_attempts_hash
                ON request_attempts(request_hash, attempt_number);
            CREATE TABLE IF NOT EXISTS cache_imports (
                import_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_cache_path TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                imported_response_count INTEGER NOT NULL,
                valid_response_count INTEGER NOT NULL,
                invalid_response_count INTEGER NOT NULL,
                import_reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS response_import_provenance (
                request_hash TEXT PRIMARY KEY,
                source_cache_path TEXT NOT NULL,
                source_request_hash TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                import_reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS response_aliases (
                target_request_hash TEXT PRIMARY KEY,
                source_request_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reuse_reason TEXT NOT NULL,
                source_generation_config_json TEXT NOT NULL,
                target_generation_config_json TEXT NOT NULL,
                validation_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_response_aliases_source
                ON response_aliases(source_request_hash);
            """
        )
        self._auto_import_smoke_cache()
        self._backfill_request_states()

    def _auto_import_smoke_cache(self) -> None:
        if self.path.name != "gemini_cache.sqlite":
            return
        if self.connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]:
            return
        source = self.path.parent / "access_smoke_2x1" / "gemini_cache.sqlite"
        if source.exists() and source.resolve() != self.path.resolve():
            self.import_smoke_cache(source)

    def import_smoke_cache(self, source_path: str | Path) -> int:
        """Import the immutable 26-attempt Phase-B cache into an empty root cache."""
        source_path = Path(source_path).resolve()
        if source_path == self.path.resolve():
            raise ValueError("source and target cache paths must differ")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if self.connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]:
            raise RuntimeError("smoke cache import requires an empty responses table")

        source = sqlite3.connect(
            f"file:{source_path}?mode=ro", uri=True, timeout=30.0
        )
        source.row_factory = sqlite3.Row
        try:
            integrity = str(source.execute("PRAGMA quick_check").fetchone()[0])
            if integrity != "ok":
                raise RuntimeError(f"source smoke cache failed quick_check: {integrity}")
            rows = source.execute("SELECT * FROM responses ORDER BY rowid").fetchall()
            valid_count = sum(bool(row["valid"]) for row in rows)
            invalid_count = len(rows) - valid_count
            if (len(rows), valid_count, invalid_count) != (26, 20, 6):
                raise RuntimeError(
                    "source smoke cache must contain exactly 26 responses "
                    "(20 valid and 6 invalid)"
                )
            source_columns = {str(row["name"]) for row in source.execute(
                "PRAGMA table_info(responses)"
            )}
            target_columns = [
                str(row["name"])
                for row in self.connection.execute("PRAGMA table_info(responses)")
                if str(row["name"]) in source_columns
            ]
            placeholders = ",".join("?" for _ in target_columns)
            now = _utc_now()
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    cursor = self.connection.execute(
                        f"INSERT INTO responses ({','.join(target_columns)}) "
                        f"VALUES ({placeholders}) ON CONFLICT(request_hash) DO NOTHING",
                        tuple(row[name] for name in target_columns),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("smoke response import encountered a hash conflict")
                    self.connection.execute(
                        """
                        INSERT INTO response_import_provenance (
                            request_hash, source_cache_path, source_request_hash,
                            imported_at, import_reason
                        ) VALUES (?, ?, ?, ?, 'immutable_phase_b_smoke_cache_import')
                        """,
                        (row["request_hash"], str(source_path), row["request_hash"], now),
                    )
                self.connection.execute(
                    """
                    INSERT INTO cache_imports (
                        source_cache_path, imported_at, imported_response_count,
                        valid_response_count, invalid_response_count, import_reason
                    ) VALUES (?, ?, 26, 20, 6,
                              'immutable_phase_b_smoke_cache_import')
                    """,
                    (str(source_path), now),
                )
                self.connection.execute("COMMIT")
            except Exception:
                if self.connection.in_transaction:
                    self.connection.execute("ROLLBACK")
                raise
        finally:
            source.close()
        return 26

    def _migrate_response_columns(self) -> None:
        existing = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(responses)")
        }
        additions = {
            "response_model": "TEXT",
            "service_tier": "TEXT",
            "response_created_at": "TEXT",
            "response_updated_at": "TEXT",
            "response_status": "TEXT",
            "protocol_id": "TEXT",
            "replicate_id": "TEXT",
            "namespace": "TEXT",
        }
        for name, column_type in additions.items():
            if name not in existing:
                self.connection.execute(
                    f"ALTER TABLE responses ADD COLUMN {name} {column_type}"
                )

    @staticmethod
    def _response_lifecycle_status(row: sqlite3.Row | dict[str, Any]) -> str:
        if bool(row["valid"]):
            return "ABSTAIN" if bool(row["abstain"]) else "SUCCEEDED"
        keys = set(row.keys())
        fallback = str(row["fallback_reason"] or "") if "fallback_reason" in keys else ""
        if fallback.startswith("http_") and fallback not in {
            "http_429",
            "http_500",
            "http_502",
            "http_503",
            "http_504",
        }:
            return "PERMANENT_FAILED"
        return "TECHNICAL_FALLBACK"

    def _backfill_request_states(self) -> None:
        now = _utc_now()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            for row in self.connection.execute(
                "SELECT request_hash, sample_id, model_id, valid, abstain, fallback_reason, "
                "created_at, request_id, protocol_id, replicate_id, namespace "
                "FROM responses"
            ).fetchall():
                self.connection.execute(
                    """
                    INSERT INTO request_states (
                        request_hash, sample_id, model_id, protocol_id,
                        replicate_id, namespace, status, owner_id, reserved_usd,
                        created_at, updated_at, response_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0.0, ?, ?, ?)
                    ON CONFLICT(request_hash) DO NOTHING
                    """,
                    (
                        row["request_hash"],
                        row["sample_id"],
                        row["model_id"],
                        row["protocol_id"],
                        row["replicate_id"],
                        row["namespace"],
                        self._response_lifecycle_status(row),
                        row["created_at"] or now,
                        now,
                        row["request_id"],
                    ),
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.connection.close()

    def _legacy_missing_request_id_evidence(
        self, source_request_hash: str
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT p.source_cache_path, p.imported_at AS response_imported_at,
                   p.import_reason, i.imported_response_count,
                   i.valid_response_count, i.invalid_response_count,
                   i.imported_at AS cache_imported_at
            FROM response_import_provenance AS p
            JOIN cache_imports AS i
              ON i.source_cache_path = p.source_cache_path
             AND i.import_reason = p.import_reason
            WHERE p.request_hash = ?
              AND p.source_request_hash = ?
              AND p.import_reason = 'immutable_phase_b_smoke_cache_import'
              AND i.imported_response_count = 26
              AND i.valid_response_count = 20
              AND i.invalid_response_count = 6
            ORDER BY i.import_id DESC LIMIT 1
            """,
            (source_request_hash, source_request_hash),
        ).fetchone()
        return None if row is None else dict(row)

    def lower_ceiling_candidates(
        self, *, sample_id: str, model_id: str, target_max_output_tokens: int
    ) -> list[dict[str, Any]]:
        """Return paid valid sources; callers must still verify the source hash."""
        candidates = []
        for row in self.connection.execute(
            "SELECT * FROM responses WHERE sample_id = ? AND model_id = ? "
            "AND valid = 1 ORDER BY created_at DESC",
            (str(sample_id), str(model_id)),
        ).fetchall():
            item = dict(row)
            generation = json.loads(item["generation_config_json"])
            usage = json.loads(item["usage_json"] or "{}")
            source_max = int(generation.get("max_output_tokens", 0) or 0)
            paid_tokens = sum(
                int(usage.get(name, 0) or 0)
                for name in (
                    "total_input_tokens",
                    "total_output_tokens",
                    "total_thought_tokens",
                )
            )
            legacy_import_evidence = None
            if not item.get("request_id"):
                legacy_import_evidence = self._legacy_missing_request_id_evidence(
                    str(item["request_hash"])
                )
            if (
                0 < source_max < int(target_max_output_tokens)
                and paid_tokens > 0
                and (item.get("request_id") or legacy_import_evidence is not None)
            ):
                item["generation_config"] = generation
                item["model_metadata"] = json.loads(item["model_metadata_json"])
                item["legacy_import_evidence"] = legacy_import_evidence
                candidates.append(item)
        candidates.sort(
            key=lambda item: int(item["generation_config"]["max_output_tokens"]),
            reverse=True,
        )
        return candidates

    def add_response_alias(
        self,
        *,
        target_request_hash: str,
        source_request_hash: str,
        target_generation_config: dict[str, Any],
        reuse_reason: str = LOWER_CEILING_REUSE_REASON,
    ) -> bool:
        """Create an explicit, non-overwriting alias to a paid valid response."""
        if target_request_hash == source_request_hash:
            raise ValueError("an alias target must differ from its source")
        if reuse_reason != LOWER_CEILING_REUSE_REASON:
            raise ValueError("unsupported response reuse reason")
        source = self.connection.execute(
            "SELECT * FROM responses WHERE request_hash = ?", (source_request_hash,)
        ).fetchone()
        if source is None:
            raise KeyError(f"alias source response is absent: {source_request_hash}")
        if not bool(source["valid"]) or not source["parsed_output_json"]:
            raise ValueError("invalid or truncated responses cannot be aliased")
        usage = json.loads(source["usage_json"] or "{}")
        paid_tokens = sum(
            int(usage.get(name, 0) or 0)
            for name in (
                "total_input_tokens",
                "total_output_tokens",
                "total_thought_tokens",
            )
        )
        legacy_import_evidence = None
        if not source["request_id"]:
            legacy_import_evidence = self._legacy_missing_request_id_evidence(
                source_request_hash
            )
        if paid_tokens <= 0 or (
            not source["request_id"] and legacy_import_evidence is None
        ):
            raise ValueError("alias source lacks paid-response provenance")
        source_generation = json.loads(source["generation_config_json"])
        target_generation = dict(target_generation_config)
        source_max = int(source_generation.get("max_output_tokens", 0) or 0)
        target_max = int(target_generation.get("max_output_tokens", 0) or 0)
        if source_max <= 0 or target_max <= source_max:
            raise ValueError("alias requires a strictly higher target output-token ceiling")
        source_contract = {
            key: value
            for key, value in source_generation.items()
            if key != "max_output_tokens"
        }
        target_contract = {
            key: value
            for key, value in target_generation.items()
            if key != "max_output_tokens"
        }
        if source_contract != target_contract:
            raise ValueError(
                "source and target generation configs differ beyond token ceiling"
            )

        now = _utc_now()
        validation = {
            "source_valid": True,
            "source_paid_tokens": paid_tokens,
            "provider_request_id_present": bool(source["request_id"]),
            "provider_request_id_missing_legacy": not bool(source["request_id"]),
            "legacy_import_evidence": legacy_import_evidence,
            "source_max_output_tokens": source_max,
            "target_max_output_tokens": target_max,
            "only_output_token_ceiling_changed": True,
        }
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if self.connection.execute(
                "SELECT 1 FROM responses WHERE request_hash = ?",
                (target_request_hash,),
            ).fetchone():
                raise RuntimeError("alias target already has an immutable response")
            existing = self.connection.execute(
                "SELECT * FROM response_aliases WHERE target_request_hash = ?",
                (target_request_hash,),
            ).fetchone()
            if existing is not None:
                same = (
                    existing["source_request_hash"] == source_request_hash
                    and existing["reuse_reason"] == reuse_reason
                    and json.loads(existing["target_generation_config_json"])
                    == target_generation
                )
                if same:
                    self.connection.execute("COMMIT")
                    return False
                raise RuntimeError("an existing response alias cannot be overwritten")
            self.connection.execute(
                """
                INSERT INTO response_aliases (
                    target_request_hash, source_request_hash, created_at,
                    reuse_reason, source_generation_config_json,
                    target_generation_config_json, validation_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_request_hash,
                    source_request_hash,
                    now,
                    reuse_reason,
                    json.dumps(source_generation, sort_keys=True, allow_nan=False),
                    json.dumps(target_generation, sort_keys=True, allow_nan=False),
                    json.dumps(validation, sort_keys=True, allow_nan=False),
                ),
            )
            self.connection.execute("COMMIT")
            return True
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def get(self, digest: str) -> dict[str, Any] | None:
        alias = None
        row = self.connection.execute(
            "SELECT * FROM responses WHERE request_hash = ?", (digest,)
        ).fetchone()
        if row is None:
            alias = self.connection.execute(
                "SELECT * FROM response_aliases WHERE target_request_hash = ?",
                (digest,),
            ).fetchone()
            if alias is None:
                return None
            row = self.connection.execute(
                "SELECT * FROM responses WHERE request_hash = ?",
                (alias["source_request_hash"],),
            ).fetchone()
            if row is None or not bool(row["valid"]):
                raise RuntimeError("response alias points to a missing or invalid source")
        result = dict(row)
        for name in (
            "mapping_json",
            "usage_json",
            "parsed_output_json",
            "model_metadata_json",
            "generation_config_json",
        ):
            if result.get(name):
                result[name.removesuffix("_json")] = json.loads(result[name])
        result["valid"] = bool(result["valid"])
        result["abstain"] = bool(result["abstain"])
        result["cache_hit"] = True
        result["status"] = "success" if result["valid"] else "fallback"
        result["lifecycle_status"] = self._response_lifecycle_status(result)
        source_request_hash = str(row["request_hash"])
        if alias is not None:
            result["request_hash"] = digest
            result["source_request_hash"] = source_request_hash
            result["reuse_reason"] = str(alias["reuse_reason"])
            result["response_alias"] = True
            result["source_generation_config"] = result["generation_config"]
            result["target_generation_config"] = json.loads(
                alias["target_generation_config_json"]
            )
            result["alias_validation"] = json.loads(alias["validation_json"])
        provenance = self.connection.execute(
            "SELECT * FROM response_import_provenance WHERE request_hash = ?",
            (source_request_hash,),
        ).fetchone()
        if provenance is not None:
            result["import_provenance"] = dict(provenance)
        # The immutable response is the source of truth if a process crashed
        # after committing it but before clearing its lease/state.
        now = _utc_now()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                """
                INSERT INTO request_states (
                    request_hash, sample_id, model_id, protocol_id, replicate_id,
                    namespace, status, owner_id, reserved_usd, created_at,
                    updated_at, response_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0.0, ?, ?, ?)
                ON CONFLICT(request_hash) DO UPDATE SET
                    status = excluded.status,
                    owner_id = NULL,
                    reserved_usd = 0.0,
                    updated_at = excluded.updated_at,
                    response_id = excluded.response_id
                """,
                (
                    digest,
                    result["sample_id"],
                    result["model_id"],
                    result.get("protocol_id"),
                    result.get("replicate_id"),
                    result.get("namespace"),
                    result["lifecycle_status"],
                    result["created_at"],
                    now,
                    result.get("request_id"),
                ),
            )
            self.connection.execute(
                "DELETE FROM request_leases WHERE request_hash = ?", (digest,)
            )
            self.connection.execute("COMMIT")
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        return result

    def put(self, record: dict[str, Any]) -> bool:
        forbidden = {"api_key", "headers", "x-goog-api-key", "authorization", "correctness", "ground_truth"}
        if forbidden & {str(key).lower() for key in record}:
            raise ValueError("cache record contains a forbidden secret/evaluation field")
        values = {
            **record,
            "mapping_json": json.dumps(record["mapping"], sort_keys=True, allow_nan=False),
            "usage_json": json.dumps(record.get("usage", {}), sort_keys=True, allow_nan=False),
            "parsed_output_json": (
                None
                if record.get("parsed_output") is None
                else json.dumps(record["parsed_output"], sort_keys=True, allow_nan=False)
            ),
            "model_metadata_json": json.dumps(record.get("model_metadata", {}), sort_keys=True, allow_nan=False),
            "generation_config_json": json.dumps(record["generation_config"], sort_keys=True, allow_nan=False),
            "valid": int(bool(record["valid"])),
            "abstain": int(bool(record["abstain"])),
        }
        columns = (
            "request_hash", "sample_id", "frame_id", "model_id", "image_hash", "mapping_json",
            "created_at", "request_id", "http_status", "latency_seconds", "usage_json", "retry_count",
            "raw_output", "parsed_output_json", "valid", "abstain", "fallback_reason",
            "model_metadata_json", "prompt_hash", "schema_hash", "renderer_hash",
            "evidence_schema_hash", "generation_config_json", "estimated_charge_usd",
            "response_model", "service_tier", "response_created_at",
            "response_updated_at", "response_status", "protocol_id",
            "replicate_id", "namespace",
        )
        placeholders = ",".join("?" for _ in columns)
        cursor = self.connection.execute(
            f"INSERT INTO responses ({','.join(columns)}) VALUES ({placeholders}) "
            "ON CONFLICT(request_hash) DO NOTHING",
            tuple(values.get(name) for name in columns),
        )
        # Final responses are immutable. A resolved response can never be
        # overwritten by a retry, replicate, or resumed process.
        return cursor.rowcount == 1

    def plan_request(
        self,
        *,
        request_hash: str,
        sample_id: str,
        model_id: str,
        protocol_id: str | None = None,
        replicate_id: int | str | None = None,
        namespace: str | None = None,
    ) -> str:
        now = _utc_now()
        self.connection.execute(
            """
            INSERT INTO request_states (
                request_hash, sample_id, model_id, protocol_id, replicate_id,
                namespace, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'PLANNED', ?, ?)
            ON CONFLICT(request_hash) DO NOTHING
            """,
            (
                request_hash,
                str(sample_id),
                str(model_id),
                protocol_id,
                None if replicate_id is None else str(replicate_id),
                namespace,
                now,
                now,
            ),
        )
        return str(
            self.connection.execute(
                "SELECT status FROM request_states WHERE request_hash = ?",
                (request_hash,),
            ).fetchone()[0]
        )

    def request_state(self, digest: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM request_states WHERE request_hash = ?", (digest,)
        ).fetchone()
        return None if row is None else dict(row)

    def outstanding_reserve_usd(self) -> float:
        return float(
            self.connection.execute(
                "SELECT COALESCE(SUM(reserved_usd), 0.0) FROM request_states "
                "WHERE status = 'IN_FLIGHT'"
            ).fetchone()[0]
        )

    def _next_attempt_number(self, digest: str) -> int:
        return int(
            self.connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 "
                "FROM request_attempts WHERE request_hash = ?",
                (digest,),
            ).fetchone()[0]
        )

    def claim_request(
        self,
        digest: str,
        *,
        owner_id: str,
        lease_seconds: float,
        reserved_usd: float = 0.0,
        now_epoch: float | None = None,
        max_spend_usd: float | None = None,
        already_estimated_usd: float = 0.0,
    ) -> ClaimDecision:
        """Atomically acquire one request or report the current owner/state."""
        if not owner_id:
            raise ValueError("owner_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if reserved_usd < 0:
            raise ValueError("reserved_usd must be non-negative")
        now_epoch = time.time() if now_epoch is None else float(now_epoch)
        now = datetime.fromtimestamp(now_epoch, timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            response = self.connection.execute(
                "SELECT valid, abstain, fallback_reason FROM responses WHERE request_hash = ?",
                (digest,),
            ).fetchone()
            state = self.connection.execute(
                "SELECT * FROM request_states WHERE request_hash = ?", (digest,)
            ).fetchone()
            if response is not None:
                status = self._response_lifecycle_status(response)
                if state is not None and state["status"] != status:
                    self.connection.execute(
                        "UPDATE request_states SET status = ?, owner_id = NULL, "
                        "reserved_usd = 0.0, updated_at = ? WHERE request_hash = ?",
                        (status, now, digest),
                    )
                self.connection.execute(
                    "DELETE FROM request_leases WHERE request_hash = ?", (digest,)
                )
                self.connection.execute("COMMIT")
                return ClaimDecision(False, status, "cached_final", None)
            if state is None:
                self.connection.execute("ROLLBACK")
                raise KeyError(f"request is not planned: {digest}")
            if state["status"] in TERMINAL_REQUEST_STATUSES:
                self.connection.execute("COMMIT")
                return ClaimDecision(
                    False, str(state["status"]), "terminal_state", state["owner_id"]
                )

            lease = self.connection.execute(
                "SELECT * FROM request_leases WHERE request_hash = ?", (digest,)
            ).fetchone()
            recovered = False
            recovery_count = int(state["recovery_count"] or 0)
            if lease is not None and float(lease["expires_at"]) > now_epoch:
                self.connection.execute("COMMIT")
                return ClaimDecision(
                    False,
                    "IN_FLIGHT",
                    "active_lease",
                    str(lease["owner_id"]),
                    False,
                    int(lease["recovery_count"] or 0),
                )
            if lease is not None:
                recovered = True
                recovery_count = int(lease["recovery_count"] or 0) + 1
                latest_attempt = self.connection.execute(
                    "SELECT * FROM request_attempts WHERE request_hash = ? "
                    "ORDER BY attempt_number DESC LIMIT 1",
                    (digest,),
                ).fetchone()
                if latest_attempt is not None and str(latest_attempt["status"]) in TERMINAL_REQUEST_STATUSES:
                    # The provider result reached the durable attempt ledger,
                    # but the process crashed before the immutable response was
                    # committed.  Re-sending could duplicate a paid request, so
                    # fail closed to q-only and preserve the terminal receipt.
                    self.connection.execute(
                        "DELETE FROM request_leases WHERE request_hash = ?",
                        (digest,),
                    )
                    self.connection.execute(
                        "UPDATE request_states SET status = 'TECHNICAL_FALLBACK', "
                        "owner_id = NULL, reserved_usd = 0.0, updated_at = ?, "
                        "last_error = 'uncertain_provider_completion_after_crash', "
                        "response_id = ?, recovery_count = ? WHERE request_hash = ?",
                        (
                            now,
                            latest_attempt["response_id"],
                            recovery_count,
                            digest,
                        ),
                    )
                    self.connection.execute("COMMIT")
                    return ClaimDecision(
                        False,
                        "TECHNICAL_FALLBACK",
                        "uncertain_provider_completion_after_crash",
                        str(lease["owner_id"]),
                        True,
                        recovery_count,
                    )
                attempt_number = self._next_attempt_number(digest)
                self.connection.execute(
                    """
                    INSERT INTO request_attempts (
                        request_hash, attempt_number, owner_id, started_at,
                        completed_at, status, error_type, error_message,
                        latency_seconds, usage_json, estimated_charge_usd
                    ) VALUES (?, ?, ?, ?, ?, 'RETRYABLE_FAILED',
                              'StaleLeaseRecovery', 'stale_lease_recovered',
                              0.0, '{}', 0.0)
                    """,
                    (digest, attempt_number, str(lease["owner_id"]), now, now),
                )

            outstanding = float(
                self.connection.execute(
                    "SELECT COALESCE(SUM(reserved_usd), 0.0) "
                    "FROM request_states WHERE status = 'IN_FLIGHT'"
                ).fetchone()[0]
            )
            if state["status"] == "IN_FLIGHT":
                outstanding = max(0.0, outstanding - float(state["reserved_usd"] or 0.0))
            if (
                max_spend_usd is not None
                and float(already_estimated_usd) + outstanding + reserved_usd
                > float(max_spend_usd)
            ):
                if recovered:
                    self.connection.execute(
                        "DELETE FROM request_leases WHERE request_hash = ?", (digest,)
                    )
                    self.connection.execute(
                        "UPDATE request_states SET status = 'RETRYABLE_FAILED', "
                        "owner_id = NULL, reserved_usd = 0.0, updated_at = ?, "
                        "last_error = 'budget_would_be_exceeded', recovery_count = ? "
                        "WHERE request_hash = ?",
                        (now, recovery_count, digest),
                    )
                self.connection.execute("COMMIT")
                return ClaimDecision(
                    False,
                    str(state["status"]),
                    "budget_would_be_exceeded",
                    None,
                    recovered,
                    recovery_count,
                )

            expires_at = now_epoch + float(lease_seconds)
            self.connection.execute(
                """
                INSERT INTO request_leases (
                    request_hash, owner_id, acquired_at, expires_at, recovery_count
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(request_hash) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    acquired_at = excluded.acquired_at,
                    expires_at = excluded.expires_at,
                    recovery_count = excluded.recovery_count
                """,
                (digest, owner_id, now, expires_at, recovery_count),
            )
            self.connection.execute(
                """
                UPDATE request_states SET status = 'IN_FLIGHT', owner_id = ?,
                    reserved_usd = ?, updated_at = ?, last_error = NULL,
                    recovery_count = ? WHERE request_hash = ?
                """,
                (owner_id, reserved_usd, now, recovery_count, digest),
            )
            self.connection.execute("COMMIT")
            return ClaimDecision(
                True,
                "IN_FLIGHT",
                None,
                owner_id,
                recovered,
                recovery_count,
            )
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def renew_lease(
        self,
        digest: str,
        *,
        owner_id: str,
        lease_seconds: float,
        now_epoch: float | None = None,
    ) -> bool:
        now_epoch = time.time() if now_epoch is None else float(now_epoch)
        cursor = self.connection.execute(
            "UPDATE request_leases SET expires_at = ? "
            "WHERE request_hash = ? AND owner_id = ?",
            (now_epoch + float(lease_seconds), digest, owner_id),
        )
        return cursor.rowcount == 1

    def append_attempt(
        self,
        digest: str,
        *,
        owner_id: str,
        started_at: str,
        completed_at: str,
        status: str,
        latency_seconds: float,
        http_status: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        estimated_charge_usd: float = 0.0,
    ) -> int:
        if status not in REQUEST_STATUSES:
            raise ValueError(f"invalid request status: {status}")
        metadata = dict(metadata or {})
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            attempt_number = self._next_attempt_number(digest)
            self.connection.execute(
                """
                INSERT INTO request_attempts (
                    request_hash, attempt_number, owner_id, started_at,
                    completed_at, status, http_status, error_type,
                    error_message, latency_seconds, response_id, response_model,
                    service_tier, response_created_at, response_updated_at,
                    response_status, usage_json, estimated_charge_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    digest,
                    attempt_number,
                    owner_id,
                    started_at,
                    completed_at,
                    status,
                    http_status,
                    error_type,
                    error_message,
                    float(latency_seconds),
                    metadata.get("response_id"),
                    metadata.get("response_model"),
                    metadata.get("service_tier"),
                    metadata.get("response_created_at"),
                    metadata.get("response_updated_at"),
                    metadata.get("response_status"),
                    json.dumps(usage or {}, sort_keys=True, allow_nan=False),
                    float(estimated_charge_usd),
                ),
            )
            self.connection.execute("COMMIT")
            return attempt_number
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def attempts(self, digest: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM request_attempts WHERE request_hash = ? "
            "ORDER BY attempt_number",
            (digest,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["usage"] = json.loads(item["usage_json"])
            result.append(item)
        return result

    def retryable_attempt_count(self, digest: str) -> int:
        """Return durable provider retryable failures for one request hash.

        Stale-lease recovery rows are audit events rather than provider calls,
        so they neither consume the retry cap nor inflate API progress.
        """
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM request_attempts "
                "WHERE request_hash = ? AND status = 'RETRYABLE_FAILED' "
                "AND COALESCE(error_type, '') != 'StaleLeaseRecovery'",
                (digest,),
            ).fetchone()[0]
        )

    def attempt_statistics(self) -> dict[str, int]:
        """Return cumulative provider-attempt counts from the SQLite ledger."""
        row = self.connection.execute(
            """
            SELECT
                COALESCE(SUM(CASE
                    WHEN COALESCE(error_type, '') != 'StaleLeaseRecovery'
                    THEN 1 ELSE 0 END), 0) AS api_attempts,
                COALESCE(SUM(CASE
                    WHEN status = 'RETRYABLE_FAILED'
                         AND COALESCE(error_type, '') != 'StaleLeaseRecovery'
                    THEN 1 ELSE 0 END), 0) AS retryable_attempts
            FROM request_attempts
            """
        ).fetchone()
        return {
            "api_attempts": int(row["api_attempts"]),
            "retryable_attempts": int(row["retryable_attempts"]),
        }

    def complete_request(
        self,
        digest: str,
        *,
        owner_id: str,
        status: str,
        last_error: str | None = None,
        response_id: str | None = None,
    ) -> None:
        if status not in REQUEST_STATUSES - {"PLANNED", "IN_FLIGHT"}:
            raise ValueError(f"invalid completion status: {status}")
        now = _utc_now()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT owner_id FROM request_states WHERE request_hash = ?",
                (digest,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown request: {digest}")
            if row["owner_id"] not in (None, owner_id):
                raise RuntimeError("request lease is owned by another process")
            self.connection.execute(
                """
                UPDATE request_states SET status = ?, owner_id = NULL,
                    reserved_usd = 0.0, updated_at = ?, last_error = ?,
                    response_id = ? WHERE request_hash = ?
                """,
                (status, now, last_error, response_id, digest),
            )
            self.connection.execute(
                "DELETE FROM request_leases WHERE request_hash = ? AND owner_id = ?",
                (digest, owner_id),
            )
            self.connection.execute("COMMIT")
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def budget_state(self) -> tuple[float, dict[str, int]]:
        response_spend: dict[str, float] = {}
        rows = self.connection.execute(
            "SELECT request_hash, model_id, usage_json, estimated_charge_usd FROM responses"
        ).fetchall()
        for row in rows:
            usage = json.loads(row["usage_json"] or "{}")
            if usage and row["model_id"] in MODEL_PRICING_USD_PER_MILLION:
                charge = estimate_usage_cost_usd(str(row["model_id"]), usage)
            else:
                charge = float(row["estimated_charge_usd"] or 0.0)
            response_spend[str(row["request_hash"])] = charge
        attempt_spend = {
            str(request_hash): float(charge or 0.0)
            for request_hash, charge in self.connection.execute(
                "SELECT request_hash, SUM(estimated_charge_usd) "
                "FROM request_attempts GROUP BY request_hash"
            )
        }
        spent = sum(
            attempt_spend.get(request_hash, 0.0) or response_charge
            for request_hash, response_charge in response_spend.items()
        ) + sum(
            charge for request_hash, charge in attempt_spend.items()
            if request_hash not in response_spend
        )
        counts = {
            str(model): int(count)
            for model, count in self.connection.execute(
                "SELECT model_id, COUNT(*) FROM request_states GROUP BY model_id"
            )
        }
        return spent, counts


def _status_code(exc: Exception) -> int | None:
    for name in ("status_code", "code"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
        if callable(value):
            try:
                result = value()
                if isinstance(result, int):
                    return result
            except Exception:
                pass
    match = __import__("re").search(r"\b(400|401|403|404|429|500|502|503|504)\b", str(exc))
    return None if match is None else int(match.group(1))


def _retry_after_seconds(exc: Exception) -> float | None:
    """Return an HTTP Retry-After delay from an SDK exception, if available."""
    sources = (getattr(exc, "response", None), exc)
    for source in sources:
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
                retry_at = email.utils.parsedate_to_datetime(str(value))
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    return None


def _usage_dict(interaction: Any) -> dict[str, Any]:
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    if isinstance(usage, dict):
        return usage
    result = {}
    for name in (
        "total_input_tokens", "total_output_tokens", "total_thought_tokens", "total_tokens",
        "total_cached_tokens", "total_tool_use_tokens", "input_tokens_by_modality", "output_tokens_by_modality",
    ):
        value = getattr(usage, name, None)
        if value is not None:
            result[name] = value
    return result


class GoogleInteractionsRunner:
    def __init__(
        self,
        *,
        cache: GeminiCache,
        budget: BudgetGuard,
        api_key: str | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 5,
        max_cumulative_retryable_attempts: int | None = None,
        owner_id: str | None = None,
        lease_seconds: float = 1800.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self.cache = cache
        self.budget = budget
        self.sleep = sleep
        self.max_retries = int(max_retries)
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if max_cumulative_retryable_attempts is None:
            raw_cumulative_cap = os.environ.get(
                MAX_CUMULATIVE_RETRYABLE_ATTEMPTS_ENV,
                str(DEFAULT_MAX_CUMULATIVE_RETRYABLE_ATTEMPTS),
            )
            try:
                max_cumulative_retryable_attempts = int(raw_cumulative_cap)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max cumulative retryable attempts must be a positive integer"
                ) from exc
        self.max_cumulative_retryable_attempts = int(
            max_cumulative_retryable_attempts
        )
        if self.max_cumulative_retryable_attempts < 1:
            raise ValueError(
                "max cumulative retryable attempts must be a positive integer"
            )
        self.owner_id = owner_id or f"pid-{os.getpid()}:{uuid.uuid4().hex}"
        self.lease_seconds = float(lease_seconds)
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._client = client

    def _client_or_raise(self) -> Any:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )
        return self._client

    def run(
        self,
        *,
        sample_id: str,
        frame_id: str,
        model_id: str,
        image_path: str | Path,
        system_instruction: str,
        metadata_prompt: str,
        mapping: dict[str, Any],
        prompt_hash: str,
        schema_hash: str,
        renderer_hash: str,
        evidence_schema_hash: str,
        model_metadata: dict[str, Any] | None = None,
        request_upper_bound_usd: float = 0.10,
        image_resolution: str = DEFAULT_IMAGE_RESOLUTION,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        protocol_id: str | None = None,
        replicate_id: int | str | None = None,
        namespace: str | None = None,
        service_tier: str = "standard",
    ) -> dict[str, Any]:
        if model_id not in MODEL_IDS:
            raise ValueError(f"model substitution is forbidden: {model_id}")
        if service_tier != "standard":
            raise ValueError("formal transport requires service_tier='standard'")
        mapping_for_assert = dict(mapping)
        if "candidate_to_display" not in mapping_for_assert:
            # Compatibility for the Phase-B v1 caller, which persisted only
            # display_to_candidate. The asserted inverse is deterministic and
            # the original mapping remains unchanged for the legacy hash.
            display = mapping_for_assert.get("display_to_candidate", {})
            mapping_for_assert["candidate_to_display"] = {
                str(candidate_id): str(display_id)
                for display_id, candidate_id in display.items()
            }
        assert_display_mapping(mapping_for_assert)
        image_path = Path(image_path)
        image_bytes = image_path.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        generation_config = {
            "thinking_level": str(thinking_level),
            "max_output_tokens": int(max_output_tokens),
        }
        model_metadata = dict(model_metadata or {"requested_model_id": model_id})
        digest = request_hash(
            model_id=model_id,
            model_metadata=model_metadata,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            renderer_hash=renderer_hash,
            evidence_schema_hash=evidence_schema_hash,
            image_sha256=image_hash,
            candidate_mapping=mapping,
            serialized_metadata=metadata_prompt,
            generation_config={**generation_config, "image_resolution": image_resolution},
            protocol_id=protocol_id,
            replicate_id=replicate_id,
            namespace=namespace,
        )
        request_contract = {
            "model": model_id,
            "input": [
                {"type": "text", "text": metadata_prompt},
                {
                    "type": "image",
                    "data_sha256": image_hash,
                    "mime_type": "image/png",
                    "resolution": image_resolution,
                },
            ],
            "store": False,
            "background": False,
            "stream": False,
            "service_tier": service_tier,
            "system_instruction": system_instruction,
            "generation_config": generation_config,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": response_json_schema(),
            },
        }
        # Leak and contract validation precede even cache lookup/claim so a bad
        # payload cannot leave a stale lease or be hidden by an old response.
        canonical_request_json(request_contract)
        cached = self.cache.get(digest)
        if cached is not None:
            return cached
        if replicate_id is None:
            target_generation_config = {
                **generation_config,
                "image_resolution": image_resolution,
            }
            for source in self.cache.lower_ceiling_candidates(
                sample_id=str(sample_id),
                model_id=model_id,
                target_max_output_tokens=max_output_tokens,
            ):
                source_generation = source["generation_config"]
                reconstructed_source_hash = request_hash(
                    model_id=model_id,
                    model_metadata=model_metadata,
                    prompt_hash=prompt_hash,
                    schema_hash=schema_hash,
                    renderer_hash=renderer_hash,
                    evidence_schema_hash=evidence_schema_hash,
                    image_sha256=image_hash,
                    candidate_mapping=mapping,
                    serialized_metadata=metadata_prompt,
                    generation_config=source_generation,
                    protocol_id=source.get("protocol_id"),
                    replicate_id=source.get("replicate_id"),
                    namespace=source.get("namespace"),
                )
                if reconstructed_source_hash != source["request_hash"]:
                    continue
                self.cache.add_response_alias(
                    target_request_hash=digest,
                    source_request_hash=source["request_hash"],
                    target_generation_config=target_generation_config,
                )
                cached = self.cache.get(digest)
                if cached is not None:
                    return cached
        self.cache.plan_request(
            request_hash=digest,
            sample_id=str(sample_id),
            model_id=model_id,
            protocol_id=protocol_id,
            replicate_id=replicate_id,
            namespace=namespace,
        )
        if not self.api_key:
            return {
                "request_hash": digest,
                "status": "blocked",
                "lifecycle_status": "PLANNED",
                "valid": False,
                "abstain": False,
                "fallback_reason": "missing_api_key",
                "cache_hit": False,
            }
        decision = self.budget.check(
            model_id,
            float(request_upper_bound_usd),
            outstanding_request_reserve_usd=self.cache.outstanding_reserve_usd(),
        )
        if not decision.allowed:
            return {
                "request_hash": digest,
                "status": "blocked",
                "lifecycle_status": "PLANNED",
                "valid": False,
                "abstain": False,
                "fallback_reason": decision.reason,
                "cache_hit": False,
            }
        reserve_usd = decision.next_request_reserve_usd
        claim = self.cache.claim_request(
            digest,
            owner_id=self.owner_id,
            lease_seconds=self.lease_seconds,
            reserved_usd=reserve_usd,
            max_spend_usd=self.budget.max_spend_usd,
            already_estimated_usd=self.budget.estimated_spend_usd,
        )
        if not claim.acquired:
            raced_cache = self.cache.get(digest)
            if raced_cache is not None:
                return raced_cache
            return {
                "request_hash": digest,
                "status": (
                    "blocked"
                    if claim.reason == "budget_would_be_exceeded"
                    else "fallback"
                    if claim.status in TERMINAL_REQUEST_STATUSES
                    else "in_flight"
                ),
                "lifecycle_status": claim.status,
                "valid": False,
                "abstain": False,
                "fallback_reason": claim.reason,
                "cache_hit": False,
                "lease_owner": claim.owner_id,
                "stale_lease_recovered": claim.stale_lease_recovered,
            }
        cumulative_retryable_attempts = self.cache.retryable_attempt_count(digest)
        if (
            cumulative_retryable_attempts
            >= self.max_cumulative_retryable_attempts
        ):
            # The immutable fallback response makes the cap terminal across
            # every later resume.  FullRunOrchestrator maps any invalid
            # technical response to the frozen q-only candidate c0.
            fallback_reason = "cumulative_retryable_attempt_cap_reached"
            record = {
                "request_hash": digest,
                "sample_id": str(sample_id),
                "frame_id": str(frame_id),
                "model_id": model_id,
                "image_hash": image_hash,
                "mapping": mapping,
                "created_at": _utc_now(),
                "request_id": None,
                "http_status": None,
                "latency_seconds": 0.0,
                "usage": {},
                "retry_count": cumulative_retryable_attempts,
                "raw_output": None,
                "parsed_output": None,
                "valid": False,
                "abstain": False,
                "fallback_reason": fallback_reason,
                "model_metadata": model_metadata,
                "prompt_hash": prompt_hash,
                "schema_hash": schema_hash,
                "renderer_hash": renderer_hash,
                "evidence_schema_hash": evidence_schema_hash,
                "generation_config": {
                    **generation_config,
                    "image_resolution": image_resolution,
                },
                "estimated_charge_usd": 0.0,
                "response_model": None,
                "service_tier": None,
                "response_created_at": None,
                "response_updated_at": None,
                "response_status": None,
                "protocol_id": protocol_id,
                "replicate_id": (
                    None if replicate_id is None else str(replicate_id)
                ),
                "namespace": namespace,
            }
            safe_record = redact_sensitive(record, api_key=self.api_key)
            self.cache.put(safe_record)
            self.cache.complete_request(
                digest,
                owner_id=self.owner_id,
                status="TECHNICAL_FALLBACK",
                last_error=fallback_reason,
            )
            return {
                **safe_record,
                "status": "fallback",
                "lifecycle_status": "TECHNICAL_FALLBACK",
                "cache_hit": False,
                "api_attempted": False,
                "stale_lease_recovered": claim.stale_lease_recovered,
            }
        reserved_locally = self.budget.reserve(model_id, float(request_upper_bound_usd))
        if abs(reserved_locally - reserve_usd) > 1e-12:
            self.cache.complete_request(
                digest,
                owner_id=self.owner_id,
                status="RETRYABLE_FAILED",
                last_error="budget_reservation_mismatch",
            )
            self.budget.release(reserved_locally)
            raise RuntimeError("budget reservation changed after request claim")
        start = time.perf_counter()
        retry_count = 0
        raw_output = None
        parsed: GeminiRankingResponse | None = None
        fallback_reason = None
        http_status = None
        request_id = None
        usage: dict[str, Any] = {}
        actual_model_metadata = dict(model_metadata)
        provider_metadata: dict[str, Any] = {}
        estimated_charge_usd = 0.0
        lifecycle_status = "TECHNICAL_FALLBACK"
        provider_responded = False
        while True:
            if not self.cache.renew_lease(
                digest,
                owner_id=self.owner_id,
                lease_seconds=self.lease_seconds,
            ):
                self.budget.release(reserved_locally)
                raise RuntimeError("request lease was lost before API attempt")
            if (
                self.budget.max_spend_usd is not None
                and self.budget.estimated_spend_usd + self.cache.outstanding_reserve_usd()
                > self.budget.max_spend_usd
            ):
                fallback_reason = "budget_would_be_exceeded_before_retry"
                lifecycle_status = "RETRYABLE_FAILED"
                break
            attempt_started_at = _utc_now()
            attempt_start = time.perf_counter()
            try:
                client = self._client_or_raise()
                interaction = client.interactions.create(
                    model=model_id,
                    input=[
                        {"type": "text", "text": metadata_prompt},
                        {
                            "type": "image",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                            "mime_type": "image/png",
                            "resolution": image_resolution,
                        },
                    ],
                    stream=False,
                    store=False,
                    background=False,
                    service_tier=service_tier,
                    system_instruction=system_instruction,
                    generation_config=generation_config,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": response_json_schema(),
                    },
                )
                provider_responded = True
                raw_output = str(getattr(interaction, "output_text", "") or "")
                provider_metadata = response_metadata(interaction)
                request_id = provider_metadata.get("response_id")
                actual_model_metadata.update(provider_metadata)
                usage = _usage_dict(interaction)
                estimated_charge_usd = (
                    estimate_usage_cost_usd(model_id, usage)
                    if usage
                    else reserved_locally
                )
                try:
                    parsed = parse_ranking_response(raw_output)
                    lifecycle_status = (
                        "ABSTAIN" if parsed.decision == "abstain" else "SUCCEEDED"
                    )
                    fallback_reason = None
                except (json.JSONDecodeError, ValueError) as exc:
                    lifecycle_status = "TECHNICAL_FALLBACK"
                    fallback_reason = (
                        f"invalid_structured_response:{type(exc).__name__}"
                    )
                    self.cache.append_attempt(
                        digest,
                        owner_id=self.owner_id,
                        started_at=attempt_started_at,
                        completed_at=_utc_now(),
                        status=lifecycle_status,
                        latency_seconds=time.perf_counter() - attempt_start,
                        error_type=type(exc).__name__,
                        error_message=fallback_reason,
                        metadata=provider_metadata,
                        usage=usage,
                        estimated_charge_usd=estimated_charge_usd,
                    )
                    break
                self.cache.append_attempt(
                    digest,
                    owner_id=self.owner_id,
                    started_at=attempt_started_at,
                    completed_at=_utc_now(),
                    status=lifecycle_status,
                    latency_seconds=time.perf_counter() - attempt_start,
                    metadata=provider_metadata,
                    usage=usage,
                    estimated_charge_usd=estimated_charge_usd,
                )
                break
            except Exception as exc:
                http_status = _status_code(exc)
                message = str(exc).lower()
                if http_status in {401, 403}:
                    fallback_reason = f"http_{http_status}"
                    # Authentication failures are hard stops for the runner,
                    # but remain re-claimable after an out-of-band key rotation.
                    # The failed attempt is immutable audit evidence; it must
                    # never become a final cache response for this request hash.
                    lifecycle_status = "RETRYABLE_FAILED"
                    self.cache.append_attempt(
                        digest,
                        owner_id=self.owner_id,
                        started_at=attempt_started_at,
                        completed_at=_utc_now(),
                        status=lifecycle_status,
                        latency_seconds=time.perf_counter() - attempt_start,
                        http_status=http_status,
                        error_type=type(exc).__name__,
                        error_message=str(redact_sensitive(str(exc), api_key=self.api_key)),
                    )
                    break
                if "safety" in message or "blocked" in message:
                    fallback_reason = "safety_block"
                    lifecycle_status = "TECHNICAL_FALLBACK"
                    self.cache.append_attempt(
                        digest,
                        owner_id=self.owner_id,
                        started_at=attempt_started_at,
                        completed_at=_utc_now(),
                        status=lifecycle_status,
                        latency_seconds=time.perf_counter() - attempt_start,
                        http_status=http_status,
                        error_type=type(exc).__name__,
                        error_message=str(redact_sensitive(str(exc), api_key=self.api_key)),
                    )
                    break
                retryable = http_status in RETRYABLE_STATUS_CODES or any(
                    token in message for token in ("timeout", "connection reset", "temporarily unavailable")
                )
                if http_status in NON_RETRYABLE_STATUS_CODES or not retryable:
                    fallback_reason = f"http_{http_status or 'permanent_error'}"
                    lifecycle_status = "PERMANENT_FAILED"
                    self.cache.append_attempt(
                        digest,
                        owner_id=self.owner_id,
                        started_at=attempt_started_at,
                        completed_at=_utc_now(),
                        status=lifecycle_status,
                        latency_seconds=time.perf_counter() - attempt_start,
                        http_status=http_status,
                        error_type=type(exc).__name__,
                        error_message=str(redact_sensitive(str(exc), api_key=self.api_key)),
                    )
                    break
                self.cache.append_attempt(
                    digest,
                    owner_id=self.owner_id,
                    started_at=attempt_started_at,
                    completed_at=_utc_now(),
                    status="RETRYABLE_FAILED",
                    latency_seconds=time.perf_counter() - attempt_start,
                    http_status=http_status,
                    error_type=type(exc).__name__,
                    error_message=str(redact_sensitive(str(exc), api_key=self.api_key)),
                )
                cumulative_retryable_attempts = self.cache.retryable_attempt_count(
                    digest
                )
                if (
                    cumulative_retryable_attempts
                    >= self.max_cumulative_retryable_attempts
                ):
                    fallback_reason = (
                        "cumulative_retryable_attempt_cap_reached"
                    )
                    lifecycle_status = "TECHNICAL_FALLBACK"
                    break
                if retry_count >= self.max_retries:
                    fallback_reason = "retry_exhausted"
                    lifecycle_status = "RETRYABLE_FAILED"
                    break
                retry_after = _retry_after_seconds(exc)
                delay = (
                    retry_after
                    if retry_after is not None
                    else min(
                        30.0,
                        (2.0**retry_count)
                        + random.Random(f"{digest}:{retry_count}").random(),
                    )
                )
                retry_count += 1
                self.sleep(delay)
        latency = time.perf_counter() - start
        cumulative_retryable_attempts = self.cache.retryable_attempt_count(digest)
        valid = parsed is not None
        abstain = bool(valid and parsed.decision == "abstain")
        if usage:
            estimated_charge_usd = self.budget.settle(
                model_id, reserved_locally, usage
            )
        elif provider_responded:
            estimated_charge_usd = self.budget.settle_reserved_upper_bound(
                reserved_locally
            )
        else:
            self.budget.release(reserved_locally)
        record = {
            "request_hash": digest,
            "sample_id": str(sample_id),
            "frame_id": str(frame_id),
            "model_id": model_id,
            "image_hash": image_hash,
            "mapping": mapping,
            "created_at": _utc_now(),
            "request_id": None if request_id is None else str(request_id),
            "http_status": http_status,
            "latency_seconds": latency,
            "usage": usage,
            "retry_count": cumulative_retryable_attempts,
            "raw_output": raw_output,
            "parsed_output": None if parsed is None else parsed.model_dump(mode="json"),
            "valid": valid,
            "abstain": abstain,
            "fallback_reason": fallback_reason,
            "model_metadata": actual_model_metadata,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "renderer_hash": renderer_hash,
            "evidence_schema_hash": evidence_schema_hash,
            "generation_config": {**generation_config, "image_resolution": image_resolution},
            "estimated_charge_usd": estimated_charge_usd,
            "response_model": provider_metadata.get("response_model"),
            "service_tier": provider_metadata.get("service_tier"),
            "response_created_at": provider_metadata.get("response_created_at"),
            "response_updated_at": provider_metadata.get("response_updated_at"),
            "response_status": provider_metadata.get("response_status"),
            "protocol_id": protocol_id,
            "replicate_id": None if replicate_id is None else str(replicate_id),
            "namespace": namespace,
        }
        safe_record = redact_sensitive(record, api_key=self.api_key)
        # A per-invocation retry exhaustion remains resumable.  The cumulative
        # cap above instead becomes an immutable terminal technical fallback.
        if lifecycle_status != "RETRYABLE_FAILED":
            self.cache.put(safe_record)
        self.cache.complete_request(
            digest,
            owner_id=self.owner_id,
            status=lifecycle_status,
            last_error=fallback_reason,
            response_id=None if request_id is None else str(request_id),
        )
        return {
            **safe_record,
            "status": "success" if valid else "fallback",
            "lifecycle_status": lifecycle_status,
            "cache_hit": False,
            "stale_lease_recovered": claim.stale_lease_recovered,
        }
