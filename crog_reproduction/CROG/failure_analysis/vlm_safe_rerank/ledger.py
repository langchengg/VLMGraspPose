from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PAIRWISE_REQUEST_HASH_VERSION = "pairwise_request_hash_v2"
LEGACY_PAIRWISE_REQUEST_HASH_VERSION = "pairwise_request_hash_v1"
TERMINAL_STATUSES = {"SUCCEEDED", "PERMANENT_FAILED", "SCHEMA_FAILED", "TECHNICAL_FALLBACK"}


def pairwise_request_hash(
    *,
    sample_id: str,
    baseline_candidate_id: str,
    challenger_candidate_id: str,
    model_id: str,
    protocol: str,
    prompt_hash: str,
    schema_hash: str,
    renderer_hash: str,
    board_sha256: str,
    evidence_hash: str,
    generation: Mapping[str, Any],
    perturbation_variant: str,
) -> str:
    payload = {
        "hash_version": PAIRWISE_REQUEST_HASH_VERSION,
        "sample_id": sample_id,
        "baseline_candidate_id": baseline_candidate_id,
        "challenger_candidate_id": challenger_candidate_id,
        "model_id": model_id,
        "protocol": protocol,
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "renderer_hash": renderer_hash,
        "board_sha256": board_sha256,
        "evidence_hash": evidence_hash,
        "generation": dict(generation),
        "perturbation_variant": perturbation_variant,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def legacy_pairwise_request_hash(
    *,
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
) -> str:
    """Reconstruct the pre-v2 key solely to migrate paid terminal cache rows."""

    payload = {
        "hash_version": LEGACY_PAIRWISE_REQUEST_HASH_VERSION,
        "sample_id": sample_id,
        "baseline_candidate_id": baseline_candidate_id,
        "challenger_candidate_id": challenger_candidate_id,
        "model_id": model_id,
        "protocol": protocol,
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "renderer_hash": renderer_hash,
        "evidence_hash": evidence_hash,
        "generation": dict(generation),
        "perturbation_variant": perturbation_variant,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CachedResponse:
    request_hash: str
    status: str
    raw_response: str | None
    parsed_json: dict[str, Any] | None
    api_request_id: str | None
    latency_seconds: float | None
    estimated_cost_usd: float
    requested_model: str | None
    response_model: str | None


class PairwiseLedger:
    """Exact-hash cache and durable attempt ledger; API secrets are never accepted."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS requests (
              request_hash TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              owner TEXT,
              lease_expires REAL,
              raw_response TEXT,
              parsed_json TEXT,
              api_request_id TEXT,
              latency_seconds REAL,
              estimated_cost_usd REAL NOT NULL DEFAULT 0,
              requested_model TEXT,
              response_model TEXT,
              updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
              attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
              request_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              latency_seconds REAL,
              estimated_cost_usd REAL NOT NULL DEFAULT 0,
              error_class TEXT,
              created_at REAL NOT NULL,
              FOREIGN KEY(request_hash) REFERENCES requests(request_hash)
            );
            CREATE INDEX IF NOT EXISTS attempts_request_hash ON attempts(request_hash);
            CREATE TABLE IF NOT EXISTS cache_migrations (
              old_request_hash TEXT PRIMARY KEY,
              new_request_hash TEXT NOT NULL UNIQUE,
              board_sha256 TEXT NOT NULL,
              reason TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            """
        )
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(requests)")
        }
        for column in ("requested_model", "response_model"):
            if column not in columns:
                self.connection.execute(f"ALTER TABLE requests ADD COLUMN {column} TEXT")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "PairwiseLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def acquire(self, request_hash: str, owner: str, *, lease_seconds: float = 300.0) -> str:
        now = time.time()
        with self.connection:
            row = self.connection.execute(
                "SELECT status, owner, lease_expires FROM requests WHERE request_hash=?", (request_hash,)
            ).fetchone()
            if row is not None and row["status"] in TERMINAL_STATUSES:
                return "CACHED"
            if row is not None and row["status"] == "IN_FLIGHT" and float(row["lease_expires"] or 0) > now and row["owner"] != owner:
                return "BUSY"
            self.connection.execute(
                """INSERT INTO requests(request_hash,status,owner,lease_expires,updated_at)
                   VALUES(?,?,?,?,?) ON CONFLICT(request_hash) DO UPDATE SET
                   status='IN_FLIGHT', owner=excluded.owner,
                   lease_expires=excluded.lease_expires, updated_at=excluded.updated_at""",
                (request_hash, "IN_FLIGHT", owner, now + lease_seconds, now),
            )
        return "ACQUIRED"

    def record_attempt(
        self,
        request_hash: str,
        status: str,
        *,
        latency_seconds: float | None = None,
        estimated_cost_usd: float = 0.0,
        error_class: str | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO attempts(request_hash,status,latency_seconds,estimated_cost_usd,error_class,created_at) VALUES(?,?,?,?,?,?)",
                (request_hash, status, latency_seconds, float(estimated_cost_usd), error_class, time.time()),
            )

    def release(self, request_hash: str, *, status: str = "RETRYABLE_FAILED") -> None:
        """Release an owned lease without manufacturing a terminal response."""

        with self.connection:
            cursor = self.connection.execute(
                "UPDATE requests SET status=?,owner=NULL,lease_expires=NULL,updated_at=? WHERE request_hash=? AND status='IN_FLIGHT'",
                (status, time.time(), request_hash),
            )
            if cursor.rowcount != 1:
                raise KeyError("in-flight request must exist before release")

    def migrate_terminal_hash(
        self,
        old_request_hash: str,
        new_request_hash: str,
        *,
        board_sha256: str,
    ) -> bool:
        """Atomically re-key a v1 terminal row after deterministically rebuilding its board.

        This preserves all paid attempts and avoids a second provider request.  The
        migration table makes the one-time compatibility assumption auditable.
        """

        if old_request_hash == new_request_hash:
            return False
        with self.connection:
            existing = self.connection.execute(
                "SELECT 1 FROM requests WHERE request_hash=?", (new_request_hash,)
            ).fetchone()
            if existing is not None:
                return False
            row = self.connection.execute(
                "SELECT * FROM requests WHERE request_hash=?", (old_request_hash,)
            ).fetchone()
            if (
                row is None
                or row["status"] not in TERMINAL_STATUSES
                or row["status"] == "TECHNICAL_FALLBACK"
            ):
                return False
            self.connection.execute(
                """INSERT INTO requests(
                     request_hash,status,owner,lease_expires,raw_response,parsed_json,
                     api_request_id,latency_seconds,estimated_cost_usd,requested_model,
                     response_model,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_request_hash, row["status"], None, None, row["raw_response"],
                    row["parsed_json"], row["api_request_id"], row["latency_seconds"],
                    row["estimated_cost_usd"], row["requested_model"],
                    row["response_model"], time.time(),
                ),
            )
            self.connection.execute(
                "UPDATE attempts SET request_hash=? WHERE request_hash=?",
                (new_request_hash, old_request_hash),
            )
            self.connection.execute("DELETE FROM requests WHERE request_hash=?", (old_request_hash,))
            self.connection.execute(
                "INSERT INTO cache_migrations(old_request_hash,new_request_hash,board_sha256,reason,created_at) VALUES(?,?,?,?,?)",
                (
                    old_request_hash, new_request_hash, board_sha256,
                    "v1_key_lacked_board_sha256; board deterministically rebuilt before replay",
                    time.time(),
                ),
            )
        return True

    def backfill_response_metadata(
        self,
        request_hash: str,
        *,
        requested_model: str | None,
        response_model: str | None,
        api_request_id: str | None = None,
    ) -> None:
        """Backfill metadata from immutable raw audit files created by v1."""

        with self.connection:
            self.connection.execute(
                """UPDATE requests SET
                     requested_model=COALESCE(requested_model,?),
                     response_model=COALESCE(response_model,?),
                     api_request_id=COALESCE(api_request_id,?)
                   WHERE request_hash=?""",
                (requested_model, response_model, api_request_id, request_hash),
            )

    def normalize_conservative_attempt_costs(
        self,
        *,
        er2_reserve_usd: float,
        flash_reserve_usd: float,
    ) -> int:
        """Idempotently repair pre-guard attempt accounting.

        ER2 has no independently verified price in this experiment, so every
        attempt is reserved at the configured cap.  Failed Flash attempts have
        no trustworthy usage object and are conservatively reserved as well.
        """

        with self.connection:
            er2 = self.connection.execute(
                """UPDATE attempts SET estimated_cost_usd=? WHERE request_hash IN
                   (SELECT request_hash FROM requests WHERE requested_model=?)
                   AND ABS(estimated_cost_usd-?)>1e-12""",
                (float(er2_reserve_usd), "gemini-robotics-er-2-preview", float(er2_reserve_usd)),
            ).rowcount
            flash = self.connection.execute(
                """UPDATE attempts SET estimated_cost_usd=? WHERE status!='SUCCEEDED'
                   AND request_hash IN
                   (SELECT request_hash FROM requests WHERE requested_model=?)
                   AND estimated_cost_usd<=0""",
                (float(flash_reserve_usd), "gemini-3.6-flash"),
            ).rowcount
            self.connection.execute(
                """UPDATE requests SET estimated_cost_usd=COALESCE(
                     (SELECT SUM(a.estimated_cost_usd) FROM attempts a
                      WHERE a.request_hash=requests.request_hash), estimated_cost_usd)"""
            )
        return int(er2 + flash)

    def finish(
        self,
        request_hash: str,
        status: str,
        *,
        raw_response: str | None = None,
        parsed_json: Mapping[str, Any] | None = None,
        api_request_id: str | None = None,
        latency_seconds: float | None = None,
        estimated_cost_usd: float = 0.0,
        requested_model: str | None = None,
        response_model: str | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("finish requires a terminal status")
        with self.connection:
            cursor = self.connection.execute(
                """UPDATE requests SET status=?,owner=NULL,lease_expires=NULL,raw_response=?,parsed_json=?,
                   api_request_id=?,latency_seconds=?,estimated_cost_usd=?,requested_model=?,
                   response_model=?,updated_at=? WHERE request_hash=?""",
                (status, raw_response, json.dumps(dict(parsed_json), sort_keys=True) if parsed_json is not None else None,
                 api_request_id, latency_seconds, float(estimated_cost_usd), requested_model,
                 response_model, time.time(), request_hash),
            )
            if cursor.rowcount != 1:
                raise KeyError("request must be acquired before finish")

    def cached(self, request_hash: str) -> CachedResponse | None:
        row = self.connection.execute("SELECT * FROM requests WHERE request_hash=?", (request_hash,)).fetchone()
        if row is None or row["status"] not in TERMINAL_STATUSES:
            return None
        return CachedResponse(
            request_hash, str(row["status"]), row["raw_response"],
            json.loads(row["parsed_json"]) if row["parsed_json"] else None,
            row["api_request_id"], row["latency_seconds"], float(row["estimated_cost_usd"]),
            row["requested_model"], row["response_model"],
        )

    def summary(self) -> dict[str, Any]:
        request_counts = {row[0]: row[1] for row in self.connection.execute("SELECT status,COUNT(*) FROM requests GROUP BY status")}
        attempt_counts = {row[0]: row[1] for row in self.connection.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status")}
        attempts, cost = self.connection.execute("SELECT COUNT(*),COALESCE(SUM(estimated_cost_usd),0) FROM attempts").fetchone()
        duplicate_success = self.connection.execute(
            "SELECT COUNT(*) FROM (SELECT request_hash FROM attempts WHERE status='SUCCEEDED' GROUP BY request_hash HAVING COUNT(*)>1)"
        ).fetchone()[0]
        cache_migrations = self.connection.execute("SELECT COUNT(*) FROM cache_migrations").fetchone()[0]
        return {
            "request_status_counts": request_counts,
            "attempt_status_counts": attempt_counts,
            "attempts": int(attempts),
            "attempt_estimated_cost_usd": float(cost),
            "duplicate_successful_request_hashes": int(duplicate_success),
            "cache_hash_migrations": int(cache_migrations),
        }
