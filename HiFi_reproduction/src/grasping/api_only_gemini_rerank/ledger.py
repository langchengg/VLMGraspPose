"""Durable exact-hash cache and attempt ledger; credentials are never accepted."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import assert_no_gt_payload


TERMINAL = {"SUCCEEDED", "PERMANENT_FAILED", "SCHEMA_FAILED", "TECHNICAL_FALLBACK"}


@dataclass(frozen=True)
class CachedRequest:
    request_hash: str
    status: str
    raw_response: str | None
    parsed_response: dict[str, Any] | None
    metadata: dict[str, Any]
    requested_model: str
    response_model: str | None
    request_id: str | None
    latency_seconds: float | None
    estimated_cost_usd: float | None


class ApiLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS requests(
              request_hash TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              owner TEXT,
              lease_expires REAL,
              requested_model TEXT NOT NULL,
              response_model TEXT,
              endpoint_type TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              raw_response TEXT,
              parsed_response_json TEXT,
              request_id TEXT,
              latency_seconds REAL,
              estimated_cost_usd REAL,
              cache_hit_count INTEGER NOT NULL DEFAULT 0,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts(
              attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
              request_hash TEXT NOT NULL,
              attempt_number INTEGER NOT NULL,
              status TEXT NOT NULL,
              latency_seconds REAL,
              error_class TEXT,
              error_message TEXT,
              usage_json TEXT,
              estimated_cost_usd REAL,
              response_model TEXT,
              request_id TEXT,
              created_at REAL NOT NULL,
              UNIQUE(request_hash, attempt_number),
              FOREIGN KEY(request_hash) REFERENCES requests(request_hash)
            );
            CREATE INDEX IF NOT EXISTS attempts_by_request ON attempts(request_hash);
            CREATE TABLE IF NOT EXISTS request_aliases(
              source_request_hash TEXT PRIMARY KEY,
              target_request_hash TEXT NOT NULL,
              reason TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempt_reservations(
              reservation_id TEXT PRIMARY KEY,
              request_hash TEXT NOT NULL,
              attempt_number INTEGER NOT NULL,
              owner TEXT NOT NULL,
              reserved_cost_usd REAL NOT NULL,
              created_at REAL NOT NULL,
              UNIQUE(request_hash, attempt_number),
              FOREIGN KEY(request_hash) REFERENCES requests(request_hash)
            );
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "ApiLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def acquire(
        self, request_hash: str, owner: str, *, requested_model: str,
        endpoint_type: str, metadata: Mapping[str, Any], lease_seconds: float = 600,
    ) -> str:
        assert_no_gt_payload(metadata)
        now = time.time()
        serialized = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.db:
            row = self.db.execute(
                "SELECT status,owner,lease_expires FROM requests WHERE request_hash=?", (request_hash,)
            ).fetchone()
            if row is not None and row["status"] in TERMINAL:
                return "CACHED"
            if row is not None and row["status"] == "IN_FLIGHT" and float(row["lease_expires"] or 0) > now and row["owner"] != owner:
                return "BUSY"
            if row is not None and row["owner"] != owner and float(row["lease_expires"] or 0) <= now:
                reservation = self.db.execute(
                    "SELECT reservation_id,attempt_number,reserved_cost_usd FROM attempt_reservations WHERE request_hash=?",
                    (request_hash,),
                ).fetchone()
                if reservation is not None:
                    # The previous owner crossed the local send boundary but
                    # did not commit a response.  Treat the hash as terminal
                    # unknown-remote instead of risking a duplicate paid call.
                    self.db.execute(
                        """INSERT INTO attempts(request_hash,attempt_number,status,latency_seconds,error_class,error_message,
                           usage_json,estimated_cost_usd,response_model,request_id,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (request_hash, int(reservation["attempt_number"]), "PERMANENT_FAILED", None,
                         "StaleReservedAttempt", "stale owner with an unresolved provider reservation; no resend",
                         "{}", float(reservation["reserved_cost_usd"]), None, None, now),
                    )
                    self.db.execute("DELETE FROM attempt_reservations WHERE request_hash=?", (request_hash,))
                    self.db.execute(
                        """UPDATE requests SET status='TECHNICAL_FALLBACK',owner=NULL,lease_expires=NULL,
                           estimated_cost_usd=?,updated_at=? WHERE request_hash=?""",
                        (float(reservation["reserved_cost_usd"]), now, request_hash),
                    )
                    return "CACHED"
            self.db.execute(
                """INSERT INTO requests(request_hash,status,owner,lease_expires,requested_model,endpoint_type,metadata_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(request_hash) DO UPDATE SET
                   status='IN_FLIGHT', owner=excluded.owner, lease_expires=excluded.lease_expires,
                   updated_at=excluded.updated_at""",
                (request_hash, "IN_FLIGHT", owner, now + lease_seconds, requested_model, endpoint_type, serialized, now, now),
            )
        return "ACQUIRED"

    def renew(self, request_hash: str, owner: str, *, lease_seconds: float = 600) -> None:
        with self.db:
            cursor = self.db.execute(
                "UPDATE requests SET lease_expires=?,updated_at=? WHERE request_hash=? AND status='IN_FLIGHT' AND owner=?",
                (time.time()+lease_seconds, time.time(), request_hash, owner),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("request lease ownership was lost")

    def reserve_attempt(
        self, request_hash: str, owner: str, *, reserve_cost_usd: float,
        max_cost_usd: float, max_attempts: int,
    ) -> str:
        reservation_id = uuid.uuid4().hex
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT status,owner,lease_expires FROM requests WHERE request_hash=?", (request_hash,)
            ).fetchone()
            if row is None or row["status"] != "IN_FLIGHT" or row["owner"] != owner or float(row["lease_expires"] or 0) <= time.time():
                raise RuntimeError("request lease ownership was lost before budget reservation")
            counts = self.db.execute(
                "SELECT (SELECT COUNT(*) FROM attempts)+(SELECT COUNT(*) FROM attempt_reservations) n, "
                "(SELECT COALESCE(SUM(estimated_cost_usd),0) FROM attempts)+(SELECT COALESCE(SUM(reserved_cost_usd),0) FROM attempt_reservations) cost"
            ).fetchone()
            if int(counts["n"])+1 > int(max_attempts):
                raise RuntimeError("next attempt would exceed MAX_PROVIDER_REQUESTS")
            if float(counts["cost"])+float(reserve_cost_usd) > float(max_cost_usd)+1e-12:
                raise RuntimeError("next attempt reserve would exceed MAX_API_COST_USD")
            attempt_number = self.next_attempt_number(request_hash)
            self.db.execute(
                "INSERT INTO attempt_reservations(reservation_id,request_hash,attempt_number,owner,reserved_cost_usd,created_at) VALUES(?,?,?,?,?,?)",
                (reservation_id, request_hash, attempt_number, owner, float(reserve_cost_usd), time.time()),
            )
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        return reservation_id

    def cached(self, request_hash: str) -> CachedRequest | None:
        alias = self.db.execute(
            "SELECT target_request_hash FROM request_aliases WHERE source_request_hash=?", (request_hash,)
        ).fetchone()
        if alias is not None:
            return self.cached(str(alias["target_request_hash"]))
        row = self.db.execute("SELECT * FROM requests WHERE request_hash=?", (request_hash,)).fetchone()
        if row is None or row["status"] not in TERMINAL:
            return None
        return CachedRequest(
            request_hash=request_hash, status=str(row["status"]), raw_response=row["raw_response"],
            parsed_response=json.loads(row["parsed_response_json"]) if row["parsed_response_json"] else None,
            metadata=json.loads(row["metadata_json"]), requested_model=str(row["requested_model"]),
            response_model=row["response_model"], request_id=row["request_id"],
            latency_seconds=row["latency_seconds"], estimated_cost_usd=row["estimated_cost_usd"],
        )

    def set_alias(self, source_request_hash: str, target_request_hash: str, *, reason: str) -> None:
        if source_request_hash == target_request_hash:
            raise ValueError("request alias cannot point to itself")
        with self.db:
            self.db.execute(
                "INSERT INTO request_aliases(source_request_hash,target_request_hash,reason,created_at) VALUES(?,?,?,?) "
                "ON CONFLICT(source_request_hash) DO UPDATE SET target_request_hash=excluded.target_request_hash,reason=excluded.reason",
                (source_request_hash, target_request_hash, reason, time.time()),
            )

    def mark_cache_hit(self, request_hash: str) -> None:
        alias = self.db.execute(
            "SELECT target_request_hash FROM request_aliases WHERE source_request_hash=?", (request_hash,)
        ).fetchone()
        if alias is not None:
            request_hash = str(alias["target_request_hash"])
        with self.db:
            self.db.execute(
                "UPDATE requests SET cache_hit_count=cache_hit_count+1,updated_at=? WHERE request_hash=?",
                (time.time(), request_hash),
            )

    def next_attempt_number(self, request_hash: str) -> int:
        row = self.db.execute("SELECT COALESCE(MAX(attempt_number),0)+1 n FROM attempts WHERE request_hash=?", (request_hash,)).fetchone()
        return int(row["n"])

    def attempt_count(self, request_hash: str, *, status: str | None = None) -> int:
        if status is None:
            row = self.db.execute(
                "SELECT COUNT(*) n FROM attempts WHERE request_hash=?", (request_hash,)
            ).fetchone()
        else:
            row = self.db.execute(
                "SELECT COUNT(*) n FROM attempts WHERE request_hash=? AND status=?",
                (request_hash, status),
            ).fetchone()
        return int(row["n"])

    def record_attempt(
        self, request_hash: str, *, status: str, latency_seconds: float | None,
        error: BaseException | None = None, usage: Mapping[str, Any] | None = None,
        estimated_cost_usd: float | None = None, response_model: str | None = None,
        request_id: str | None = None, reservation_id: str | None = None,
    ) -> None:
        error_message = None if error is None else str(error)[:1000]
        with self.db:
            attempt_number = self.next_attempt_number(request_hash)
            if reservation_id is not None:
                reservation = self.db.execute(
                    "SELECT attempt_number,request_hash FROM attempt_reservations WHERE reservation_id=?", (reservation_id,)
                ).fetchone()
                if reservation is None or reservation["request_hash"] != request_hash:
                    raise RuntimeError("attempt budget reservation is missing or mismatched")
                attempt_number = int(reservation["attempt_number"])
                self.db.execute("DELETE FROM attempt_reservations WHERE reservation_id=?", (reservation_id,))
            self.db.execute(
                """INSERT INTO attempts(request_hash,attempt_number,status,latency_seconds,error_class,error_message,usage_json,estimated_cost_usd,response_model,request_id,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (request_hash, attempt_number, status, latency_seconds,
                 None if error is None else type(error).__name__, error_message,
                 json.dumps(dict(usage or {}), sort_keys=True), estimated_cost_usd,
                 response_model, request_id, time.time()),
            )

    def finalize(
        self, request_hash: str, *, status: str, raw_response: str | None,
        parsed_response: Mapping[str, Any] | None, latency_seconds: float,
        estimated_cost_usd: float | None, response_model: str | None,
        request_id: str | None, owner: str | None = None,
    ) -> None:
        if status not in TERMINAL:
            raise ValueError("final status is not terminal")
        with self.db:
            where = "request_hash=?" if owner is None else "request_hash=? AND owner=?"
            values = (status, raw_response,
                      None if parsed_response is None else json.dumps(dict(parsed_response), sort_keys=True),
                      latency_seconds, estimated_cost_usd, response_model, request_id, time.time(), request_hash)
            if owner is not None:
                values = (*values, owner)
            cursor = self.db.execute(
                """UPDATE requests SET status=?,owner=NULL,lease_expires=NULL,raw_response=?,
                   parsed_response_json=?,latency_seconds=?,estimated_cost_usd=?,response_model=?,request_id=?,updated_at=?
                   WHERE """ + where,
                values,
            )
            if cursor.rowcount != 1:
                raise RuntimeError("request finalization lost its owner fence")

    def record_attempt_and_finalize(
        self, request_hash: str, *, attempt_status: str, final_status: str,
        latency_seconds: float | None, total_latency_seconds: float,
        error: BaseException | None = None, usage: Mapping[str, Any] | None = None,
        attempt_cost_usd: float | None = None, total_cost_usd: float | None = None,
        raw_response: str | None, parsed_response: Mapping[str, Any] | None,
        response_model: str | None, request_id: str | None,
        reservation_id: str, owner: str,
    ) -> None:
        """Atomically commit the final provider attempt and terminal cache row."""
        if final_status not in TERMINAL:
            raise ValueError("final status is not terminal")
        error_message = None if error is None else str(error)[:1000]
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            request = self.db.execute(
                "SELECT status,owner FROM requests WHERE request_hash=?", (request_hash,)
            ).fetchone()
            if request is None or request["status"] != "IN_FLIGHT" or request["owner"] != owner:
                raise RuntimeError("request finalization lost its owner fence")
            reservation = self.db.execute(
                "SELECT attempt_number,request_hash FROM attempt_reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if reservation is None or reservation["request_hash"] != request_hash:
                raise RuntimeError("attempt budget reservation is missing or mismatched")
            self.db.execute(
                """INSERT INTO attempts(request_hash,attempt_number,status,latency_seconds,error_class,error_message,
                   usage_json,estimated_cost_usd,response_model,request_id,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (request_hash, int(reservation["attempt_number"]), attempt_status, latency_seconds,
                 None if error is None else type(error).__name__, error_message,
                 json.dumps(dict(usage or {}), sort_keys=True), attempt_cost_usd,
                 response_model, request_id, now),
            )
            self.db.execute("DELETE FROM attempt_reservations WHERE reservation_id=?", (reservation_id,))
            cursor = self.db.execute(
                """UPDATE requests SET status=?,owner=NULL,lease_expires=NULL,raw_response=?,
                   parsed_response_json=?,latency_seconds=?,estimated_cost_usd=?,response_model=?,request_id=?,updated_at=?
                   WHERE request_hash=? AND owner=?""",
                (final_status, raw_response,
                 None if parsed_response is None else json.dumps(dict(parsed_response), sort_keys=True),
                 total_latency_seconds, total_cost_usd, response_model, request_id, now,
                 request_hash, owner),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("request finalization lost its owner fence")
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def release(self, request_hash: str, owner: str | None = None) -> None:
        with self.db:
            where = "request_hash=? AND status='IN_FLIGHT'" if owner is None else "request_hash=? AND status='IN_FLIGHT' AND owner=?"
            values = (time.time(), request_hash) if owner is None else (time.time(), request_hash, owner)
            self.db.execute(
                "UPDATE requests SET status='RETRYABLE_FAILED',owner=NULL,lease_expires=NULL,updated_at=? WHERE " + where,
                values,
            )

    def summary(self) -> dict[str, Any]:
        status = {str(r["status"]): int(r["n"]) for r in self.db.execute("SELECT status,COUNT(*) n FROM requests GROUP BY status")}
        attempts = self.db.execute(
            "SELECT COUNT(*) n,COALESCE(SUM(estimated_cost_usd),0) cost FROM attempts"
        ).fetchone()
        duplicate = self.db.execute(
            "SELECT COUNT(*) n FROM (SELECT request_hash FROM attempts WHERE status='SUCCEEDED' GROUP BY request_hash HAVING COUNT(*)>1)"
        ).fetchone()
        costs = self.db.execute(
            """SELECT
               COALESCE(SUM(CASE WHEN r.requested_model='gemini-3.6-flash' AND a.status='SUCCEEDED' THEN a.estimated_cost_usd ELSE 0 END),0) flash_priced,
               COALESCE(SUM(CASE WHEN r.requested_model='gemini-robotics-er-2-preview' THEN a.estimated_cost_usd ELSE 0 END),0) er2_reserve,
               COALESCE(SUM(CASE WHEN r.requested_model='gemini-3.6-flash' AND a.status!='SUCCEEDED' THEN a.estimated_cost_usd ELSE 0 END),0) failure_reserve
               FROM attempts a JOIN requests r USING(request_hash)"""
        ).fetchone()
        cache = self.db.execute("SELECT COALESCE(SUM(cache_hit_count),0) n FROM requests").fetchone()
        return {
            "logical_requests": sum(status.values()), "status_counts": status,
            "provider_attempts": int(attempts["n"]), "attempt_estimated_cost_usd": float(attempts["cost"]),
            "cache_hits": int(cache["n"]), "duplicate_successful_request_hashes": int(duplicate["n"]),
            "cost_accounting": {
                "flash_token_priced_estimate_usd": float(costs["flash_priced"]),
                "er2_conservative_budget_reserve_usd": float(costs["er2_reserve"]),
                "failure_unknown_charge_budget_reserve_usd": float(costs["failure_reserve"]),
                "provider_invoice_verified": False,
                "legacy_mixed_total_usd": float(attempts["cost"]),
            },
        }
