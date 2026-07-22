"""Crash-resilient SQLite state store for long-running VGN experiments."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

from .failure_taxonomy import is_retryable, is_terminal


class ManifestCountMismatch(ValueError):
    """Raised when run state and the supplied manifest cannot be identical."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"sample rows must be mappings or dataclasses, got {type(value)!r}")


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class ExperimentStore:
    """Persist run state with atomic worker claims and resumable leases.

    Each worker/process should create its own instance. SQLite WAL permits
    readers during a write transaction, while ``BEGIN IMMEDIATE`` serializes
    the short claim transaction so two workers cannot claim the same sample.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        run_id: str,
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id).strip()
        if not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        self._connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_ms / 1_000.0,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA wal_autocheckpoint=1000")
        self._create_schema()

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Underlying connection for read-only diagnostics and tests."""

        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None  # type: ignore[assignment]

    def checkpoint(self, *, truncate: bool = False) -> tuple[int, int, int]:
        """Checkpoint the WAL and return SQLite's checkpoint result tuple."""

        mode = "TRUNCATE" if truncate else "PASSIVE"
        row = self._connection.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return tuple(int(value) for value in row)

    @contextmanager
    def _transaction(self, mode: str = "IMMEDIATE") -> Iterator[sqlite3.Connection]:
        self._connection.execute(f"BEGIN {mode}")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO schema_info(singleton, schema_version) VALUES (1, 1);

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                manifest_count INTEGER NOT NULL CHECK (manifest_count >= 0),
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS samples (
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                dataset_index INTEGER NOT NULL,
                scene_id TEXT NOT NULL,
                instruction TEXT NOT NULL DEFAULT '',
                view TEXT,
                cluster_id TEXT NOT NULL,
                manifest_row_json TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending'
                    CHECK (state IN ('pending', 'running', 'terminal', 'failed')),
                outcome_status TEXT,
                failure_reason TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                claimed_by TEXT,
                lease_expires_at REAL,
                started_at REAL,
                finished_at REAL,
                result_json TEXT,
                official_candidate_count INTEGER,
                target_candidate_count INTEGER,
                top1_vgn_quality REAL,
                processing_time_total REAL,
                PRIMARY KEY (run_id, sample_id),
                UNIQUE (run_id, dataset_index),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS samples_claim_queue
                ON samples(run_id, state, dataset_index);
            CREATE INDEX IF NOT EXISTS samples_lease
                ON samples(run_id, state, lease_expires_at);

            CREATE TABLE IF NOT EXISTS sample_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                worker_id TEXT,
                event_time REAL NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (run_id, sample_id)
                    REFERENCES samples(run_id, sample_id) ON DELETE CASCADE
            );
            """
        )
        version = self._connection.execute(
            "SELECT schema_version FROM schema_info WHERE singleton=1"
        ).fetchone()[0]
        if int(version) != self.SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported experiment-store schema {version}; expected {self.SCHEMA_VERSION}"
            )

    def initialize_run(self, metadata: Mapping[str, Any], manifest_count: int) -> None:
        """Create a run, or prove that an existing run has identical inputs."""

        if manifest_count < 0:
            raise ValueError("manifest_count must be non-negative")
        metadata_json = _canonical_json(dict(metadata))
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT manifest_count, metadata_json FROM runs WHERE run_id=?",
                (self.run_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO runs(run_id, manifest_count, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (self.run_id, int(manifest_count), metadata_json, now, now),
                )
                return
            if int(row["manifest_count"]) != int(manifest_count):
                raise ManifestCountMismatch(
                    f"run {self.run_id!r} expects {row['manifest_count']} manifest rows, "
                    f"received {manifest_count}"
                )
            if row["metadata_json"] != metadata_json:
                raise ValueError(
                    f"run {self.run_id!r} already exists with different reproducibility metadata"
                )

    def get_run(self) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (self.run_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def register_samples(
        self,
        rows: Iterable[Mapping[str, Any] | Any],
        *,
        expected_count: int | None = None,
    ) -> int:
        """Atomically register an exact manifest without silent partial inserts."""

        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_indices: set[int] = set()
        for raw in rows:
            row = dict(_mapping(raw))
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError("every manifest row must have a non-empty sample_id")
            if "dataset_index" not in row or row["dataset_index"] is None:
                raise ValueError(f"sample {sample_id!r} is missing dataset_index")
            dataset_index = int(row["dataset_index"])
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
            if dataset_index in seen_indices:
                raise ValueError(f"duplicate dataset_index in manifest: {dataset_index}")
            seen_ids.add(sample_id)
            seen_indices.add(dataset_index)
            row["sample_id"] = sample_id
            row["dataset_index"] = dataset_index
            normalized.append(row)

        run = self.get_run()
        if run is None:
            raise RuntimeError("initialize_run must be called before register_samples")
        required_count = int(run["manifest_count"])
        if expected_count is not None and int(expected_count) != required_count:
            raise ManifestCountMismatch(
                f"expected_count {expected_count} differs from run manifest_count {required_count}"
            )
        if len(normalized) != required_count:
            raise ManifestCountMismatch(
                f"manifest contains {len(normalized)} rows; run requires exactly {required_count}"
            )

        now = time.time()
        with self._transaction() as connection:
            existing_rows = connection.execute(
                """
                SELECT sample_id, dataset_index, scene_id, instruction, view, manifest_row_json
                FROM samples WHERE run_id=?
                """,
                (self.run_id,),
            ).fetchall()
            existing = {row["sample_id"]: row for row in existing_rows}
            if existing and set(existing) != seen_ids:
                missing = sorted(seen_ids - set(existing))[:5]
                extra = sorted(set(existing) - seen_ids)[:5]
                raise ManifestCountMismatch(
                    "registered sample identities differ from supplied manifest; "
                    f"missing_in_store={missing}, extra_in_store={extra}"
                )

            for row in normalized:
                sample_id = row["sample_id"]
                scene_id = str(row.get("scene_id", "")).strip()
                if not scene_id:
                    raise ValueError(f"sample {sample_id!r} is missing scene_id")
                instruction = str(row.get("instruction", row.get("text", "")))
                view_value = row.get("view")
                view = None if view_value is None else str(view_value)
                cluster_id = str(row.get("cluster_id", scene_id)).strip()
                if not cluster_id:
                    raise ValueError(f"sample {sample_id!r} has an empty cluster_id")
                serialized = _canonical_json(row)
                old = existing.get(sample_id)
                if old is not None:
                    identity = (
                        int(old["dataset_index"]),
                        old["scene_id"],
                        old["instruction"],
                        old["view"],
                        old["manifest_row_json"],
                    )
                    supplied = (
                        int(row["dataset_index"]),
                        scene_id,
                        instruction,
                        view,
                        serialized,
                    )
                    if identity != supplied:
                        raise ValueError(
                            f"sample {sample_id!r} differs from its registered manifest row"
                        )
                    continue
                connection.execute(
                    """
                    INSERT INTO samples(
                        run_id, sample_id, dataset_index, scene_id, instruction,
                        view, cluster_id, manifest_row_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.run_id,
                        sample_id,
                        row["dataset_index"],
                        scene_id,
                        instruction,
                        view,
                        cluster_id,
                        serialized,
                    ),
                )
            connection.execute(
                "UPDATE runs SET updated_at=? WHERE run_id=?", (now, self.run_id)
            )
        return len(normalized)

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 900.0,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Atomically claim the lowest-index pending sample."""

        worker = str(worker_id).strip()
        if not worker:
            raise ValueError("worker_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = time.time() if now is None else float(now)
        with self._transaction() as connection:
            self._recover_expired(connection, timestamp)
            row = connection.execute(
                """
                SELECT sample_id FROM samples
                WHERE run_id=? AND state='pending'
                ORDER BY dataset_index, sample_id
                LIMIT 1
                """,
                (self.run_id,),
            ).fetchone()
            if row is None:
                return None
            sample_id = row["sample_id"]
            cursor = connection.execute(
                """
                UPDATE samples
                SET state='running', attempts=attempts+1, claimed_by=?,
                    lease_expires_at=?, started_at=COALESCE(started_at, ?),
                    finished_at=NULL
                WHERE run_id=? AND sample_id=? AND state='pending'
                """,
                (
                    worker,
                    timestamp + float(lease_seconds),
                    timestamp,
                    self.run_id,
                    sample_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("atomic claim invariant violated")
            self._event(connection, sample_id, "claimed", worker, timestamp, {})
            claimed = connection.execute(
                "SELECT * FROM samples WHERE run_id=? AND sample_id=?",
                (self.run_id, sample_id),
            ).fetchone()
        return self._decode_sample_row(claimed)

    def claim_sample(
        self,
        sample_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 900.0,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Atomically claim one explicit pending sample.

        This is used by deterministic start/end/sample-id shards.  Returning
        ``None`` for a non-pending row makes resume idempotent without changing
        the row's recorded outcome.
        """

        worker = str(worker_id).strip()
        identifier = str(sample_id).strip()
        if not worker or not identifier:
            raise ValueError("sample_id and worker_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = time.time() if now is None else float(now)
        with self._transaction() as connection:
            self._recover_expired(connection, timestamp)
            cursor = connection.execute(
                """
                UPDATE samples
                SET state='running', attempts=attempts+1, claimed_by=?,
                    lease_expires_at=?, started_at=COALESCE(started_at, ?),
                    finished_at=NULL
                WHERE run_id=? AND sample_id=? AND state='pending'
                """,
                (
                    worker,
                    timestamp + float(lease_seconds),
                    timestamp,
                    self.run_id,
                    identifier,
                ),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT 1 FROM samples WHERE run_id=? AND sample_id=?",
                    (self.run_id, identifier),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown sample_id {identifier!r}")
                return None
            self._event(connection, identifier, "claimed", worker, timestamp, {})
            claimed = connection.execute(
                "SELECT * FROM samples WHERE run_id=? AND sample_id=?",
                (self.run_id, identifier),
            ).fetchone()
        return self._decode_sample_row(claimed)

    def heartbeat(
        self,
        sample_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 900.0,
        now: float | None = None,
    ) -> None:
        """Extend the lease held by a running worker."""

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = time.time() if now is None else float(now)
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE samples SET lease_expires_at=?
                WHERE run_id=? AND sample_id=? AND state='running' AND claimed_by=?
                """,
                (
                    timestamp + float(lease_seconds),
                    self.run_id,
                    str(sample_id),
                    str(worker_id),
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("heartbeat rejected: worker does not hold this sample")
            self._event(
                connection,
                str(sample_id),
                "heartbeat",
                str(worker_id),
                timestamp,
                {"lease_seconds": float(lease_seconds)},
            )

    def complete_sample(
        self,
        sample_id: str,
        status: str,
        *,
        result: Mapping[str, Any] | None = None,
        failure_reason: str = "",
        worker_id: str | None = None,
        now: float | None = None,
    ) -> None:
        """Persist a terminal scientific outcome or deterministic failure."""

        if not is_terminal(status):
            raise ValueError(f"status {status!r} is not terminal")
        payload = dict(result or {})
        timestamp = time.time() if now is None else float(now)
        self._finish(
            sample_id,
            state="terminal",
            status=status,
            payload=payload,
            failure_reason=failure_reason,
            worker_id=worker_id,
            timestamp=timestamp,
        )

    def fail_sample(
        self,
        sample_id: str,
        status: str,
        *,
        failure_reason: str,
        result: Mapping[str, Any] | None = None,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> None:
        """Persist a retryable failure without marking the sample terminal."""

        if not is_retryable(status):
            if is_terminal(status):
                self.complete_sample(
                    sample_id,
                    status,
                    result=result,
                    failure_reason=failure_reason,
                    worker_id=worker_id,
                    now=now,
                )
                return
            raise ValueError(f"status {status!r} is neither terminal nor retryable")
        timestamp = time.time() if now is None else float(now)
        self._finish(
            sample_id,
            state="failed",
            status=status,
            payload=dict(result or {}),
            failure_reason=failure_reason,
            worker_id=worker_id,
            timestamp=timestamp,
        )

    def _finish(
        self,
        sample_id: str,
        *,
        state: str,
        status: str,
        payload: Mapping[str, Any],
        failure_reason: str,
        worker_id: str | None,
        timestamp: float,
    ) -> None:
        official_count = _optional_int(payload.get("official_candidate_count"))
        target_count = _optional_int(payload.get("target_candidate_count"))
        quality = _optional_float(payload.get("top1_vgn_quality"))
        total_time = _optional_float(payload.get("processing_time_total"))
        if total_time is None:
            parts = [
                _optional_float(payload.get(name))
                for name in (
                    "processing_time_depth",
                    "processing_time_tsdf",
                    "processing_time_vgn",
                )
            ]
            finite_parts = [part for part in parts if part is not None]
            total_time = sum(finite_parts) if finite_parts else None
        with self._transaction() as connection:
            current = connection.execute(
                "SELECT state, claimed_by FROM samples WHERE run_id=? AND sample_id=?",
                (self.run_id, str(sample_id)),
            ).fetchone()
            if current is None:
                raise KeyError(f"unknown sample_id {sample_id!r}")
            if current["state"] not in {"pending", "running", "failed"}:
                raise RuntimeError(
                    f"sample {sample_id!r} is already terminal and cannot be overwritten"
                )
            if worker_id is not None and current["claimed_by"] != str(worker_id):
                raise RuntimeError("completion rejected: worker does not hold this sample")
            connection.execute(
                """
                UPDATE samples
                SET state=?, outcome_status=?, failure_reason=?, finished_at=?,
                    claimed_by=NULL, lease_expires_at=NULL, result_json=?,
                    official_candidate_count=?, target_candidate_count=?,
                    top1_vgn_quality=?, processing_time_total=?
                WHERE run_id=? AND sample_id=?
                """,
                (
                    state,
                    str(status),
                    str(failure_reason),
                    timestamp,
                    _canonical_json(dict(payload)),
                    official_count,
                    target_count,
                    quality,
                    total_time,
                    self.run_id,
                    str(sample_id),
                ),
            )
            self._event(
                connection,
                str(sample_id),
                "completed" if state == "terminal" else "failed",
                worker_id,
                timestamp,
                {"status": status, "failure_reason": failure_reason},
            )

    def recover_expired_claims(self, *, now: float | None = None) -> int:
        """Return samples with expired worker leases to the pending queue."""

        timestamp = time.time() if now is None else float(now)
        with self._transaction() as connection:
            return self._recover_expired(connection, timestamp)

    def _recover_expired(
        self, connection: sqlite3.Connection, timestamp: float
    ) -> int:
        rows = connection.execute(
            """
            SELECT sample_id, claimed_by FROM samples
            WHERE run_id=? AND state='running' AND lease_expires_at < ?
            ORDER BY dataset_index
            """,
            (self.run_id, timestamp),
        ).fetchall()
        for row in rows:
            connection.execute(
                """
                UPDATE samples
                SET state='pending', claimed_by=NULL, lease_expires_at=NULL,
                    outcome_status=NULL, failure_reason='worker_lost: lease expired',
                    finished_at=NULL
                WHERE run_id=? AND sample_id=? AND state='running'
                """,
                (self.run_id, row["sample_id"]),
            )
            self._event(
                connection,
                row["sample_id"],
                "lease_recovered",
                row["claimed_by"],
                timestamp,
                {"reason": "worker_lost"},
            )
        return len(rows)

    def requeue_retryable(self, sample_id: str | None = None) -> int:
        """Move retryable failed rows back to pending for an explicit retry."""

        clauses = ["run_id=?", "state='failed'"]
        parameters: list[Any] = [self.run_id]
        if sample_id is not None:
            clauses.append("sample_id=?")
            parameters.append(str(sample_id))
        with self._transaction() as connection:
            rows = connection.execute(
                f"SELECT sample_id, outcome_status FROM samples WHERE {' AND '.join(clauses)}",
                parameters,
            ).fetchall()
            retry_ids = [
                row["sample_id"]
                for row in rows
                if row["outcome_status"] and is_retryable(row["outcome_status"])
            ]
            for retry_id in retry_ids:
                connection.execute(
                    """
                    UPDATE samples SET state='pending', outcome_status=NULL,
                        claimed_by=NULL, lease_expires_at=NULL, finished_at=NULL
                    WHERE run_id=? AND sample_id=?
                    """,
                    (self.run_id, retry_id),
                )
            return len(retry_ids)

    def requeue_statuses(
        self,
        statuses: Iterable[str],
        *,
        sample_ids: Iterable[str] | None = None,
    ) -> int:
        """Explicitly reset selected recorded outcomes to pending.

        This is the only path used for ``--retry-model-outcomes`` and
        ``--force``.  Normal resume never modifies completed rows.
        """

        selected_statuses = {str(value) for value in statuses}
        if not selected_statuses:
            return 0
        selected_ids = None if sample_ids is None else {str(value) for value in sample_ids}
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT sample_id, outcome_status FROM samples WHERE run_id=?",
                (self.run_id,),
            ).fetchall()
            reset = [
                row["sample_id"]
                for row in rows
                if row["outcome_status"] in selected_statuses
                and (selected_ids is None or row["sample_id"] in selected_ids)
            ]
            for identifier in reset:
                connection.execute(
                    """
                    UPDATE samples
                    SET state='pending', outcome_status=NULL, failure_reason='',
                        claimed_by=NULL, lease_expires_at=NULL, started_at=NULL,
                        finished_at=NULL, result_json=NULL,
                        official_candidate_count=NULL, target_candidate_count=NULL,
                        top1_vgn_quality=NULL, processing_time_total=NULL
                    WHERE run_id=? AND sample_id=?
                    """,
                    (self.run_id, identifier),
                )
                self._event(
                    connection,
                    identifier,
                    "explicit_requeue",
                    None,
                    time.time(),
                    {"prior_status": next(
                        row["outcome_status"] for row in rows if row["sample_id"] == identifier
                    )},
                )
            return len(reset)

    def get_sample(self, sample_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM samples WHERE run_id=? AND sample_id=?",
            (self.run_id, str(sample_id)),
        ).fetchone()
        return None if row is None else self._decode_sample_row(row)

    def sample_rows(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM samples WHERE run_id=? ORDER BY dataset_index, sample_id",
            (self.run_id,),
        ).fetchall()
        return [self._decode_sample_row(row) for row in rows]

    def status_counts(self) -> dict[str, int]:
        rows = self._connection.execute(
            """
            SELECT COALESCE(outcome_status, state) AS status, COUNT(*) AS count
            FROM samples WHERE run_id=?
            GROUP BY COALESCE(outcome_status, state)
            """,
            (self.run_id,),
        ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def events(self, sample_id: str | None = None) -> list[dict[str, Any]]:
        if sample_id is None:
            rows = self._connection.execute(
                """
                SELECT * FROM sample_events WHERE run_id=?
                ORDER BY event_id
                """,
                (self.run_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT * FROM sample_events WHERE run_id=? AND sample_id=?
                ORDER BY event_id
                """,
                (self.run_id, str(sample_id)),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result

    def _event(
        self,
        connection: sqlite3.Connection,
        sample_id: str,
        event_type: str,
        worker_id: str | None,
        timestamp: float,
        details: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO sample_events(
                run_id, sample_id, event_type, worker_id, event_time, details_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                sample_id,
                event_type,
                worker_id,
                timestamp,
                _canonical_json(dict(details)),
            ),
        )

    @staticmethod
    def _decode_sample_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        manifest_row = json.loads(item.pop("manifest_row_json"))
        result_json = item.pop("result_json")
        result = {} if result_json is None else json.loads(result_json)
        # Preserve all original manifest/result fields while making authoritative
        # database state columns win on name collisions.
        merged = dict(manifest_row)
        merged.update(result)
        merged.update(item)
        merged["manifest_row"] = manifest_row
        merged["result"] = result
        merged["status"] = merged.get("outcome_status") or merged["state"]
        return merged
