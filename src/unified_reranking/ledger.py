"""SQLite stage ledger for resumable experiment execution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .hashing import atomic_text


LEDGER_COLUMNS = (
    "stage",
    "substage",
    "route",
    "evidence_track",
    "pool",
    "method",
    "feature_set",
    "loss",
    "encoder",
    "seed",
    "status",
    "start_time",
    "end_time",
    "command",
    "return_code",
    "artifact_path",
    "artifact_sha256",
    "error_summary",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_ledger(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                substage TEXT NOT NULL DEFAULT '',
                route TEXT NOT NULL DEFAULT '',
                evidence_track TEXT NOT NULL DEFAULT '',
                pool TEXT NOT NULL DEFAULT '',
                method TEXT NOT NULL DEFAULT '',
                feature_set TEXT NOT NULL DEFAULT '',
                loss TEXT NOT NULL DEFAULT '',
                encoder TEXT NOT NULL DEFAULT '',
                seed INTEGER NOT NULL DEFAULT -1,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                command TEXT NOT NULL DEFAULT '',
                return_code INTEGER,
                artifact_path TEXT NOT NULL DEFAULT '',
                artifact_sha256 TEXT NOT NULL DEFAULT '',
                error_summary TEXT NOT NULL DEFAULT '',
                UNIQUE(stage, substage, route, evidence_track, pool, method,
                       feature_set, loss, encoder, seed)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS stages_status_idx ON stages(status)"
        )
        # Early development ledgers used NULL for a stage without a learned-model
        # seed.  SQLite treats NULL values as distinct in UNIQUE constraints, so
        # resuming could append a duplicate stage.  Canonicalise that sentinel.
        connection.execute("UPDATE stages SET seed=-1 WHERE seed IS NULL")
    return destination


def render_ledger_commands(path: str | Path) -> str:
    """Render the current ledger as deterministic JSONL without mutation."""

    ledger_path = Path(path).resolve()
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise ValueError(f"expected a regular ledger database: {ledger_path}")
    with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, stage, substage, route, evidence_track, pool, method,
                   feature_set, loss, encoder, seed, status, start_time,
                   end_time, command, return_code, artifact_path,
                   artifact_sha256, error_summary
            FROM stages
            ORDER BY stage, substage, route, evidence_track, pool, method,
                     feature_set, loss, encoder, seed, id
            """
        ).fetchall()
    return "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )


def export_ledger_commands(path: str | Path) -> Path:
    """Export the current SQLite ledger as deterministic JSONL.

    The SQLite table is authoritative.  A separate advisory lock prevents two
    concurrently completing stages from replacing ``commands.log`` with an
    older snapshot.  Each export queries only after acquiring that lock.
    """

    ledger_path = initialize_ledger(path).resolve()
    destination = ledger_path.parent / "commands.log"
    lock_path = ledger_path.parent / ".commands.log.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        payload = render_ledger_commands(ledger_path)
        atomic_text(destination, payload)
        with destination.open("rb") as stream:
            os.fsync(stream.fileno())
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return destination


@contextmanager
def ledger_stage(
    path: str | Path,
    *,
    stage: str,
    substage: str = "",
    route: str = "",
    evidence_track: str = "",
    pool: str = "",
    method: str = "",
    feature_set: str = "",
    loss: str = "",
    encoder: str = "",
    seed: int | None = None,
    command: str = "",
) -> Iterator[dict[str, str]]:
    initialize_ledger(path)
    started = utc_now()
    command_digest = hashlib.sha256(command.encode("utf-8")).hexdigest()[:16]
    bound_substage = (
        f"{substage}__cmd_{command_digest}"
        if command and substage
        else f"cmd_{command_digest}"
        if command
        else substage
    )
    identity = (
        stage,
        bound_substage,
        route,
        evidence_track,
        pool,
        method,
        feature_set,
        loss,
        encoder,
        -1 if seed is None else int(seed),
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO stages (
                stage, substage, route, evidence_track, pool, method, feature_set,
                loss, encoder, seed, status, start_time, command
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'RUNNING', ?, ?)
            ON CONFLICT(stage, substage, route, evidence_track, pool, method,
                        feature_set, loss, encoder, seed)
            DO UPDATE SET status='RUNNING', start_time=excluded.start_time,
                          end_time=NULL, command=excluded.command,
                          return_code=NULL, error_summary=''
            """,
            (*identity, started, command),
        )
    state: dict[str, str] = {}
    try:
        yield state
    except Exception as error:
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                UPDATE stages SET status='FAILED', end_time=?, return_code=1,
                                  error_summary=?
                WHERE stage=? AND substage=? AND route=? AND evidence_track=?
                  AND pool=? AND method=? AND feature_set=? AND loss=?
                  AND encoder=? AND seed=?
                """,
                (utc_now(), f"{type(error).__name__}: {error}", *identity),
            )
        export_ledger_commands(path)
        raise
    else:
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                UPDATE stages SET status='COMPLETE', end_time=?, return_code=0,
                                  artifact_path=?, artifact_sha256=?
                WHERE stage=? AND substage=? AND route=? AND evidence_track=?
                  AND pool=? AND method=? AND feature_set=? AND loss=?
                  AND encoder=? AND seed=?
                """,
                (
                    utc_now(),
                    state.get("artifact_path", ""),
                    state.get("artifact_sha256", ""),
                    *identity,
                ),
            )
        export_ledger_commands(path)
