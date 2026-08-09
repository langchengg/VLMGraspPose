"""SQLite-backed, resumable stage ledger for the complete G1/C1 experiment."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .artifacts import utc_now


SCHEMA = """
CREATE TABLE IF NOT EXISTS run_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    substage TEXT NOT NULL DEFAULT '',
    backend TEXT NOT NULL DEFAULT '',
    pool TEXT NOT NULL DEFAULT '',
    method TEXT NOT NULL DEFAULT '',
    feature_set TEXT NOT NULL DEFAULT '',
    seed INTEGER,
    status TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    command TEXT NOT NULL DEFAULT '',
    return_code INTEGER,
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    error_summary TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS run_ledger_lookup
ON run_ledger(stage, substage, backend, pool, method, feature_set, seed, status);
"""


def ledger_path(run_dir: str | Path) -> Path:
    return Path(run_dir).expanduser().resolve() / "run_ledger.sqlite"


def initialize_ledger(run_dir: str | Path) -> Path:
    path = ledger_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)
    return path


def completed(
    run_dir: str | Path,
    *,
    stage: str,
    substage: str = "",
    backend: str = "",
    pool: str = "",
    method: str = "",
    feature_set: str = "",
    seed: int | None = None,
) -> bool:
    path = initialize_ledger(run_dir)
    query = """
        SELECT 1 FROM run_ledger
        WHERE stage=? AND substage=? AND backend=? AND pool=? AND method=?
          AND feature_set=? AND seed IS ? AND status='COMPLETE'
        ORDER BY id DESC LIMIT 1
    """
    with sqlite3.connect(path) as connection:
        return connection.execute(
            query,
            (stage, substage, backend, pool, method, feature_set, seed),
        ).fetchone() is not None


@contextmanager
def ledger_stage(
    run_dir: str | Path,
    *,
    stage: str,
    substage: str = "",
    backend: str = "",
    pool: str = "",
    method: str = "",
    feature_set: str = "",
    seed: int | None = None,
    command: str = "",
) -> Iterator[dict[str, Any]]:
    """Record one stage attempt, including failure evidence, without hiding retries."""

    path = initialize_ledger(run_dir)
    started = utc_now()
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO run_ledger(
                stage,substage,backend,pool,method,feature_set,seed,status,start_time,command
            ) VALUES(?,?,?,?,?,?,?,'RUNNING',?,?)
            """,
            (stage, substage, backend, pool, method, feature_set, seed, started, command),
        )
        row_id = int(cursor.lastrowid)
        connection.commit()
    result: dict[str, Any] = {
        "artifact_path": "",
        "artifact_sha256": "",
        "return_code": 0,
    }
    try:
        yield result
    except BaseException as error:
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                UPDATE run_ledger SET status='FAILED',end_time=?,return_code=?,error_summary=?
                WHERE id=?
                """,
                (utc_now(), 1, f"{type(error).__name__}: {error}"[:4000], row_id),
            )
            connection.commit()
        raise
    else:
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                UPDATE run_ledger SET status='COMPLETE',end_time=?,return_code=?,
                    artifact_path=?,artifact_sha256=? WHERE id=?
                """,
                (
                    utc_now(),
                    int(result.get("return_code", 0)),
                    str(result.get("artifact_path", "")),
                    str(result.get("artifact_sha256", "")),
                    row_id,
                ),
            )
            connection.commit()


__all__ = ["completed", "initialize_ledger", "ledger_path", "ledger_stage"]
