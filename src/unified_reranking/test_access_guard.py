"""Fail-closed candidate-level test-label access guard."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .lock import verify_formal_test_lock


TEST_LABEL_FILENAMES = {
    "canonical_candidates.parquet",
    "per_sample_metrics.parquet",
    "test_labels.parquet",
}


def append_access_log(run_dir: str | Path, event: dict[str, Any]) -> None:
    destination = Path(run_dir) / "09_formal_test" / "test_access.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    line = json.dumps(payload, sort_keys=True) + "\n"
    with destination.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            event_id = payload.get("event_id")
            if isinstance(event_id, str) and event_id:
                stream.seek(0)
                for existing_line in stream:
                    try:
                        existing = json.loads(existing_line)
                    except json.JSONDecodeError:
                        continue
                    if existing.get("event_id") == event_id:
                        return
            stream.seek(0, os.SEEK_END)
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def assert_test_labels_unlocked(run_dir: str | Path) -> None:
    root = Path(run_dir)
    lock = root / "08_lock" / "FORMAL_TEST_LOCK.json"
    try:
        verify_formal_test_lock(root)
    except PermissionError as error:
        append_access_log(
            root,
            {"event": "candidate_test_label_access_denied", "reason": str(error)},
        )
        raise
    execution = root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    if not execution.is_file():
        append_access_log(root, {"event": "candidate_test_label_access_denied", "reason": "execution_not_claimed"})
        raise PermissionError("formal Test execution must be claimed before candidate labels are read")
    execution_payload = json.loads(execution.read_text(encoding="utf-8"))
    # Only the active, already-claimed transaction may open the formal label
    # source. COMPLETE is deliberately rejected so the exactly-once reader
    # cannot be reused after finalization.
    if execution_payload.get("execution_count") != 1 or execution_payload.get("status") != "RUNNING":
        append_access_log(root, {"event": "candidate_test_label_access_denied", "reason": "invalid_execution_claim"})
        raise PermissionError("invalid formal Test execution claim")
    append_access_log(
        root,
        {
            "event": "candidate_test_label_access_authorized",
            "lock_path": str(lock.resolve()),
            "lock_file_sha256": sha256_file(lock),
        },
    )
