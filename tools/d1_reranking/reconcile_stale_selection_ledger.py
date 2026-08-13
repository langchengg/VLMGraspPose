"""Reconcile one stale concurrent P9 selector attempt against the formal lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import export_ledger_commands, ledger_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stale-row-id", required=True, type=int)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _selection_record_from_lock(root: Path) -> dict[str, object]:
    lock_path = root / "08_lock/FORMAL_TEST_LOCK.json"
    digest_path = root / "08_lock/FORMAL_TEST_LOCK.sha256"
    if digest_path.read_text(encoding="ascii").strip() != sha256_file(lock_path):
        raise RuntimeError("D1 formal-lock detached digest differs")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    unsigned = dict(lock)
    recorded = unsigned.pop("self_sha256", None)
    if lock.get("status") != "LOCKED" or recorded != canonical_sha256(unsigned):
        raise RuntimeError("D1 formal-lock self hash differs")
    target = (root / "07_validation/selected_primary_ungated.json").resolve()
    records = [
        dict(record)
        for record in lock.get("inventory", {}).values()
        if isinstance(record, dict)
        and Path(str(record.get("path", ""))).resolve() == target
    ]
    unique = {(r.get("path"), r.get("sha256"), r.get("bytes")) for r in records}
    if len(unique) != 1:
        raise RuntimeError("D1 locked primary selection record is absent/ambiguous")
    record = records[0]
    if (
        sha256_file(target) != record.get("sha256")
        or target.stat().st_size != record.get("bytes")
    ):
        raise RuntimeError("D1 locked primary selection bytes differ")
    return record


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    if any((root / name).exists() for name in ("FINAL_RUN_LOCK.json", "COMPLETE")):
        raise PermissionError("D1 stale-ledger reconciliation is pre-final only")
    ledger = root / "run_ledger.sqlite"
    locked_selection = _selection_record_from_lock(root)
    with sqlite3.connect(ledger) as connection:
        connection.row_factory = sqlite3.Row
        stale = connection.execute(
            "SELECT * FROM stages WHERE id=?", (args.stale_row_id,)
        ).fetchone()
        if stale is None or stale["status"] != "RUNNING":
            raise RuntimeError("target D1 ledger row is not uniquely RUNNING")
        base_substage = str(stale["substage"]).split("__cmd_", 1)[0]
        if (
            stale["stage"] != "P9"
            or base_substage != "d1_select_primary_ungated"
            or Path(str(stale["artifact_path"])).resolve()
            != Path(str(locked_selection["path"])).resolve()
        ):
            raise RuntimeError("target D1 ledger row is not the stale P9 selector")
        completed = connection.execute(
            "SELECT * FROM stages WHERE stage='P9' AND status='COMPLETE' "
            "AND substage LIKE 'd1_select_primary_ungated__cmd_%' "
            "AND end_time>? ORDER BY end_time DESC",
            (stale["start_time"],),
        ).fetchall()
        valid = [
            row
            for row in completed
            if Path(str(row["artifact_path"])).resolve()
            == Path(str(locked_selection["path"])).resolve()
            and row["artifact_sha256"] == locked_selection["sha256"]
        ]
        if len(valid) != 1:
            raise RuntimeError("stale selector lacks one later locked COMPLETE attempt")
        connection.execute(
            "UPDATE stages SET status='FAILED', end_time=?, return_code=130, "
            "error_summary=? WHERE id=? AND status='RUNNING'",
            (
                _utc_now(),
                f"Interrupted concurrent attempt; superseded by locked COMPLETE row {valid[0]['id']}",
                args.stale_row_id,
            ),
        )
    export_ledger_commands(ledger)

    destination = root / "00_audit/STALE_SELECTION_LEDGER_RECONCILIATION.json"
    command = " ".join(map(str, sys.argv))
    with ledger_stage(
        ledger,
        stage="P9",
        substage="d1_select_primary_ungated",
        route="D1",
        evidence_track="T2_matched_common",
        pool="top5",
        method="R3_R5_R6",
        command=command,
    ) as state:
        audit: dict[str, object] = {
            "schema_version": 1,
            "status": "PASS",
            "stale_row_id": args.stale_row_id,
            "stale_disposition": "FAILED_INTERRUPTED_CONCURRENT_ATTEMPT",
            "superseding_complete_row_id": int(valid[0]["id"]),
            "locked_primary_selection": locked_selection,
            "formal_lock": {
                "path": str((root / "08_lock/FORMAL_TEST_LOCK.json").resolve()),
                "sha256": sha256_file(root / "08_lock/FORMAL_TEST_LOCK.json"),
                "bytes": (root / "08_lock/FORMAL_TEST_LOCK.json").stat().st_size,
            },
            "selection_recomputed": False,
            "formal_test_reexecuted": False,
        }
        audit["content_sha256"] = canonical_sha256(audit)
        atomic_json(destination, audit)
        state["artifact_path"] = str(destination.resolve())
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
