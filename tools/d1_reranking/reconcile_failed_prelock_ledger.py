"""Record a later audited completion for a superseded failed prelock stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--failed-row-id", required=True, type=int)
    parser.add_argument("--evidence-artifact", required=True, type=Path)
    parser.add_argument(
        "--authority", required=True, choices=("formal_lock", "later_complete_ledger")
    )
    return parser.parse_args()


def _record(path: Path) -> dict[str, object]:
    value = path.expanduser().resolve()
    if value.is_symlink() or not value.is_file():
        raise RuntimeError(f"D1 reconciliation evidence is absent/not regular: {value}")
    return {"path": str(value), "sha256": sha256_file(value), "bytes": value.stat().st_size}


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    if any((root / name).exists() for name in ("FINAL_RUN_LOCK.json", "COMPLETE")):
        raise PermissionError("D1 failed-stage reconciliation is pre-final only")
    evidence = _record(args.evidence_artifact)
    ledger = root / "run_ledger.sqlite"
    with sqlite3.connect(ledger) as connection:
        connection.row_factory = sqlite3.Row
        failed = connection.execute(
            "SELECT * FROM stages WHERE id=?", (args.failed_row_id,)
        ).fetchone()
        if failed is None or failed["status"] != "FAILED":
            raise RuntimeError("target D1 ledger row is not FAILED")
        if args.authority == "formal_lock":
            lock = json.loads(
                (root / "08_lock/FORMAL_TEST_LOCK.json").read_text(encoding="utf-8")
            )
            matches = [
                record
                for record in lock.get("inventory", {}).values()
                if isinstance(record, dict)
                and record.get("path") == evidence["path"]
                and record.get("sha256") == evidence["sha256"]
                and record.get("bytes") == evidence["bytes"]
            ]
            if len(matches) != 1:
                raise RuntimeError("reconciliation evidence is not uniquely formal-locked")
            authority_record: object = _record(root / "08_lock/FORMAL_TEST_LOCK.json")
        else:
            rows = connection.execute(
                "SELECT id,artifact_path,artifact_sha256 FROM stages "
                "WHERE id>? AND status='COMPLETE' AND artifact_path=? "
                "AND artifact_sha256=?",
                (failed["id"], evidence["path"], evidence["sha256"]),
            ).fetchall()
            if len(rows) != 1:
                raise RuntimeError("reconciliation evidence lacks one later COMPLETE row")
            authority_record = {"ledger_complete_row_id": int(rows[0]["id"])}
        base_substage = str(failed["substage"]).split("__cmd_", 1)[0]
        identity = {
            field: failed[field]
            for field in (
                "stage",
                "route",
                "evidence_track",
                "pool",
                "method",
                "feature_set",
                "loss",
                "encoder",
                "seed",
            )
        }

    destination = (
        root
        / "00_audit"
        / f"FAILED_LEDGER_RECONCILIATION_{args.failed_row_id}.json"
    )
    command = " ".join(map(str, sys.argv))
    with ledger_stage(
        ledger,
        stage=str(identity["stage"]),
        substage=base_substage,
        route=str(identity["route"]),
        evidence_track=str(identity["evidence_track"]),
        pool=str(identity["pool"]),
        method=str(identity["method"]),
        feature_set=str(identity["feature_set"]),
        loss=str(identity["loss"]),
        encoder=str(identity["encoder"]),
        seed=int(identity["seed"]),
        command=command,
    ) as state:
        audit: dict[str, object] = {
            "schema_version": 1,
            "status": "PASS",
            "failed_row_id": args.failed_row_id,
            "failed_error_summary": str(failed["error_summary"]),
            "reconciliation_authority": args.authority,
            "authority_record": authority_record,
            "evidence_artifact": evidence,
            "stage_reexecuted": False,
            "formal_test_reexecuted": False,
        }
        audit["content_sha256"] = canonical_sha256(audit)
        atomic_json(destination, audit)
        state["artifact_path"] = str(destination.resolve())
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
