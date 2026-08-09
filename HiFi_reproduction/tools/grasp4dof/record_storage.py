#!/usr/bin/env python3
"""Record compact-run storage use without deleting or mutating source runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


LIMIT_BYTES = 12 * 1024**3


def directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    if not (run / ".RUN_ACTIVE").is_file() or not (run / ".DO_NOT_PRUNE").is_file():
        raise ValueError("active/protection markers are missing")
    total = directory_bytes(run)
    budget_path = run / "storage_budget.json"
    budget = {
        "schema_version": 1,
        "budget_bytes": LIMIT_BYTES,
        "budget_gib": 12.0,
        "scope": "new run excluding original dataset and pinned third-party source",
        "compact_policy": {
            "rgb_depth": "references only",
            "probabilities": "reuse verified compressed NPZ by reference",
            "candidate_tables": "Parquet ZSTD",
            "heatmaps": "selected gallery only",
            "checkpoints": "best only after training completes",
        },
    }
    if not budget_path.exists():
        temporary = budget_path.with_name(f".{budget_path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, budget_path)
    elif json.loads(budget_path.read_text()) != budget:
        raise ValueError("storage budget artifact drift")
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "run_bytes": total,
        "run_gib": total / 1024**3,
        "budget_bytes": LIMIT_BYTES,
        "remaining_bytes": LIMIT_BYTES - total,
        "within_budget": total <= LIMIT_BYTES,
    }
    log = run / "storage_usage_by_stage.csv"
    exists = log.exists()
    with log.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, sort_keys=True))
    if total > LIMIT_BYTES:
        raise RuntimeError("run exceeds 12 GiB compact-storage budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
