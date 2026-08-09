#!/usr/bin/env python3
"""Select an evenly spaced, GT-free subset for cache-cold VLM repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_evenly_spaced(
    rows: list[dict[str, Any]], sample_count: int
) -> list[dict[str, str]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    eligible = [row for row in rows if row.get("candidate_ids")]
    if len(eligible) < sample_count:
        raise ValueError(
            f"only {len(eligible)} nonempty rows for {sample_count} repeats"
        )
    indices = [
        min(
            len(eligible) - 1,
            math.floor((index + 0.5) * len(eligible) / sample_count),
        )
        for index in range(sample_count)
    ]
    if len(indices) != len(set(indices)):
        raise RuntimeError("repeat subset index selection is not unique")
    selected = [{"sample_id": str(eligible[index]["sample_id"])} for index in indices]
    if len({row["sample_id"] for row in selected}) != sample_count:
        raise ValueError("repeat subset sample IDs are not unique")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    input_path = args.input_jsonl.expanduser().resolve()
    rows = _rows(input_path)
    if not rows:
        raise ValueError("input JSONL is empty")
    observed_splits = {str(row.get("split", "")) for row in rows}
    if observed_splits - {"val", "validation"}:
        raise ValueError(
            "VLM repeat subset may consume validation rows only; "
            f"observed={sorted(observed_splits)}"
        )
    selected = select_evenly_spaced(rows, args.sample_count)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    selection_path = root / "selection_manifest.jsonl"
    temporary = root / f".selection_manifest.{os.getpid()}.tmp"
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    temporary.replace(selection_path)
    summary = {
        "schema_version": 1,
        "selection_split": "validation",
        "selection_method": "evenly_spaced_over_nonempty_pilot_order",
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": _sha256(input_path),
        "input_sample_count": len(rows),
        "input_nonempty_sample_count": sum(
            bool(row.get("candidate_ids")) for row in rows
        ),
        "selected_sample_count": len(selected),
        "selected_sample_ids": [row["sample_id"] for row in selected],
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "uses_gt": False,
    }
    summary_path = root / "summary.json"
    temporary = root / f".summary.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
