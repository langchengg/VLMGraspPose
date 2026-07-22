#!/usr/bin/env python3
"""Deterministically preregister a stratified real-robot trial manifest.

This script only prepares immutable trial intents.  It cannot command hardware
and does not create real-robot success labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.grasping.vgn_pipeline import atomic_write_json


SUCCESS_DEFINITIONS = ("lift_10cm_hold_3s", "placed_in_bin")
STRATIFICATION_FIELDS = (
    "object_category",
    "geometry_shape",
    "clutter_level",
    "approach_direction",
    "seen_status",
)
REQUIRED_FIELDS = (
    "sample_id",
    "instruction",
    "mask_iou",
    "vgn_quality",
    *STRATIFICATION_FIELDS,
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("trials", []) if isinstance(payload, Mapping) else payload
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        raise ValueError("trial candidates must be JSONL, JSON, or CSV")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("trial candidate input must contain object rows")
    return [dict(row) for row in rows]


def _stable_key(row: Mapping[str, Any], seed: int) -> str:
    identity = "|".join(
        (str(seed), str(row.get("sample_id", "")), str(row.get("instruction", "")))
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_trial_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    success_definition: str,
) -> list[dict[str, Any]]:
    """Select trials by deterministic round-robin over declared strata."""

    if success_definition not in SUCCESS_DEFINITIONS:
        raise ValueError(f"unsupported success definition: {success_definition}")
    if count < 50:
        raise ValueError("real-robot preregistration requires at least 50 trials")
    candidates = [dict(row) for row in rows]
    if count > len(candidates):
        raise ValueError(f"requested {count} trials but only {len(candidates)} candidates exist")
    for index, row in enumerate(candidates):
        missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "")]
        if missing:
            raise ValueError(f"candidate row {index} is missing required fields: {missing}")
        for field in ("mask_iou", "vgn_quality"):
            value = float(row[field])
            if not np.isfinite(value):
                raise ValueError(f"candidate row {index} has non-finite {field}")
            row[field] = value

    quality_edges = np.quantile(
        [float(row["vgn_quality"]) for row in candidates], [0.25, 0.5, 0.75]
    )
    iou_edges = np.quantile(
        [float(row["mask_iou"]) for row in candidates], [0.25, 0.5, 0.75]
    )
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        quality_bin = int(np.searchsorted(quality_edges, row["vgn_quality"], side="right"))
        iou_bin = int(np.searchsorted(iou_edges, row["mask_iou"], side="right"))
        row["vgn_quality_quartile"] = quality_bin
        row["mask_iou_quartile"] = iou_bin
        stratum = (
            *(str(row[field]) for field in STRATIFICATION_FIELDS),
            f"quality_q{quality_bin}",
            f"mask_iou_q{iou_bin}",
        )
        groups[stratum].append(row)
    queues: dict[tuple[str, ...], deque[dict[str, Any]]] = {}
    for stratum, values in groups.items():
        values.sort(key=lambda row: _stable_key(row, seed))
        queues[stratum] = deque(values)

    selected: list[dict[str, Any]] = []
    strata = sorted(queues)
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            queue = queues[stratum]
            if not queue:
                continue
            selected.append(queue.popleft())
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError("candidate queues exhausted before requested trial count")

    manifest: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        item = dict(row)
        item.update(
            {
                "trial_id": f"robot_trial_{index:04d}",
                "preregistered_order": index,
                "selection_seed": int(seed),
                "success_definition": success_definition,
                "one_query_one_grounding_one_top1_one_execution": True,
                "retry_same_trial_allowed": False,
                "physical_execution_status": "not_attempted",
            }
        )
        manifest.append(item)
    return manifest


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--success-definition", choices=SUCCESS_DEFINITIONS, default="lift_10cm_hold_3s"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = _read_rows(args.input.expanduser().resolve())
    manifest = build_trial_manifest(
        rows,
        count=args.count,
        seed=args.seed,
        success_definition=args.success_definition,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "trials_manifest.jsonl", manifest)
    atomic_write_json(
        output / "protocol.json",
        {
            "status": "preregistered_not_executed",
            "executor_mode": "dry_run",
            "hardware_enabled": False,
            "trial_count": len(manifest),
            "seed": args.seed,
            "success_definition": args.success_definition,
            "minimum_lift_m": 0.10 if args.success_definition == "lift_10cm_hold_3s" else None,
            "minimum_hold_s": 3.0 if args.success_definition == "lift_10cm_hold_3s" else None,
            "stratification_fields": list(STRATIFICATION_FIELDS),
            "real_robot_grasp_success_rate": None,
            "reason": "no physical robot execution logs",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
