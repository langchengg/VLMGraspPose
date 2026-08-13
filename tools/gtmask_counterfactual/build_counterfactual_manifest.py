#!/usr/bin/env python3
"""Publish the exact 7,675-row P1 denominator using identity joins only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.audit import transition_pipeline_status  # noqa: E402
from gtmask_counterfactual.contracts import RunState  # noqa: E402
from gtmask_counterfactual.io import artifact_record, sha256_file  # noqa: E402
from gtmask_counterfactual.manifest import (  # noqa: E402
    EXPECTED_SAMPLE_COUNT,
    build_counterfactual_manifest,
)
from gtmask_counterfactual.mapping_pipeline import (  # noqa: E402
    publish_manifest_artifacts,
)


DENOMINATOR_COLUMNS = (
    "sample_id",
    "scene_id",
    "frame_id",
    "query_id",
    "query_type",
    "target_instance_id",
    "gt_grasp_target_instance_id",
    "language_prompt",
    "rgb_path",
    "rgb_sha256",
    "rgb_height",
    "rgb_width",
    "depth_path",
    "depth_sha256",
    "depth_height",
    "depth_width",
    "intrinsics_path",
    "intrinsics_sha256",
    "prepared_gt_mask_path",
    "prepared_gt_mask_sha256",
    "source_instance_mask_path",
    "source_instance_mask_sha256",
    "gt_grasp_set_path",
    "gt_grasp_set_sha256",
)
ROUTE_OPTIONAL_COLUMNS = (
    "sample_id",
    "scene_id",
    "frame_id",
    "query_id",
    "target_instance_id",
    "rgb_sha256",
    "depth_sha256",
    "language_prompt",
    "source_identity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--denominator", required=True, type=Path)
    parser.add_argument("--g1-source", required=True, type=Path)
    parser.add_argument("--c1-source", required=True, type=Path)
    parser.add_argument("--d1-source", required=True, type=Path)
    parser.add_argument("--unified-final-lock", required=True, type=Path)
    parser.add_argument("--d1-final-lock", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_SAMPLE_COUNT)
    return parser.parse_args()


def _regular(path: Path) -> Path:
    value = path.expanduser().resolve()
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {value}")
    return value


def _denominator(path: Path) -> list[dict[str, Any]]:
    source = _regular(path)
    schema = pq.read_schema(source)
    missing = sorted(set(DENOMINATOR_COLUMNS).difference(schema.names))
    if missing:
        raise ValueError(f"denominator misses required columns: {missing}")
    # Paths, hashes, identities, prompt and declared shapes only: no mask pixels
    # and no GT-grasp rows are materialised at P1.
    return pq.read_table(source, columns=list(DENOMINATOR_COLUMNS)).to_pylist()


def _route(path: Path) -> list[dict[str, Any]]:
    source = _regular(path)
    schema = pq.read_schema(source)
    columns = [name for name in ROUTE_OPTIONAL_COLUMNS if name in schema.names]
    if "source_identity" not in columns:
        raise ValueError(f"route source misses source_identity: {source}")
    if "sample_id" not in columns and not {
        "scene_id",
        "frame_id",
        "query_id",
        "target_instance_id",
    }.issubset(columns):
        raise ValueError(f"route source lacks an exact identity key: {source}")
    return pq.read_table(source, columns=columns).to_pylist()


def _inventory_records(value: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            records.append(value)
        for child in value.values():
            records.extend(_inventory_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_inventory_records(child))
    return records


def _require_locked_source(path: Path, lock_path: Path) -> None:
    source = _regular(path)
    final_lock = _regular(lock_path)
    value = json.loads(final_lock.read_text(encoding="utf-8"))
    inventory = value.get("inventory") if isinstance(value, dict) else None
    records = _inventory_records(inventory)
    observed_path = str(source)
    observed_hash = sha256_file(source)
    for record in records:
        record_path = record.get("path")
        if record_path is None and record.get("relative_path") is not None:
            record_path = str(
                (final_lock.parent / str(record["relative_path"])).resolve()
            )
        if str(Path(str(record_path)).expanduser().resolve()) == observed_path and str(
            record.get("sha256")
        ) == observed_hash:
            return
    raise PermissionError(f"source is not byte-bound by final lock: {source}")


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    status = json.loads((root / "pipeline_status.json").read_text(encoding="utf-8"))
    if status.get("status") != RunState.P1_BASELINE_REPLAY_PASS.value:
        raise PermissionError("P1 manifest requires baseline replay PASS first")
    sources = {
        "denominator": _regular(args.denominator),
        "g1": _regular(args.g1_source),
        "c1": _regular(args.c1_source),
        "d1": _regular(args.d1_source),
        "unified_final_lock": _regular(args.unified_final_lock),
        "d1_final_lock": _regular(args.d1_final_lock),
    }
    _require_locked_source(sources["g1"], sources["unified_final_lock"])
    _require_locked_source(sources["c1"], sources["unified_final_lock"])
    _require_locked_source(sources["d1"], sources["d1_final_lock"])
    try:
        _require_locked_source(sources["denominator"], sources["unified_final_lock"])
    except PermissionError:
        _require_locked_source(sources["denominator"], sources["d1_final_lock"])
    result = build_counterfactual_manifest(
        _denominator(sources["denominator"]),
        {route: _route(sources[route]) for route in ("g1", "c1", "d1")},
        expected_count=args.expected_count,
    )
    publish_manifest_artifacts(
        root,
        result,
        source_records={name: artifact_record(path) for name, path in sources.items()},
    )
    transition_pipeline_status(
        root,
        RunState.P1_BASELINE_REPLAY_PASS,
        first_incomplete_stage=RunState.P2_GT_MAPPING_PASS.value,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
