#!/usr/bin/env python3
"""Project retained repeated-FiLM test candidates into compact stage tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


FORBIDDEN_GT_COLUMNS = {
    "candidate_positive",
    "candidate_gt_iou",
    "angle_error",
    "correct_candidate_id",
    "gt_grasp",
    "gt_mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--nms-candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--expected-raw", type=int, default=1_466_046)
    parser.add_argument("--expected-mask-valid", type=int, default=1_449_011)
    parser.add_argument("--expected-nms", type=int, default=187_077)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def _validate_scope(
    output_root: Path, tmp_root: Path
) -> tuple[Path, Path]:
    output = output_root.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    run_root = _protected_run_root(output)
    if _protected_run_root(temporary) != run_root:
        raise ValueError("output-root and tmp-root belong to different active runs")
    configured_tmp = (run_root / "tmp").resolve()
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    if output == temporary or temporary in output.parents:
        raise ValueError("final candidate tables cannot be stored below tmp-root")
    output.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    return output, temporary


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _stage_values(path: Path) -> set[str]:
    values: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["stage"], batch_size=100_000):
        values.update(map(str, batch.column(0).to_pylist()))
    return values


def _verify_primary_keys(
    path: Path,
    *,
    expected_rows: int,
    require_valid: bool | None,
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["sample_id", "candidate_id", "valid"], batch_size=100_000
    ):
        sample_ids = batch.column(0).to_pylist()
        candidate_ids = batch.column(1).to_pylist()
        valid = batch.column(2).to_pylist()
        if require_valid is not None and any(
            bool(value) is not require_valid for value in valid
        ):
            raise ValueError(f"candidate validity contract differs: {path}")
        local = set(zip(map(str, sample_ids), map(str, candidate_ids), strict=True))
        if len(local) != batch.num_rows or keys & local:
            raise ValueError(f"duplicate sample_id+candidate_id: {path}")
        keys.update(local)
        rows += batch.num_rows
    if rows != expected_rows or len(keys) != expected_rows:
        raise ValueError(f"candidate table row count differs: {path}")
    return keys


def _write_mask_validated(
    source: Path, destination_tmp: Path
) -> tuple[int, pa.Schema]:
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    stage_index = schema.get_field_index("stage")
    valid_index = schema.get_field_index("valid")
    if stage_index < 0 or valid_index < 0:
        raise ValueError("raw candidate schema omits stage/valid")
    writer = pq.ParquetWriter(
        destination_tmp, schema, compression="zstd", use_dictionary=True
    )
    rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=100_000):
            filtered = batch.filter(pc.equal(batch.column(valid_index), True))
            if filtered.num_rows == 0:
                continue
            arrays = [filtered.column(index) for index in range(filtered.num_columns)]
            arrays[stage_index] = pa.array(
                ["mask_validated"] * filtered.num_rows,
                type=schema.field(stage_index).type,
            )
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
            rows += filtered.num_rows
    finally:
        writer.close()
    return rows, schema


def _atomic_hardlink(source: Path, destination: Path, tmp_root: Path) -> None:
    temporary = tmp_root / f"{destination.name}.{uuid.uuid4().hex}.link"
    os.link(source, temporary)
    os.replace(temporary, destination)


def main() -> int:
    args = parse_args()
    raw = args.raw_candidates.expanduser().resolve()
    nms = args.nms_candidates.expanduser().resolve()
    output, tmp_root = _validate_scope(args.output_root, args.tmp_root)
    destinations = {
        "raw": output / "raw_candidates.parquet",
        "mask_validated": output / "mask_validated_candidates.parquet",
        "nms": output / "nms_candidates.parquet",
        "manifest": output / "candidate_tables.manifest.json",
    }
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError("retained candidate projection already exists")

    raw_file = pq.ParquetFile(raw)
    nms_file = pq.ParquetFile(nms)
    if raw_file.schema_arrow != nms_file.schema_arrow:
        raise ValueError("retained raw/NMS schemas differ")
    if FORBIDDEN_GT_COLUMNS & set(raw_file.schema_arrow.names):
        raise ValueError("retained candidate table contains GT-derived columns")
    if raw_file.metadata.num_rows != args.expected_raw:
        raise ValueError("retained raw count differs")
    if nms_file.metadata.num_rows != args.expected_nms:
        raise ValueError("retained NMS count differs")
    if _stage_values(raw) != {"raw"} or _stage_values(nms) != {"nms"}:
        raise ValueError("retained candidate stage labels differ")

    raw_keys = _verify_primary_keys(
        raw, expected_rows=args.expected_raw, require_valid=None
    )
    nms_keys = _verify_primary_keys(
        nms, expected_rows=args.expected_nms, require_valid=True
    )
    if not nms_keys <= raw_keys:
        raise ValueError("retained NMS pool is not a subset of raw candidates")

    mask_tmp = (
        tmp_root
        / f"mask_validated_candidates.parquet.{uuid.uuid4().hex}.tmp"
    )
    mask_count, schema = _write_mask_validated(raw, mask_tmp)
    if mask_count != args.expected_mask_valid:
        raise ValueError("retained mask-valid count differs")
    mask_file = pq.ParquetFile(mask_tmp)
    if (
        mask_file.schema_arrow != schema
        or mask_file.metadata.num_rows != args.expected_mask_valid
        or _stage_values(mask_tmp) != {"mask_validated"}
    ):
        raise ValueError("mask-valid candidate projection verification failed")
    mask_keys = _verify_primary_keys(
        mask_tmp,
        expected_rows=args.expected_mask_valid,
        require_valid=True,
    )
    if not nms_keys <= mask_keys <= raw_keys:
        raise ValueError("candidate-stage subset invariant failed")

    _atomic_hardlink(raw, destinations["raw"], tmp_root)
    os.replace(mask_tmp, destinations["mask_validated"])
    _atomic_hardlink(nms, destinations["nms"], tmp_root)
    artifacts = {}
    for stage in ("raw", "mask_validated", "nms"):
        path = destinations[stage]
        artifacts[stage] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": pq.ParquetFile(path).metadata.num_rows,
            "primary_key": ["sample_id", "candidate_id"],
            "compression": "zstd",
            "gt_free": True,
            "hardlinked_from_retained_source": stage in {"raw", "nms"},
        }
    manifest = {
        "schema_version": 1,
        "status": "COMPLETED",
        "lineage": "hierarchical_repeated_film_only",
        "split": "test",
        "single_film_used": False,
        "source_raw_candidates": str(raw),
        "source_raw_candidates_sha256": sha256_file(raw),
        "source_nms_candidates": str(nms),
        "source_nms_candidates_sha256": sha256_file(nms),
        "candidate_stage_subset_invariant": True,
        "candidate_primary_keys_unique": True,
        "schema_sha256": _canonical_hash(
            [(field.name, str(field.type), field.nullable) for field in schema]
        ),
        "artifacts": artifacts,
    }
    temporary_manifest = (
        tmp_root / f"candidate_tables.manifest.{uuid.uuid4().hex}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, destinations["manifest"])
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
