#!/usr/bin/env python3
"""Build the 7,675-row pre-lock GT path/hash registry without opening pixels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.mapping import (  # noqa: E402
    EXPECTED_SAMPLE_COUNT,
    GTAuthorityColumns,
    GTMappingError,
    build_prelock_registry,
    join_real_authority_rows,
)
from gtmask_counterfactual.contracts import RunState  # noqa: E402
from unified_reranking.hashing import canonical_sha256, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--prepared-label-manifest", required=True, type=Path)
    parser.add_argument("--visual-paired-manifest", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_SAMPLE_COUNT)
    return parser.parse_args()


def _read_columns(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"manifest must be a regular non-symlink file: {source}")
    schema = pq.read_schema(source)
    missing = sorted(set(columns).difference(schema.names))
    if missing:
        raise ValueError(f"manifest {source} misses columns: {missing}")
    # The explicit projection is the safety boundary: grasp labels, outcomes,
    # and mask pixels are neither requested nor materialized.
    return pq.read_table(source, columns=columns).to_pylist()


def _question_column(path: Path) -> str:
    names = pq.read_schema(path.expanduser().resolve()).names
    if "query_id" in names:
        return "query_id"
    if "question_index" in names:
        return "question_index"
    raise ValueError(f"manifest lacks query_id/question_index: {path}")


def _unique(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{label} contains an empty sample_id")
        if sample_id in result:
            raise ValueError(f"{label} contains duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pacsv.write_csv(pa.Table.from_pylist(rows), temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    status = json.loads((root / "pipeline_status.json").read_text(encoding="utf-8"))
    if status.get("status") != RunState.P1_BASELINE_REPLAY_PASS.value:
        raise PermissionError("pre-lock GT registry requires P1_BASELINE_REPLAY_PASS")
    fields = GTAuthorityColumns()
    sample_question = _question_column(args.sample_manifest)
    samples = _read_columns(
        args.sample_manifest, ["sample_id", "scene_id", sample_question]
    )
    if sample_question == "query_id":
        for row in samples:
            row["question_index"] = row.pop("query_id")
    prepared_question = _question_column(args.prepared_label_manifest)
    prepared_columns = [
        "sample_id",
        "scene_id",
        prepared_question,
        "target_object_id",
        "prepared_gt_mask_path",
        "prepared_gt_mask_sha256",
    ]
    visual_question = _question_column(args.visual_paired_manifest)
    visual_columns = [
        "sample_id",
        "scene_id",
        visual_question,
        "target_instance_id",
        "gt_mask_path",
        "gt_mask_sha256",
    ]
    prepared = _read_columns(args.prepared_label_manifest, prepared_columns)
    visual = _read_columns(args.visual_paired_manifest, visual_columns)
    for rows, question in (
        (prepared, prepared_question),
        (visual, visual_question),
    ):
        if question == "query_id":
            for row in rows:
                row["question_index"] = row.pop("query_id")
    denominator_ids = {str(row["sample_id"]) for row in samples}
    prepared_by_id = _unique(prepared, label="prepared-label manifest")
    visual_by_id = _unique(visual, label="visual-paired manifest")
    extras = sorted((set(prepared_by_id) | set(visual_by_id)) - denominator_ids)
    if extras:
        raise ValueError(f"GT authority has samples outside denominator: {extras[:5]}")
    authorities: list[dict[str, Any]] = []
    unresolved_reasons: dict[str, str] = {}
    for sample_id in sorted(denominator_ids):
        left = prepared_by_id.get(sample_id)
        right = visual_by_id.get(sample_id)
        if left is None or right is None:
            missing = []
            if left is None:
                missing.append("prepared GT authority")
            if right is None:
                missing.append("visual/instance GT authority")
            unresolved_reasons[sample_id] = "missing " + " and ".join(missing)
            continue
        try:
            authorities.extend(join_real_authority_rows([left], [right]))
        except GTMappingError as error:
            unresolved_reasons[sample_id] = str(error)
    registry = build_prelock_registry(
        samples,
        authorities,
        columns=fields,
        expected_count=args.expected_count,
    )
    for row in registry:
        reason = unresolved_reasons.get(str(row["sample_id"]))
        if reason is not None:
            row["mapping_reason"] = reason
    output = root / "03_gt_mask_registry"
    parquet = output / "gt_mask_registry_prelock.parquet"
    csv = output / "gt_mask_registry_prelock.csv"
    _atomic_parquet(parquet, registry)
    _atomic_csv(csv, registry)
    access: dict[str, Any] = {
        "schema_version": 1,
        "status": "PATH_HASH_REGISTRY_COMPLETE",
        "stage": "P2_GT_MAPPING_PASS",
        "purpose": "prelock_path_hash_mapping_only",
        "sample_count": len(registry),
        "evaluable_mapping_count": sum(
            row["mapping_status"] == "PATH_HASH_INSTANCE_AUTHORITY_MAPPED"
            for row in registry
        ),
        "unresolved_mapping_count": sum(
            row["mapping_status"] != "PATH_HASH_INSTANCE_AUTHORITY_MAPPED"
            for row in registry
        ),
        "partition_total": len(registry),
        "sample_manifest": {
            "path": str(args.sample_manifest.expanduser().resolve()),
            "sha256": sha256_file(args.sample_manifest.expanduser().resolve()),
            "columns_read": ["sample_id", "scene_id", sample_question],
        },
        "prepared_label_manifest": {
            "path": str(args.prepared_label_manifest.expanduser().resolve()),
            "sha256": sha256_file(args.prepared_label_manifest.expanduser().resolve()),
            "columns_read": prepared_columns,
        },
        "visual_paired_manifest": {
            "path": str(args.visual_paired_manifest.expanduser().resolve()),
            "sha256": sha256_file(args.visual_paired_manifest.expanduser().resolve()),
            "columns_read": visual_columns,
        },
        "outputs": {
            "parquet": {"path": str(parquet), "sha256": sha256_file(parquet)},
            "csv": {"path": str(csv), "sha256": sha256_file(csv)},
        },
        "gt_mask_pixels_read": False,
        "gt_grasp_rows_read": False,
        "candidate_generation_allowed": False,
    }
    access["content_sha256"] = canonical_sha256(access)
    _atomic_json(output / "PRELOCK_ACCESS.json", access)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
