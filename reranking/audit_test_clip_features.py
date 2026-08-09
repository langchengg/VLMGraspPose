"""Audit the frozen test-split dense CLIP candidate-feature cache.

The audit is intentionally independent of model selection and held-out labels.
It checks the cache against the frozen test manifest and Stage-1 proposal
candidate identities, then publishes an atomic manifest, checksum list, and
human-readable receipt inside the canonical reranking run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


FORBIDDEN_FIELDS = {
    "answer_instance_value",
    "candidate_correctness",
    "candidate_iou",
    "correctness",
    "ground_truth",
    "ground_truth_iou",
    "gt_correctness",
    "gt_iou",
    "gt_mask",
    "label",
    "target_instance_id",
    "y90",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _schema_descriptor(archive: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    descriptor = []
    for name in sorted(archive.files):
        value = np.asarray(archive[name])
        # NumPy chooses a per-array fixed Unicode width (for example U35 or
        # U45) from the longest candidate ID in that sample.  That storage
        # width is not a semantic schema difference; exact, non-truncated IDs
        # are checked separately against candidate_index.parquet below.
        dtype = (
            "unicode"
            if value.dtype.kind == "U"
            else "bytes"
            if value.dtype.kind == "S"
            else str(value.dtype)
        )
        descriptor.append(
            {
                "name": name,
                "dtype": dtype,
                "rank": int(value.ndim),
                "tail_shape": list(value.shape[1:]) if value.ndim else [],
                "sample_axis": bool(value.ndim),
            }
        )
    return descriptor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repository.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    hifi = repository / "HiFi_reproduction"
    experiment = hifi / "outputs/sam3_proposal_bank_p90_v1"
    source_manifest = (
        hifi
        / "runs/modular_reranking_repeatedfilm_v1_20260729_203147"
        / "compact_inputs/test/manifest.jsonl"
    )
    clip_root = experiment / "cache/clip_embeddings"
    cache_root = clip_root / "dense_candidates"
    canonical_path = clip_root / "manifest_stage1_test.json"
    shard_paths = sorted(clip_root.glob("manifest_stage1_test.shard-*-of-004.json"))

    errors: list[str] = []
    rows = [
        json.loads(line)
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_ids = [str(row["sample_id"]) for row in rows]
    if len(expected_ids) != 7675:
        errors.append(f"expected manifest contains {len(expected_ids)} samples, not 7675")
    if len(set(expected_ids)) != len(expected_ids):
        errors.append("expected manifest contains duplicate sample IDs")

    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical_records = canonical.get("records", [])
    canonical_ids = [str(record.get("sample_id")) for record in canonical_records]
    if canonical.get("status") != "COMPLETE":
        errors.append("canonical CLIP manifest is not COMPLETE")
    if canonical_ids != expected_ids:
        errors.append("canonical CLIP record order/identity differs from frozen test manifest")
    if int(canonical.get("samples", -1)) != len(expected_ids):
        errors.append("canonical CLIP sample count differs from frozen test manifest")

    if len(shard_paths) != 4:
        errors.append(f"found {len(shard_paths)} CLIP shard manifests, expected 4")
    shard_ids: list[str] = []
    shard_counts: list[dict[str, Any]] = []
    for expected_index, path in enumerate(shard_paths):
        shard = json.loads(path.read_text(encoding="utf-8"))
        records = shard.get("records", [])
        ids = [str(record.get("sample_id")) for record in records]
        shard_ids.extend(ids)
        shard_counts.append(
            {
                "path": str(path),
                "status": shard.get("status"),
                "shard_index": shard.get("shard_index"),
                "num_shards": shard.get("num_shards"),
                "samples": len(ids),
            }
        )
        if shard.get("status") != "COMPLETE":
            errors.append(f"incomplete shard manifest: {path.name}")
        if shard.get("shard_index") != expected_index or shard.get("num_shards") != 4:
            errors.append(f"invalid shard identity: {path.name}")
        if int(shard.get("samples", -1)) != len(ids):
            errors.append(f"shard sample-count mismatch: {path.name}")
    shard_counter = Counter(shard_ids)
    duplicate_shard_ids = sorted(key for key, count in shard_counter.items() if count > 1)
    if duplicate_shard_ids:
        errors.append(f"CLIP shard manifests contain {len(duplicate_shard_ids)} duplicates")
    missing_shard_ids = sorted(set(expected_ids) - set(shard_ids))
    extra_shard_ids = sorted(set(shard_ids) - set(expected_ids))
    if missing_shard_ids:
        errors.append(f"CLIP shard manifests omit {len(missing_shard_ids)} samples")
    if extra_shard_ids:
        errors.append(f"CLIP shard manifests contain {len(extra_shard_ids)} extra samples")

    record_by_id = {str(record.get("sample_id")): record for record in canonical_records}
    schema_hash_counts: Counter[str] = Counter()
    feature_dimensions: Counter[int] = Counter()
    output_records: list[dict[str, Any]] = []
    checksum_rows: list[tuple[str, Path]] = []
    candidate_rows = 0
    unreadable = 0
    non_finite_values = 0
    candidate_identity_mismatches = 0
    duplicate_candidate_ids = 0
    forbidden_fields_seen: set[str] = set()

    for number, sample_id in enumerate(expected_ids, start=1):
        path = cache_root / f"{sample_id}.npz"
        if not path.is_file():
            errors.append(f"missing CLIP cache: {sample_id}")
            continue
        checksum = _sha256(path)
        checksum_rows.append((checksum, path))
        try:
            with np.load(path, allow_pickle=False) as archive:
                descriptor = _schema_descriptor(archive)
                schema_hash = _stable_hash(descriptor)
                schema_hash_counts[schema_hash] += 1
                names = set(archive.files)
                forbidden_fields_seen.update(names & FORBIDDEN_FIELDS)
                candidate_ids = np.asarray(archive["candidate_id"]).astype(str)
                count = len(candidate_ids)
                candidate_rows += count
                if len(set(candidate_ids.tolist())) != count:
                    duplicate_candidate_ids += 1
                for name in archive.files:
                    value = np.asarray(archive[name])
                    if value.ndim and value.shape[0] != count:
                        errors.append(
                            f"sample-axis mismatch for {sample_id}:{name}: "
                            f"{value.shape[0]} != {count}"
                        )
                    if value.dtype.kind in "fciu":
                        non_finite_values += int(value.size - np.isfinite(value).sum())
                for name in ("masked_embeddings", "box_embeddings", "background_embeddings"):
                    value = np.asarray(archive[name])
                    if value.ndim != 2:
                        errors.append(f"invalid embedding rank for {sample_id}:{name}")
                    elif value.shape[1]:
                        feature_dimensions[int(value.shape[1])] += 1
        except Exception as error:
            unreadable += 1
            errors.append(f"unreadable CLIP cache {sample_id}: {type(error).__name__}: {error}")
            continue

        proposal_index = experiment / "proposals" / sample_id / "candidate_index.parquet"
        try:
            proposal_ids = (
                pq.read_table(proposal_index, columns=["candidate_id"])
                .column("candidate_id")
                .to_pylist()
            )
            proposal_ids = [str(value) for value in proposal_ids]
        except Exception as error:
            errors.append(
                f"unreadable proposal candidate index {sample_id}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        if candidate_ids.tolist() != sorted(proposal_ids):
            candidate_identity_mismatches += 1
        expected_count = int(record_by_id.get(sample_id, {}).get("candidate_count", -1))
        if expected_count != count:
            errors.append(
                f"canonical candidate count mismatch for {sample_id}: "
                f"{expected_count} != {count}"
            )
        output_records.append(
            {
                "sample_id": sample_id,
                "candidate_count": count,
                "schema_sha256": schema_hash,
                "sha256": checksum,
                "path": str(path),
            }
        )
        if number % 250 == 0:
            print(f"audited {number}/{len(expected_ids)}", flush=True)

    if len(schema_hash_counts) != 1:
        errors.append(f"observed {len(schema_hash_counts)} CLIP cache schemas, expected 1")
    if set(feature_dimensions) != {512}:
        errors.append(f"embedding dimensions differ from 512: {dict(feature_dimensions)}")
    if unreadable:
        errors.append(f"{unreadable} CLIP cache files were unreadable")
    if non_finite_values:
        errors.append(f"CLIP cache contains {non_finite_values} NaN/Inf values")
    if candidate_identity_mismatches:
        errors.append(
            f"{candidate_identity_mismatches} CLIP caches disagree with proposal candidate IDs"
        )
    if duplicate_candidate_ids:
        errors.append(f"{duplicate_candidate_ids} samples contain duplicate candidate IDs")
    if forbidden_fields_seen:
        errors.append(f"forbidden fields present: {sorted(forbidden_fields_seen)}")
    if candidate_rows != int(canonical.get("candidate_rows", -1)):
        errors.append(
            f"candidate row total differs from canonical manifest: "
            f"{candidate_rows} != {canonical.get('candidate_rows')}"
        )
    if len(output_records) != len(expected_ids):
        errors.append(f"audited {len(output_records)}/{len(expected_ids)} sample caches")

    for path in [canonical_path, *shard_paths, source_manifest]:
        checksum_rows.append((_sha256(path), path))

    completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    passed = not errors
    payload = {
        "status": "PASSED" if passed else "FAILED",
        "completed_at_utc": completed_at,
        "expected_samples": len(expected_ids),
        "unique_expected_samples": len(set(expected_ids)),
        "audited_samples": len(output_records),
        "missing_samples": len(set(expected_ids) - {row["sample_id"] for row in output_records}),
        "duplicate_manifest_samples": len(duplicate_shard_ids),
        "candidate_rows": candidate_rows,
        "unreadable_files": unreadable,
        "non_finite_values": non_finite_values,
        "candidate_identity_mismatches": candidate_identity_mismatches,
        "duplicate_candidate_id_samples": duplicate_candidate_ids,
        "composite_index_unique": (
            len(output_records) == len(expected_ids)
            and not duplicate_candidate_ids
            and len(set(expected_ids)) == len(expected_ids)
        ),
        "schema_sha256_counts": dict(sorted(schema_hash_counts.items())),
        "embedding_dimension_counts": {
            str(key): value for key, value in sorted(feature_dimensions.items())
        },
        "forbidden_fields_present": sorted(forbidden_fields_seen),
        "canonical_manifest": str(canonical_path),
        "canonical_manifest_sha256": _sha256(canonical_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "shards": shard_counts,
        "errors": errors,
        "records": output_records,
    }
    manifest_path = run_root / "manifests/test_clip_feature_manifest.json"
    checksum_path = run_root / "manifests/test_clip_feature_checksums.sha256"
    audit_path = run_root / "audit/TEST_CLIP_FEATURE_AUDIT.md"
    _atomic_text(
        manifest_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    checksum_lines = [
        f"{checksum}  {path}" for checksum, path in sorted(checksum_rows, key=lambda item: str(item[1]))
    ]
    _atomic_text(checksum_path, "\n".join(checksum_lines) + "\n")
    schema_summary = ", ".join(
        f"`{key}`: {value}" for key, value in sorted(schema_hash_counts.items())
    )
    error_lines = "\n".join(f"- {error}" for error in errors) or "- None."
    audit_text = f"""# Test CLIP feature audit

- Status: **{'PASSED' if passed else 'FAILED'}**
- Completed: `{completed_at}`
- Expected / audited samples: {len(expected_ids)} / {len(output_records)}
- Candidate rows: {candidate_rows}
- Missing samples: {len(set(expected_ids) - {row['sample_id'] for row in output_records})}
- Duplicate manifest samples: {len(duplicate_shard_ids)}
- Unreadable files: {unreadable}
- NaN/Inf values: {non_finite_values}
- Candidate-ID mismatches: {candidate_identity_mismatches}
- Samples with duplicate candidate IDs: {duplicate_candidate_ids}
- Composite `(sample_id, candidate_id)` index unique: {payload['composite_index_unique']}
- Embedding dimension counts: `{dict(sorted(feature_dimensions.items()))}`
- Schema SHA-256 counts: {schema_summary}
- Forbidden fields: `{sorted(forbidden_fields_seen)}`
- Canonical manifest was atomically rebuilt by the existing extraction utility from four COMPLETE shard manifests and its record order exactly matches the frozen test manifest.

## Errors

{error_lines}

## Published evidence

- `{manifest_path}`
- `{checksum_path}`
"""
    _atomic_text(audit_path, audit_text)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "records"},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
