"""Stable sample and candidate identities for frozen-pool re-ranking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CANDIDATE_POSE_FIELDS = (
    "center_uv",
    "center_depth_m",
    "center_camera_xyz_m",
    "angle_rad",
    "width_m",
    "width_px",
    "endpoints_uv",
    "T_camera_grasp_fixed_approach",
)


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without changing the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_prefix_sha256(path: str | Path, row_count: int) -> str | None:
    """Hash exactly the first N canonical JSONL records, including newlines."""

    count = int(row_count)
    if count < 0:
        raise ValueError("JSONL prefix row count must be non-negative")
    if count == 0:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for index in range(count):
            line = stream.readline()
            if not line or not line.strip() or not line.endswith(b"\n"):
                raise ValueError(
                    f"JSONL has no canonical row {index + 1} for prefix hash"
                )
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"JSONL prefix row {index + 1} is invalid"
                ) from error
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"JSONL prefix row {index + 1} is not an object"
                )
            digest.update(line)
    return digest.hexdigest()


def stable_sample_id(scene_id: str, question_index: int) -> str:
    """Reproduce the repository's split-local stable sample identifier."""

    index = int(question_index)
    identity = f"{scene_id}\t{index}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:16]
    return f"q{index:07d}_{digest}"


def _record_value(record: Mapping[str, Any], field: str) -> Any:
    if field == "center_uv":
        return record.get(
            field, [record.get("center_u_px"), record.get("center_v_px")]
        )
    if field == "endpoints_uv":
        return record.get(
            field, [record.get("endpoint_1_uv"), record.get("endpoint_2_uv")]
        )
    return record.get(field)


def candidate_identity_sha256(record: Mapping[str, Any]) -> str:
    """Hash the immutable candidate ID and pose using float64 canonical bytes."""

    sample_id = str(record.get("sample_id", ""))
    candidate_id = str(record.get("candidate_id", ""))
    if not sample_id or not candidate_id:
        raise ValueError("candidate identity requires sample_id and candidate_id")
    digest = hashlib.sha256()
    digest.update(sample_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(candidate_id.encode("utf-8"))
    for field in CANDIDATE_POSE_FIELDS:
        try:
            value = np.asarray(_record_value(record, field), dtype="<f8")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{sample_id}/{candidate_id} has invalid immutable field {field}"
            ) from error
        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"{sample_id}/{candidate_id} has non-finite immutable field {field}"
            )
        digest.update(b"\0")
        digest.update(field.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def assert_candidate_identity_invariant(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> None:
    """Assert that a result is a pure permutation of an immutable candidate pool."""

    if len(before) != len(after):
        raise AssertionError(
            f"candidate count changed during re-ranking: {len(before)} -> {len(after)}"
        )
    before_hashes = {
        (str(row["sample_id"]), str(row["candidate_id"])): candidate_identity_sha256(
            row
        )
        for row in before
    }
    after_hashes = {
        (str(row["sample_id"]), str(row["candidate_id"])): candidate_identity_sha256(
            row
        )
        for row in after
    }
    if len(before_hashes) != len(before) or len(after_hashes) != len(after):
        raise AssertionError("candidate IDs are not unique within their sample")
    if before_hashes.keys() != after_hashes.keys():
        missing = sorted(before_hashes.keys() - after_hashes.keys())[:5]
        added = sorted(after_hashes.keys() - before_hashes.keys())[:5]
        raise AssertionError(
            f"candidate ID set changed: missing={missing}, added={added}"
        )
    changed = [
        key for key in before_hashes if before_hashes[key] != after_hashes[key]
    ]
    if changed:
        raise AssertionError(f"candidate pose changed for {changed[:5]}")
