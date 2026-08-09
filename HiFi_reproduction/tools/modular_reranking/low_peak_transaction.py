"""Shared safety primitives for low-peak compact-shard merge transactions.

The helpers in this module deliberately support only exact compact Parquet
files below one active run's ``tmp/`` directory.  They never remove
directories, metadata, manifests, summaries, labels, or files outside that
scope.  A caller must first make a durable output and journal entry, then call
``release_verified_parquets`` explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq


DEFAULT_MAX_RUN_BYTES = 18 * 1024**3
DEFAULT_OUTPUT_HEADROOM_BYTES = 64 * 1024**2


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def is_below(path: Path, parent: Path) -> bool:
    return path != parent and parent in path.parents


def validate_release_scope(
    *,
    output: Path,
    tmp_root: Path,
    input_parquets: Sequence[Path],
) -> tuple[Path, Path, Path, list[Path]]:
    """Resolve and validate one low-peak transaction's filesystem boundary."""

    resolved_output = output.expanduser().resolve()
    resolved_tmp = tmp_root.expanduser().resolve()
    run_root = protected_run_root(resolved_output)
    if protected_run_root(resolved_tmp) != run_root:
        raise ValueError("output and tmp-root belong to different active runs")
    configured_tmp = (run_root / "tmp").resolve()
    if (
        resolved_tmp != configured_tmp
        and configured_tmp not in resolved_tmp.parents
    ):
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    resolved_inputs = [path.expanduser().resolve() for path in input_parquets]
    for path in resolved_inputs:
        if protected_run_root(path) != run_root or not is_below(
            path, configured_tmp
        ):
            raise ValueError(
                f"release input must be below the current run tmp/: {path}"
            )
        if path.suffix != ".parquet":
            raise ValueError(f"release input is not a compact Parquet: {path}")
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise ValueError("release input list contains duplicates")
    return resolved_output, resolved_tmp, run_root, resolved_inputs


def default_receipt_path(
    run_root: Path, *, family: str, output: Path
) -> Path:
    output_id = canonical_json_sha256(str(output.resolve()))[:16]
    return (
        run_root
        / "manifests"
        / "low_peak_merge"
        / f"{family}_{output_id}.json"
    )


def validate_receipt_path(path: Path, *, run_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    expected_root = (run_root / "manifests" / "low_peak_merge").resolve()
    if not is_below(resolved, expected_root):
        raise ValueError(
            f"low-peak receipt must be below {expected_root}: {resolved}"
        )
    return resolved


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_atomic_json(
    path: Path, payload: Mapping[str, Any], *, tmp_root: Path
) -> None:
    """Write JSON durably before any source file can be released."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def load_receipt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"low-peak receipt is not a JSON object: {path}")
    return value


def bind_or_resume_receipt(
    *,
    receipt_path: Path,
    plan: Mapping[str, Any],
    tmp_root: Path,
) -> dict[str, Any]:
    """Create a durable journal or reject any resume-parameter drift."""

    plan_sha256 = canonical_json_sha256(plan)
    existing = load_receipt(receipt_path)
    if existing is not None:
        if (
            existing.get("plan_sha256") != plan_sha256
            or existing.get("plan") != plan
        ):
            raise ValueError(
                "low-peak resume parameters or source identities changed"
            )
        return existing
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "plan_sha256": plan_sha256,
        "plan": dict(plan),
        "stages": {},
        "released_inputs": [],
    }
    durable_atomic_json(receipt_path, receipt, tmp_root=tmp_root)
    return receipt


def allocated_bytes(root: Path) -> int:
    """Return allocated bytes without double-counting hard-linked inodes."""

    total = 0
    visited: set[tuple[int, int]] = set()
    for directory, directory_names, filenames in os.walk(
        root, followlinks=False
    ):
        directory_path = Path(directory)
        for name in list(directory_names):
            item = directory_path / name
            if item.is_symlink():
                directory_names.remove(name)
                stat = item.lstat()
                identity = (int(stat.st_dev), int(stat.st_ino))
                if identity not in visited:
                    visited.add(identity)
                    total += int(stat.st_blocks) * 512
        for filename in filenames:
            item = directory_path / filename
            stat = item.lstat() if item.is_symlink() else item.stat()
            identity = (int(stat.st_dev), int(stat.st_ino))
            if identity in visited:
                continue
            visited.add(identity)
            total += int(stat.st_blocks) * 512
    return total


def enforce_budget_preflight(
    *,
    run_root: Path,
    input_paths: Sequence[Path],
    max_run_bytes: int,
    headroom_bytes: int = DEFAULT_OUTPUT_HEADROOM_BYTES,
) -> dict[str, int]:
    """Refuse a merge whose conservative stage peak exceeds the run budget."""

    if max_run_bytes <= 0 or headroom_bytes < 0:
        raise ValueError("storage budget values must be non-negative")
    current = allocated_bytes(run_root)
    source_logical = sum(path.stat().st_size for path in input_paths)
    # The merged table uses the same schema and ZSTD codec as its inputs.
    # Ten percent plus a fixed footer/buffer allowance is intentionally
    # conservative for changed row-group boundaries.
    projected_output = (
        (source_logical * 110 + 99) // 100 + headroom_bytes
    )
    projected_peak = current + projected_output
    if projected_peak > max_run_bytes:
        raise ValueError(
            "low-peak storage preflight exceeds budget: "
            f"current={current} projected_output={projected_output} "
            f"projected_peak={projected_peak} max={max_run_bytes}"
        )
    return {
        "current_allocated_bytes": current,
        "source_logical_bytes": source_logical,
        "projected_output_estimate_bytes": projected_output,
        "projected_peak_bytes": projected_peak,
        "max_run_bytes": max_run_bytes,
        "headroom_bytes": headroom_bytes,
    }


def parquet_key_contract(
    path: Path,
    *,
    key_columns: Sequence[str],
    require_sample_grouping: bool,
    include_all_columns: bool = False,
) -> dict[str, Any]:
    """Audit exact row/key order while keeping memory bounded per sample."""

    parquet = pq.ParquetFile(path)
    missing = sorted(set(key_columns) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Parquet key columns are missing in {path}: {missing}")
    digest = hashlib.sha256()
    digest.update(b"[")
    row_digest = hashlib.sha256()
    row_digest.update(b"[")
    row_count = 0
    first = True
    current_sample: str | None = None
    current_candidates: set[tuple[str, ...]] = set()
    closed_samples: set[str] = set()
    selected_columns = (
        list(parquet.schema_arrow.names)
        if include_all_columns
        else list(key_columns)
    )
    key_indices = [selected_columns.index(name) for name in key_columns]
    for batch in parquet.iter_batches(columns=selected_columns, batch_size=50_000):
        columns = [batch.column(index).to_pylist() for index in range(batch.num_columns)]
        for raw_values in zip(*columns, strict=True):
            raw_key = tuple(raw_values[index] for index in key_indices)
            if any(value is None for value in raw_key):
                raise ValueError(f"null Parquet primary key in {path}")
            key = tuple(map(str, raw_key))
            sample_id = key[0]
            if require_sample_grouping:
                if current_sample != sample_id:
                    if current_sample is not None:
                        closed_samples.add(current_sample)
                    if sample_id in closed_samples:
                        raise ValueError(
                            f"sample rows are not contiguous in {path}: {sample_id}"
                        )
                    current_sample = sample_id
                    current_candidates = set()
                candidate_key = key[1:]
                if candidate_key in current_candidates:
                    raise ValueError(
                        f"duplicate Parquet primary key in {path}: {key}"
                    )
                current_candidates.add(candidate_key)
            if not first:
                digest.update(b",")
            first = False
            digest.update(
                json.dumps(
                    key,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            if include_all_columns:
                if row_count:
                    row_digest.update(b",")
                row_digest.update(
                    json.dumps(
                        dict(zip(selected_columns, raw_values, strict=True)),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=True,
                    ).encode("utf-8")
                )
            row_count += 1
    digest.update(b"]")
    row_digest.update(b"]")
    contract = {
        "rows": row_count,
        "ordered_primary_key_sha256": digest.hexdigest(),
        "schema_sha256": canonical_json_sha256(
            [
                (field.name, str(field.type), field.nullable)
                for field in parquet.schema_arrow
            ]
        ),
    }
    if include_all_columns:
        contract["ordered_logical_rows_sha256"] = row_digest.hexdigest()
    return contract


def ordered_key_contract_for_paths(
    paths: Sequence[Path],
    *,
    key_columns: Sequence[str],
    require_sample_grouping: bool,
    include_all_columns: bool = False,
) -> dict[str, Any]:
    """Audit the exact logical concatenation of several Parquet fragments."""

    digest = hashlib.sha256()
    digest.update(b"[")
    row_digest = hashlib.sha256()
    row_digest.update(b"[")
    row_count = 0
    first = True
    current_sample: str | None = None
    current_candidates: set[tuple[str, ...]] = set()
    closed_samples: set[str] = set()
    schema_sha256: str | None = None
    for path in paths:
        parquet = pq.ParquetFile(path)
        current_schema_sha256 = canonical_json_sha256(
            [
                (field.name, str(field.type), field.nullable)
                for field in parquet.schema_arrow
            ]
        )
        if schema_sha256 is None:
            schema_sha256 = current_schema_sha256
        elif current_schema_sha256 != schema_sha256:
            raise ValueError("source Parquet schemas differ")
        missing = sorted(set(key_columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(
                f"Parquet key columns are missing in {path}: {missing}"
            )
        selected_columns = (
            list(parquet.schema_arrow.names)
            if include_all_columns
            else list(key_columns)
        )
        key_indices = [selected_columns.index(name) for name in key_columns]
        for batch in parquet.iter_batches(
            columns=selected_columns, batch_size=50_000
        ):
            columns = [
                batch.column(index).to_pylist()
                for index in range(batch.num_columns)
            ]
            for raw_values in zip(*columns, strict=True):
                raw_key = tuple(raw_values[index] for index in key_indices)
                if any(value is None for value in raw_key):
                    raise ValueError(f"null Parquet primary key in {path}")
                key = tuple(map(str, raw_key))
                sample_id = key[0]
                if require_sample_grouping:
                    if current_sample != sample_id:
                        if current_sample is not None:
                            closed_samples.add(current_sample)
                        if sample_id in closed_samples:
                            raise ValueError(
                                "sample rows are not contiguous across source "
                                f"Parquets: {sample_id}"
                            )
                        current_sample = sample_id
                        current_candidates = set()
                    candidate_key = key[1:]
                    if candidate_key in current_candidates:
                        raise ValueError(
                            "duplicate primary key across source Parquets: "
                            f"{key}"
                        )
                    current_candidates.add(candidate_key)
                if not first:
                    digest.update(b",")
                first = False
                digest.update(
                    json.dumps(
                        key,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                )
                if include_all_columns:
                    if row_count:
                        row_digest.update(b",")
                    row_digest.update(
                        json.dumps(
                            dict(
                                zip(
                                    selected_columns,
                                    raw_values,
                                    strict=True,
                                )
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=True,
                        ).encode("utf-8")
                    )
                row_count += 1
    digest.update(b"]")
    row_digest.update(b"]")
    if schema_sha256 is None:
        raise ValueError("no source Parquet paths")
    contract = {
        "rows": row_count,
        "ordered_primary_key_sha256": digest.hexdigest(),
        "schema_sha256": schema_sha256,
    }
    if include_all_columns:
        contract["ordered_logical_rows_sha256"] = row_digest.hexdigest()
    return contract


def parquet_key_set_contract(
    paths: Sequence[Path],
    *,
    key_columns: Sequence[str],
    include_all_columns: bool = False,
) -> dict[str, Any]:
    """Return an exact order-independent PK digest, rejecting duplicates."""

    keys: set[tuple[str, ...]] = set()
    row_hashes: list[bytes] = []
    schema_sha256_values: set[str] = set()
    for path in paths:
        parquet = pq.ParquetFile(path)
        schema_sha256_values.add(
            canonical_json_sha256(
                [
                    (field.name, str(field.type), field.nullable)
                    for field in parquet.schema_arrow
                ]
            )
        )
        missing = sorted(set(key_columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(
                f"Parquet key columns are missing in {path}: {missing}"
            )
        selected_columns = (
            list(parquet.schema_arrow.names)
            if include_all_columns
            else list(key_columns)
        )
        key_indices = [selected_columns.index(name) for name in key_columns]
        for batch in parquet.iter_batches(
            columns=selected_columns, batch_size=50_000
        ):
            columns = [
                batch.column(index).to_pylist()
                for index in range(batch.num_columns)
            ]
            for raw_values in zip(*columns, strict=True):
                raw_key = tuple(raw_values[index] for index in key_indices)
                if any(value is None for value in raw_key):
                    raise ValueError(f"null Parquet primary key in {path}")
                key = tuple(map(str, raw_key))
                if key in keys:
                    raise ValueError(
                        f"duplicate Parquet primary key across inputs: {key}"
                    )
                keys.add(key)
                if include_all_columns:
                    row_hashes.append(
                        hashlib.sha256(
                            json.dumps(
                                dict(
                                    zip(
                                        selected_columns,
                                        raw_values,
                                        strict=True,
                                    )
                                ),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=True,
                            ).encode("utf-8")
                        ).digest()
                    )
    if len(schema_sha256_values) != 1:
        raise ValueError("source Parquet schemas differ")
    contract = {
        "rows": len(keys),
        "sorted_primary_key_sha256": canonical_json_sha256(sorted(keys)),
        "schema_sha256": next(iter(schema_sha256_values)),
    }
    if include_all_columns:
        rows_digest = hashlib.sha256()
        for row_hash in sorted(row_hashes):
            rows_digest.update(row_hash)
        contract["sorted_logical_rows_sha256"] = rows_digest.hexdigest()
    return contract


def validate_exact_release_file(
    path: Path, *, expected_sha256: str, tmp_root: Path
) -> None:
    resolved = path.resolve()
    configured_tmp = (protected_run_root(resolved) / "tmp").resolve()
    if (
        not is_below(resolved, configured_tmp)
        or not is_below(resolved, tmp_root.resolve())
        or resolved.suffix != ".parquet"
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError(
            f"refusing to release non-compact or out-of-scope input: {resolved}"
        )
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"release input hash changed: {resolved}")


def release_verified_parquets(
    *,
    paths_and_hashes: Sequence[tuple[Path, str]],
    tmp_root: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
    stage_name: str,
) -> None:
    """Unlink exact files one at a time, journaling interruption-safe progress."""

    released = set(map(str, receipt.get("released_inputs", [])))
    for path, expected_sha256 in paths_and_hashes:
        if path.is_symlink():
            raise ValueError(f"refusing to release symlinked input: {path}")
        resolved = path.resolve()
        value = str(resolved)
        if not path.exists():
            if value not in released:
                # The stage-level VERIFIED_READY_TO_RELEASE journal is durable
                # before this function starts.  A missing file can therefore
                # only be accepted when that stage still binds its exact hash.
                stage = receipt.get("stages", {}).get(stage_name, {})
                expected = {
                    str(item["path"]): str(item["sha256"])
                    for item in stage.get("inputs", [])
                }
                if expected.get(value) != expected_sha256:
                    raise ValueError(
                        f"untracked missing release input: {resolved}"
                    )
                released.add(value)
                receipt["released_inputs"] = sorted(released)
                durable_atomic_json(
                    receipt_path, receipt, tmp_root=tmp_root
                )
            continue
        validate_exact_release_file(
            resolved,
            expected_sha256=expected_sha256,
            tmp_root=tmp_root,
        )
        resolved.unlink()
        fsync_directory(resolved.parent)
        released.add(value)
        receipt["released_inputs"] = sorted(released)
        durable_atomic_json(receipt_path, receipt, tmp_root=tmp_root)
        fault_point(f"{stage_name}:after_unlink")


def fault_point(_name: str) -> None:
    """No-op hook monkeypatched by interruption-recovery tests."""
