"""Atomic serialization helpers for resumable SAM 3 CPU runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .sam3_serialization import (
    assert_no_ground_truth_leakage,
    file_manifest,
    save_strict_json,
    sha256_file,
    verify_file_manifest,
    write_jsonl,
)

__all__ = [
    "assert_no_ground_truth_leakage",
    "atomic_output_directory",
    "atomic_write_json",
    "atomic_write_jsonl",
    "file_manifest",
    "save_strict_json",
    "sha256_file",
    "verify_file_manifest",
    "write_jsonl",
]


@contextmanager
def atomic_output_directory(destination: Path | str) -> Iterator[Path]:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"stale incomplete output requires review: {temporary}")
    temporary.mkdir()
    try:
        yield temporary
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def atomic_write_json(path: Path | str, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def atomic_write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(row, sort_keys=True, allow_nan=False) for row in rows
    )
    if payload:
        payload += "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path
