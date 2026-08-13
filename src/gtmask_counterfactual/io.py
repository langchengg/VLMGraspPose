"""Durable atomic I/O specialised for the counterfactual run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from d1_reranking.io import atomic_copy
from unified_reranking.hashing import canonical_sha256, sha256_file

__all__ = [
    "artifact_record",
    "atomic_copy",
    "atomic_csv",
    "atomic_json",
    "atomic_parquet",
    "atomic_text",
    "canonical_sha256",
    "exclusive_json",
    "exclusive_text",
    "sha256_file",
]


def _fsync_parent(destination: Path) -> None:
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_text(path: str | Path, value: str) -> Path:
    """Write, fsync, and atomically replace one UTF-8 artifact."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return atomic_text(
        path,
        json.dumps(
            dict(value), indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n",
    )


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Durably publish one CSV through a same-directory temporary file."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Durably publish one Parquet artifact without exposing partial bytes."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            frame.to_parquet(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def exclusive_text(path: str | Path, value: str) -> Path:
    """Publish an immutable claim with kernel-enforced exactly-once semantics."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_parent(destination)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    return destination


def exclusive_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return exclusive_text(
        path,
        json.dumps(
            dict(value), indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n",
    )


def artifact_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }
