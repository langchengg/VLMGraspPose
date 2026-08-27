"""Deterministic hashing and durable atomic writes for the 6-DoF pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical UTF-8 representation used by all hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"expected a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_destination(path: str | os.PathLike[str]) -> tuple[Path, Path]:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{id(destination)}.tmp"
    )
    return destination, temporary


def _fsync_parent(destination: Path) -> None:
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_text(path: str | os.PathLike[str], value: str) -> Path:
    """Write and fsync a UTF-8 file before atomically publishing it."""

    destination, temporary = _prepare_destination(path)
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


def atomic_json(path: str | os.PathLike[str], value: Any) -> Path:
    return atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
    )


def atomic_jsonl(
    path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]
) -> Path:
    lines = [
        json.dumps(
            dict(row),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        for row in rows
    ]
    return atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def atomic_npz(path: str | os.PathLike[str], **arrays: Any) -> Path:
    """Atomically publish a compressed, pickle-free NumPy archive."""

    destination, temporary = _prepare_destination(path)
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
