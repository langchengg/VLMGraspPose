"""Deterministic hashing and atomic artifact writers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"expected a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_text(path: str | os.PathLike[str], value: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def atomic_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> Path:
    return atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
    )
