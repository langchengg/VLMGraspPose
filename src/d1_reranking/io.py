"""Small atomic I/O helpers for D1 artifacts."""

from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_parquet(frame: pd.DataFrame, destination: str | Path) -> Path:
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return path


def atomic_csv(frame: pd.DataFrame, destination: str | Path) -> Path:
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def atomic_copy(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(
            f"copy source must be a regular non-symlink file: {source_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(
        f".{destination_path.name}.{os.getpid()}.tmp"
    )
    with source_path.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination_path)
    return destination_path


def atomic_pickle(value: Any, destination: str | Path) -> Path:
    """Persist a primitive experiment payload without a partial destination."""

    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def atomic_hardlink(source: str | Path, destination: str | Path) -> Path:
    """Create an immutable-in-place alias and reject divergent existing bytes."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"hardlink source must be a regular file: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if destination_path.is_symlink():
            raise RuntimeError(
                f"hardlink destination must not be a symlink: {destination_path}"
            )
        from unified_reranking.hashing import sha256_file

        if sha256_file(destination_path) != sha256_file(source_path):
            raise RuntimeError(f"existing hardlink alias differs: {destination_path}")
        return destination_path
    temporary = destination_path.with_name(
        f".{destination_path.name}.{os.getpid()}.tmp"
    )
    os.link(source_path, temporary)
    os.replace(temporary, destination_path)
    return destination_path
