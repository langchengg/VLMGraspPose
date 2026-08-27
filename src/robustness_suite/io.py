"""Small provenance and atomic-output helpers for the robustness suite."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    return Path(name)


def atomic_text(destination: str | Path, value: str) -> None:
    target = Path(destination)
    temporary = _temporary(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(destination: str | Path, value: Any) -> None:
    atomic_text(
        destination,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def atomic_csv(destination: str | Path, frame: pd.DataFrame) -> None:
    target = Path(destination)
    temporary = _temporary(target)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_parquet(destination: str | Path, frame: pd.DataFrame) -> None:
    target = Path(destination)
    temporary = _temporary(target)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(run_dir: str | Path, **changes: Any) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = read_manifest(root)
    manifest.update(changes)
    atomic_json(root / "run_manifest.json", manifest)
    return manifest


def update_progress(run_dir: str | Path, stage: str, status: str) -> None:
    root = Path(run_dir)
    manifest = read_manifest(root)
    progress = dict(manifest.get("progress", {}))
    progress[str(stage)] = str(status)
    manifest["progress"] = progress
    manifest["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(root / "run_manifest.json", manifest)


def record_command(run_dir: str | Path, command: list[str]) -> None:
    root = Path(run_dir)
    manifest = read_manifest(root)
    commands = list(manifest.get("commands", []))
    commands.append(
        {
            "argv": [str(item) for item in command],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    manifest["commands"] = commands
    atomic_json(root / "run_manifest.json", manifest)


__all__ = [
    "atomic_csv",
    "atomic_json",
    "atomic_parquet",
    "atomic_text",
    "read_manifest",
    "record_command",
    "sha256_file",
    "update_manifest",
    "update_progress",
]
