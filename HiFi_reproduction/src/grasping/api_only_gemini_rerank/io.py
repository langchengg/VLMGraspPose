"""Atomic local artifact helpers; never serialize credentials."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, target)


def atomic_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)


def initialize_run(path: str | Path) -> Path:
    run = Path(path).expanduser().resolve()
    for relative in (
        "prompts", "boards/smoke", "boards/diagnostic", "boards/validation",
        "boards/formal", "raw_api/er2", "raw_api/flash", "parsed_api",
        "plots", "failure_gallery/recovered", "failure_gallery/harmful",
        "failure_gallery/neutral_correct", "failure_gallery/neutral_wrong",
        "failure_gallery/missed_recoverable", "failure_gallery/unrecoverable",
        "failure_gallery/disagreement", "failure_gallery/unstable",
        "failure_gallery/api_failure", "audit", "data", "logs",
        "request_manifests", "stage_results",
    ):
        (run / relative).mkdir(parents=True, exist_ok=True)
    (run / ".DO_NOT_PRUNE").touch(exist_ok=True)
    return run
