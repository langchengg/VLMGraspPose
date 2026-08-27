"""Small dependency-free integrity and atomic-I/O helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PREREGISTRATION_SHA256 = (
    "451c030eede0c1d04edb8a747065944698ae2111f2223d510b6a2dae2c6c9e22"
)
RUN_ID = "20260822_194405_remaining_robustness"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def require_run_dir(repo: Path, run_dir: Path) -> Path:
    repo = repo.resolve()
    run_dir = run_dir.resolve()
    allowed = (repo / "artifacts" / "robustness_completion").resolve()
    if run_dir.parent != allowed or run_dir.name != RUN_ID:
        raise ValueError(f"output must be the locked child run {allowed / RUN_ID}")
    if run_dir == repo or run_dir in {
        (repo / "artifacts" / "robustness_suite" / "20260822_085125_robustness_suite").resolve(),
        (repo / "artifacts" / "graspnet6d" / "20260819_221819_graspnet6d_vgn_lambdamart").resolve(),
    }:
        raise ValueError("source run cannot be used as output")
    return run_dir


def verify_preregistration(run_dir: Path) -> None:
    prereg = run_dir / "PRE_REGISTRATION.md"
    digest_file = run_dir / "PRE_REGISTRATION.sha256"
    if not prereg.is_file() or not digest_file.is_file():
        raise RuntimeError("locked preregistration is missing")
    actual = sha256_file(prereg)
    declared = digest_file.read_text(encoding="utf-8").split()[0]
    if actual != PREREGISTRATION_SHA256 or declared != actual:
        raise RuntimeError("PRE_REGISTRATION hash mismatch")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def atomic_frame(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        if suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError(f"unsupported frame suffix: {suffix}")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

