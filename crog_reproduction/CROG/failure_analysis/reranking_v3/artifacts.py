from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from . import SCHEMA_VERSION
from .schema import artifact_identity, atomic_write_json, atomic_write_text, canonical_json, sha256_bytes, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "failure_analysis" / "reranking_outputs"
CODE_ROOT = REPO_ROOT / "failure_analysis" / "reranking_v3"


def assert_v3_output_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise PermissionError(f"V3 output must be below {OUTPUT_ROOT}: {resolved}") from exc
    if not relative.parts or not relative.parts[0].startswith("v3_fullchain_"):
        raise PermissionError(f"V3 output must use a v3_fullchain_<run_id> root: {resolved}")
    return resolved


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    roots = (CODE_ROOT, REPO_ROOT / "model", REPO_ROOT / "utils", REPO_ROOT / "failure_analysis" / "reranking_v2")
    for path in sorted({p.resolve() for root in roots for p in root.rglob("*.py") if p.is_file()}):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_metadata(device: str) -> dict[str, Any]:
    import cv2
    import scipy
    import skimage

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def create_run_dir(output_dir: str | Path, *, force_new_run: bool) -> Path:
    output = assert_v3_output_path(output_dir)
    if output.exists():
        raise FileExistsError(f"V3 run directory already exists and is immutable: {output}")
    if not force_new_run:
        raise PermissionError("creating a V3 run requires --force-new-run")
    output.mkdir(parents=True, exist_ok=False)
    return output


def append_command_log(run_dir: str | Path, argv: list[str]) -> None:
    run = assert_v3_output_path(run_dir)
    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cwd": str(Path.cwd().resolve()),
        "argv": list(argv),
        "pid": os.getpid(),
    }
    path = run / "commands.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_environment(run_dir: str | Path, *, device: str) -> Path:
    run = assert_v3_output_path(run_dir)
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": "environment",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": git_head,
        "code_fingerprint": code_fingerprint(),
        "runtime": runtime_metadata(device),
    }
    return atomic_write_json(run / "environment.json", value)


def write_sha_sidecar(path: str | Path) -> Path:
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    return atomic_write_text(sidecar, f"{sha256_file(source)}  {source.name}\n")


def build_artifact_manifest(
    *,
    artifact_type: str,
    status: str,
    parents: Iterable[str | Path],
    candidate_source: str | Path,
    checkpoint: str | Path,
    split_manifest: str | Path,
    evaluator: str | Path,
    config: dict[str, Any],
    feature_schema_hash: str,
    seed: int,
    device: str,
    dtype: str,
    row_count: int,
    unique_sample_count: int,
    unique_candidate_count: int,
    missing_count: int = 0,
    fallback_count: int = 0,
    outputs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    output_identities = [artifact_identity(path) for path in outputs]
    payload = {
        "artifact_type": artifact_type,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "parent_hashes": [artifact_identity(path) for path in parents],
        "candidate_source": artifact_identity(candidate_source),
        "crog_checkpoint": artifact_identity(checkpoint),
        "split_manifest": artifact_identity(split_manifest),
        "evaluator": artifact_identity(evaluator),
        "code_fingerprint": code_fingerprint(),
        "config_hash": sha256_bytes(canonical_json(config).encode()),
        "feature_schema_hash": feature_schema_hash,
        "seed": int(seed),
        "device": device,
        "dtype": dtype,
        "row_count": int(row_count),
        "unique_sample_count": int(unique_sample_count),
        "unique_candidate_count": int(unique_candidate_count),
        "missing_count": int(missing_count),
        "fallback_count": int(fallback_count),
        "creation_time": now,
        "completion_time": now if status == "complete" else None,
        "outputs": output_identities,
    }
    payload["content_sha256"] = sha256_bytes(canonical_json(payload).encode())
    return payload

