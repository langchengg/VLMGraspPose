"""Immutable GO locks and formal paid-run authorization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .io import atomic_json, sha256_file, utc_now
from .constants import EXACT_MODEL_IDS


def lock_path(run_dir: str | Path, backend: str) -> Path:
    if backend not in {"G1", "C1"}:
        raise ValueError("backend must be G1 or C1")
    return Path(run_dir) / f"LOCKED_MANIFEST_{backend}.json"


def create_go_lock(run_dir: str | Path, backend: str, payload: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    if payload.get("validation_decision") != "GO":
        raise ValueError("only an exact validation GO may be locked")
    primary = str(payload.get("locked_primary", ""))
    expected_models: list[str]
    if "CONSENSUS" in primary:
        expected_models = list(EXACT_MODEL_IDS)
    elif "robotics-er-2" in primary:
        expected_models = [EXACT_MODEL_IDS[0]]
    elif "3.6-flash" in primary:
        expected_models = [EXACT_MODEL_IDS[1]]
    else:
        raise ValueError("GO lock primary is not an exact Gemini method")
    if list(payload.get("required_models", [])) != expected_models:
        raise ValueError("GO lock required_models do not match locked_primary")
    availability = dict(payload.get("model_availability", {}))
    if not all(availability.get(model) is True for model in expected_models):
        raise ValueError("GO lock requires an unavailable exact model")
    metric = dict(payload.get("validation_metric", {}))
    if metric.get("method") != primary or metric.get("validation_decision") != "GO":
        raise ValueError("GO lock lacks the matching untouched-validation GO metric")
    path = lock_path(run_dir, backend)
    if path.exists():
        raise FileExistsError("experiment lock is immutable and already exists")
    value = {"schema_version": 1, "backend": backend, "locked_at_utc": utc_now(), **dict(payload)}
    if not dry_run:
        atomic_json(path, value)
        value["lock_sha256"] = sha256_file(path)
    return value


def verify_go_lock(run_dir: str | Path, backend: str) -> dict[str, Any]:
    run = Path(run_dir)
    path = lock_path(run, backend)
    if not path.is_file():
        raise FileNotFoundError("locked manifest is absent")
    value = json.loads(path.read_text())
    if value.get("backend") != backend or value.get("validation_decision") != "GO":
        raise ValueError("invalid GO lock")
    required_hashes = {
        "development_policy_lock_sha256": run / "DEVELOPMENT_POLICY_LOCK.json",
        "candidate_manifest_sha256": run / f"CANDIDATE_MANIFEST_{backend}.parquet",
        "validation_results_sha256": run / "VALIDATION_RESULTS.json",
    }
    for field, target in required_hashes.items():
        if not target.is_file() or value.get(field) != sha256_file(target):
            raise ValueError(f"formal lock dependency hash mismatch: {field}")
    return value


def require_formal_authorization(run_dir: str | Path, backend: str, *, allow_formal_argument: bool) -> dict[str, Any]:
    lock = verify_go_lock(run_dir, backend)
    if not allow_formal_argument:
        raise PermissionError("--allow-formal is required")
    if os.environ.get("ALLOW_FORMAL_GEMINI_RERANK") != "1":
        raise PermissionError("ALLOW_FORMAL_GEMINI_RERANK=1 is required")
    if os.environ.get("ALLOW_PAID_API_RUN") != "1":
        raise PermissionError("ALLOW_PAID_API_RUN=1 is required")
    return lock
