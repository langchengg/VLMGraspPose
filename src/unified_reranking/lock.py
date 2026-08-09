"""Hash-verified formal-Test lock and single-execution claim."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .hashing import atomic_json, canonical_sha256, sha256_file


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_formal_test_lock(
    run_dir: str | Path,
    *,
    declaration: Mapping[str, Any],
    locked_files: Mapping[str, str | Path],
) -> Path:
    root = Path(run_dir).resolve()
    destination = root / "08_lock" / "FORMAL_TEST_LOCK.json"
    execution = root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    if destination.exists() or execution.exists():
        raise FileExistsError("formal Test lock/execution already exists")
    files = {}
    for name, raw_path in sorted(locked_files.items()):
        path = Path(raw_path).resolve()
        files[str(name)] = {"path": str(path), "sha256": sha256_file(path)}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED",
        "created_at_utc": _now(),
        "benchmark_description": "locked retrospective paired test benchmark",
        "formal_test_max_execution_count": 1,
        "declaration": dict(declaration),
        "locked_files": files,
    }
    payload["self_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return destination


def verify_formal_test_lock(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    lock = root / "08_lock" / "FORMAL_TEST_LOCK.json"
    if not lock.is_file():
        raise PermissionError("candidate-level test labels are locked before FORMAL_TEST_LOCK.json")
    payload = json.loads(lock.read_text(encoding="utf-8"))
    if payload.get("status") != "LOCKED":
        raise PermissionError("formal Test lock status is not LOCKED")
    recorded = payload.get("self_sha256")
    unsigned = dict(payload)
    unsigned.pop("self_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(unsigned):
        raise PermissionError("formal Test lock self-hash mismatch")
    for name, artifact in payload.get("locked_files", {}).items():
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise PermissionError(f"formal Test locked-file hash mismatch: {name}")
    return payload


def claim_formal_test_execution(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    lock = verify_formal_test_lock(root)
    execution_path = root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    if execution_path.exists():
        raise PermissionError("formal Test has already been claimed/executed")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("formal_test_execution_count", 0)) != 0:
        raise PermissionError("formal Test execution count is already non-zero")
    claim = {
        "schema_version": 1,
        "status": "RUNNING",
        "claimed_at_utc": _now(),
        "execution_count": 1,
        "formal_lock_file_sha256": sha256_file(root / "08_lock" / "FORMAL_TEST_LOCK.json"),
        "formal_lock_self_sha256": lock["self_sha256"],
    }
    atomic_json(execution_path, claim)
    manifest["formal_test_execution_count"] = 1
    manifest["status"] = "FORMAL_TEST_EXECUTION_CLAIMED"
    manifest["test_label_state"] = "FORMAL_TEST_EXECUTION_CLAIMED"
    atomic_json(manifest_path, manifest)
    return claim


def finalize_formal_test_execution(run_dir: str | Path, artifacts: Mapping[str, str | Path]) -> Path:
    root = Path(run_dir).resolve()
    execution_path = root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    if not execution_path.is_file():
        raise RuntimeError("formal Test execution was not claimed")
    payload = json.loads(execution_path.read_text(encoding="utf-8"))
    if payload.get("status") != "RUNNING" or payload.get("execution_count") != 1:
        raise RuntimeError("formal Test execution state is not a single RUNNING claim")
    payload["status"] = "COMPLETE"
    payload["completed_at_utc"] = _now()
    payload["artifacts"] = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(artifacts.items())
    }
    atomic_json(execution_path, payload)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "FORMAL_TEST_COMPLETE"
    manifest["test_label_state"] = "FORMAL_TEST_COMPLETE"
    atomic_json(manifest_path, manifest)
    return execution_path
