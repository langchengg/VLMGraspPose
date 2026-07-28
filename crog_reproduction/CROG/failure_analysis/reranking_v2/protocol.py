from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .schema import (
    artifact_identity,
    atomic_write_json,
    atomic_write_text,
    code_fingerprint,
    sha256_bytes,
    sha256_file,
)


def split_ids(
    split_manifest_path: str | Path, partition: str
) -> set[str]:
    payload = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    return {
        str(row["sample_id"])
        for row in payload["rows"]
        if row["development_partition"] == partition
    }


def fold_lookup(split_manifest_path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    return {
        str(row["sample_id"]): int(row["oof_fold"])
        for row in payload["rows"]
        if row["development_partition"] == "train"
    }


def lock_experiment(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    primary_method: str,
    exploratory_methods: list[str],
    split_manifest: str | Path,
    checkpoint: str | Path,
    configs: dict[str, Any],
    seeds: list[int],
    thresholds: dict[str, Any],
    feature_lists: dict[str, list[str]],
    evaluator_source: str | Path,
    model_artifacts: list[str | Path],
    test_command: str,
    aggregate_test_exposure_disclosure: str,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            "frozen_experiment_manifest.json is immutable and already exists"
        )
    payload = {
        "schema_version": "2.0.0",
        "kind": "frozen_experiment_manifest",
        "status": "locked",
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "primary_method": str(primary_method),
        "exploratory_methods": list(exploratory_methods),
        "split_manifest": artifact_identity(split_manifest),
        "checkpoint": artifact_identity(checkpoint),
        "configs": configs,
        "seeds": [int(seed) for seed in seeds],
        "thresholds": thresholds,
        "feature_lists": feature_lists,
        "evaluator": artifact_identity(evaluator_source),
        "code_sha256": code_fingerprint(repo_root),
        "model_artifacts": [
            artifact_identity(path) for path in model_artifacts
        ],
        "test_command": str(test_command),
        "disclosure": aggregate_test_exposure_disclosure,
    }
    payload["lock_sha256"] = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    atomic_write_json(output_path, payload)
    sha_path = output_path.with_suffix(output_path.suffix + ".sha256")
    atomic_write_text(
        sha_path,
        f"{sha256_file(output_path)}  {output_path.name}\n",
    )
    return payload


def verify_lock(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    lock_sha = payload.pop("lock_sha256")
    observed = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    payload["lock_sha256"] = lock_sha
    if observed != lock_sha:
        raise ValueError("frozen experiment manifest content hash is invalid")
    for name in ("split_manifest", "checkpoint", "evaluator"):
        identity = payload[name]
        if sha256_file(identity["path"]) != identity["sha256"]:
            raise ValueError(f"locked {name} hash changed")
    for identity in payload["model_artifacts"]:
        if sha256_file(identity["path"]) != identity["sha256"]:
            raise ValueError(f"locked model artifact changed: {identity['path']}")
    if code_fingerprint(repo_root) != payload["code_sha256"]:
        raise ValueError("code fingerprint changed after experiment lock")
    return payload


def claim_test_once(
    test_run_dir: str | Path,
    lock_path: str | Path,
    *,
    resume: bool = False,
) -> Path:
    test_run_dir = Path(test_run_dir)
    claim = test_run_dir / "TEST_RUN_CLAIM.json"
    if claim.exists():
        if resume:
            existing = json.loads(claim.read_text(encoding="utf-8"))
            if existing["lock_file_sha256"] != sha256_file(lock_path):
                raise ValueError("formal test resume uses a different lock")
            if (test_run_dir / "TEST_RUN_COMPLETE.json").exists():
                raise FileExistsError("formal V2 test is already complete")
            return claim
        raise FileExistsError("formal V2 test has already been claimed")
    test_run_dir.mkdir(parents=True, exist_ok=True)
    value = {
        "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lock_path": str(Path(lock_path).resolve()),
        "lock_file_sha256": sha256_file(lock_path),
        "pid": os.getpid(),
    }
    atomic_write_json(claim, value)
    return claim
