"""Immutable authorization, claim, and event contracts for K-sensitivity."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256

from .execution import (
    AUTHORIZATION_START_WINDOW_SECONDS,
    CANONICAL_RANK1_RUN_DIR,
    GATE_CLOCK_TOLERANCE_SECONDS,
    LEASE_HEARTBEAT_MAX_AGE_SECONDS,
    artifact_record,
    load_content_manifest,
    validate_resource_gate,
)
from .resource_gate import host_contract, resource_thresholds


K_EXECUTION_ID_ENV = "D1_K_SENSITIVITY_EXECUTION_ID"
K_CLAIM_SHA256_ENV = "D1_K_SENSITIVITY_CLAIM_SHA256"
K_EXECUTION_POINTER_RELATIVE = Path("configs/d1_k_sensitivity_execution.json")
K_EXECUTION_REGISTRY_RELATIVE = Path("configs/k_sensitivity_executions")
K_EXECUTION_SCOPE = {
    "route": "D1",
    "stage": "P10",
    "analysis": "K_sensitivity",
    "device": "cpu",
    "max_parallel": 1,
    "job_count": 54,
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def execution_source_records(
    *,
    plan_path: Path,
    resource_gate_path: Path,
    resume_from: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Return the exact code/data authority closure for one execution."""

    records = {
        "plan": artifact_record(plan_path),
        "resource_gate": artifact_record(resource_gate_path),
        "contract": artifact_record(Path(__file__)),
        "runner": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/run_k_sensitivity_cell.py"
        ),
        "authorizer": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/authorize_k_sensitivity_execution.py"
        ),
        "orchestrator": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/run_k_sensitivity_matrix.py"
        ),
    }
    if resume_from is not None:
        records["resume_from"] = {
            "path": str(resume_from["path"]),
            "sha256": str(resume_from["sha256"]),
        }
    return records


def k_sensitivity_resource_policy(
    *, plan_path: Path, source_paths: Sequence[Path]
) -> dict[str, Any]:
    """Freeze the K-specific 3x5-minute CPU resource-gate policy."""

    from .k_sensitivity import load_k_sensitivity_plan

    plan = load_k_sensitivity_plan(plan_path)
    if plan.get("job_count") != 54:
        raise RuntimeError("D1 K resource policy requires the exact 54-job plan")
    sources = [artifact_record(path) for path in source_paths]
    if not sources:
        raise ValueError("D1 K resource policy requires source records")
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY",
        "gate_type": "d1_k_sensitivity_cpu_three_continuous_five_minute_windows_v1",
        "scope": K_EXECUTION_SCOPE,
        "rank1_run_dir": str(CANONICAL_RANK1_RUN_DIR),
        "plan": artifact_record(plan_path),
        "thresholds": resource_thresholds(),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "sources": sources,
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def execution_directory(root: Path, execution_id: str) -> Path:
    if not execution_id or any(
        character not in "0123456789abcdef" for character in execution_id
    ):
        raise RuntimeError("D1 K execution identifier is invalid")
    return root / K_EXECUTION_REGISTRY_RELATIVE / execution_id


def _current_process_ancestors() -> set[int]:
    ancestors: set[int] = set()
    cursor = os.getpid()
    for _ in range(64):
        if cursor <= 1 or cursor in ancestors:
            break
        ancestors.add(cursor)
        result = subprocess.run(
            ["/bin/ps", "-o", "ppid=", "-p", str(cursor)],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            cursor = int(result.stdout.strip())
        except ValueError:
            break
    return ancestors


def exclusive_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Create one immutable JSON record with kernel-enforced O_EXCL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
    return path


def create_job_claim(
    root: Path,
    *,
    execution: Mapping[str, Any],
    job: Mapping[str, Any],
    owner_pid: int,
    command: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    """Exclusively claim one planned job for one orchestrator child."""

    execution_id = str(execution.get("execution_id", ""))
    job_id = str(job.get("job_id", ""))
    configuration = _mapping(
        job.get("configuration"), name="D1 K claimed job configuration"
    )
    if canonical_sha256(configuration)[:16] != job_id:
        raise RuntimeError("D1 K claimed job differs from its configuration")
    if (
        owner_pid <= 1
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise RuntimeError("D1 K job claim owner/command is invalid")
    claim: dict[str, Any] = {
        "schema_version": 1,
        "status": "CLAIMED",
        "execution_id": execution_id,
        "job_id": job_id,
        "configuration_sha256": canonical_sha256(configuration),
        "owner_pid": owner_pid,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    claim["content_sha256"] = canonical_sha256(claim)
    path = execution_directory(root, execution_id) / "claims" / f"{job_id}.json"
    exclusive_json(path, claim)
    return path, claim


def write_execution_event(
    root: Path,
    *,
    execution: Mapping[str, Any],
    sequence: int,
    status: str,
    owner_pid: int | None,
    current_job_id: str | None,
    claim: Mapping[str, Any] | None,
    outputs: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    failure: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Append an immutable event, then atomically advance the fixed pointer."""

    if status not in {"ACTIVE", "COMPLETE", "FAILED"} or sequence < 0:
        raise ValueError("D1 K execution event state is invalid")
    execution_id = str(execution.get("execution_id", ""))
    output_map = {str(key): value for key, value in outputs.items()}
    event: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "execution_id": execution_id,
        "sequence": sequence,
        "owner_pid": owner_pid,
        "heartbeat_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_job_id": current_job_id,
        "claim": None if claim is None else dict(claim),
        "completed_jobs": len(output_map),
        "expected_jobs": 54,
        "outputs": output_map,
        "commands": [dict(command) for command in commands],
        "failure": failure,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    event["content_sha256"] = canonical_sha256(event)
    event_path = (
        execution_directory(root, execution_id)
        / "events"
        / f"{sequence:04d}_{status.lower()}.json"
    )
    exclusive_json(event_path, event)
    execution_path = execution_directory(root, execution_id) / "execution.json"
    pointer: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "execution_id": execution_id,
        "execution": artifact_record(execution_path),
        "latest_event": artifact_record(event_path),
        "completed_jobs": len(output_map),
        "expected_jobs": 54,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(root / K_EXECUTION_POINTER_RELATIVE, pointer)
    return event_path, event


def validate_active_execution(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    require_fresh_gate: bool,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Validate the fixed ACTIVE pointer and immutable execution authority."""

    pointer_path = root / K_EXECUTION_POINTER_RELATIVE
    pointer = load_content_manifest(
        pointer_path, name="D1 K execution pointer", statuses=("ACTIVE",)
    )
    execution_id = str(pointer.get("execution_id", ""))
    expected_directory = execution_directory(root, execution_id)
    execution_path = verified_artifact_path(
        _mapping(pointer.get("execution"), name="D1 K execution record"),
        name="D1 K execution authority",
    )
    if execution_path != expected_directory / "execution.json":
        raise RuntimeError("D1 K execution authority path differs")
    execution = load_content_manifest(
        execution_path, name="D1 K execution authority", statuses=("ACTIVE",)
    )
    gate_record = _mapping(
        execution.get("resource_gate"), name="D1 K execution resource gate"
    )
    gate_path = verified_artifact_path(gate_record, name="D1 K resource gate")
    resume_from = execution.get("resume_from")
    expected_sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=gate_path,
        resume_from=(
            _mapping(resume_from, name="D1 K resume source")
            if resume_from is not None
            else None
        ),
    )
    if (
        execution.get("execution_id") != execution_id
        or execution.get("run_dir") != str(root)
        or execution.get("rank1_run_dir") != str(CANONICAL_RANK1_RUN_DIR)
        or execution.get("scope") != K_EXECUTION_SCOPE
        or execution.get("plan") != artifact_record(plan_path)
        or execution.get("job_count") != 54
        or execution.get("job_ids_sha256") != plan.get("job_ids_sha256")
        or execution.get("candidate_test_labels_read") is not False
        or execution.get("test_inputs_referenced") is not False
        or execution.get("direct_cli_execution_permitted") is not False
        or execution.get("host") != host_contract()
        or execution.get("sources") != expected_sources
        or execution.get("source_signature_sha256")
        != canonical_sha256(expected_sources)
    ):
        raise RuntimeError("D1 K execution authority contract differs")
    verify_artifact_records_recursive(
        expected_sources, name="D1 K execution sources", require_at_least_one=True
    )
    validate_resource_gate(
        gate_path,
        run_dir=root,
        plan_path=plan_path,
        expected_scope=K_EXECUTION_SCOPE,
        require_fresh=require_fresh_gate,
    )
    if require_fresh_gate:
        current = datetime.now(timezone.utc)
        expires = datetime.fromisoformat(str(execution.get("expires_at_utc")))
        if current > expires:
            raise RuntimeError("D1 K execution authority expired before start/resume")
    event_path = verified_artifact_path(
        _mapping(pointer.get("latest_event"), name="D1 K latest event record"),
        name="D1 K latest execution event",
    )
    event = load_content_manifest(
        event_path, name="D1 K latest execution event", statuses=("ACTIVE",)
    )
    if (
        event_path.parent != expected_directory / "events"
        or event.get("execution_id") != execution_id
        or pointer.get("completed_jobs") != event.get("completed_jobs")
        or pointer.get("expected_jobs") != 54
    ):
        raise RuntimeError("D1 K ACTIVE event/pointer binding differs")
    return execution_path, execution, event_path, event


def validate_worker_context(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    """Prove the cell is a claimed child of the active lease-holding orchestrator."""

    execution_path, execution, event_path, event = validate_active_execution(
        root,
        plan_path=plan_path,
        plan=plan,
        require_fresh_gate=False,
    )
    execution_id = str(execution["execution_id"])
    job_id = str(job.get("job_id", ""))
    claim_record = _mapping(event.get("claim"), name="D1 K active job claim")
    claim_path = verified_artifact_path(claim_record, name="D1 K active job claim")
    expected_claim_path = (
        execution_directory(root, execution_id) / "claims" / f"{job_id}.json"
    )
    claim = load_content_manifest(
        claim_path, name="D1 K active job claim", statuses=("CLAIMED",)
    )
    owner_pid = int(event.get("owner_pid", -1))
    current = datetime.now(timezone.utc)
    heartbeat = datetime.fromisoformat(str(event.get("heartbeat_at_utc")))
    heartbeat_age = (current - heartbeat).total_seconds()
    configuration = _mapping(
        job.get("configuration"), name="D1 K worker job configuration"
    )
    if (
        event.get("current_job_id") != job_id
        or claim_path != expected_claim_path
        or claim.get("execution_id") != execution_id
        or claim.get("job_id") != job_id
        or claim.get("configuration_sha256") != canonical_sha256(configuration)
        or claim.get("owner_pid") != owner_pid
        or os.environ.get(K_EXECUTION_ID_ENV) != execution_id
        or os.environ.get(K_CLAIM_SHA256_ENV) != claim_record.get("sha256")
        or owner_pid <= 1
    ):
        raise RuntimeError("D1 K worker claim/execution identity differs")
    if heartbeat_age < -GATE_CLOCK_TOLERANCE_SECONDS:
        raise RuntimeError("D1 K execution heartbeat is in the future")
    if heartbeat_age > LEASE_HEARTBEAT_MAX_AGE_SECONDS:
        raise RuntimeError("D1 K execution heartbeat is stale")
    try:
        os.kill(owner_pid, 0)
    except OSError as error:
        raise RuntimeError("D1 K orchestrator owner is not alive") from error
    if owner_pid not in _current_process_ancestors():
        raise RuntimeError("D1 K orchestrator owner is not an ancestor of this cell")
    return execution_path, execution, event_path, event, claim_path, claim


def authorization_expiry(now: datetime) -> str:
    return (now + timedelta(seconds=AUTHORIZATION_START_WINDOW_SECONDS)).isoformat()


__all__ = [
    "K_CLAIM_SHA256_ENV",
    "K_EXECUTION_ID_ENV",
    "K_EXECUTION_POINTER_RELATIVE",
    "K_EXECUTION_REGISTRY_RELATIVE",
    "K_EXECUTION_SCOPE",
    "authorization_expiry",
    "create_job_claim",
    "execution_directory",
    "execution_source_records",
    "exclusive_json",
    "k_sensitivity_resource_policy",
    "validate_active_execution",
    "validate_worker_context",
    "write_execution_event",
]
