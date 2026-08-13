"""Immutable authorization, claims, and events for nine raw-feature jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256

from .execution import (
    AUTHORIZATION_START_WINDOW_SECONDS,
    CANONICAL_RANK1_RUN_DIR,
    FEATURE_RESOURCE_SCOPE,
    GATE_CLOCK_TOLERANCE_SECONDS,
    LEASE_HEARTBEAT_MAX_AGE_SECONDS,
    artifact_record,
    load_content_manifest,
    validate_feature_resource_gate,
)
from .resource_gate import host_contract


FEATURE_EXECUTION_ID_ENV = "D1_FEATURE_EXECUTION_ID"
FEATURE_CLAIM_SHA256_ENV = "D1_FEATURE_CLAIM_SHA256"
FEATURE_EXECUTION_POINTER_RELATIVE = Path("configs/d1_feature_execution.json")
FEATURE_EXECUTION_REGISTRY_RELATIVE = Path("configs/feature_executions")
FEATURE_JOB_COUNT = 9
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def execution_directory(root: Path, execution_id: str) -> Path:
    if not execution_id or any(
        character not in "0123456789abcdef" for character in execution_id
    ):
        raise RuntimeError("D1 feature execution identifier is invalid")
    return root / FEATURE_EXECUTION_REGISTRY_RELATIVE / execution_id


def execution_source_records(
    *,
    plan_path: Path,
    resource_gate_path: Path,
    resume_from: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records: dict[str, Any] = {
        "plan": artifact_record(plan_path),
        "resource_gate": artifact_record(resource_gate_path),
        "contract": artifact_record(Path(__file__)),
        "runner": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/extract_common_features.py"
        ),
        "authorizer": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/authorize_feature_extraction.py"
        ),
        "orchestrator": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/run_feature_extraction_matrix.py"
        ),
    }
    if resume_from is not None:
        records["resume_from"] = dict(resume_from)
    return records


def exclusive_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Create one immutable JSON record using O_EXCL and O_NOFOLLOW."""

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
    execution_id = str(execution.get("execution_id", ""))
    job_id = str(job.get("job_id", ""))
    configuration = _mapping(
        job.get("configuration"), name="D1 feature claimed job configuration"
    )
    if (
        canonical_sha256(configuration)[:16] != job_id
        or owner_pid <= 1
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise RuntimeError("D1 feature job claim differs from its planned job")
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
        "test_inputs_referenced": configuration.get("split") == "test",
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
    if status not in {"ACTIVE", "COMPLETE", "FAILED"} or sequence < 0:
        raise ValueError("D1 feature execution event state is invalid")
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
        "expected_jobs": FEATURE_JOB_COUNT,
        "outputs": output_map,
        "commands": [dict(command) for command in commands],
        "failure": failure,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
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
        "expected_jobs": FEATURE_JOB_COUNT,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(root / FEATURE_EXECUTION_POINTER_RELATIVE, pointer)
    return event_path, event


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


def _assert_repository_lease_is_held(root: Path) -> None:
    path = (root.parent / ".d1_heavy_resource.lock").resolve()
    descriptor = os.open(
        path,
        os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("D1 feature resource lease is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        raise RuntimeError("D1 feature orchestrator does not hold the global lease")
    finally:
        os.close(descriptor)


def validate_active_execution(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    require_fresh_gate: bool,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    pointer_path = root / FEATURE_EXECUTION_POINTER_RELATIVE
    pointer = load_content_manifest(
        pointer_path, name="D1 feature execution pointer", statuses=("ACTIVE",)
    )
    execution_id = str(pointer.get("execution_id", ""))
    directory = execution_directory(root, execution_id)
    execution_path = verified_artifact_path(
        _mapping(pointer.get("execution"), name="D1 feature execution record"),
        name="D1 feature execution authority",
    )
    if execution_path != directory / "execution.json":
        raise RuntimeError("D1 feature execution authority path differs")
    execution = load_content_manifest(
        execution_path, name="D1 feature execution authority", statuses=("ACTIVE",)
    )
    gate_path = verified_artifact_path(
        _mapping(execution.get("resource_gate"), name="D1 feature resource gate"),
        name="D1 feature resource gate",
    )
    resume_from = execution.get("resume_from")
    expected_sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=gate_path,
        resume_from=(
            _mapping(resume_from, name="D1 feature resume event")
            if resume_from is not None
            else None
        ),
    )
    if (
        execution.get("execution_id") != execution_id
        or execution.get("run_dir") != str(root)
        or execution.get("rank1_run_dir") != str(CANONICAL_RANK1_RUN_DIR)
        or execution.get("scope") != FEATURE_RESOURCE_SCOPE
        or execution.get("plan") != artifact_record(plan_path)
        or execution.get("job_count") != FEATURE_JOB_COUNT
        or execution.get("job_ids_sha256") != plan.get("job_ids_sha256")
        or execution.get("max_parallel") != 1
        or execution.get("device") != "cpu"
        or execution.get("direct_cli_execution_permitted") is not False
        or execution.get("candidate_test_labels_read") is not False
        or execution.get("host") != host_contract()
        or execution.get("sources") != expected_sources
        or execution.get("source_signature_sha256")
        != canonical_sha256(expected_sources)
    ):
        raise RuntimeError("D1 feature execution authority contract differs")
    verify_artifact_records_recursive(
        expected_sources, name="D1 feature execution sources", require_at_least_one=True
    )
    current_gate = validate_feature_resource_gate(
        root, require_fresh=require_fresh_gate
    )
    if artifact_record(gate_path) != artifact_record(
        verified_artifact_path(
            load_content_manifest(
                root / "00_audit/RESOURCE_AUDIT.json",
                name="D1 feature resource pointer",
                statuses=("PASS",),
            )["latest_gate"],
            name="D1 feature active gate",
        )
    ) or current_gate.get("gate_id") != execution.get("gate_id"):
        raise RuntimeError("D1 feature authority no longer binds the active gate")
    if require_fresh_gate:
        expires = datetime.fromisoformat(str(execution.get("expires_at_utc")))
        if datetime.now(timezone.utc) > expires:
            raise RuntimeError("D1 feature execution authority expired")
    event_path = verified_artifact_path(
        _mapping(pointer.get("latest_event"), name="D1 feature latest event"),
        name="D1 feature latest event",
    )
    event = load_content_manifest(
        event_path, name="D1 feature latest event", statuses=("ACTIVE",)
    )
    if (
        event_path.parent != directory / "events"
        or event.get("execution_id") != execution_id
        or pointer.get("completed_jobs") != event.get("completed_jobs")
        or pointer.get("expected_jobs") != FEATURE_JOB_COUNT
    ):
        raise RuntimeError("D1 feature ACTIVE event/pointer binding differs")
    return execution_path, execution, event_path, event


def validate_worker_context(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    execution_path, execution, event_path, event = validate_active_execution(
        root,
        plan_path=plan_path,
        plan=plan,
        require_fresh_gate=False,
    )
    execution_id = str(execution["execution_id"])
    job_id = str(job.get("job_id", ""))
    claim_record = _mapping(event.get("claim"), name="D1 feature active job claim")
    claim_path = verified_artifact_path(
        claim_record, name="D1 feature active job claim"
    )
    expected_claim = (
        execution_directory(root, execution_id) / "claims" / f"{job_id}.json"
    )
    claim = load_content_manifest(
        claim_path, name="D1 feature active job claim", statuses=("CLAIMED",)
    )
    owner_pid = int(event.get("owner_pid", -1))
    heartbeat = datetime.fromisoformat(str(event.get("heartbeat_at_utc")))
    heartbeat_age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    configuration = _mapping(
        job.get("configuration"), name="D1 feature worker configuration"
    )
    if (
        event.get("current_job_id") != job_id
        or claim_path != expected_claim
        or claim.get("execution_id") != execution_id
        or claim.get("job_id") != job_id
        or claim.get("configuration_sha256") != canonical_sha256(configuration)
        or claim.get("owner_pid") != owner_pid
        or os.environ.get(FEATURE_EXECUTION_ID_ENV) != execution_id
        or os.environ.get(FEATURE_CLAIM_SHA256_ENV) != claim_record.get("sha256")
        or owner_pid <= 1
    ):
        raise RuntimeError("D1 feature worker claim/execution identity differs")
    if heartbeat_age < -GATE_CLOCK_TOLERANCE_SECONDS:
        raise RuntimeError("D1 feature execution heartbeat is in the future")
    if heartbeat_age > LEASE_HEARTBEAT_MAX_AGE_SECONDS:
        raise RuntimeError("D1 feature execution heartbeat is stale")
    try:
        os.kill(owner_pid, 0)
    except OSError as error:
        raise RuntimeError("D1 feature orchestrator owner is not alive") from error
    if owner_pid not in _current_process_ancestors():
        raise RuntimeError("D1 feature orchestrator is not a worker ancestor")
    _assert_repository_lease_is_held(root)
    return execution_path, execution, event_path, event, claim_path, claim


def authorization_expiry(now: datetime) -> str:
    return (now + timedelta(seconds=AUTHORIZATION_START_WINDOW_SECONDS)).isoformat()


__all__ = [
    "FEATURE_CLAIM_SHA256_ENV",
    "FEATURE_EXECUTION_ID_ENV",
    "FEATURE_EXECUTION_POINTER_RELATIVE",
    "FEATURE_EXECUTION_REGISTRY_RELATIVE",
    "FEATURE_JOB_COUNT",
    "authorization_expiry",
    "create_job_claim",
    "execution_directory",
    "execution_source_records",
    "exclusive_json",
    "validate_active_execution",
    "validate_worker_context",
    "write_execution_event",
]
