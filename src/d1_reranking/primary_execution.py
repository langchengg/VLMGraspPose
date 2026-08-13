"""Immutable authorization, claim, event, and recovery contracts for P9."""

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
from .plan import load_active_primary_plan, load_primary_plan
from .resource_gate import host_contract, resource_thresholds


PRIMARY_EXECUTION_ID_ENV = "D1_PRIMARY_EXECUTION_ID"
PRIMARY_CLAIM_SHA256_ENV = "D1_PRIMARY_CLAIM_SHA256"
PRIMARY_EXECUTION_POINTER_RELATIVE = Path("configs/d1_primary_matrix_execution.json")
PRIMARY_EXECUTION_REGISTRY_RELATIVE = Path("configs/primary_matrix_executions")
PRIMARY_RESOURCE_POLICY_POINTER_RELATIVE = Path(
    "configs/d1_primary_resource_gate_policy_active.json"
)
PRIMARY_RESOURCE_POLICY_REGISTRY_RELATIVE = Path(
    "configs/primary_resource_policies"
)
PRIMARY_EXECUTION_SCOPE = {
    "route": "D1",
    "stage": "P9",
    "operation": "primary_matrix",
    "device": "cpu",
    "max_parallel": 1,
    "job_count": 360,
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _same_artifact_identity(observed: Any, expected: Mapping[str, Any]) -> bool:
    """Match path/SHA exactly while allowing optional byte/count metadata."""

    if not isinstance(observed, Mapping) or any(
        observed.get(key) != expected.get(key) for key in ("path", "sha256")
    ):
        return False
    return not (
        "bytes" in observed
        and "bytes" in expected
        and observed.get("bytes") != expected.get("bytes")
    )


def exclusive_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Create one immutable JSON record using kernel O_EXCL."""

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


def execution_directory(root: Path, execution_id: str) -> Path:
    if not execution_id or any(
        character not in "0123456789abcdef" for character in execution_id
    ):
        raise RuntimeError("D1 primary execution identifier is invalid")
    return root / PRIMARY_EXECUTION_REGISTRY_RELATIVE / execution_id


def execution_source_records(
    *,
    plan_path: Path,
    resource_gate_path: Path,
    resume_from: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    records = {
        "plan": artifact_record(plan_path),
        "resource_gate": artifact_record(resource_gate_path),
        "contract": artifact_record(Path(__file__)),
        "runner": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/train_primary_cell.py"
        ),
        "authorizer": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/authorize_primary_execution.py"
        ),
        "orchestrator": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/run_primary_matrix.py"
        ),
    }
    if resume_from is not None:
        records["resume_from"] = {
            "path": str(resume_from["path"]),
            "sha256": str(resume_from["sha256"]),
        }
    return records


def primary_resource_policy(
    *, plan_path: Path, source_paths: Sequence[Path]
) -> dict[str, Any]:
    plan = load_primary_plan(plan_path)
    if plan.get("job_count") != 360:
        raise RuntimeError("D1 primary resource policy requires the exact 360-job plan")
    sources = [artifact_record(path) for path in source_paths]
    if not sources:
        raise ValueError("D1 primary resource policy requires implementation sources")
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY",
        "gate_type": "d1_primary_cpu_three_continuous_five_minute_windows_v2",
        "scope": PRIMARY_EXECUTION_SCOPE,
        "rank1_run_dir": str(CANONICAL_RANK1_RUN_DIR),
        "plan": artifact_record(plan_path),
        "thresholds": resource_thresholds(),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def load_primary_resource_policy(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and semantically replay the active immutable primary policy."""

    root = Path(run_dir).expanduser().resolve()
    pointer_path = root / PRIMARY_RESOURCE_POLICY_POINTER_RELATIVE
    plan_path, _plan = load_active_primary_plan(root)
    if not pointer_path.exists():
        legacy = root / "configs/d1_resource_gate_policy.json"
        policy = load_content_manifest(
            legacy, name="D1 primary resource policy", statuses=("LOCKED_POLICY",)
        )
        raw_sources = policy.get("sources")
        if not isinstance(raw_sources, list):
            raise RuntimeError("D1 primary resource-policy sources are invalid")
        sources = tuple(
            verified_artifact_path(record, name="D1 primary resource-policy source")
            for record in raw_sources
        )
        if policy != primary_resource_policy(
            plan_path=plan_path, source_paths=sources
        ):
            raise RuntimeError("D1 primary resource-policy code/plan binding differs")
        return legacy, policy
    pointer = load_content_manifest(
        pointer_path,
        name="D1 active primary resource-policy pointer",
        statuses=("LOCKED_POLICY_POINTER",),
    )
    policy_path = verified_artifact_path(
        _mapping(pointer.get("active_policy"), name="D1 primary policy record"),
        name="D1 active primary resource policy",
    )
    if policy_path.parent != (root / PRIMARY_RESOURCE_POLICY_REGISTRY_RELATIVE).resolve():
        raise RuntimeError("D1 primary resource policy is outside its registry")
    policy = load_content_manifest(
        policy_path, name="D1 primary resource policy", statuses=("LOCKED_POLICY",)
    )
    raw_sources = policy.get("sources")
    if not isinstance(raw_sources, list):
        raise RuntimeError("D1 primary resource-policy sources are invalid")
    sources = tuple(
        verified_artifact_path(record, name="D1 primary resource-policy source")
        for record in raw_sources
    )
    expected_plan = artifact_record(plan_path)
    expected_policy = artifact_record(policy_path)
    if (
        policy != primary_resource_policy(plan_path=plan_path, source_paths=sources)
        or pointer.get("active_policy") != expected_policy
        or pointer.get("plan") != expected_plan
        or pointer.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 active primary policy/plan binding differs")
    return policy_path, policy


def create_job_claim(
    root: Path,
    *,
    execution: Mapping[str, Any],
    job: Mapping[str, Any],
    owner_pid: int,
    command: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    configuration = _mapping(
        job.get("configuration"), name="D1 primary claimed configuration"
    )
    job_id = str(job.get("job_id", ""))
    if canonical_sha256(configuration)[:16] != job_id:
        raise RuntimeError("D1 primary claimed job/configuration differs")
    if (
        owner_pid <= 1
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise RuntimeError("D1 primary claim owner/command differs")
    claim: dict[str, Any] = {
        "schema_version": 1,
        "status": "CLAIMED",
        "execution_id": execution["execution_id"],
        "job_id": job_id,
        "configuration_sha256": canonical_sha256(configuration),
        "owner_pid": owner_pid,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    claim["content_sha256"] = canonical_sha256(claim)
    path = (
        execution_directory(root, str(execution["execution_id"]))
        / "claims"
        / f"{job_id}.json"
    )
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
        raise ValueError("D1 primary execution event status/sequence differs")
    output_map = dict(outputs)
    if status == "COMPLETE" and len(output_map) != 360:
        raise RuntimeError("D1 primary execution cannot complete without 360 results")
    event: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "execution_id": execution["execution_id"],
        "sequence": sequence,
        "owner_pid": owner_pid,
        "heartbeat_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_job_id": current_job_id,
        "claim": None if claim is None else dict(claim),
        "completed_jobs": len(output_map),
        "expected_jobs": 360,
        "outputs": output_map,
        "commands": [dict(command) for command in commands],
        "failure": failure,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    event["content_sha256"] = canonical_sha256(event)
    directory = execution_directory(root, str(execution["execution_id"]))
    event_path = directory / "events" / f"{sequence:04d}_{status.lower()}.json"
    exclusive_json(event_path, event)
    pointer: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "execution_id": execution["execution_id"],
        "execution": artifact_record(directory / "execution.json"),
        "latest_event": artifact_record(event_path),
        "completed_jobs": len(output_map),
        "expected_jobs": 360,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(root / PRIMARY_EXECUTION_POINTER_RELATIVE, pointer)
    return event_path, event


def create_execution_authority(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    gate_path: Path,
    owner: str,
    gate_id: object,
    live_recheck: Mapping[str, Any],
    resume_from: Mapping[str, Any] | None,
    resume_outputs: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    owner_name = owner.strip()
    if not owner_name:
        raise ValueError("D1 primary execution owner must be non-empty")
    now = datetime.now(timezone.utc)
    sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=gate_path,
        resume_from=resume_from,
    )
    execution_id = canonical_sha256(
        {
            "run_dir": str(root),
            "owner": owner_name,
            "plan": artifact_record(plan_path),
            "resource_gate": artifact_record(gate_path),
            "authorized_at_utc": now.isoformat(),
            "resume_from": resume_from,
        }
    )[:24]
    execution: dict[str, Any] = {
        "schema_version": 1,
        "status": "ACTIVE",
        "execution_id": execution_id,
        "authorized_at_utc": now.isoformat(),
        "expires_at_utc": (
            now + timedelta(seconds=AUTHORIZATION_START_WINDOW_SECONDS)
        ).isoformat(),
        "owner": owner_name,
        "run_dir": str(root),
        "rank1_run_dir": str(CANONICAL_RANK1_RUN_DIR),
        "host": host_contract(),
        "scope": PRIMARY_EXECUTION_SCOPE,
        "plan": artifact_record(plan_path),
        "resource_gate": artifact_record(gate_path),
        "gate_id": gate_id,
        "job_count": 360,
        "job_ids_sha256": plan["job_ids_sha256"],
        "max_parallel": 1,
        "device": "cpu",
        "direct_cli_execution_permitted": False,
        "fresh_gate_required_on_resume": True,
        "resume_from": None if resume_from is None else dict(resume_from),
        "resume_outputs": dict(resume_outputs),
        "live_recheck": dict(live_recheck),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    execution["content_sha256"] = canonical_sha256(execution)
    path = execution_directory(root, execution_id) / "execution.json"
    exclusive_json(path, execution)
    write_execution_event(
        root,
        execution=execution,
        sequence=0,
        status="ACTIVE",
        owner_pid=None,
        current_job_id=None,
        claim=None,
        outputs=resume_outputs,
        commands=(),
    )
    return path, execution


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


def validate_active_execution(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    require_fresh_gate: bool,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    pointer = load_content_manifest(
        root / PRIMARY_EXECUTION_POINTER_RELATIVE,
        name="D1 primary execution pointer",
        statuses=("ACTIVE",),
    )
    execution_id = str(pointer.get("execution_id", ""))
    directory = execution_directory(root, execution_id)
    execution_path = verified_artifact_path(
        _mapping(pointer.get("execution"), name="D1 primary execution record"),
        name="D1 primary execution authority",
    )
    if execution_path != directory / "execution.json":
        raise RuntimeError("D1 primary execution authority path differs")
    execution = load_content_manifest(
        execution_path, name="D1 primary execution authority", statuses=("ACTIVE",)
    )
    gate_path = verified_artifact_path(
        _mapping(execution.get("resource_gate"), name="D1 primary gate record"),
        name="D1 primary execution gate",
    )
    resume_from = execution.get("resume_from")
    expected_sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=gate_path,
        resume_from=(
            _mapping(resume_from, name="D1 primary resume source")
            if resume_from is not None
            else None
        ),
    )
    if (
        load_primary_plan(plan_path) != plan
        or execution.get("execution_id") != execution_id
        or execution.get("run_dir") != str(root)
        or execution.get("rank1_run_dir") != str(CANONICAL_RANK1_RUN_DIR)
        or execution.get("scope") != PRIMARY_EXECUTION_SCOPE
        or execution.get("plan") != artifact_record(plan_path)
        or execution.get("job_count") != 360
        or execution.get("job_ids_sha256") != plan.get("job_ids_sha256")
        or execution.get("candidate_test_labels_read") is not False
        or execution.get("test_inputs_referenced") is not False
        or execution.get("direct_cli_execution_permitted") is not False
        or execution.get("host") != host_contract()
        or execution.get("sources") != expected_sources
        or execution.get("source_signature_sha256")
        != canonical_sha256(expected_sources)
    ):
        raise RuntimeError("D1 primary execution authority contract differs")
    verify_artifact_records_recursive(
        expected_sources,
        name="D1 primary execution sources",
        require_at_least_one=True,
    )
    validate_resource_gate(
        gate_path,
        run_dir=root,
        plan_path=plan_path,
        expected_scope=PRIMARY_EXECUTION_SCOPE,
        require_fresh=require_fresh_gate,
    )
    if require_fresh_gate and datetime.now(timezone.utc) > datetime.fromisoformat(
        str(execution.get("expires_at_utc"))
    ):
        raise RuntimeError("D1 primary execution authority expired before start/resume")
    event_path = verified_artifact_path(
        _mapping(pointer.get("latest_event"), name="D1 primary latest event record"),
        name="D1 primary latest event",
    )
    event = load_content_manifest(
        event_path, name="D1 primary latest event", statuses=("ACTIVE",)
    )
    if (
        event_path.parent != directory / "events"
        or event.get("execution_id") != execution_id
        or pointer.get("completed_jobs") != event.get("completed_jobs")
        or pointer.get("expected_jobs") != 360
    ):
        raise RuntimeError("D1 primary ACTIVE event/pointer binding differs")
    return execution_path, execution, event_path, event


def validate_worker_context(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    execution_path, execution, event_path, event = validate_active_execution(
        root, plan_path=plan_path, plan=plan, require_fresh_gate=False
    )
    execution_id = str(execution["execution_id"])
    job_id = str(job.get("job_id", ""))
    claim_record = _mapping(event.get("claim"), name="D1 primary active claim")
    claim_path = verified_artifact_path(
        claim_record, name="D1 primary active job claim"
    )
    expected_claim = (
        execution_directory(root, execution_id) / "claims" / f"{job_id}.json"
    )
    claim = load_content_manifest(
        claim_path, name="D1 primary active claim", statuses=("CLAIMED",)
    )
    owner_pid = int(event.get("owner_pid", -1))
    heartbeat = datetime.fromisoformat(str(event.get("heartbeat_at_utc")))
    heartbeat_age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    configuration = _mapping(job.get("configuration"), name="D1 primary worker job")
    if (
        event.get("current_job_id") != job_id
        or claim_path != expected_claim
        or claim.get("execution_id") != execution_id
        or claim.get("job_id") != job_id
        or claim.get("configuration_sha256") != canonical_sha256(configuration)
        or claim.get("owner_pid") != owner_pid
        or os.environ.get(PRIMARY_EXECUTION_ID_ENV) != execution_id
        or os.environ.get(PRIMARY_CLAIM_SHA256_ENV) != claim_record.get("sha256")
        or owner_pid <= 1
    ):
        raise RuntimeError("D1 primary worker claim/execution identity differs")
    if heartbeat_age < -GATE_CLOCK_TOLERANCE_SECONDS:
        raise RuntimeError("D1 primary execution heartbeat is in the future")
    if heartbeat_age > LEASE_HEARTBEAT_MAX_AGE_SECONDS:
        raise RuntimeError("D1 primary execution heartbeat is stale")
    try:
        os.kill(owner_pid, 0)
    except OSError as error:
        raise RuntimeError("D1 primary orchestrator owner is not alive") from error
    if owner_pid not in _current_process_ancestors():
        raise RuntimeError("D1 primary orchestrator owner is not an ancestor")
    return execution_path, execution, event_path, event, claim_path, claim


def validate_primary_result(
    value: Mapping[str, Any],
    *,
    root: Path,
    plan_path: Path,
    job: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    result = _mapping(value, name="D1 primary result")
    unsigned = dict(result)
    recorded = unsigned.pop("content_sha256", None)
    configuration = _mapping(
        result.get("configuration"), name="D1 primary result configuration"
    )
    planned = _mapping(job.get("configuration"), name="D1 primary planned job")
    job_id = str(job.get("job_id", ""))
    mode = str(planned.get("mode", ""))
    cell_key = str(result.get("cell_key", ""))
    expected = (
        root
        / ("06_oof" if mode == "oof" else "07_validation")
        / "primary_cells"
        / cell_key
        / "manifest.json"
    ).resolve()
    sources = _mapping(result.get("sources"), name="D1 primary result sources")
    if (
        result.get("status") != "COMPLETE"
        or recorded != canonical_sha256(unsigned)
        or canonical_sha256(planned)[:16] != job_id
        or configuration.get("planned_job_id") != job_id
        or configuration.get("planned_configuration") != planned
        or canonical_sha256(configuration)[:16] != cell_key
        or manifest_path.resolve() != expected
        or not _same_artifact_identity(sources.get("plan"), artifact_record(plan_path))
        or result.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 primary result contract differs for {job_id}")
    verify_artifact_records_recursive(
        {
            "sources": sources,
            "artifacts": result.get("artifacts"),
            "execution": result.get("execution_provenance"),
        },
        name=f"D1 primary result {job_id}",
        require_at_least_one=True,
    )
    return result


__all__ = [
    "PRIMARY_CLAIM_SHA256_ENV",
    "PRIMARY_EXECUTION_ID_ENV",
    "PRIMARY_EXECUTION_POINTER_RELATIVE",
    "PRIMARY_EXECUTION_REGISTRY_RELATIVE",
    "PRIMARY_EXECUTION_SCOPE",
    "PRIMARY_RESOURCE_POLICY_POINTER_RELATIVE",
    "PRIMARY_RESOURCE_POLICY_REGISTRY_RELATIVE",
    "create_execution_authority",
    "create_job_claim",
    "execution_directory",
    "execution_source_records",
    "exclusive_json",
    "primary_resource_policy",
    "load_primary_resource_policy",
    "validate_active_execution",
    "validate_primary_result",
    "validate_worker_context",
    "write_execution_event",
]
