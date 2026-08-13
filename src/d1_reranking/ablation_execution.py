"""Independent serial execution authority for P10 ablation jobs."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256

from .ablation import (
    ABLATION_EXECUTION_POINTER_RELATIVE,
    ABLATION_EXECUTION_REGISTRY_RELATIVE,
)
from .execution import (
    CANONICAL_RANK1_RUN_DIR,
    GATE_CLOCK_TOLERANCE_SECONDS,
    LEASE_HEARTBEAT_MAX_AGE_SECONDS,
    artifact_record,
    load_content_manifest,
)
from .k_execution import authorization_expiry, exclusive_json
from .resource_gate import host_contract
from .resource_gate import resource_thresholds


ABLATION_EXECUTION_ID_ENV = "D1_ABLATION_EXECUTION_ID"
ABLATION_CLAIM_SHA256_ENV = "D1_ABLATION_CLAIM_SHA256"
ABLATION_RESOURCE_POLICY_POINTER_RELATIVE = Path(
    "configs/d1_ablation_resource_gate_policy_active.json"
)
ABLATION_RESOURCE_POLICY_REGISTRY_RELATIVE = Path("configs/ablation_resource_policies")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def ablation_execution_scope(plan: Mapping[str, Any]) -> dict[str, Any]:
    count = plan.get("job_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise RuntimeError("D1 P10 execution requires a non-empty plan")
    return {
        "route": "D1",
        "stage": "P10",
        "analysis": "evidence_track_and_feature_family_ablation",
        "device": "cpu",
        "max_parallel": 1,
        "job_count": count,
    }


def ablation_policy_source_paths() -> tuple[Path, ...]:
    """Return the exact code closure authorized to plan and execute P10."""

    root = Path(__file__).resolve().parents[2]
    return (
        root / "src/d1_reranking/ablation.py",
        root / "src/d1_reranking/ablation_execution.py",
        root / "src/d1_reranking/fold_calibration.py",
        root / "src/d1_reranking/models.py",
        root / "src/d1_reranking/resource_gate.py",
        root / "src/unified_reranking/gate.py",
        root / "src/unified_reranking/models/lightgbm_ranker.py",
        root / "tools/d1_reranking/audit_resources.py",
        root / "tools/d1_reranking/authorize_ablation_execution.py",
        root / "tools/d1_reranking/run_ablation_cell.py",
        root / "tools/d1_reranking/run_ablation_matrix.py",
        root / "tools/d1_reranking/write_ablation_resource_policy.py",
    )


def ablation_resource_policy(
    *, plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the immutable policy for exactly one current P10 plan/code closure."""

    if plan.get("candidate_test_labels_read") is not False:
        raise RuntimeError("D1 P10 resource policy violates Test isolation")
    scope = ablation_execution_scope(plan)
    sources = [artifact_record(path) for path in ablation_policy_source_paths()]
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY",
        "gate_type": "d1_ablation_three_continuous_five_minute_windows_v1",
        "scope": scope,
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


def load_ablation_resource_policy(
    run_dir: str | Path, *, plan_path: Path, plan: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Resolve and replay the active versioned P10 plan/code policy."""

    root = Path(run_dir).expanduser().resolve()
    pointer = load_content_manifest(
        root / ABLATION_RESOURCE_POLICY_POINTER_RELATIVE,
        name="D1 active P10 resource-policy pointer",
        statuses=("LOCKED_POLICY_POINTER",),
    )
    policy_path = verified_artifact_path(
        pointer.get("active_policy", {}), name="D1 active P10 resource policy"
    )
    if (
        policy_path.parent
        != (root / ABLATION_RESOURCE_POLICY_REGISTRY_RELATIVE).resolve()
    ):
        raise RuntimeError("D1 P10 resource policy is outside its immutable registry")
    policy = load_content_manifest(
        policy_path, name="D1 P10 resource policy", statuses=("LOCKED_POLICY",)
    )
    expected = ablation_resource_policy(plan_path=plan_path, plan=plan)
    if (
        policy != expected
        or pointer.get("active_policy") != artifact_record(policy_path)
        or pointer.get("plan") != artifact_record(plan_path)
        or pointer.get("scope") != ablation_execution_scope(plan)
    ):
        raise RuntimeError("D1 active P10 resource policy/plan/code binding differs")
    return policy_path, policy


def execution_directory(root: Path, execution_id: str) -> Path:
    if not execution_id or any(
        character not in "0123456789abcdef" for character in execution_id
    ):
        raise RuntimeError("D1 P10 execution identifier is invalid")
    return root / ABLATION_EXECUTION_REGISTRY_RELATIVE / execution_id


def execution_source_records(
    *,
    plan_path: Path,
    resource_gate_path: Path,
    resume_from: Mapping[str, Any] | None,
) -> dict[str, dict[str, str]]:
    root = Path(__file__).resolve().parents[2]
    gate = load_content_manifest(
        resource_gate_path, name="D1 P10 resource gate", statuses=("PASS",)
    )
    policy_path = verified_artifact_path(
        gate.get("policy", {}), name="D1 P10 resource policy"
    )
    records = {
        "plan": artifact_record(plan_path),
        "resource_gate": artifact_record(resource_gate_path),
        "resource_policy": artifact_record(policy_path),
        "contract": artifact_record(Path(__file__)),
        "runner": artifact_record(root / "tools/d1_reranking/run_ablation_cell.py"),
        "authorizer": artifact_record(
            root / "tools/d1_reranking/authorize_ablation_execution.py"
        ),
        "orchestrator": artifact_record(
            root / "tools/d1_reranking/run_ablation_matrix.py"
        ),
    }
    if resume_from is not None:
        records["resume_from"] = {
            "path": str(resume_from["path"]),
            "sha256": str(resume_from["sha256"]),
        }
    return records


def create_execution_authority(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    resource_gate_path: Path,
    owner: str,
    resume_from: Mapping[str, Any] | None = None,
    resume_outputs: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create one immutable ACTIVE authority after the caller validates its gate."""

    owner_name = owner.strip()
    if not owner_name:
        raise ValueError("D1 P10 execution owner must be non-empty")
    inherited = {} if resume_outputs is None else dict(resume_outputs)
    if resume_from is None and inherited:
        raise RuntimeError("D1 P10 inherited outputs require a recovery source")
    now = datetime.now(timezone.utc)
    scope = ablation_execution_scope(plan)
    sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=resource_gate_path,
        resume_from=resume_from,
    )
    execution_id = canonical_sha256(
        {
            "plan": artifact_record(plan_path),
            "resource_gate": artifact_record(resource_gate_path),
            "owner": owner_name,
            "created_at_utc": now.isoformat(),
            "resume_from": resume_from,
        }
    )[:24]
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "ACTIVE",
        "execution_id": execution_id,
        "run_dir": str(root.resolve()),
        "rank1_run_dir": str(CANONICAL_RANK1_RUN_DIR),
        "scope": scope,
        "plan": artifact_record(plan_path),
        "job_count": int(plan["job_count"]),
        "job_ids_sha256": plan["job_ids_sha256"],
        "owner": owner_name,
        "created_at_utc": now.isoformat(),
        "expires_at_utc": authorization_expiry(now),
        "resource_gate": artifact_record(resource_gate_path),
        "resume_from": None if resume_from is None else dict(resume_from),
        "resume_outputs": inherited,
        "host": host_contract(),
        "max_parallel": 1,
        "device": "cpu",
        "direct_cli_execution_permitted": False,
        "fresh_gate_required_on_resume": True,
        "global_heavy_resource_lease_required": True,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    value["content_sha256"] = canonical_sha256(value)
    path = execution_directory(root, execution_id) / "execution.json"
    exclusive_json(path, value)
    return path, value


def create_job_claim(
    root: Path,
    *,
    execution: Mapping[str, Any],
    job: Mapping[str, Any],
    owner_pid: int,
    command: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    configuration = _mapping(job.get("configuration"), name="D1 P10 job configuration")
    job_id = str(job.get("job_id", ""))
    if (
        canonical_sha256(configuration)[:16] != job_id
        or owner_pid <= 1
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise RuntimeError("D1 P10 claim identity differs")
    claim: dict[str, Any] = {
        "schema_version": 1,
        "status": "CLAIMED",
        "execution_id": str(execution["execution_id"]),
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
        raise ValueError("D1 P10 execution event state differs")
    expected_jobs = int(execution["job_count"])
    output_map = {str(key): value for key, value in outputs.items()}
    if len(output_map) > expected_jobs:
        raise RuntimeError("D1 P10 execution output count exceeds plan")
    event: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "execution_id": str(execution["execution_id"]),
        "sequence": sequence,
        "owner_pid": owner_pid,
        "heartbeat_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_job_id": current_job_id,
        "claim": None if claim is None else dict(claim),
        "completed_jobs": len(output_map),
        "expected_jobs": expected_jobs,
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
    execution_path = directory / "execution.json"
    pointer: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "execution_id": str(execution["execution_id"]),
        "execution": artifact_record(execution_path),
        "latest_event": artifact_record(event_path),
        "completed_jobs": len(output_map),
        "expected_jobs": expected_jobs,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(root / ABLATION_EXECUTION_POINTER_RELATIVE, pointer)
    return event_path, event


def _ancestors() -> set[int]:
    result: set[int] = set()
    cursor = os.getpid()
    for _ in range(64):
        if cursor <= 1 or cursor in result:
            break
        result.add(cursor)
        process = subprocess.run(
            ["/bin/ps", "-o", "ppid=", "-p", str(cursor)],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            cursor = int(process.stdout.strip())
        except ValueError:
            break
    return result


def validate_worker_context(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    pointer = load_content_manifest(
        root / ABLATION_EXECUTION_POINTER_RELATIVE,
        name="D1 P10 execution pointer",
        statuses=("ACTIVE",),
    )
    execution_id = str(pointer["execution_id"])
    directory = execution_directory(root, execution_id)
    execution_path = verified_artifact_path(
        pointer["execution"], name="D1 P10 authority"
    )
    event_path = verified_artifact_path(
        pointer["latest_event"], name="D1 P10 active event"
    )
    execution = load_content_manifest(
        execution_path, name="D1 P10 authority", statuses=("ACTIVE",)
    )
    event = load_content_manifest(
        event_path, name="D1 P10 active event", statuses=("ACTIVE",)
    )
    sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=verified_artifact_path(
            execution["resource_gate"], name="D1 P10 resource gate"
        ),
        resume_from=(
            _mapping(execution["resume_from"], name="D1 P10 resume")
            if execution.get("resume_from") is not None
            else None
        ),
    )
    job_id = str(job["job_id"])
    claim_path = verified_artifact_path(event["claim"], name="D1 P10 claim")
    claim = load_content_manifest(
        claim_path, name="D1 P10 claim", statuses=("CLAIMED",)
    )
    owner_pid = int(event.get("owner_pid", -1))
    heartbeat_age = (
        datetime.now(timezone.utc)
        - datetime.fromisoformat(str(event["heartbeat_at_utc"]))
    ).total_seconds()
    if (
        execution_path != directory / "execution.json"
        or event_path.parent != directory / "events"
        or execution.get("scope") != ablation_execution_scope(plan)
        or execution.get("plan") != artifact_record(plan_path)
        or execution.get("job_ids_sha256") != plan.get("job_ids_sha256")
        or execution.get("sources") != sources
        or execution.get("source_signature_sha256") != canonical_sha256(sources)
        or execution.get("direct_cli_execution_permitted") is not False
        or event.get("current_job_id") != job_id
        or claim_path != directory / "claims" / f"{job_id}.json"
        or claim.get("execution_id") != execution_id
        or claim.get("job_id") != job_id
        or claim.get("configuration_sha256") != canonical_sha256(job["configuration"])
        or claim.get("owner_pid") != owner_pid
        or os.environ.get(ABLATION_EXECUTION_ID_ENV) != execution_id
        or os.environ.get(ABLATION_CLAIM_SHA256_ENV) != event["claim"]["sha256"]
        or owner_pid <= 1
    ):
        raise RuntimeError("D1 P10 worker authority/claim differs")
    if (
        heartbeat_age < -GATE_CLOCK_TOLERANCE_SECONDS
        or heartbeat_age > LEASE_HEARTBEAT_MAX_AGE_SECONDS
    ):
        raise RuntimeError("D1 P10 worker heartbeat differs")
    try:
        os.kill(owner_pid, 0)
    except OSError as error:
        raise RuntimeError("D1 P10 orchestrator is not alive") from error
    if owner_pid not in _ancestors():
        raise RuntimeError("D1 P10 orchestrator is not an ancestor")
    verify_artifact_records_recursive(
        sources, name="D1 P10 execution sources", require_at_least_one=True
    )
    return execution_path, execution, event_path, event, claim_path, claim


__all__ = [
    "ABLATION_CLAIM_SHA256_ENV",
    "ABLATION_EXECUTION_ID_ENV",
    "ABLATION_RESOURCE_POLICY_POINTER_RELATIVE",
    "ABLATION_RESOURCE_POLICY_REGISTRY_RELATIVE",
    "ablation_policy_source_paths",
    "ablation_resource_policy",
    "ablation_execution_scope",
    "create_execution_authority",
    "create_job_claim",
    "execution_directory",
    "execution_source_records",
    "load_ablation_resource_policy",
    "validate_worker_context",
    "write_execution_event",
]
