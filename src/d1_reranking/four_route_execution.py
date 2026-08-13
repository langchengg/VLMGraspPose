"""Immutable serial execution contracts for the P12 Validation producers."""

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

from .execution import (
    CANONICAL_RANK1_RUN_DIR,
    artifact_record,
    load_content_manifest,
    validate_resource_gate,
)
from .four_route_plan import PLAN_RELATIVE_PATH, PRODUCER_JOB_STAGES
from .k_execution import authorization_expiry, exclusive_json
from .resource_gate import host_contract, resource_thresholds


P12_EXECUTION_ID_ENV = "D1_P12_VALIDATION_EXECUTION_ID"
P12_CLAIM_SHA256_ENV = "D1_P12_VALIDATION_CLAIM_SHA256"
P12_EXECUTION_POINTER_RELATIVE = Path("configs/d1_four_route_validation_execution.json")
P12_EXECUTION_REGISTRY_RELATIVE = Path("configs/four_route_validation_executions")
P12_POLICY_POINTER_RELATIVE = Path(
    "configs/d1_four_route_validation_resource_policy.json"
)
P12_POLICY_REGISTRY_RELATIVE = Path("configs/four_route_validation_resource_policies")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _artifact_record_binds_path(
    value: Any, *, expected_path: Path, name: str
) -> bool:
    """Verify a possibly enriched artifact record against one current file.

    Producer manifests include a ``bytes`` field while execution authority
    records intentionally contain only ``path`` and ``sha256``.  Integrity is
    defined by the verified file identity, not by exact equality of those two
    record schemas.
    """

    if not isinstance(value, Mapping):
        return False
    try:
        observed_path = verified_artifact_path(value, name=name)
    except (RuntimeError, ValueError):
        return False
    expected = expected_path.expanduser().resolve()
    if observed_path != expected:
        return False
    if "bytes" in value:
        declared_bytes = value.get("bytes")
        if (
            not isinstance(declared_bytes, int)
            or isinstance(declared_bytes, bool)
            or declared_bytes != expected.stat().st_size
        ):
            return False
    return True


def load_four_route_execution_plan(root: Path) -> tuple[Path, dict[str, Any]]:
    path = (root / PLAN_RELATIVE_PATH).resolve()
    plan = load_content_manifest(
        path, name="P12 four-route plan", statuses=("PLANNED",)
    )
    producer = _mapping(
        plan.get("sources", {}).get("validation_producer"), name="P12 producer contract"
    )
    jobs = producer.get("jobs")
    if (
        plan.get("validation_producer_ready") is not True
        or producer.get("status") != "READY"
        or producer.get("candidate_test_labels_read") is not False
        or producer.get("test_inputs_referenced") is not False
        or producer.get("device") != "cpu"
        or producer.get("max_parallel") != 1
        or not isinstance(jobs, list)
        or len(jobs) != len(PRODUCER_JOB_STAGES)
        or producer.get("job_count") != len(jobs)
        or [job.get("configuration", {}).get("stage") for job in jobs]
        != list(PRODUCER_JOB_STAGES)
        or producer.get("job_ids_sha256")
        != canonical_sha256([job.get("job_id") for job in jobs])
    ):
        raise RuntimeError("P12 Validation execution plan contract differs")
    for job in jobs:
        configuration = _mapping(job.get("configuration"), name="P12 producer job")
        if (
            canonical_sha256(configuration)[:16] != job.get("job_id")
            or configuration.get("device") != "cpu"
            or configuration.get("threads") != 1
        ):
            raise RuntimeError("P12 producer job identity differs")
    verify_artifact_records_recursive(
        producer, name="P12 producer input closure", require_at_least_one=True
    )
    spec_path = verified_artifact_path(
        producer.get("spec", {}), name="P12 producer spec"
    )
    spec = load_content_manifest(
        spec_path,
        name="P12 producer spec",
        statuses=("VALIDATION_PRODUCER_DECLARED",),
    )
    if (
        spec.get("candidate_test_labels_read") is not False
        or spec.get("selection_used_test_metrics") is not False
        or set(spec.get("t4", {})) != {"train", "validation"}
    ):
        raise PermissionError("P12 producer spec Test-isolation semantics differ")

    def matches(record: Mapping[str, Any], raw: Any) -> bool:
        if not isinstance(raw, str) or not raw:
            return False
        candidate = Path(raw).expanduser()
        resolved = (
            (spec_path.parent / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
        return record.get("path") == str(resolved)

    spec_router = _mapping(spec.get("router"), name="P12 router spec")
    bound_router = _mapping(producer.get("router"), name="P12 router contract")
    spec_union = _mapping(spec.get("union"), name="P12 union spec")
    bound_union = _mapping(producer.get("union"), name="P12 union contract")
    if (
        not matches(bound_router["train_oof"], spec_router.get("train_oof"))
        or not matches(bound_router["validation"], spec_router.get("validation"))
        or bound_router.get("feature_columns") != spec_router.get("feature_columns")
        or any(
            not matches(bound_union[key], spec_union.get(key))
            for key in (
                "train_top20",
                "validation_top20",
                "folds",
                "train_denominator",
                "validation_denominator",
            )
        )
        or bound_union.get("feature_columns") != spec_union.get("feature_columns")
    ):
        raise RuntimeError("P12 producer spec expansion differs from the frozen plan")
    for split in ("train", "validation"):
        spec_split = _mapping(spec["t4"][split], name=f"P12 {split} T4 spec")
        bound_split = _mapping(producer["t4"][split], name=f"P12 {split} T4 contract")
        if (
            not matches(bound_split["d1_top5"], spec_split.get("d1_top5"))
            or not matches(bound_split["t3_manifest"], spec_split.get("t3_manifest"))
            or set(spec_split.get("peer_top5", {})) != {"CROG", "G1", "C1"}
            or any(
                not matches(
                    bound_split["peer_top5"][route], spec_split["peer_top5"][route]
                )
                for route in ("CROG", "G1", "C1")
            )
        ):
            raise RuntimeError(
                "P12 producer T4 spec expansion differs from the frozen plan"
            )
    return path, plan


def execution_scope(plan: Mapping[str, Any]) -> dict[str, Any]:
    producer = _mapping(
        plan.get("sources", {}).get("validation_producer"), name="P12 producer contract"
    )
    return {
        "route": "CROG_G1_C1_D1",
        "stage": "P12",
        "operation": "four_route_validation_producers",
        "device": "cpu",
        "max_parallel": 1,
        "job_count": int(producer["job_count"]),
    }


def policy_source_paths() -> tuple[Path, ...]:
    return (
        REPOSITORY_ROOT / "src/d1_reranking/four_route_execution.py",
        REPOSITORY_ROOT / "src/d1_reranking/four_route_plan.py",
        REPOSITORY_ROOT / "src/d1_reranking/four_route_producer.py",
        REPOSITORY_ROOT / "src/d1_reranking/four_route_t4.py",
        REPOSITORY_ROOT / "src/d1_reranking/resource_gate.py",
        REPOSITORY_ROOT / "src/unified_reranking/models/lightgbm_ranker.py",
        REPOSITORY_ROOT / "tools/d1_reranking/audit_resources.py",
        REPOSITORY_ROOT / "tools/d1_reranking/run_four_route_validation.py",
    )


def resource_policy(*, plan_path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    sources = [artifact_record(path) for path in policy_source_paths()]
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY",
        "gate_type": "d1_p12_validation_three_continuous_five_minute_windows_v1",
        "scope": execution_scope(plan),
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


def write_resource_policy(root: Path, *, resume: bool) -> tuple[Path, dict[str, Any]]:
    plan_path, plan = load_four_route_execution_plan(root)
    policy = resource_policy(plan_path=plan_path, plan=plan)
    policy_id = str(policy["content_sha256"])[:24]
    path = root / P12_POLICY_REGISTRY_RELATIVE / f"{policy_id}.json"
    if path.exists():
        existing = load_content_manifest(
            path, name="P12 resource policy", statuses=("LOCKED_POLICY",)
        )
        if not resume or existing != policy:
            raise RuntimeError("immutable P12 resource policy exists and differs")
    else:
        exclusive_json(path, policy)
    pointer: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY_POINTER",
        "active_policy": artifact_record(path),
        "plan": artifact_record(plan_path),
        "scope": execution_scope(plan),
        "candidate_test_labels_read": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    pointer_path = root / P12_POLICY_POINTER_RELATIVE
    if pointer_path.exists():
        existing_pointer = load_content_manifest(
            pointer_path,
            name="P12 resource policy pointer",
            statuses=("LOCKED_POLICY_POINTER",),
        )
        if not resume or existing_pointer != pointer:
            raise RuntimeError("immutable P12 policy pointer exists and differs")
    else:
        atomic_json(pointer_path, pointer)
    return path, policy


def load_resource_policy(
    root: Path, *, plan_path: Path, plan: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    pointer = load_content_manifest(
        root / P12_POLICY_POINTER_RELATIVE,
        name="P12 resource policy pointer",
        statuses=("LOCKED_POLICY_POINTER",),
    )
    path = verified_artifact_path(
        pointer.get("active_policy", {}), name="P12 active resource policy"
    )
    if path.parent != (root / P12_POLICY_REGISTRY_RELATIVE).resolve():
        raise RuntimeError("P12 resource policy escaped its immutable registry")
    policy = load_content_manifest(
        path, name="P12 resource policy", statuses=("LOCKED_POLICY",)
    )
    if (
        policy != resource_policy(plan_path=plan_path, plan=plan)
        or pointer.get("plan") != artifact_record(plan_path)
        or pointer.get("scope") != execution_scope(plan)
    ):
        raise RuntimeError("P12 resource policy plan/code binding differs")
    return path, policy


def execution_directory(root: Path, execution_id: str) -> Path:
    if not execution_id or any(
        character not in "0123456789abcdef" for character in execution_id
    ):
        raise RuntimeError("P12 execution identifier is invalid")
    return root / P12_EXECUTION_REGISTRY_RELATIVE / execution_id


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
    owner_name = owner.strip()
    if not owner_name:
        raise ValueError("P12 execution owner must be non-empty")
    inherited = {} if resume_outputs is None else dict(resume_outputs)
    if resume_from is None and inherited:
        raise RuntimeError("P12 inherited outputs require a failed execution")
    if inherited:
        verify_artifact_records_recursive(
            inherited, name="P12 inherited outputs", require_at_least_one=True
        )
    now = datetime.now(timezone.utc)
    sources: dict[str, Any] = {
        "plan": artifact_record(plan_path),
        "resource_gate": artifact_record(resource_gate_path),
        "contract": artifact_record(Path(__file__)),
        "runner": artifact_record(
            REPOSITORY_ROOT / "tools/d1_reranking/run_four_route_validation.py"
        ),
    }
    if resume_from is not None:
        sources["resume_from"] = dict(resume_from)
    execution_id = canonical_sha256(
        {
            "plan": sources["plan"],
            "gate": sources["resource_gate"],
            "owner": owner_name,
            "created": now.isoformat(),
        }
    )[:24]
    producer = _mapping(
        plan["sources"]["validation_producer"], name="P12 producer contract"
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "ACTIVE",
        "execution_id": execution_id,
        "run_dir": str(root.resolve()),
        "rank1_run_dir": str(CANONICAL_RANK1_RUN_DIR),
        "scope": execution_scope(plan),
        "plan": artifact_record(plan_path),
        "job_count": producer["job_count"],
        "job_ids_sha256": producer["job_ids_sha256"],
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
    configuration = _mapping(job.get("configuration"), name="P12 job configuration")
    job_id = str(job.get("job_id", ""))
    if canonical_sha256(configuration)[:16] != job_id or owner_pid <= 1 or not command:
        raise RuntimeError("P12 claim identity differs")
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "CLAIMED",
        "execution_id": execution["execution_id"],
        "job_id": job_id,
        "configuration_sha256": canonical_sha256(configuration),
        "owner_pid": owner_pid,
        "command": list(command),
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    value["content_sha256"] = canonical_sha256(value)
    path = (
        execution_directory(root, str(execution["execution_id"]))
        / "claims"
        / f"{job_id}.json"
    )
    exclusive_json(path, value)
    return path, value


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
        raise ValueError("P12 execution event state differs")
    output_map = dict(outputs)
    expected = int(execution["job_count"])
    if status == "COMPLETE" and len(output_map) != expected:
        raise RuntimeError("P12 execution cannot complete with missing outputs")
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
        "expected_jobs": expected,
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
        "expected_jobs": expected,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(root / P12_EXECUTION_POINTER_RELATIVE, pointer)
    return event_path, event


def _ancestors() -> set[int]:
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


def validate_worker_context(
    root: Path, *, job_id: str, lease_path: Path | None
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    expected_lease = (root.parent / ".d1_heavy_resource.lock").resolve()
    if lease_path is None or lease_path.resolve() != expected_lease:
        raise PermissionError("P12 worker requires the global heavy-resource lease")
    plan_path, plan = load_four_route_execution_plan(root)
    pointer = load_content_manifest(
        root / P12_EXECUTION_POINTER_RELATIVE,
        name="P12 execution pointer",
        statuses=("ACTIVE",),
    )
    execution_path = verified_artifact_path(
        pointer["execution"], name="P12 execution authority"
    )
    execution = load_content_manifest(
        execution_path, name="P12 execution authority", statuses=("ACTIVE",)
    )
    event_path = verified_artifact_path(
        pointer["latest_event"], name="P12 active event"
    )
    event = load_content_manifest(
        event_path, name="P12 active event", statuses=("ACTIVE",)
    )
    claim_path = verified_artifact_path(event.get("claim", {}), name="P12 active claim")
    claim = load_content_manifest(claim_path, name="P12 claim", statuses=("CLAIMED",))
    if (
        execution.get("plan") != artifact_record(plan_path)
        or execution.get("job_ids_sha256")
        != plan["sources"]["validation_producer"]["job_ids_sha256"]
        or event.get("current_job_id") != job_id
        or claim.get("job_id") != job_id
        or claim.get("execution_id") != execution.get("execution_id")
        or claim.get("owner_pid") not in _ancestors()
        or os.environ.get(P12_EXECUTION_ID_ENV) != execution.get("execution_id")
        or os.environ.get(P12_CLAIM_SHA256_ENV) != event["claim"]["sha256"]
    ):
        raise PermissionError("P12 worker authority/claim differs")
    return plan_path, plan, execution_path, claim_path


def validate_complete_execution(root: Path) -> dict[str, Any]:
    plan_path, plan = load_four_route_execution_plan(root)
    pointer = load_content_manifest(
        root / P12_EXECUTION_POINTER_RELATIVE,
        name="P12 execution pointer",
        statuses=("COMPLETE",),
    )
    execution_path = verified_artifact_path(
        pointer["execution"], name="P12 execution authority"
    )
    execution = load_content_manifest(
        execution_path, name="P12 execution authority", statuses=("ACTIVE",)
    )
    event_path = verified_artifact_path(
        pointer["latest_event"], name="P12 complete event"
    )
    event = load_content_manifest(
        event_path, name="P12 complete event", statuses=("COMPLETE",)
    )
    jobs = plan["sources"]["validation_producer"]["jobs"]
    expected_ids = {str(job["job_id"]) for job in jobs}
    if (
        execution.get("plan") != artifact_record(plan_path)
        or pointer.get("execution_id") != execution.get("execution_id")
        or event.get("execution_id") != execution.get("execution_id")
        or set(event.get("outputs", {})) != expected_ids
        or event.get("completed_jobs") != len(expected_ids)
        or event.get("expected_jobs") != len(expected_ids)
        or event.get("current_job_id") is not None
        or event.get("claim") is not None
        or event.get("candidate_test_labels_read") is not False
        or event.get("test_inputs_referenced") is not False
    ):
        raise RuntimeError("P12 complete execution semantics differ")
    verify_artifact_records_recursive(
        event["outputs"], name="P12 execution outputs", require_at_least_one=True
    )
    expected_outputs = {
        "router_validation": root
        / "13_four_route_extension/router/router_selection_manifest.json",
        "top20_union_validation": root
        / "13_four_route_extension/union/selected_union_ranker.json",
        "t4_train": root
        / "03_features/train/top5/T4_four_route_consensus/manifest.json",
        "t4_validation": root
        / "03_features/validation/top5/T4_four_route_consensus/manifest.json",
        "validation_summary": root
        / "13_four_route_extension/validation_router_union_manifest.json",
    }
    for job in jobs:
        job_id = str(job["job_id"])
        stage = str(job["configuration"]["stage"])
        result_path = verified_artifact_path(
            event["outputs"][job_id], name=f"P12 {stage} result"
        )
        if (
            result_path
            != (
                root / "13_four_route_extension/execution_results" / f"{job_id}.json"
            ).resolve()
        ):
            raise RuntimeError(f"P12 {stage} result path differs")
        result = load_content_manifest(
            result_path, name=f"P12 {stage} result", statuses=("COMPLETE",)
        )
        verify_artifact_records_recursive(
            {"sources": result.get("sources"), "artifacts": result.get("artifacts")},
            name=f"P12 {stage} result closure",
            require_at_least_one=True,
        )
        producer_path = verified_artifact_path(
            result.get("artifacts", {}).get("producer_manifest", {}),
            name=f"P12 {stage} producer",
        )
        if (
            result.get("job_id") != job_id
            or result.get("configuration") != job["configuration"]
            or result.get("sources", {}).get("plan") != artifact_record(plan_path)
            or result.get("candidate_test_labels_read") is not False
            or result.get("test_inputs_referenced") is not False
            or producer_path != expected_outputs[stage].resolve()
        ):
            raise RuntimeError(f"P12 {stage} result semantics differ")
        claim_path = verified_artifact_path(
            result.get("sources", {}).get("execution_claim", {}),
            name=f"P12 {stage} claim",
        )
        claim = load_content_manifest(
            claim_path, name=f"P12 {stage} claim", statuses=("CLAIMED",)
        )
        job_execution_path = verified_artifact_path(
            result.get("sources", {}).get("execution_authority", {}),
            name=f"P12 {stage} execution authority",
        )
        job_execution = load_content_manifest(
            job_execution_path,
            name=f"P12 {stage} execution authority",
            statuses=("ACTIVE",),
        )
        job_execution_id = str(job_execution.get("execution_id", ""))
        if (
            claim.get("job_id") != job_id
            or claim.get("configuration_sha256")
            != canonical_sha256(job["configuration"])
            or claim.get("execution_id") != job_execution_id
            or job_execution.get("plan") != artifact_record(plan_path)
            or job_execution_path
            != execution_directory(root, job_execution_id) / "execution.json"
            or claim_path
            != execution_directory(root, job_execution_id) / "claims" / f"{job_id}.json"
        ):
            raise RuntimeError(f"P12 {stage} claim semantics differ")
        producer_manifest = load_content_manifest(
            producer_path,
            name=f"P12 {stage} producer",
            statuses=("COMPLETE", "VALIDATION_LOCKED"),
        )
        if producer_manifest.get("candidate_test_labels_read") is not False:
            raise PermissionError(f"P12 {stage} producer violates Test isolation")
        if stage == "top20_union_validation":
            internal_plan_path = verified_artifact_path(
                producer_manifest.get("sources", {}).get("plan", {}),
                name="P12 Top20 internal plan",
            )
            internal_plan = load_content_manifest(
                internal_plan_path,
                name="P12 Top20 internal plan",
                statuses=("PLANNED",),
            )
            bound_plan = internal_plan.get("sources", {}).get("p12_plan")
        else:
            bound_plan = producer_manifest.get("sources", {}).get("p12_plan")
        if not _artifact_record_binds_path(
            bound_plan,
            expected_path=plan_path,
            name=f"P12 {stage} producer plan",
        ):
            raise RuntimeError(f"P12 {stage} producer does not bind the current plan")
    return {
        "execution": artifact_record(execution_path),
        "event": artifact_record(event_path),
    }


def validate_gate_for_execution(
    root: Path, *, gate_path: Path, require_fresh: bool
) -> dict[str, Any]:
    plan_path, plan = load_four_route_execution_plan(root)
    policy_path, _policy = load_resource_policy(root, plan_path=plan_path, plan=plan)
    return validate_resource_gate(
        gate_path,
        run_dir=root,
        plan_path=plan_path,
        policy_path=policy_path,
        expected_scope=execution_scope(plan),
        require_fresh=require_fresh,
    )


__all__ = [
    "P12_CLAIM_SHA256_ENV",
    "P12_EXECUTION_ID_ENV",
    "P12_EXECUTION_POINTER_RELATIVE",
    "create_execution_authority",
    "create_job_claim",
    "execution_directory",
    "execution_scope",
    "load_four_route_execution_plan",
    "load_resource_policy",
    "validate_complete_execution",
    "validate_gate_for_execution",
    "validate_worker_context",
    "write_execution_event",
    "write_resource_policy",
]
