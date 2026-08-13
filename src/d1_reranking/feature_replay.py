"""Exact P13 replay for the nine heavy raw matched-common feature jobs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import canonical_sha256

from .execution import FEATURE_RESOURCE_SCOPE, artifact_record, load_content_manifest
from .feature_execution import (
    FEATURE_EXECUTION_POINTER_RELATIVE,
    execution_directory,
    execution_source_records,
)
from .feature_plan import (
    FEATURE_JOB_COUNT,
    FEATURE_POOLS,
    FEATURE_SPLITS,
    load_active_feature_extraction_plan,
    validate_feature_extraction_result,
)


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _resume_lineage_execution_ids(
    root: Path, resume_from: Mapping[str, Any] | None
) -> set[str]:
    identifiers: set[str] = set()
    cursor = dict(resume_from) if resume_from is not None else None
    while cursor is not None:
        failed_path = verified_artifact_path(cursor, name="D1 feature resume event")
        failed = load_content_manifest(
            failed_path, name="D1 feature resume event", statuses=("FAILED",)
        )
        execution_id = str(failed.get("execution_id", ""))
        if not execution_id or execution_id in identifiers:
            raise RuntimeError("D1 feature resume lineage is cyclic or invalid")
        identifiers.add(execution_id)
        authority_path = execution_directory(root, execution_id) / "execution.json"
        authority = load_content_manifest(
            authority_path,
            name="D1 feature resumed authority",
            statuses=("ACTIVE",),
        )
        if (
            failed_path.parent != authority_path.parent / "events"
            or authority.get("execution_id") != execution_id
        ):
            raise RuntimeError("D1 feature resume event/authority lineage differs")
        prior = authority.get("resume_from")
        cursor = (
            _mapping(prior, name="D1 feature prior resume event")
            if prior is not None
            else None
        )
    return identifiers


def _validate_test_access_events(
    root: Path, *, jobs: Mapping[str, Mapping[str, Any]], outputs: Mapping[str, Any]
) -> dict[str, str]:
    path = root / "09_formal_test/test_access.log"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(
            "D1 raw-feature replay requires the Test access log"
        ) from error
    events: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"D1 Test access log line {line_number} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError(f"D1 Test access log line {line_number} is invalid")
        event_id = value.get("event_id")
        if isinstance(event_id, str) and event_id:
            if event_id in events and events[event_id] != value:
                raise RuntimeError("D1 Test access event identifier is ambiguous")
            events[event_id] = value
    matched: dict[str, str] = {}
    for job_id, job in jobs.items():
        configuration = _mapping(
            job.get("configuration"), name=f"D1 raw-feature job {job_id} configuration"
        )
        if configuration.get("split") != "test":
            continue
        event_id = canonical_sha256(
            {"stage": "d1_raw_feature_extraction", "job_id": job_id}
        )[:24]
        event = events.get(event_id)
        output = _mapping(outputs.get(job_id), name=f"D1 Test feature output {job_id}")
        if (
            event is None
            or event.get("event") != "prelock_label_free_test_stage"
            or event.get("stage") != "d1_raw_feature_extraction"
            or event.get("job_id") != job_id
            or event.get("pool") != configuration.get("pool")
            or event.get("output_manifest") != output.get("path")
            or event.get("output_manifest_sha256") != output.get("sha256")
            or event.get("candidate_labels_opened_as_table") is not False
            or event.get("candidate_test_labels_read") is not False
        ):
            raise RuntimeError(f"D1 Test feature access event differs for {job_id}")
        matched[job_id] = event_id
    if len(matched) != len(FEATURE_POOLS):
        raise RuntimeError("D1 raw-feature replay requires exactly three Test events")
    return {"path": str(path.resolve()), "sha256": artifact_record(path)["sha256"]}


def validate_feature_execution_replay(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay authority/events/claims/commands/results as one exact bijection."""

    root = Path(run_dir).expanduser().resolve()
    plan_path, plan = load_active_feature_extraction_plan(root)
    pointer_path = root / FEATURE_EXECUTION_POINTER_RELATIVE
    pointer = load_content_manifest(
        pointer_path, name="D1 feature execution pointer", statuses=("COMPLETE",)
    )
    execution_id = str(pointer.get("execution_id", ""))
    directory = execution_directory(root, execution_id)
    execution_path = verified_artifact_path(
        _mapping(pointer.get("execution"), name="D1 feature execution record"),
        name="D1 feature execution authority",
    )
    execution = load_content_manifest(
        execution_path, name="D1 feature execution authority", statuses=("ACTIVE",)
    )
    if execution_path != directory / "execution.json":
        raise RuntimeError("D1 feature execution authority path differs")
    gate_path = verified_artifact_path(
        _mapping(execution.get("resource_gate"), name="D1 feature resource gate"),
        name="D1 feature resource gate",
    )
    raw_resume = execution.get("resume_from")
    resume_from = (
        _mapping(raw_resume, name="D1 feature resume event")
        if raw_resume is not None
        else None
    )
    expected_sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=gate_path,
        resume_from=resume_from,
    )
    event_path = verified_artifact_path(
        _mapping(pointer.get("latest_event"), name="D1 feature COMPLETE event"),
        name="D1 feature COMPLETE event",
    )
    event = load_content_manifest(
        event_path, name="D1 feature COMPLETE event", statuses=("COMPLETE",)
    )
    if (
        execution.get("execution_id") != execution_id
        or execution.get("scope") != FEATURE_RESOURCE_SCOPE
        or execution.get("plan") != artifact_record(plan_path)
        or execution.get("job_count") != FEATURE_JOB_COUNT
        or execution.get("max_parallel") != 1
        or execution.get("device") != "cpu"
        or execution.get("direct_cli_execution_permitted") is not False
        or execution.get("candidate_test_labels_read") is not False
        or execution.get("sources") != expected_sources
        or execution.get("source_signature_sha256")
        != canonical_sha256(expected_sources)
        or event_path.parent != directory / "events"
        or event.get("execution_id") != execution_id
        or event.get("current_job_id") is not None
        or event.get("claim") is not None
        or event.get("failure") is not None
        or pointer.get("completed_jobs") != FEATURE_JOB_COUNT
        or pointer.get("expected_jobs") != FEATURE_JOB_COUNT
        or event.get("completed_jobs") != FEATURE_JOB_COUNT
        or event.get("expected_jobs") != FEATURE_JOB_COUNT
    ):
        raise RuntimeError("D1 feature COMPLETE execution semantics differ")
    verify_artifact_records_recursive(
        expected_sources,
        name="D1 feature execution source closure",
        require_at_least_one=True,
    )
    raw_jobs = plan.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != FEATURE_JOB_COUNT:
        raise RuntimeError("D1 feature plan does not contain exactly nine jobs")
    jobs = {
        str(job["job_id"]): job
        for job in raw_jobs
        if isinstance(job, Mapping) and isinstance(job.get("configuration"), Mapping)
    }
    outputs = _mapping(event.get("outputs"), name="D1 feature COMPLETE outputs")
    inherited = _mapping(
        execution.get("resume_outputs"), name="D1 feature inherited outputs"
    )
    if resume_from is None:
        if inherited:
            raise RuntimeError(
                "D1 feature execution has unauthorised inherited outputs"
            )
    else:
        failed_path = verified_artifact_path(
            resume_from, name="D1 feature failed event"
        )
        failed = load_content_manifest(
            failed_path, name="D1 feature failed event", statuses=("FAILED",)
        )
        if failed.get("outputs") != inherited:
            raise RuntimeError("D1 feature inherited outputs differ from failed event")
    allowed_result_execution_ids = {execution_id} | _resume_lineage_execution_ids(
        root, resume_from
    )
    commands = event.get("commands")
    planned_order = [str(job["job_id"]) for job in raw_jobs]
    expected_command_jobs = [
        job_id for job_id in planned_order if job_id not in inherited
    ]
    expected_indices = [
        index for index, job_id in enumerate(planned_order) if job_id not in inherited
    ]
    if (
        len(jobs) != FEATURE_JOB_COUNT
        or set(outputs) != set(jobs)
        or not set(inherited).issubset(jobs)
        or not isinstance(commands, list)
        or len(commands) != FEATURE_JOB_COUNT - len(inherited)
    ):
        raise RuntimeError("D1 feature job/result/command bijection differs")
    for offset, raw_command in enumerate(commands):
        command = _mapping(raw_command, name=f"D1 feature command {offset}")
        job_id = expected_command_jobs[offset]
        job = jobs[job_id]
        argv = command.get("argv")
        expected_suffix = [*job["worker_argv"], "--run-dir", str(root)]
        if (
            command.get("job_id") != job_id
            or command.get("index") != expected_indices[offset]
            or not isinstance(argv, list)
            or len(argv) != len(expected_suffix) + 1
            or argv[1:] != expected_suffix
            or command.get("returncode") != 0
            or not isinstance(command.get("live_recheck"), Mapping)
        ):
            raise RuntimeError(f"D1 feature command differs for {job_id}")
        claim_path = verified_artifact_path(
            _mapping(command.get("claim"), name=f"D1 feature claim {job_id}"),
            name=f"D1 feature claim {job_id}",
        )
        claim = load_content_manifest(
            claim_path, name=f"D1 feature claim {job_id}", statuses=("CLAIMED",)
        )
        active_event_path = verified_artifact_path(
            _mapping(
                command.get("active_event"), name=f"D1 feature active event {job_id}"
            ),
            name=f"D1 feature active event {job_id}",
        )
        active_event = load_content_manifest(
            active_event_path,
            name=f"D1 feature active event {job_id}",
            statuses=("ACTIVE",),
        )
        if (
            claim_path != directory / "claims" / f"{job_id}.json"
            or claim.get("execution_id") != execution_id
            or claim.get("job_id") != job_id
            or claim.get("command") != argv
            or claim.get("configuration_sha256")
            != canonical_sha256(job["configuration"])
            or active_event_path.parent != directory / "events"
            or active_event.get("execution_id") != execution_id
            or active_event.get("current_job_id") != job_id
            or active_event.get("claim") != artifact_record(claim_path)
        ):
            raise RuntimeError(f"D1 feature claim/event binding differs for {job_id}")
    command_by_job = {
        str(command["job_id"]): command
        for command in commands
        if isinstance(command, Mapping)
    }
    result_records: dict[str, dict[str, str]] = {}
    for job_id in planned_order:
        result_path = verified_artifact_path(
            _mapping(outputs.get(job_id), name=f"D1 feature output {job_id}"),
            name=f"D1 feature output {job_id}",
        )
        expected_path = (root / str(jobs[job_id]["output_manifest"])).resolve()
        if result_path != expected_path:
            raise RuntimeError(f"D1 feature result path differs for {job_id}")
        result = load_content_manifest(
            result_path, name=f"D1 feature result {job_id}", statuses=("COMPLETE",)
        )
        validate_feature_extraction_result(
            result,
            plan_path=plan_path,
            job=jobs[job_id],
            manifest_path=result_path,
        )
        result_sources = _mapping(
            result.get("sources"), name=f"D1 feature result sources {job_id}"
        )
        result_execution_id = result.get("execution_provenance", {}).get(  # type: ignore[union-attr]
            "execution_id"
        )
        if result_execution_id not in allowed_result_execution_ids or (
            job_id not in inherited and result_execution_id != execution_id
        ):
            raise RuntimeError(f"D1 feature result execution differs for {job_id}")
        if job_id not in inherited:
            command = command_by_job[job_id]
            if (
                result_sources.get("execution_manifest")
                != artifact_record(execution_path)
                or result_sources.get("execution_claim") != command.get("claim")
                or result_sources.get("execution_event") != command.get("active_event")
            ):
                raise RuntimeError(
                    f"D1 feature result/command provenance differs for {job_id}"
                )
        result_records[job_id] = artifact_record(result_path)
    access_record = _validate_test_access_events(root, jobs=jobs, outputs=outputs)
    records = {
        "plan": artifact_record(plan_path),
        "execution_pointer": artifact_record(pointer_path),
        "execution_authority": artifact_record(execution_path),
        "complete_event": artifact_record(event_path),
        "results": result_records,
        "test_access_log": access_record,
    }
    return records, {
        "job_count": FEATURE_JOB_COUNT,
        "splits": list(FEATURE_SPLITS),
        "pools": list(FEATURE_POOLS),
        "max_parallel": 1,
        "candidate_test_labels_read": False,
        "exact_job_result_bijection": True,
        "fresh_gated_resume_replayed": resume_from is not None,
    }


def feature_lifecycle_evidence(run_dir: str | Path) -> dict[str, Any]:
    """Return the exact COMPLETE replay subset bound by FEATURES_READY."""

    records, _checks = validate_feature_execution_replay(run_dir)
    return {
        "plan": records["plan"],
        "execution_pointer": records["execution_pointer"],
        "execution_authority": records["execution_authority"],
        "complete_event": records["complete_event"],
        "outputs": records["results"],
    }


__all__ = ["feature_lifecycle_evidence", "validate_feature_execution_replay"]
