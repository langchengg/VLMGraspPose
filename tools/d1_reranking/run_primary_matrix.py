"""Execute or fresh-gated resume the exact serial 360-job D1 primary matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.contracts import RunState  # noqa: E402
from d1_reranking.execution import (  # noqa: E402
    CANONICAL_RANK1_RUN_DIR,
    artifact_record,
    exclusive_heavy_resource_lease,
    python_invocation_path,
)
from d1_reranking.plan import (  # noqa: E402
    PRIMARY_PLAN_POINTER_RELATIVE,
    load_active_primary_plan,
    load_primary_plan,
)
from d1_reranking.primary_execution import (  # noqa: E402
    PRIMARY_CLAIM_SHA256_ENV,
    PRIMARY_EXECUTION_ID_ENV,
    create_job_claim,
    exclusive_json,
    validate_active_execution,
    validate_primary_result,
    write_execution_event,
)
from d1_reranking.resource_gate import (  # noqa: E402
    collect_resource_snapshot,
    evaluate_resource_snapshot,
)
from d1_reranking.run import (  # noqa: E402
    assert_writable_prelock,
    transition_pipeline_status,
)
from unified_reranking.artifacts import verified_artifact_path  # noqa: E402
from unified_reranking.hashing import canonical_sha256, sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / "HiFi_reproduction/.venv-grasp4dof/bin/python",
    )
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )
    return parser.parse_args()


def _child_result(
    stdout: str,
    *,
    root: Path,
    plan_path: Path,
    job: dict[str, Any],
) -> dict[str, str]:
    job_id = str(job["job_id"])
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or value.get("job_id") != job_id:
            continue
        path = Path(str(value.get("manifest", ""))).resolve()
        if sha256_file(path) != value.get("sha256"):
            raise RuntimeError(f"D1 primary job {job_id} returned a stale manifest")
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_primary_result(
            result,
            root=root,
            plan_path=plan_path,
            job=job,
            manifest_path=path,
        )
        return artifact_record(path)
    raise RuntimeError(f"D1 primary job {job_id} emitted no exact manifest record")


def _live_recheck(rank1_root: Path, *, prefix: str) -> dict[str, Any]:
    snapshot = collect_resource_snapshot(repo_root=ROOT, rank1_run_dir=rank1_root)
    failures = evaluate_resource_snapshot(snapshot, prefix=prefix)
    if failures:
        raise RuntimeError(f"D1 primary live resource recheck failed: {failures}")
    return snapshot


def _planned_python(plan: dict[str, Any]) -> Path:
    sources = plan.get("sources")
    environment = sources.get("environment") if isinstance(sources, dict) else None
    record = (
        environment.get("python_executable") if isinstance(environment, dict) else None
    )
    return verified_artifact_path(record or {}, name="D1 primary planned Python")


def _validate_resume_outputs(
    *,
    root: Path,
    plan_path: Path,
    plan: dict[str, Any],
    execution: dict[str, Any],
    initial_event: dict[str, Any],
) -> dict[str, dict[str, str]]:
    raw = execution.get("resume_outputs")
    if not isinstance(raw, dict) or initial_event.get("outputs") != raw:
        raise RuntimeError("D1 primary resume output authority/event binding differs")
    if (
        initial_event.get("sequence") != 0
        or initial_event.get("current_job_id") is not None
        or initial_event.get("claim") is not None
        or initial_event.get("commands") != []
        or initial_event.get("completed_jobs") != len(raw)
    ):
        raise RuntimeError("D1 primary initial ACTIVE resume event differs")
    jobs = {str(job["job_id"]): job for job in plan["jobs"] if isinstance(job, dict)}
    outputs: dict[str, dict[str, str]] = {}
    for job_id, record in raw.items():
        if job_id not in jobs or not isinstance(record, dict):
            raise RuntimeError("D1 primary inherited output is outside the plan")
        path = verified_artifact_path(record, name=f"D1 primary inherited {job_id}")
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_primary_result(
            value,
            root=root,
            plan_path=plan_path,
            job=jobs[job_id],
            manifest_path=path,
        )
        outputs[job_id] = artifact_record(path)
    return outputs


def _validate_bijection(
    *, root: Path, plan_path: Path, plan: dict[str, Any], outputs: dict[str, Any]
) -> None:
    jobs = plan["jobs"]
    job_ids = [str(job["job_id"]) for job in jobs]
    if len(job_ids) != 360 or len(set(job_ids)) != 360 or set(outputs) != set(job_ids):
        raise RuntimeError("D1 primary execution result/job bijection is incomplete")
    paths: set[str] = set()
    for job in jobs:
        job_id = str(job["job_id"])
        record = outputs.get(job_id)
        if not isinstance(record, dict):
            raise RuntimeError(f"D1 primary output record is absent for {job_id}")
        path = verified_artifact_path(record, name=f"D1 primary output {job_id}")
        if str(path) in paths:
            raise RuntimeError("D1 primary jobs collide on one result manifest")
        paths.add(str(path))
        validate_primary_result(
            json.loads(path.read_text(encoding="utf-8")),
            root=root,
            plan_path=plan_path,
            job=job,
            manifest_path=path,
        )


def _run_under_lease(
    run_dir: Path, *, python: Path, rank1_run_dir: Path
) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    executable = python_invocation_path(python)
    rank1_root = rank1_run_dir.expanduser().resolve()
    if rank1_root != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("D1 primary runner rank1 interlock path differs")
    if (root / PRIMARY_PLAN_POINTER_RELATIVE).exists():
        plan_path, plan = load_active_primary_plan(root)
    else:
        plan_path = root / "configs/d1_primary_matrix_plan.json"
        plan = load_primary_plan(plan_path)
    summary_path = root / "07_validation/primary_matrix_execution.json"
    if summary_path.exists():
        raise RuntimeError("D1 immutable primary COMPLETE summary already exists")
    if artifact_record(executable) != artifact_record(_planned_python(plan)):
        raise RuntimeError("D1 primary runner Python differs from the frozen plan")
    execution_path, execution, _event_path, initial_event = validate_active_execution(
        root,
        plan_path=plan_path,
        plan=plan,
        require_fresh_gate=True,
    )
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 360:
        raise RuntimeError("D1 primary orchestrator requires the exact 360-job plan")
    _live_recheck(rank1_root, prefix="primary_executor_start_live")
    execution_id = str(execution["execution_id"])
    outputs: dict[str, Any] = _validate_resume_outputs(
        root=root,
        plan_path=plan_path,
        plan=plan,
        execution=execution,
        initial_event=initial_event,
    )
    commands: list[dict[str, Any]] = []
    sequence = int(initial_event.get("sequence", 0)) + 1
    try:
        for index, raw_job in enumerate(jobs):
            if not isinstance(raw_job, dict):
                raise RuntimeError("D1 primary plan contains an invalid job")
            job = {str(key): value for key, value in raw_job.items()}
            job_id = str(job.get("job_id", ""))
            if job_id in outputs:
                continue
            worker_argv = job.get("worker_argv")
            if not isinstance(worker_argv, list) or not all(
                isinstance(item, str) for item in worker_argv
            ):
                raise RuntimeError(f"D1 primary job {job_id} argv differs")
            live = _live_recheck(rank1_root, prefix=f"primary_job[{index}]_live")
            command = [str(executable), *worker_argv, "--run-dir", str(root)]
            claim_path, _claim = create_job_claim(
                root,
                execution=execution,
                job=job,
                owner_pid=os.getpid(),
                command=command,
            )
            claim_record = artifact_record(claim_path)
            command_record: dict[str, Any] = {
                "index": index,
                "job_id": job_id,
                "argv": command,
                "claim": claim_record,
                "live_recheck": live,
            }
            commands.append(command_record)
            write_execution_event(
                root,
                execution=execution,
                sequence=sequence,
                status="ACTIVE",
                owner_pid=os.getpid(),
                current_job_id=job_id,
                claim=claim_record,
                outputs=outputs,
                commands=commands,
            )
            sequence += 1
            environment = dict(os.environ)
            for name in THREAD_ENVIRONMENT:
                environment[name] = "1"
            environment["PYTHONHASHSEED"] = "0"
            environment[PRIMARY_EXECUTION_ID_ENV] = execution_id
            environment[PRIMARY_CLAIM_SHA256_ENV] = claim_record["sha256"]
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            command_record["returncode"] = process.returncode
            if process.returncode != 0:
                raise RuntimeError(
                    f"D1 primary job {job_id} failed ({process.returncode}): "
                    f"{process.stderr[-2000:]}"
                )
            outputs[job_id] = _child_result(
                process.stdout,
                root=root,
                plan_path=plan_path,
                job=job,
            )
        _validate_bijection(root=root, plan_path=plan_path, plan=plan, outputs=outputs)
        complete_event_path, _complete = write_execution_event(
            root,
            execution=execution,
            sequence=sequence,
            status="COMPLETE",
            owner_pid=os.getpid(),
            current_job_id=None,
            claim=None,
            outputs=outputs,
            commands=commands,
        )
        summary: dict[str, Any] = {
            "schema_version": 2,
            "status": "COMPLETE",
            "execution_id": execution_id,
            "plan": artifact_record(plan_path),
            "execution_authority": artifact_record(execution_path),
            "complete_event": artifact_record(complete_event_path),
            "expected_jobs": 360,
            "completed_jobs": 360,
            "outputs": outputs,
            "commands": commands,
            "max_parallel": 1,
            "device": "cpu",
            "candidate_test_labels_read": False,
            "test_inputs_referenced": False,
        }
        summary["content_sha256"] = canonical_sha256(summary)
        exclusive_json(summary_path, summary)
        transition_pipeline_status(
            root,
            status=RunState.VALIDATION_SCREEN,
            first_incomplete_stage="P13_PRELOCK_READINESS",
            formal_test_executed=False,
            test_candidate_labels_read=False,
            formal_test_execution_count=0,
        )
        return summary
    except BaseException as error:
        write_execution_event(
            root,
            execution=execution,
            sequence=sequence,
            status="FAILED",
            owner_pid=os.getpid(),
            current_job_id=None,
            claim=None,
            outputs=outputs,
            commands=commands,
            failure=f"{type(error).__name__}: {error}",
        )
        raise


def run(run_dir: Path, *, python: Path, rank1_run_dir: Path) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with exclusive_heavy_resource_lease(root, purpose="D1 P9 primary 360-job matrix"):
        return _run_under_lease(root, python=python, rank1_run_dir=rank1_run_dir)


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    output_path = root / "07_validation/primary_matrix_execution.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P9",
        substage="d1_primary_matrix_execution",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R2-R6",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, python=args.python, rank1_run_dir=args.rank1_run_dir)
        state["artifact_path"] = str(output_path)
        state["artifact_sha256"] = sha256_file(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
