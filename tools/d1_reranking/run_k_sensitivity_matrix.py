"""Run the exact 54-job K-sensitivity matrix under one repository-wide lease."""

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

from d1_reranking.execution import (  # noqa: E402
    CANONICAL_RANK1_RUN_DIR,
    artifact_record,
    exclusive_heavy_resource_lease,
    python_invocation_path,
)
from d1_reranking.k_execution import (  # noqa: E402
    K_CLAIM_SHA256_ENV,
    K_EXECUTION_ID_ENV,
    create_job_claim,
    validate_active_execution,
    write_execution_event,
)
from d1_reranking.k_sensitivity import (  # noqa: E402
    load_k_sensitivity_plan,
    validate_k_sensitivity_result,
)
from d1_reranking.resource_gate import (  # noqa: E402
    collect_resource_snapshot,
    evaluate_resource_snapshot,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


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
    expected_path = (root / str(job["output_manifest"])).resolve()
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or value.get("job_id") != job_id:
            continue
        observed_path = Path(str(value.get("manifest", ""))).resolve()
        if observed_path != expected_path or sha256_file(observed_path) != value.get(
            "sha256"
        ):
            raise RuntimeError(f"D1 K job {job_id} returned a stale/wrong manifest")
        result = json.loads(observed_path.read_text(encoding="utf-8"))
        validate_k_sensitivity_result(
            result,
            plan_path=plan_path,
            job=job,
            manifest_path=observed_path,
        )
        return artifact_record(observed_path)
    raise RuntimeError(f"D1 K job {job_id} emitted no exact manifest record")


def _live_recheck(rank1_root: Path, *, prefix: str) -> dict[str, Any]:
    snapshot = collect_resource_snapshot(repo_root=ROOT, rank1_run_dir=rank1_root)
    failures = evaluate_resource_snapshot(snapshot, prefix=prefix)
    if failures:
        raise RuntimeError(f"D1 K live resource recheck failed: {failures}")
    return snapshot


def _validate_bijection(
    *, root: Path, plan_path: Path, plan: dict[str, Any], outputs: dict[str, Any]
) -> None:
    jobs = plan["jobs"]
    job_ids = [str(job["job_id"]) for job in jobs]
    if (
        set(outputs) != set(job_ids)
        or len(outputs) != len(job_ids)
        or len(job_ids) != 54
    ):
        raise RuntimeError("D1 K execution result/job bijection is incomplete")
    paths: set[str] = set()
    for job in jobs:
        job_id = str(job["job_id"])
        expected_path = (root / str(job["output_manifest"])).resolve()
        record = outputs.get(job_id)
        if not isinstance(record, dict) or record != artifact_record(expected_path):
            raise RuntimeError(f"D1 K output record differs for {job_id}")
        if str(expected_path) in paths:
            raise RuntimeError("D1 K jobs collide on one result manifest")
        paths.add(str(expected_path))
        result = json.loads(expected_path.read_text(encoding="utf-8"))
        validate_k_sensitivity_result(
            result,
            plan_path=plan_path,
            job=job,
            manifest_path=expected_path,
        )


def _validate_resume_outputs(
    *,
    root: Path,
    plan_path: Path,
    plan: dict[str, Any],
    execution: dict[str, Any],
    initial_event: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Revalidate inherited cells before skipping them in a fresh execution."""

    raw_outputs = execution.get("resume_outputs")
    event_outputs = initial_event.get("outputs")
    if not isinstance(raw_outputs, dict) or event_outputs != raw_outputs:
        raise RuntimeError("D1 K resume output authority/event binding differs")
    if (
        initial_event.get("sequence") != 0
        or initial_event.get("current_job_id") is not None
        or initial_event.get("claim") is not None
        or initial_event.get("commands") != []
        or initial_event.get("completed_jobs") != len(raw_outputs)
    ):
        raise RuntimeError("D1 K initial ACTIVE resume event differs")
    planned = {str(job["job_id"]): job for job in plan["jobs"] if isinstance(job, dict)}
    outputs: dict[str, dict[str, str]] = {}
    for job_id, record in raw_outputs.items():
        if job_id not in planned or not isinstance(record, dict):
            raise RuntimeError("D1 K resume output falls outside the frozen plan")
        expected_path = (root / str(planned[job_id]["output_manifest"])).resolve()
        if record != artifact_record(expected_path):
            raise RuntimeError(f"D1 K resume output record differs for {job_id}")
        result = json.loads(expected_path.read_text(encoding="utf-8"))
        validate_k_sensitivity_result(
            result,
            plan_path=plan_path,
            job=planned[job_id],
            manifest_path=expected_path,
        )
        outputs[job_id] = artifact_record(expected_path)
    return outputs


def _run_under_lease(
    run_dir: Path, *, python: Path, rank1_run_dir: Path
) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    executable = python_invocation_path(python)
    rank1_root = rank1_run_dir.expanduser().resolve()
    if rank1_root != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("D1 K orchestrator rank1 interlock path differs")
    plan_path = root / "configs/d1_k_sensitivity_plan.json"
    plan = load_k_sensitivity_plan(plan_path)
    execution_path, execution, _event_path, initial_event = validate_active_execution(
        root,
        plan_path=plan_path,
        plan=plan,
        require_fresh_gate=True,
    )
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 54:
        raise RuntimeError("D1 K orchestrator requires the exact 54-job plan")
    _live_recheck(rank1_root, prefix="k_executor_start_live")
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
                raise RuntimeError("D1 K plan contains an invalid job")
            job = {str(key): value for key, value in raw_job.items()}
            job_id = str(job.get("job_id", ""))
            if job_id in outputs:
                continue
            worker_argv = job.get("worker_argv")
            if not isinstance(worker_argv, list) or not all(
                isinstance(item, str) for item in worker_argv
            ):
                raise RuntimeError(f"D1 K job {job_id} worker argv is invalid")
            live = _live_recheck(rank1_root, prefix=f"k_job[{index}]_live")
            command = [str(executable), *worker_argv, "--run-dir", str(root)]
            claim_path, _claim = create_job_claim(
                root,
                execution=execution,
                job=job,
                owner_pid=os.getpid(),
                command=command,
            )
            claim_record = artifact_record(claim_path)
            command_record = {
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
            environment[K_EXECUTION_ID_ENV] = execution_id
            environment[K_CLAIM_SHA256_ENV] = claim_record["sha256"]
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
                    f"D1 K job {job_id} failed ({process.returncode}): "
                    f"{process.stderr[-2000:]}"
                )
            outputs[job_id] = _child_result(
                process.stdout, root=root, plan_path=plan_path, job=job
            )
        _validate_bijection(root=root, plan_path=plan_path, plan=plan, outputs=outputs)
        _event_path, complete = write_execution_event(
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
        return complete
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
    with exclusive_heavy_resource_lease(
        root, purpose="D1 P10 K-sensitivity 54-job execution"
    ):
        return _run_under_lease(root, python=python, rank1_run_dir=rank1_run_dir)


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    pointer_path = root / "configs/d1_k_sensitivity_execution.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="d1_k_sensitivity_matrix_execution",
        route="D1",
        pool="top10_allnms",
        evidence_track="T2_T3",
        method="selected_primary_fixed",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, python=args.python, rank1_run_dir=args.rank1_run_dir)
        state["artifact_path"] = str(pointer_path)
        state["artifact_sha256"] = sha256_file(pointer_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
