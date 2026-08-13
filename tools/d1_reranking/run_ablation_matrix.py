"""Execute the frozen P10 ablation matrix serially under one global lease."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.ablation import (  # noqa: E402
    ABLATION_EXECUTION_POINTER_RELATIVE,
    ABLATION_PLAN_RELATIVE,
    load_ablation_plan,
)
from d1_reranking.ablation_execution import (  # noqa: E402
    ABLATION_CLAIM_SHA256_ENV,
    ABLATION_EXECUTION_ID_ENV,
    ablation_execution_scope,
    create_job_claim,
    execution_directory,
    load_ablation_resource_policy,
    write_execution_event,
)
from d1_reranking.execution import (  # noqa: E402
    CANONICAL_RANK1_RUN_DIR,
    artifact_record,
    exclusive_heavy_resource_lease,
    load_content_manifest,
    python_invocation_path,
    validate_resource_gate,
)
from d1_reranking.resource_gate import (  # noqa: E402
    collect_resource_snapshot,
    evaluate_resource_snapshot,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import verified_artifact_path  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--rank1-run-dir", type=Path, default=CANONICAL_RANK1_RUN_DIR)
    return parser.parse_args()


def _load_authority(
    root: Path, plan: dict[str, object]
) -> tuple[Path, dict[str, object], dict[str, object]]:
    pointer = load_content_manifest(
        root / ABLATION_EXECUTION_POINTER_RELATIVE,
        name="D1 P10 execution pointer",
        statuses=("ACTIVE",),
    )
    execution_path = verified_artifact_path(
        pointer["execution"], name="D1 P10 authority"
    )
    execution = load_content_manifest(
        execution_path, name="D1 P10 authority", statuses=("ACTIVE",)
    )
    latest_path = verified_artifact_path(pointer["latest_event"], name="D1 P10 event")
    event = load_content_manifest(
        latest_path, name="D1 P10 event", statuses=("ACTIVE",)
    )
    if (
        execution_path
        != execution_directory(root, str(execution["execution_id"])) / "execution.json"
        or execution.get("scope") != ablation_execution_scope(plan)
        or execution.get("plan") != artifact_record(root / ABLATION_PLAN_RELATIVE)
        or execution.get("job_ids_sha256") != plan.get("job_ids_sha256")
        or event.get("execution_id") != execution.get("execution_id")
        or event.get("current_job_id") is not None
        or event.get("claim") is not None
    ):
        raise RuntimeError("D1 P10 ACTIVE authority differs")
    return execution_path, execution, event


def run(run_dir: Path, *, python: Path, rank1_run_dir: Path) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    executable = python_invocation_path(python)
    rank1 = rank1_run_dir.expanduser().resolve()
    if rank1 != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("D1 P10 runner rank1 interlock differs")
    plan_path = root / ABLATION_PLAN_RELATIVE
    plan = load_ablation_plan(plan_path)
    _execution_path, execution, initial = _load_authority(root, plan)
    policy_path, _policy = load_ablation_resource_policy(
        root, plan_path=plan_path, plan=plan
    )
    validate_resource_gate(
        verified_artifact_path(execution["resource_gate"], name="D1 P10 gate"),
        run_dir=root,
        plan_path=plan_path,
        policy_path=policy_path,
        expected_scope=ablation_execution_scope(plan),
        require_fresh=True,
    )
    jobs = list(plan["jobs"])
    outputs = dict(initial.get("outputs", {}))
    commands: list[dict[str, object]] = []
    sequence = 1
    for index, job_value in enumerate(jobs):
        job = dict(job_value)
        job_id = str(job["job_id"])
        if job_id in outputs:
            continue
        live = collect_resource_snapshot(repo_root=ROOT, rank1_run_dir=rank1)
        failures = evaluate_resource_snapshot(live, prefix=f"ablation_job_{index}")
        if failures:
            write_execution_event(
                root,
                execution=execution,
                sequence=sequence,
                status="FAILED",
                owner_pid=os.getpid(),
                current_job_id=job_id,
                claim=None,
                outputs=outputs,
                commands=commands,
                failure=f"live resource recheck failed: {failures}",
            )
            raise RuntimeError(f"D1 P10 live resource recheck failed: {failures}")
        argv = [
            str(executable),
            *map(str, job["worker_argv"]),
            "--run-dir",
            str(root),
        ]
        claim_path, _claim = create_job_claim(
            root,
            execution=execution,
            job=job,
            owner_pid=os.getpid(),
            command=argv,
        )
        claim_record = artifact_record(claim_path)
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
        environment = dict(os.environ)
        environment[ABLATION_EXECUTION_ID_ENV] = str(execution["execution_id"])
        environment[ABLATION_CLAIM_SHA256_ENV] = str(claim_record["sha256"])
        completed = subprocess.run(
            argv, cwd=ROOT, env=environment, check=False, text=True
        )
        command = {
            "index": index,
            "job_id": job_id,
            "argv": argv,
            "returncode": completed.returncode,
            "claim": claim_record,
            "live_recheck": live,
        }
        commands.append(command)
        if completed.returncode != 0:
            write_execution_event(
                root,
                execution=execution,
                sequence=sequence + 1,
                status="FAILED",
                owner_pid=os.getpid(),
                current_job_id=job_id,
                claim=claim_record,
                outputs=outputs,
                commands=commands,
                failure=f"worker return code {completed.returncode}",
            )
            raise RuntimeError(f"D1 P10 worker failed: {job_id}")
        result_path = (root / str(job["output_manifest"])).resolve()
        result = load_content_manifest(
            result_path, name=f"D1 P10 cell {job_id}", statuses=("COMPLETE",)
        )
        if (
            result.get("job_id") != job_id
            or result.get("configuration") != job["configuration"]
        ):
            raise RuntimeError(f"D1 P10 worker output differs: {job_id}")
        outputs[job_id] = artifact_record(result_path)
        sequence += 2
        write_execution_event(
            root,
            execution=execution,
            sequence=sequence,
            status="ACTIVE",
            owner_pid=os.getpid(),
            current_job_id=None,
            claim=None,
            outputs=outputs,
            commands=commands,
        )
        sequence += 1
    _path, complete = write_execution_event(
        root,
        execution=execution,
        sequence=sequence,
        status="COMPLETE",
        owner_pid=None,
        current_job_id=None,
        claim=None,
        outputs=outputs,
        commands=commands,
    )
    return complete


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="d1_ablation_serial_matrix",
        route="D1",
        pool="top5",
        method="selected_primary_fixed",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        with exclusive_heavy_resource_lease(
            root, purpose="D1 P10 evidence and feature ablations"
        ):
            event = run(root, python=args.python, rank1_run_dir=args.rank1_run_dir)
        pointer = root / ABLATION_EXECUTION_POINTER_RELATIVE
        state["artifact_path"] = str(pointer.resolve())
        state["artifact_sha256"] = sha256_file(pointer)
    print(
        json.dumps(
            {"status": event["status"], "completed_jobs": event["completed_jobs"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
