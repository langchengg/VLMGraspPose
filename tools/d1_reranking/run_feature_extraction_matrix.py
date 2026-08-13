"""Run the exact nine raw-feature jobs serially under one global lease."""

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
from d1_reranking.feature_execution import (  # noqa: E402
    FEATURE_CLAIM_SHA256_ENV,
    FEATURE_EXECUTION_ID_ENV,
    FEATURE_EXECUTION_POINTER_RELATIVE,
    FEATURE_JOB_COUNT,
    create_job_claim,
    validate_active_execution,
    write_execution_event,
)
from d1_reranking.feature_plan import (  # noqa: E402
    load_active_feature_extraction_plan,
    validate_feature_extraction_result,
)
from d1_reranking.feature_replay import feature_lifecycle_evidence  # noqa: E402
from d1_reranking.contracts import RunState  # noqa: E402
from d1_reranking.resource_gate import (  # noqa: E402
    collect_resource_snapshot,
    evaluate_resource_snapshot,
)
from d1_reranking.run import (  # noqa: E402
    assert_writable_prelock,
    transition_pipeline_status,
)
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


def _live_recheck(rank1_root: Path, *, prefix: str) -> dict[str, Any]:
    snapshot = collect_resource_snapshot(repo_root=ROOT, rank1_run_dir=rank1_root)
    failures = evaluate_resource_snapshot(snapshot, prefix=prefix)
    if failures:
        raise RuntimeError(f"D1 feature live resource recheck failed: {failures}")
    return snapshot


def _child_result(
    stdout: str, *, root: Path, plan_path: Path, job: dict[str, Any]
) -> dict[str, str]:
    job_id = str(job["job_id"])
    expected = (root / str(job["output_manifest"])).resolve()
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or value.get("job_id") != job_id:
            continue
        observed = Path(str(value.get("manifest", ""))).resolve()
        if observed != expected or value.get("sha256") != sha256_file(observed):
            raise RuntimeError(f"D1 feature job {job_id} returned a stale manifest")
        result = json.loads(observed.read_text(encoding="utf-8"))
        validate_feature_extraction_result(
            result, plan_path=plan_path, job=job, manifest_path=observed
        )
        return artifact_record(observed)
    raise RuntimeError(f"D1 feature job {job_id} emitted no exact result record")


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
        raise RuntimeError("D1 feature resume authority/event outputs differ")
    if (
        initial_event.get("sequence") != 0
        or initial_event.get("current_job_id") is not None
        or initial_event.get("claim") is not None
        or initial_event.get("commands") != []
        or initial_event.get("completed_jobs") != len(raw)
    ):
        raise RuntimeError("D1 feature initial ACTIVE resume event differs")
    planned = {str(job["job_id"]): job for job in plan["jobs"] if isinstance(job, dict)}
    outputs: dict[str, dict[str, str]] = {}
    for job_id, record in raw.items():
        if job_id not in planned or not isinstance(record, dict):
            raise RuntimeError("D1 feature inherited output is outside the plan")
        expected = (root / str(planned[job_id]["output_manifest"])).resolve()
        if record != artifact_record(expected):
            raise RuntimeError(f"D1 feature inherited output differs for {job_id}")
        result = json.loads(expected.read_text(encoding="utf-8"))
        validate_feature_extraction_result(
            result,
            plan_path=plan_path,
            job=planned[job_id],
            manifest_path=expected,
        )
        outputs[job_id] = artifact_record(expected)
    return outputs


def _validate_bijection(
    *, root: Path, plan_path: Path, plan: dict[str, Any], outputs: dict[str, Any]
) -> None:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != FEATURE_JOB_COUNT:
        raise RuntimeError("D1 feature execution requires exactly nine jobs")
    job_ids = [str(job["job_id"]) for job in jobs]
    if set(outputs) != set(job_ids) or len(outputs) != FEATURE_JOB_COUNT:
        raise RuntimeError("D1 feature job/result bijection is incomplete")
    paths: set[str] = set()
    for job in jobs:
        job_id = str(job["job_id"])
        expected = (root / str(job["output_manifest"])).resolve()
        if outputs.get(job_id) != artifact_record(expected) or str(expected) in paths:
            raise RuntimeError(f"D1 feature output record/path differs for {job_id}")
        paths.add(str(expected))
        result = json.loads(expected.read_text(encoding="utf-8"))
        validate_feature_extraction_result(
            result, plan_path=plan_path, job=job, manifest_path=expected
        )


def _run_under_lease(
    run_dir: Path, *, python: Path, rank1_run_dir: Path
) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    executable = python_invocation_path(python)
    rank1_root = rank1_run_dir.expanduser().resolve()
    if rank1_root != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("D1 feature orchestrator rank1 path differs")
    plan_path, plan = load_active_feature_extraction_plan(root)
    planned_python = (
        plan.get("sources", {})
        .get("environment", {})
        .get(  # type: ignore[union-attr]
            "python_executable"
        )
    )
    if planned_python != artifact_record(executable):
        raise RuntimeError(
            "D1 feature orchestrator Python differs from the frozen plan"
        )
    _execution_path, execution, _event_path, initial_event = validate_active_execution(
        root,
        plan_path=plan_path,
        plan=plan,
        require_fresh_gate=True,
    )
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != FEATURE_JOB_COUNT:
        raise RuntimeError("D1 feature orchestrator requires exactly nine jobs")
    _live_recheck(rank1_root, prefix="feature_executor_start_live")
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
                raise RuntimeError("D1 feature plan contains an invalid job")
            job = {str(key): value for key, value in raw_job.items()}
            job_id = str(job["job_id"])
            if job_id in outputs:
                continue
            assert_writable_prelock(root)
            worker_argv = job.get("worker_argv")
            if not isinstance(worker_argv, list) or not all(
                isinstance(item, str) for item in worker_argv
            ):
                raise RuntimeError(f"D1 feature job {job_id} argv is invalid")
            live = _live_recheck(rank1_root, prefix=f"feature_job[{index}]_live")
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
            assert_writable_prelock(root)
            active_event_path, _active_event = write_execution_event(
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
            command_record["active_event"] = artifact_record(active_event_path)
            sequence += 1
            environment = dict(os.environ)
            environment[FEATURE_EXECUTION_ID_ENV] = execution_id
            environment[FEATURE_CLAIM_SHA256_ENV] = claim_record["sha256"]
            assert_writable_prelock(root)
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
                    f"D1 feature job {job_id} failed ({process.returncode}): "
                    f"{process.stderr[-2000:]}"
                )
            outputs[job_id] = _child_result(
                process.stdout, root=root, plan_path=plan_path, job=job
            )
        assert_writable_prelock(root)
        _validate_bijection(root=root, plan_path=plan_path, plan=plan, outputs=outputs)
        _path, complete = write_execution_event(
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
    except BaseException as error:
        assert_writable_prelock(root)
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
    assert_writable_prelock(root)
    feature_lifecycle_evidence(root)
    transition_pipeline_status(
        root,
        status=RunState.FEATURES_READY,
        first_incomplete_stage="P7_SPLITS_AND_OOF",
        formal_test_executed=False,
        test_candidate_labels_read=False,
        formal_test_execution_count=0,
    )
    return complete


def run(run_dir: Path, *, python: Path, rank1_run_dir: Path) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with exclusive_heavy_resource_lease(
        root, purpose="D1 P5 raw matched-common feature extraction (9 jobs)"
    ):
        return _run_under_lease(root, python=python, rank1_run_dir=rank1_run_dir)


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    pointer = root / FEATURE_EXECUTION_POINTER_RELATIVE
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P5",
        substage="d1_raw_feature_extraction_matrix",
        route="D1",
        pool="top5_top10_allnms",
        evidence_track="matched_common_raw",
        method="unified_common_extractor",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, python=args.python, rank1_run_dir=args.rank1_run_dir)
        state["artifact_path"] = str(pointer)
        state["artifact_sha256"] = sha256_file(pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
