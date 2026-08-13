"""Plan-bound P12 Validation producer policy, authorization, and serial runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping


# This entry point imports the plan contract, which transitively imports NumPy.
# Pin native runtimes before any project import so the worker cannot initialize
# a wider pool before the producer's own single-thread safeguards run.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import (  # noqa: E402
    CANONICAL_RANK1_RUN_DIR,
    artifact_record,
    exclusive_heavy_resource_lease,
    load_content_manifest,
    python_invocation_path,
)
from d1_reranking.four_route_execution import (  # noqa: E402
    P12_CLAIM_SHA256_ENV,
    P12_EXECUTION_ID_ENV,
    P12_EXECUTION_POINTER_RELATIVE,
    create_execution_authority,
    create_job_claim,
    execution_directory,
    execution_scope,
    load_four_route_execution_plan,
    validate_gate_for_execution,
    validate_worker_context,
    write_execution_event,
    write_resource_policy,
)
from d1_reranking.resource_gate import (  # noqa: E402
    collect_resource_snapshot,
    evaluate_resource_snapshot,
)
from d1_reranking.k_execution import exclusive_json  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import canonical_sha256, sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


WorkerExecutor = Callable[[Path, Mapping[str, Any], Path, Path, Path], Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    policy = subparsers.add_parser("write-policy")
    policy.add_argument("--run-dir", required=True, type=Path)
    policy.add_argument("--resume", action="store_true")
    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--run-dir", required=True, type=Path)
    authorize.add_argument("--gate-manifest", required=True, type=Path)
    authorize.add_argument("--owner", required=True)
    authorize.add_argument(
        "--rank1-run-dir", type=Path, default=CANONICAL_RANK1_RUN_DIR
    )
    execute = subparsers.add_parser("run")
    execute.add_argument("--run-dir", required=True, type=Path)
    execute.add_argument("--python", type=Path, default=Path(sys.executable))
    execute.add_argument("--rank1-run-dir", type=Path, default=CANONICAL_RANK1_RUN_DIR)
    worker = subparsers.add_parser("_worker")
    worker.add_argument("--run-dir", required=True, type=Path)
    worker.add_argument("--job-id", required=True)
    return parser.parse_args()


def _job_result_path(root: Path, job_id: str) -> Path:
    return root / "13_four_route_extension/execution_results" / f"{job_id}.json"


def _producer_output_path(root: Path, stage: str) -> Path:
    paths = {
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
    try:
        return paths[stage]
    except KeyError as error:
        raise RuntimeError(f"unsupported P12 Validation job stage: {stage}") from error


def write_job_result(
    root: Path,
    *,
    job: Mapping[str, Any],
    plan_path: Path,
    execution_path: Path,
    claim_path: Path,
    producer_manifest_path: Path,
    resume: bool,
) -> Path:
    configuration = dict(job["configuration"])
    job_id = str(job["job_id"])
    producer = load_content_manifest(
        producer_manifest_path,
        name=f"P12 {configuration['stage']} producer",
        statuses=("COMPLETE", "VALIDATION_LOCKED"),
    )
    if (
        producer.get("candidate_test_labels_read") is not False
        or producer.get("selection_used_test_metrics") is True
    ):
        raise PermissionError("P12 Validation producer violates Test isolation")
    verify_artifact_records_recursive(
        producer.get("artifacts"),
        name="P12 producer artifacts",
        require_at_least_one=True,
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "job_id": job_id,
        "configuration": configuration,
        "sources": {
            "plan": artifact_record(plan_path),
            "execution_authority": artifact_record(execution_path),
            "execution_claim": artifact_record(claim_path),
        },
        "artifacts": {"producer_manifest": artifact_record(producer_manifest_path)},
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    value["content_sha256"] = canonical_sha256(value)
    path = _job_result_path(root, job_id)
    if path.exists():
        existing = load_content_manifest(
            path, name="P12 job result", statuses=("COMPLETE",)
        )
        if resume and existing == value:
            return path
        raise RuntimeError("immutable P12 job result exists and differs")
    exclusive_json(path, value)
    return path


def _record_path(record: Mapping[str, Any], *, name: str) -> Path:
    return verified_artifact_path(record, name=name)


def run_worker(root: Path, *, job_id: str, resource_lease_path: Path | None) -> Path:
    assert_writable_prelock(root)
    plan_path, plan, execution_path, claim_path = validate_worker_context(
        root, job_id=job_id, lease_path=resource_lease_path
    )
    producer = plan["sources"]["validation_producer"]
    jobs = {str(job["job_id"]): job for job in producer["jobs"]}
    if job_id not in jobs:
        raise RuntimeError("P12 worker job is not in the frozen plan")
    job = jobs[job_id]
    stage = str(job["configuration"]["stage"])
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    source_paths = {
        "p12_plan": plan_path,
        "producer_spec": _record_path(producer["spec"], name="P12 producer spec"),
        "producer_runner": Path(__file__).resolve(),
    }
    if stage == "router_validation":
        from d1_reranking.four_route_producer import run_four_route_router_validation

        router = producer["router"]
        run_four_route_router_validation(
            train_oof_path=_record_path(
                router["train_oof"], name="P12 router Train OOF"
            ),
            validation_path=_record_path(
                router["validation"], name="P12 router Validation"
            ),
            feature_columns=router["feature_columns"],
            output_dir=root / "13_four_route_extension/router",
            source_paths=source_paths,
            resume=True,
        )
    elif stage == "top20_union_validation":
        from d1_reranking.four_route_producer import run_top20_union_validation

        union = producer["union"]
        run_top20_union_validation(
            train_path=_record_path(union["train_top20"], name="P12 union Train"),
            validation_path=_record_path(
                union["validation_top20"], name="P12 union Validation"
            ),
            folds_path=_record_path(union["folds"], name="P12 union folds"),
            train_denominator_path=_record_path(
                union["train_denominator"], name="P12 union Train denominator"
            ),
            validation_denominator_path=_record_path(
                union["validation_denominator"], name="P12 union Validation denominator"
            ),
            feature_columns=union["feature_columns"],
            output_dir=root / "13_four_route_extension/union",
            source_paths=source_paths,
            resume=True,
        )
    elif stage in {"t4_train", "t4_validation"}:
        from d1_reranking.four_route_t4 import write_t4_features

        split = stage.removeprefix("t4_")
        t4 = producer["t4"][split]
        write_t4_features(
            split=split,
            d1_top5_path=_record_path(t4["d1_top5"], name=f"P12 {split} D1 Top5"),
            t3_manifest_path=_record_path(t4["t3_manifest"], name=f"P12 {split} T3"),
            peer_top5_paths={
                route: _record_path(record, name=f"P12 {split} {route} Top5")
                for route, record in t4["peer_top5"].items()
            },
            output_dir=root / f"03_features/{split}/top5/T4_four_route_consensus",
            source_paths=source_paths,
            resume=True,
        )
    elif stage == "validation_summary":
        from d1_reranking.four_route_producer import finalize_p12_validation

        union = producer["union"]
        router = producer["router"]
        finalize_p12_validation(
            run_dir=root,
            router_selection_path=_producer_output_path(root, "router_validation"),
            router_validation_input_path=_record_path(
                router["validation"], name="P12 router Validation"
            ),
            union_selection_path=_producer_output_path(root, "top20_union_validation"),
            validation_top20_path=_record_path(
                union["validation_top20"], name="P12 union Validation"
            ),
            source_paths=source_paths,
            resume=True,
        )
    else:  # pragma: no cover - plan validation rejects this first
        raise RuntimeError(f"unsupported P12 Validation job: {stage}")
    return write_job_result(
        root,
        job=job,
        plan_path=plan_path,
        execution_path=execution_path,
        claim_path=claim_path,
        producer_manifest_path=_producer_output_path(root, stage),
        resume=True,
    )


def authorize(
    root: Path,
    *,
    gate_manifest: Path,
    owner: str,
    rank1_run_dir: Path,
    collect_snapshot: Callable[..., dict[str, Any]] = collect_resource_snapshot,
    evaluate_snapshot: Callable[..., list[str]] = evaluate_resource_snapshot,
) -> dict[str, Any]:
    assert_writable_prelock(root)
    rank1 = rank1_run_dir.expanduser().resolve()
    if rank1 != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("P12 authorization rank1 interlock differs")
    plan_path, plan = load_four_route_execution_plan(root)
    pointer_path = root / P12_EXECUTION_POINTER_RELATIVE
    resume_from = None
    resume_outputs: dict[str, Any] = {}
    if pointer_path.exists():
        prior = load_content_manifest(
            pointer_path,
            name="prior P12 execution",
            statuses=("ACTIVE", "COMPLETE", "FAILED"),
        )
        if prior["status"] != "FAILED":
            raise RuntimeError(f"P12 execution is already {prior['status']}")
        failed_path = verified_artifact_path(
            prior["latest_event"], name="P12 failed event"
        )
        failed = load_content_manifest(
            failed_path, name="P12 failed event", statuses=("FAILED",)
        )
        resume_from = artifact_record(failed_path)
        resume_outputs = dict(failed.get("outputs", {}))
        verify_artifact_records_recursive(
            resume_outputs,
            name="P12 recovery outputs",
            require_at_least_one=bool(resume_outputs),
        )
    gate_path = gate_manifest.expanduser().resolve()
    validate_gate_for_execution(root, gate_path=gate_path, require_fresh=True)
    live = collect_snapshot(repo_root=ROOT, rank1_run_dir=rank1)
    failures = evaluate_snapshot(live, prefix="p12_validation_authorization_live")
    if failures:
        raise RuntimeError(f"P12 live resource recheck failed: {failures}")
    execution_path, execution = create_execution_authority(
        root,
        plan_path=plan_path,
        plan=plan,
        resource_gate_path=gate_path,
        owner=owner,
        resume_from=resume_from,
        resume_outputs=resume_outputs,
    )
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
    return execution


def _default_worker(
    root: Path,
    job: Mapping[str, Any],
    claim_path: Path,
    execution_path: Path,
    lease_path: Path,
    *,
    python: Path,
) -> Path:
    executable = python_invocation_path(python)
    argv = [
        str(executable),
        str(Path(__file__).resolve()),
        "_worker",
        "--run-dir",
        str(root),
        "--job-id",
        str(job["job_id"]),
    ]
    environment = dict(os.environ)
    execution = load_content_manifest(
        execution_path, name="P12 execution", statuses=("ACTIVE",)
    )
    environment[P12_EXECUTION_ID_ENV] = str(execution["execution_id"])
    environment[P12_CLAIM_SHA256_ENV] = artifact_record(claim_path)["sha256"]
    completed = subprocess.run(argv, cwd=ROOT, env=environment, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"P12 worker returned {completed.returncode}")
    return _job_result_path(root, str(job["job_id"]))


def run_serial(
    root: Path,
    *,
    python: Path,
    rank1_run_dir: Path,
    resource_lease_path: Path,
    worker_executor: WorkerExecutor | None = None,
    collect_snapshot: Callable[..., dict[str, Any]] = collect_resource_snapshot,
    evaluate_snapshot: Callable[..., list[str]] = evaluate_resource_snapshot,
) -> dict[str, Any]:
    assert_writable_prelock(root)
    executable = python_invocation_path(python)
    rank1 = rank1_run_dir.expanduser().resolve()
    if rank1 != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("P12 runner rank1 interlock differs")
    expected_lease = (root.parent / ".d1_heavy_resource.lock").resolve()
    if resource_lease_path.resolve() != expected_lease:
        raise PermissionError("P12 serial runner requires the global heavy lease")
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
    initial_path = verified_artifact_path(
        pointer["latest_event"], name="P12 active event"
    )
    initial = load_content_manifest(
        initial_path, name="P12 active event", statuses=("ACTIVE",)
    )
    if (
        execution.get("plan") != artifact_record(plan_path)
        or execution.get("scope") != execution_scope(plan)
        or initial.get("current_job_id") is not None
        or initial.get("claim") is not None
    ):
        raise RuntimeError("P12 ACTIVE authority differs")
    validate_gate_for_execution(
        root,
        gate_path=verified_artifact_path(
            execution["resource_gate"], name="P12 resource gate"
        ),
        require_fresh=True,
    )
    jobs = list(plan["sources"]["validation_producer"]["jobs"])
    outputs = dict(initial.get("outputs", {}))
    commands: list[dict[str, Any]] = []
    sequence = 1
    for index, job in enumerate(jobs):
        job_id = str(job["job_id"])
        if job_id in outputs:
            continue
        live = collect_snapshot(repo_root=ROOT, rank1_run_dir=rank1)
        failures = evaluate_snapshot(live, prefix=f"p12_validation_job_{index}")
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
            raise RuntimeError(f"P12 live resource recheck failed: {failures}")
        command = [
            str(executable),
            str(Path(__file__).resolve()),
            "_worker",
            "--run-dir",
            str(root),
            "--job-id",
            job_id,
        ]
        claim_path, _claim = create_job_claim(
            root, execution=execution, job=job, owner_pid=os.getpid(), command=command
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
        try:
            if worker_executor is None:
                result_path = _default_worker(
                    root,
                    job,
                    claim_path,
                    execution_path,
                    resource_lease_path,
                    python=python,
                )
            else:
                result_path = worker_executor(
                    root, job, claim_path, execution_path, resource_lease_path
                )
            result = load_content_manifest(
                result_path, name=f"P12 job {job_id}", statuses=("COMPLETE",)
            )
            if (
                result.get("job_id") != job_id
                or result.get("configuration") != job["configuration"]
            ):
                raise RuntimeError("P12 worker result identity differs")
            if result.get("sources", {}).get("execution_claim") != claim_record:
                raise RuntimeError("P12 worker result claim binding differs")
            outputs[job_id] = artifact_record(result_path)
            commands.append(
                {
                    "index": index,
                    "job_id": job_id,
                    "argv": command,
                    "claim": claim_record,
                    "live_recheck": live,
                    "returncode": 0,
                }
            )
        except BaseException as error:
            commands.append(
                {
                    "index": index,
                    "job_id": job_id,
                    "argv": command,
                    "claim": claim_record,
                    "live_recheck": live,
                    "failure": str(error),
                }
            )
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
                failure=str(error),
            )
            raise
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
    if args.action == "_worker":
        expected_lease = (root.parent / ".d1_heavy_resource.lock").resolve()
        result = run_worker(
            root, job_id=args.job_id, resource_lease_path=expected_lease
        )
        print(json.dumps({"status": "COMPLETE", "result": str(result)}))
        return 0
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P12",
        substage=f"d1_four_route_validation_{args.action.replace('-', '_')}",
        route="CROG_G1_C1_D1",
        pool="four_route_top20_no_dedup",
        evidence_track="validation_only",
        method="serial_single_thread_cpu",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        if args.action == "write-policy":
            path, value = write_resource_policy(root, resume=args.resume)
            result: dict[str, Any] = {"status": value["status"], "path": str(path)}
            artifact = path
        elif args.action == "authorize":
            value = authorize(
                root,
                gate_manifest=args.gate_manifest,
                owner=args.owner,
                rank1_run_dir=args.rank1_run_dir,
            )
            artifact = (
                execution_directory(root, str(value["execution_id"])) / "execution.json"
            )
            result = {"status": "ACTIVE", "execution_id": value["execution_id"]}
        elif args.action == "run":
            with exclusive_heavy_resource_lease(
                root, purpose="D1 P12 four-route Validation producers"
            ) as lease:
                value = run_serial(
                    root,
                    python=args.python,
                    rank1_run_dir=args.rank1_run_dir,
                    resource_lease_path=lease,
                )
            artifact = root / P12_EXECUTION_POINTER_RELATIVE
            result = {
                "status": value["status"],
                "completed_jobs": value["completed_jobs"],
            }
        else:  # pragma: no cover
            raise RuntimeError(f"unsupported action: {args.action}")
        state["artifact_path"] = str(artifact.resolve())
        state["artifact_sha256"] = sha256_file(artifact)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
