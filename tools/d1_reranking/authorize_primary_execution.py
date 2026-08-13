"""Authorize one CPU-only D1 primary-matrix run from a fresh PASS gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import (  # noqa: E402
    CANONICAL_RANK1_RUN_DIR,
    artifact_record,
    load_content_manifest,
    validate_resource_gate,
)
from d1_reranking.plan import (  # noqa: E402
    PRIMARY_PLAN_POINTER_RELATIVE,
    load_active_primary_plan,
    load_primary_plan,
)
from d1_reranking.primary_execution import (  # noqa: E402
    PRIMARY_EXECUTION_POINTER_RELATIVE,
    PRIMARY_EXECUTION_SCOPE,
    create_execution_authority,
    execution_directory,
    load_primary_resource_policy,
    validate_primary_result,
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
    parser.add_argument("--gate-manifest", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs" / "reranking_complete_20260803_094159",
    )
    return parser.parse_args()


def run(
    run_dir: Path, *, gate_manifest: Path, owner: str, rank1_run_dir: Path
) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    if not owner.strip():
        raise ValueError("D1 execution owner must be non-empty")
    rank1_root = rank1_run_dir.expanduser().resolve()
    if rank1_root != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("D1 authorization rank1 interlock path differs")
    active_plan = (root / PRIMARY_PLAN_POINTER_RELATIVE).exists()
    if active_plan:
        plan_path, plan = load_active_primary_plan(root)
    else:
        plan_path = root / "configs/d1_primary_matrix_plan.json"
        plan = load_primary_plan(plan_path)
    policy_path = None
    if active_plan:
        policy_path, _policy = load_primary_resource_policy(root)
    pointer_path = root / PRIMARY_EXECUTION_POINTER_RELATIVE
    prior: dict[str, object] | None = None
    if pointer_path.exists():
        prior = load_content_manifest(
            pointer_path,
            name="prior D1 primary execution pointer",
            statuses=("ACTIVE", "COMPLETE", "FAILED"),
        )
        if prior.get("status") == "ACTIVE":
            raise RuntimeError("D1 primary execution is already ACTIVE")
        if prior.get("status") == "COMPLETE":
            raise RuntimeError("D1 primary matrix is already COMPLETE")
    resume_from: dict[str, str] | None = None
    resume_outputs: dict[str, dict[str, str]] = {}
    prior_gate_record: dict[str, str] | None = None
    if prior is not None and prior.get("status") == "FAILED":
        resume_from = {
            str(key): str(value)
            for key, value in dict(prior["latest_event"]).items()  # type: ignore[arg-type]
        }
        failed_event_path = verified_artifact_path(
            resume_from, name="D1 primary failed execution event"
        )
        failed_event = load_content_manifest(
            failed_event_path,
            name="D1 primary failed execution event",
            statuses=("FAILED",),
        )
        prior_execution_path = verified_artifact_path(
            dict(prior["execution"]),  # type: ignore[arg-type]
            name="D1 primary failed execution authority",
        )
        prior_execution = load_content_manifest(
            prior_execution_path,
            name="D1 primary failed execution authority",
            statuses=("ACTIVE",),
        )
        if (
            failed_event.get("execution_id") != prior.get("execution_id")
            or prior_execution.get("execution_id") != prior.get("execution_id")
            or failed_event_path.parent
            != execution_directory(root, str(prior["execution_id"])) / "events"
        ):
            raise RuntimeError("D1 primary failed authority/event binding differs")
        prior_gate_record = {
            str(key): str(value)
            for key, value in dict(prior_execution["resource_gate"]).items()  # type: ignore[arg-type]
        }
        failed_outputs = failed_event.get("outputs")
        if not isinstance(failed_outputs, dict):
            raise RuntimeError("D1 primary failed outputs are invalid")
        planned = {
            str(job["job_id"]): job for job in plan["jobs"] if isinstance(job, dict)
        }
        for job_id, raw_record in failed_outputs.items():
            if job_id not in planned or not isinstance(raw_record, dict):
                raise RuntimeError("D1 primary resumable output is outside the plan")
            result_path = verified_artifact_path(
                raw_record, name=f"D1 primary resumable output {job_id}"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_primary_result(
                result,
                root=root,
                plan_path=plan_path,
                job=planned[job_id],
                manifest_path=result_path,
            )
            resume_outputs[job_id] = artifact_record(result_path)
    gate_path = gate_manifest.expanduser().resolve()
    if (
        prior_gate_record is not None
        and artifact_record(gate_path) == prior_gate_record
    ):
        raise RuntimeError(
            "D1 primary resume requires a newly completed fresh resource gate"
        )
    gate = validate_resource_gate(
        gate_path,
        run_dir=root,
        plan_path=plan_path,
        policy_path=policy_path,
        expected_scope=PRIMARY_EXECUTION_SCOPE,
        require_fresh=True,
    )
    live = collect_resource_snapshot(
        repo_root=ROOT,
        rank1_run_dir=rank1_root,
    )
    live_failures = evaluate_resource_snapshot(live, prefix="authorization_live")
    if live_failures:
        raise RuntimeError(f"D1 live resource recheck failed: {live_failures}")
    _path, execution = create_execution_authority(
        root,
        plan_path=plan_path,
        plan=plan,
        gate_path=gate_path,
        owner=owner,
        gate_id=gate.get("gate_id"),
        live_recheck=live,
        resume_from=resume_from,
        resume_outputs=resume_outputs,
    )
    return execution


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P9",
        substage="d1_primary_matrix_authorization",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R2-R6",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(
            root,
            gate_manifest=args.gate_manifest,
            owner=args.owner,
            rank1_run_dir=args.rank1_run_dir,
        )
        path = execution_directory(root, str(result["execution_id"])) / "execution.json"
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
