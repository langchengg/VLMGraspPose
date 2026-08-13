"""Authorize one serial P10 ablation execution from a fresh PASS resource gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    ablation_execution_scope,
    create_execution_authority,
    load_ablation_resource_policy,
    write_execution_event,
)
from d1_reranking.execution import (  # noqa: E402
    CANONICAL_RANK1_RUN_DIR,
    artifact_record,
    load_content_manifest,
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
    parser.add_argument("--gate-manifest", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--rank1-run-dir", type=Path, default=CANONICAL_RANK1_RUN_DIR)
    return parser.parse_args()


def run(
    run_dir: Path, *, gate_manifest: Path, owner: str, rank1_run_dir: Path
) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    rank1 = rank1_run_dir.expanduser().resolve()
    if rank1 != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("D1 P10 authorization rank1 interlock differs")
    plan_path = root / ABLATION_PLAN_RELATIVE
    plan = load_ablation_plan(plan_path)
    pointer_path = root / ABLATION_EXECUTION_POINTER_RELATIVE
    resume_from = None
    resume_outputs: dict[str, object] = {}
    if pointer_path.exists():
        prior = load_content_manifest(
            pointer_path,
            name="prior D1 P10 execution",
            statuses=("ACTIVE", "COMPLETE", "FAILED"),
        )
        if prior["status"] != "FAILED":
            raise RuntimeError(f"D1 P10 execution is already {prior['status']}")
        failed_path = verified_artifact_path(
            prior["latest_event"], name="D1 P10 failed event"
        )
        failed = load_content_manifest(
            failed_path, name="D1 P10 failed event", statuses=("FAILED",)
        )
        resume_from = artifact_record(failed_path)
        resume_outputs = dict(failed.get("outputs", {}))
    gate_path = gate_manifest.expanduser().resolve()
    scope = ablation_execution_scope(plan)
    policy_path, _policy = load_ablation_resource_policy(
        root, plan_path=plan_path, plan=plan
    )
    validate_resource_gate(
        gate_path,
        run_dir=root,
        plan_path=plan_path,
        policy_path=policy_path,
        expected_scope=scope,
        require_fresh=True,
    )
    live = collect_resource_snapshot(repo_root=ROOT, rank1_run_dir=rank1)
    failures = evaluate_resource_snapshot(live, prefix="ablation_authorization_live")
    if failures:
        raise RuntimeError(f"D1 P10 live resource recheck failed: {failures}")
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


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="d1_ablation_authorization",
        route="D1",
        pool="top5",
        method="serial_fresh_gate_authority",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        execution = run(
            root,
            gate_manifest=args.gate_manifest,
            owner=args.owner,
            rank1_run_dir=args.rank1_run_dir,
        )
        path = (
            root
            / "configs/ablation_executions"
            / str(execution["execution_id"])
            / "execution.json"
        )
        state["artifact_path"] = str(path.resolve())
        state["artifact_sha256"] = sha256_file(path)
    print(json.dumps({"status": "ACTIVE", "execution_id": execution["execution_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
