"""Authorize one exact K-sensitivity matrix from a fresh PASS resource gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

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
from d1_reranking.k_execution import (  # noqa: E402
    K_EXECUTION_POINTER_RELATIVE,
    K_EXECUTION_SCOPE,
    authorization_expiry,
    execution_directory,
    execution_source_records,
    exclusive_json,
    write_execution_event,
)
from d1_reranking.k_sensitivity import (  # noqa: E402
    load_k_sensitivity_plan,
    validate_k_sensitivity_result,
)
from d1_reranking.resource_gate import (  # noqa: E402
    collect_resource_snapshot,
    evaluate_resource_snapshot,
    host_contract,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import verified_artifact_path  # noqa: E402
from unified_reranking.hashing import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gate-manifest", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )
    return parser.parse_args()


def run(
    run_dir: Path, *, gate_manifest: Path, owner: str, rank1_run_dir: Path
) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    owner_name = owner.strip()
    if not owner_name:
        raise ValueError("D1 K execution owner must be non-empty")
    rank1_root = rank1_run_dir.expanduser().resolve()
    if rank1_root != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("D1 K authorization rank1 interlock path differs")
    pointer_path = root / K_EXECUTION_POINTER_RELATIVE
    prior: dict[str, object] | None = None
    if pointer_path.exists():
        prior = load_content_manifest(
            pointer_path,
            name="prior D1 K execution pointer",
            statuses=("ACTIVE", "COMPLETE", "FAILED"),
        )
        if prior.get("status") == "ACTIVE":
            raise RuntimeError("D1 K execution is already ACTIVE")
        if prior.get("status") == "COMPLETE":
            raise RuntimeError("D1 K matrix is already COMPLETE")
    plan_path = root / "configs/d1_k_sensitivity_plan.json"
    plan = load_k_sensitivity_plan(plan_path)
    resume_from: dict[str, str] | None = None
    resume_outputs: dict[str, dict[str, str]] = {}
    prior_gate_record: dict[str, str] | None = None
    if prior is not None and prior.get("status") == "FAILED":
        resume_from = {
            str(key): str(value)
            for key, value in dict(prior["latest_event"]).items()  # type: ignore[arg-type]
        }
        failed_event_path = verified_artifact_path(
            resume_from, name="D1 K failed execution event"
        )
        failed_event = load_content_manifest(
            failed_event_path,
            name="D1 K failed execution event",
            statuses=("FAILED",),
        )
        prior_execution_path = verified_artifact_path(
            dict(prior["execution"]),  # type: ignore[arg-type]
            name="D1 K failed execution authority",
        )
        prior_execution = load_content_manifest(
            prior_execution_path,
            name="D1 K failed execution authority",
            statuses=("ACTIVE",),
        )
        if (
            failed_event.get("execution_id") != prior.get("execution_id")
            or prior_execution.get("execution_id") != prior.get("execution_id")
            or failed_event_path.parent
            != execution_directory(root, str(prior["execution_id"])) / "events"
        ):
            raise RuntimeError("D1 K failed execution authority/event binding differs")
        prior_gate_record = {
            str(key): str(value)
            for key, value in dict(prior_execution["resource_gate"]).items()  # type: ignore[arg-type]
        }
        failed_outputs = failed_event.get("outputs")
        if not isinstance(failed_outputs, dict):
            raise RuntimeError("D1 K failed execution outputs are invalid")
        planned = {
            str(job["job_id"]): job for job in plan["jobs"] if isinstance(job, dict)
        }
        for job_id, raw_record in failed_outputs.items():
            if job_id not in planned or not isinstance(raw_record, dict):
                raise RuntimeError("D1 K failed execution output is outside plan")
            result_path = verified_artifact_path(
                raw_record, name=f"D1 K resumable output {job_id}"
            )
            expected_path = (root / str(planned[job_id]["output_manifest"])).resolve()
            if result_path != expected_path:
                raise RuntimeError(f"D1 K resumable result path differs for {job_id}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_k_sensitivity_result(
                result,
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
        raise RuntimeError("D1 K resume requires a newly completed fresh resource gate")
    gate = validate_resource_gate(
        gate_path,
        run_dir=root,
        plan_path=plan_path,
        expected_scope=K_EXECUTION_SCOPE,
        require_fresh=True,
    )
    live = collect_resource_snapshot(repo_root=ROOT, rank1_run_dir=rank1_root)
    failures = evaluate_resource_snapshot(live, prefix="k_authorization_live")
    if failures:
        raise RuntimeError(f"D1 K live resource recheck failed: {failures}")
    now = datetime.now(timezone.utc)
    execution_id = canonical_sha256(
        {
            "run_dir": str(root),
            "owner": owner_name,
            "plan": artifact_record(plan_path),
            "resource_gate": artifact_record(gate_path),
            "authorized_at_utc": now.isoformat(),
            "resume_from": resume_from,
        }
    )[:24]
    sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=gate_path,
        resume_from=resume_from,
    )
    execution: dict[str, object] = {
        "schema_version": 1,
        "status": "ACTIVE",
        "execution_id": execution_id,
        "authorized_at_utc": now.isoformat(),
        "expires_at_utc": authorization_expiry(now),
        "owner": owner_name,
        "run_dir": str(root),
        "rank1_run_dir": str(rank1_root),
        "host": host_contract(),
        "scope": K_EXECUTION_SCOPE,
        "plan": artifact_record(plan_path),
        "resource_gate": artifact_record(gate_path),
        "gate_id": gate.get("gate_id"),
        "job_count": 54,
        "job_ids_sha256": plan["job_ids_sha256"],
        "max_parallel": 1,
        "device": "cpu",
        "direct_cli_execution_permitted": False,
        "fresh_gate_required_on_resume": True,
        "resume_from": resume_from,
        "resume_outputs": resume_outputs,
        "live_recheck": live,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    execution["content_sha256"] = canonical_sha256(execution)
    execution_path = execution_directory(root, execution_id) / "execution.json"
    if execution_path.exists():
        raise FileExistsError(f"D1 K execution already exists: {execution_path}")
    exclusive_json(execution_path, execution)
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
        substage="d1_k_sensitivity_authorization",
        route="D1",
        pool="top10_allnms",
        evidence_track="T2_T3",
        method="selected_primary_fixed",
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
