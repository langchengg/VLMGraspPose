"""Authorize the exact nine raw-feature jobs from a fresh scoped gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import (  # noqa: E402
    CANONICAL_RANK1_RUN_DIR,
    FEATURE_RESOURCE_SCOPE,
    artifact_record,
    exclusive_heavy_resource_lease,
    load_content_manifest,
    validate_feature_resource_gate,
)
from d1_reranking.feature_execution import (  # noqa: E402
    FEATURE_EXECUTION_POINTER_RELATIVE,
    FEATURE_JOB_COUNT,
    authorization_expiry,
    execution_directory,
    execution_source_records,
    exclusive_json,
    write_execution_event,
)
from d1_reranking.feature_plan import (  # noqa: E402
    load_active_feature_extraction_plan,
    load_feature_extraction_plan,
    validate_feature_extraction_result,
)
from d1_reranking.resource_gate import (  # noqa: E402
    collect_resource_snapshot,
    evaluate_resource_snapshot,
    host_contract,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import verified_artifact_path  # noqa: E402
from unified_reranking.hashing import canonical_sha256, sha256_file  # noqa: E402
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


def _run_under_lease(
    run_dir: Path, *, gate_manifest: Path, owner: str, rank1_run_dir: Path
) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    owner_name = owner.strip()
    if not owner_name:
        raise ValueError("D1 feature execution owner must be non-empty")
    rank1_root = rank1_run_dir.expanduser().resolve()
    if rank1_root != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("D1 feature authorization rank1 path differs")
    plan_path, plan = load_active_feature_extraction_plan(root)
    pointer_path = root / FEATURE_EXECUTION_POINTER_RELATIVE
    prior: dict[str, Any] | None = None
    if pointer_path.exists():
        prior = load_content_manifest(
            pointer_path,
            name="prior D1 feature execution",
            statuses=("ACTIVE", "COMPLETE", "FAILED"),
        )
        if prior.get("status") == "ACTIVE":
            raise RuntimeError("D1 feature execution is already ACTIVE")
        if prior.get("status") == "COMPLETE":
            raise RuntimeError("D1 feature extraction is already COMPLETE")

    resume_from: dict[str, str] | None = None
    resume_outputs: dict[str, dict[str, str]] = {}
    prior_gate: dict[str, str] | None = None
    if prior is not None:
        resume_from = {
            str(key): str(value)
            for key, value in dict(prior["latest_event"]).items()  # type: ignore[arg-type]
        }
        failed_path = verified_artifact_path(
            resume_from, name="D1 feature failed event"
        )
        failed = load_content_manifest(
            failed_path, name="D1 feature failed event", statuses=("FAILED",)
        )
        authority_path = verified_artifact_path(
            dict(prior["execution"]),  # type: ignore[arg-type]
            name="D1 feature failed authority",
        )
        authority = load_content_manifest(
            authority_path, name="D1 feature failed authority", statuses=("ACTIVE",)
        )
        execution_id = str(prior["execution_id"])
        if (
            failed.get("execution_id") != execution_id
            or authority.get("execution_id") != execution_id
            or failed_path.parent != execution_directory(root, execution_id) / "events"
        ):
            raise RuntimeError("D1 feature failed authority/event binding differs")
        prior_gate = {
            str(key): str(value)
            for key, value in dict(authority["resource_gate"]).items()  # type: ignore[arg-type]
        }
        outputs = failed.get("outputs")
        if not isinstance(outputs, dict):
            raise RuntimeError("D1 feature failed outputs are invalid")
        planned = {
            str(job["job_id"]): job for job in plan["jobs"] if isinstance(job, dict)
        }
        for job_id, raw_record in outputs.items():
            if job_id not in planned or not isinstance(raw_record, dict):
                raise RuntimeError("D1 feature resumable output is outside the plan")
            result_path = verified_artifact_path(
                raw_record, name=f"D1 feature resumable output {job_id}"
            )
            expected_path = (root / str(planned[job_id]["output_manifest"])).resolve()
            result = load_content_manifest(
                result_path, name=f"D1 feature result {job_id}", statuses=("COMPLETE",)
            )
            validate_feature_extraction_result(
                result,
                plan_path=plan_path,
                job=planned[job_id],
                manifest_path=expected_path,
            )
            if result_path != expected_path:
                raise RuntimeError(f"D1 feature resume path differs for {job_id}")
            resume_outputs[job_id] = artifact_record(result_path)

    gate_path = gate_manifest.expanduser().resolve()
    if prior_gate is not None and artifact_record(gate_path) == prior_gate:
        raise RuntimeError("D1 feature resume requires a newly completed fresh gate")
    gate = validate_feature_resource_gate(root, require_fresh=True)
    pointer = load_content_manifest(
        root / "00_audit/RESOURCE_AUDIT.json",
        name="D1 feature resource pointer",
        statuses=("PASS",),
    )
    if pointer.get("latest_gate") != artifact_record(gate_path):
        raise RuntimeError("D1 feature authorization gate is not the active gate")
    live = collect_resource_snapshot(repo_root=ROOT, rank1_run_dir=rank1_root)
    failures = evaluate_resource_snapshot(live, prefix="feature_authorization_live")
    if failures:
        raise RuntimeError(f"D1 feature authorization live check failed: {failures}")
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
        "scope": FEATURE_RESOURCE_SCOPE,
        "plan": artifact_record(plan_path),
        "resource_gate": artifact_record(gate_path),
        "gate_id": gate.get("gate_id"),
        "job_count": FEATURE_JOB_COUNT,
        "job_ids_sha256": plan["job_ids_sha256"],
        "max_parallel": 1,
        "device": "cpu",
        "direct_cli_execution_permitted": False,
        "fresh_gate_required_on_resume": True,
        "resume_from": resume_from,
        "resume_outputs": resume_outputs,
        "live_recheck": live,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    execution["content_sha256"] = canonical_sha256(execution)
    execution_path = execution_directory(root, execution_id) / "execution.json"
    if execution_path.exists():
        raise FileExistsError(f"D1 feature execution already exists: {execution_path}")
    if load_feature_extraction_plan(plan_path) != plan:
        raise RuntimeError("D1 feature plan changed during authorization")
    final_gate = validate_feature_resource_gate(root, require_fresh=True)
    if final_gate.get("gate_id") != gate.get("gate_id"):
        raise RuntimeError("D1 feature resource gate changed during authorization")
    assert_writable_prelock(root)
    exclusive_json(execution_path, execution)
    assert_writable_prelock(root)
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


def run(
    run_dir: Path, *, gate_manifest: Path, owner: str, rank1_run_dir: Path
) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with exclusive_heavy_resource_lease(
        root, purpose="D1 P5 raw-feature execution authorization"
    ):
        return _run_under_lease(
            root,
            gate_manifest=gate_manifest,
            owner=owner,
            rank1_run_dir=rank1_run_dir,
        )


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P5_RESOURCE",
        substage="d1_feature_extraction_authorization",
        route="D1",
        pool="top5_top10_allnms",
        evidence_track="matched_common_raw",
        method="unified_common_extractor",
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
