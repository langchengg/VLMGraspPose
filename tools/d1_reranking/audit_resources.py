"""Run the exact 3x5-minute resource gate before heavy D1 work."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import (  # noqa: E402
    CANONICAL_RANK1_RUN_DIR,
    artifact_record,
    exclusive_heavy_resource_lease,
    load_candidate_resource_policy,
    load_content_manifest,
    load_feature_resource_policy,
)
from d1_reranking.ablation import (  # noqa: E402
    ABLATION_PLAN_RELATIVE,
    load_ablation_plan,
)
from d1_reranking.ablation_execution import (  # noqa: E402
    ablation_execution_scope,
    load_ablation_resource_policy,
)
from d1_reranking.provenance import load_source_closure  # noqa: E402
from d1_reranking.resource_gate import (  # noqa: E402
    RESOURCE_SNAPSHOT_PERIOD_SECONDS,
    RESOURCE_WINDOW_COUNT,
    RESOURCE_WINDOW_DURATION_SECONDS,
    collect_resource_snapshot,
    evaluate_resource_gate,
    host_contract,
    resource_thresholds,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs" / "reranking_complete_20260803_094159",
    )
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--scope",
        choices=(
            "ablation",
            "candidates",
            "features",
            "four_route_validation",
            "primary",
            "k_sensitivity",
        ),
        required=True,
    )
    return parser.parse_args()


def _run_under_lease(
    run_dir: Path, *, rank1_run_dir: Path, owner: str, scope: str
) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    if not owner.strip():
        raise ValueError("resource owner must be non-empty")
    rank1_root = rank1_run_dir.expanduser().resolve()
    if rank1_root != CANONICAL_RANK1_RUN_DIR:
        raise RuntimeError("D1 resource gate rank1 interlock path differs")
    if scope == "ablation":
        plan_path = root / ABLATION_PLAN_RELATIVE
        plan = load_ablation_plan(plan_path)
        policy_path, policy = load_ablation_resource_policy(
            root, plan_path=plan_path, plan=plan
        )
        prerequisite = None
    elif scope == "four_route_validation":
        from d1_reranking.four_route_execution import (
            load_four_route_execution_plan,
            load_resource_policy,
        )

        plan_path, plan = load_four_route_execution_plan(root)
        policy_path, policy = load_resource_policy(root, plan_path=plan_path, plan=plan)
        prerequisite = None
    elif scope == "primary":
        from d1_reranking.plan import load_active_primary_plan
        from d1_reranking.primary_execution import (
            load_primary_resource_policy,
        )

        plan_path, _plan = load_active_primary_plan(root)
        policy_path, policy = load_primary_resource_policy(root)
        prerequisite: dict[str, str] | None = None
    elif scope == "k_sensitivity":
        from d1_reranking.k_execution import K_EXECUTION_SCOPE

        policy_path = root / "configs/d1_k_sensitivity_resource_gate_policy.json"
        plan_path = root / "configs/d1_k_sensitivity_plan.json"
        load_content_manifest(
            plan_path, name="D1 K-sensitivity plan", statuses=("PLANNED",)
        )
        prerequisite = None
    elif scope == "features":
        from d1_reranking.execution import FEATURE_RESOURCE_SCOPE
        from d1_reranking.feature_plan import load_active_feature_extraction_plan

        policy_path, policy = load_feature_resource_policy(root)
        plan_path, _plan = load_active_feature_extraction_plan(root)
        closure_path, _closure = load_source_closure(root)
        prerequisite = artifact_record(closure_path)
    elif scope == "candidates":
        policy_path, policy = load_candidate_resource_policy(root)
        plan_path = None
        closure_path, _closure = load_source_closure(root)
        prerequisite = artifact_record(closure_path)
    else:  # pragma: no cover - argparse also enforces this
        raise ValueError(f"unsupported D1 resource scope: {scope}")
    if scope not in {"candidates", "features"}:
        policy = load_content_manifest(
            policy_path, name="D1 resource gate policy", statuses=("LOCKED_POLICY",)
        )
    if policy.get("rank1_run_dir") != str(CANONICAL_RANK1_RUN_DIR):
        raise RuntimeError("D1 resource policy rank1 interlock path differs")
    if plan_path is not None and policy.get("plan") != artifact_record(plan_path):
        raise RuntimeError("D1 resource policy does not bind the current plan")
    if scope == "k_sensitivity" and policy.get("scope") != K_EXECUTION_SCOPE:
        raise RuntimeError("D1 K resource policy scope differs")
    if scope == "features" and policy.get("scope") != FEATURE_RESOURCE_SCOPE:
        raise RuntimeError("D1 feature resource policy scope differs")
    if scope == "ablation" and policy.get("scope") != ablation_execution_scope(plan):
        raise RuntimeError("D1 P10 ablation resource policy scope differs")
    if scope == "four_route_validation":
        from d1_reranking.four_route_execution import execution_scope

        if policy.get("scope") != execution_scope(plan):
            raise RuntimeError("D1 P12 Validation resource policy scope differs")
    if prerequisite is not None and policy.get("prerequisite") != prerequisite:
        raise RuntimeError("D1 resource policy does not bind the source closure")
    started_at = datetime.now(timezone.utc)
    gate_started = time.monotonic()
    windows = []
    for index in range(RESOURCE_WINDOW_COUNT):
        window_started = time.monotonic()
        observations = []
        while True:
            snapshot = collect_resource_snapshot(
                repo_root=ROOT,
                rank1_run_dir=rank1_root,
            )
            snapshot["monotonic_offset_seconds"] = time.monotonic() - gate_started
            observations.append(snapshot)
            elapsed = time.monotonic() - window_started
            if elapsed >= RESOURCE_WINDOW_DURATION_SECONDS:
                break
            time.sleep(
                min(
                    RESOURCE_SNAPSHOT_PERIOD_SECONDS,
                    RESOURCE_WINDOW_DURATION_SECONDS - elapsed,
                )
            )
        windows.append(
            {
                "index": index,
                "monotonic_start_seconds": window_started - gate_started,
                "monotonic_end_seconds": time.monotonic() - gate_started,
                "observations": observations,
            }
        )
    passed, reasons = evaluate_resource_gate(windows)
    finished_at = datetime.now(timezone.utc)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "owner": owner.strip(),
        "run_dir": str(root),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "monotonic_elapsed_seconds": time.monotonic() - gate_started,
        "exclusive_resource_ownership": passed,
        "host": host_contract(),
        "thresholds": resource_thresholds(),
        "policy": artifact_record(policy_path),
        "scope": policy["scope"],
        "rank1_run_dir": str(rank1_root),
        "windows": windows,
        "failure_reasons": reasons,
        "candidate_test_labels_read": False,
        "sources": {
            "tool": artifact_record(Path(__file__)),
            "resource_gate": artifact_record(
                ROOT / "src/d1_reranking/resource_gate.py"
            ),
        },
    }
    if plan_path is not None:
        result["plan"] = artifact_record(plan_path)
    if prerequisite is not None:
        result["prerequisite"] = prerequisite
    result["content_sha256"] = canonical_sha256(result)
    gate_id = canonical_sha256(result)[:20]
    result["gate_id"] = gate_id
    result["content_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )
    gate_path = root / "00_audit" / "resource_gates" / f"{gate_id}.json"
    if gate_path.exists():
        raise FileExistsError(f"D1 resource gate evidence already exists: {gate_path}")
    atomic_json(gate_path, result)
    pointer: dict[str, object] = {
        "schema_version": 1,
        "status": result["status"],
        "latest_gate": artifact_record(gate_path),
        "scope": result["scope"],
        "policy": artifact_record(policy_path),
        "candidate_test_labels_read": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(root / "00_audit" / "RESOURCE_AUDIT.json", pointer)
    result["artifact_path"] = str(gate_path)
    return result


def run(
    run_dir: Path, *, rank1_run_dir: Path, owner: str, scope: str
) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with exclusive_heavy_resource_lease(
        root, purpose=f"D1 {scope} three-window resource audit"
    ):
        return _run_under_lease(
            root,
            rank1_run_dir=rank1_run_dir,
            owner=owner,
            scope=scope,
        )


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with exclusive_heavy_resource_lease(
        root, purpose=f"D1 {args.scope} three-window resource audit"
    ):
        assert_writable_prelock(root)
        with ledger_stage(
            root / "run_ledger.sqlite",
            stage=(
                "P2_RESOURCE"
                if args.scope == "candidates"
                else "P5_RESOURCE"
                if args.scope == "features"
                else "P10_RESOURCE"
                if args.scope in {"ablation", "k_sensitivity"}
                else "P12_RESOURCE"
                if args.scope == "four_route_validation"
                else "P9_RESOURCE"
            ),
            substage=f"d1_{args.scope}_resource_gate_3x5min",
            route="D1",
            command=" ".join(map(str, sys.argv)),
        ) as state:
            result = _run_under_lease(
                root,
                rank1_run_dir=args.rank1_run_dir,
                owner=args.owner,
                scope=args.scope,
            )
            state["artifact_path"] = str(result["artifact_path"])
            state["artifact_sha256"] = sha256_file(result["artifact_path"])
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
