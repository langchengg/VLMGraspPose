"""Freeze the versioned resource policy for the exact current P10 plan/code."""

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
    ABLATION_PLAN_RELATIVE,
    load_ablation_plan,
)
from d1_reranking.ablation_execution import (  # noqa: E402
    ABLATION_RESOURCE_POLICY_POINTER_RELATIVE,
    ABLATION_RESOURCE_POLICY_REGISTRY_RELATIVE,
    ablation_execution_scope,
    ablation_resource_policy,
)
from d1_reranking.execution import (  # noqa: E402
    artifact_record,
    load_content_manifest,
)
from d1_reranking.k_execution import exclusive_json  # noqa: E402
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
    return parser.parse_args()


def run(run_dir: Path) -> tuple[Path, dict[str, object]]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    plan_path = root / ABLATION_PLAN_RELATIVE
    plan = load_ablation_plan(plan_path)
    policy = ablation_resource_policy(plan_path=plan_path, plan=plan)
    policy_id = canonical_sha256(policy)[:24]
    path = root / ABLATION_RESOURCE_POLICY_REGISTRY_RELATIVE / f"{policy_id}.json"
    if path.exists():
        existing = load_content_manifest(
            path, name="D1 P10 resource policy", statuses=("LOCKED_POLICY",)
        )
        if existing != policy:
            raise RuntimeError("D1 P10 resource policy registry entry differs")
    else:
        exclusive_json(path, policy)
    pointer: dict[str, object] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY_POINTER",
        "active_policy": artifact_record(path),
        "plan": artifact_record(plan_path),
        "scope": ablation_execution_scope(plan),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(root / ABLATION_RESOURCE_POLICY_POINTER_RELATIVE, pointer)
    return path, policy


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10_RESOURCE",
        substage="d1_ablation_resource_policy_v1",
        route="D1",
        pool="top5",
        method="exact_plan_code_source_policy",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        path, policy = run(root)
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    print(json.dumps({"status": policy["status"], "policy": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
