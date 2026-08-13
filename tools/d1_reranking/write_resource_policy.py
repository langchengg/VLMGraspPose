"""Freeze the CPU-only D1 primary-matrix resource policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, resource_policy  # noqa: E402
from d1_reranking.plan import load_active_primary_plan  # noqa: E402
from d1_reranking.primary_execution import (  # noqa: E402
    PRIMARY_RESOURCE_POLICY_POINTER_RELATIVE,
    PRIMARY_RESOURCE_POLICY_REGISTRY_RELATIVE,
    exclusive_json,
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
    return parser.parse_args()


def run(run_dir: Path) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    plan_path, _plan = load_active_primary_plan(root)
    value = resource_policy(
        plan_path=plan_path,
        source_paths=(
            ROOT / "src/d1_reranking/resource_gate.py",
            ROOT / "src/d1_reranking/execution.py",
            ROOT / "src/d1_reranking/primary_execution.py",
            ROOT / "src/d1_reranking/plan.py",
            ROOT / "tools/d1_reranking/audit_resources.py",
            ROOT / "tools/d1_reranking/authorize_primary_execution.py",
            ROOT / "tools/d1_reranking/run_primary_matrix.py",
            Path(__file__),
        ),
    )
    policy_id = str(value["content_sha256"])[:20]
    destination = root / PRIMARY_RESOURCE_POLICY_REGISTRY_RELATIVE / f"{policy_id}.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError("immutable D1 primary resource policy differs")
    else:
        exclusive_json(destination, value)
    pointer: dict[str, object] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY_POINTER",
        "active_policy": artifact_record(destination),
        "plan": artifact_record(plan_path),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    assert_writable_prelock(root)
    atomic_json(root / PRIMARY_RESOURCE_POLICY_POINTER_RELATIVE, pointer)
    return value


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    plan_path, _plan = load_active_primary_plan(root)
    expected = resource_policy(
        plan_path=plan_path,
        source_paths=(
            ROOT / "src/d1_reranking/resource_gate.py",
            ROOT / "src/d1_reranking/execution.py",
            ROOT / "src/d1_reranking/primary_execution.py",
            ROOT / "src/d1_reranking/plan.py",
            ROOT / "tools/d1_reranking/audit_resources.py",
            ROOT / "tools/d1_reranking/authorize_primary_execution.py",
            ROOT / "tools/d1_reranking/run_primary_matrix.py",
            Path(__file__),
        ),
    )
    policy_id = str(expected["content_sha256"])[:20]
    destination = root / PRIMARY_RESOURCE_POLICY_REGISTRY_RELATIVE / f"{policy_id}.json"
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P9_RESOURCE",
        substage=f"d1_primary_resource_policy_{policy_id}",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R2_R6_primary_matrix",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root)
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
