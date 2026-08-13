"""Freeze the resource-gate policy for the exact 54-job K-sensitivity plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.k_execution import k_sensitivity_resource_policy  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import atomic_json, sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def run(run_dir: Path) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    plan_path = root / "configs/d1_k_sensitivity_plan.json"
    value = k_sensitivity_resource_policy(
        plan_path=plan_path,
        source_paths=(
            ROOT / "src/d1_reranking/k_execution.py",
            ROOT / "src/d1_reranking/resource_gate.py",
            ROOT / "tools/d1_reranking/audit_resources.py",
            Path(__file__),
        ),
    )
    atomic_json(root / "configs/d1_k_sensitivity_resource_gate_policy.json", value)
    return value


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    destination = root / "configs/d1_k_sensitivity_resource_gate_policy.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="d1_k_sensitivity_resource_policy",
        route="D1",
        pool="top10_allnms",
        evidence_track="T2_T3",
        method="resource_policy",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root)
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
