"""Write the immutable D1 Top5/T2 R0-R7 plan before any model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.plan import (  # noqa: E402
    PRIMARY_PLAN_POINTER_RELATIVE,
    publish_primary_plan,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _tool_paths() -> tuple[Path, ...]:
    return (
        ROOT / "src/d1_reranking/plan.py",
        ROOT / "src/d1_reranking/models.py",
        ROOT / "src/d1_reranking/fold_calibration.py",
        ROOT / "src/d1_reranking/rules.py",
        ROOT / "src/d1_reranking/resource_gate.py",
        ROOT / "src/d1_reranking/execution.py",
        ROOT / "src/d1_reranking/primary_execution.py",
        ROOT / "src/d1_reranking/selection.py",
        ROOT / "src/d1_reranking/gate_inputs.py",
        ROOT / "src/unified_reranking/datasets.py",
        ROOT / "src/unified_reranking/training.py",
        ROOT / "src/unified_reranking/losses.py",
        ROOT / "src/unified_reranking/metrics.py",
        ROOT / "src/unified_reranking/telemetry.py",
        ROOT / "src/unified_reranking/models/lightgbm_ranker.py",
        ROOT / "tools/d1_reranking/train_primary_cell.py",
        ROOT / "tools/d1_reranking/run_r0_r1.py",
        ROOT / "tools/d1_reranking/audit_resources.py",
        ROOT / "tools/d1_reranking/write_resource_policy.py",
        ROOT / "tools/d1_reranking/authorize_primary_execution.py",
        ROOT / "tools/d1_reranking/run_primary_matrix.py",
        ROOT / "tools/d1_reranking/select_primary_ranker.py",
        ROOT / "tools/d1_reranking/prepare_gate_inputs.py",
        ROOT / "tools/d1_reranking/plan_gate_grid.py",
        ROOT / "tools/d1_reranking/select_gate.py",
        Path(__file__),
    )


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    pointer_path = root / PRIMARY_PLAN_POINTER_RELATIVE
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P9",
        substage="d1_primary_matrix_plan",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R0-R7",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        destination, _plan = publish_primary_plan(
            root,
            python_path=args.python,
            tool_paths=_tool_paths(),
            resume=args.resume,
        )
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
        print(
            f"D1 primary plan: {destination} "
            f"(active pointer {pointer_path})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
