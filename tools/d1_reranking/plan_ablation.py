"""Freeze the versioned P10 evidence/feature ablation universe."""

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
    write_ablation_plan,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(run_dir: Path, *, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    destination = root / ABLATION_PLAN_RELATIVE
    tools = (
        ROOT / "src/d1_reranking/ablation.py",
        ROOT / "src/d1_reranking/ablation_execution.py",
        ROOT / "src/d1_reranking/ablation_selection.py",
        ROOT / "src/d1_reranking/ablation_replay.py",
        ROOT / "src/d1_reranking/gate_inputs.py",
        ROOT / "src/d1_reranking/gate_validation.py",
        ROOT / "src/d1_reranking/fold_calibration.py",
        ROOT / "src/d1_reranking/models.py",
        ROOT / "src/unified_reranking/gate.py",
        ROOT / "src/unified_reranking/datasets.py",
        ROOT / "src/unified_reranking/training.py",
        ROOT / "src/unified_reranking/models/lightgbm_ranker.py",
        ROOT / "tools/d1_reranking/run_ablation_cell.py",
        ROOT / "tools/d1_reranking/authorize_ablation_execution.py",
        ROOT / "tools/d1_reranking/run_ablation_matrix.py",
        ROOT / "tools/d1_reranking/write_ablation_resource_policy.py",
        ROOT / "tools/d1_reranking/select_ablation.py",
        Path(__file__),
    )
    return write_ablation_plan(
        destination, run_dir=root, tool_paths=tools, resume=resume
    )


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    destination = root / ABLATION_PLAN_RELATIVE
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="d1_ablation_plan_v1",
        route="D1",
        pool="top5",
        evidence_track="T1_T2_T3_T4_and_feature_families",
        method="selected_primary_fixed_no_retune",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        plan = run(root, resume=args.resume)
        state["artifact_path"] = str(destination.resolve())
        state["artifact_sha256"] = sha256_file(destination)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "job_count": plan["job_count"],
                "plan": str(destination.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
