"""Freeze the non-executable D1 Top10/AllNMS K-sensitivity job universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.k_sensitivity import write_k_sensitivity_plan  # noqa: E402
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
    destination = root / "configs" / "d1_k_sensitivity_plan.json"
    return write_k_sensitivity_plan(
        destination,
        run_dir=root,
        tool_paths=(
            ROOT / "src/d1_reranking/k_sensitivity.py",
            ROOT / "src/d1_reranking/k_execution.py",
            ROOT / "src/d1_reranking/k_replay.py",
            ROOT / "src/d1_reranking/k_formal_replay.py",
            ROOT / "src/d1_reranking/models.py",
            ROOT / "src/d1_reranking/fold_calibration.py",
            ROOT / "src/d1_reranking/selection.py",
            ROOT / "src/unified_reranking/datasets.py",
            ROOT / "src/unified_reranking/training.py",
            ROOT / "src/unified_reranking/losses.py",
            ROOT / "src/unified_reranking/models/lightgbm_ranker.py",
            ROOT / "tools/d1_reranking/run_k_sensitivity_cell.py",
            ROOT / "tools/d1_reranking/authorize_k_sensitivity_execution.py",
            ROOT / "tools/d1_reranking/run_k_sensitivity_matrix.py",
            ROOT / "tools/d1_reranking/select_k_sensitivity.py",
            ROOT / "tools/d1_reranking/build_k_scenario_gates.py",
            ROOT / "tools/d1_reranking/build_d1_formal_inputs.py",
            ROOT / "tools/unified_reranking/apply_locked_matrix_cell.py",
            ROOT / "tools/d1_reranking/write_k_sensitivity_resource_policy.py",
            ROOT / "tools/d1_reranking/audit_resources.py",
            Path(__file__),
        ),
        resume=resume,
    )


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    destination = root / "configs" / "d1_k_sensitivity_plan.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="d1_k_sensitivity_plan",
        route="D1",
        pool="top10_allnms",
        evidence_track="T2_T3",
        method="selected_primary_fixed",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        plan = run(root, resume=args.resume)
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
        print(
            json.dumps(
                {
                    "status": plan["status"],
                    "job_count": plan["job_count"],
                    "plan": str(destination),
                    "sha256": state["artifact_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
