"""Replay COMPLETE P10 cells and publish frozen Validation-only ablation tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.ablation_replay import (  # noqa: E402
    validate_ablation_execution_results,
    validate_ablation_selection,
)
from d1_reranking.ablation_selection import (  # noqa: E402
    SELECTION_RELATIVE,
    write_ablation_selection,
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
    execution_records, values = validate_ablation_execution_results(root)
    selection = write_ablation_selection(
        root,
        execution_records=execution_records,
        result_values=values["results"],
        resume=resume,
    )
    validate_ablation_selection(
        root,
        execution_records=execution_records,
        execution_values=values,
    )
    return selection


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="d1_evidence_feature_ablation_selection",
        route="D1",
        pool="top5",
        evidence_track="T1_T2_T3_T4",
        method="fixed_selected_primary_no_retune",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        selection = run(root, resume=args.resume)
        path = root / SELECTION_RELATIVE
        state["artifact_path"] = str(path.resolve())
        state["artifact_sha256"] = sha256_file(path)
    print(json.dumps({"status": selection["status"], "selection": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
