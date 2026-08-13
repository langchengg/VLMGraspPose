"""Publish P10 tables through the audited primary-gate schema adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.ablation_schema_adapter import (  # noqa: E402
    ADAPTER_RELATIVE,
    run_adapted_ablation_selection,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    output = root / ADAPTER_RELATIVE
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="d1_ablation_selection_schema_adapter_v1",
        route="D1",
        pool="top5",
        evidence_track="T1_T2_T3_T4",
        method="fixed_selected_primary_no_retune",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        selection = run_adapted_ablation_selection(root, resume=args.resume)
        state["artifact_path"] = str(output)
        state["artifact_sha256"] = sha256_file(output)
    print(
        json.dumps(
            {
                "status": selection["status"],
                "schema_adapter": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
