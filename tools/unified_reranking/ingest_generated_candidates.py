"""Freeze a completed fair G1/C1 development inference into canonical pools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.candidates import ingest_generated_candidate_pool
from unified_reranking.hashing import sha256_file
from unified_reranking.ledger import ledger_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("g1", "c1"))
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    parser.add_argument(
        "--fair-run",
        type=Path,
        default=ROOT / "runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P1",
        substage=f"ingest_generated_{args.route}_{args.split}",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        ingest_generated_candidate_pool(
            run_dir, args.fair_run.resolve(), route=args.route, split=args.split
        )
        artifact = run_dir / "02_candidates" / "candidate_contract_hashes.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
