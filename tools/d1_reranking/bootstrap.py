"""Create the lightweight auditable D1 extension run skeleton."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.run import bootstrap_run  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", default="audit")
    parser.add_argument("--split", default="none")
    parser.add_argument("--pool", default="none")
    parser.add_argument("--evidence-track", default="none")
    parser.add_argument("--method", default="none")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--unified-run",
        type=Path,
        default=ROOT / "runs" / "fair_unified_reranking_20260809_103012",
    )
    parser.add_argument(
        "--snapshot-a",
        type=Path,
        default=ROOT
        / "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ROOT / "runs" / f"fair_d1_reranking_extension_{stamp}"
    run_dir = run_dir.expanduser().resolve()
    evaluator = args.unified_run / "configs" / "canonical_evaluator.py"
    bootstrap_run(
        repo_root=ROOT,
        run_dir=run_dir,
        unified_run=args.unified_run,
        snapshot_a=args.snapshot_a,
        evaluator_path=evaluator,
        resume=args.resume,
    )
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P0",
        substage="bootstrap_and_source_audit",
        route="D1",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        artifact = run_dir / "00_audit" / "SOURCE_RUN_IMMUTABILITY_BEFORE.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
