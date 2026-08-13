"""Copy and validate the grouped five-fold D1 development split contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.splits import build_split_audit  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", default="splits")
    parser.add_argument("--split", default="train")
    parser.add_argument("--pool", default="none")
    parser.add_argument("--evidence-track", default="none")
    parser.add_argument("--method", default="grouped_rgbd")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--unified-run",
        type=Path,
        default=ROOT / "runs/fair_unified_reranking_20260809_103012",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = args.run_dir.expanduser().resolve()
    assert_writable_prelock(run)
    output = run / "04_splits" / "split_leakage_audit.json"
    if output.exists() and not args.resume:
        raise FileExistsError(f"D1 split audit exists; pass --resume: {output}")
    with ledger_stage(
        run / "run_ledger.sqlite",
        stage="P7",
        substage="grouped_oof_and_split_audit",
        route="D1",
        method="grouped_rgbd",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        build_split_audit(run_dir=run, unified_run=args.unified_run)
        state["artifact_path"] = str(output)
        state["artifact_sha256"] = sha256_file(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
