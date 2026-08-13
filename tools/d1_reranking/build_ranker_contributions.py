"""Replay the locked D1 R5 ensemble and persist label-free feature contributions."""

from __future__ import annotations

# Keep the native runtime import order aligned with locked Test inference.
try:
    import lightgbm as _lightgbm  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _lightgbm = None

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.ranker_contributions import (  # noqa: E402
    MANIFEST_RELATIVE_PATH,
    build_ranker_contributions,
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
    destination = root / MANIFEST_RELATIVE_PATH
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_r5_candidate_contributions",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R5_three_seed_native_pred_contrib",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        build_ranker_contributions(root, resume=args.resume)
        state["artifact_path"] = str(destination.resolve())
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
