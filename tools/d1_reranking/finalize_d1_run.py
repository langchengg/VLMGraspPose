"""Fail-closed D1 final-run lock and COMPLETE marker assembly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.postformal import (  # noqa: E402
    POSTFORMAL_MANIFEST_RELATIVE_PATH,
    assert_writable_finalization,
    finalize_d1_run,
)
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_finalization(root)
    postformal = root / POSTFORMAL_MANIFEST_RELATIVE_PATH
    # This context must close and export commands.log before the immutable final
    # inventory is assembled.  No ledger or command mutation follows the lock.
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="PFINAL",
        substage="d1_finalization_preflight",
        route="D1",
        pool="all_locked_outputs",
        evidence_track="P14_COMPLETE_P17_PASS_P15_COMPLETE",
        method="fresh_exact_inventory_finalizer",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        if not postformal.is_file():
            raise FileNotFoundError(f"D1 postformal manifest is missing: {postformal}")
        state["artifact_path"] = str(postformal)
        state["artifact_sha256"] = sha256_file(postformal)
    finalize_d1_run(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
