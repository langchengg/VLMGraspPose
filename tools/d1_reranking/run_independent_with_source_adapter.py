"""Run locked D1 P17 through the exact audited router-rank source adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from d1_reranking.independent_source_adapter import (
    run_adapted_independent_recompute,
)
from tools.d1_reranking import independent_recompute as locked_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    command = " ".join(map(str, sys.argv))
    try:
        run_adapted_independent_recompute(root)
    except Exception as error:
        locked_module._record_ledger(
            root, command=command, status="FAILED", error=repr(error)
        )
        raise
    locked_module._record_ledger(root, command=command, status="COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
