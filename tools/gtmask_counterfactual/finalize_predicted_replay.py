#!/usr/bin/env python3
"""Close P1 only after derived and exact G1/C1/D1 predicted replay PASS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.predicted_replay import close_predicted_replay  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--derived-reconciliation", required=True, type=Path)
    parser.add_argument("--g1-manifest", required=True, type=Path)
    parser.add_argument("--c1-manifest", required=True, type=Path)
    parser.add_argument("--d1-manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = close_predicted_replay(
        args.run_dir,
        derived_reconciliation=args.derived_reconciliation,
        route_manifests={
            "g1": args.g1_manifest,
            "c1": args.c1_manifest,
            "d1": args.d1_manifest,
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
