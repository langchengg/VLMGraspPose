"""Run the production P9 gallery and P10 independent acceptance gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gallery", "independent"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "gallery":
        # P9 owns the gallery producer/validator dependency graph.  Keep this
        # import action-local so the independent process never imports it.
        from gtmask_counterfactual.acceptance import accept_gallery

        result = accept_gallery(args.run_dir, resume=args.resume)
    else:
        from gtmask_counterfactual.independent_acceptance import (
            accept_independent_recompute,
        )

        result = accept_independent_recompute(args.run_dir, resume=args.resume)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
