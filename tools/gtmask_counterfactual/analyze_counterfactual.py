#!/usr/bin/env python3
"""Analyze six hash-bound saved GT-mask frames; never generate candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.postprocess import (  # noqa: E402
    INPUT_MANIFEST_RELATIVE_PATH,
    run_postprocess,
)
from gtmask_counterfactual.protocol import LOCK_RELATIVE_PATH  # noqa: E402
from gtmask_counterfactual.statistics import BOOTSTRAP_ITERATIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("test",), default="test")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    root = arguments.run_dir.expanduser().resolve()
    result = run_postprocess(
        root,
        protocol_lock=root / LOCK_RELATIVE_PATH,
        input_manifest=root / INPUT_MANIFEST_RELATIVE_PATH,
        resume=arguments.resume,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
    )
    print(
        json.dumps(
            {
                "status": json.loads(result.read_text(encoding="utf-8"))["status"],
                "scientific_role": "post-formal oracle stage-replacement diagnostic",
                "route_status": str(result),
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "training_or_selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
