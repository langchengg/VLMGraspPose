#!/usr/bin/env python3
"""Prepare or sign the fixed P2 mapping visual-review sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.mapping_review import (  # noqa: E402
    prepare_mapping_review_template,
    sign_mapping_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-dir", required=True, type=Path)
    prepare.add_argument("--resume", action="store_true")
    sign = subparsers.add_parser("sign")
    sign.add_argument("--run-dir", required=True, type=Path)
    sign.add_argument("--review-csv", required=True, type=Path)
    sign.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        path = prepare_mapping_review_template(args.run_dir, resume=args.resume)
        status = "PENDING_MANUAL_QA"
    else:
        path = sign_mapping_review(args.run_dir, args.review_csv, resume=args.resume)
        status = "SIGNED"
    print(json.dumps({"status": status, "artifact": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
