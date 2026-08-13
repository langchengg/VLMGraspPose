#!/usr/bin/env python3
"""Build or manually accept the canonical anti-cherry-picking gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.gallery_pipeline import (  # noqa: E402
    accept_gallery_manual_qa,
    prepare_gallery,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-dir", required=True, type=Path)
    prepare.add_argument("--resume", action="store_true")
    gate = prepare.add_mutually_exclusive_group(required=True)
    gate.add_argument("--resource-gate", type=Path)
    gate.add_argument("--collect-resource-gate", action="store_true")
    prepare.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )
    accept = subparsers.add_parser("accept")
    accept.add_argument("--run-dir", required=True, type=Path)
    accept.add_argument("--manual-qa-csv", required=True, type=Path)
    accept.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.command == "prepare":
        path = prepare_gallery(
            arguments.run_dir,
            resume=arguments.resume,
            resource_gate=arguments.resource_gate,
            collect_resource_gate=arguments.collect_resource_gate,
            rank1_run_dir=arguments.rank1_run_dir,
        )
        status = "PENDING_MANUAL_QA"
    else:
        path = accept_gallery_manual_qa(
            arguments.run_dir,
            manual_qa_csv=arguments.manual_qa_csv,
            resume=arguments.resume,
        )
        status = "COMPLETE"
    print(json.dumps({"status": status, "manifest": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
