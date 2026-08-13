#!/usr/bin/env python3
"""Build or finalize downstream GT-mask counterfactual artifacts.

The CLI intentionally cannot construct tables from raw Test annotations.  It
only consumes a previously hash-bound table manifest produced by the execution
pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gtmask_counterfactual.figures import render_all_figures
from gtmask_counterfactual.finalize import assert_counterfactual_run, finalize_run
from gtmask_counterfactual.reporting import write_reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="render figures and reports from bound tables")
    build.add_argument("--run-dir", required=True, type=Path)
    build.add_argument("--table-manifest", type=Path)
    build.add_argument("--d1-blocker-json", type=Path)
    finalize = subparsers.add_parser("finalize", help="create COMPLETE or PARTIAL final lock")
    finalize.add_argument("--run-dir", required=True, type=Path)
    finalize.add_argument("--d1-blocker-json", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    root = assert_counterfactual_run(arguments.run_dir)
    blocker = None
    if arguments.d1_blocker_json is not None:
        blocker_path = arguments.d1_blocker_json.expanduser().resolve()
        if blocker_path.is_symlink() or not blocker_path.is_file():
            raise ValueError(f"D1 blocker JSON is not a regular file: {blocker_path}")
        blocker = json.loads(blocker_path.read_text(encoding="utf-8"))
        if not isinstance(blocker, dict):
            raise ValueError("D1 blocker JSON must be an object")
    if arguments.command == "build":
        figures = render_all_figures(
            root,
            arguments.table_manifest,
            allow_missing_d1_primary=blocker is not None,
        )
        reports = write_reports(root, arguments.table_manifest, d1_blocker=blocker)
        figure_status = json.loads(figures.read_text(encoding="utf-8"))["status"]
        report_status = json.loads(reports.read_text(encoding="utf-8"))["status"]
        if figure_status != report_status:
            raise RuntimeError("figure/report completion scopes differ")
        print(
            json.dumps(
                {
                    "status": figure_status,
                    "figures": str(figures),
                    "reports": str(reports),
                },
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(finalize_run(root, d1_blocker=blocker), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
