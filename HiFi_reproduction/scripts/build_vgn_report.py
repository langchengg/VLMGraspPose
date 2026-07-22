#!/usr/bin/env python3
"""Build the truthful OCID-VLG/VGN report and qualitative gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.experiments.report_builder import build_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocid-output", type=Path, required=True)
    parser.add_argument("--sim-output", type=Path)
    parser.add_argument("--real-robot-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = build_report(
        args.ocid_output,
        args.sim_output,
        args.output,
        real_robot_output=args.real_robot_output,
    )
    print(
        json.dumps(
            {"status": "report_built", **{key: str(value) for key, value in paths.items()}},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
