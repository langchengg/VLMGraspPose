#!/usr/bin/env python3
"""Materialise canonical frozen final outcomes from protocol-locked sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.final_outcomes import (  # noqa: E402
    materialize_frozen_final_outcomes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    outcomes, authority = materialize_frozen_final_outcomes(
        arguments.run_dir, resume=arguments.resume
    )
    print(
        json.dumps(
            {
                "status": "LOCKED",
                "final_outcomes": str(outcomes),
                "authority": str(authority),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
