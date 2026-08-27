#!/usr/bin/env python3
"""Materialize the protocol-bound Test grasp geometry after the P3 claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.gt_grasp_authority import (  # noqa: E402
    materialize_gt_grasp_authority,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    output = materialize_gt_grasp_authority(
        arguments.run_dir,
        protocol_lock=arguments.protocol_lock,
        resume=arguments.resume,
    )
    print(json.dumps({"status": "COMPLETE", "manifest": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
