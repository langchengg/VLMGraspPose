#!/usr/bin/env python3
"""Reconcile locked label-free visual assets for GT-mask case boards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.io import artifact_record  # noqa: E402
from gtmask_counterfactual.visual_assets import (  # noqa: E402
    write_visual_asset_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--unified-samples", required=True, type=Path)
    parser.add_argument("--d1-samples", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output = write_visual_asset_registry(
        arguments.run_dir,
        protocol_lock=arguments.protocol_lock,
        unified_samples=artifact_record(arguments.unified_samples.expanduser().resolve()),
        d1_samples=artifact_record(arguments.d1_samples.expanduser().resolve()),
    )
    print(json.dumps({"status": "COMPLETE", "manifest": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
