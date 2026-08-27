#!/usr/bin/env python3
"""Assemble canonical P6 postprocess inputs without caller artifact paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.postprocess_inputs import (  # noqa: E402
    PostprocessInputAssemblyError,
    assemble_postprocess_inputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inputs, assembly = assemble_postprocess_inputs(args.run_dir, resume=args.resume)
    except PostprocessInputAssemblyError as error:
        print(json.dumps(error.as_dict(), sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "READY",
                "postprocess_inputs": str(inputs),
                "assembly_manifest": str(assembly),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
