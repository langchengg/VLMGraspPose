"""Assemble the unique production P3 bindings/declaration from canonical P2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.protocol_inputs import assemble_protocol_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bindings, declaration = assemble_protocol_inputs(args.run_dir, resume=args.resume)
    print(f"bindings={bindings}")
    print(f"declaration={declaration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
