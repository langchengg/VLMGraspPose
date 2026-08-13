"""Create or verify the exactly-once counterfactual protocol lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.protocol import (  # noqa: E402
    claim_bulk_execution,
    create_protocol_lock,
    verify_protocol_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--claim-bulk", action="store_true")
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--declaration", type=Path)
    return parser.parse_args()


def _object(path: Path | None, *, name: str) -> dict[str, object]:
    if path is None:
        raise SystemExit(f"--{name} is required with --create")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"--{name} must contain a JSON object")
    return value


def main() -> int:
    args = parse_args()
    if args.create:
        result = create_protocol_lock(
            args.run_dir,
            bindings=_object(args.bindings, name="bindings"),
            declaration=_object(args.declaration, name="declaration"),
        )
        print(result)
    elif args.verify:
        value = verify_protocol_lock(args.run_dir)
        print(value["self_sha256"])
    else:
        print(claim_bulk_execution(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
