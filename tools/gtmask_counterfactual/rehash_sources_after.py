#!/usr/bin/env python3
"""Rehash both frozen source inventories under a fresh heavy-work lease."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.audit import write_source_immutability_after  # noqa: E402
from gtmask_counterfactual.io import artifact_record  # noqa: E402
from gtmask_counterfactual.resource import (  # noqa: E402
    exclusive_d1_flock,
    validate_fresh_gate,
    validate_live_resources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resource-gate", required=True, type=Path)
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )
    return parser.parse_args()


def _load_gate(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"resource gate is not a regular file: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("resource gate must contain a JSON object")
    return value


def main() -> int:
    args = parse_args()
    run = args.run_dir.expanduser().resolve()
    with exclusive_d1_flock(run, purpose="GT-mask terminal source rehash"):
        gate = _load_gate(args.resource_gate)
        validate_fresh_gate(gate)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_source_rehash_after_launch",
        )
        output = write_source_immutability_after(run)
    print(json.dumps({"status": "PASS", "artifact": artifact_record(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
