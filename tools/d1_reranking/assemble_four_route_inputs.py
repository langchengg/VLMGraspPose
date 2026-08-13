"""Assemble fixed Train/Validation-only P12 router and Top20 inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.four_route_inputs import (  # noqa: E402
    assemble_four_route_development_inputs,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--three-route-run", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    output = root / "configs/p12_validation_producer_spec.json"
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P12",
        substage="d1_four_route_development_inputs",
        route="CROG_G1_C1_D1",
        pool="exact_four_route_top20",
        evidence_track="T2_router_union",
        method="deterministic_input_assembly",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        assemble_four_route_development_inputs(
            d1_run_dir=root,
            three_route_run=args.three_route_run,
            resume=args.resume,
        )
        state["artifact_path"] = str(output)
        state["artifact_sha256"] = sha256_file(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
