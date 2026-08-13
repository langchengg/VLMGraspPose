"""Normalize frozen three-route router/Top15 references for D1 P14."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.reference_systems import assemble_reference_systems  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--three-route-decisions", required=True, type=Path)
    parser.add_argument("--top15-decisions", required=True, type=Path)
    parser.add_argument("--top15-ranking", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_read_only_reference_systems",
        route="CROG_G1_C1",
        evidence_track="completed_three_route_formal_outputs",
        pool="three_route_top1_top15",
        method="exact_normalization_without_labels",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        results = assemble_reference_systems(
            root,
            three_route_decisions_path=args.three_route_decisions,
            top15_decisions_path=args.top15_decisions,
            top15_ranking_path=args.top15_ranking,
            resume=args.resume,
        )
        paths = {
            name: root / f"08_lock/formal_inputs/{name}/manifest.json"
            for name in results
        }
        state["artifact_path"] = json.dumps(
            {name: str(path) for name, path in paths.items()}, sort_keys=True
        )
        state["artifact_sha256"] = json.dumps(
            {name: sha256_file(path) for name, path in paths.items()}, sort_keys=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
