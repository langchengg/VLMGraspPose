"""Write the immutable, preformal D1 statistics declaration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.formal import (  # noqa: E402
    STATISTICS_CONFIG_RELATIVE_PATH,
    assemble_statistics_config,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    destination = root / STATISTICS_CONFIG_RELATIVE_PATH
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_predeclared_statistics_config",
        route="D1",
        pool="top5_top10_allnms_four_route_top20",
        evidence_track="formal_statistics_declaration",
        method="exact_mcnemar_scene_bootstrap_holm",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        assemble_statistics_config(
            root,
            bootstrap_seed=args.bootstrap_seed,
            resume=args.resume,
        )
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
