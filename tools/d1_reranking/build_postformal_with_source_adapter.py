"""Build D1 P15 through the audited dataclass evaluator-loader adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from d1_reranking.postformal import (
    POSTFORMAL_MANIFEST_RELATIVE_PATH,
    assert_writable_postformal,
)
from d1_reranking.postformal_source_adapter import run_adapted_postformal
from unified_reranking.hashing import sha256_file
from unified_reranking.ledger import ledger_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_postformal(root)
    destination = root / POSTFORMAL_MANIFEST_RELATIVE_PATH
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P15",
        substage="d1_postformal_artifacts",
        route="D1",
        pool="top5_top10_allnms_four_route_top20",
        evidence_track="formal_test_postformal_only",
        method="failure_funnel_objective_mismatch_reports",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run_adapted_postformal(root)
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
