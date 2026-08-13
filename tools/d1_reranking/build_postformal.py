"""Build D1 P15 failure analysis, tables, figures, cases, and reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.postformal import (  # noqa: E402
    POSTFORMAL_MANIFEST_RELATIVE_PATH,
    assert_writable_postformal,
    build_postformal_artifacts,
)
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


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
        build_postformal_artifacts(root)
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
