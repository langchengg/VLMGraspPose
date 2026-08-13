"""Idempotently recover the FEATURES_READY transition after COMPLETE replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.contracts import RunState  # noqa: E402
from d1_reranking.feature_replay import feature_lifecycle_evidence  # noqa: E402
from d1_reranking.run import (  # noqa: E402
    assert_writable_prelock,
    transition_pipeline_status,
)
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def run(run_dir: Path) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    feature_lifecycle_evidence(root)
    record = transition_pipeline_status(
        root,
        status=RunState.FEATURES_READY,
        first_incomplete_stage="P7_SPLITS_AND_OOF",
        formal_test_executed=False,
        test_candidate_labels_read=False,
        formal_test_execution_count=0,
    )
    return {"pipeline_status": record}


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P5",
        substage="d1_raw_feature_extraction_finalize",
        route="D1",
        pool="top5_top10_allnms",
        evidence_track="matched_common_raw",
        method="unified_common_extractor",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root)
        status_path = root / "pipeline_status.json"
        state["artifact_path"] = str(status_path)
        state["artifact_sha256"] = sha256_file(status_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
