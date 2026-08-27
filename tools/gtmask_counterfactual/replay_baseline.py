"""Replay frozen predicted-mask baselines without opening raw Test GT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.audit import require_verified_source  # noqa: E402
from gtmask_counterfactual.baseline import (  # noqa: E402
    BaselineReplayMismatch,
    load_frozen_formal_parquet,
    load_frozen_full_pool_taxonomy,
    replay_baselines,
)
from gtmask_counterfactual.io import (  # noqa: E402
    artifact_record,
    atomic_json,
    atomic_text,
    canonical_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unified-run", type=Path, required=True)
    parser.add_argument("--d1-run", type=Path, required=True)
    parser.add_argument("--unified-outcomes", type=Path)
    parser.add_argument("--d1-outcomes", type=Path)
    parser.add_argument("--unified-full-pool-summary", type=Path)
    return parser.parse_args()


def _mismatch_markdown(error: BaselineReplayMismatch) -> str:
    lines = [
        "# Baseline replay mismatch",
        "",
        "Counterfactual bulk execution is blocked. No raw Test ground-truth rows were read.",
        "",
        "| route | field | observed | expected |",
        "|---|---|---:|---:|",
    ]
    for route, fields in sorted(error.mismatches.items()):
        for field, (observed, expected) in sorted(fields.items()):
            lines.append(f"| {route} | {field} | {observed} | {expected} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    require_verified_source(
        args.run_dir, source_name="unified", source_run=args.unified_run
    )
    require_verified_source(args.run_dir, source_name="d1", source_run=args.d1_run)
    unified_path = args.unified_outcomes or (
        args.unified_run / "09_formal_test/formal_test_per_sample.parquet"
    )
    d1_path = args.d1_outcomes or (
        args.d1_run / "09_formal_test/formal_candidate_score_decision_bundle.parquet"
    )
    full_pool_path = args.unified_full_pool_summary or (
        args.unified_run / "13_failure_galleries/failure_taxonomy_per_sample.parquet"
    )
    unified = load_frozen_formal_parquet(unified_path, source_run=args.unified_run)
    d1 = load_frozen_formal_parquet(d1_path, source_run=args.d1_run)
    full_pool = load_frozen_full_pool_taxonomy(
        full_pool_path, source_run=args.unified_run
    )
    output_dir = args.run_dir.expanduser().resolve() / "04_predicted_replay"
    try:
        result = replay_baselines(unified, d1, unified_full_pool=full_pool)
    except BaselineReplayMismatch as error:
        atomic_text(
            args.run_dir.expanduser().resolve()
            / "00_audit"
            / "BASELINE_REPLAY_MISMATCH.md",
            _mismatch_markdown(error),
        )
        return 2
    result["source_artifacts"] = {
        "d1_formal_outcomes": artifact_record(d1_path),
        "unified_formal_outcomes": artifact_record(unified_path),
        "unified_full_pool_taxonomy": artifact_record(full_pool_path),
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(output_dir / "derived_baseline_reconciliation.json", result)
    atomic_text(
        output_dir / "BASELINE_REPLAY_REPORT.md",
        "# Predicted-mask baseline replay\n\n"
        "Status: PASS. Metrics were recomputed from frozen post-formal outcomes; "
        "raw Test labels and ground-truth rows were not opened.\n\n"
        f"```json\n{json.dumps(result['routes'], indent=2, sort_keys=True)}\n```\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
