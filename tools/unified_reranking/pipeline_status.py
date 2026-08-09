"""Report resumable P2-P15 readiness and fail closed on unsafe stage requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.pipeline_status import (
    STAGE_BY_NAME,
    assert_stage_ready,
    audit_pipeline_readiness,
    write_pipeline_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--require-ready",
        choices=tuple(STAGE_BY_NAME),
        help="Exit non-zero unless this stage is already complete or safe to resume.",
    )
    parser.add_argument("--write-status", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report = audit_pipeline_readiness(run_dir)
    if args.write_status:
        write_pipeline_status(run_dir, report)
    if args.require_ready:
        assert_stage_ready(report, args.require_ready)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        workers = report["legacy_ranker_workers"]
        print(f"Run: {run_dir}")
        print(
            f"State: {report['status']}; Test labels read: "
            f"{'yes' if report['candidate_test_labels_read'] else 'no'} "
            f"({report['candidate_test_label_access_state']}, "
            f"events={report['candidate_test_label_read_event_count']})"
        )
        print(
            "Legacy ranker workers: "
            + (", ".join(str(worker.get("pid")) for worker in workers) if workers else "none")
        )
        for name, stage in report["stages"].items():
            print(f"{name:11s} {stage['status']:30s} {stage['title']}")
        next_stage = report["first_actionable_stage"]
        if next_stage:
            print(f"\nNext stage: {next_stage}")
            for command in report["stages"][next_stage]["next_commands"]:
                print(f"  {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
