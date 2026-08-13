"""Freeze the exact nine-job D1 raw matched-common feature plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.feature_plan import (  # noqa: E402
    FEATURE_PLAN_POINTER_RELATIVE,
    publish_feature_extraction_plan,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / "HiFi_reproduction/.venv-grasp4dof/bin/python",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _tool_paths() -> tuple[Path, ...]:
    return (
        ROOT / "src/d1_reranking/contracts.py",
        ROOT / "src/d1_reranking/execution.py",
        ROOT / "src/d1_reranking/feature_execution.py",
        ROOT / "src/d1_reranking/feature_plan.py",
        ROOT / "src/d1_reranking/feature_replay.py",
        ROOT / "src/d1_reranking/provenance.py",
        ROOT / "src/d1_reranking/resource_gate.py",
        ROOT / "src/d1_reranking/run.py",
        ROOT / "src/unified_reranking/artifacts.py",
        ROOT / "src/unified_reranking/feature_extractors/common.py",
        ROOT / "src/unified_reranking/hashing.py",
        ROOT / "tools/unified_reranking/extract_common_features.py",
        ROOT / "tools/d1_reranking/extract_common_features.py",
        ROOT / "tools/d1_reranking/authorize_feature_extraction.py",
        ROOT / "tools/d1_reranking/run_feature_extraction_matrix.py",
        ROOT / "tools/d1_reranking/finalize_feature_extraction.py",
        ROOT / "tools/d1_reranking/write_feature_resource_policy.py",
        ROOT / "tools/d1_reranking/audit_resources.py",
        Path(__file__),
    )


def run(
    run_dir: Path, *, python: Path, resume: bool
) -> tuple[Path, dict[str, object]]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    return publish_feature_extraction_plan(
        root,
        python_path=python,
        tool_paths=_tool_paths(),
        resume=resume,
    )


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    pointer_path = root / FEATURE_PLAN_POINTER_RELATIVE
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P5",
        substage="d1_raw_feature_extraction_plan",
        route="D1",
        pool="top5_top10_allnms",
        evidence_track="matched_common_raw",
        method="unified_common_extractor",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        destination, plan = run(root, python=args.python, resume=args.resume)
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
        print(
            json.dumps(
                {
                    "status": plan["status"],
                    "job_count": plan["job_count"],
                    "plan": str(destination),
                    "active_pointer": str(pointer_path),
                    "sha256": state["artifact_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
