"""Reconcile historical D1 snapshots and freeze the unique source contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.provenance import (  # noqa: E402
    SOURCE_CLOSURE_POINTER,
    publish_source_closure,
    reconcile_sources,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.test_access_guard import append_access_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", default="source_reconciliation")
    parser.add_argument("--split", default="none")
    parser.add_argument("--pool", default="none")
    parser.add_argument("--evidence-track", default="none")
    parser.add_argument("--method", default="provenance")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--snapshot-a",
        type=Path,
        default=ROOT
        / "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528",
    )
    parser.add_argument(
        "--compact-source",
        type=Path,
        default=ROOT
        / "HiFi_reproduction/runs/modular_reranking_repeatedfilm_v1_20260729_203147",
    )
    parser.add_argument(
        "--unified-run",
        type=Path,
        default=ROOT / "runs/fair_unified_reranking_20260809_103012",
    )
    parser.add_argument(
        "--snapshot-b",
        type=Path,
        default=ROOT / "HiFi_reproduction/docs",
    )
    parser.add_argument(
        "--snapshot-c",
        type=Path,
        default=ROOT
        / "HiFi_reproduction/runs/grasp_backend_comparison_20260807_090155",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    assert_writable_prelock(run_dir)
    pointer_path = run_dir / "00_audit" / SOURCE_CLOSURE_POINTER
    if pointer_path.exists() and not args.resume:
        raise FileExistsError(
            f"D1 active source closure exists; pass --resume: {pointer_path}"
        )
    closure = reconcile_sources(
        repo_root=ROOT,
        run_dir=run_dir,
        snapshot_a=args.snapshot_a,
        compact_source=args.compact_source,
        unified_run=args.unified_run,
        snapshot_b=args.snapshot_b,
        snapshot_c=args.snapshot_c,
    )
    closure_path = (
        run_dir
        / "00_audit"
        / "source_reconciliations"
        / str(closure["closure_id"])
        / "manifest.json"
    ).resolve()
    assert_writable_prelock(run_dir)
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P1",
        substage=f"d1_snapshot_reconciliation_{closure['closure_id']}",
        route="D1",
        method="provenance",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        state["artifact_path"] = str(closure_path)
        state["artifact_sha256"] = sha256_file(closure_path)
    assert_writable_prelock(run_dir)
    publish_source_closure(run_dir, closure_path)
    opaque_ground_truth = closure["canonical_inputs"]["test"]["opaque_ground_truth"]
    append_access_log(
        run_dir,
        {
            "event": "d1_test_ground_truth_hash_only_source_closure",
            "event_id": f"d1_source_closure:{closure['closure_id']}:opaque_test_gt_hash",
            "path": opaque_ground_truth["path"],
            "sha256": opaque_ground_truth["sha256"],
            "candidate_labels_opened_as_table": False,
            "candidate_label_rows_read": 0,
            "purpose": "opaque prelock authority binding; no Parquet rows parsed",
        },
    )
    opaque_visual = closure["canonical_inputs"]["test"]["opaque_visual_ground_truth"]
    append_access_log(
        run_dir,
        {
            "event": "d1_test_visual_ground_truth_hash_only_source_closure",
            "event_id": (
                f"d1_source_closure:{closure['closure_id']}:opaque_test_visual_gt_hash"
            ),
            "path": opaque_visual["path"],
            "sha256": opaque_visual["sha256"],
            "candidate_labels_opened_as_table": False,
            "candidate_label_rows_read": 0,
            "visual_ground_truth_opened_as_table": False,
            "purpose": "opaque postformal-visual authority binding; no Parquet rows parsed",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
