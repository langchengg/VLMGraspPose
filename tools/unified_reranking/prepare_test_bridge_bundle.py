"""Freeze label-free Test inputs for the post-lock 2x2 attribution bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.hashing import sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log
from unified_reranking.test_bridge import build_label_free_test_bridge


HISTORICAL_RUN = (
    ROOT / "HiFi_reproduction" / "runs" / "g1_c1_complete_reranking_20260806T084131Z"
)
MODULAR_RUN = (
    ROOT
    / "HiFi_reproduction"
    / "runs"
    / "modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--historical-run", type=Path, default=HISTORICAL_RUN)
    parser.add_argument("--modular-run", type=Path, default=MODULAR_RUN)
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--denominator", type=Path)
    return parser.parse_args()


def run(
    *,
    run_dir: Path,
    historical_run: Path,
    modular_run: Path,
    evaluator_path: Path | None = None,
    denominator_path: Path | None = None,
) -> dict[str, object]:
    root = run_dir.resolve()
    historical = historical_run.resolve()
    modular = modular_run.resolve()
    manifest = build_label_free_test_bridge(
        run_dir=root,
        historical_candidates={
            route: historical / "data" / f"frozen_{route}_test_top5_candidates.parquet"
            for route in ("g1", "c1")
        },
        historical_ground_truth_path=modular / "manifests" / "test_labels.parquet",
        historical_source_manifest_path=modular / "manifests" / "experiment_lock.json",
        historical_inventory_path=historical / "audit" / "frozen_pool_inventory.json",
        evaluator_path=(evaluator_path or root / "configs" / "canonical_evaluator.py"),
        denominator_path=denominator_path,
    )
    append_access_log(
        root,
        {
            "event": "label_free_test_bridge_input",
            "output_manifest": str(
                (
                    root
                    / "11_attribution_bridge"
                    / "test_bridge_input"
                    / "manifest.json"
                ).resolve()
            ),
            "output_manifest_sha256": sha256_file(
                root / "11_attribution_bridge" / "test_bridge_input" / "manifest.json"
            ),
            "candidate_test_labels_read": False,
            "historical_test_ground_truth_rows_read": False,
            "historical_test_ground_truth_sha256": manifest["sources"][
                "historical_ground_truth"
            ]["sha256"],
        },
    )
    return manifest


def main() -> int:
    args = parse_args()
    root = args.run_dir.resolve()
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="label_free_test_bridge_input",
        route="G1+C1",
        evidence_track="attribution_bridge",
        pool="fair_gaussian+historical_nms",
        method="label_free_selector_pool_bridge",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(
            run_dir=root,
            historical_run=args.historical_run,
            modular_run=args.modular_run,
            evaluator_path=args.evaluator,
            denominator_path=args.denominator,
        )
        manifest = (
            root / "11_attribution_bridge" / "test_bridge_input" / "manifest.json"
        )
        state["artifact_path"] = str(manifest.resolve())
        state["artifact_sha256"] = sha256_file(manifest)
        state["candidate_test_labels_read"] = False
        state["historical_test_ground_truth_rows_read"] = False
        state["manifest_status"] = result["status"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run"]
