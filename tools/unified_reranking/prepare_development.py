"""Prepare grouped folds and candidate supervision for available development pools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.development import prepare_development_contracts
from unified_reranking.hashing import sha256_file
from unified_reranking.ledger import ledger_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--fair-run",
        type=Path,
        default=ROOT / "runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523",
    )
    parser.add_argument(
        "--modular-run",
        type=Path,
        default=ROOT / "HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500",
    )
    parser.add_argument(
        "--fold-source",
        type=Path,
        default=ROOT / "HiFi_reproduction/runs/g1_c1_complete_reranking_20260806T084131Z/03_splits/fold_assignments.parquet",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P2_P5",
        substage="development_labels_and_grouped_folds",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        prepare_development_contracts(
            run_dir,
            args.fair_run.resolve(),
            args.modular_run.resolve(),
            args.fold_source.resolve(),
        )
        artifact = run_dir / "03_features" / "development_label_audit.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
