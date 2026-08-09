"""Assemble and validate the P11 bundle without opening Test labels or locking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from unified_reranking.hashing import sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.prelock import LABEL_NORMALIZATION, assemble_prelock_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--candidate-test-labels",
        type=Path,
        help="Immutable fair-source label Parquet; byte-hashed only before the formal claim.",
    )
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument(
        "--code-root",
        action="append",
        type=Path,
        help="File/directory included in the deterministic code bundle (repeatable).",
    )
    parser.add_argument("--label-route-column", default=LABEL_NORMALIZATION["route_column"])
    parser.add_argument("--label-variant-column", default=LABEL_NORMALIZATION["variant_column"])
    parser.add_argument(
        "--include-variant",
        action="append",
        dest="include_variants",
        help="Native source variant to retain after the post-claim read.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    variants = args.include_variants or list(LABEL_NORMALIZATION["include_variants"])
    normalization = {
        "route_column": args.label_route_column,
        "variant_column": args.label_variant_column,
        "include_variants": variants,
    }
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P11_PRELOCK",
        substage="assemble_and_semantically_validate",
        method="hash_bound_label_free_bundle",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = assemble_prelock_bundle(
            run_dir,
            candidate_test_labels_path=args.candidate_test_labels,
            evaluator_path=args.evaluator,
            code_roots=args.code_root,
            normalization=normalization,
        )
        marker = run_dir / "08_lock" / "prelock_assembly_manifest.json"
        state["artifact_path"] = str(marker)
        state["artifact_sha256"] = sha256_file(marker)
    print(
        f"P11_PRELOCK COMPLETE: {result['artifacts']['formal_evaluation_plan']['path']}\n"
        "FORMAL_TEST_LOCK.json was not created; run create_formal_test_lock only after review."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

