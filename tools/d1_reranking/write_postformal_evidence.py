"""Write the label-free, pre-P13 D1 postformal evidence declaration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.postformal_evidence import (  # noqa: E402
    EVIDENCE_RELATIVE_PATH,
    assemble_postformal_evidence,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--q-saturation-threshold", required=True, type=float)
    parser.add_argument("--mask-quality-threshold", required=True, type=float)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    destination = root / EVIDENCE_RELATIVE_PATH
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_postformal_evidence_predeclaration",
        route="D1",
        evidence_track="validation_and_label_free_test_only",
        method="exact_hash_predeclaration",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        assemble_postformal_evidence(
            root,
            q_saturation_threshold=args.q_saturation_threshold,
            mask_quality_threshold=args.mask_quality_threshold,
            resume=args.resume,
        )
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
