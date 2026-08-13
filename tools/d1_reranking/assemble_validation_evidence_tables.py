"""Assemble fixed, Validation-only D1 evidence tables for P13."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from d1_reranking.run import assert_writable_prelock  # noqa: E402
from d1_reranking.validation_evidence import (  # noqa: E402
    MANIFEST_RELATIVE_PATH,
    assemble_validation_evidence_tables,
)
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    output = root / MANIFEST_RELATIVE_PATH
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_validation_evidence_tables",
        route="D1",
        evidence_track="validation_only",
        method="exact_semantic_replay",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        assemble_validation_evidence_tables(root, resume=args.resume)
        state["artifact_path"] = str(output.resolve())
        state["artifact_sha256"] = sha256_file(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
