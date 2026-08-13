"""Freeze the audited K-plan native-rank dtype compatibility record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.k_source_adapter import (  # noqa: E402
    ADAPTER_RELATIVE,
    prepare_k_source_adapter,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
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
    destination = root / ADAPTER_RELATIVE
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_k_source_adapter_v1",
        route="D1",
        pool="top10_allnms",
        evidence_track="T2_T3",
        method="selected_primary_fixed",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        value = prepare_k_source_adapter(root, resume=args.resume)
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
    print(json.dumps({"status": value["status"], "adapter": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
