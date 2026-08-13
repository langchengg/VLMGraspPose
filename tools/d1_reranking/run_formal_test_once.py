"""Execute the single locked D1 formal-Test transaction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.formal import run_formal_test_once  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def _execute(root: Path, *, command: str) -> None:
    execution_path = root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    if execution_path.exists():
        raise PermissionError("D1 formal Test has already been claimed/executed")

    def after_claim(evaluate):  # type: ignore[no-untyped-def]
        with ledger_stage(
            root / "run_ledger.sqlite",
            stage="P14",
            substage="d1_exactly_once_formal_test",
            route="D1",
            pool="top5_top10_allnms",
            evidence_track="locked_formal_bundle",
            method="R0_R7_K_router_union",
            command=command,
        ) as state:
            result = evaluate()
            state["artifact_path"] = str(execution_path)
            state["artifact_sha256"] = sha256_file(execution_path)
            return result

    # The core creates the O_EXCL claim before invoking this ledger wrapper.
    run_formal_test_once(root, transaction_wrapper=after_claim)


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    _execute(root, command=" ".join(map(str, sys.argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
