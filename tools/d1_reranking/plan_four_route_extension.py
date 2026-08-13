"""Freeze the P12 four-route router/Top20 extension source and execution plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.contracts import EXPECTED_UNIFIED_FINAL_LOCK_SHA256  # noqa: E402
from d1_reranking.four_route_plan import (  # noqa: E402
    D1_SOURCE_NAMES,
    PLAN_RELATIVE_PATH,
    THREE_ROUTE_SOURCE_NAMES,
    write_four_route_plan,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def _named_paths(values: list[str], *, names: tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"source must use NAME=PATH syntax: {raw}")
        if name in result:
            raise ValueError(f"source name is duplicated: {name}")
        result[name] = Path(path)
    if set(result) != set(names):
        missing = sorted(set(names).difference(result))
        extra = sorted(set(result).difference(names))
        raise ValueError(f"source names differ; missing={missing}, extra={extra}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--three-route-run", required=True, type=Path)
    parser.add_argument(
        "--producer-spec",
        required=True,
        type=Path,
        help="immutable Train/Validation-only router, union, and T4 input declaration",
    )
    parser.add_argument(
        "--three-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeat once for each required completed three-route source",
    )
    parser.add_argument(
        "--d1-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeat once for each required D1 application source",
    )
    parser.add_argument(
        "--expected-final-lock-sha256",
        default=EXPECTED_UNIFIED_FINAL_LOCK_SHA256,
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(
    *,
    run_dir: Path,
    three_route_run: Path,
    three_route_sources: dict[str, Path],
    d1_sources: dict[str, Path],
    producer_spec: Path,
    expected_final_lock_sha256: str,
    resume: bool,
) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    return write_four_route_plan(
        d1_run_dir=root,
        completed_three_route_run=three_route_run,
        three_route_sources=three_route_sources,
        d1_sources=d1_sources,
        producer_spec_path=producer_spec,
        code_paths=(
            ROOT / "src/d1_reranking/four_route.py",
            ROOT / "src/d1_reranking/four_route_inputs.py",
            ROOT / "src/d1_reranking/four_route_plan.py",
            ROOT / "src/d1_reranking/four_route_producer.py",
            ROOT / "src/d1_reranking/four_route_t4.py",
            ROOT / "src/d1_reranking/four_route_validation.py",
            ROOT / "src/d1_reranking/four_route_execution.py",
            ROOT / "src/d1_reranking/contracts.py",
            ROOT / "src/unified_reranking/gate.py",
            ROOT / "src/unified_reranking/route_router.py",
            ROOT / "src/unified_reranking/statistics.py",
            ROOT / "tools/d1_reranking/apply_locked_four_route_test.py",
            ROOT / "tools/d1_reranking/assemble_four_route_inputs.py",
            ROOT / "tools/d1_reranking/run_four_route_validation.py",
            Path(__file__),
        ),
        resume=resume,
        expected_final_lock_sha256=expected_final_lock_sha256,
    )


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    three_sources = _named_paths(args.three_source, names=THREE_ROUTE_SOURCE_NAMES)
    d1_sources = _named_paths(args.d1_source, names=D1_SOURCE_NAMES)
    destination = root / PLAN_RELATIVE_PATH
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P12",
        substage="d1_four_route_extension_plan",
        route="CROG_G1_C1_D1",
        pool="four_route_top20_no_dedup",
        evidence_track="validation_router_union",
        method="crog_default_expected_gain",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        plan = run(
            run_dir=root,
            three_route_run=args.three_route_run,
            three_route_sources=three_sources,
            d1_sources=d1_sources,
            producer_spec=args.producer_spec,
            expected_final_lock_sha256=args.expected_final_lock_sha256,
            resume=args.resume,
        )
        state["artifact_path"] = str(destination.resolve())
        state["artifact_sha256"] = sha256_file(destination)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan": str(destination.resolve()),
                "sha256": sha256_file(destination),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
