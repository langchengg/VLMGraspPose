"""Assemble exact label-free P12 Test router, Top20, and T4 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.four_route_test_inputs import (  # noqa: E402
    TEST_INPUTS_RELATIVE,
    assemble_four_route_test_inputs,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.test_access_guard import append_access_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--three-route-run", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    manifest_path = root / TEST_INPUTS_RELATIVE / "manifest.json"
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P12",
        substage="d1_four_route_label_free_test_inputs",
        route="CROG_G1_C1_D1",
        pool="exact_four_route_top20",
        evidence_track="router_top20_t4_test",
        method="deterministic_label_free_assembly",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = assemble_four_route_test_inputs(
            d1_run_dir=root,
            three_route_run=args.three_route_run,
            resume=args.resume,
        )
        append_access_log(
            root,
            {
                "event": "prelock_label_free_test_stage",
                "event_id": "d1_p12_four_route_test_inputs",
                "stage": "d1_p12_four_route_test_inputs",
                "candidate_test_labels_read": False,
                "candidate_labels_opened_as_table": False,
                "inputs": [
                    record
                    for record in result["sources"].values()
                    if isinstance(record, dict)
                    and "path" in record
                    and "sha256" in record
                ]
                + list(result["sources"]["candidate_top5"].values()),
                "outputs": {
                    name: record
                    for name, record in result["artifacts"].items()
                },
            },
        )
        state["artifact_path"] = str(manifest_path.resolve())
        state["artifact_sha256"] = sha256_file(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
                "candidate_test_labels_read": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
