"""Apply P12 Validation locks to Test without opening candidate labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record  # noqa: E402
from d1_reranking.four_route_producer import (  # noqa: E402
    apply_locked_four_route_test,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.test_access_guard import append_access_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--router-test-input", required=True, type=Path)
    parser.add_argument("--union-test-features", required=True, type=Path)
    parser.add_argument("--denominator", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(
    run_dir: Path,
    *,
    router_test_input: Path,
    union_test_features: Path,
    denominator: Path | None,
    resume: bool,
) -> dict[str, dict[str, object]]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    router_selection = (
        root / "13_four_route_extension/router/router_selection_manifest.json"
    )
    union_selection = root / "13_four_route_extension/union/selected_union_ranker.json"
    denominator_path = (
        denominator.expanduser().resolve()
        if denominator is not None
        else root / "01_manifests/d1_paired_manifest.parquet"
    )
    results = apply_locked_four_route_test(
        router_selection_path=router_selection,
        router_test_input_path=router_test_input,
        union_selection_path=union_selection,
        union_test_feature_path=union_test_features,
        denominator_path=denominator_path,
        output_dir=root / "08_lock/formal_inputs",
        resume=resume,
    )
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "event_id": "d1_p12_four_route_test_application",
            "stage": "d1_p12_four_route_test_application",
            "candidate_test_labels_read": False,
            "candidate_labels_opened_as_table": False,
            "inputs": [
                artifact_record(router_selection),
                artifact_record(union_selection),
                artifact_record(router_test_input),
                artifact_record(union_test_features),
                artifact_record(denominator_path),
            ],
            "outputs": {
                name: artifact_record(
                    root / f"08_lock/formal_inputs/{name}/manifest.json"
                )
                for name in results
            },
        },
    )
    return results


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    output = root / "08_lock/formal_inputs"
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P12",
        substage="d1_four_route_label_free_test_application",
        route="CROG_G1_C1_D1",
        pool="four_route_top20_no_dedup",
        evidence_track="locked_label_free_formal_inputs",
        method="crog_default_router_and_selected_top20_union",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        results = run(
            root,
            router_test_input=args.router_test_input,
            union_test_features=args.union_test_features,
            denominator=args.denominator,
            resume=args.resume,
        )
        manifests = {name: output / name / "manifest.json" for name in sorted(results)}
        state["artifact_path"] = json.dumps(
            {name: str(path.resolve()) for name, path in manifests.items()},
            sort_keys=True,
        )
        state["artifact_sha256"] = json.dumps(
            {name: sha256_file(path) for name, path in manifests.items()},
            sort_keys=True,
        )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "systems": sorted(results),
                "output": str(output.resolve()),
                "candidate_test_labels_read": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
