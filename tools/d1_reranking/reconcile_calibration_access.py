"""Reconcile access events for already-frozen label-free Test calibrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import canonical_sha256  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.test_access_guard import append_access_log  # noqa: E402


POOLS = ("top5", "top10", "allnms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def reconcile(root: Path) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for pool in POOLS:
        manifest_path = root / f"05_calibration/{pool}/test_application_manifest.json"
        manifest = load_content_manifest(
            manifest_path,
            name=f"D1 {pool} Test calibration application",
            statuses=("COMPLETE",),
        )
        if (
            manifest.get("candidate_test_labels_read") is not False
            or manifest.get("configuration", {}).get("split") != "test"
            or manifest.get("configuration", {}).get("pool") != pool
        ):
            raise PermissionError(f"D1 {pool} Test calibration provenance differs")
        verify_artifact_records_recursive(
            {"sources": manifest["sources"], "artifacts": manifest["artifacts"]},
            name=f"D1 {pool} Test calibration application",
            require_at_least_one=True,
        )
        output_record = artifact_record(manifest_path)
        identity = {
            "stage": f"d1_calibration_{pool}_test",
            "output_manifest": output_record,
        }
        append_access_log(
            root,
            {
                "event": "prelock_label_free_test_stage",
                "event_id": canonical_sha256(identity)[:24],
                **identity,
                "purpose": "reconcile frozen label-free Test calibration access evidence",
                "candidate_labels_opened_as_table": False,
                "candidate_test_labels_read": False,
                "inputs": [
                    manifest["sources"]["candidates"],
                    manifest["sources"]["calibration_manifest"],
                ],
            },
        )
        outputs[pool] = output_record
    return outputs


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_calibration_access_reconciliation",
        route="D1",
        pool="top5_top10_allnms",
        evidence_track="label_free_test_calibration",
        method="hash_verified_access_reconciliation",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        outputs = reconcile(root)
        state["artifact_path"] = json.dumps(
            {name: value["path"] for name, value in outputs.items()}, sort_keys=True
        )
        state["artifact_sha256"] = json.dumps(
            {name: value["sha256"] for name, value in outputs.items()}, sort_keys=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
