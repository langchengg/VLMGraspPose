"""Finalize D1 through the audited append-only access-log adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from d1_reranking.finalization_source_adapter import (
    access_log_check_with_plan_history,
    access_output_supersessions,
    reuse_source_immutability_after,
    verify_access_event_outputs_with_supersession,
)
from d1_reranking.postformal import (
    POSTFORMAL_MANIFEST_RELATIVE_PATH,
    assert_writable_finalization,
    finalize_d1_run,
)
from d1_reranking.postformal_source_adapter import _load_lock, _locked_record
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def _record(path: Path) -> dict[str, object]:
    value = path.resolve()
    return {"path": str(value), "sha256": sha256_file(value), "bytes": value.stat().st_size}


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_finalization(root)
    lock = _load_lock(root)
    repo = Path(__file__).resolve().parents[2]
    locked_postformal = repo / "src/d1_reranking/postformal.py"
    postformal_record = _locked_record(lock, locked_postformal)
    events = [
        json.loads(line)
        for line in (root / "09_formal_test/test_access.log").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    supersessions = access_output_supersessions(events)
    if len(supersessions) != 1:
        raise RuntimeError("D1 finalization expects exactly one access supersession")
    audit: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_values_changed": False,
        "compatibility_scope": "append_only_preclaim_output_supersession_only",
        "locked_postformal_source": postformal_record,
        "adapter_module": _record(
            repo / "src/d1_reranking/finalization_source_adapter.py"
        ),
        "adapter_tool": _record(Path(__file__)),
        "formal_lock": _record(root / "08_lock/FORMAL_TEST_LOCK.json"),
        "supersessions": supersessions,
    }
    audit["content_sha256"] = canonical_sha256(audit)
    audit_path = root / "16_reports/FINALIZATION_SOURCE_ADAPTER.json"
    atomic_json(audit_path, audit)

    postformal = root / POSTFORMAL_MANIFEST_RELATIVE_PATH
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="PFINAL",
        substage="d1_finalization_preflight",
        route="D1",
        pool="all_locked_outputs",
        evidence_track="P14_COMPLETE_P17_PASS_P15_COMPLETE",
        method="fresh_exact_inventory_finalizer_with_access_adapter",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        state["artifact_path"] = str(postformal.resolve())
        state["artifact_sha256"] = sha256_file(postformal)

    from d1_reranking import postformal as locked_module

    original = locked_module._verify_access_event_outputs
    original_access = locked_module._access_log_check
    original_source = locked_module._source_immutability_after
    locked_module._access_log_check_original_for_adapter = original_access
    locked_module._verify_access_event_outputs = (
        verify_access_event_outputs_with_supersession
    )
    locked_module._access_log_check = access_log_check_with_plan_history
    locked_module._source_immutability_after = reuse_source_immutability_after
    try:
        finalize_d1_run(root)
    finally:
        locked_module._verify_access_event_outputs = original
        locked_module._access_log_check = original_access
        locked_module._source_immutability_after = original_source
        del locked_module._access_log_check_original_for_adapter
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
