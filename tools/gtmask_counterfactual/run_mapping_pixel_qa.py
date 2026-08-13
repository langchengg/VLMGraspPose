#!/usr/bin/env python3
"""Run the authorized P2 mapping/annotation pixel QA and no candidate generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pyarrow.csv as pacsv
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.audit import transition_pipeline_status  # noqa: E402
from gtmask_counterfactual.contracts import RunState  # noqa: E402
from gtmask_counterfactual.io import (  # noqa: E402
    artifact_record,
    atomic_json,
)
from gtmask_counterfactual.mapping import EXPECTED_SAMPLE_COUNT  # noqa: E402
from gtmask_counterfactual.mapping_pipeline import (  # noqa: E402
    run_mapping_pixel_qa_bulk,
)
from gtmask_counterfactual.resource import (  # noqa: E402
    collect_fresh_three_by_five_gate,
    exclusive_d1_flock,
    validate_fresh_gate,
    validate_live_resources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_SAMPLE_COUNT)
    parser.add_argument("--minimum-contact-cases", type=int, default=150)
    parser.add_argument("--resume", action="store_true")
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--resource-gate", type=Path)
    gate.add_argument("--collect-resource-gate", action="store_true")
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )
    parser.add_argument(
        "--manual-qa-csv",
        type=Path,
        help=(
            "Human review with sample_id, review_status, reviewer, "
            "reviewed_at_utc, review_signature"
        ),
    )
    return parser.parse_args()


def _regular(path: Path) -> Path:
    value = path.expanduser().resolve()
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"required P1/P2 artifact is absent: {value}")
    return value


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    status = json.loads((root / "pipeline_status.json").read_text(encoding="utf-8"))
    if status.get("status") != RunState.P1_BASELINE_REPLAY_PASS.value:
        raise PermissionError("P2 mapping QA requires P1_BASELINE_REPLAY_PASS")
    manifest_path = _regular(
        root / "02_sample_manifest" / "counterfactual_manifest.parquet"
    )
    join_audit_path = _regular(root / "02_sample_manifest" / "JOIN_AUDIT.json")
    unresolved_path = _regular(
        root / "02_sample_manifest" / "unresolved_samples.csv"
    )
    prelock_path = _regular(
        root / "03_gt_mask_registry" / "gt_mask_registry_prelock.parquet"
    )
    manual_path = _regular(args.manual_qa_csv) if args.manual_qa_csv else None
    manual_rows = pacsv.read_csv(manual_path).to_pylist() if manual_path else None
    with exclusive_d1_flock(root, purpose="GT-mask P2 bulk pixel QA"):
        if args.collect_resource_gate:
            gate_value = collect_fresh_three_by_five_gate(
                repo_root=ROOT,
                rank1_run_dir=args.rank1_run_dir.expanduser().resolve(),
            )
            gate_path = root / "00_audit/resource_gates" / (
                f"p2_mapping_{gate_value['content_sha256'][:20]}.json"
            )
            atomic_json(gate_path, gate_value)
        else:
            gate_path = _regular(args.resource_gate)
            gate_value = json.loads(gate_path.read_text(encoding="utf-8"))
            if not isinstance(gate_value, dict):
                raise ValueError("P2 resource gate must contain a JSON object")
        validate_fresh_gate(gate_value)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_p2_mapping_pixel_qa_launch",
        )
        validate_fresh_gate(gate_value)
        result = run_mapping_pixel_qa_bulk(
            pq.read_table(manifest_path).to_pylist(),
            pq.read_table(prelock_path).to_pylist(),
            run_dir=root,
            expected_count=args.expected_count,
            minimum_contact_cases=args.minimum_contact_cases,
            input_records={
                "counterfactual_manifest": artifact_record(manifest_path),
                "join_audit": artifact_record(join_audit_path),
                "unresolved_samples": artifact_record(unresolved_path),
                "prelock_gt_mask_registry": artifact_record(prelock_path),
                "resource_gate": artifact_record(gate_path),
            },
            manual_qa_rows=manual_rows,
            manual_qa_record=artifact_record(manual_path) if manual_path else None,
            resume=args.resume,
        )
    if result["status"] != "PASS":
        return 2
    transition_pipeline_status(
        root,
        RunState.P2_GT_MAPPING_PASS,
        first_incomplete_stage=RunState.P3_PROTOCOL_LOCKED.value,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
