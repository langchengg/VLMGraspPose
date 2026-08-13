"""Publish a versioned active policy for the nine raw-feature jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import (  # noqa: E402
    artifact_record,
    feature_resource_policy,
)
from d1_reranking.feature_plan import (  # noqa: E402
    load_active_feature_extraction_plan,
)
from d1_reranking.feature_execution import exclusive_json  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def _policy_value(root: Path) -> dict[str, object]:
    plan_path, _plan = load_active_feature_extraction_plan(root)
    return feature_resource_policy(
        plan_path=plan_path,
        source_paths=(
            ROOT / "src/d1_reranking/execution.py",
            ROOT / "src/d1_reranking/feature_execution.py",
            ROOT / "src/d1_reranking/feature_plan.py",
            ROOT / "src/d1_reranking/resource_gate.py",
            ROOT / "tools/d1_reranking/audit_resources.py",
            ROOT / "tools/d1_reranking/authorize_feature_extraction.py",
            ROOT / "tools/d1_reranking/run_feature_extraction_matrix.py",
            ROOT / "tools/d1_reranking/extract_common_features.py",
            Path(__file__),
        ),
    )


def run(
    run_dir: Path, *, expected_value: dict[str, object] | None = None
) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    value = _policy_value(root)
    if expected_value is not None and value != expected_value:
        raise RuntimeError("D1 feature resource policy changed before publication")
    policy_id = str(value["content_sha256"])[:20]
    destination = root / "configs/feature_resource_policies" / f"{policy_id}.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError("immutable D1 feature resource policy differs")
    else:
        assert_writable_prelock(root)
        try:
            exclusive_json(destination, value)
        except FileExistsError:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != value:
                raise RuntimeError("immutable D1 feature resource policy differs")
    plan_path, _plan = load_active_feature_extraction_plan(root)
    source_closure = value["prerequisite"]
    pointer: dict[str, object] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY_POINTER",
        "active_policy": artifact_record(destination),
        "plan": artifact_record(plan_path),
        "source_closure": source_closure,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    assert_writable_prelock(root)
    atomic_json(root / "configs/d1_feature_resource_gate_policy_active.json", pointer)
    return value


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    value = _policy_value(root)
    policy_id = str(value["content_sha256"])[:20]
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P5_RESOURCE",
        substage=f"d1_feature_resource_policy_{policy_id}",
        route="D1",
        pool="top5_top10_allnms",
        evidence_track="matched_common_raw",
        method="unified_common_extractor",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        value = run(root, expected_value=value)
        destination = (
            root
            / "configs/feature_resource_policies"
            / f"{str(value['content_sha256'])[:20]}.json"
        )
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
