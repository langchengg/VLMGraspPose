"""Freeze the P2 resource policy against the immutable D1 source closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import candidate_resource_policy  # noqa: E402
from d1_reranking.provenance import load_source_closure  # noqa: E402
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
    closure_path, _closure = load_source_closure(root)
    return candidate_resource_policy(
        source_closure_path=closure_path,
        source_paths=(
            ROOT / "src/d1_reranking/candidates.py",
            ROOT / "src/d1_reranking/contracts.py",
            ROOT / "src/d1_reranking/io.py",
            ROOT / "src/d1_reranking/resource_gate.py",
            ROOT / "src/d1_reranking/execution.py",
            ROOT / "src/unified_reranking/artifacts.py",
            ROOT / "src/unified_reranking/hashing.py",
            ROOT / "tools/d1_reranking/audit_resources.py",
            ROOT / "tools/d1_reranking/build_candidates.py",
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
        raise RuntimeError("D1 candidate resource policy changed before publication")
    policy_id = str(value["content_sha256"])[:20]
    destination = root / "configs" / "candidate_resource_policies" / f"{policy_id}.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError("immutable D1 candidate resource policy differs")
    else:
        atomic_json(destination, value)
    pointer: dict[str, object] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY_POINTER",
        "active_policy": {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        },
        "source_closure": value["prerequisite"],
        "candidate_test_labels_read": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(
        root / "configs" / "d1_candidate_resource_gate_policy_active.json", pointer
    )
    return value


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    value = _policy_value(root)
    policy_id = str(value["content_sha256"])[:20]
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P2_RESOURCE",
        substage=f"d1_candidate_resource_policy_{policy_id}",
        route="D1",
        pool="allnms_top10_top5",
        evidence_track="none",
        method="candidate_projection",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        value = run(root, expected_value=value)
        destination = (
            root
            / "configs"
            / "candidate_resource_policies"
            / f"{str(value['content_sha256'])[:20]}.json"
        )
        state["artifact_path"] = str(destination)
        state["artifact_sha256"] = sha256_file(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
