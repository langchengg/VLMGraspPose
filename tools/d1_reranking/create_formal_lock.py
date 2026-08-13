"""Create the D1 P14 formal-Test lock from a label-free evaluation plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.formal import create_formal_lock  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--evaluation-plan", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    lock_path = root / "08_lock" / "FORMAL_TEST_LOCK.json"
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P14",
        substage="d1_formal_test_lock",
        route="D1",
        pool="top5_top10_allnms",
        evidence_track="T2_plus_locked_secondary",
        method="R0_R7_K_router_union",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        create_formal_lock(
            root,
            evaluation_plan_path=args.evaluation_plan,
            code_paths=(
                ROOT / "src/d1_reranking/formal.py",
                ROOT / "src/d1_reranking/contracts.py",
                ROOT / "src/d1_reranking/prelock.py",
                ROOT / "src/d1_reranking/four_route_test_inputs.py",
                ROOT / "src/d1_reranking/four_route_validation.py",
                ROOT / "src/d1_reranking/postformal.py",
                ROOT / "src/d1_reranking/postformal_sources.py",
                ROOT / "src/d1_reranking/ranker_contributions.py",
                ROOT / "src/d1_reranking/postformal_evidence.py",
                ROOT / "src/d1_reranking/validation_evidence.py",
                ROOT / "src/d1_reranking/ablation_schema_adapter.py",
                ROOT / "src/d1_reranking/primary_source_adapter.py",
                ROOT / "src/d1_reranking/feature_source_adapter.py",
                ROOT / "src/d1_reranking/feature_replay_adapter.py",
                ROOT / "src/d1_reranking/k_source_adapter.py",
                ROOT / "src/d1_reranking/k_replay_adapter.py",
                ROOT / "src/d1_reranking/k_formal_replay_adapter.py",
                ROOT / "src/d1_reranking/lightweight_audits.py",
                ROOT / "src/unified_reranking/artifacts.py",
                ROOT / "src/unified_reranking/hashing.py",
                Path(__file__),
                ROOT / "tools/d1_reranking/run_formal_test_once.py",
                ROOT / "tools/d1_reranking/independent_recompute.py",
                ROOT / "tools/d1_reranking/build_postformal.py",
                ROOT / "tools/d1_reranking/prepare_postformal_sources.py",
                ROOT / "tools/d1_reranking/build_ranker_contributions.py",
                ROOT / "tools/d1_reranking/assemble_validation_evidence_tables.py",
                ROOT / "tools/d1_reranking/select_ablation_with_schema_adapter.py",
                ROOT / "tools/d1_reranking/prepare_primary_source_adapter.py",
                ROOT / "tools/d1_reranking/prepare_feature_source_adapter.py",
                ROOT / "tools/d1_reranking/prepare_k_source_adapter.py",
                ROOT / "tools/d1_reranking/assemble_lightweight_audits.py",
                ROOT / "tools/d1_reranking/write_postformal_evidence.py",
                ROOT / "tools/d1_reranking/assemble_four_route_test_inputs.py",
                ROOT / "tools/d1_reranking/apply_locked_four_route_test.py",
                ROOT / "tools/d1_reranking/finalize_d1_run.py",
            ),
        )
        state["artifact_path"] = str(lock_path)
        state["artifact_sha256"] = sha256_file(lock_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
