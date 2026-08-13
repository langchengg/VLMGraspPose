"""Freeze canonical label-free D1 All-NMS, Top-10 and Top-5 pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.candidates import build_canonical_candidate_split  # noqa: E402
from d1_reranking.execution import (  # noqa: E402
    artifact_record,
    exclusive_heavy_resource_lease,
    load_content_manifest,
    validate_candidate_resource_gate,
)
from d1_reranking.provenance import load_source_closure  # noqa: E402
from d1_reranking.resource_gate import (  # noqa: E402
    collect_resource_snapshot,
    evaluate_resource_snapshot,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.artifacts import verified_artifact_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", default="candidates")
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="all"
    )
    parser.add_argument(
        "--pool", choices=("all", "allnms", "top10", "top5"), default="all"
    )
    parser.add_argument("--evidence-track", default="none")
    parser.add_argument("--method", default="native")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def _run_under_lease(args: argparse.Namespace, run_dir: Path) -> int:
    """Revalidate authorization inside the held repository-wide lease."""

    assert_writable_prelock(run_dir)
    closure_path, closure = load_source_closure(run_dir)
    gate = validate_candidate_resource_gate(run_dir, require_fresh=True)
    pointer = load_content_manifest(
        run_dir / "00_audit" / "RESOURCE_AUDIT.json",
        name="D1 candidate resource gate pointer",
        statuses=("PASS",),
    )
    gate_path = verified_artifact_path(
        pointer.get("latest_gate", {}), name="D1 candidate resource gate"
    )
    splits = ("train", "validation", "test") if args.split == "all" else (args.split,)
    for split in splits:
        assert_writable_prelock(run_dir)
        live = collect_resource_snapshot(
            repo_root=ROOT,
            rank1_run_dir=ROOT / "runs" / "reranking_complete_20260803_094159",
        )
        failures = evaluate_resource_snapshot(
            live, prefix=f"candidate_projection_{split}_live"
        )
        if failures:
            raise RuntimeError(
                f"D1 candidate projection live resource check failed: {failures}"
            )
        split_inputs = closure.get("canonical_inputs", {}).get(split)  # type: ignore[union-attr]
        if not isinstance(split_inputs, dict):
            raise RuntimeError(f"D1 source closure misses {split} inputs")
        candidate_paths = [
            verified_artifact_path(record, name=f"D1 {split} candidate source")
            for record in split_inputs.get("candidate_sources", [])
        ]
        score_paths = [
            verified_artifact_path(record, name=f"D1 {split} score source")
            for record in split_inputs.get("score_sources", [])
        ]
        paired_path = verified_artifact_path(
            split_inputs.get("paired_manifest", {}),
            name=f"D1 {split} paired source",
        )
        if not candidate_paths or len(candidate_paths) != len(score_paths):
            raise RuntimeError(f"incomplete D1 source shard inventory for {split}")
        assert_writable_prelock(run_dir)
        with ledger_stage(
            run_dir / "run_ledger.sqlite",
            stage="P2",
            substage=f"freeze_{split}_candidate_contract",
            route="D1",
            pool=args.pool,
            method=args.method,
            command=" ".join(map(str, sys.argv)),
        ) as state:
            manifest = build_canonical_candidate_split(
                run_dir=run_dir,
                split=split,
                candidate_paths=candidate_paths,
                score_paths=score_paths,
                paired_path=paired_path,
                resume=args.resume,
                source_contract={
                    "snapshot": "A",
                    "source_closure": artifact_record(closure_path),
                    "source_paths_from_closure_only": True,
                    "projection_code": {
                        "candidates": artifact_record(
                            ROOT / "src/d1_reranking/candidates.py"
                        ),
                        "contracts": artifact_record(
                            ROOT / "src/d1_reranking/contracts.py"
                        ),
                        "io": artifact_record(ROOT / "src/d1_reranking/io.py"),
                        "tool": artifact_record(Path(__file__)),
                    },
                    "candidate_membership": "frozen post-NMS; no regeneration or re-NMS",
                    "candidate_test_labels_read": False,
                },
                execution_contract={
                    "resource_gate": artifact_record(gate_path),
                    "resource_policy": gate.get("policy"),
                    "live_resource_recheck": live,
                    "exclusive_heavy_resource_lease": str(
                        (run_dir.parent / ".d1_heavy_resource.lock").resolve()
                    ),
                    "candidate_test_labels_read": False,
                },
            )
            output_dir = (
                run_dir / "02_candidates"
                if split == "test"
                else run_dir / "02_candidates" / split
            )
            path = output_dir / (
                "test_manifest.json" if split == "test" else "manifest.json"
            )
            state["artifact_path"] = str(path)
            state["artifact_sha256"] = sha256_file(path)
            print(
                json.dumps(
                    {"split": split, "summaries": manifest["summaries"]}, sort_keys=True
                )
            )
    return 0


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    assert_writable_prelock(run_dir)
    with exclusive_heavy_resource_lease(
        run_dir, purpose="D1 P2 canonical candidate projection"
    ):
        return _run_under_lease(args, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
