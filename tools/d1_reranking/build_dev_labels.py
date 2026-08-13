"""Build physically separate Train/Validation D1 labels with the frozen evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.evaluator_adapter import build_candidate_labels  # noqa: E402
from d1_reranking.candidates import artifact_record  # noqa: E402
from d1_reranking.provenance import load_source_closure  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    load_verified_json,
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", default="labels")
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--pool", choices=("allnms", "top10", "top5"), required=True)
    parser.add_argument("--evidence-track", default="none")
    parser.add_argument("--method", default="canonical_evaluator")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def _candidate_contract(
    run_dir: Path, split: str, pool: str, closure_path: Path
) -> tuple[Path, Path, Path]:
    split_dir = run_dir / "02_candidates" / split
    manifest_path = split_dir / "manifest.json"
    manifest = load_verified_json(manifest_path, name=f"D1 {split} candidate manifest")
    unsigned = dict(manifest)
    observed_content = unsigned.pop("content_sha256", None)
    if observed_content != canonical_sha256(unsigned):
        raise RuntimeError(f"D1 {split} candidate manifest content hash mismatch")
    configuration = manifest.get("configuration")
    observed_closure = (
        configuration.get("source_contract", {}).get("source_closure")
        if isinstance(configuration, dict)
        else None
    )
    expected_closure = artifact_record(closure_path)
    closure_matches = isinstance(observed_closure, dict) and all(
        observed_closure.get(key) == expected_closure.get(key)
        for key in ("path", "sha256")
    )
    if (
        closure_matches
        and "bytes" in observed_closure
        and observed_closure.get("bytes") != expected_closure.get("bytes")
    ):
        closure_matches = False
    if not isinstance(configuration, dict) or (
        configuration.get("route") != "D1"
        or configuration.get("split") != split
        or not closure_matches
    ):
        raise RuntimeError(f"D1 {split} candidate/source-closure binding differs")
    verify_artifact_records_recursive(
        manifest.get("artifacts"),
        name=f"D1 {split} candidate artifacts",
        require_at_least_one=True,
    )
    candidates = verified_artifact_path(
        manifest.get("artifacts", {}).get(pool, {}),  # type: ignore[union-attr]
        name=f"D1 {split}/{pool} candidates",
    )
    expected = split_dir / f"d1_{pool}_candidates.parquet"
    if candidates != expected.resolve():
        raise RuntimeError(f"D1 {split}/{pool} candidate path differs")
    hashes = verified_artifact_path(
        manifest.get("artifacts", {}).get("candidate_hashes", {}),  # type: ignore[union-attr]
        name=f"D1 {split} candidate hashes",
    )
    return candidates, manifest_path, hashes


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    assert_writable_prelock(run_dir)
    closure_path, closure = load_source_closure(run_dir)
    split_inputs = closure.get("canonical_inputs", {}).get(args.split)  # type: ignore[union-attr]
    if not isinstance(split_inputs, dict):
        raise RuntimeError(f"D1 source closure misses {args.split} inputs")
    labels_source = verified_artifact_path(
        split_inputs.get("development_labels", {}),
        name=f"D1 {args.split} development labels",
    )
    candidates, candidate_manifest, candidate_hashes = _candidate_contract(
        run_dir, args.split, args.pool, closure_path
    )
    output_dir = run_dir / "03_features" / args.split / args.pool / "labels"
    destination = output_dir / "candidate_labels.parquet"
    manifest_path = output_dir / "manifest.json"
    evaluator = run_dir / "configs" / "canonical_evaluator.py"
    evaluator_records = [
        record
        for record in closure.get("verified_artifacts", [])
        if isinstance(record, dict) and record.get("name") == "canonical_evaluator"
    ]
    if len(evaluator_records) != 1:
        raise RuntimeError("D1 source closure canonical evaluator authority is absent")
    evaluator_authority = verified_artifact_path(
        evaluator_records[0], name="D1 canonical evaluator authority"
    )
    if sha256_file(evaluator) != sha256_file(evaluator_authority):
        raise RuntimeError("D1 run evaluator differs from the source closure authority")
    signature = canonical_sha256(
        {
            "split": args.split,
            "pool": args.pool,
            "source_closure": artifact_record(closure_path),
            "candidate_manifest": artifact_record(candidate_manifest),
            "candidate_hashes": artifact_record(candidate_hashes),
            "candidates": artifact_record(candidates),
            "sample_labels": artifact_record(labels_source),
            "evaluator_authority": artifact_record(evaluator_authority),
            "evaluator": artifact_record(evaluator),
            "tool": artifact_record(Path(__file__)),
        }
    )
    if manifest_path.is_file():
        existing = load_verified_json(
            manifest_path, name=f"D1 {args.split}/{args.pool} labels"
        )
        unsigned = dict(existing)
        observed_content = unsigned.pop("content_sha256", None)
        if observed_content != canonical_sha256(unsigned):
            raise RuntimeError("development label manifest content hash mismatch")
        verify_artifact_records_recursive(
            {
                "sources": existing.get("sources"),
                "artifact": existing.get("artifact"),
            },
            name=f"D1 {args.split}/{args.pool} labels resume",
            require_at_least_one=True,
        )
        if args.resume and existing.get("source_signature_sha256") == signature:
            return 0
        raise RuntimeError(
            "development label manifest exists with a different or corrupt contract"
        )
    assert_writable_prelock(run_dir)
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P4",
        substage=f"labels_{args.split}_{args.pool}",
        route="D1",
        pool=args.pool,
        method="canonical_evaluator",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        summary = build_candidate_labels(
            candidates,
            labels_source,
            destination,
            evaluator,
            sha256_file(evaluator),
            split=args.split,
        )
        manifest = {
            "schema_version": 1,
            "status": "COMPLETE",
            "split": args.split,
            "pool": args.pool,
            "candidate_test_labels_read": False,
            "source_signature_sha256": signature,
            "sources": {
                "source_closure": artifact_record(closure_path),
                "candidate_manifest": artifact_record(candidate_manifest),
                "candidate_hashes": artifact_record(candidate_hashes),
                "candidates": artifact_record(candidates),
                "sample_labels": artifact_record(labels_source),
                "evaluator_authority": artifact_record(evaluator_authority),
                "evaluator": artifact_record(evaluator),
                "tool": artifact_record(Path(__file__)),
            },
            "artifact": {
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
            },
            "summary": summary,
        }
        manifest["content_sha256"] = canonical_sha256(manifest)
        atomic_json(manifest_path, manifest)
        state["artifact_path"] = str(manifest_path)
        state["artifact_sha256"] = sha256_file(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
