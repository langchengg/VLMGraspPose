"""Project the frozen GT-free D1 feature allowlist into T1 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.features import write_native_feature_track  # noqa: E402
from d1_reranking.candidates import artifact_record  # noqa: E402
from d1_reranking.execution import load_content_manifest  # noqa: E402
from d1_reranking.provenance import load_source_closure  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.hashing import sha256_file  # noqa: E402
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    verified_artifact_path,
    verify_artifact_records_recursive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", default="features")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--pool", choices=("allnms", "top10", "top5"), required=True)
    parser.add_argument("--evidence-track", default="T1_native_available")
    parser.add_argument("--method", default="native_projection")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def _candidate_path(run_dir: Path, split: str, pool: str) -> Path:
    base = (
        run_dir / "02_candidates"
        if split == "test"
        else run_dir / "02_candidates" / split
    )
    return base / f"d1_{pool}_candidates.parquet"


def _candidate_contract(
    run_dir: Path, split: str, pool: str, closure_path: Path
) -> tuple[Path, Path, Path]:
    base = (
        run_dir / "02_candidates"
        if split == "test"
        else run_dir / "02_candidates" / split
    )
    manifest_path = base / (
        "test_manifest.json" if split == "test" else "manifest.json"
    )
    manifest = load_content_manifest(
        manifest_path, name=f"D1 {split} candidate manifest", statuses=("COMPLETE",)
    )
    configuration = manifest.get("configuration")
    closure_record = (
        configuration.get("source_contract", {}).get("source_closure")
        if isinstance(configuration, dict)
        else None
    )
    expected_closure = artifact_record(closure_path)
    closure_matches = isinstance(closure_record, dict) and all(
        closure_record.get(key) == expected_closure.get(key)
        for key in ("path", "sha256")
    )
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
    candidate_path = verified_artifact_path(
        manifest.get("artifacts", {}).get(pool, {}),  # type: ignore[union-attr]
        name=f"D1 {split}/{pool} candidates",
    )
    if candidate_path != _candidate_path(run_dir, split, pool).resolve():
        raise RuntimeError(f"D1 {split}/{pool} candidate path differs")
    hashes_path = verified_artifact_path(
        manifest.get("artifacts", {}).get("candidate_hashes", {}),  # type: ignore[union-attr]
        name=f"D1 {split} candidate hashes",
    )
    return candidate_path, manifest_path, hashes_path


def main() -> int:
    args = parse_args()
    run = args.run_dir.expanduser().resolve()
    assert_writable_prelock(run)
    closure_path, closure = load_source_closure(run)
    split_inputs = closure.get("canonical_inputs", {}).get(args.split)  # type: ignore[union-attr]
    if not isinstance(split_inputs, dict):
        raise RuntimeError(f"D1 source closure misses {args.split} feature inputs")
    source_features = verified_artifact_path(
        split_inputs.get("native_features", {}),
        name=f"D1 {args.split} native feature source",
    )
    allowlist = verified_artifact_path(
        split_inputs.get("feature_allowlist", {}),
        name=f"D1 {args.split} feature allowlist",
    )
    candidates, candidate_manifest, candidate_hashes = _candidate_contract(
        run, args.split, args.pool, closure_path
    )
    output_manifest = (
        run
        / "03_features"
        / args.split
        / args.pool
        / "T1_native_available"
        / "manifest.json"
    )
    assert_writable_prelock(run)
    with ledger_stage(
        run / "run_ledger.sqlite",
        stage="P5",
        substage=f"T1_{args.split}_{args.pool}",
        route="D1",
        pool=args.pool,
        evidence_track="T1_native_available",
        method="native_projection",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        write_native_feature_track(
            run_dir=run,
            split=args.split,
            pool=args.pool,
            candidates_path=candidates,
            candidate_manifest_path=candidate_manifest,
            candidate_hashes_path=candidate_hashes,
            source_features_path=source_features,
            allowlist_path=allowlist,
            resume=args.resume,
            source_closure_path=closure_path,
            projection_tool_path=Path(__file__),
        )
        state["artifact_path"] = str(output_manifest)
        state["artifact_sha256"] = sha256_file(output_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
