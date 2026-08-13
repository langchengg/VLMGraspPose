"""Run the verified common extractor for raw D1 matched-common evidence.

This is a thin D1 path/manifest adapter. The numerical implementation remains
``tools.unified_reranking.extract_common_features.run`` so D1 uses exactly the
same coordinate, missingness and relation definitions as CROG/G1/C1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import (  # noqa: E402
    FEATURE_RESOURCE_SCOPE,
    artifact_record,
)
from d1_reranking.feature_execution import validate_worker_context  # noqa: E402
from d1_reranking.feature_plan import (  # noqa: E402
    load_active_feature_extraction_plan,
    validate_feature_extraction_result,
)
from d1_reranking.io import atomic_csv, atomic_hardlink  # noqa: E402
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
from unified_reranking.test_access_guard import append_access_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", default="features")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--pool", choices=("allnms", "top10", "top5"), required=True)
    parser.add_argument("--evidence-track", default="matched_common_raw")
    parser.add_argument("--method", default="unified_common_extractor")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="formal")
    return parser.parse_args()


def _actual_candidate(run: Path, split: str, pool: str) -> Path:
    base = run / "02_candidates" if split == "test" else run / "02_candidates" / split
    return base / f"d1_{pool}_candidates.parquet"


def _verified_candidate(
    run: Path, split: str, pool: str
) -> tuple[Path, Path, Path, dict[str, object]]:
    base = run / "02_candidates" if split == "test" else run / "02_candidates" / split
    manifest_path = base / (
        "test_manifest.json" if split == "test" else "manifest.json"
    )
    manifest = load_verified_json(manifest_path, name=f"D1 {split} candidate manifest")
    unsigned = dict(manifest)
    observed_content = unsigned.pop("content_sha256", None)
    if observed_content != canonical_sha256(unsigned):
        raise RuntimeError(f"D1 {split} candidate manifest content hash mismatch")
    verify_artifact_records_recursive(
        manifest.get("artifacts"),
        name=f"D1 {split} candidate artifacts",
        require_at_least_one=True,
    )
    record = manifest.get("artifacts", {}).get(pool)  # type: ignore[union-attr]
    source = verified_artifact_path(
        record if isinstance(record, dict) else {},
        name=f"D1 {split}/{pool} canonical candidates",
    )
    expected = _actual_candidate(run, split, pool).resolve()
    if source != expected:
        raise RuntimeError(f"D1 {split}/{pool} candidate artifact path differs")
    hashes = verified_artifact_path(
        manifest.get("artifacts", {}).get("candidate_hashes", {}),  # type: ignore[union-attr]
        name=f"D1 {split} candidate hashes",
    )
    return source, manifest_path, hashes, manifest


def _unified_pool_name(pool: str) -> str:
    return "all" if pool == "allnms" else pool


def _record_test_access(
    run_dir: Path, *, job_id: str, pool: str, manifest_path: Path
) -> None:
    assert_writable_prelock(run_dir)
    append_access_log(
        run_dir,
        {
            "event": "prelock_label_free_test_stage",
            "event_id": canonical_sha256(
                {"stage": "d1_raw_feature_extraction", "job_id": job_id}
            )[:24],
            "stage": "d1_raw_feature_extraction",
            "job_id": job_id,
            "pool": pool,
            "output_manifest": str(manifest_path.resolve()),
            "output_manifest_sha256": sha256_file(manifest_path),
            "candidate_labels_opened_as_table": False,
            "candidate_test_labels_read": False,
        },
    )


def _verified_common_manifest(
    path: Path,
    *,
    split: str,
    pool: str,
    compatibility: Path,
    paired_alias: Path,
) -> dict[str, object]:
    value = load_verified_json(path, name=f"D1 {split}/{pool} unified common")
    if (
        value.get("route") != "d1"
        or value.get("split") != split
        or value.get("pool") != _unified_pool_name(pool)
        or value.get("tag") != "formal"
        or value.get("candidate_manifest_sha256") != sha256_file(compatibility)
        or value.get("paired_manifest_sha256") != sha256_file(paired_alias)
    ):
        raise RuntimeError(f"D1 {split}/{pool} unified common semantics differ")
    verify_artifact_records_recursive(
        value.get("artifacts"),
        name=f"D1 {split}/{pool} unified common artifacts",
        require_at_least_one=True,
    )
    return value


def run(
    args: argparse.Namespace, *, resource_lease_path: Path | None = None
) -> dict[str, object]:
    run_dir = args.run_dir.expanduser().resolve()
    assert_writable_prelock(run_dir)
    expected_lease = (run_dir.parent / ".d1_heavy_resource.lock").resolve()
    if resource_lease_path is None or resource_lease_path.resolve() != expected_lease:
        raise RuntimeError("D1 raw-feature worker requires the repository-wide lease")
    plan_path, plan = load_active_feature_extraction_plan(run_dir)
    matching = [
        job
        for job in plan.get("jobs", ())
        if isinstance(job, dict) and job.get("job_id") == args.job_id
    ]
    if len(matching) != 1:
        raise RuntimeError("D1 raw-feature job is outside the frozen nine-job plan")
    job = matching[0]
    planned = job["configuration"]
    if (
        args.stage != "features"
        or args.evidence_track != "matched_common_raw"
        or args.method != "unified_common_extractor"
        or args.seed != -1
        or args.config is not None
        or args.split != planned["split"]
        or args.pool != planned["pool"]
        or args.chunk_size != planned["chunk_size"]
        or args.tag != planned["tag"]
        or args.limit is not planned["limit"]
        or canonical_sha256(planned)[:16] != args.job_id
    ):
        raise ValueError(
            "D1 raw-feature CLI differs from the frozen planned configuration"
        )
    (
        execution_path,
        execution,
        event_path,
        _event,
        claim_path,
        _claim,
    ) = validate_worker_context(
        run_dir,
        plan_path=plan_path,
        plan=plan,
        job=job,
    )
    paired_source_name = (
        "d1_paired_manifest.parquet"
        if args.split == "test"
        else f"d1_paired_{args.split}.parquet"
    )
    closure_path, closure = load_source_closure(run_dir)
    split_inputs = closure.get("canonical_inputs", {}).get(args.split)  # type: ignore[union-attr]
    if not isinstance(split_inputs, dict):
        raise RuntimeError(f"D1 source closure misses {args.split} paired input")
    paired_source = verified_artifact_path(
        split_inputs.get("paired_manifest", {}),
        name=f"D1 {args.split} paired source",
    )
    paired_copy = run_dir / "01_manifests" / paired_source_name
    if sha256_file(paired_copy) != sha256_file(paired_source):
        raise RuntimeError(f"D1 {args.split} paired copy differs from source closure")
    paired_alias = run_dir / "01_manifests" / f"paired_{args.split}.parquet"
    source, candidate_manifest, candidate_hashes, candidate_contract = (
        _verified_candidate(run_dir, args.split, args.pool)
    )
    source_contract = candidate_contract.get("configuration", {}).get(  # type: ignore[union-attr]
        "source_contract"
    )
    bound_closure = (
        verified_artifact_path(
            source_contract.get("source_closure", {}),
            name="D1 candidate source closure",
        )
        if isinstance(source_contract, dict)
        else None
    )
    if bound_closure != closure_path:
        raise RuntimeError("D1 candidate/common source-closure binding differs")
    if source.is_symlink():
        raise ValueError(f"canonical D1 candidate pool must not be a symlink: {source}")
    unified_pool = _unified_pool_name(args.pool)
    compatibility = (
        run_dir / "02_candidates" / f"d1_{args.split}_{unified_pool}.parquet"
    )
    pool_suffix = "" if unified_pool == "top5" else f"_{unified_pool}"
    output_name = f"d1_{args.split}{pool_suffix}"
    common_dir = run_dir / "03_features" / "common" / output_name
    common_manifest = common_dir / "feature_manifest.json"
    track_dir = run_dir / "03_features" / args.split / args.pool / "matched_common_raw"
    track_manifest = track_dir / "manifest.json"
    extractor_configuration = {
        "schema_version": 1,
        "route": "D1",
        "split": args.split,
        "pool": args.pool,
        "unified_pool": unified_pool,
        "track": "matched_common_raw",
        "chunk_size": int(args.chunk_size),
        "tag": "formal",
        "limit": None,
        "candidate_test_labels_read": False,
    }

    def current_sources() -> dict[str, object]:
        _verified_common_manifest(
            common_manifest,
            split=args.split,
            pool=args.pool,
            compatibility=compatibility,
            paired_alias=paired_alias,
        )
        return {
            "source_closure": artifact_record(closure_path),
            "candidate_manifest": artifact_record(candidate_manifest),
            "candidate_hashes": artifact_record(candidate_hashes),
            "canonical_candidates": artifact_record(source),
            "paired_source": artifact_record(paired_source),
            "paired_copy": artifact_record(paired_copy),
            "paired_manifest": artifact_record(paired_alias),
            "compatibility_candidates": artifact_record(compatibility),
            "unified_common_manifest": artifact_record(common_manifest),
            "shared_numerical_extractor": artifact_record(
                ROOT / "src/unified_reranking/feature_extractors/common.py"
            ),
            "unified_tool": artifact_record(
                ROOT / "tools/unified_reranking/extract_common_features.py"
            ),
            "wrapper_tool": artifact_record(Path(__file__)),
        }

    if track_manifest.is_file():
        if not args.resume:
            raise FileExistsError(
                f"D1 raw common track exists; pass --resume: {track_manifest}"
            )
        existing = load_verified_json(
            track_manifest, name=f"D1 {args.split}/{args.pool} raw common result"
        )
        validate_feature_extraction_result(
            existing,
            plan_path=plan_path,
            job=job,
            manifest_path=track_manifest,
        )
        if args.split == "test":
            _record_test_access(
                run_dir,
                job_id=args.job_id,
                pool=args.pool,
                manifest_path=track_manifest,
            )
        return existing

    assert_writable_prelock(run_dir)
    atomic_hardlink(paired_copy, paired_alias)
    atomic_hardlink(source, compatibility)

    # Import only the label-free extractor after the D1 contract is verified.
    # This module has no evaluator or candidate-label dependency.
    from tools.unified_reranking.extract_common_features import run as run_common

    common_args = argparse.Namespace(
        run_dir=run_dir,
        route="d1",
        split=args.split,
        pool=unified_pool,
        chunk_size=args.chunk_size,
        limit=args.limit,
        tag=args.tag,
        suppress_access_log=True,
    )
    assert_writable_prelock(run_dir)
    run_common(common_args)
    assert_writable_prelock(run_dir)
    result = _verified_common_manifest(
        common_manifest,
        split=args.split,
        pool=args.pool,
        compatibility=compatibility,
        paired_alias=paired_alias,
    )
    sources = {
        "plan": artifact_record(plan_path),
        "planned_inputs": job["inputs"],
        "execution_manifest": artifact_record(execution_path),
        "execution_event": artifact_record(event_path),
        "execution_claim": artifact_record(claim_path),
        "extractor": current_sources(),
    }
    signature = canonical_sha256(sources)
    artifacts = {}
    assert_writable_prelock(run_dir)
    for kind in ("candidate_features", "candidate_relations", "sample_context"):
        linked = atomic_hardlink(
            common_dir / f"{kind}.parquet", track_dir / f"{kind}.parquet"
        )
        artifacts[kind] = artifact_record(linked)
    feature_columns = tuple(str(item) for item in result["model_feature_columns"])
    schema_path = track_dir / "feature_schema.json"
    atomic_json(
        schema_path,
        {
            "schema_version": 1,
            "track": "matched_common_raw",
            "route": "D1",
            "model_columns": list(feature_columns),
            "model_schema_sha256": canonical_sha256(list(feature_columns)),
            "shared_numerical_extractor": artifact_record(
                ROOT / "src/unified_reranking/feature_extractors/common.py"
            ),
            "candidate_test_labels_read": False,
        },
    )
    feature_frame = pd.read_parquet(
        track_dir / "candidate_features.parquet", columns=list(feature_columns)
    )
    missingness = pd.DataFrame(
        {
            "column": feature_columns,
            "missing_fraction": [
                float(
                    1.0
                    - np.isfinite(
                        pd.to_numeric(feature_frame[column], errors="coerce")
                    ).mean()
                )
                for column in feature_columns
            ],
        }
    )
    missing_path = atomic_csv(missingness, track_dir / "missingness_report.csv")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "job_id": args.job_id,
        "route": "D1",
        "split": args.split,
        "pool": args.pool,
        "track": "matched_common_raw",
        "configuration": planned,
        "configuration_sha256": canonical_sha256(planned),
        "extractor_configuration": extractor_configuration,
        "model_feature_columns": list(feature_columns),
        "model_feature_schema_sha256": canonical_sha256(list(feature_columns)),
        "source_signature_sha256": signature,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": args.split == "test",
        "sources": sources,
        "artifacts": {
            **artifacts,
            "feature_schema": artifact_record(schema_path),
            "missingness_report": artifact_record(missing_path),
        },
        "telemetry": result.get("telemetry"),
        "feature_extraction_latency_ms": result.get("feature_extraction_latency_ms"),
        "execution_provenance": {
            "execution_id": execution["execution_id"],
            "execution_manifest": artifact_record(execution_path),
            "execution_event": artifact_record(event_path),
            "execution_claim": artifact_record(claim_path),
            "resource_gate": execution["resource_gate"],
            "resource_lease_path": str(expected_lease),
            "scope": FEATURE_RESOURCE_SCOPE,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    assert_writable_prelock(run_dir)
    atomic_json(track_manifest, manifest)
    validate_feature_extraction_result(
        manifest,
        plan_path=plan_path,
        job=job,
        manifest_path=track_manifest,
    )
    if args.split == "test":
        _record_test_access(
            run_dir,
            job_id=args.job_id,
            pool=args.pool,
            manifest_path=track_manifest,
        )
    return manifest


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    track_manifest = (
        run_dir
        / "03_features"
        / args.split
        / args.pool
        / "matched_common_raw"
        / "manifest.json"
    )
    assert_writable_prelock(run_dir)
    lease_path = (run_dir.parent / ".d1_heavy_resource.lock").resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P5",
        substage=f"matched_common_raw_{args.split}_{args.pool}_{args.tag}",
        route="D1",
        pool=args.pool,
        evidence_track="matched_common_raw",
        method="unified_common_extractor",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(args, resource_lease_path=lease_path)
        state["artifact_path"] = str(track_manifest)
        state["artifact_sha256"] = sha256_file(track_manifest)
        print(
            json.dumps(
                {
                    "job_id": result["job_id"],
                    "manifest": str(track_manifest),
                    "sha256": state["artifact_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
