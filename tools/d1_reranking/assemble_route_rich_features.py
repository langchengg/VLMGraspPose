"""Assemble D1 T3 route-rich evidence as the exact T2+T1 feature superset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.features import assemble_route_rich_features  # noqa: E402
from d1_reranking.io import atomic_csv, atomic_parquet  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from d1_reranking.tracks import missingness_rows  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.telemetry import (  # noqa: E402
    flatten_telemetry,
    missing_feature_rate,
    telemetry_payload,
)
from unified_reranking.test_access_guard import append_access_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    parser.add_argument("--pool", required=True, choices=("top5", "top10", "allnms"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _candidate_manifest(root: Path, split: str) -> Path:
    return (
        root / "02_candidates" / "test_manifest.json"
        if split == "test"
        else root / "02_candidates" / split / "manifest.json"
    )


def _same_artifact_record(
    observed: object, expected: dict[str, str], *, name: str
) -> None:
    if not isinstance(observed, dict) or any(
        observed.get(key) != expected.get(key) for key in ("path", "sha256")
    ):
        raise RuntimeError(f"{name} does not bind the current artifact")
    if (
        "bytes" in observed
        and "bytes" in expected
        and observed.get("bytes") != expected.get("bytes")
    ):
        raise RuntimeError(f"{name} byte count differs")


def run(run_dir: Path, *, split: str, pool: str, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    t1_path = (
        root / "03_features" / split / pool / "T1_native_available" / "manifest.json"
    )
    t2_path = (
        root / "03_features" / split / pool / "T2_matched_common" / "manifest.json"
    )
    candidates_path = _candidate_manifest(root, split)
    t1 = load_content_manifest(
        t1_path, name=f"D1 {split}/{pool} T1", statuses=("COMPLETE",)
    )
    t2 = load_content_manifest(
        t2_path, name=f"D1 {split}/{pool} T2", statuses=("COMPLETE",)
    )
    candidates_manifest = load_content_manifest(
        candidates_path, name=f"D1 {split} candidates", statuses=("COMPLETE",)
    )
    for name, value in (("T1", t1), ("T2", t2), ("candidates", candidates_manifest)):
        if value.get("candidate_test_labels_read") is not False:
            raise RuntimeError(
                f"D1 {split}/{pool} {name} violates Test-label isolation"
            )
        verify_artifact_records_recursive(
            value,
            name=f"D1 {split}/{pool} {name}",
            require_at_least_one=True,
        )
    t1_table = verified_artifact_path(t1.get("artifact", {}), name="D1 T1 features")
    t2_table = verified_artifact_path(
        t2.get("artifacts", {}).get("candidate_features", {}),  # type: ignore[union-attr]
        name="D1 T2 features",
    )
    candidate_table = verified_artifact_path(
        candidates_manifest.get("artifacts", {}).get(pool, {}),  # type: ignore[union-attr]
        name=f"D1 {split}/{pool} candidates",
    )
    candidate_manifest_record = artifact_record(candidates_path)
    candidates_record = artifact_record(candidate_table)
    candidate_hashes = verified_artifact_path(
        candidates_manifest.get("artifacts", {}).get("candidate_hashes", {}),  # type: ignore[union-attr]
        name=f"D1 {split} candidate hashes",
    )
    candidate_hashes_record = artifact_record(candidate_hashes)
    candidate_configuration = candidates_manifest.get("configuration")
    t1_configuration = t1.get("configuration")
    t2_configuration = t2.get("configuration")
    t2_sources = t2.get("sources")
    if not isinstance(candidate_configuration, dict) or (
        candidate_configuration.get("route") != "D1"
        or candidate_configuration.get("split") != split
    ):
        raise RuntimeError(f"D1 {split} candidate manifest semantics differ")
    if not isinstance(t1_configuration, dict) or (
        t1_configuration.get("split") != split
        or t1_configuration.get("pool") != pool
        or t1_configuration.get("track") != "T1_native_available"
    ):
        raise RuntimeError(f"D1 {split}/{pool} T1 candidate binding differs")
    _same_artifact_record(
        t1_configuration.get("candidate_manifest"),
        candidate_manifest_record,
        name=f"D1 {split}/{pool} T1 candidate manifest",
    )
    _same_artifact_record(
        t1_configuration.get("candidates"),
        candidates_record,
        name=f"D1 {split}/{pool} T1 candidates",
    )
    _same_artifact_record(
        t1_configuration.get("candidate_hashes"),
        candidate_hashes_record,
        name=f"D1 {split}/{pool} T1 candidate hashes",
    )
    if not isinstance(t2_configuration, dict) or (
        t2_configuration.get("route") != "D1"
        or t2_configuration.get("split") != split
        or t2_configuration.get("pool") != pool
        or t2_configuration.get("track") != "T2_matched_common"
        or not isinstance(t2_sources, dict)
    ):
        raise RuntimeError(f"D1 {split}/{pool} T2 candidate binding differs")
    _same_artifact_record(
        t2_sources.get("candidate_manifest"),
        candidate_manifest_record,
        name=f"D1 {split}/{pool} T2 candidate manifest",
    )
    _same_artifact_record(
        t2_sources.get("candidates"),
        candidates_record,
        name=f"D1 {split}/{pool} T2 candidates",
    )
    _same_artifact_record(
        t2_sources.get("candidate_hashes"),
        candidate_hashes_record,
        name=f"D1 {split}/{pool} T2 candidate hashes",
    )
    sources = {
        "T1_manifest": artifact_record(t1_path),
        "T1_features": artifact_record(t1_table),
        "T2_manifest": artifact_record(t2_path),
        "T2_features": artifact_record(t2_table),
        "candidate_manifest": artifact_record(candidates_path),
        "candidate_hashes": candidate_hashes_record,
        "candidates": artifact_record(candidate_table),
        "assembler": artifact_record(ROOT / "src/d1_reranking/features.py"),
        "tool": artifact_record(Path(__file__)),
    }
    configuration = {
        "schema_version": 1,
        "route": "D1",
        "split": split,
        "pool": pool,
        "track": "T3_route_rich",
        "contract": "exact T2 matched-common plus prefixed T1-only D1 evidence",
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = root / "03_features" / split / pool / "T3_route_rich"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name=f"D1 {split}/{pool} T3", statuses=("COMPLETE",)
        )
        if (
            resume
            and existing.get("source_signature_sha256") == signature
            and existing.get("configuration") == configuration
            and existing.get("sources") == sources
        ):
            verify_artifact_records_recursive(
                existing,
                name=f"D1 {split}/{pool} T3 resume",
                require_at_least_one=True,
            )
            return existing
        raise RuntimeError("D1 T3 feature manifest differs or is corrupt")

    started = time.perf_counter()
    t1_frame = pd.read_parquet(t1_table)
    t2_frame = pd.read_parquet(t2_table)
    candidates = pd.read_parquet(candidate_table, columns=["sample_id", "candidate_id"])
    frame, model_columns = assemble_route_rich_features(t2_frame, t1_frame)
    expected = set(map(tuple, candidates.astype(str).to_numpy()))
    observed = set(
        map(tuple, frame[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    if (
        expected != observed
        or len(frame) != len(candidates)
        or frame.duplicated(["sample_id", "candidate_id"]).any()
    ):
        raise RuntimeError("D1 T3 does not exactly preserve candidate membership")
    assembly_latency_ms = (time.perf_counter() - started) * 1000.0 / len(frame)
    t1_latency = float(t1.get("feature_extraction_latency_ms", -1))
    t2_latency = float(t2.get("feature_extraction_latency_ms", -1))
    if min(t1_latency, t2_latency) < 0:
        raise RuntimeError("D1 T3 source extraction telemetry is incomplete")
    composite_latency_ms = t1_latency + t2_latency + assembly_latency_ms
    feature_table = atomic_parquet(frame, output / "candidate_features.parquet")
    schema_path = output / "feature_schema.json"
    atomic_json(
        schema_path,
        {
            "schema_version": 1,
            "route": "D1",
            "track": "T3_route_rich",
            "model_columns": list(model_columns),
            "model_schema_sha256": canonical_sha256(model_columns),
            "candidate_test_labels_read": False,
        },
    )
    missing_path = atomic_csv(
        missingness_rows(frame, model_columns), output / "missingness_report.csv"
    )
    telemetry = telemetry_payload(
        phase=f"d1_T3_{split}_{pool}_assembly",
        parameter_count=None,
        ranker_latency_ms=None,
        feature_latency_ms=assembly_latency_ms,
        missing_feature_rate_value=missing_feature_rate(frame, model_columns),
        not_applicable_fields=("parameter_count", "ranker_latency_ms"),
    )
    artifacts = {
        "candidate_features": artifact_record(feature_table),
        "feature_schema": artifact_record(schema_path),
        "missingness_report": artifact_record(missing_path),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": configuration,
        "source_signature_sha256": signature,
        "model_feature_columns": list(model_columns),
        "model_feature_schema_sha256": canonical_sha256(model_columns),
        "feature_extraction_latency_ms": composite_latency_ms,
        "feature_track_assembly_latency_ms": assembly_latency_ms,
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "candidate_test_labels_read": False,
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    if split == "test":
        append_access_log(
            root,
            {
                "event": "prelock_label_free_test_stage",
                "stage": "d1_route_rich_test_features",
                "route": "D1",
                "pool": pool,
                "output_manifest": str(manifest_path.resolve()),
                "output_manifest_sha256": sha256_file(manifest_path),
                "candidate_labels_opened_as_table": False,
            },
        )
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    path = (
        root
        / "03_features"
        / args.split
        / args.pool
        / "T3_route_rich"
        / "manifest.json"
    )
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P5",
        substage=f"T3_{args.split}_{args.pool}",
        route="D1",
        pool=args.pool,
        evidence_track="T3_route_rich",
        method="T2_plus_D1_native",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, split=args.split, pool=args.pool, resume=args.resume)
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
