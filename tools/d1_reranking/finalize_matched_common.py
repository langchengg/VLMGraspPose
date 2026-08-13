"""Finalize calibrated D1 T2 tracks and isolated training compatibility paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.candidates import artifact_record  # noqa: E402
from d1_reranking.io import atomic_csv, atomic_hardlink, atomic_parquet  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from d1_reranking.tracks import (  # noqa: E402
    finalize_matched_common,
    missingness_rows,
)
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
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    parser.add_argument("--pool", default="top5", choices=("top5", "top10", "allnms"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _verified_content_manifest(path: Path, *, name: str) -> dict[str, object]:
    value = load_verified_json(path, name=name)
    unsigned = dict(value)
    expected = unsigned.pop("content_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash mismatch")
    return value


def _same_artifact_record(
    observed: object, expected: dict[str, object], *, name: str
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


def _validate_resumed_track(
    manifest: dict[str, object], *, candidate_path: Path
) -> None:
    if manifest.get("candidate_test_labels_read") is not False:
        raise RuntimeError("D1 T2 resume violates Test isolation")
    columns = tuple(map(str, manifest.get("model_feature_columns", ())))
    if not columns or manifest.get("model_feature_schema_sha256") != canonical_sha256(
        list(columns)
    ):
        raise RuntimeError("D1 T2 resume model schema hash differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("D1 T2 resume artifact inventory is absent")
    feature_path = verified_artifact_path(
        artifacts.get("candidate_features", {}), name="D1 T2 resumed features"
    )
    schema_path = verified_artifact_path(
        artifacts.get("feature_schema", {}), name="D1 T2 resumed feature schema"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if (
        schema.get("route") != "D1"
        or schema.get("track") != "T2_matched_common"
        or tuple(map(str, schema.get("model_columns", ()))) != columns
        or schema.get("model_schema_sha256") != canonical_sha256(list(columns))
        or schema.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 T2 resumed feature schema differs")
    feature_keys = pd.read_parquet(feature_path, columns=["sample_id", "candidate_id"])
    candidate_keys = pd.read_parquet(
        candidate_path, columns=["sample_id", "candidate_id"]
    )
    if (
        feature_keys.duplicated(["sample_id", "candidate_id"]).any()
        or candidate_keys.duplicated(["sample_id", "candidate_id"]).any()
        or len(feature_keys) != len(candidate_keys)
        or set(map(tuple, feature_keys.astype(str).to_numpy()))
        != set(map(tuple, candidate_keys.astype(str).to_numpy()))
    ):
        raise RuntimeError("D1 T2 resumed candidate membership differs")


def _candidate_source(
    root: Path, split: str, pool: str
) -> tuple[Path, Path, dict[str, object]]:
    base = root / "02_candidates" if split == "test" else root / "02_candidates" / split
    manifest_path = base / (
        "test_manifest.json" if split == "test" else "manifest.json"
    )
    manifest = _verified_content_manifest(
        manifest_path, name=f"D1 {split} candidate manifest"
    )
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict) or (
        configuration.get("route") != "D1"
        or configuration.get("split") != split
        or manifest.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 {split} candidate manifest semantics differ")
    verify_artifact_records_recursive(
        manifest.get("artifacts"),
        name=f"D1 {split} candidate artifacts",
        require_at_least_one=True,
    )
    record = manifest.get("artifacts", {}).get(pool)  # type: ignore[union-attr]
    if not isinstance(record, dict):
        raise ValueError(f"D1 {split}/{pool} candidate artifact is absent")
    path = verified_artifact_path(record, name=f"D1 {split}/{pool} candidates")
    return path, manifest_path, manifest


def _calibration_source(
    root: Path,
    split: str,
    pool: str,
    *,
    candidate_manifest_record: dict[str, object],
    candidates_record: dict[str, object],
) -> tuple[Path, Path, dict[str, object], Path, dict[str, object]]:
    manifest_path = root / "05_calibration" / pool / "calibration_manifest.json"
    manifest = _verified_content_manifest(
        manifest_path, name=f"D1 {pool} calibration manifest"
    )
    if (
        manifest.get("configuration", {}).get("route") != "D1"  # type: ignore[union-attr]
        or manifest.get("configuration", {}).get("pool") != pool  # type: ignore[union-attr]
        or manifest.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 {pool} calibration manifest semantics differ")
    verify_artifact_records_recursive(
        {"sources": manifest.get("sources"), "artifacts": manifest.get("artifacts")},
        name=f"D1 {pool} calibration",
        require_at_least_one=True,
    )
    if split == "test":
        application_path = (
            root / "05_calibration" / pool / "test_application_manifest.json"
        )
        application = _verified_content_manifest(
            application_path, name=f"D1 {pool} Test calibration application"
        )
        expected_configuration = {
            "route": "D1",
            "split": "test",
            "pool": pool,
            "selected_method": manifest.get("selected_method"),
        }
        application_configuration = application.get("configuration")
        if not isinstance(application_configuration, dict) or any(
            application_configuration.get(key) != value
            for key, value in expected_configuration.items()
        ):
            raise RuntimeError("D1 Test calibration application semantics differ")
        application_sources = application.get("sources")
        if not isinstance(application_sources, dict) or (
            application_sources.get("calibration_manifest")
            != artifact_record(manifest_path)
            or application_sources.get("candidate_manifest")
            != candidate_manifest_record
            or application_sources.get("candidates") != candidates_record
        ):
            raise RuntimeError(
                "D1 Test calibration application candidate/calibrator binding differs"
            )
        verify_artifact_records_recursive(
            {"sources": application_sources, "artifacts": application.get("artifacts")},
            name=f"D1 {pool} Test calibration application",
            require_at_least_one=True,
        )
        record = application.get("artifacts", {}).get("test_predictions")  # type: ignore[union-attr]
        if not isinstance(record, dict):
            raise ValueError("D1 Test calibration application is incomplete")
        path = verified_artifact_path(
            record, name=f"D1 test/{pool} calibration predictions"
        )
        return path, application_path, application, manifest_path, manifest
    artifact_name = "train_oof" if split == "train" else "validation"
    split_sources = manifest.get("sources", {}).get(split)  # type: ignore[union-attr]
    if not isinstance(split_sources, dict) or (
        split_sources.get("candidate_manifest") != candidate_manifest_record
        or split_sources.get("candidates") != candidates_record
    ):
        raise RuntimeError(f"D1 {split}/{pool} calibration candidate binding differs")
    record = manifest.get("artifacts", {}).get(artifact_name)  # type: ignore[union-attr]
    if not isinstance(record, dict):
        raise ValueError(f"D1 calibration misses {artifact_name}")
    path = verified_artifact_path(
        record, name=f"D1 {split}/{pool} calibration predictions"
    )
    return path, manifest_path, manifest, manifest_path, manifest


def run(run_dir: Path, *, split: str, pool: str, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    candidate_path, candidate_manifest_path, candidate_manifest = _candidate_source(
        root, split, pool
    )
    candidate_manifest_record = artifact_record(candidate_manifest_path)
    candidates_record = artifact_record(candidate_path)
    candidate_hashes_record = candidate_manifest.get("artifacts", {}).get(  # type: ignore[union-attr]
        "candidate_hashes"
    )
    if not isinstance(candidate_hashes_record, dict):
        raise ValueError(f"D1 {split} candidate-hashes artifact is absent")
    candidate_hashes_path = verified_artifact_path(
        candidate_hashes_record, name=f"D1 {split} candidate hashes"
    )
    raw_dir = root / "03_features" / split / pool / "matched_common_raw"
    raw_manifest_path = raw_dir / "manifest.json"
    raw_manifest = _verified_content_manifest(
        raw_manifest_path, name=f"D1 {split}/{pool} raw common manifest"
    )
    if (
        raw_manifest.get("candidate_test_labels_read") is not False
        or raw_manifest.get("route") != "D1"
        or raw_manifest.get("split") != split
        or raw_manifest.get("pool") != pool
        or raw_manifest.get("track") != "matched_common_raw"
    ):
        raise RuntimeError("D1 raw common manifest semantics differ")
    raw_sources = raw_manifest.get("sources")
    extractor_sources = (
        raw_sources.get("extractor") if isinstance(raw_sources, dict) else None
    )
    if not isinstance(extractor_sources, dict):
        raise RuntimeError("D1 raw common extractor sources are absent")
    _same_artifact_record(
        extractor_sources.get("candidate_manifest"),
        candidate_manifest_record,
        name="D1 raw common candidate manifest",
    )
    _same_artifact_record(
        extractor_sources.get("canonical_candidates"),
        candidates_record,
        name="D1 raw common canonical candidates",
    )
    verify_artifact_records_recursive(
        {
            "sources": raw_manifest.get("sources"),
            "artifacts": raw_manifest.get("artifacts"),
        },
        name=f"D1 {split}/{pool} raw common",
        require_at_least_one=True,
    )
    raw_record = raw_manifest.get("artifacts", {}).get("candidate_features")  # type: ignore[union-attr]
    relation_record = raw_manifest.get("artifacts", {}).get("candidate_relations")  # type: ignore[union-attr]
    context_record = raw_manifest.get("artifacts", {}).get("sample_context")  # type: ignore[union-attr]
    if not all(
        isinstance(record, dict)
        for record in (raw_record, relation_record, context_record)
    ):
        raise ValueError("D1 raw common artifacts are incomplete")
    raw_path = verified_artifact_path(raw_record, name="D1 raw common features")
    relations_path = verified_artifact_path(
        relation_record, name="D1 raw common relations"
    )
    context_path = verified_artifact_path(
        context_record, name="D1 raw common sample context"
    )
    (
        calibration_path,
        calibration_source_path,
        _calibration_source_manifest,
        calibration_manifest_path,
        _calibration_manifest,
    ) = _calibration_source(
        root,
        split,
        pool,
        candidate_manifest_record=candidate_manifest_record,
        candidates_record=candidates_record,
    )
    sources = {
        "candidate_manifest": candidate_manifest_record,
        "candidates": candidates_record,
        "candidate_hashes": artifact_record(candidate_hashes_path),
        "raw_common_manifest": artifact_record(raw_manifest_path),
        "raw_common_features": artifact_record(raw_path),
        "calibration_manifest": artifact_record(calibration_manifest_path),
        "calibration_source_manifest": artifact_record(calibration_source_path),
        "calibration_predictions": artifact_record(calibration_path),
        "assembler": artifact_record(ROOT / "src/d1_reranking/tracks.py"),
        "tool": artifact_record(Path(__file__)),
    }
    feature_extraction_latency_ms = float(
        raw_manifest.get("feature_extraction_latency_ms", -1)
    )
    if feature_extraction_latency_ms < 0:
        raise RuntimeError("D1 raw common extraction latency is absent")
    configuration = {
        "schema_version": 1,
        "route": "D1",
        "split": split,
        "pool": pool,
        "track": "T2_matched_common",
        "candidate_test_labels_read": False,
        "calibration_columns": ["calibrated_native_probability", "base_logit"],
        "training_use": (
            "R0/R1/reporting and label-free inference; R2-R6 outer OOF cells must "
            "refit the selected calibration family on their own fit partition"
        ),
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = root / "03_features" / split / pool / "T2_matched_common"
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = _verified_content_manifest(
            manifest_path, name=f"D1 {split}/{pool} T2 manifest"
        )
        if (
            resume
            and existing.get("source_signature_sha256") == signature
            and existing.get("configuration") == configuration
            and existing.get("sources") == sources
        ):
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                },
                name=f"D1 {split}/{pool} T2 resume",
                require_at_least_one=True,
            )
            _validate_resumed_track(existing, candidate_path=candidate_path)
            return existing
        raise RuntimeError("D1 T2 manifest exists with a different or corrupt contract")

    frame, model_columns = finalize_matched_common(
        pd.read_parquet(candidate_path),
        pd.read_parquet(raw_path),
        pd.read_parquet(calibration_path),
    )
    feature_path = atomic_parquet(frame, output / "candidate_features.parquet")
    relation_alias = atomic_hardlink(
        relations_path, output / "candidate_relations.parquet"
    )
    context_alias = atomic_hardlink(context_path, output / "sample_context.parquet")
    missing_path = atomic_csv(
        missingness_rows(frame, model_columns), output / "missingness_report.csv"
    )
    schema_path = output / "feature_schema.json"
    atomic_json(
        schema_path,
        {
            "schema_version": 1,
            "route": "D1",
            "track": "T2_matched_common",
            "model_columns": list(model_columns),
            "model_schema_sha256": canonical_sha256(list(model_columns)),
            "candidate_test_labels_read": False,
        },
    )
    artifacts = {
        "candidate_features": artifact_record(feature_path),
        "candidate_relations": artifact_record(relation_alias),
        "sample_context": artifact_record(context_alias),
        "feature_schema": artifact_record(schema_path),
        "missingness_report": artifact_record(missing_path),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": configuration,
        "source_signature_sha256": signature,
        "candidate_test_labels_read": False,
        "model_feature_columns": list(model_columns),
        "model_feature_schema_sha256": canonical_sha256(list(model_columns)),
        "feature_extraction_latency_ms": feature_extraction_latency_ms,
        "sources": sources,
        "artifacts": artifacts,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    manifest_path = (
        root
        / "03_features"
        / args.split
        / args.pool
        / "T2_matched_common"
        / "manifest.json"
    )
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P5",
        substage=f"T2_finalize_{args.split}_{args.pool}",
        route="D1",
        pool=args.pool,
        evidence_track="T2_matched_common",
        method="calibrated_common_assembly",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, split=args.split, pool=args.pool, resume=args.resume)
        state["artifact_path"] = str(manifest_path)
        state["artifact_sha256"] = sha256_file(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
