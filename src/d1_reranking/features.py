"""Leakage-safe D1 feature projection and route-rich track assembly."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.feature_tracks import select_numeric_model_columns
from unified_reranking.hashing import atomic_json, canonical_sha256
from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.telemetry import (
    flatten_telemetry,
    missing_feature_rate,
    telemetry_payload,
)

from .candidates import artifact_record, verify_canonical_candidate_frame
from .io import atomic_csv, atomic_parquet


JOIN_COLUMNS = ("sample_id", "candidate_id")
STRUCTURAL_COLUMNS = (
    "sample_id",
    "candidate_id",
    "route",
    "native_rank",
    "native_score_raw",
    "cx_px",
    "cy_px",
    "center_depth_m",
    "theta_deg",
    "width_px",
    "height_px",
)

# The legacy feature table used a contact-derived width convention in some
# migrated artifacts. The canonical configured width comes from the candidate
# contract, so the legacy field is excluded rather than silently substituted.
LEGACY_EXCLUDED_FEATURES = frozenset({"q_raw", "width_px"})


def load_feature_allowlist(path: str | Path) -> tuple[str, ...]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    features = value.get("features")
    if value.get("ground_truth_allowed") is not False or not isinstance(features, list):
        raise ValueError("D1 inference feature allowlist is not explicitly GT-free")
    names = tuple(
        str(item) for item in features if str(item) not in LEGACY_EXCLUDED_FEATURES
    )
    if int(value.get("count", -1)) != len(features):
        raise ValueError("D1 inference feature allowlist count mismatch")
    return assert_model_feature_columns(names)


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, frame[list(JOIN_COLUMNS)].astype(str).to_numpy()))


def project_native_available_features(
    candidates: pd.DataFrame,
    source_features: pd.DataFrame,
    allowlist: Sequence[str],
    *,
    split: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build T1 from exact candidate keys and a fixed GT-free allowlist."""

    verify_canonical_candidate_frame(candidates, split=split)
    required_source = {*JOIN_COLUMNS, "original_gqcnn_rank", "q_raw", *allowlist}
    missing = sorted(required_source.difference(source_features.columns))
    if missing:
        raise ValueError(f"D1 native feature source misses columns: {missing}")
    if (
        source_features[list(JOIN_COLUMNS)].isna().any().any()
        or source_features.duplicated(list(JOIN_COLUMNS)).any()
    ):
        raise ValueError("D1 native feature source has invalid candidate keys")
    expected = _key_set(candidates)
    source_keys = _key_set(source_features)
    if not expected.issubset(source_keys):
        raise ValueError(
            f"D1 native features miss canonical candidates: {len(expected - source_keys)}"
        )
    selected = source_features.loc[
        source_features["sample_id"]
        .astype(str)
        .isin(candidates["sample_id"].astype(str)),
        [*JOIN_COLUMNS, "original_gqcnn_rank", "q_raw", *allowlist],
    ].copy()
    selected_keys = pd.MultiIndex.from_frame(selected[list(JOIN_COLUMNS)].astype(str))
    expected_index = pd.MultiIndex.from_tuples(
        sorted(expected), names=list(JOIN_COLUMNS)
    )
    selected = selected.loc[selected_keys.isin(expected_index)].copy()
    if _key_set(selected) != expected or len(selected) != len(candidates):
        raise ValueError("D1 native feature projection has non-exact membership")
    canonical = candidates.loc[
        :,
        [
            *JOIN_COLUMNS,
            "route",
            "native_rank",
            "native_score",
            "cx_px",
            "cy_px",
            "center_depth_m",
            "theta_deg",
            "width_px",
            "height_px",
        ],
    ].rename(columns={"native_score": "native_score_raw"})
    result = canonical.merge(
        selected, on=list(JOIN_COLUMNS), how="left", validate="one_to_one"
    )
    if not np.array_equal(
        result["native_rank"].to_numpy(int),
        result["original_gqcnn_rank"].to_numpy(int),
    ):
        raise ValueError("D1 native feature rank differs from frozen candidate rank")
    if not np.array_equal(
        result["native_score_raw"].to_numpy(np.float64),
        result["q_raw"].to_numpy(np.float64),
    ):
        raise ValueError("D1 native feature q differs from frozen candidate q")
    result = result.drop(columns=["original_gqcnn_rank", "q_raw"])
    ordered = [*STRUCTURAL_COLUMNS, *allowlist]
    result = (
        result.loc[:, ordered]
        .sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    model_columns = select_numeric_model_columns(result)
    if not set(allowlist).issubset(model_columns):
        missing_model = sorted(set(allowlist).difference(model_columns))
        raise ValueError(
            f"D1 allowlisted features are not numeric model columns: {missing_model}"
        )
    return result, model_columns


def assemble_route_rich_features(
    matched_common: pd.DataFrame,
    native_available: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build T3 as T2 plus prefixed D1-only evidence without supervision."""

    for name, frame in (("T2", matched_common), ("T1", native_available)):
        if (
            frame[list(JOIN_COLUMNS)].isna().any().any()
            or frame.duplicated(list(JOIN_COLUMNS)).any()
        ):
            raise ValueError(f"D1 {name} features have invalid candidate keys")
    if _key_set(matched_common) != _key_set(native_available):
        raise ValueError("D1 T1/T2 feature membership mismatch")
    t1_model = select_numeric_model_columns(native_available)
    t2_model = select_numeric_model_columns(matched_common)
    route_only = [column for column in t1_model if column not in t2_model]
    supplement = native_available.loc[:, [*JOIN_COLUMNS, *route_only]].rename(
        columns={column: f"d1_route_{column}" for column in route_only}
    )
    result = matched_common.merge(
        supplement, on=list(JOIN_COLUMNS), how="left", validate="one_to_one"
    )
    model_columns = select_numeric_model_columns(result)
    return result, model_columns


def write_native_feature_track(
    *,
    run_dir: str | Path,
    split: str,
    pool: str,
    candidates_path: str | Path,
    candidate_manifest_path: str | Path,
    candidate_hashes_path: str | Path,
    source_features_path: str | Path,
    allowlist_path: str | Path,
    source_closure_path: str | Path,
    projection_tool_path: str | Path,
    resume: bool,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    output_dir = root / "03_features" / split / pool / "T1_native_available"
    manifest_path = output_dir / "manifest.json"
    configuration = {
        "schema_version": 1,
        "split": split,
        "pool": pool,
        "track": "T1_native_available",
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "candidates": artifact_record(candidates_path),
        "candidate_hashes": artifact_record(candidate_hashes_path),
        "source_features": artifact_record(source_features_path),
        "allowlist": artifact_record(allowlist_path),
        "source_closure": artifact_record(source_closure_path),
        "projection_primitive": artifact_record(Path(__file__)),
        "projection_tool": artifact_record(projection_tool_path),
        "legacy_excluded_features": sorted(LEGACY_EXCLUDED_FEATURES),
        "candidate_test_labels_read": False,
    }
    source_signature = canonical_sha256(configuration)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        unsigned = dict(existing)
        expected_content = unsigned.pop("content_sha256", None)
        if (
            resume
            and existing.get("status") == "COMPLETE"
            and existing.get("source_signature_sha256") == source_signature
            and expected_content == canonical_sha256(unsigned)
        ):
            verify_artifact_records_recursive(
                {
                    "configuration": existing.get("configuration"),
                    "artifact": existing.get("artifact"),
                    "feature_schema": existing.get("feature_schema"),
                    "missingness_report": existing.get("missingness_report"),
                },
                name=f"D1 {split}/{pool} T1 resume",
                require_at_least_one=True,
            )
            return existing
        raise RuntimeError(
            "D1 T1 feature manifest exists with a different or corrupt contract"
        )
    started = time.perf_counter()
    candidates = pd.read_parquet(candidates_path)
    allowlist = load_feature_allowlist(allowlist_path)
    source = pd.read_parquet(
        source_features_path,
        columns=[*JOIN_COLUMNS, "original_gqcnn_rank", "q_raw", *allowlist],
    )
    frame, model_columns = project_native_available_features(
        candidates, source, allowlist, split=split
    )
    feature_extraction_latency_ms = (
        (time.perf_counter() - started) * 1000.0 / len(frame)
    )
    artifact_path = atomic_parquet(frame, output_dir / "candidate_features.parquet")
    missingness = pd.DataFrame(
        {
            "column": model_columns,
            "missing_fraction": [
                float(
                    1.0
                    - np.isfinite(pd.to_numeric(frame[column], errors="coerce")).mean()
                )
                for column in model_columns
            ],
        }
    )
    missing_path = atomic_csv(missingness, output_dir / "missingness_report.csv")
    schema = {
        "schema_version": 1,
        "track": "T1_native_available",
        "model_columns": list(model_columns),
        "model_schema_sha256": canonical_sha256(list(model_columns)),
        "forbidden_ground_truth": True,
        "candidate_test_labels_read": False,
    }
    schema_path = output_dir / "feature_schema.json"
    atomic_json(schema_path, schema)
    telemetry = telemetry_payload(
        phase=f"d1_T1_{split}_{pool}_projection",
        parameter_count=None,
        ranker_latency_ms=None,
        feature_latency_ms=feature_extraction_latency_ms,
        missing_feature_rate_value=missing_feature_rate(frame, model_columns),
        not_applicable_fields=("parameter_count", "ranker_latency_ms"),
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_signature_sha256": source_signature,
        "configuration": configuration,
        "artifact": {**artifact_record(artifact_path), "rows": len(frame)},
        "feature_schema": artifact_record(schema_path),
        "missingness_report": artifact_record(missing_path),
        "feature_extraction_latency_ms": feature_extraction_latency_ms,
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "candidate_test_labels_read": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    return manifest
