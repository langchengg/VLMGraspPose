from __future__ import annotations

import pandas as pd
import pytest

from d1_reranking.candidates import artifact_record
from d1_reranking.tracks import finalize_matched_common
from tools.d1_reranking.finalize_matched_common import (
    _calibration_source,
    _validate_resumed_track,
)
from unified_reranking.hashing import atomic_json, canonical_sha256


def _content(path, value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(value)
    atomic_json(path, value)


def test_finalize_matched_common_adds_only_label_free_calibration() -> None:
    candidates = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "route": ["D1", "D1"],
        }
    )
    common = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "route": ["D1", "D1"],
            "native_rank": [1, 2],
            "p_center": [0.8, 0.4],
        }
    )
    calibration = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "calibrated_native_probability": [0.7, 0.3],
            "base_logit": [0.847, -0.847],
            "candidate_success": [1, 0],
        }
    )
    frame, columns = finalize_matched_common(candidates, common, calibration)
    assert {
        "native_rank",
        "p_center",
        "calibrated_native_probability",
        "base_logit",
    }.issubset(columns)
    assert "candidate_success" not in frame


def test_finalize_matched_common_requires_exact_calibration_membership() -> None:
    candidates = pd.DataFrame(
        {"sample_id": ["s0"], "candidate_id": ["a"], "route": ["D1"]}
    )
    common = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "candidate_id": ["a"],
            "route": ["D1"],
            "native_rank": [1],
        }
    )
    calibration = pd.DataFrame(
        {
            "sample_id": ["other"],
            "candidate_id": ["a"],
            "calibrated_native_probability": [0.5],
            "base_logit": [0.0],
        }
    )
    with pytest.raises(ValueError, match="candidate membership differs"):
        finalize_matched_common(candidates, common, calibration)


def test_finalize_matched_common_rejects_extra_or_rank_drift() -> None:
    candidates = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "candidate_id": ["a"],
            "route": ["D1"],
            "native_rank": [1],
            "native_score": [0.9],
        }
    )
    common = candidates.assign(p_center=[0.8])
    calibration = candidates.assign(
        calibrated_native_probability=[0.7], base_logit=[0.847]
    )
    extra = pd.concat(
        [
            calibration,
            calibration.assign(candidate_id="b", native_rank=2, native_score=0.8),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="candidate membership differs"):
        finalize_matched_common(candidates, common, extra)
    with pytest.raises(ValueError, match="native_rank differs"):
        finalize_matched_common(candidates, common.assign(native_rank=2), calibration)


def test_t2_calibration_manifest_must_bind_current_candidate_contract(
    tmp_path,
) -> None:
    candidate_manifest = tmp_path / "candidate_manifest.json"
    candidates = tmp_path / "candidates.parquet"
    predictions = tmp_path / "calibration.parquet"
    for path, payload in (
        (candidate_manifest, b"manifest"),
        (candidates, b"candidates"),
        (predictions, b"calibration"),
    ):
        path.write_bytes(payload)
    full_path = tmp_path / "05_calibration" / "top5" / "calibration_manifest.json"
    full: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": {"route": "D1", "pool": "top5"},
        "candidate_test_labels_read": False,
        "selected_method": "platt",
        "sources": {
            "train": {
                "candidate_manifest": artifact_record(candidate_manifest),
                "candidates": artifact_record(candidates),
            }
        },
        "artifacts": {"train_oof": artifact_record(predictions)},
    }
    _content(full_path, full)
    observed, *_rest = _calibration_source(
        tmp_path,
        "train",
        "top5",
        candidate_manifest_record=artifact_record(candidate_manifest),
        candidates_record=artifact_record(candidates),
    )
    assert observed == predictions.resolve()

    other = tmp_path / "other_candidates.parquet"
    other.write_bytes(b"other")
    full["sources"]["train"]["candidates"] = artifact_record(other)  # type: ignore[index]
    full.pop("content_sha256")
    _content(full_path, full)
    with pytest.raises(RuntimeError, match="candidate binding differs"):
        _calibration_source(
            tmp_path,
            "train",
            "top5",
            candidate_manifest_record=artifact_record(candidate_manifest),
            candidates_record=artifact_record(candidates),
        )


def test_t2_resume_rederives_schema_and_membership(tmp_path) -> None:
    candidates = tmp_path / "candidates.parquet"
    features = tmp_path / "features.parquet"
    schema = tmp_path / "schema.json"
    pd.DataFrame({"sample_id": ["s0"], "candidate_id": ["a"]}).to_parquet(
        candidates, index=False
    )
    pd.DataFrame(
        {"sample_id": ["s0"], "candidate_id": ["a"], "native_rank": [1.0]}
    ).to_parquet(features, index=False)
    columns = ("native_rank",)
    atomic_json(
        schema,
        {
            "schema_version": 1,
            "route": "D1",
            "track": "T2_matched_common",
            "model_columns": list(columns),
            "model_schema_sha256": canonical_sha256(list(columns)),
            "candidate_test_labels_read": False,
        },
    )
    manifest = {
        "candidate_test_labels_read": False,
        "model_feature_columns": list(columns),
        "model_feature_schema_sha256": canonical_sha256(list(columns)),
        "artifacts": {
            "candidate_features": artifact_record(features),
            "feature_schema": artifact_record(schema),
        },
    }
    _validate_resumed_track(manifest, candidate_path=candidates)
    manifest["model_feature_schema_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="model schema hash differs"):
        _validate_resumed_track(manifest, candidate_path=candidates)
