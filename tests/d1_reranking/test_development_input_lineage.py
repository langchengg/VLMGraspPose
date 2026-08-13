from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from d1_reranking.candidates import artifact_record
from tools.d1_reranking.run_r0_r1 import _load_split as load_rule_split
from tools.d1_reranking.train_primary_cell import _load_split as load_primary_split
from unified_reranking.hashing import atomic_json, canonical_sha256


def _write_manifest(path: Path, value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(value)
    atomic_json(path, value)


def _build_split(root: Path, split: str = "train") -> Path:
    rows = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["c1", "c2"],
            "native_rank": [1, 2],
            "native_score": [0.8, 0.2],
            "native_score_raw": [0.8, 0.2],
            "candidate_identity_sha256": ["i1", "i2"],
            "candidate_geometry_sha256": ["g1", "g2"],
            "p_center": [0.7, 0.3],
        }
    )
    candidate_dir = root / "02_candidates" / split
    candidate_dir.mkdir(parents=True)
    candidates = candidate_dir / "d1_top5_candidates.parquet"
    hashes = candidate_dir / "candidate_hashes.parquet"
    rows.to_parquet(candidates, index=False)
    pd.DataFrame({"sample_id": ["s0"], "candidate_count": [2]}).to_parquet(
        hashes, index=False
    )
    candidate_manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "configuration": {"route": "D1", "split": split},
        "artifacts": {
            "top5": artifact_record(candidates),
            "candidate_hashes": artifact_record(hashes),
        },
    }
    candidate_manifest_path = candidate_dir / "manifest.json"
    _write_manifest(candidate_manifest_path, candidate_manifest)

    raw_dir = root / "03_features" / split / "top5" / "matched_common_raw"
    raw_dir.mkdir(parents=True)
    raw_features = raw_dir / "candidate_features.parquet"
    rows.drop(columns="native_score").to_parquet(raw_features, index=False)
    raw_manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "route": "D1",
        "split": split,
        "pool": "top5",
        "track": "matched_common_raw",
        "candidate_test_labels_read": False,
        "sources": {
            "extractor": {
                "candidate_manifest": artifact_record(candidate_manifest_path),
                "candidate_hashes": artifact_record(hashes),
                "canonical_candidates": artifact_record(candidates),
            },
        },
        "artifacts": {"candidate_features": artifact_record(raw_features)},
    }
    raw_manifest_path = raw_dir / "manifest.json"
    _write_manifest(raw_manifest_path, raw_manifest)

    calibration_dir = root / "05_calibration" / "top5"
    calibration_dir.mkdir(parents=True)
    calibration_manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "selected_method": "platt",
    }
    calibration_manifest_path = calibration_dir / "calibration_manifest.json"
    _write_manifest(calibration_manifest_path, calibration_manifest)

    final_dir = root / "03_features" / split / "top5" / "T2_matched_common"
    final_dir.mkdir(parents=True)
    final_features = final_dir / "candidate_features.parquet"
    final = rows.drop(columns="native_score").copy()
    final["calibrated_native_probability"] = [0.75, 0.25]
    final["base_logit"] = [1.1, -1.1]
    final.to_parquet(final_features, index=False)
    model_columns = [
        "native_rank",
        "native_score_raw",
        "p_center",
        "calibrated_native_probability",
        "base_logit",
    ]
    final_manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "configuration": {
            "route": "D1",
            "split": split,
            "pool": "top5",
            "track": "T2_matched_common",
        },
        "model_feature_columns": model_columns,
        "model_feature_schema_sha256": canonical_sha256(model_columns),
        "feature_extraction_latency_ms": 1.0,
        "sources": {
            "candidate_manifest": artifact_record(candidate_manifest_path),
            "candidate_hashes": artifact_record(hashes),
            "candidates": artifact_record(candidates),
            "raw_common_manifest": artifact_record(raw_manifest_path),
            "raw_common_features": artifact_record(raw_features),
            "calibration_manifest": artifact_record(calibration_manifest_path),
        },
        "artifacts": {"candidate_features": artifact_record(final_features)},
    }
    final_manifest_path = final_dir / "manifest.json"
    _write_manifest(final_manifest_path, final_manifest)

    label_dir = root / "03_features" / split / "top5" / "labels"
    label_dir.mkdir(parents=True)
    labels = label_dir / "candidate_labels.parquet"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["c1", "c2"],
            "native_rank": [1, 2],
            "candidate_success": [1, 0],
            "jacquard_margin": [0.4, -0.2],
        }
    ).to_parquet(labels, index=False)
    label_manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "split": split,
        "pool": "top5",
        "candidate_test_labels_read": False,
        "sources": {
            "candidate_manifest": artifact_record(candidate_manifest_path),
            "candidate_hashes": artifact_record(hashes),
            "candidates": artifact_record(candidates),
        },
        "artifact": artifact_record(labels),
    }
    _write_manifest(label_dir / "manifest.json", label_manifest)

    denominator = root / "01_manifests" / f"d1_paired_{split}.parquet"
    denominator.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": ["s0"]}).to_parquet(denominator, index=False)
    return final_manifest_path


def test_primary_and_rule_loaders_accept_exact_development_lineage(
    tmp_path: Path,
) -> None:
    _build_split(tmp_path)
    primary, primary_columns, _primary_sources = load_primary_split(tmp_path, "train")
    rules, denominator, _rule_sources = load_rule_split(tmp_path, "train")
    assert len(primary) == len(rules) == 2
    assert "base_logit" in primary_columns
    assert denominator == ["s0"]


@pytest.mark.parametrize("loader", [load_primary_split, load_rule_split])
def test_development_loaders_reject_rehashed_stale_candidate_binding(
    tmp_path: Path, loader: object
) -> None:
    manifest_path = _build_split(tmp_path)
    stale = tmp_path / "stale_candidates.parquet"
    stale.write_bytes(b"old candidate contract")
    value = pd.read_json(manifest_path, typ="series").to_dict()
    value["sources"]["candidates"] = artifact_record(stale)
    value.pop("content_sha256")
    _write_manifest(manifest_path, value)
    with pytest.raises(RuntimeError, match="final T2"):
        loader(tmp_path, "train")  # type: ignore[operator]
