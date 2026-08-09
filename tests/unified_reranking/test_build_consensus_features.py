import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from tools.unified_reranking.build_consensus_features import run
from unified_reranking.hashing import canonical_sha256, sha256_file


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    for route in ("crog", "g1", "c1"):
        candidates = pd.DataFrame(
            {
                "sample_id": ["s"],
                "candidate_id": [f"{route}-0"],
                "route": [route.upper()],
                "native_rank": [1],
                "native_score": [0.8],
            }
        )
        candidate_path = run_dir / "02_candidates" / f"{route}_validation_top5.parquet"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(candidate_path, index=False)
        calibration_path = run_dir / "05_calibration" / f"{route}_validation.parquet"
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        candidates[["sample_id", "candidate_id"]].assign(
            calibrated_native_probability=0.7, base_logit=0.8
        ).to_parquet(calibration_path, index=False)
        _write_json(
            run_dir / "05_calibration" / f"{route}_calibration_manifest.json",
            {
                "status": "COMPLETE",
                "artifacts": {
                    "validation": {
                        "path": str(calibration_path),
                        "sha256": sha256_file(calibration_path),
                    }
                },
            },
        )

        common_dir = run_dir / "03_features" / "common" / f"{route}_validation"
        common_path = common_dir / "candidate_features.parquet"
        common_dir.mkdir(parents=True, exist_ok=True)
        candidates[["sample_id", "candidate_id"]].assign(common_signal=1.0).to_parquet(
            common_path, index=False
        )
        _write_json(
            common_dir / "feature_manifest.json",
            {
                "status": "COMPLETE",
                "artifacts": {
                    "candidate_features": {
                        "path": str(common_path),
                        "sha256": sha256_file(common_path),
                    }
                },
            },
        )

        rgb_dir = run_dir / "03_features" / "rgb" / f"{route}_validation"
        rgb_path = rgb_dir / "candidate_features.parquet"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        candidates[["sample_id", "candidate_id"]].assign(rgb_missing=0.0).to_parquet(
            rgb_path, index=False
        )
        _write_json(
            rgb_dir / "feature_manifest.json",
            {
                "status": "COMPLETE",
                "artifact": {"path": str(rgb_path), "sha256": sha256_file(rgb_path)},
            },
        )

        dense_dir = run_dir / "03_features" / "tri_backend_dense" / "validation" / route
        dense_path = dense_dir / "candidate_features.parquet"
        dense_dir.mkdir(parents=True, exist_ok=True)
        dense = candidates[["sample_id", "candidate_id"]].assign(
            dense_crog_support=0.5,
            dense_g1_support=0.6,
            dense_c1_support=0.7,
        )
        dense.to_parquet(dense_path, index=False)
        dense_columns = ["dense_crog_support", "dense_g1_support", "dense_c1_support"]
        _write_json(
            dense_dir / "feature_manifest.json",
            {
                "status": "COMPLETE",
                "route": route,
                "split": "validation",
                "dense_sampling_verified": True,
                "original_model_original_roundtrip_verified": True,
                "model_feature_schema_sha256": canonical_sha256(dense_columns),
                "artifact": {"path": str(dense_path), "sha256": sha256_file(dense_path)},
            },
        )
    return run_dir


def test_t3_builder_requires_and_merges_all_dense_backend_families(tmp_path: Path):
    run_dir = _fixture(tmp_path)
    result = run(argparse.Namespace(run_dir=run_dir, split="validation"))
    assert result["dense_sampling_verified"] is True
    for route in ("crog", "g1", "c1"):
        track = pd.read_parquet(
            run_dir
            / "03_features"
            / "tracks"
            / "T3_tri_backend"
            / f"{route}_validation"
            / "candidate_features.parquet"
        )
        assert {"dense_crog_support", "dense_g1_support", "dense_c1_support"}.issubset(
            track.columns
        )


def test_t3_builder_rejects_tampered_dense_artifact(tmp_path: Path):
    run_dir = _fixture(tmp_path)
    path = (
        run_dir
        / "03_features"
        / "tri_backend_dense"
        / "validation"
        / "g1"
        / "candidate_features.parquet"
    )
    frame = pd.read_parquet(path)
    frame["dense_g1_support"] = 9.0
    frame.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        run(argparse.Namespace(run_dir=run_dir, split="validation"))
