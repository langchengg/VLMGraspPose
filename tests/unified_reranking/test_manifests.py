from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.unified_reranking.manifests import _load_safe_samples


def test_safe_manifest_rejects_duplicate_sample_identity(tmp_path: Path) -> None:
    row = {
        "schema_version": 1,
        "sample_id": "q1",
        "sample_index": 0,
        "scene_id": "scene",
        "expression_index": 0,
        "question_index": 0,
        "source_rgb_path": "/rgb",
        "source_rgb_sha256": "a",
        "source_depth_path": "/depth",
        "source_depth_sha256": "b",
        "rgbd_pair_sha256": "c",
        "language": "pick",
        "language_sha256": "d",
        "predicted_mask_path": "/mask",
        "predicted_mask_sha256": "e",
        "predicted_probability_path": "/prob",
        "predicted_probability_sha256": "f",
        "intrinsics_path": None,
        "intrinsics_sha256": None,
        "intrinsics_status": "missing",
        "intrinsics_provenance": "{}",
        "split": "train",
        "checkpoint_sha256": "g",
        "config_sha256": "h",
        "inference_contract_sha256": "i",
    }
    path = tmp_path / "samples.parquet"
    pd.DataFrame([row, row]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        _load_safe_samples(path)
