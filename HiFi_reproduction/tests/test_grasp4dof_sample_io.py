from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.grasping.common.sample_io import CompactSampleLoader


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deployment(tmp_path: Path, *, probability_format: str) -> dict[str, object]:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    mask = tmp_path / "mask.png"
    probability = tmp_path / f"probability.{probability_format}"
    Image.fromarray(np.full((8, 10, 3), 127, dtype=np.uint8)).save(rgb)
    Image.fromarray(np.full((8, 10), 1000, dtype=np.uint16)).save(depth)
    Image.fromarray(np.ones((8, 10), dtype=np.uint8) * 255).save(mask)
    value = np.full((4, 5), 0.75, dtype=np.float32)
    if probability_format == "npz":
        np.savez_compressed(probability, probability=value)
    else:
        np.save(probability, value)
    language = "grasp the cup"
    return {
        "sample_id": "sample",
        "scene_id": "scene",
        "language": language,
        "language_sha256": hashlib.sha256(language.encode()).hexdigest(),
        "source_rgb_path": str(rgb),
        "source_rgb_sha256": _sha(rgb),
        "source_depth_path": str(depth),
        "source_depth_sha256": _sha(depth),
        "predicted_mask_path": str(mask),
        "predicted_mask_sha256": _sha(mask),
        "predicted_probability_path": str(probability),
        "predicted_probability_sha256": _sha(probability),
        "intrinsics_path": None,
        "intrinsics_provenance": "{}",
    }


@pytest.mark.parametrize("probability_format", ["npy", "npz"])
def test_loader_supports_both_audited_probability_formats(
    tmp_path: Path, probability_format: str
) -> None:
    arrays = CompactSampleLoader().load(
        _deployment(tmp_path, probability_format=probability_format)
    )
    assert arrays.rgb.shape == (8, 10, 3)
    assert arrays.depth_m.shape == (8, 10)
    assert arrays.probability.shape == (4, 5)


def test_loader_rejects_source_file_drift(tmp_path: Path) -> None:
    row = _deployment(tmp_path, probability_format="npz")
    Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(row["predicted_mask_path"])
    with pytest.raises(ValueError, match="predicted mask SHA-256 mismatch"):
        CompactSampleLoader().load(row)


def test_oracle_loader_verifies_separate_gt_mask(tmp_path: Path) -> None:
    row = _deployment(tmp_path, probability_format="npz")
    gt = tmp_path / "gt.png"
    Image.fromarray(np.ones((4, 5), dtype=np.uint8) * 255).save(gt)
    labels = {
        "sample_id": "sample",
        "prepared_gt_mask_path": str(gt),
        "prepared_gt_mask_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="prepared GT oracle mask SHA-256 mismatch"):
        CompactSampleLoader().load(row, mask_source="gt_mask_oracle", labels=labels)
