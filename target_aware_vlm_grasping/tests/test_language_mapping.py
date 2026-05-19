from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from target.language_mapping import ObjectLanguageMapper
from utils.data_types import DatasetSample


def test_language_mapper_uses_dataset_sentence() -> None:
    sample = DatasetSample(
        dataset_name="OCID-VLG",
        sample_id="s0",
        rgb_path=Path("rgb.png"),
        depth_path=Path("depth.png"),
        sentence="Grasp the red cup",
        target_label="cup",
        target_id=2,
        target_bbox_gt=[1, 2, 3, 4],
    )
    entry = ObjectLanguageMapper().entry_from_sample(sample)
    assert entry.command == "Grasp the red cup"
    assert entry.target_label == "cup"
    assert entry.disambiguation_type == "dataset_sentence"


def test_language_mapper_disambiguates_duplicate_labels() -> None:
    mask = np.zeros((12, 30), dtype=np.uint8)
    mask[2:8, 1:6] = 1
    mask[2:8, 23:28] = 2
    sample = DatasetSample(
        dataset_name="OCID-Grasp",
        sample_id="s1",
        rgb_path=Path("rgb.png"),
        depth_path=Path("depth.png"),
        sentence="",
        target_label="cup",
    )
    entries = ObjectLanguageMapper().entries_from_instances(sample, mask, labels={1: "cup", 2: "cup"})
    assert [entry.command for entry in entries] == ["pick the left cup", "pick the right cup"]
