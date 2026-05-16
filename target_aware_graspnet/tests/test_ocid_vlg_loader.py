from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.ocid_vlg_loader import OCIDGraspIndexBuilder, OCIDVLGIndexBuilder, OCIDVLGLoader


def test_ocid_vlg_index_uses_sentence_as_command_and_target_box() -> None:
    dataset_root = ROOT.parent / "data" / "raw" / "OCID-VLG"
    samples = OCIDVLGIndexBuilder(dataset_root, ROOT / "outputs" / "test_ocid").build(
        refer_split="multiple",
        split="test",
        max_samples=2,
    )

    assert samples
    sample = samples[0]
    assert sample.command == sample.sentence
    assert sample.target_label == "flashlight_1"
    assert sample.target_index == 2
    assert sample.target_bbox == [259, 381, 344, 408]
    assert sample.rgb_path.exists()
    assert sample.depth_path.exists()
    assert len(sample.grasp_rectangles) == 7


def test_ocid_vlg_loader_loads_depth_in_meters_and_instance_mask() -> None:
    dataset_root = ROOT.parent / "data" / "raw" / "OCID-VLG"
    sample = OCIDVLGIndexBuilder(dataset_root, ROOT / "outputs" / "test_ocid").build(
        refer_split="multiple",
        split="test",
        max_samples=1,
    )[0]

    loaded = OCIDVLGLoader(depth_scale=1000.0).load_sample(sample)

    assert loaded["rgb"].shape[:2] == (480, 640)
    assert loaded["depth"].dtype == np.float32
    assert 0.5 < float(loaded["depth"][loaded["depth"] > 0].mean()) < 2.0
    assert loaded["target"].command == sample.sentence
    assert loaded["target"].mask is not None
    assert loaded["target"].mask.sum() > 0
    assert loaded["target"].bbox == [259, 381, 344, 408]


def test_ocid_grasp_fallback_generates_language_from_class_name() -> None:
    dataset_root = ROOT.parent / "data" / "raw" / "OCID-VLG"
    samples = OCIDGraspIndexBuilder(dataset_root, ROOT / "outputs" / "test_ocid_grasp").build(
        max_samples=1,
    )

    assert samples
    sample = samples[0]
    assert sample.sentence == ""
    assert sample.command.startswith("pick the ")
    assert sample.target_label
    assert sample.target_index >= 1
    assert sample.rgb_path.exists()
    assert sample.grasp_rectangles
