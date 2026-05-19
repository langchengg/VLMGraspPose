from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from target.object_language_mapping import ObjectLanguageMapper
from utils.data_types import GraspNetSample


def _sample(tmp_path: Path) -> GraspNetSample:
    return GraspNetSample(
        split="test_seen",
        scene_id="scene_0100",
        camera="realsense",
        frame_id="0000",
        rgb_path=tmp_path / "rgb.png",
        depth_path=tmp_path / "depth.png",
        annotation_path=None,
        camera_intrinsic_path=None,
        output_dir=tmp_path / "outputs" / "test_seen" / "scene_0100" / "realsense" / "0000",
        label_path=tmp_path / "label.png",
    )


def test_mapper_generates_one_entry_per_visible_target(tmp_path: Path) -> None:
    label = np.zeros((20, 30), dtype=np.uint8)
    label[2:8, 2:10] = 1
    label[10:18, 15:25] = 2
    mapper = ObjectLanguageMapper(output_root=tmp_path / "outputs")

    entries = mapper.entries_for_sample(_sample(tmp_path), label, all_targets=True)

    assert [entry.target_id for entry in entries] == [0, 1]
    assert [entry.command for entry in entries] == ["pick object_000", "pick object_001"]
    assert entries[0].bbox == [2, 2, 9, 7]
    assert entries[1].mask_path.endswith("target_001.png")


def test_mapper_disambiguates_duplicate_category_labels(tmp_path: Path) -> None:
    label = np.zeros((20, 40), dtype=np.uint8)
    label[4:12, 1:8] = 1
    label[4:12, 31:38] = 2
    mapper = ObjectLanguageMapper(
        output_root=tmp_path / "outputs",
        category_labels={0: "mug", 1: "mug"},
    )

    entries = mapper.entries_for_sample(_sample(tmp_path), label, all_targets=True)

    commands = [entry.command for entry in entries]
    assert commands == ["pick the left mug", "pick the right mug"]
    assert all(entry.disambiguation_type == "position" for entry in entries)


def test_mapper_uses_category_label_when_available(tmp_path: Path) -> None:
    label = np.zeros((20, 30), dtype=np.uint8)
    label[3:12, 4:16] = 17
    mapper = ObjectLanguageMapper(output_root=tmp_path / "outputs", category_labels={16: "mug"})

    entries = mapper.entries_for_sample(_sample(tmp_path), label, all_targets=True)

    assert len(entries) == 1
    assert entries[0].target_id == 16
    assert entries[0].target_label == "mug"
    assert entries[0].command == "pick the mug"
    assert entries[0].disambiguation_type == "category"


def test_mapper_one_target_per_frame_chooses_largest_visible_object(tmp_path: Path) -> None:
    label = np.zeros((20, 30), dtype=np.uint8)
    label[1:4, 1:4] = 1
    label[6:18, 8:25] = 2
    mapper = ObjectLanguageMapper(output_root=tmp_path / "outputs")

    entries = mapper.entries_for_sample(_sample(tmp_path), label, all_targets=False)

    assert len(entries) == 1
    assert entries[0].target_id == 1
    assert entries[0].command == "pick object_001"
