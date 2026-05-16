from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.split_manager import SplitManager


def test_split_ranges_are_exact() -> None:
    manager = SplitManager()
    assert manager.get_scene_ids("train")[0] == "scene_0000"
    assert manager.get_scene_ids("train")[-1] == "scene_0089"
    assert manager.get_scene_ids("val") == [f"scene_{idx:04d}" for idx in range(90, 100)]
    assert manager.get_scene_ids("test_seen")[0] == "scene_0100"
    assert manager.get_scene_ids("test_novel")[-1] == "scene_0189"


def test_scene_to_split() -> None:
    manager = SplitManager()
    assert manager.scene_to_split("scene_0000") == "train"
    assert manager.scene_to_split("scene_0095") == "val"
    assert manager.scene_to_split("scene_0140") == "test_similar"
    assert manager.scene_to_split("scene_0189") == "test_novel"


def test_invalid_split_and_scene_raise_clear_errors() -> None:
    manager = SplitManager()
    with pytest.raises(ValueError):
        manager.get_scene_ids("bad")
    with pytest.raises(ValueError):
        manager.scene_to_split("scene_9999")
