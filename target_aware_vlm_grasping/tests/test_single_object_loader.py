from __future__ import annotations

import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.single_object_loader import SingleObjectIndexBuilder, SingleObjectRGBDLoader


def test_single_object_loader_builds_command_and_resizes_to_depth(tmp_path: Path) -> None:
    data = tmp_path / "data"
    obj = data / "001_chips_can"
    masks = obj / "masks"
    masks.mkdir(parents=True)

    rgb = np.zeros((1024, 1280, 3), dtype=np.uint8)
    rgb[300:700, 520:760] = [0, 0, 255]
    cv2.imwrite(str(obj / "NP1_0.jpg"), rgb)

    depth = np.full((480, 640), 11000, dtype=np.uint16)
    depth[150:330, 260:380] = 8500
    with h5py.File(obj / "NP1_0.h5", "w") as handle:
        handle.create_dataset("depth", data=depth)

    mask = np.zeros((1024, 1280), dtype=np.uint8)
    mask[300:700, 520:760] = 255
    cv2.imwrite(str(masks / "NP1_0_mask.pbm"), mask)

    samples = SingleObjectIndexBuilder(data, tmp_path / "outputs").build()
    assert len(samples) == 1
    sample = samples[0]
    assert sample.command == "pick the chips can"
    assert sample.target_label == "chips can"
    assert sample.depth_path.suffix == ".h5"

    loaded = SingleObjectRGBDLoader().load_sample(sample)
    assert loaded["rgb"].shape[:2] == (480, 640)
    assert loaded["depth"].shape == (480, 640)
    assert 0.8 < float(loaded["depth"][loaded["depth"] > 0].min()) < 0.9
    assert loaded["target"].mask is not None
    assert loaded["target"].mask.shape == (480, 640)
    assert loaded["target"].bbox is not None
