from __future__ import annotations

from pathlib import Path

from dataset.camera_loader import load_depth, load_intrinsics, load_label, load_rgb
from utils.data_types import GraspNetSample


class GraspNetLoader:
    def __init__(self, depth_scale: float = 1000.0, fallback_intrinsics: dict | None = None):
        self.depth_scale = depth_scale
        self.fallback_intrinsics = fallback_intrinsics

    def load_sample(self, sample: GraspNetSample) -> dict:
        data = {
            "rgb": load_rgb(sample.rgb_path),
            "depth": load_depth(sample.depth_path, self.depth_scale),
            "intrinsics": load_intrinsics(sample.camera_intrinsic_path, self.fallback_intrinsics),
            "label": None,
        }
        if sample.label_path and Path(sample.label_path).exists():
            data["label"] = load_label(sample.label_path)
        return data
