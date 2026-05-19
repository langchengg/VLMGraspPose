from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.data_types import DatasetSample, GraspCandidate, TargetRegion


def test_numpy_and_path_values_serialize_to_jsonable_types(tmp_path: Path) -> None:
    sample = DatasetSample(
        dataset_name="synthetic",
        sample_id="sample_000",
        rgb_path=tmp_path / "rgb.png",
        depth_path=tmp_path / "depth.png",
        sentence="pick the red box",
        target_label="red box",
    )
    target = TargetRegion(
        target_id=1,
        label="red box",
        bbox=[1, 2, 3, 4],
        mask=np.ones((2, 2), dtype=bool),
        grounding_score=1.0,
        center_2d=(2.0, 3.0),
    )
    candidate = GraspCandidate(
        position=np.array([0.1, 0.2, 0.3]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        approach_vector=np.array([0.0, 0.0, -1.0]),
        closing_direction=np.array([1.0, 0.0, 0.0]),
        gripper_width=0.05,
        grasp_type="top_down",
        initial_geometric_score=0.7,
    )
    assert isinstance(sample.to_json()["rgb_path"], str)
    assert target.to_json()["mask"] is None
    assert candidate.to_json()["position"] == [0.1, 0.2, 0.3]
