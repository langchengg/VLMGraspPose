from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_sampler.geometric_sampler import GeometricGraspSampler
from utils.data_types import GraspCandidate


def test_grasp_candidate_validation_accepts_normalized_candidate() -> None:
    candidate = GraspCandidate(
        position=np.array([0.0, 0.0, 1.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        approach_vector=np.array([0.0, 0.0, -1.0]),
        closing_direction=np.array([1.0, 0.0, 0.0]),
        gripper_width=0.04,
        grasp_type="top_down",
        initial_geometric_score=0.8,
    )
    assert GeometricGraspSampler({"min_width": 0.02, "max_width": 0.10})._valid(candidate)
