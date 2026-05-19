from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.data_types import CandidateFeatureVector, GraspCandidate, ScoredGrasp, TargetRegion
from visualization.visualize_rgb import save_rgb_overlay


def _scored_grasp(x: float, y: float, z: float, rank: int) -> ScoredGrasp:
    candidate = GraspCandidate(
        position=np.array([x, y, z], dtype=float),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        approach_vector=np.array([0.0, 0.0, -1.0], dtype=float),
        closing_direction=np.array([1.0, 0.0, 0.0], dtype=float),
        gripper_width=0.04,
        grasp_type="top_down",
        initial_geometric_score=0.8,
    )
    features = CandidateFeatureVector(
        target_overlap=1.0,
        center_alignment=1.0,
        distance_to_target_center=0.0,
        gripper_width_match=1.0,
        approach_direction_score=1.0,
        depth_stability=1.0,
        collision_penalty=0.0,
        boundary_penalty=0.0,
        initial_geometric_score=0.8,
    )
    return ScoredGrasp(candidate=candidate, features=features, final_score=0.8, rank=rank)


def test_rgb_overlay_draws_topk_grasp_rectangles(tmp_path: Path) -> None:
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    target = TargetRegion(
        target_id=1,
        label="box",
        bbox=[30, 30, 70, 70],
        mask=mask,
        grounding_score=1.0,
        center_2d=(50.0, 50.0),
        command="pick the box",
    )
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    grasps = [_scored_grasp(0.0, 0.0, 1.0, 1), _scored_grasp(0.1, 0.0, 1.0, 2)]
    out = tmp_path / "overlay.png"

    save_rgb_overlay(out, rgb, target, grasps[0], intrinsics, top_k=grasps)

    image = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert image is not None
    # Top-1 is drawn in red and fallback Top-K in blue in BGR image space.
    red_pixels = np.count_nonzero((image[:, :, 2] > 180) & (image[:, :, 1] < 120) & (image[:, :, 0] < 120))
    blue_pixels = np.count_nonzero((image[:, :, 0] > 180) & (image[:, :, 1] < 160) & (image[:, :, 2] < 160))
    assert red_pixels > 0
    assert blue_pixels > 0

