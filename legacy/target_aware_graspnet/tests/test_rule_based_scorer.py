from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scoring.rule_based_scorer import RuleBasedScorer
from utils.data_types import CandidateFeatureVector, GraspCandidate


def _candidate(score: float) -> GraspCandidate:
    return GraspCandidate(
        position=np.array([0.0, 0.0, 0.1]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        approach_vector=np.array([0.0, 0.0, -1.0]),
        closing_direction=np.array([1.0, 0.0, 0.0]),
        gripper_width=0.05,
        grasp_type="top_down",
        initial_geometric_score=score,
    )


def test_rule_based_scorer_uses_configured_formula_and_ranks_descending() -> None:
    low = CandidateFeatureVector(
        target_overlap=0.1,
        center_alignment=0.1,
        distance_to_target_center=0.1,
        gripper_width_match=0.1,
        approach_direction_score=0.1,
        depth_stability=0.1,
        collision_penalty=1.0,
        boundary_penalty=1.0,
        initial_geometric_score=0.1,
    )
    high = CandidateFeatureVector(
        target_overlap=1.0,
        center_alignment=1.0,
        distance_to_target_center=0.0,
        gripper_width_match=1.0,
        approach_direction_score=1.0,
        depth_stability=1.0,
        collision_penalty=0.0,
        boundary_penalty=0.0,
        initial_geometric_score=1.0,
    )

    ranked = RuleBasedScorer().score([_candidate(0.1), _candidate(1.0)], [low, high])

    assert ranked[0].final_score == pytest.approx(0.9)
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[0].metadata["score_breakdown"]["target_overlap"] == 0.25
