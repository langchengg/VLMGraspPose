from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scoring.factory import build_scorer
from scoring.mlp_scorer import MLPScorer
from utils.data_types import CandidateFeatureVector, GraspCandidate


def _candidate(score: float) -> GraspCandidate:
    return GraspCandidate(
        position=np.array([0.0, 0.0, 1.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        approach_vector=np.array([0.0, 0.0, -1.0]),
        closing_direction=np.array([1.0, 0.0, 0.0]),
        gripper_width=0.05,
        grasp_type="top_down",
        initial_geometric_score=score,
    )


def _features(value: float) -> CandidateFeatureVector:
    return CandidateFeatureVector(
        target_overlap=value,
        center_alignment=value,
        distance_to_target_center=0.0 if value > 0.5 else 0.2,
        gripper_width_match=value,
        approach_direction_score=value,
        depth_stability=value,
        collision_penalty=1.0 - value,
        boundary_penalty=1.0 - value,
        initial_geometric_score=value,
    )


def test_mlp_scorer_ranks_candidates_and_records_metadata() -> None:
    scorer = MLPScorer.from_config({"init": "rule_based"})

    ranked = scorer.score([_candidate(0.1), _candidate(1.0)], [_features(0.1), _features(1.0)])

    assert ranked[0].final_score > ranked[1].final_score
    assert ranked[0].rank == 1
    assert ranked[0].metadata["scorer"] == "mlp"
    assert "mlp_feature_vector" in ranked[0].metadata


def test_build_scorer_uses_mlp_when_configured() -> None:
    scorer = build_scorer({"method": "mlp", "mlp": {"init": "rule_based"}})

    assert isinstance(scorer, MLPScorer)

