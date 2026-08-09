from __future__ import annotations

import pandas as pd

from tools.grasp4dof.analyze_subgroups import _aggregate, _fixed_bins


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_id": "G1",
                "sample_id": "s1",
                "query_type": "name",
                "target_area_fraction": 0.004,
                "predicted_mask_iou": 0.2,
                "predicted_mask_confidence": 0.4,
                "predicted_target_depth_valid_fraction": 0.8,
                "target_object_width_px": 31,
                "scene_clutter_instance_count": 3,
                "j_at_1": True,
                "j_at_5": True,
                "candidate_pool_oracle": True,
                "non_empty": True,
                "nms_candidate_count": 4,
                "empty_reason": None,
                "first_valid_rank": 1,
                "latency_seconds": 0.1,
            },
            {
                "method_id": "G1",
                "sample_id": "s2",
                "query_type": "relation",
                "target_area_fraction": 0.05,
                "predicted_mask_iou": 0.9,
                "predicted_mask_confidence": 0.9,
                "predicted_target_depth_valid_fraction": 1.0,
                "target_object_width_px": 150,
                "scene_clutter_instance_count": 9,
                "j_at_1": False,
                "j_at_5": False,
                "candidate_pool_oracle": False,
                "non_empty": False,
                "nms_candidate_count": 0,
                "empty_reason": "empty_mask",
                "first_valid_rank": None,
                "latency_seconds": 0.2,
            },
        ]
    )


def test_fixed_subgroup_bins_cover_required_dimensions() -> None:
    result = _fixed_bins(_rows())
    assert result.target_area_group.tolist() == ["<0.5%", ">=4%"]
    assert result.candidate_count_group.tolist() == ["1-5", "0"]
    assert result.no_grasp_reason_group.tolist() == ["has_grasp", "empty_mask"]
    assert result.first_valid_rank_group.tolist() == ["rank1", "no_positive"]
    assert result.scene_clutter_group.tolist() == ["1-3", ">=8"]


def test_subgroup_aggregation_keeps_empty_samples_as_failures() -> None:
    result = _aggregate(_fixed_bins(_rows()))
    query = result.loc[
        (result.dimension == "query_type") & (result.group == "relation")
    ].iloc[0]
    assert query.sample_count == 1
    assert query.j_at_1 == 0.0
    assert query.non_empty_rate == 0.0
