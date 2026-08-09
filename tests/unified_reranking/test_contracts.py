from __future__ import annotations

import pytest

from src.unified_reranking.contracts import assert_model_feature_columns


@pytest.mark.parametrize(
    "column",
    [
        "candidate_success",
        "jacquard_margin",
        "pool_solvable",
        "native_correct",
        "first_positive_rank",
        "object_category",
        "scene_id",
        "sample_id",
        "candidate_id",
        "gt_mask_iou",
        "source_rgb_path",
    ],
)
def test_forbidden_feature_columns_fail_closed(column: str) -> None:
    with pytest.raises(ValueError, match="forbidden model feature"):
        assert_model_feature_columns(["native_score", column])


def test_safe_feature_columns_are_preserved() -> None:
    columns = ("native_score", "p_center", "contact_depth_symmetry")
    assert_model_feature_columns(columns) == columns
