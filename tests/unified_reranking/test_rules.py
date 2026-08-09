from __future__ import annotations

import numpy as np
import pytest

from unified_reranking.rules import (
    SignConstrainedLinearUtility,
    resolve_rule_features,
    single_rule_scores,
)


def test_rule_resolution_is_predeclared_and_label_free() -> None:
    columns = ["center_prob", "p_center", "width_symmetry", "angle_consistency"]
    resolved = resolve_rule_features(columns)
    assert [(item.family, item.column, item.direction) for item in resolved] == [
        ("soft_target_support", "p_center", 1),
        ("jaw_support", "width_symmetry", 1),
        ("angle_agreement", "angle_consistency", 1),
    ]
    with pytest.raises(ValueError, match="depth_contact"):
        resolve_rule_features(columns, require_all=True)


def test_single_rule_uses_orientation_and_alpha_zero_control() -> None:
    matrix = np.asarray([[1.0, 2.0], [-1.0, -2.0]])
    base = np.asarray([0.25, -0.5])
    columns = ["finger_sweep_obstacle_max", "p_center"]
    np.testing.assert_array_equal(
        single_rule_scores(base, matrix, columns, family="soft_target_support", alpha=0),
        base,
    )
    np.testing.assert_allclose(
        single_rule_scores(base, matrix, columns, family="collision_proxy", alpha=0.5),
        base - 0.5 * matrix[:, 0],
    )


def test_sign_constrained_utility_respects_directions_and_query_weighting() -> None:
    matrix = np.asarray(
        [
            [2.0, -2.0],
            [-2.0, 2.0],
            [1.5, -1.0],
            [-1.5, 1.0],
        ]
    )
    labels = [1, 0, 1, 0]
    queries = ["a", "a", "b", "b"]
    base = np.zeros(4)
    utility = SignConstrainedLinearUtility(alpha=1.0, l2=1e-3).fit(
        matrix,
        labels,
        queries,
        base,
        ["p_center", "finger_sweep_obstacle_max"],
    )
    assert np.all(utility.weights_ >= 0)
    scores = utility.predict(matrix, base, ["p_center", "finger_sweep_obstacle_max"])
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]
    artifact = utility.artifact()
    assert artifact["features"][1]["direction"] == -1

