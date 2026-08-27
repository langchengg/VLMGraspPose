import numpy as np
import pytest

from graspnet6d.evaluator import (
    EvaluatorAdapterError,
    FORMAL_PARITY_STATUS,
    associate_candidates_to_models,
    ensure_graspnetapi_source,
    evaluate_frozen_candidates,
    relevance_from_friction,
    validate_graspnet_rows,
)


def _pose(x: float) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    return value


def test_graspnet_evaluator_adapter_matches_reference() -> None:
    """The adapter's association matches the official low-level helper exactly.

    This is a unit-level helper parity check, not the required real-scene
    collision/friction parity gate for formal evaluation.
    """

    pytest.importorskip("open3d")
    ensure_graspnetapi_source()
    from graspnetAPI.utils.eval_utils import compute_closest_points, transform_points

    models = [
        np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [-0.01, 0.0, 0.0]]),
    ]
    poses = [_pose(0.0), _pose(1.0)]
    translations = np.array([[0.002, 0.0, 0.0], [0.997, 0.0, 0.0]])
    instances, object_ids, transformed = associate_candidates_to_models(
        translations, models, poses, [11, 29]
    )
    reference_scene = np.concatenate(
        [transform_points(model, pose) for model, pose in zip(models, poses)], axis=0
    )
    reference_point_indices = compute_closest_points(translations, reference_scene)
    reference_instances = np.array([0, 0, 1, 1])[reference_point_indices]
    assert np.array_equal(instances, reference_instances)
    assert np.array_equal(object_ids, np.array([11, 29]))
    assert all(np.array_equal(left, right) for left, right in zip(transformed, [reference_scene[:2], reference_scene[2:]]))
    assert FORMAL_PARITY_STATUS.startswith("blocked_")


@pytest.mark.parametrize(
    ("friction", "expected"),
    [(0.2, 6), (0.4, 5), (0.6, 4), (0.8, 3), (1.0, 2), (1.2, 1), (-1.0, 0), (1.3, 0)],
)
def test_relevance_mapping(friction: float, expected: int) -> None:
    assert (
        relevance_from_friction(
            friction, correct_target=True, collision=False, valid_geometry=True
        )
        == expected
    )


def test_wrong_target_is_relevance_zero() -> None:
    assert relevance_from_friction(0.2, correct_target=False, collision=False) == 0
    assert relevance_from_friction(0.2, correct_target=True, collision=True) == 0
    assert relevance_from_friction(0.2, correct_target=True, collision=False, valid_geometry=False) == 0


def test_graspnet_row_geometry_contract() -> None:
    row = np.zeros((1, 17), dtype=np.float64)
    row[0, 1:4] = [0.05, 0.02, 0.04]
    row[0, 4:13] = np.eye(3).reshape(-1)
    row[0, 13:16] = [0.0, 0.0, 0.5]
    assert validate_graspnet_rows(row).tolist() == [True]
    row[0, 1] = 0.2
    assert validate_graspnet_rows(row).tolist() == [False]


def test_batch_evaluator_fails_closed_without_real_parity_gate() -> None:
    with pytest.raises(EvaluatorAdapterError, match="ParityGate"):
        evaluate_frozen_candidates(
            np.empty((0, 17)),
            models_object_m=[],
            dexnet_models=[],
            poses_object_to_camera=[],
            object_ids=[],
            target_object_id=1,
            dexnet_config={},
            table_points_camera_m=np.empty((0, 3)),
        )
