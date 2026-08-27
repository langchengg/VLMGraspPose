"""Deterministic unit tests for coordinate and gripper-axis transforms."""

from __future__ import annotations

import numpy as np
import pytest

from graspnet6d.geometry import (
    VGN_TO_GRASPNET_AXIS_MAP,
    VGN_TO_GRASPNET_AXIS_MAP_STATUS,
    CameraIntrinsics,
    as_rotation_matrix,
    as_transform,
    backproject_pixels,
    compose_transforms,
    invert_transform,
    make_transform,
    project_points,
    rotation_geodesic_deg,
    transform_points,
    transform_pose,
    vgn_rotation_to_graspnet,
)


def _z_rotation_90() -> np.ndarray:
    return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_depth_projection_round_trip() -> None:
    intrinsics = CameraIntrinsics(
        fx=500.0, fy=505.0, cx=320.0, cy=240.0, width=640, height=480
    )
    pixels = np.array([[320.0, 240.0], [120.5, 100.25], [500.0, 350.0]])
    depths = np.array([0.75, 1.20, 0.55])

    points = backproject_pixels(pixels, depths, intrinsics)
    projected = project_points(points, intrinsics, require_in_image=True)

    np.testing.assert_allclose(projected, pixels, atol=1e-12, rtol=0.0)


def test_camera_table_camera_and_local_table_local_round_trips() -> None:
    table_from_camera = make_transform(_z_rotation_90(), (0.40, -0.20, 0.10))
    table_from_local = make_transform(np.eye(3), (0.30, 0.05, 0.02))
    point_camera = np.array([0.10, 0.20, 0.80])
    point_local = np.array([0.02, 0.03, 0.04])

    camera_round_trip = transform_points(
        invert_transform(table_from_camera),
        transform_points(table_from_camera, point_camera),
    )
    local_round_trip = transform_points(
        invert_transform(table_from_local),
        transform_points(table_from_local, point_local),
    )

    np.testing.assert_allclose(camera_round_trip, point_camera, atol=1e-12)
    np.testing.assert_allclose(local_round_trip, point_local, atol=1e-12)
    np.testing.assert_allclose(
        compose_transforms(table_from_camera, invert_transform(table_from_camera)),
        np.eye(4),
        atol=1e-12,
    )


def test_pose_transform_preserves_so3() -> None:
    table_from_camera = make_transform(_z_rotation_90(), (0.4, 0.0, 0.0))
    rotation_table, translation_table = transform_pose(
        table_from_camera, np.eye(3), (0.1, 0.2, 0.8)
    )

    np.testing.assert_allclose(rotation_table.T @ rotation_table, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation_table) == pytest.approx(1.0)
    np.testing.assert_allclose(translation_table, (0.2, 0.1, 0.8), atol=1e-12)


def test_vgn_to_graspnet_axis_hypothesis_is_explicit_and_right_handed() -> None:
    converted = vgn_rotation_to_graspnet(np.eye(3))

    assert VGN_TO_GRASPNET_AXIS_MAP_STATUS == "unvalidated_hypothesis"
    np.testing.assert_array_equal(converted, VGN_TO_GRASPNET_AXIS_MAP)
    np.testing.assert_array_equal(converted[:, 0], (0.0, 0.0, 1.0))
    np.testing.assert_array_equal(converted[:, 1], (0.0, 1.0, 0.0))
    np.testing.assert_array_equal(converted[:, 2], (-1.0, 0.0, 0.0))
    assert np.linalg.det(converted) == pytest.approx(1.0)


def test_rotation_geodesic_distance() -> None:
    assert rotation_geodesic_deg(np.eye(3), np.eye(3)) == pytest.approx(0.0)
    assert rotation_geodesic_deg(np.eye(3), _z_rotation_90()) == pytest.approx(90.0)


def test_geometry_validation_rejects_reflection_and_bad_homogeneous_row() -> None:
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        as_rotation_matrix(np.diag((1.0, 1.0, -1.0)))
    invalid = np.eye(4)
    invalid[3, 0] = 1.0
    with pytest.raises(ValueError, match="homogeneous"):
        as_transform(invalid)


def test_projection_requires_positive_camera_depth() -> None:
    intrinsics = CameraIntrinsics(500.0, 500.0, 320.0, 240.0)
    with pytest.raises(ValueError, match="positive"):
        project_points((0.0, 0.0, 0.0), intrinsics)
