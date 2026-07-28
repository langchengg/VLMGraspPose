from __future__ import annotations

import importlib
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.grasping.vgn_adapter import (
    Candidate,
    OFFICIAL_VOXEL_SIZE_M,
    build_tsdf_grid,
    decode_candidates,
    ensure_official_vgn_path,
    filter_target_candidates,
    process,
    select,
    select_candidate,
    sort_candidates_by_quality,
    validate_tsdf_grid,
)
from src.grasping.vgn_geometry import (
    CameraIntrinsics,
    GeometryError,
    build_task_frame,
    invert_transform,
    plane_z_in_task,
    prepare_target_mask,
    resize_mask_nearest,
    resolve_depth_m,
    transform_points,
)
from src.grasping.vgn_pipeline import candidate_status


ROOT = Path(__file__).resolve().parents[1]
VGN_ROOT = ROOT / "third_party" / "vgn"


def _candidate(
    position: tuple[float, float, float], quality: float = 0.95, index: int = 0
) -> Candidate:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return Candidate(
        official_selection_index=index,
        raw_order_index=index,
        vgn_quality=quality,
        voxel_index_ijk=(1, 2, 3),
        position_task_m=np.asarray(position, dtype=np.float64),
        quaternion_task_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        rotation_task_3x3=np.eye(3),
        width_m=0.04,
        T_task_grasp=transform,
    )


def test_depth_mm_to_m() -> None:
    raw = np.array([[0, 1000, 2500]], dtype=np.uint16)
    result = resolve_depth_m(raw, unit="mm", depth_scale=1000.0)
    np.testing.assert_allclose(result.depth_m, [[0.0, 1.0, 2.5]])
    assert result.source_unit == "mm"


def test_depth_m_not_rescaled_twice() -> None:
    raw = np.array([[0.0, 0.75, 1.25]], dtype=np.float32)
    result = resolve_depth_m(raw, unit="m", depth_scale=1000.0)
    np.testing.assert_allclose(result.depth_m, raw)
    assert result.source_unit == "m"


def test_mask_nearest_resize() -> None:
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    resized = resize_mask_nearest(mask, (4, 4))
    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(resized, expected)


def test_task_frame_is_right_handed() -> None:
    frame = build_task_frame([0.0, 0.0, 0.8], [0.0, 0.0, -1.0, 1.0])
    rotation = frame.T_camera_task[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-10)
    projection_task = transform_points(frame.T_task_camera, frame.target_projection_camera_m)
    np.testing.assert_allclose(projection_task, [0.15, 0.15, 0.05], atol=1e-10)


def test_transform_round_trip() -> None:
    frame = build_task_frame([0.1, -0.2, 0.8], [0.0, 0.0, -1.0, 1.0])
    points = np.array([[0.01, 0.02, 0.03], [0.2, 0.1, 0.05]])
    camera = transform_points(frame.T_camera_task, points)
    recovered = transform_points(invert_transform(frame.T_camera_task), camera)
    np.testing.assert_allclose(recovered, points, atol=1e-10)
    plane_points = np.array([[0.0, 0.0, 1.0], [0.2, -0.1, 1.0]])
    np.testing.assert_allclose(
        plane_z_in_task([0.0, 0.0, -1.0, 1.0], frame.T_camera_task, plane_points),
        0.05,
        atol=1e-10,
    )


def test_tsdf_shape_is_1_40_40_40() -> None:
    depth = np.full((48, 64), 1.0, dtype=np.float32)
    intrinsics = CameraIntrinsics(64, 48, 55.0, 55.0, 31.5, 23.5, source="synthetic_test")
    transform = np.eye(4)
    transform[:3, 3] = [-0.15, -0.15, 0.8]
    result = build_tsdf_grid(depth, intrinsics, transform, vgn_root=VGN_ROOT)
    assert result.grid.shape == (1, 40, 40, 40)
    assert result.voxel_size_m == pytest.approx(0.0075)


def test_vgn_input_shape_validation() -> None:
    assert validate_tsdf_grid(np.zeros((1, 40, 40, 40))).shape == (1, 40, 40, 40)
    with pytest.raises(ValueError, match="shape"):
        validate_tsdf_grid(np.zeros((40, 40, 40)))


def test_official_width_conversion() -> None:
    grasp = SimpleNamespace(
        pose=SimpleNamespace(translation=np.array([2.0, 3.0, 4.0]), rotation=Rotation.identity()),
        width=5.0,
    )
    candidate = decode_candidates([grasp], [0.91])[0]
    assert candidate.width_m == pytest.approx(5.0 * OFFICIAL_VOXEL_SIZE_M)
    np.testing.assert_allclose(
        candidate.position_task_m, np.array([2.0, 3.0, 4.0]) * OFFICIAL_VOXEL_SIZE_M
    )


def test_candidate_projection() -> None:
    intrinsics = CameraIntrinsics(11, 11, 10.0, 10.0, 5.0, 5.0, source="synthetic_test")
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True
    annotated, accepted = filter_target_candidates(
        [_candidate((0.0, 0.0, 1.0))],
        intrinsics=intrinsics,
        raw_target_mask=mask,
        dilated_target_mask=mask,
        depth_m=np.ones((11, 11), dtype=np.float32),
        target_points_camera=np.array([[0.0, 0.0, 1.0]]),
        T_camera_task=np.eye(4),
    )
    np.testing.assert_allclose(annotated[0].projected_uv, [5.0, 5.0])
    assert annotated[0].inside_raw_target_mask is True
    assert len(accepted) == 1


def test_empty_mask_failure() -> None:
    with pytest.raises(GeometryError) as error:
        prepare_target_mask(
            np.zeros((8, 8), dtype=np.uint8),
            np.ones((8, 8), dtype=np.float32),
            min_area_px=1,
            min_valid_depth_points=1,
        )
    assert error.value.status == "empty_mask"


def test_no_candidate_returns_clean_status() -> None:
    assert candidate_status(0, 0) == "no_official_grasp"
    assert candidate_status(3, 0) == "no_target_grasp"
    assert select_candidate([], policy="highest_vgn_quality") is None


def test_score_sort_is_descending() -> None:
    candidates = [_candidate((0, 0, 1), 0.91, 0), _candidate((0, 0, 1), 0.99, 1)]
    ordered = sort_candidates_by_quality(candidates)
    assert [item.vgn_quality for item in ordered] == [0.99, 0.91]
    assert [item.score_rank for item in ordered] == [1, 2]


def test_no_custom_score_fields() -> None:
    record = _candidate((0, 0, 1)).to_record()
    forbidden = {"combined_score", "custom_score", "semantic_score", "mask_score", "collision_score"}
    assert forbidden.isdisjoint(record)
    assert record["score_source"] == "official_vgn_processed_quality"
    assert isinstance(record["rank"], int)
    assert isinstance(record["score_rank"], int)
    assert record["rank"] == record["score_rank"] == 1


def test_off_target_candidate_is_rejected() -> None:
    intrinsics = CameraIntrinsics(11, 11, 10.0, 10.0, 5.0, 5.0, source="synthetic_test")
    mask = np.zeros((11, 11), dtype=bool)
    mask[0, 0] = True
    annotated, accepted = filter_target_candidates(
        [_candidate((0.0, 0.0, 1.0))],
        intrinsics=intrinsics,
        raw_target_mask=mask,
        dilated_target_mask=mask,
        T_camera_task=np.eye(4),
    )
    assert annotated[0].projection_in_image is True
    assert annotated[0].inside_dilated_target_mask is False
    assert accepted == []


def test_official_postprocessing_matches_upstream() -> None:
    ensure_official_vgn_path(VGN_ROOT)
    dummy_vis = types.ModuleType("vgn.vis")
    dummy_vis.draw_quality = lambda *args, **kwargs: None
    sys.modules["vgn.vis"] = dummy_vis
    import vgn

    vgn.vis = dummy_vis
    upstream = importlib.import_module("vgn.detection")

    rng = np.random.RandomState(123)
    # Exercise every upstream TSDF region and the modernized
    # scipy.ndimage.binary_dilation(mask=..., iterations=2) path.
    tsdf = rng.choice(
        np.array([0.0, 0.25, 0.75], dtype=np.float64),
        size=(1, 40, 40, 40),
        p=(0.25, 0.35, 0.40),
    )
    quality = rng.uniform(0.0, 0.2, size=(40, 40, 40))
    quality[4:13, 4:13, 4:13] = 1.0
    quality[24:34, 20:31, 8:18] = 0.98
    rotation = rng.normal(size=(4, 40, 40, 40))
    rotation /= np.linalg.norm(rotation, axis=0, keepdims=True)
    width = rng.uniform(2.0, 8.0, size=(40, 40, 40))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        up_processed = upstream.process(
            tsdf.copy(), quality.copy(), rotation.copy(), width.copy()
        )
    adapted = process(tsdf.copy(), quality.copy(), rotation.copy(), width.copy())
    for actual, expected in zip(adapted, up_processed):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    up_grasps, up_scores = upstream.select(
        up_processed[0].copy(), up_processed[1], up_processed[2]
    )
    our_grasps, our_scores = select(
        adapted[0].copy(), adapted[1], adapted[2]
    )
    assert len(our_grasps) == len(up_grasps)
    np.testing.assert_allclose(our_scores, up_scores, rtol=0.0, atol=1e-12)
    for ours, official in zip(our_grasps, up_grasps):
        np.testing.assert_array_equal(ours.pose.translation, official.pose.translation)
        np.testing.assert_allclose(
            ours.pose.rotation.as_quat(), official.pose.rotation.as_quat(), atol=1e-12
        )
        assert ours.width == pytest.approx(official.width, abs=1e-12)
