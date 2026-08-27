import numpy as np
import pytest

from graspnet6d.tsdf import (
    CameraIntrinsics,
    OFFICIAL_TSDF_CONTRACT,
    backproject_masked_depth,
    build_target_centered_tsdf,
    compute_target_workspace,
    inspect_upstream_tsdf_contract,
)


def _example() -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    depth = np.full((24, 32), 0.8, dtype=np.float32)
    # Neighbouring geometry is deliberately different and outside the target mask.
    depth[4:9, 4:9] = 0.72
    mask = np.zeros_like(depth, dtype=bool)
    mask[10:15, 14:19] = True
    intrinsic = CameraIntrinsics(32, 24, 40.0, 40.0, 15.5, 11.5)
    return depth, mask, intrinsic


def test_target_workspace_uses_fixed_physical_scale() -> None:
    depth, mask, intrinsic = _example()
    workspace = compute_target_workspace(depth, mask, intrinsic)
    assert workspace.physical_size_m == pytest.approx(0.30)
    assert workspace.contains_camera_points(workspace.target_centroid_camera_m[None]).item()
    assert np.allclose(
        workspace.workspace_origin_camera_m,
        workspace.target_centroid_camera_m - 0.15,
    )


def test_mask_backprojection_is_metric() -> None:
    depth, mask, intrinsic = _example()
    points = backproject_masked_depth(depth, mask, intrinsic)
    assert points.shape == (25, 3)
    assert np.allclose(points[:, 2], 0.8)


def test_workspace_axes_are_table_aligned() -> None:
    depth, mask, intrinsic = _example()
    camera_to_table = np.array(
        [
            [0.0, -1.0, 0.0, 0.2],
            [1.0, 0.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    workspace = compute_target_workspace(
        depth, mask, intrinsic, T_camera_to_table=camera_to_table
    )
    assert np.allclose(camera_to_table @ workspace.T_local_to_camera, workspace.T_local_to_table)
    assert np.allclose(workspace.T_local_to_table[:3, :3], np.eye(3))


def test_upstream_tsdf_contract_is_executable() -> None:
    pytest.importorskip("open3d")
    observed = inspect_upstream_tsdf_contract()
    assert observed.input_shape == (1, 40, 40, 40)
    assert observed.voxel_size_m == pytest.approx(0.0075)
    assert observed.truncation_distance_m == pytest.approx(0.03)


def test_target_centered_tsdf_integrates_full_scene_depth() -> None:
    pytest.importorskip("open3d")
    depth_m, mask, intrinsic = _example()
    result = build_target_centered_tsdf(
        depth_m,
        mask,
        intrinsic,
        depth_scale=1.0,
        T_camera_to_table=np.eye(4),
        source_view_id=17,
    )
    assert result.tsdf.shape == OFFICIAL_TSDF_CONTRACT.input_shape
    assert np.all(np.isfinite(result.tsdf))
    assert result.full_scene_depth_integrated is True
    assert result.source_view_ids == (17,)
    assert result.T_local_to_table is not None
    assert 0.0 <= result.valid_voxel_fraction <= 1.0
