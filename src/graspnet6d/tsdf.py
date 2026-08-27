"""Target-centred VGN TSDF construction that retains local scene geometry.

The target mask is used only to estimate a metric target centroid.  The TSDF is
integrated from the unmasked scene depth, so neighbouring objects, the table,
and other collision geometry inside the fixed local cube remain present.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VGN_ROOT = REPOSITORY_ROOT / "HiFi_reproduction" / "third_party" / "vgn"


@dataclass(frozen=True)
class TSDFContract:
    physical_size_m: float = 0.30
    resolution: int = 40
    truncation_distance_m: float = 0.03
    depth_truncation_m: float = 2.0
    source: str = "VGN corl2020 perception.py and detection.py"

    @property
    def voxel_size_m(self) -> float:
        return self.physical_size_m / self.resolution

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, self.resolution, self.resolution, self.resolution)

    def validate(self) -> None:
        if self.physical_size_m <= 0 or self.resolution <= 0:
            raise ValueError("TSDF physical size and resolution must be positive")
        if not np.isclose(self.truncation_distance_m, 4 * self.voxel_size_m):
            raise ValueError("VGN TSDF truncation must be four voxels")
        if self.depth_truncation_m <= 0:
            raise ValueError("depth truncation must be positive")


OFFICIAL_TSDF_CONTRACT = TSDFContract()


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_value(cls, value: Any, *, image_shape: tuple[int, int] | None = None) -> "CameraIntrinsics":
        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            result = cls(
                int(value["width"]),
                int(value["height"]),
                float(value["fx"]),
                float(value["fy"]),
                float(value["cx"]),
                float(value["cy"]),
            )
        else:
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3, 3) or image_shape is None:
                raise ValueError("intrinsics must be CameraIntrinsics, mapping, or 3x3 K with image_shape")
            result = cls(
                int(image_shape[1]), int(image_shape[0]), array[0, 0], array[1, 1], array[0, 2], array[1, 2]
            )
        values = np.asarray([result.fx, result.fy, result.cx, result.cy], dtype=np.float64)
        if result.width <= 0 or result.height <= 0 or result.fx <= 0 or result.fy <= 0:
            raise ValueError("invalid camera intrinsics")
        if not np.all(np.isfinite(values)):
            raise ValueError("camera intrinsics contain NaN or Inf")
        return result


@dataclass(frozen=True)
class TargetWorkspace:
    target_centroid_camera_m: np.ndarray
    target_centroid_table_m: np.ndarray
    workspace_origin_camera_m: np.ndarray
    workspace_origin_table_m: np.ndarray
    T_local_to_camera: np.ndarray
    T_local_to_table: np.ndarray
    physical_size_m: float

    def contains_camera_points(self, points_camera_m: np.ndarray, *, tolerance: float = 1e-9) -> np.ndarray:
        points = np.asarray(points_camera_m, dtype=np.float64)
        inverse = np.linalg.inv(self.T_local_to_camera)
        local = transform_points(points, inverse)
        return np.all((local >= -tolerance) & (local <= self.physical_size_m + tolerance), axis=-1)


@dataclass(frozen=True)
class TSDFBuildResult:
    tsdf: np.ndarray
    voxel_size_m: float
    physical_size_m: float
    workspace_origin_camera_m: np.ndarray
    T_local_to_camera: np.ndarray
    T_local_to_table: np.ndarray | None
    depth_scale: float
    valid_voxel_fraction: float
    source_view_ids: tuple[int, ...]
    full_scene_depth_integrated: bool = True

    def cache_record(self) -> dict[str, object]:
        return {
            "tsdf": self.tsdf,
            "voxel_size": np.float64(self.voxel_size_m),
            "physical_size": np.float64(self.physical_size_m),
            "workspace_origin": self.workspace_origin_camera_m,
            "T_local_to_camera": self.T_local_to_camera,
            "T_local_to_table": (
                np.empty((0, 0), dtype=np.float64)
                if self.T_local_to_table is None
                else self.T_local_to_table
            ),
            "depth_scale": np.float64(self.depth_scale),
            "valid_voxel_fraction": np.float64(self.valid_voxel_fraction),
            "source_view_ids": np.asarray(self.source_view_ids, dtype=np.int64),
            "full_scene_depth_integrated": np.bool_(self.full_scene_depth_integrated),
        }


def ensure_vgn_source(vgn_root: Path | str = DEFAULT_VGN_ROOT) -> Path:
    root = Path(vgn_root).expanduser().resolve()
    source = root / "src"
    expected = source / "vgn" / "perception.py"
    if not expected.is_file():
        raise FileNotFoundError(f"VGN source snapshot missing: {expected}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return root


def inspect_upstream_tsdf_contract(vgn_root: Path | str = DEFAULT_VGN_ROOT) -> TSDFContract:
    """Check the installed upstream implementation against the pretrained contract."""

    ensure_vgn_source(vgn_root)
    from vgn.perception import TSDFVolume

    # Instantiation makes voxel/truncation values executable source evidence.
    # It requires Open3D, just like formal TSDF construction.
    volume = TSDFVolume(OFFICIAL_TSDF_CONTRACT.physical_size_m, OFFICIAL_TSDF_CONTRACT.resolution)
    observed = TSDFContract(
        physical_size_m=float(volume.size),
        resolution=int(volume.resolution),
        truncation_distance_m=float(volume.sdf_trunc),
        depth_truncation_m=2.0,
        source=str(Path(vgn_root).expanduser().resolve() / "src" / "vgn" / "perception.py"),
    )
    observed.validate()
    if observed.input_shape != (1, 40, 40, 40):
        raise RuntimeError(f"unexpected VGN input shape derived from source: {observed.input_shape}")
    return observed


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("points must have final dimension 3")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite 4x4 matrix")
    flat = values.reshape(-1, 3)
    result = flat @ matrix[:3, :3].T + matrix[:3, 3]
    return result.reshape(values.shape)


def depth_to_meters(depth: np.ndarray, *, depth_scale: float) -> np.ndarray:
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive")
    value = np.asarray(depth)
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
        raise ValueError("depth must be a numeric HxW array")
    meters = value.astype(np.float32) / float(depth_scale)
    if not np.all(np.isfinite(meters)) or np.any(meters < 0):
        raise ValueError("depth must be finite and non-negative")
    return meters


def backproject_masked_depth(
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: CameraIntrinsics | Mapping[str, float] | np.ndarray,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth and target mask must have the same HxW shape")
    intrinsic = CameraIntrinsics.from_value(intrinsics, image_shape=depth.shape)
    if depth.shape != (intrinsic.height, intrinsic.width):
        raise ValueError("depth shape does not match camera intrinsics")
    valid = mask & np.isfinite(depth) & (depth > 0)
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        raise ValueError("target mask contains no valid positive-depth pixels")
    z = depth[rows, columns]
    x = (columns - intrinsic.cx) * z / intrinsic.fx
    y = (rows - intrinsic.cy) * z / intrinsic.fy
    return np.column_stack((x, y, z))


def compute_target_workspace(
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: CameraIntrinsics | Mapping[str, float] | np.ndarray,
    *,
    T_camera_to_table: np.ndarray | None = None,
    contract: TSDFContract = OFFICIAL_TSDF_CONTRACT,
) -> TargetWorkspace:
    """Centre a fixed-scale, table-axis-aligned cube on the target centroid.

    Passing no table transform is equivalent to identity and is useful only for
    geometry-unit tests.  Formal construction requires the GraspNet
    ``cam0_wrt_table @ camera_pose`` camera-to-table transform.
    """

    contract.validate()
    target_points = backproject_masked_depth(depth_m, target_mask, intrinsics)
    centroid_camera = np.median(target_points, axis=0)
    camera_to_table = (
        np.eye(4, dtype=np.float64)
        if T_camera_to_table is None
        else np.asarray(T_camera_to_table, dtype=np.float64)
    )
    if camera_to_table.shape != (4, 4) or not np.all(np.isfinite(camera_to_table)):
        raise ValueError("T_camera_to_table must be a finite 4x4 transform")
    if not np.allclose(camera_to_table[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("T_camera_to_table has an invalid homogeneous last row")
    centroid_table = transform_points(centroid_camera[None], camera_to_table)[0]
    origin_table = centroid_table - contract.physical_size_m / 2.0
    local_to_table = np.eye(4, dtype=np.float64)
    local_to_table[:3, 3] = origin_table
    local_to_camera = np.linalg.inv(camera_to_table) @ local_to_table
    workspace = TargetWorkspace(
        centroid_camera,
        centroid_table,
        local_to_camera[:3, 3],
        origin_table,
        local_to_camera,
        local_to_table,
        contract.physical_size_m,
    )
    if not bool(workspace.contains_camera_points(centroid_camera[None])[0]):
        raise RuntimeError("computed target centroid is outside its target-centred workspace")
    return workspace


def _safe_volume_grid(volume: Any, resolution: int, voxel_size_m: float) -> np.ndarray:
    cloud = volume.extract_voxel_point_cloud()
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    grid = np.zeros((1, resolution, resolution, resolution), dtype=np.float32)
    if len(points) == 0:
        return grid
    distances = colors[:, 0]
    for point, distance in zip(points, distances):
        index = np.floor(point / voxel_size_m).astype(np.int64)
        if np.all((0 <= index) & (index < resolution)):
            grid[(0, *index)] = float(distance)
    return grid


def build_target_centered_tsdf(
    depth: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: CameraIntrinsics | Mapping[str, float] | np.ndarray,
    *,
    T_camera_to_table: np.ndarray,
    depth_scale: float = 1000.0,
    source_view_id: int = 0,
    contract: TSDFContract = OFFICIAL_TSDF_CONTRACT,
) -> TSDFBuildResult:
    """Integrate one complete scene depth map into a target-centred TSDF.

    No mask is applied to ``depth`` during integration.  Open3D's volume is in
    local coordinates ``[0, physical_size]^3`` and receives local-to-camera as
    its world-to-camera extrinsic.
    """

    import open3d as o3d

    contract.validate()
    depth_m = depth_to_meters(depth, depth_scale=depth_scale)
    intrinsic = CameraIntrinsics.from_value(intrinsics, image_shape=depth_m.shape)
    workspace = compute_target_workspace(
        depth_m,
        target_mask,
        intrinsic,
        T_camera_to_table=T_camera_to_table,
        contract=contract,
    )

    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=contract.physical_size_m,
        resolution=contract.resolution,
        sdf_trunc=contract.truncation_distance_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.zeros_like(depth_m, dtype=np.float32)),
        o3d.geometry.Image(depth_m.astype(np.float32, copy=False)),
        depth_scale=1.0,
        depth_trunc=contract.depth_truncation_m,
        convert_rgb_to_intensity=False,
    )
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic.width,
        height=intrinsic.height,
        fx=intrinsic.fx,
        fy=intrinsic.fy,
        cx=intrinsic.cx,
        cy=intrinsic.cy,
    )
    volume.integrate(rgbd, pinhole, workspace.T_local_to_camera)
    grid = _safe_volume_grid(volume, contract.resolution, contract.voxel_size_m)
    if grid.shape != contract.input_shape or not np.all(np.isfinite(grid)):
        raise RuntimeError("constructed TSDF violates the VGN input contract")

    valid_fraction = float(np.count_nonzero(grid) / grid.size)
    return TSDFBuildResult(
        tsdf=grid,
        voxel_size_m=contract.voxel_size_m,
        physical_size_m=contract.physical_size_m,
        workspace_origin_camera_m=workspace.workspace_origin_camera_m,
        T_local_to_camera=workspace.T_local_to_camera,
        T_local_to_table=workspace.T_local_to_table,
        depth_scale=float(depth_scale),
        valid_voxel_fraction=valid_fraction,
        source_view_ids=(int(source_view_id),),
    )
