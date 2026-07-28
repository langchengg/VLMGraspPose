"""Minimal, deterministic adapter for the official CoRL 2020 VGN code.

The post-processing in :func:`process` and :func:`select` is adapted from the
BSD-3-Clause licensed upstream implementation at
https://github.com/ethz-asl/vgn/blob/d7af0622433f52ae88ebe81533f12b46b33e951a/src/vgn/detection.py
(Copyright 2020 ETHZ ASL).  It intentionally preserves the upstream operation
order and numerical definitions while replacing the removed
``scipy.ndimage.morphology.binary_dilation`` alias with
``scipy.ndimage.binary_dilation``.

No score other than the processed official VGN quality is computed here.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


LOGGER = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_VGN_ROOT = REPOSITORY_ROOT / "third_party" / "vgn"
OFFICIAL_VGN_REPOSITORY = "https://github.com/ethz-asl/vgn"
OFFICIAL_VGN_BRANCH = "corl2020"
OFFICIAL_VGN_COMMIT = "d7af0622433f52ae88ebe81533f12b46b33e951a"

OFFICIAL_WORKSPACE_SIZE_M = 0.30
OFFICIAL_RESOLUTION = 40
OFFICIAL_VOXEL_SIZE_M = OFFICIAL_WORKSPACE_SIZE_M / OFFICIAL_RESOLUTION
OFFICIAL_FINGER_DEPTH_M = 0.05
OFFICIAL_TABLE_HEIGHT_M = 0.05
OFFICIAL_GAUSSIAN_FILTER_SIGMA = 1.0
OFFICIAL_MIN_WIDTH_VOXELS = 1.33
OFFICIAL_MAX_WIDTH_VOXELS = 9.33
OFFICIAL_QUALITY_THRESHOLD = 0.90
OFFICIAL_MAX_FILTER_SIZE = 4
OFFICIAL_TSDF_INPUT_SHAPE = (1, 40, 40, 40)
OFFICIAL_DEPTH_TRUNC_M = 2.0
SCORE_SOURCE = "official_vgn_processed_quality"


class VGNAdapterError(RuntimeError):
    """Base class for adapter failures with actionable messages."""


@dataclass(frozen=True)
class OfficialVGNPreset:
    """Physical and post-processing constants of the pretrained model."""

    workspace_size_m: float = OFFICIAL_WORKSPACE_SIZE_M
    resolution: int = OFFICIAL_RESOLUTION
    voxel_size_m: float = OFFICIAL_VOXEL_SIZE_M
    finger_depth_m: float = OFFICIAL_FINGER_DEPTH_M
    table_height_m: float = OFFICIAL_TABLE_HEIGHT_M
    gaussian_filter_sigma: float = OFFICIAL_GAUSSIAN_FILTER_SIGMA
    min_width_voxels: float = OFFICIAL_MIN_WIDTH_VOXELS
    max_width_voxels: float = OFFICIAL_MAX_WIDTH_VOXELS
    quality_threshold: float = OFFICIAL_QUALITY_THRESHOLD
    maximum_filter_size: int = OFFICIAL_MAX_FILTER_SIZE


OFFICIAL_PRESET = OfficialVGNPreset()


@dataclass(frozen=True)
class DeviceSelection:
    """Resolved torch device and any explicit MPS fallback information."""

    requested: str
    resolved: str
    mps_smoke_tested: bool = False
    fallback_reason: str | None = None

    @property
    def fell_back(self) -> bool:
        return self.fallback_reason is not None


@dataclass(frozen=True)
class PredictionResult:
    """Official network outputs plus the device actually used."""

    qual_vol: np.ndarray
    rot_vol: np.ndarray
    width_vol: np.ndarray
    requested_device: str
    used_device: str
    mps_fallback_reason: str | None = None

    def __iter__(self) -> Iterator[np.ndarray]:
        # Allows ``qual, rot, width = predict_official(...)`` without dropping
        # the separately inspectable device/fallback metadata.
        yield self.qual_vol
        yield self.rot_vol
        yield self.width_vol


@dataclass(frozen=True)
class TSDFBuildResult:
    """TSDF grid together with the volume handle used for diagnostics."""

    grid: np.ndarray
    volume: Any
    voxel_size_m: float
    depth_trunc_m: float
    implementation: str

    def get_cloud(self) -> Any:
        return self.volume.get_cloud()


@dataclass(frozen=True)
class Candidate:
    """One official VGN local maximum, decoded without a half-voxel offset."""

    official_selection_index: int
    raw_order_index: int
    vgn_quality: float
    voxel_index_ijk: tuple[int, int, int]
    position_task_m: np.ndarray
    quaternion_task_xyzw: np.ndarray
    rotation_task_3x3: np.ndarray
    width_m: float
    T_task_grasp: np.ndarray
    score_rank: int | None = None
    position_camera_m: np.ndarray | None = None
    quaternion_camera_xyzw: np.ndarray | None = None
    T_camera_grasp: np.ndarray | None = None
    projected_uv: np.ndarray | None = None
    inside_raw_target_mask: bool = False
    inside_dilated_target_mask: bool = False
    nearest_target_point_distance_m: float | None = None
    projected_depth_difference_m: float | None = None
    positive_camera_z: bool = False
    projection_in_image: bool = False
    target_filter_accepted: bool = False

    @property
    def score(self) -> float:
        """Alias for the unmodified official VGN processed quality."""

        return self.vgn_quality

    def to_record(self) -> dict[str, Any]:
        """Return the research JSON schema without any custom score fields."""

        def array_or_none(value: np.ndarray | None) -> Any:
            return None if value is None else np.asarray(value).tolist()

        # Production callers annotate the quality rank before serialization.
        # The fallback keeps standalone diagnostic records schema-valid while
        # preserving the upstream candidate order.
        rank = int(
            self.score_rank
            if self.score_rank is not None
            else self.raw_order_index + 1
        )
        return {
            "rank": rank,
            "score_rank": rank,
            "official_selection_index": self.official_selection_index,
            "raw_order_index": self.raw_order_index,
            "vgn_quality": self.vgn_quality,
            "voxel_index_ijk": list(self.voxel_index_ijk),
            "position_task_m": self.position_task_m.tolist(),
            "quaternion_task_xyzw": self.quaternion_task_xyzw.tolist(),
            "rotation_task_3x3": self.rotation_task_3x3.tolist(),
            "width_m": self.width_m,
            "T_task_grasp": self.T_task_grasp.tolist(),
            "position_camera_m": array_or_none(self.position_camera_m),
            "quaternion_camera_xyzw": array_or_none(self.quaternion_camera_xyzw),
            "T_camera_grasp": array_or_none(self.T_camera_grasp),
            "projected_uv": array_or_none(self.projected_uv),
            "inside_raw_target_mask": self.inside_raw_target_mask,
            "inside_dilated_target_mask": self.inside_dilated_target_mask,
            "nearest_target_point_distance_m": self.nearest_target_point_distance_m,
            "projected_depth_difference_m": self.projected_depth_difference_m,
            "positive_camera_z": self.positive_camera_z,
            "projection_in_image": self.projection_in_image,
            "target_filter_accepted": self.target_filter_accepted,
            "score_source": SCORE_SOURCE,
        }


@dataclass(frozen=True)
class PostprocessingResult:
    processed_quality: np.ndarray
    rotation: np.ndarray
    width_voxels: np.ndarray
    candidates: tuple[Candidate, ...]


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def ensure_official_vgn_path(vgn_root: Path | str | None = None) -> Path:
    """Add only the official checkout's ``src`` directory to ``sys.path``.

    A previously imported, unrelated package named ``vgn`` is rejected instead
    of being silently mixed with the pinned checkout.
    """

    root = Path(vgn_root or OFFICIAL_VGN_ROOT).expanduser().resolve()
    source = root / "src"
    marker = source / "vgn" / "detection.py"
    if not marker.is_file():
        raise FileNotFoundError(
            f"Official VGN checkout missing at {root}; expected {marker}. "
            f"Clone {OFFICIAL_VGN_REPOSITORY} branch {OFFICIAL_VGN_BRANCH}."
        )
    loaded = sys.modules.get("vgn")
    loaded_path = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_path is not None and not _path_is_within(Path(loaded_path), source):
        raise ImportError(
            f"An incompatible vgn package is already imported from {loaded_path}; "
            f"expected the pinned checkout under {source}"
        )
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return root


def official_checkout_commit(vgn_root: Path | str | None = None) -> str:
    """Read and verify the immutable upstream checkout commit."""

    root = ensure_official_vgn_path(vgn_root)
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise VGNAdapterError(f"Cannot verify official VGN git commit at {root}") from error
    if commit != OFFICIAL_VGN_COMMIT:
        raise VGNAdapterError(
            f"Official VGN checkout commit mismatch: found {commit}, "
            f"required {OFFICIAL_VGN_COMMIT} ({OFFICIAL_VGN_BRANCH})"
        )
    try:
        tracked_changes = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise VGNAdapterError(f"Cannot verify official VGN worktree at {root}") from error
    if tracked_changes:
        raise VGNAdapterError(
            "Official VGN checkout has tracked modifications; refusing to label it "
            f"as pristine commit {commit}: {tracked_changes}"
        )
    return commit


def checkpoint_sha256(path: Path | str) -> str:
    """Require a real checkpoint and return its SHA256 digest."""

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"VGN checkpoint not found: {checkpoint}. Download the official data "
            "bundle linked from the VGN README and provide data/models/vgn_conv.pth. "
            "Random weights are never used for inference."
        )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mps_conv3d_smoke_test() -> tuple[bool, str | None]:
    import torch

    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_built() or not backend.is_available():
        return False, "torch MPS backend is not built and available"
    try:
        layer = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1).to("mps")
        value = torch.zeros((1, 1, 5, 5, 5), dtype=torch.float32, device="mps")
        with torch.no_grad():
            output = layer(value)
        # Materialize on CPU so deferred backend failures are observed here.
        output.cpu()
    except Exception as error:  # MPS backend error types differ across torch versions.
        return False, f"{type(error).__name__}: {error}"
    return True, None


def resolve_device_info(
    requested: str = "auto", *, logger: logging.Logger | None = None
) -> DeviceSelection:
    """Resolve ``auto|cuda|cpu|mps`` using the project's conservative policy."""

    import torch

    logger = logger or LOGGER
    name = str(requested).lower()
    if name not in {"auto", "cuda", "cpu", "mps"}:
        raise ValueError("device must be one of: auto, cuda, cpu, mps")
    if name == "auto":
        if torch.cuda.is_available():
            return DeviceSelection(name, "cuda")
        # Auto never selects MPS.  This is explicit even on non-Darwin hosts.
        return DeviceSelection(name, "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise VGNAdapterError("CUDA was requested but torch.cuda.is_available() is false")
        return DeviceSelection(name, "cuda")
    if name == "cpu":
        return DeviceSelection(name, "cpu")

    passed, reason = _mps_conv3d_smoke_test()
    if not passed:
        message = f"MPS Conv3d smoke test failed; falling back to CPU: {reason}"
        logger.warning(message)
        return DeviceSelection(name, "cpu", mps_smoke_tested=True, fallback_reason=message)
    return DeviceSelection(name, "mps", mps_smoke_tested=True)


def resolve_device(requested: str = "auto", *, logger: logging.Logger | None = None) -> Any:
    """Return a ``torch.device`` while retaining detailed info via the companion API."""

    import torch

    return torch.device(resolve_device_info(requested, logger=logger).resolved)


def load_official_network(
    weights_path: Path | str,
    *,
    device: str | Any = "auto",
    vgn_root: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Construct the untouched official ConvNet and safely load its state dict."""

    import torch

    root = ensure_official_vgn_path(vgn_root)
    official_checkout_commit(root)
    checkpoint = Path(weights_path).expanduser().resolve()
    digest = checkpoint_sha256(checkpoint)
    selection = (
        resolve_device_info(device, logger=logger)
        if isinstance(device, str)
        else DeviceSelection(str(device), str(torch.device(device)))
    )
    torch_device = torch.device(selection.resolved)

    from vgn.networks import get_network

    parts = checkpoint.stem.split("_")
    if len(parts) < 2 or parts[1].lower() != "conv":
        raise ValueError(
            f"Official checkpoint filename must identify the conv architecture, got {checkpoint.name}"
        )
    net = get_network("conv").to(torch_device)
    try:
        state_dict = torch.load(
            checkpoint, map_location=torch_device, weights_only=True
        )
    except TypeError:
        # Compatibility only for old torch releases without ``weights_only``.
        state_dict = torch.load(checkpoint, map_location=torch_device)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    # Non-persistent provenance is convenient to the caller and does not alter
    # architecture, state_dict keys, or checkpoint parameters.
    net._vgn_checkpoint_sha256 = digest
    net._vgn_checkpoint_path = str(checkpoint)
    net._vgn_device_selection = selection
    return net


def validate_tsdf_grid(tsdf_vol: np.ndarray) -> np.ndarray:
    """Validate the official network input shape and return contiguous float32."""

    array = np.asarray(tsdf_vol)
    if array.shape != OFFICIAL_TSDF_INPUT_SHAPE:
        raise ValueError(
            f"VGN TSDF input must have shape {OFFICIAL_TSDF_INPUT_SHAPE}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"VGN TSDF input must be numeric, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("VGN TSDF input contains NaN or Inf")
    return np.ascontiguousarray(array, dtype=np.float32)


def _intrinsic_values(intrinsics: Any) -> tuple[int, int, float, float, float, float]:
    def get(name: str) -> Any:
        if isinstance(intrinsics, Mapping):
            if name not in intrinsics:
                raise ValueError(f"camera intrinsics missing {name}")
            return intrinsics[name]
        if not hasattr(intrinsics, name):
            raise ValueError(f"camera intrinsics missing {name}")
        return getattr(intrinsics, name)

    width, height = int(get("width")), int(get("height"))
    fx, fy, cx, cy = (float(get(name)) for name in ("fx", "fy", "cx", "cy"))
    values = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    if width <= 0 or height <= 0 or not np.all(np.isfinite(values)) or fx <= 0 or fy <= 0:
        raise ValueError("invalid camera intrinsics")
    return width, height, fx, fy, cx, cy


class _ConfigurableTSDFVolume:
    """Strict Open3D equivalent of upstream TSDFVolume with depth_trunc exposed."""

    def __init__(self, size: float, resolution: int, depth_trunc_m: float):
        import open3d as o3d

        self.size = float(size)
        self.resolution = int(resolution)
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4.0 * self.voxel_size
        self.depth_trunc_m = float(depth_trunc_m)
        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    def integrate(self, depth_img: np.ndarray, intrinsic: Any, extrinsic: Any) -> None:
        import open3d as o3d

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=self.depth_trunc_m,
            convert_rgb_to_intensity=False,
        )
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )
        matrix = extrinsic.as_matrix() if hasattr(extrinsic, "as_matrix") else extrinsic
        self._volume.integrate(rgbd, o3d_intrinsic, np.asarray(matrix, dtype=np.float64))

    def get_grid(self) -> np.ndarray:
        cloud = self._volume.extract_voxel_point_cloud()
        points = np.asarray(cloud.points)
        distances = np.asarray(cloud.colors)[:, [0]]
        grid = np.zeros((1, self.resolution, self.resolution, self.resolution), dtype=np.float32)
        for index, point in enumerate(points):
            ijk = np.floor(point / self.voxel_size).astype(int)
            if np.all((0 <= ijk) & (ijk < self.resolution)):
                grid[(0, *ijk)] = distances[index, 0]
        return grid

    def get_cloud(self) -> Any:
        return self._volume.extract_point_cloud()


def build_tsdf_volume(
    depth_m: np.ndarray,
    intrinsics: Any,
    T_camera_task: np.ndarray,
    *,
    vgn_root: Path | str | None = None,
    workspace_size_m: float = OFFICIAL_WORKSPACE_SIZE_M,
    resolution: int = OFFICIAL_RESOLUTION,
    depth_trunc_m: float = OFFICIAL_DEPTH_TRUNC_M,
    preset: str = "official",
    logger: logging.Logger | None = None,
) -> Any:
    """Integrate one metric depth image using task-to-camera extrinsics."""

    logger = logger or LOGGER
    if preset != "official":
        raise ValueError("only the official VGN preset is supported")
    size = float(workspace_size_m)
    resolution = int(resolution)
    depth_trunc_m = float(depth_trunc_m)
    if size <= 0 or resolution <= 0 or depth_trunc_m <= 0:
        raise ValueError("TSDF size, resolution, and depth truncation must be positive")
    if size != OFFICIAL_WORKSPACE_SIZE_M or resolution != OFFICIAL_RESOLUTION:
        logger.warning(
            "changing physical scale invalidates strict pretrained-model comparability: "
            "workspace_size_m=%s resolution=%s", size, resolution
        )
    depth = np.asarray(depth_m, dtype=np.float32)
    width, height, fx, fy, cx, cy = _intrinsic_values(intrinsics)
    if depth.shape != (height, width):
        raise ValueError(
            f"metric depth shape {depth.shape} does not match intrinsics {(height, width)}"
        )
    if not np.all(np.isfinite(depth)) or np.any(depth < 0):
        raise ValueError("metric depth must be finite and non-negative")
    transform = np.asarray(T_camera_task, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_camera_task must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("T_camera_task must be a homogeneous transform")

    ensure_official_vgn_path(vgn_root)
    from vgn.perception import CameraIntrinsic, TSDFVolume
    from vgn.utils.transform import Transform

    intrinsic = CameraIntrinsic(width, height, fx, fy, cx, cy)
    if depth_trunc_m == OFFICIAL_DEPTH_TRUNC_M:
        volume: Any = TSDFVolume(size, resolution)
    else:
        volume = _ConfigurableTSDFVolume(size, resolution, depth_trunc_m)
    volume.integrate(depth, intrinsic, Transform.from_matrix(transform))
    return volume


def build_tsdf_grid(
    depth_m: np.ndarray,
    intrinsics: Any,
    T_camera_task: np.ndarray,
    **kwargs: Any,
) -> TSDFBuildResult:
    """Build a TSDF and return both its official-shaped grid and volume handle."""

    depth_trunc_m = float(kwargs.get("depth_trunc_m", OFFICIAL_DEPTH_TRUNC_M))
    volume = build_tsdf_volume(depth_m, intrinsics, T_camera_task, **kwargs)
    grid = validate_tsdf_grid(tsdf_grid_from_volume(volume))
    implementation = (
        "official_vgn.perception.TSDFVolume"
        if depth_trunc_m == OFFICIAL_DEPTH_TRUNC_M
        else "equivalent_open3d_uniform_tsdf"
    )
    return TSDFBuildResult(
        grid=grid,
        volume=volume,
        voxel_size_m=float(volume.voxel_size),
        depth_trunc_m=depth_trunc_m,
        implementation=implementation,
    )


def tsdf_grid_from_volume(volume: Any) -> np.ndarray:
    """Extract an upstream-compatible grid under both NumPy 1.x and 2.x.

    CoRL 2020 ``TSDFVolume.get_grid`` assigns a length-one array into a scalar
    grid element.  NumPy 2 rejects that formerly tolerated assignment.  The
    calculation below is otherwise identical and reads the same official
    Open3D volume; it does not alter TSDF values or voxel coordinates.
    """

    resolution = int(volume.resolution)
    voxel_size = float(volume.voxel_size)
    open3d_volume = getattr(volume, "_volume", None)
    if open3d_volume is None:
        return np.asarray(volume.get_grid())
    cloud = open3d_volume.extract_voxel_point_cloud()
    points = np.asarray(cloud.points)
    distances = np.asarray(cloud.colors)[:, 0]
    grid = np.zeros((1, resolution, resolution, resolution), dtype=np.float32)
    for point, distance in zip(points, distances):
        ijk = np.floor(point / voxel_size).astype(int)
        if np.all((0 <= ijk) & (ijk < resolution)):
            grid[(0, *ijk)] = float(distance)
    return grid


def _run_network(tsdf: np.ndarray, net: Any, device: Any) -> tuple[np.ndarray, ...]:
    import torch

    tensor = torch.from_numpy(tsdf).unsqueeze(0).to(device)
    with torch.no_grad():
        qual, rot, width = net(tensor)
    return (
        qual.detach().cpu().squeeze().numpy(),
        rot.detach().cpu().squeeze().numpy(),
        width.detach().cpu().squeeze().numpy(),
    )


def predict_official(
    tsdf_vol: np.ndarray,
    net: Any,
    device: str | Any,
    *,
    logger: logging.Logger | None = None,
) -> PredictionResult:
    """Run official VGN inference, retrying on CPU only after an MPS failure."""

    import torch

    logger = logger or LOGGER
    tsdf = validate_tsdf_grid(tsdf_vol)
    requested = str(device)
    torch_device = torch.device(device)
    try:
        qual, rot, width = _run_network(tsdf, net, torch_device)
        used = torch_device.type
        reason = None
    except Exception as error:
        if torch_device.type != "mps":
            raise
        reason = (
            "MPS inference failed; falling back to CPU: "
            f"{type(error).__name__}: {error}"
        )
        logger.warning(reason)
        cpu = torch.device("cpu")
        net = net.to(cpu)
        net.eval()
        qual, rot, width = _run_network(tsdf, net, cpu)
        used = "cpu"
    if qual.shape != (40, 40, 40):
        raise VGNAdapterError(f"official quality output has invalid shape {qual.shape}")
    if rot.shape != (4, 40, 40, 40):
        raise VGNAdapterError(f"official orientation output has invalid shape {rot.shape}")
    if width.shape != (40, 40, 40):
        raise VGNAdapterError(f"official width output has invalid shape {width.shape}")
    return PredictionResult(qual, rot, width, requested, used, reason)


def predict(tsdf_vol: np.ndarray, net: Any, device: str | Any) -> tuple[np.ndarray, ...]:
    """Signature-compatible strict wrapper around upstream ``predict``."""

    result = predict_official(tsdf_vol, net, device)
    return result.qual_vol, result.rot_vol, result.width_vol


def _validate_prediction_volumes(
    qual_vol: np.ndarray, rot_vol: np.ndarray, width_vol: np.ndarray
) -> None:
    if np.asarray(qual_vol).shape != (40, 40, 40):
        raise ValueError(f"quality volume must be 40x40x40, got {np.shape(qual_vol)}")
    if np.asarray(rot_vol).shape != (4, 40, 40, 40):
        raise ValueError(f"rotation volume must be 4x40x40x40, got {np.shape(rot_vol)}")
    if np.asarray(width_vol).shape != (40, 40, 40):
        raise ValueError(f"width volume must be 40x40x40, got {np.shape(width_vol)}")


def process(
    tsdf_vol: np.ndarray,
    qual_vol: np.ndarray,
    rot_vol: np.ndarray,
    width_vol: np.ndarray,
    gaussian_filter_sigma: float = OFFICIAL_GAUSSIAN_FILTER_SIGMA,
    min_width: float = OFFICIAL_MIN_WIDTH_VOXELS,
    max_width: float = OFFICIAL_MAX_WIDTH_VOXELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official VGN processing order with only the SciPy namespace updated."""

    # Do not cast here: upstream only squeezes the input.  In particular, a
    # float64 value immediately adjacent to 0.5 must retain upstream's exact
    # surface-mask classification.  ``validate_tsdf_grid`` remains strict for
    # the network input boundary.
    tsdf = np.asarray(tsdf_vol)
    if tsdf.shape != OFFICIAL_TSDF_INPUT_SHAPE:
        raise ValueError(
            f"VGN TSDF input must have shape {OFFICIAL_TSDF_INPUT_SHAPE}, got {tsdf.shape}"
        )
    tsdf = tsdf.squeeze()
    _validate_prediction_volumes(qual_vol, rot_vol, width_vol)

    quality = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )
    outside_voxels = tsdf > 0.5
    inside_voxels = np.logical_and(1e-3 < tsdf, tsdf < 0.5)
    valid_voxels = ndimage.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    quality[valid_voxels == False] = 0.0  # noqa: E712 - mirrors upstream exactly.
    quality[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0
    return quality, rot_vol, width_vol


def process_official(
    tsdf_vol: np.ndarray,
    qual_vol: np.ndarray,
    rot_vol: np.ndarray,
    width_vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Named entry point locked to all official post-processing parameters."""

    return process(
        tsdf_vol,
        qual_vol,
        rot_vol,
        width_vol,
        gaussian_filter_sigma=OFFICIAL_GAUSSIAN_FILTER_SIGMA,
        min_width=OFFICIAL_MIN_WIDTH_VOXELS,
        max_width=OFFICIAL_MAX_WIDTH_VOXELS,
    )


def select_index(
    qual_vol: np.ndarray,
    rot_vol: np.ndarray,
    width_vol: np.ndarray,
    index: Sequence[int],
) -> tuple[Any, float]:
    """Decode one upstream voxel-space grasp, including SciPy quaternion semantics."""

    ensure_official_vgn_path()
    from vgn.grasp import Grasp
    from vgn.utils.transform import Rotation, Transform

    i, j, k = (int(value) for value in index)
    score = qual_vol[i, j, k]
    orientation = Rotation.from_quat(rot_vol[:, i, j, k])
    position = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(orientation, position), width), score


def select(
    qual_vol: np.ndarray,
    rot_vol: np.ndarray,
    width_vol: np.ndarray,
    threshold: float = OFFICIAL_QUALITY_THRESHOLD,
    max_filter_size: int = OFFICIAL_MAX_FILTER_SIZE,
) -> tuple[list[Any], list[float]]:
    """Mirror upstream thresholding, NMS, iteration order, and voxel decoding."""

    _validate_prediction_volumes(qual_vol, rot_vol, width_vol)
    # Upstream mutates this input at thresholding; retain that behavior so the
    # regression comparison catches future drift.
    qual_vol[qual_vol < threshold] = 0.0
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    selected_quality = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(selected_quality, 1.0, 0.0)
    grasps: list[Any] = []
    scores: list[float] = []
    for index in np.argwhere(mask):
        grasp, score = select_index(selected_quality, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)
    return grasps, scores


def _candidate_from_voxel_grasp(
    grasp: Any,
    score: float,
    official_selection_index: int,
    voxel_size_m: float,
) -> Candidate:
    position_voxels = np.asarray(grasp.pose.translation, dtype=np.float64)
    ijk = tuple(int(value) for value in position_voxels)
    position_task = position_voxels * voxel_size_m
    quaternion = np.asarray(grasp.pose.rotation.as_quat(), dtype=np.float64)
    rotation = np.asarray(grasp.pose.rotation.as_matrix(), dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = position_task
    return Candidate(
        official_selection_index=official_selection_index,
        raw_order_index=official_selection_index,
        vgn_quality=float(score),
        voxel_index_ijk=ijk,
        position_task_m=position_task,
        quaternion_task_xyzw=quaternion,
        rotation_task_3x3=rotation,
        width_m=float(grasp.width) * voxel_size_m,
        T_task_grasp=pose,
    )


def decode_candidates(
    grasps: Sequence[Any],
    scores: Sequence[float],
    *,
    voxel_size_m: float = OFFICIAL_VOXEL_SIZE_M,
) -> list[Candidate]:
    """Convert official voxel-space grasps to metres exactly as upstream does."""

    if len(grasps) != len(scores):
        raise ValueError("grasp and score counts differ")
    voxel_size_m = float(voxel_size_m)
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be finite and positive")
    return [
        _candidate_from_voxel_grasp(grasp, score, index, voxel_size_m)
        for index, (grasp, score) in enumerate(zip(grasps, scores))
    ]


def select_official_candidates(
    qual_vol: np.ndarray,
    rot_vol: np.ndarray,
    width_vol: np.ndarray,
    *,
    voxel_size_m: float = OFFICIAL_VOXEL_SIZE_M,
) -> list[Candidate]:
    """Apply official threshold/NMS and decode candidates in original order."""

    grasps, scores = select(
        qual_vol,
        rot_vol,
        width_vol,
        threshold=OFFICIAL_QUALITY_THRESHOLD,
        max_filter_size=OFFICIAL_MAX_FILTER_SIZE,
    )
    return decode_candidates(grasps, scores, voxel_size_m=voxel_size_m)


def run_official_postprocessing(
    tsdf_vol: np.ndarray,
    qual_vol: np.ndarray,
    rot_vol: np.ndarray,
    width_vol: np.ndarray,
    *,
    voxel_size_m: float = OFFICIAL_VOXEL_SIZE_M,
) -> PostprocessingResult:
    """Run the immutable official process/select chain without random shuffling."""

    quality, rotation, width = process_official(tsdf_vol, qual_vol, rot_vol, width_vol)
    candidates = select_official_candidates(
        quality.copy(), rotation, width, voxel_size_m=voxel_size_m
    )
    return PostprocessingResult(quality, rotation, width, tuple(candidates))


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px < 0:
        raise ValueError("mask dilation radius cannot be negative")
    if radius_px == 0:
        return mask.copy()
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    structure = xx * xx + yy * yy <= radius_px * radius_px
    return ndimage.binary_dilation(mask, structure=structure)


def filter_target_candidates(
    candidates: Iterable[Candidate],
    *,
    intrinsics: Any,
    raw_target_mask: np.ndarray,
    dilated_target_mask: np.ndarray | None = None,
    target_mask_dilation_px: int = 3,
    depth_m: np.ndarray | None = None,
    target_points_camera: np.ndarray | None = None,
    T_camera_task: np.ndarray,
) -> tuple[list[Candidate], list[Candidate]]:
    """Annotate every candidate and accept only centres in the dilated mask.

    Distance and depth-consistency values are diagnostics only.  They never
    modify ``vgn_quality`` and are not acceptance conditions.
    """

    width, height, fx, fy, cx, cy = _intrinsic_values(intrinsics)
    raw = np.asarray(raw_target_mask, dtype=bool)
    if raw.shape != (height, width):
        raise ValueError(f"raw target mask shape {raw.shape} != {(height, width)}")
    if dilated_target_mask is None:
        dilated = _dilate_mask(raw, int(target_mask_dilation_px))
    else:
        dilated = np.asarray(dilated_target_mask, dtype=bool)
        if dilated.shape != raw.shape:
            raise ValueError("raw and dilated target mask shapes differ")
    depth = None if depth_m is None else np.asarray(depth_m, dtype=np.float32)
    if depth is not None and depth.shape != raw.shape:
        raise ValueError("depth and target mask shapes differ")
    target_points = (
        np.empty((0, 3), dtype=np.float64)
        if target_points_camera is None
        else np.asarray(target_points_camera, dtype=np.float64)
    )
    if target_points.ndim != 2 or target_points.shape[1:] != (3,):
        raise ValueError("target_points_camera must have shape Nx3")
    target_points = target_points[np.all(np.isfinite(target_points), axis=1)]
    tree = cKDTree(target_points) if len(target_points) else None

    camera_from_task = np.asarray(T_camera_task, dtype=np.float64)
    if camera_from_task.shape != (4, 4) or not np.all(np.isfinite(camera_from_task)):
        raise ValueError("T_camera_task must be a finite 4x4 matrix")

    annotated: list[Candidate] = []
    accepted: list[Candidate] = []
    for candidate in candidates:
        camera_pose = camera_from_task @ candidate.T_task_grasp
        position = camera_pose[:3, 3]
        positive_z = bool(np.isfinite(position[2]) and position[2] > 0.0)
        projected = None
        in_image = False
        raw_inside = False
        dilated_inside = False
        pixel_u = pixel_v = -1
        if positive_z:
            u = fx * position[0] / position[2] + cx
            v = fy * position[1] / position[2] + cy
            projected = np.asarray([u, v], dtype=np.float64)
            if np.all(np.isfinite(projected)):
                pixel_u, pixel_v = int(np.rint(u)), int(np.rint(v))
                in_image = 0 <= pixel_u < width and 0 <= pixel_v < height
                if in_image:
                    raw_inside = bool(raw[pixel_v, pixel_u])
                    dilated_inside = bool(dilated[pixel_v, pixel_u])

        nearest_distance = None
        if tree is not None and np.all(np.isfinite(position)):
            nearest_distance = float(tree.query(position, k=1)[0])
        depth_difference = None
        if depth is not None and in_image:
            observed_depth = float(depth[pixel_v, pixel_u])
            if np.isfinite(observed_depth) and observed_depth > 0.0:
                depth_difference = abs(float(position[2]) - observed_depth)

        from scipy.spatial.transform import Rotation

        camera_rotation = np.asarray(camera_pose[:3, :3], dtype=np.float64)
        camera_quaternion = Rotation.from_matrix(camera_rotation).as_quat()
        item = replace(
            candidate,
            position_camera_m=position.copy(),
            quaternion_camera_xyzw=np.asarray(camera_quaternion, dtype=np.float64),
            T_camera_grasp=camera_pose,
            projected_uv=projected,
            inside_raw_target_mask=raw_inside,
            inside_dilated_target_mask=dilated_inside,
            nearest_target_point_distance_m=nearest_distance,
            projected_depth_difference_m=depth_difference,
            positive_camera_z=positive_z,
            projection_in_image=in_image,
            target_filter_accepted=dilated_inside,
        )
        annotated.append(item)
        if item.target_filter_accepted:
            accepted.append(item)
    return annotated, accepted


def sort_candidates_by_quality(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Stable descending official-quality order, tied by upstream selection index."""

    ordered = sorted(
        candidates,
        key=lambda item: (-float(item.vgn_quality), int(item.official_selection_index)),
    )
    return [replace(item, score_rank=rank) for rank, item in enumerate(ordered, start=1)]


def select_candidate(
    candidates: Iterable[Candidate],
    *,
    policy: str = "highest_vgn_quality",
    seed: int = 42,
) -> Candidate | None:
    """Apply one explicitly named execution-selection policy."""

    items = list(candidates)
    if not items:
        return None
    if policy == "highest_vgn_quality":
        return sort_candidates_by_quality(items)[0]
    raw_order = sorted(items, key=lambda item: item.raw_order_index)
    if policy == "official_sim_random":
        index = int(np.random.RandomState(int(seed)).permutation(len(raw_order))[0])
        return raw_order[index]
    if policy == "official_panda_highest_z":
        # Panda's upstream np.argmax returns the first maximum.  Raw official
        # selection order is the deterministic tie breaker here.
        heights = np.asarray([item.position_task_m[2] for item in raw_order])
        return raw_order[int(np.argmax(heights))]
    raise ValueError(
        "selection policy must be highest_vgn_quality, official_sim_random, "
        "or official_panda_highest_z"
    )


def runtime_metadata(
    *,
    vgn_root: Path | str | None = None,
    weights_path: Path | str | None = None,
) -> dict[str, Any]:
    """Collect reproducibility facts without loading a model or random weights."""

    import scipy
    import torch

    metadata: dict[str, Any] = {
        "repository_url": OFFICIAL_VGN_REPOSITORY,
        "repository_branch": OFFICIAL_VGN_BRANCH,
        "repository_commit": official_checkout_commit(vgn_root),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "workspace_size_m": OFFICIAL_WORKSPACE_SIZE_M,
        "resolution": OFFICIAL_RESOLUTION,
        "voxel_size_m": OFFICIAL_VOXEL_SIZE_M,
        "quality_threshold": OFFICIAL_QUALITY_THRESHOLD,
    }
    try:
        import open3d

        metadata["open3d_version"] = open3d.__version__
    except ImportError:
        metadata["open3d_version"] = None
    if weights_path is not None:
        checkpoint = Path(weights_path).expanduser().resolve()
        metadata["checkpoint_path"] = str(checkpoint)
        metadata["checkpoint_sha256"] = checkpoint_sha256(checkpoint)
    return metadata


__all__ = [
    "Candidate",
    "DeviceSelection",
    "OFFICIAL_DEPTH_TRUNC_M",
    "OFFICIAL_FINGER_DEPTH_M",
    "OFFICIAL_GAUSSIAN_FILTER_SIGMA",
    "OFFICIAL_MAX_FILTER_SIZE",
    "OFFICIAL_MAX_WIDTH_VOXELS",
    "OFFICIAL_MIN_WIDTH_VOXELS",
    "OFFICIAL_PRESET",
    "OFFICIAL_QUALITY_THRESHOLD",
    "OFFICIAL_RESOLUTION",
    "OFFICIAL_TABLE_HEIGHT_M",
    "OFFICIAL_TSDF_INPUT_SHAPE",
    "OFFICIAL_VGN_BRANCH",
    "OFFICIAL_VGN_COMMIT",
    "OFFICIAL_VGN_REPOSITORY",
    "OFFICIAL_VGN_ROOT",
    "OFFICIAL_VOXEL_SIZE_M",
    "OFFICIAL_WORKSPACE_SIZE_M",
    "OfficialVGNPreset",
    "PostprocessingResult",
    "PredictionResult",
    "SCORE_SOURCE",
    "TSDFBuildResult",
    "VGNAdapterError",
    "build_tsdf_grid",
    "build_tsdf_volume",
    "checkpoint_sha256",
    "decode_candidates",
    "ensure_official_vgn_path",
    "filter_target_candidates",
    "load_official_network",
    "official_checkout_commit",
    "predict",
    "predict_official",
    "process",
    "process_official",
    "resolve_device",
    "resolve_device_info",
    "run_official_postprocessing",
    "runtime_metadata",
    "select",
    "select_candidate",
    "select_index",
    "select_official_candidates",
    "sort_candidates_by_quality",
    "tsdf_grid_from_volume",
    "validate_tsdf_grid",
]
