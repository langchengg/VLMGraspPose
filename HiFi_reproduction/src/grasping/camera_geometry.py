"""Metric camera geometry for fixed-approach planar grasps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


T_CAMERA_GRASP_FIXED_APPROACH_KEY = "T_camera_grasp_fixed_approach"


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class CameraIntrinsicsData:
    """Validated pinhole intrinsics in an explicitly named camera frame."""

    frame: str
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float
    height: int
    width: int

    def __post_init__(self) -> None:
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("camera frame must be a non-empty string")
        for name in ("fx", "fy", "cx", "cy", "skew"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "width", int(self.width))
        if self.height <= 0 or self.width <= 0:
            raise ValueError("image height and width must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CameraIntrinsicsData":
        return cls(
            frame=str(values["frame"]),
            fx=values["fx"],
            fy=values["fy"],
            cx=values["cx"],
            cy=values["cy"],
            skew=values.get("skew", 0.0),
            height=values["height"],
            width=values["width"],
        )

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, self.skew, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_perception(self) -> Any:
        """Construct the official camera-intrinsics type used by ``gqcnn``.

        The pinned GQ-CNN code imports this type from ``autolab_core`` (the
        modern ``perception`` package does not re-export it).
        """

        from autolab_core import CameraIntrinsics  # Imported lazily by design.

        return CameraIntrinsics(
            self.frame,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.skew,
            self.height,
            self.width,
        )

    def save_intr(self, path: Path | str) -> Path:
        """Export an official ``.intr`` file via the installed perception API."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        intrinsics = self.to_perception()
        save = getattr(intrinsics, "save", None)
        if not callable(save):
            raise RuntimeError("installed CameraIntrinsics object has no save method")
        save(str(destination))
        return destination


def depth_mm_to_meters(
    depth_mm: np.ndarray,
    *,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
) -> np.ndarray:
    """Convert millimetres to float32 metres, mapping invalid samples to zero."""

    minimum = _finite_scalar(min_depth_m, "min_depth_m")
    if minimum < 0:
        raise ValueError("min_depth_m cannot be negative")
    maximum = None if max_depth_m is None else _finite_scalar(max_depth_m, "max_depth_m")
    if maximum is not None and maximum <= minimum:
        raise ValueError("max_depth_m must be greater than min_depth_m")
    source = np.asarray(depth_mm)
    if source.ndim != 2:
        raise ValueError(f"depth image must be HxW, got {source.shape}")
    depth_m = source.astype(np.float32, copy=True) * np.float32(0.001)
    valid = np.isfinite(depth_m) & (depth_m > minimum)
    if maximum is not None:
        valid &= depth_m <= maximum
    depth_m[~valid] = np.float32(0.0)
    return depth_m


def normalize_depth_meters(
    depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
) -> np.ndarray:
    """Copy metric depth to float32 and normalize invalid samples to zero."""

    minimum = _finite_scalar(min_depth_m, "min_depth_m")
    maximum = None if max_depth_m is None else _finite_scalar(max_depth_m, "max_depth_m")
    if minimum < 0 or (maximum is not None and maximum <= minimum):
        raise ValueError("invalid depth range")
    source = np.asarray(depth_m)
    if source.ndim != 2:
        raise ValueError(f"depth image must be HxW, got {source.shape}")
    output = source.astype(np.float32, copy=True)
    valid = np.isfinite(output) & (output > minimum)
    if maximum is not None:
        valid &= output <= maximum
    output[~valid] = np.float32(0.0)
    return output


def backproject_pixel(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsicsData,
) -> np.ndarray:
    """Back-project one pixel to ``(X, Y, Z)`` in the camera frame, in metres."""

    u = _finite_scalar(u, "u")
    v = _finite_scalar(v, "v")
    z = _finite_scalar(depth_m, "depth_m")
    if z <= 0:
        raise ValueError("depth_m must be positive")
    y_normalized = (v - intrinsics.cy) / intrinsics.fy
    x_normalized = (u - intrinsics.cx - intrinsics.skew * y_normalized) / intrinsics.fx
    return np.array([x_normalized * z, y_normalized * z, z], dtype=np.float32)


def backproject_pixels(
    uv: np.ndarray,
    depth_m: np.ndarray | float,
    intrinsics: CameraIntrinsicsData,
) -> np.ndarray:
    """Vectorized pixel back-projection with output shape ``(..., 3)``."""

    pixels = np.asarray(uv, dtype=np.float64)
    if pixels.shape[-1:] != (2,):
        raise ValueError("uv must have shape (..., 2)")
    depth = np.asarray(depth_m, dtype=np.float64)
    try:
        depth = np.broadcast_to(depth, pixels.shape[:-1])
    except ValueError as error:
        raise ValueError("depth_m cannot be broadcast to uv") from error
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(depth)) or np.any(depth <= 0):
        raise ValueError("pixels must be finite and depths must be finite and positive")
    yn = (pixels[..., 1] - intrinsics.cy) / intrinsics.fy
    xn = (pixels[..., 0] - intrinsics.cx - intrinsics.skew * yn) / intrinsics.fx
    return np.stack((xn * depth, yn * depth, depth), axis=-1).astype(np.float32)


def _pixels_per_meter(
    depth_m: float,
    intrinsics: CameraIntrinsicsData,
    angle_rad: float,
) -> float:
    depth = _finite_scalar(depth_m, "depth_m")
    angle = _finite_scalar(angle_rad, "angle_rad")
    if depth <= 0:
        raise ValueError("depth_m must be positive")
    # Projection scale along a unit vector in the image plane.
    return float(
        np.hypot(intrinsics.fx * np.cos(angle), intrinsics.fy * np.sin(angle)) / depth
    )


def width_m_to_pixels(
    width_m: float,
    depth_m: float,
    intrinsics: CameraIntrinsicsData,
    angle_rad: float = 0.0,
) -> float:
    width = _finite_scalar(width_m, "width_m")
    if width < 0:
        raise ValueError("width_m cannot be negative")
    return width * _pixels_per_meter(depth_m, intrinsics, angle_rad)


def width_pixels_to_meters(
    width_px: float,
    depth_m: float,
    intrinsics: CameraIntrinsicsData,
    angle_rad: float = 0.0,
) -> float:
    width = _finite_scalar(width_px, "width_px")
    if width < 0:
        raise ValueError("width_px cannot be negative")
    return width / _pixels_per_meter(depth_m, intrinsics, angle_rad)


def grasp_endpoints_uv(
    center_uv: Sequence[float],
    width_px: float,
    angle_rad: float,
) -> np.ndarray:
    """Compute two jaw endpoints as ``[[u1,v1], [u2,v2]]``."""

    center = np.asarray(center_uv, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_uv must contain two finite values")
    width = _finite_scalar(width_px, "width_px")
    angle = _finite_scalar(angle_rad, "angle_rad")
    if width < 0:
        raise ValueError("width_px cannot be negative")
    offset = 0.5 * width * np.array([np.cos(angle), np.sin(angle)])
    return np.stack((center - offset, center + offset)).astype(np.float32)


def fixed_approach_pose(center_camera_xyz_m: Sequence[float], angle_rad: float) -> np.ndarray:
    """Return ``T_camera_grasp`` with approach constrained to camera ``+Z``.

    The grasp-frame x axis follows the planar jaw axis, its z axis is the fixed
    camera optical axis, and y completes a right-handed frame.  This is not a
    freely predicted 6-DoF pose.
    """

    center = np.asarray(center_camera_xyz_m, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_camera_xyz_m must contain three finite values")
    angle = _finite_scalar(angle_rad, "angle_rad")
    cosine, sine = np.cos(angle), np.sin(angle)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform[:3, 3] = center
    return transform


def named_fixed_approach_pose(
    center_camera_xyz_m: Sequence[float], angle_rad: float
) -> Mapping[str, np.ndarray]:
    """Return the fixed-approach transform under its required explicit name."""

    return {
        T_CAMERA_GRASP_FIXED_APPROACH_KEY: fixed_approach_pose(
            center_camera_xyz_m, angle_rad
        )
    }
