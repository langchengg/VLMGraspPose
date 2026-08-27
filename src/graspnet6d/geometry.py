"""Explicit geometry conventions shared by GraspNet and frozen VGN outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Hypothesis from the published conventions: GraspNet axes are
# (+X approach, +Y closing, +Z height), while VGN uses (+Z approach,
# +Y closing).  The columns below express [VGN +Z, VGN +Y, VGN -X].
# This must remain labelled unvalidated until checked against real GraspNet
# annotations, rendered grippers, and evaluator parity.
VGN_TO_GRASPNET_AXIS_MAP: NDArray[np.float64] = np.array(
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
VGN_TO_GRASPNET_AXIS_MAP_NAME = "vgn_z_y_minus_x_to_graspnet_x_y_z"
VGN_TO_GRASPNET_AXIS_MAP_STATUS = "unvalidated_hypothesis"


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        scalars = np.asarray((self.fx, self.fy, self.cx, self.cy), dtype=np.float64)
        if not np.isfinite(scalars).all():
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if self.width is not None and self.width <= 0:
            raise ValueError("image width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("image height must be positive")

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def as_rotation_matrix(rotation: ArrayLike, *, atol: float = 1e-6) -> NDArray[np.float64]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("rotation must be finite")
    error = np.max(np.abs(matrix.T @ matrix - np.eye(3)))
    determinant = float(np.linalg.det(matrix))
    if error > atol or abs(determinant - 1.0) > atol:
        raise ValueError(
            "rotation must be in SO(3): "
            f"orthogonality_error={error:.3g}, determinant={determinant:.9g}"
        )
    return matrix


def as_translation(translation: ArrayLike) -> NDArray[np.float64]:
    vector = np.asarray(translation, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"translation must have shape (3,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("translation must be finite")
    return vector


def make_transform(rotation: ArrayLike, translation: ArrayLike) -> NDArray[np.float64]:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = as_rotation_matrix(rotation)
    transform[:3, 3] = as_translation(translation)
    return transform


def as_transform(transform: ArrayLike, *, atol: float = 1e-6) -> NDArray[np.float64]:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("transform must be finite")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=atol, rtol=0.0):
        raise ValueError("homogeneous transform must end in [0, 0, 0, 1]")
    as_rotation_matrix(matrix[:3, :3], atol=atol)
    return matrix


def invert_transform(transform: ArrayLike) -> NDArray[np.float64]:
    matrix = as_transform(transform)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def compose_transforms(*transforms: ArrayLike) -> NDArray[np.float64]:
    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        result = result @ as_transform(transform)
    return as_transform(result)


def transform_points(transform: ArrayLike, points: ArrayLike) -> NDArray[np.float64]:
    matrix = as_transform(transform)
    values = np.asarray(points, dtype=np.float64)
    if values.shape == (3,):
        if not np.isfinite(values).all():
            raise ValueError("points must be finite")
        return matrix[:3, :3] @ values + matrix[:3, 3]
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError(f"points must end in dimension 3, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def transform_pose(
    transform_target_source: ArrayLike,
    rotation_source: ArrayLike,
    translation_source: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    transform = as_transform(transform_target_source)
    rotation = as_rotation_matrix(transform[:3, :3] @ as_rotation_matrix(rotation_source))
    translation = transform_points(transform, as_translation(translation_source))
    return rotation, translation


def backproject_pixels(
    pixels_uv: ArrayLike,
    depth_m: ArrayLike,
    intrinsics: CameraIntrinsics,
) -> NDArray[np.float64]:
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    depths = np.asarray(depth_m, dtype=np.float64)
    if pixels.shape == (2,):
        if depths.ndim > 0 and depths.size != 1:
            raise ValueError("a single pixel requires a scalar depth")
        pixels = pixels.reshape(1, 2)
        depths = np.asarray([float(depths.reshape(-1)[0])], dtype=np.float64)
        squeeze = True
    else:
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError(f"pixels must have shape (N, 2), got {pixels.shape}")
        depths = depths.reshape(-1)
        if depths.shape[0] != pixels.shape[0]:
            raise ValueError("pixel and depth counts differ")
        squeeze = False
    if not np.isfinite(pixels).all() or not np.isfinite(depths).all():
        raise ValueError("pixels and depths must be finite")
    if np.any(depths <= 0.0):
        raise ValueError("depths must be positive metres")
    x = (pixels[:, 0] - intrinsics.cx) * depths / intrinsics.fx
    y = (pixels[:, 1] - intrinsics.cy) * depths / intrinsics.fy
    points = np.column_stack((x, y, depths))
    return points[0] if squeeze else points


def project_points(
    points_camera_m: ArrayLike,
    intrinsics: CameraIntrinsics,
    *,
    require_in_image: bool = False,
) -> NDArray[np.float64]:
    points = np.asarray(points_camera_m, dtype=np.float64)
    if points.shape == (3,):
        points = points.reshape(1, 3)
        squeeze = True
    else:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        squeeze = False
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if np.any(points[:, 2] <= 0.0):
        raise ValueError("camera-frame Z must be positive")
    pixels = np.column_stack(
        (
            intrinsics.fx * points[:, 0] / points[:, 2] + intrinsics.cx,
            intrinsics.fy * points[:, 1] / points[:, 2] + intrinsics.cy,
        )
    )
    if require_in_image:
        if intrinsics.width is None or intrinsics.height is None:
            raise ValueError("image dimensions are required for bounds validation")
        inside = (
            (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < intrinsics.width)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < intrinsics.height)
        )
        if not inside.all():
            raise ValueError("projected point falls outside the image")
    return pixels[0] if squeeze else pixels


def rotation_geodesic_deg(rotation_a: ArrayLike, rotation_b: ArrayLike) -> float:
    relative = as_rotation_matrix(rotation_a).T @ as_rotation_matrix(rotation_b)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def vgn_rotation_to_graspnet(rotation: ArrayLike) -> NDArray[np.float64]:
    """Apply the explicit, still-unvalidated fixed gripper-axis hypothesis."""

    return as_rotation_matrix(as_rotation_matrix(rotation) @ VGN_TO_GRASPNET_AXIS_MAP)


def vgn_to_graspnet_provenance(*, source: str) -> dict[str, Any]:
    if not source.strip():
        raise ValueError("conversion source must be non-empty")
    return {
        "mapping_name": VGN_TO_GRASPNET_AXIS_MAP_NAME,
        "mapping_status": VGN_TO_GRASPNET_AXIS_MAP_STATUS,
        "matrix_columns": VGN_TO_GRASPNET_AXIS_MAP.tolist(),
        "source": source,
    }
