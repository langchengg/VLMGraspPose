from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def homogeneous_affine(matrix_2x3: np.ndarray | torch.Tensor) -> np.ndarray:
    value = np.asarray(matrix_2x3, dtype=np.float64)
    if value.shape != (2, 3):
        raise ValueError(f"affine must be [2,3], got {value.shape}")
    return np.vstack((value, np.asarray([0.0, 0.0, 1.0])))


def forward_from_inverse(inverse_2x3: np.ndarray | torch.Tensor) -> np.ndarray:
    return np.linalg.inv(homogeneous_affine(inverse_2x3))[:2].astype(np.float64)


def apply_affine_xy(points_xy: np.ndarray, affine_2x3: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    flat = points.reshape(-1, 2)
    homogeneous = np.concatenate((flat, np.ones((len(flat), 1))), axis=1)
    mapped = homogeneous @ np.asarray(affine_2x3, dtype=np.float64).T
    return mapped.reshape(*points.shape[:-1], 2)


def axial_radians(angle_deg: float) -> float:
    return math.radians(((float(angle_deg) + 90.0) % 180.0) - 90.0)


def axial_difference_deg(left: float | np.ndarray, right: float | np.ndarray):
    return np.abs((np.asarray(left) - np.asarray(right) + 90.0) % 180.0 - 90.0)


def candidate_original_grid(
    candidate: dict[str, Any],
    *,
    output_size: int,
    width_scale: float = 1.5,
    height_scale: float = 2.0,
) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, int(output_size), dtype=np.float64)
    vv, uu = np.meshgrid(axis, axis, indexing="ij")
    theta = axial_radians(candidate["angle_deg"])
    # Match CROG/V2 image convention: x right, y down, positive grasp angle
    # rotates the opening axis counter-clockwise in Cartesian coordinates.
    opening = np.asarray([math.cos(theta), -math.sin(theta)])
    perpendicular = np.asarray([-opening[1], opening[0]])
    half_width = max(float(candidate["width_px"]) * float(width_scale) / 2.0, 1.0)
    half_height = max(float(candidate["height_px"]) * float(height_scale) / 2.0, 1.0)
    center = np.asarray([float(candidate["cx"]), float(candidate["cy"])])
    return (
        center
        + uu[..., None] * half_width * opening
        + vv[..., None] * half_height * perpendicular
    )


def candidate_feature_grid(
    candidate: dict[str, Any],
    *,
    forward_affine: np.ndarray,
    model_input_shape: tuple[int, int] = (416, 416),
    output_size: int = 7,
    width_scale: float = 1.5,
    height_scale: float = 2.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an align_corners=False grid for any CROG feature resolution.

    A normalized feature-map location represents the same continuous position
    in the 416x416 model-input frame independent of the feature resolution.
    """
    original = candidate_original_grid(
        candidate,
        output_size=output_size,
        width_scale=width_scale,
        height_scale=height_scale,
    )
    model_xy = apply_affine_xy(original, forward_affine)
    input_h, input_w = map(int, model_input_shape)
    norm_x = 2.0 * (model_xy[..., 0] + 0.5) / input_w - 1.0
    norm_y = 2.0 * (model_xy[..., 1] + 0.5) / input_h - 1.0
    grid = np.stack((norm_x, norm_y), axis=-1)
    return torch.as_tensor(grid, device=device, dtype=dtype).unsqueeze(0)


def sample_feature_roi(
    feature: torch.Tensor,
    candidate: dict[str, Any],
    *,
    forward_affine: np.ndarray,
    output_size: int = 7,
    width_scale: float = 1.5,
    height_scale: float = 2.0,
) -> torch.Tensor:
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("feature ROI input must be [1,C,H,W]")
    grid = candidate_feature_grid(
        candidate,
        forward_affine=forward_affine,
        output_size=output_size,
        width_scale=width_scale,
        height_scale=height_scale,
        device=feature.device,
        dtype=feature.dtype,
    )
    try:
        return F.grid_sample(
            feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
    except RuntimeError:
        return F.grid_sample(
            feature.cpu(), grid.cpu(), mode="bilinear", padding_mode="zeros", align_corners=False
        ).to(feature.device)


def original_image_grid(
    candidate: dict[str, Any],
    *,
    image_shape: tuple[int, int],
    output_size: int,
    width_scale: float = 1.5,
    height_scale: float = 2.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    points = candidate_original_grid(
        candidate, output_size=output_size, width_scale=width_scale, height_scale=height_scale
    )
    height, width = map(int, image_shape)
    norm_x = 2.0 * (points[..., 0] + 0.5) / width - 1.0
    norm_y = 2.0 * (points[..., 1] + 0.5) / height - 1.0
    return torch.as_tensor(np.stack((norm_x, norm_y), -1), device=device, dtype=dtype).unsqueeze(0)


def sample_original_roi(
    image: torch.Tensor,
    candidate: dict[str, Any],
    *,
    output_size: int = 32,
    mode: str = "bilinear",
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("original image ROI input must be [1,C,H,W]")
    grid = original_image_grid(
        candidate,
        image_shape=(image.shape[-2], image.shape[-1]),
        output_size=output_size,
        device=image.device,
        dtype=image.dtype,
    )
    try:
        return F.grid_sample(image, grid, mode=mode, padding_mode="zeros", align_corners=False)
    except RuntimeError:
        return F.grid_sample(image.cpu(), grid.cpu(), mode=mode, padding_mode="zeros", align_corners=False).to(image.device)

