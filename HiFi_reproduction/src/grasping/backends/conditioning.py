"""Target-conditioned RGB-D preprocessing with reversible native coordinates.

The upstream GR-ConvNet and GG-CNN2 loaders use square inputs and per-image
zero centring.  This module preserves those semantics while making the target
conditioning and crop geometry explicit, deterministic, and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from ..common.geometry import CropTransform


ConditioningVariant = Literal["hard_mask", "dilated_crop"]


@dataclass(frozen=True, slots=True)
class ConditionedInput:
    """Model-ready arrays and their exact model-to-native transform."""

    rgb_chw: np.ndarray
    depth_chw: np.ndarray
    gate_map: np.ndarray
    probability_map: np.ndarray
    valid_depth_map: np.ndarray
    transform: CropTransform
    variant: ConditioningVariant
    mask_area_px: int
    valid_depth_fraction: float
    fill_depth_m: float


def resize_probability_to_native(
    probability: np.ndarray, native_shape: tuple[int, int]
) -> np.ndarray:
    """Resize a soft HiFi-CS probability map without binarising it."""

    value = np.asarray(probability, dtype=np.float32)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("probability must be a finite 2D array")
    if value.min(initial=0.0) < 0.0 or value.max(initial=0.0) > 1.0:
        raise ValueError("probability must be in [0, 1]")
    height, width = map(int, native_shape)
    if min(height, width) <= 0:
        raise ValueError("native_shape must be positive")
    if value.shape == (height, width):
        return np.array(value, copy=True)
    return cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR).astype(
        np.float32, copy=False
    )


def _square_bounds(
    mask: np.ndarray,
    *,
    variant: ConditioningVariant,
    dilation_fraction: float,
    minimum_side_px: int,
) -> tuple[float, float, float]:
    height, width = mask.shape
    if variant == "hard_mask":
        side = float(max(height, width))
        return (width - side) * 0.5, (height - side) * 0.5, side

    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("empty_mask")
    if not 0.0 <= float(dilation_fraction) <= 1.0:
        raise ValueError("dilation_fraction must be in [0, 1]")
    object_width = float(columns.max() - columns.min() + 1)
    object_height = float(rows.max() - rows.min() + 1)
    side = max(object_width, object_height) * (1.0 + 2.0 * dilation_fraction)
    side = max(side, float(minimum_side_px))
    side = min(side, float(max(height, width)))
    center_x = 0.5 * float(columns.min() + columns.max())
    center_y = 0.5 * float(rows.min() + rows.max())
    return center_x - side * 0.5, center_y - side * 0.5, side


def _warp_square(
    image: np.ndarray,
    *,
    crop_x: float,
    crop_y: float,
    crop_side: float,
    output_size: int,
    interpolation: int,
    border_value: float | tuple[float, ...],
) -> np.ndarray:
    scale = float(output_size) / float(crop_side)
    matrix = np.asarray(
        [[scale, 0.0, -crop_x * scale], [0.0, scale, -crop_y * scale]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (int(output_size), int(output_size)),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _depth_fill_value(depth_m: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid):
        raise ValueError("invalid_depth")
    kernel = np.ones((11, 11), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    local_background = valid & dilated & ~mask
    values = depth_m[local_background]
    if values.size < 16:
        values = depth_m[valid & mask]
    if values.size < 16:
        values = depth_m[valid]
    return float(np.median(values))


def normalise_depth_official(depth_m: np.ndarray) -> np.ndarray:
    """Match the official loaders: subtract image mean, then clip to [-1, 1]."""

    value = np.asarray(depth_m, dtype=np.float32)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("depth input must be finite and 2D before normalisation")
    return np.clip(value - np.float32(value.mean()), -1.0, 1.0).astype(
        np.float32, copy=False
    )


def normalise_rgb_official(rgb: np.ndarray) -> np.ndarray:
    """Match the official loaders: RGB / 255 followed by scalar mean subtraction."""

    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("RGB input must have shape (H, W, 3)")
    value = value.astype(np.float32) / np.float32(255.0)
    value -= np.float32(value.mean())
    return value


def condition_rgbd(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    binary_mask: np.ndarray,
    probability: np.ndarray,
    variant: ConditioningVariant,
    output_size: int = 300,
    dilation_fraction: float = 0.15,
    minimum_side_px: int = 64,
) -> ConditionedInput:
    """Create hard-mask or dilated-crop inputs without using any GT field."""

    rgb_value = np.asarray(rgb)
    depth_value = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(binary_mask).astype(bool)
    if rgb_value.ndim != 3 or rgb_value.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if depth_value.shape != rgb_value.shape[:2] or mask.shape != depth_value.shape:
        raise ValueError("RGB, depth, and binary mask shapes must agree")
    if output_size <= 0 or minimum_side_px <= 0:
        raise ValueError("output and minimum crop sizes must be positive")
    if not np.any(mask):
        raise ValueError("empty_mask")

    probability_native = resize_probability_to_native(probability, depth_value.shape)
    valid_depth = np.isfinite(depth_value) & (depth_value > 0.0)
    valid_target_fraction = float(np.count_nonzero(valid_depth & mask) / np.count_nonzero(mask))
    if not np.any(valid_depth & mask):
        raise ValueError("invalid_target_depth")
    fill_depth = _depth_fill_value(depth_value, mask)
    filled_depth = np.where(valid_depth, depth_value, fill_depth).astype(np.float32)

    if variant == "hard_mask":
        conditioned_rgb = np.where(mask[..., None], rgb_value, 0)
        conditioned_depth = np.where(mask, filled_depth, fill_depth).astype(np.float32)
    elif variant == "dilated_crop":
        conditioned_rgb = rgb_value
        conditioned_depth = filled_depth
    else:
        raise ValueError(f"unsupported conditioning variant: {variant}")

    crop_x, crop_y, crop_side = _square_bounds(
        mask,
        variant=variant,
        dilation_fraction=dilation_fraction,
        minimum_side_px=minimum_side_px,
    )
    rgb_crop = _warp_square(
        conditioned_rgb,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_side=crop_side,
        output_size=output_size,
        interpolation=cv2.INTER_LINEAR,
        border_value=(0.0, 0.0, 0.0),
    )
    depth_crop = _warp_square(
        conditioned_depth,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_side=crop_side,
        output_size=output_size,
        interpolation=cv2.INTER_LINEAR,
        border_value=fill_depth,
    ).astype(np.float32, copy=False)
    probability_crop = _warp_square(
        probability_native,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_side=crop_side,
        output_size=output_size,
        interpolation=cv2.INTER_LINEAR,
        border_value=0.0,
    ).astype(np.float32, copy=False)
    mask_crop = _warp_square(
        mask.astype(np.uint8),
        crop_x=crop_x,
        crop_y=crop_y,
        crop_side=crop_side,
        output_size=output_size,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
    ).astype(bool)
    valid_depth_crop = _warp_square(
        valid_depth.astype(np.uint8),
        crop_x=crop_x,
        crop_y=crop_y,
        crop_side=crop_side,
        output_size=output_size,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
    ).astype(bool)
    gate = np.clip(probability_crop, 0.0, 1.0)
    gate *= mask_crop.astype(np.float32)

    transform = CropTransform(
        crop_x=crop_x,
        crop_y=crop_y,
        crop_width=crop_side,
        crop_height=crop_side,
        model_width=output_size,
        model_height=output_size,
        native_width=rgb_value.shape[1],
        native_height=rgb_value.shape[0],
    )
    return ConditionedInput(
        rgb_chw=np.moveaxis(normalise_rgb_official(rgb_crop), -1, 0),
        depth_chw=normalise_depth_official(depth_crop)[None, ...],
        gate_map=gate,
        probability_map=probability_crop,
        valid_depth_map=valid_depth_crop,
        transform=transform,
        variant=variant,
        mask_area_px=int(np.count_nonzero(mask)),
        valid_depth_fraction=valid_target_fraction,
        fill_depth_m=fill_depth,
    )
