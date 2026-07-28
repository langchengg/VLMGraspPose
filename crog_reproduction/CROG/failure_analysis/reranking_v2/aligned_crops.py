from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


CROP_CHANNELS = (
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "relative_depth_m",
    "depth_valid",
    "predicted_soft_mask",
    "quality",
    "sin_2theta",
    "cos_2theta",
    "width_probability",
    "left_finger_template",
    "right_finger_template",
    "contact_template",
    "gripper_template",
)


def axial_angle_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def candidate_sampling_grid(
    candidate: dict[str, Any],
    output_size: int = 32,
    *,
    width_scale: float = 1.5,
    height_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized candidate crop pixels to source image x/y pixels."""
    size = int(output_size)
    if size <= 1:
        raise ValueError("output_size must be greater than one")
    theta = math.radians(axial_angle_deg(candidate["angle_deg"]))
    opening = np.asarray([math.cos(theta), -math.sin(theta)], dtype=np.float32)
    perpendicular = np.asarray([-opening[1], opening[0]], dtype=np.float32)
    half_width = max(float(candidate["width_px"]) * float(width_scale) / 2.0, 1.0)
    half_height = max(
        float(candidate["height_px"]) * float(height_scale) / 2.0, 1.0
    )
    normalized = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    vv, uu = np.meshgrid(normalized, normalized, indexing="ij")
    dx = uu * half_width * opening[0] + vv * half_height * perpendicular[0]
    dy = uu * half_width * opening[1] + vv * half_height * perpendicular[1]
    return (
        np.float32(float(candidate["cx"]) + dx),
        np.float32(float(candidate["cy"]) + dy),
    )


def _remap(
    array: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    interpolation: int,
    border_value: float | tuple[float, ...] = 0.0,
) -> np.ndarray:
    return cv2.remap(
        np.asarray(array),
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def gripper_templates(output_size: int = 32) -> dict[str, np.ndarray]:
    size = int(output_size)
    coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    vv, uu = np.meshgrid(coordinates, coordinates, indexing="ij")
    within_jaw_height = np.abs(vv) <= 0.5
    left = within_jaw_height & (uu >= -0.78) & (uu <= -0.58)
    right = within_jaw_height & (uu <= 0.78) & (uu >= 0.58)
    contact = within_jaw_height & (
        ((uu >= -0.60) & (uu <= -0.48))
        | ((uu <= 0.60) & (uu >= 0.48))
    )
    palm = (np.abs(vv) >= 0.42) & (np.abs(vv) <= 0.58) & (np.abs(uu) <= 0.78)
    gripper = left | right | palm
    return {
        "left": left.astype(np.float32),
        "right": right.astype(np.float32),
        "contact": contact.astype(np.float32),
        "gripper": gripper.astype(np.float32),
    }


def build_aligned_crop(
    candidate: dict[str, Any],
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    mask_probability: np.ndarray,
    quality: np.ndarray,
    sin_2theta: np.ndarray,
    cos_2theta: np.ndarray,
    width_probability: np.ndarray,
    output_size: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape = tuple(np.asarray(quality).shape)
    arrays = (mask_probability, sin_2theta, cos_2theta, width_probability)
    if np.asarray(rgb).shape[:2] != shape or any(
        np.asarray(value).shape != shape for value in arrays
    ):
        raise ValueError("RGB/map shapes do not match")
    map_x, map_y = candidate_sampling_grid(candidate, output_size)
    rgb_float = np.asarray(rgb, dtype=np.float32)
    if rgb_float.max(initial=0.0) > 1.0:
        rgb_float /= 255.0
    rgb_crop = _remap(
        rgb_float,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        border_value=(0.0, 0.0, 0.0),
    )
    map_crops = [
        _remap(
            np.asarray(value, dtype=np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
        )
        for value in (
            mask_probability,
            quality,
            sin_2theta,
            cos_2theta,
            width_probability,
        )
    ]
    missing_depth = depth_m is None or np.asarray(depth_m).shape != shape
    if missing_depth:
        relative_depth = np.zeros((output_size, output_size), dtype=np.float32)
        depth_valid = np.zeros_like(relative_depth)
        depth_median = None
    else:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        depth_crop = _remap(
            np.where(valid, depth, 0.0),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
        )
        valid_crop = _remap(
            valid.astype(np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        depth_median = (
            float(np.median(depth_crop[valid_crop])) if valid_crop.any() else None
        )
        relative_depth = np.zeros_like(depth_crop, dtype=np.float32)
        if depth_median is not None:
            relative_depth[valid_crop] = depth_crop[valid_crop] - depth_median
        depth_valid = valid_crop.astype(np.float32)
    templates = gripper_templates(output_size)
    channels = [
        rgb_crop[..., 0],
        rgb_crop[..., 1],
        rgb_crop[..., 2],
        relative_depth,
        depth_valid,
        *map_crops,
        templates["left"],
        templates["right"],
        templates["contact"],
        templates["gripper"],
    ]
    crop = np.stack(channels).astype(np.float32, copy=False)
    if crop.shape != (len(CROP_CHANNELS), output_size, output_size):
        raise AssertionError(f"unexpected crop shape: {crop.shape}")
    if not np.isfinite(crop).all():
        raise ValueError("aligned crop contains NaN or Inf")
    metadata = {
        "channels": CROP_CHANNELS,
        "output_size": int(output_size),
        "source_interpolation": {
            "rgb_and_soft_maps": "opencv_linear",
            "depth_and_validity": "opencv_nearest",
        },
        "depth_available": not missing_depth,
        "depth_valid_fraction": float(depth_valid.mean()),
        "local_depth_median_m": depth_median,
        "axial_angle_deg": axial_angle_deg(candidate["angle_deg"]),
    }
    return crop, metadata

