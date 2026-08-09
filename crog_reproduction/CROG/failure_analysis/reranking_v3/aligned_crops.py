from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .coordinate_mapping import sample_feature_roi, sample_original_roi


CROP_CHANNELS = (
    "rgb_r", "rgb_g", "rgb_b", "relative_depth_m", "depth_valid",
    "mask_raw", "mask_probability", "quality_raw", "quality_probability",
    "sin_2theta", "cos_2theta", "width_raw", "width_probability",
    "left_finger_template", "right_finger_template", "contact_template", "gripper_template",
)


def geometry_templates(size: int, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype)
    vv, uu = torch.meshgrid(axis, axis, indexing="ij")
    contact = ((uu.abs() >= 0.35) & (uu.abs() <= 0.75) & (vv.abs() <= 0.22)).to(dtype)
    left = ((uu <= -0.48) & (uu >= -0.85) & (vv.abs() <= 0.72)).to(dtype)
    right = ((uu >= 0.48) & (uu <= 0.85) & (vv.abs() <= 0.72)).to(dtype)
    gripper = ((vv.abs() <= 0.18) | (left.bool()) | (right.bool())).to(dtype)
    return torch.stack((left, right, contact, gripper), dim=0)


def build_fullchain_crop(
    candidate: dict[str, Any],
    *,
    rgb: torch.Tensor,
    depth_m: torch.Tensor,
    raw_heads: tuple[torch.Tensor, ...],
    forward_affine: np.ndarray,
    output_size: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one gripper-aligned crop without changing the candidate."""
    rgb_roi = sample_original_roi(rgb, candidate, output_size=output_size)
    depth_roi = sample_original_roi(depth_m, candidate, output_size=output_size, mode="nearest")
    valid = torch.isfinite(depth_roi) & (depth_roi > 0)
    valid_values = depth_roi[valid]
    median = valid_values.median() if valid_values.numel() else depth_roi.new_tensor(0.0)
    relative = torch.where(valid, depth_roi - median, torch.zeros_like(depth_roi))
    head_roi = torch.cat(
        [
            sample_feature_roi(head, candidate, forward_affine=forward_affine, output_size=output_size)
            for head in raw_heads
        ],
        dim=1,
    )
    mask_raw, quality_raw, sin_raw, cos_raw, width_raw = head_roi.unbind(dim=1)
    activated = torch.stack(
        (
            mask_raw, torch.sigmoid(mask_raw),
            quality_raw, torch.sigmoid(quality_raw),
            sin_raw, cos_raw,
            width_raw, torch.sigmoid(width_raw),
        ),
        dim=1,
    )
    templates = geometry_templates(output_size, device=rgb.device, dtype=rgb.dtype).unsqueeze(0)
    crop = torch.cat(
        (
            rgb_roi,
            relative,
            valid.to(rgb.dtype),
            activated,
            templates,
        ),
        dim=1,
    )
    if crop.shape[1] != len(CROP_CHANNELS):
        raise AssertionError(f"crop channel mismatch: {crop.shape}")
    array = crop.squeeze(0).float().cpu().numpy()
    return array, {
        "channels": list(CROP_CHANNELS),
        "output_size": int(output_size),
        "depth_valid_fraction": float(valid.float().mean().cpu()),
        "local_depth_median_m": float(median.cpu()) if valid_values.numel() else None,
        "align_corners": False,
        "padding_mode": "zeros",
        "candidate_identity_unchanged": True,
    }


def flip_axial_crop(crop: torch.Tensor) -> torch.Tensor:
    """Expected representation of the same axial grasp rotated by 180 degrees."""
    return torch.flip(crop, dims=(-2, -1))

