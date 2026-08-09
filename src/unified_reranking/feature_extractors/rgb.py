"""Candidate-aligned local RGB evidence shared by all three routes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import _axis, _corners, _outer_context_values, _region_values


def _rgb_patch(image: np.ndarray, point: np.ndarray, radius: int) -> np.ndarray:
    height, width = image.shape[:2]
    x, y = np.rint(point).astype(int)
    return image[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ]


def candidate_rgb_features(candidates: pd.DataFrame, rgb: np.ndarray) -> pd.DataFrame:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("RGB observation must be finite HxWx3")
    image = image.astype(np.float32)
    if image.max(initial=0.0) > 1.0:
        image /= 255.0
    image = np.clip(image, 0.0, 1.0)
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    gy, gx = np.gradient(luminance)
    gradient = np.hypot(gx, gy)
    saturation = image.max(axis=2) - image.min(axis=2)
    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        center = np.asarray([float(row.cx_px), float(row.cy_px)])
        corners = _corners(center, float(row.theta_deg), float(row.width_px), float(row.height_px))
        outer = _corners(center, float(row.theta_deg), 1.4 * float(row.width_px), 2.0 * float(row.height_px))
        channels = []
        for channel in range(3):
            values, _ = _region_values(image[..., channel], corners)
            channels.append(values)
        local_luminance, _ = _region_values(luminance, corners)
        local_gradient, _ = _region_values(gradient, corners)
        local_saturation, _ = _region_values(saturation, corners)
        outer_luminance = _outer_context_values(luminance, corners, outer)
        closing, _ = _axis(float(row.theta_deg))
        left = center - 0.5 * float(row.width_px) * closing
        right = center + 0.5 * float(row.width_px) * closing
        radius = max(2, int(round(float(row.width_px) * 0.06)))
        left_rgb = _rgb_patch(image, left, radius).reshape(-1, 3)
        right_rgb = _rgb_patch(image, right, radius).reshape(-1, 3)
        left_mean = left_rgb.mean(axis=0) if len(left_rgb) else np.zeros(3)
        right_mean = right_rgb.mean(axis=0) if len(right_rgb) else np.zeros(3)
        feature: dict[str, object] = {
            "sample_id": str(row.sample_id),
            "candidate_id": str(row.candidate_id),
            "local_rgb_luminance_mean": float(local_luminance.mean()) if len(local_luminance) else 0.0,
            "local_rgb_luminance_std": float(local_luminance.std()) if len(local_luminance) else 0.0,
            "local_rgb_saturation_mean": float(local_saturation.mean()) if len(local_saturation) else 0.0,
            "local_rgb_gradient_mean": float(local_gradient.mean()) if len(local_gradient) else 0.0,
            "local_vs_outer_luminance_contrast": (
                float(local_luminance.mean() - outer_luminance.mean())
                if len(local_luminance) and len(outer_luminance)
                else 0.0
            ),
            "left_right_contact_rgb_difference": float(np.linalg.norm(left_mean - right_mean)),
            "rgb_missing": 0.0,
        }
        for index, name in enumerate(("red", "green", "blue")):
            feature[f"local_rgb_{name}_mean"] = float(channels[index].mean()) if len(channels[index]) else 0.0
            feature[f"local_rgb_{name}_std"] = float(channels[index].std()) if len(channels[index]) else 0.0
        rows.append(feature)
    return pd.DataFrame(rows)
