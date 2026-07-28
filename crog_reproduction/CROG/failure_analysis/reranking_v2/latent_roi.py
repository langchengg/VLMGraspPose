from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F


def _axial_radians(angle_deg: float) -> float:
    return math.radians((float(angle_deg) + 90.0) % 180.0 - 90.0)


def candidate_roi_grid(
    candidate: dict[str, Any],
    *,
    image_shape: tuple[int, int],
    feature_shape: tuple[int, int],
    roi_size: int = 5,
    width_scale: float = 1.5,
    height_scale: float = 2.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a grid_sample grid for one candidate-aligned feature ROI."""
    image_h, image_w = map(int, image_shape)
    feature_h, feature_w = map(int, feature_shape)
    if min(image_h, image_w, feature_h, feature_w, roi_size) <= 0:
        raise ValueError("all shapes must be positive")
    axis = torch.linspace(-1.0, 1.0, int(roi_size), device=device, dtype=dtype)
    vv, uu = torch.meshgrid(axis, axis, indexing="ij")
    theta = _axial_radians(candidate["angle_deg"])
    opening_x, opening_y = math.cos(theta), -math.sin(theta)
    perp_x, perp_y = -opening_y, opening_x
    half_w = max(float(candidate["width_px"]) * width_scale / 2.0, 1.0)
    half_h = max(float(candidate["height_px"]) * height_scale / 2.0, 1.0)
    source_x = (
        float(candidate["cx"])
        + uu * half_w * opening_x
        + vv * half_h * perp_x
    )
    source_y = (
        float(candidate["cy"])
        + uu * half_w * opening_y
        + vv * half_h * perp_y
    )
    # align_corners=True maps image endpoints exactly onto feature endpoints.
    feature_x = source_x * (feature_w - 1) / max(1, image_w - 1)
    feature_y = source_y * (feature_h - 1) / max(1, image_h - 1)
    norm_x = 2.0 * feature_x / max(1, feature_w - 1) - 1.0
    norm_y = 2.0 * feature_y / max(1, feature_h - 1) - 1.0
    return torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)


def pool_candidate_rois(
    feature_map: torch.Tensor,
    candidates_by_batch: list[list[dict[str, Any]]],
    *,
    image_shapes: list[tuple[int, int]],
    roi_size: int = 5,
) -> torch.Tensor:
    """Return `[B,K,2C]` average+max pooled rotated ROIs."""
    if feature_map.ndim != 4:
        raise ValueError("feature_map must be [B,C,H,W]")
    batch, channels, feature_h, feature_w = feature_map.shape
    if len(candidates_by_batch) != batch or len(image_shapes) != batch:
        raise ValueError("batch metadata does not match feature map")
    pooled = []
    for batch_index, candidates in enumerate(candidates_by_batch):
        sample = feature_map[batch_index : batch_index + 1]
        sample_vectors = []
        for candidate in candidates:
            grid = candidate_roi_grid(
                candidate,
                image_shape=image_shapes[batch_index],
                feature_shape=(feature_h, feature_w),
                roi_size=roi_size,
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
            try:
                roi = F.grid_sample(
                    sample,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
            except RuntimeError:
                # Some historical MPS builds lacked grid_sample variants.
                roi = F.grid_sample(
                    sample.cpu(),
                    grid.cpu(),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).to(feature_map.device)
            average = roi.mean(dim=(-2, -1)).squeeze(0)
            maximum = roi.amax(dim=(-2, -1)).squeeze(0)
            sample_vectors.append(torch.cat((average, maximum), dim=0))
        if not sample_vectors:
            sample_vectors = [
                torch.zeros(
                    channels * 2,
                    device=feature_map.device,
                    dtype=feature_map.dtype,
                )
            ]
        pooled.append(torch.stack(sample_vectors))
    candidate_counts = {value.shape[0] for value in pooled}
    if len(candidate_counts) != 1:
        raise ValueError("candidate mask/padding is required for variable K")
    return torch.stack(pooled)


class CROGLatentCapture:
    """Read-only hooks for pre-decoder and post-decoder CROG fused features."""

    def __init__(self, model):
        self.model = model
        self.pre_decoder = None
        self.post_decoder = None
        self._handles = []

    def _capture_pre(self, _module, _inputs, output):
        self.pre_decoder = output.detach()

    def _capture_post(self, _module, _inputs, output):
        tensor = output[-1] if isinstance(output, list) else output
        self.post_decoder = tensor.detach()

    def install(self):
        if self._handles:
            raise RuntimeError("latent hooks are already installed")
        base = self.model.module if hasattr(self.model, "module") else self.model
        self._handles.append(base.neck.register_forward_hook(self._capture_pre))
        self._handles.append(base.decoder.register_forward_hook(self._capture_post))
        return self

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def suspended(self):
        """Temporarily remove hooks so output equivalence is tested honestly."""
        was_installed = bool(self._handles)
        if was_installed:
            self.remove()
        try:
            yield
        finally:
            if was_installed:
                self.install()

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, traceback):
        self.remove()

    def feature_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pre_decoder is None or self.post_decoder is None:
            raise RuntimeError("CROG forward has not populated latent hooks")
        pre = self.pre_decoder
        post = self.post_decoder
        if post.ndim == 3:
            batch, channels, flattened = post.shape
            side = int(round(math.sqrt(flattened)))
            if side * side != flattened:
                raise ValueError("decoder feature is not a square spatial map")
            post = post.reshape(batch, channels, side, side)
        if pre.ndim != 4 or post.ndim != 4:
            raise ValueError("captured CROG latents must be spatial tensors")
        return pre, post
