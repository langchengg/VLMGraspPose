from __future__ import annotations

import numpy as np
import pytest

from src.grasping.backends.conditioning import (
    condition_rgbd,
    normalise_depth_official,
    normalise_rgb_official,
    resize_probability_to_native,
)
from src.grasping.common.geometry import CropTransform
from src.grasping.common.training_targets import build_dense_grasp_targets


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.full((80, 120, 3), (40, 80, 120), dtype=np.uint8)
    depth = np.full((80, 120), 0.75, dtype=np.float32)
    depth[0, 0] = 0.0
    mask = np.zeros((80, 120), dtype=bool)
    mask[20:60, 40:80] = True
    probability = np.zeros((40, 60), dtype=np.float32)
    probability[10:30, 20:40] = 0.9
    return rgb, depth, mask, probability


def test_official_per_image_normalisation() -> None:
    depth = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert np.allclose(normalise_depth_official(depth), np.clip(depth - 2.5, -1, 1))
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    result = normalise_rgb_official(rgb)
    assert result.dtype == np.float32
    assert float(result.mean()) == pytest.approx(0.0, abs=1e-7)


def test_probability_resize_preserves_soft_values_and_range() -> None:
    _, _, _, probability = _arrays()
    resized = resize_probability_to_native(probability, (80, 120))
    assert resized.shape == (80, 120)
    assert 0.0 <= float(resized.min()) <= float(resized.max()) <= 1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resize_probability_to_native(np.asarray([[1.2]], np.float32), (2, 2))


@pytest.mark.parametrize("variant", ["hard_mask", "dilated_crop"])
def test_conditioning_shapes_finite_and_reversible(variant: str) -> None:
    rgb, depth, mask, probability = _arrays()
    conditioned = condition_rgbd(
        rgb=rgb,
        depth_m=depth,
        binary_mask=mask,
        probability=probability,
        variant=variant,
        output_size=64,
        dilation_fraction=0.15,
    )
    assert conditioned.rgb_chw.shape == (3, 64, 64)
    assert conditioned.depth_chw.shape == (1, 64, 64)
    assert conditioned.gate_map.shape == (64, 64)
    assert conditioned.valid_depth_map.shape == (64, 64)
    assert all(
        np.all(np.isfinite(value))
        for value in (conditioned.rgb_chw, conditioned.depth_chw, conditioned.gate_map)
    )
    model = conditioned.transform.native_to_model_point(60.0, 40.0)
    native = conditioned.transform.model_to_native_point(*model)
    assert native == pytest.approx((60.0, 40.0))


def test_empty_mask_and_invalid_target_depth_fail_closed() -> None:
    rgb, depth, mask, probability = _arrays()
    with pytest.raises(ValueError, match="empty_mask"):
        condition_rgbd(
            rgb=rgb,
            depth_m=depth,
            binary_mask=np.zeros_like(mask),
            probability=probability,
            variant="hard_mask",
        )
    depth[mask] = 0.0
    with pytest.raises(ValueError, match="invalid_target_depth"):
        condition_rgbd(
            rgb=rgb,
            depth_m=depth,
            binary_mask=mask,
            probability=probability,
            variant="hard_mask",
        )


def test_ocid_opening_width_not_short_edge_is_training_target() -> None:
    transform = CropTransform(
        crop_x=0,
        crop_y=0,
        crop_width=100,
        crop_height=100,
        model_width=100,
        model_height=100,
        native_width=100,
        native_height=100,
    )
    # OCID ordering: p0->p1 is the fixed-height edge; p0->p3 is jaw opening.
    corners = [[[20, 40], [20, 60], [80, 60], [80, 40]]]
    targets = build_dense_grasp_targets(
        corners, transform=transform, output_shape=(100, 100), width_scale_px=150
    )
    positive = targets.quality > 0
    assert np.any(positive)
    assert np.allclose(targets.width[positive], 60.0 / 150.0)
    assert np.allclose(targets.cos_2theta[positive], 1.0)
    assert np.allclose(targets.sin_2theta[positive], 0.0, atol=1e-7)
