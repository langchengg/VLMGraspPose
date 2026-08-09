"""Dense OCID-VLG grasp supervision in the official network output convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skimage.draw import polygon

from .evaluator import grasp_from_ocid_corners
from .geometry import CropTransform, rectangle_corners


@dataclass(frozen=True, slots=True)
class DenseGraspTargets:
    quality: np.ndarray
    cos_2theta: np.ndarray
    sin_2theta: np.ndarray
    width: np.ndarray


def build_dense_grasp_targets(
    gt_corners: Sequence[Sequence[Sequence[float]]],
    *,
    transform: CropTransform,
    output_shape: tuple[int, int],
    width_scale_px: float = 150.0,
    fixed_height_px: float = 20.0,
    gt_width_clip_px: float = 100.0,
) -> DenseGraspTargets:
    """Rasterise the central third of each GT rectangle like the official loaders.

    OCID-VLG orders its vertices with the short finger thickness edge first,
    unlike the Cornell rectangle ordering assumed by the upstream helper.  We
    therefore decode the verified OCID geometry first and rasterise it locally;
    directly passing OCID vertices to the upstream helper would swap opening
    width and finger thickness.
    """

    height, width = map(int, output_shape)
    if min(height, width) <= 0:
        raise ValueError("output_shape must be positive")
    if width_scale_px <= 0.0:
        raise ValueError("width_scale_px must be positive")
    quality = np.zeros((height, width), dtype=np.float32)
    angle = np.zeros_like(quality)
    opening_width = np.zeros_like(quality)

    model_height_scale = 0.5 * (
        transform.model_width / transform.crop_width
        + transform.model_height / transform.crop_height
    )
    for corners in gt_corners:
        grasp = grasp_from_ocid_corners(
            corners,
            fixed_height_px=fixed_height_px,
            width_clip_px=gt_width_clip_px,
        )
        center_x, center_y, angle_deg, model_width = transform.native_to_model_pose(
            grasp.center_x, grasp.center_y, grasp.angle_deg, grasp.width_px
        )
        compact = rectangle_corners(
            center_x,
            center_y,
            model_width / 3.0,
            fixed_height_px * model_height_scale,
            angle_deg,
        )
        rows, columns = polygon(compact[:, 1], compact[:, 0], shape=(height, width))
        if rows.size == 0:
            continue
        radians = np.deg2rad(angle_deg)
        quality[rows, columns] = 1.0
        angle[rows, columns] = np.float32(radians)
        opening_width[rows, columns] = np.float32(
            np.clip(model_width, 0.0, width_scale_px) / width_scale_px
        )

    return DenseGraspTargets(
        quality=quality,
        cos_2theta=np.cos(2.0 * angle).astype(np.float32),
        sin_2theta=np.sin(2.0 * angle).astype(np.float32),
        width=opening_width,
    )
