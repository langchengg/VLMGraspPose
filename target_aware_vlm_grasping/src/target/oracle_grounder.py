from __future__ import annotations

import cv2
import numpy as np

from target.base_grounder import BaseTargetGrounder
from target.mask_utils import bbox_to_mask, clean_binary_mask, compute_mask_center, load_mask, mask_to_bbox
from utils.data_types import DatasetSample, TargetRegion


class OracleTargetGrounder(BaseTargetGrounder):
    """Use dataset-provided target bbox/mask for controlled evaluation."""

    def predict(self, sample: DatasetSample, rgb_image: np.ndarray) -> TargetRegion:
        mask = _sample_mask(sample, rgb_image.shape[:2])
        bbox = sample.target_bbox_gt or sample.target_bbox
        if mask is not None:
            mask = clean_binary_mask(mask)
            bbox = bbox or mask_to_bbox(mask)
        if bbox is None:
            raise ValueError(f"Oracle target mode requires bbox or mask for sample {sample.sample_id}.")
        if mask is None:
            mask = bbox_to_mask(bbox, rgb_image.shape[:2])
        return TargetRegion(
            target_id=sample.target_id,
            label=sample.target_label,
            bbox=bbox,
            mask=mask,
            grounding_score=1.0,
            center_2d=compute_mask_center(mask),
            command=sample.command,
            target_source="oracle",
            metadata={
                "target_bbox_gt": bbox,
                "target_mask_source": "dataset",
            },
        )


def _sample_mask(sample: DatasetSample, shape: tuple[int, int]) -> np.ndarray | None:
    if isinstance(sample.target_mask_gt, np.ndarray):
        return sample.target_mask_gt.astype(bool)
    if sample.target_mask_gt:
        return load_mask(sample.target_mask_gt)
    if sample.target_mask_path:
        path = sample.target_mask_path
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3:
            img = img[:, :, 0]
        if sample.target_index is not None:
            mask = img.astype(np.int32) == int(sample.target_index)
            if mask.any():
                return mask
        return img.astype(bool)
    return None

