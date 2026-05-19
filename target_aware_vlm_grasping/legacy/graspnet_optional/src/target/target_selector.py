from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from dataset.annotation_loader import select_largest_visible_instance
from target.mask_utils import (
    bbox_to_mask,
    clean_binary_mask,
    compute_mask_center,
    load_mask,
    mask_to_bbox,
)
from target.pseudo_language import command_for_target
from utils.data_types import GraspNetSample, TargetRegion


@dataclass
class TargetSelectionConfig:
    mode: str = "pseudo"
    target_id: Optional[int] = None
    bbox: Optional[list[int]] = None
    mask_path: Optional[Path] = None


class TargetSelector:
    def __init__(self, mode: str = "pseudo"):
        self.mode = mode

    def select(
        self,
        sample: GraspNetSample,
        rgb: np.ndarray,
        label: np.ndarray | None = None,
        target_id: int | None = None,
        bbox: list[int] | None = None,
        mask_path: Path | None = None,
    ) -> TargetRegion:
        mode = self.mode
        if mode == "manual":
            return self._select_manual(rgb, target_id, bbox, mask_path)
        if mode in {"annotation", "pseudo"}:
            return self._select_from_annotation(sample, rgb, label, target_id, pseudo=(mode == "pseudo"))
        raise ValueError(f"Unknown target mode: {mode}. Use annotation, manual, or pseudo.")

    def _select_manual(
        self,
        rgb: np.ndarray,
        target_id: int | None,
        bbox: list[int] | None,
        mask_path: Path | None,
    ) -> TargetRegion:
        mask = load_mask(mask_path) if mask_path else None
        if mask is None and bbox is not None:
            mask = bbox_to_mask(bbox, rgb.shape[:2])
        if mask is not None:
            mask = clean_binary_mask(mask)
            bbox = bbox or mask_to_bbox(mask)
        if bbox is None:
            raise ValueError("Manual target selection requires --target-bbox or --target-mask.")
        center = compute_mask_center(mask) if mask is not None else (
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        )
        label = f"object_{target_id:03d}" if target_id is not None else "manual_target"
        return TargetRegion(target_id, label, bbox, mask, 1.0, center, command=command_for_target(target_id, label))

    def _select_from_annotation(
        self,
        sample: GraspNetSample,
        rgb: np.ndarray,
        label: np.ndarray | None,
        target_id: int | None,
        pseudo: bool,
    ) -> TargetRegion:
        if label is None:
            if sample.label_path is None:
                raise ValueError("Annotation target mode requires a label image.")
            from dataset.camera_loader import load_label
            label = load_label(sample.label_path)

        if target_id is None:
            selected = select_largest_visible_instance(sample.label_path)
            if selected is None:
                raise ValueError("No visible target instance found in annotation label.")
            target_id = selected["target_id"]
            mask_val = selected["mask_val"]
        else:
            mask_val = int(target_id) + 1

        mask = label == mask_val
        if not mask.any():
            raise ValueError(f"Target id {target_id} is not visible in frame {sample.frame_id}.")
        mask = clean_binary_mask(mask)
        bbox = mask_to_bbox(mask)
        if bbox is None:
            raise ValueError(f"Target id {target_id} mask is empty after cleaning.")
        center = compute_mask_center(mask)
        label_text = f"object_{target_id:03d}"
        command = command_for_target(target_id, label_text) if pseudo else None
        return TargetRegion(
            target_id=target_id,
            label=label_text,
            bbox=bbox,
            mask=mask,
            grounding_score=1.0,
            center_2d=center,
            command=command,
            metadata={"mask_val": mask_val, "selection_mode": "pseudo" if pseudo else "annotation"},
        )
