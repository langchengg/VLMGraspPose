from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SAMRefinerConfig:
    checkpoint: str = "models/vlm/sam/sam_vit_b_01ec64.pth"
    model_type: str = "vit_b"
    device: str = "cpu"
    multimask_output: bool = True
    min_area_ratio: float = 0.01
    max_area_ratio: float = 1.25
    bbox_expansion_pixels: int = 0
    bbox_expansion_ratio: float = 0.0


class SAMMaskRefiner:
    """BBox-prompted Segment Anything mask refinement.

    The import is lazy so oracle mode and Florence bbox-only mode keep running
    without the optional `segment_anything` package.
    """

    def __init__(self, config: SAMRefinerConfig | dict | None = None):
        if isinstance(config, dict):
            config = SAMRefinerConfig(**{k: v for k, v in config.items() if k in SAMRefinerConfig.__annotations__})
        self.config = config or SAMRefinerConfig()
        self._predictor = None

    def refine(self, rgb: np.ndarray, bbox: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
        self._load()
        h, w = rgb.shape[:2]
        box = np.asarray(_expand_bbox(_clip_bbox(bbox, w, h), w, h, self.config), dtype=np.float32)
        self._predictor.set_image(rgb.astype(np.uint8))
        masks, scores, logits = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=bool(self.config.multimask_output),
        )
        if masks is None or len(masks) == 0:
            raise ValueError("SAM returned no masks.")
        mask, idx, score = self._select_mask(masks, scores, box, (h, w))
        return mask.astype(bool), {
            "target_mask_source": "sam",
            "sam_model_type": self.config.model_type,
            "sam_checkpoint": self.config.checkpoint,
            "sam_score": float(score),
            "sam_mask_index": int(idx),
            "sam_box": box.astype(int).tolist(),
            "sam_mask_area": int(mask.sum()),
            "sam_num_masks": int(len(masks)),
        }

    def _load(self) -> None:
        if self._predictor is not None:
            return
        checkpoint = _resolve_project_path(self.config.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "SAM refinement requires optional dependency `segment-anything`. "
                "Install it with `python -m pip install segment-anything`."
            ) from exc
        if self.config.model_type not in sam_model_registry:
            raise ValueError(f"Unsupported SAM model_type '{self.config.model_type}'.")
        model = sam_model_registry[self.config.model_type](checkpoint=str(checkpoint))
        model.to(device=self.config.device)
        model.eval()
        self._predictor = SamPredictor(model)

    def _select_mask(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        box: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, int, float]:
        x1, y1, x2, y2 = box.astype(int).tolist()
        bbox_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
        best_idx = 0
        best_score = -float("inf")
        scores = np.asarray(scores if scores is not None else np.ones(len(masks)), dtype=float)
        for idx, mask in enumerate(masks):
            area = int(mask.sum())
            area_ratio = area / bbox_area
            if area_ratio < self.config.min_area_ratio or area_ratio > self.config.max_area_ratio:
                area_penalty = 0.4
            else:
                area_penalty = 0.0
            inside_ratio = _inside_box_ratio(mask, box, shape)
            score = float(scores[idx]) + 0.25 * inside_ratio - area_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        return masks[best_idx].astype(bool), best_idx, float(scores[best_idx])


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    project_relative = Path(__file__).resolve().parents[2] / path
    return project_relative


def _clip_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = sorted(np.clip([x1, x2], 0, width - 1).astype(int).tolist())
    y1, y2 = sorted(np.clip([y1, y2], 0, height - 1).astype(int).tolist())
    return [x1, y1, x2, y2]


def _expand_bbox(bbox: list[int], width: int, height: int, config: SAMRefinerConfig) -> list[int]:
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1 + 1, 1)
    bh = max(y2 - y1 + 1, 1)
    pad_x = max(int(config.bbox_expansion_pixels), int(round(config.bbox_expansion_ratio * bw)))
    pad_y = max(int(config.bbox_expansion_pixels), int(round(config.bbox_expansion_ratio * bh)))
    return _clip_bbox([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def _inside_box_ratio(mask: np.ndarray, box: np.ndarray, shape: tuple[int, int]) -> float:
    h, w = shape
    x1, y1, x2, y2 = box.astype(int).tolist()
    x1, x2 = sorted(np.clip([x1, x2], 0, w - 1).astype(int).tolist())
    y1, y2 = sorted(np.clip([y1, y2], 0, h - 1).astype(int).tolist())
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    inside = float(mask[y1:y2 + 1, x1:x2 + 1].sum())
    return inside / area
