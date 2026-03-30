"""
src/grounding.py — Target Grounding (Step 4)
===============================================
Migrated from stage1/grounding.py + stage1/postprocess_bbox.py.

Implementations:
  • GroundTruthGrounder  — uses GT label masks (oracle upper-bound)
  • Florence2Grounder   — Florence-2-large fine-tuned
      Tasks:
        - Box grounding:    <CAPTION_TO_PHRASE_GROUNDING>
        - Mask refinement:  <REFERRING_EXPRESSION_SEGMENTATION>

Unified output: GroundingResult {bbox, mask, confidence}
"""

import abc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ── Output dataclass ─────────────────────────────────────────────────

@dataclass
class GroundingResult:
    bbox: List[int]              # [x1, y1, x2, y2]
    mask: Optional[np.ndarray]   # HxW bool or None
    confidence: float            # 0–1

    def to_dict(self) -> dict:
        d = {"bbox": self.bbox, "confidence": self.confidence}
        d["has_mask"] = self.mask is not None
        return d


# ── Base class ───────────────────────────────────────────────────────

class TargetGrounder(abc.ABC):
    """Abstract target-grounding interface."""

    @abc.abstractmethod
    def ground(
        self,
        rgb: np.ndarray,
        text_query: str,
        **kwargs,
    ) -> Optional[GroundingResult]:
        ...


# ── Ground-Truth Grounder ────────────────────────────────────────────

class GroundTruthGrounder(TargetGrounder):
    """Use GT label mask to produce a perfect grounding (oracle)."""

    def ground(
        self,
        rgb: np.ndarray,
        text_query: str,
        *,
        label: np.ndarray = None,
        mask_val: int = None,
        **kwargs,
    ) -> Optional[GroundingResult]:
        """
        Parameters
        ----------
        label : HxW instance segmentation mask
        mask_val : pixel value in *label* corresponding to the target
                   (= obj_id + 1 in GraspNet convention)
        """
        if label is None or mask_val is None:
            raise ValueError(
                "GroundTruthGrounder requires 'label' and 'mask_val' kwargs."
            )

        mask = (label == mask_val)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None

        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        return GroundingResult(bbox=bbox, mask=mask, confidence=1.0)


# ── Florence-2 Grounder (large, fine-tuned) ─────────────────────────

class Florence2Grounder(TargetGrounder):
    """Grounding via Florence-2-large fine-tuned.

    Supported tasks:
      - 'phrase'  → <CAPTION_TO_PHRASE_GROUNDING>  (box output)
      - 'seg'     → <REFERRING_EXPRESSION_SEGMENTATION>  (mask output)
    """

    TASK_PROMPTS = {
        "phrase": "<CAPTION_TO_PHRASE_GROUNDING>",
        "seg":    "<REFERRING_EXPRESSION_SEGMENTATION>",
    }

    def __init__(
        self,
        task: str = "phrase",
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        if task not in self.TASK_PROMPTS:
            raise ValueError(
                f"Unknown Florence-2 task '{task}'. "
                f"Choose from {list(self.TASK_PROMPTS.keys())}"
            )
        self._task = task
        self._model_dir = Path(model_dir) if model_dir else config.FLORENCE2_MODEL_DIR
        self._model = None
        self._processor = None

        if device is None:
            import torch
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    # ── Lazy loading ─────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        model_path = str(self._model_dir)

        if self._model_dir.exists() and any(self._model_dir.iterdir()):
            print(f"[Florence-2] Loading from local: {model_path}")
        else:
            model_path = config.FLORENCE2_MODEL_ID
            print(f"[Florence-2] Local weights not found at {self._model_dir}")
            print(f"[Florence-2] Falling back to HuggingFace Hub: {model_path}")

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self._device)

        self._model.eval()
        print(f"[Florence-2] Model loaded on {self._device}")

    # ── Inference ────────────────────────────────────────────────────

    def ground(
        self,
        rgb: np.ndarray,
        text_query: str,
        **kwargs,
    ) -> Optional[GroundingResult]:
        """Detect the target object described by text_query."""
        self._ensure_loaded()
        import torch

        image = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
        W, H = image.size

        task_prompt = self.TASK_PROMPTS[self._task]
        prompt = task_prompt + text_query

        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(W, H),
        )

        return self._extract_result(parsed, task_prompt, W, H)

    def _extract_result(
        self, parsed: dict, task_prompt: str, W: int, H: int,
    ) -> Optional[GroundingResult]:
        """Parse Florence-2 output into a GroundingResult."""
        task_result = parsed.get(task_prompt, {})

        # ── Try bbox extraction ──────────────────────────────────────
        bboxes = task_result.get("bboxes", [])

        # Phrase grounding nests bboxes differently
        if not bboxes:
            for key in task_result:
                val = task_result[key]
                if isinstance(val, list) and len(val) > 0:
                    if isinstance(val[0], list) and len(val[0]) == 4:
                        bboxes = val
                        break

        if not bboxes:
            return None

        bbox_raw = bboxes[0]
        x1 = max(0, min(int(round(bbox_raw[0])), W - 1))
        y1 = max(0, min(int(round(bbox_raw[1])), H - 1))
        x2 = max(0, min(int(round(bbox_raw[2])), W - 1))
        y2 = max(0, min(int(round(bbox_raw[3])), H - 1))

        # ── Try mask extraction (REFERRING_EXPRESSION_SEGMENTATION) ──
        mask = None
        if self._task == "seg":
            polygons = task_result.get("polygons", [])
            if polygons and len(polygons) > 0:
                mask = self._polygons_to_mask(polygons[0], W, H)

        return GroundingResult(
            bbox=[x1, y1, x2, y2],
            mask=mask,
            confidence=1.0,  # Florence-2 doesn't output per-box conf
        )

    @staticmethod
    def _polygons_to_mask(
        polygon_points: list, W: int, H: int,
    ) -> np.ndarray:
        """Convert Florence-2 polygon output to a binary mask."""
        import cv2
        mask = np.zeros((H, W), dtype=np.uint8)
        if not polygon_points:
            return mask.astype(bool)

        # polygon_points is a flat list [x1,y1,x2,y2,...] or nested
        if isinstance(polygon_points[0], (list, tuple)):
            pts = np.array(polygon_points, dtype=np.int32)
        else:
            coords = polygon_points
            pts = np.array(
                [[int(coords[i]), int(coords[i + 1])]
                 for i in range(0, len(coords) - 1, 2)],
                dtype=np.int32,
            )

        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)


# ── Factory ──────────────────────────────────────────────────────────

def get_grounder(name: str = "gt", **kwargs) -> TargetGrounder:
    """Factory to get a grounder by name.

    Supported names:
        'gt'      — Ground-truth oracle
        'phrase'  — Florence-2 phrase grounding (box)
        'seg'     — Florence-2 referring expression segmentation (mask)
    """
    if name == "gt":
        return GroundTruthGrounder()
    elif name in ("phrase", "vlm", "florence"):
        return Florence2Grounder(task="phrase", **kwargs)
    elif name == "seg":
        return Florence2Grounder(task="seg", **kwargs)
    else:
        raise ValueError(
            f"Unknown grounder: {name}. Choose from: gt, phrase, seg"
        )


# ── Utility: pad bbox ───────────────────────────────────────────────

def pad_bbox(
    bbox: List[int],
    pad: int = 10,
    img_w: int = config.IMAGE_WIDTH,
    img_h: int = config.IMAGE_HEIGHT,
) -> List[int]:
    """Expand bounding box by pad pixels, clipping to image bounds."""
    x1, y1, x2, y2 = bbox
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(img_w - 1, x2 + pad),
        min(img_h - 1, y2 + pad),
    ]
