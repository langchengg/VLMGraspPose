"""
stage1/grounding.py — Target Grounding
========================================
Two implementations:
  • GroundTruthGrounder  — uses GT label masks (for demo / sanity-check)
  • VLMGrounder          — Florence-2 open-vocabulary grounding

Unified output:
    {bbox: [x1,y1,x2,y2], mask: HxW or None, confidence: float}
"""

import abc
from dataclasses import dataclass
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


# ── Base class ───────────────────────────────────────────────────────

class TargetGrounder(abc.ABC):
    """Abstract target-grounding interface.

    Every grounder takes an RGB image and a text query and returns a
    GroundingResult (bbox + optional mask + confidence).
    """

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
    """Use the GT label mask to produce a perfect grounding.

    This is only for **demo / debugging / sanity-check**.
    It should never be used as a Stage-1 "model" in a published experiment
    without clearly labelling it as an oracle.
    """

    def ground(
        self,
        rgb: np.ndarray,
        text_query: str,
        *,
        label: np.ndarray = None,
        instance_id: int = None,
        **kwargs,
    ) -> Optional[GroundingResult]:
        """
        Parameters
        ----------
        label : HxW uint8 instance segmentation mask
        instance_id : pixel value in *label* that corresponds to the target
        """
        if label is None or instance_id is None:
            raise ValueError(
                "GroundTruthGrounder requires 'label' and 'instance_id' kwargs."
            )

        mask = (label == instance_id)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None

        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        return GroundingResult(
            bbox=bbox,
            mask=mask,
            confidence=1.0,
        )


# ── VLM Grounder (Florence-2) ───────────────────────────────────────

class VLMGrounder(TargetGrounder):
    """Open-vocabulary grounding using Florence-2.

    Weights are loaded from the local ``models/florence-2-base/`` directory
    (downloaded via ``scripts/download_weights.py``).

    Falls back to the HuggingFace Hub ID if a local directory is absent,
    but will warn loudly.
    """

    def __init__(
        self,
        model_name: str = "florence-2",
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self._model = None
        self._processor = None

        if model_dir is None:
            model_dir = config.FLORENCE2_MODEL_DIR

        self._model_dir = Path(model_dir)

        if device is None:
            import torch
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    # ── Lazy loading ─────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Load model + processor on first call (lazy to save startup time)."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        model_path = str(self._model_dir)

        # Check local directory first
        if self._model_dir.exists() and any(self._model_dir.iterdir()):
            print(f"[Florence-2] Loading from local: {model_path}")
        else:
            # Fall back to HuggingFace Hub
            model_path = "microsoft/Florence-2-base"
            print(f"[Florence-2] Local weights not found at {self._model_dir}")
            print(f"[Florence-2] Falling back to HuggingFace Hub: {model_path}")
            print(f"[Florence-2] Run 'python scripts/download_weights.py --florence2' to cache locally.")

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
        """Detect the target object described by *text_query*.

        Uses Florence-2 ``<OPEN_VOCABULARY_DETECTION>`` task.
        Returns the highest-confidence detection, or None if nothing found.
        """
        self._ensure_loaded()

        import torch

        image = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
        W, H = image.size

        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
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

        # Extract best detection
        bboxes = parsed.get(task_prompt, {}).get("bboxes", [])
        labels = parsed.get(task_prompt, {}).get("bboxes_labels", [])

        if not bboxes:
            return None

        # Use the first (highest-confidence) detection
        bbox_raw = bboxes[0]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_raw]

        # Clamp to image bounds
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))

        # Florence-2 does not output per-box confidence directly;
        # use 1.0 and let downstream stages handle uncertainty.
        confidence = 1.0

        return GroundingResult(
            bbox=[x1, y1, x2, y2],
            mask=None,
            confidence=confidence,
        )


# ── Factory ──────────────────────────────────────────────────────────

def get_grounder(name: str = "gt", **kwargs) -> TargetGrounder:
    """Factory to get a grounder by name."""
    if name == "gt":
        return GroundTruthGrounder()
    elif name in ("vlm", "florence", "florence-2"):
        return VLMGrounder(model_name="florence-2", **kwargs)
    elif name in ("grounding_dino",):
        return VLMGrounder(model_name=name, **kwargs)
    else:
        raise ValueError(f"Unknown grounder: {name}")

