"""Lazy official Transformers SAM 3 model adapter.

This module deliberately does not import torch/transformers at import time so
prompt preparation and output validation remain usable on the Mac host.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .sam3_prompt_builder import VisualPrompt


@dataclass(frozen=True)
class Sam3InferenceResult:
    masks: tuple[np.ndarray, ...]
    probabilities: tuple[np.ndarray, ...]
    qualities: tuple[float, ...]
    runtime_seconds: float
    runtime_metadata: dict[str, Any]


class Sam3RuntimeError(RuntimeError):
    pass


def build_tracker_processor_inputs(image: Image.Image, prompt: VisualPrompt) -> dict[str, Any]:
    """Build the official one-image/one-object nested prompt structure."""

    points = [list(point) for point in (*prompt.positive_points_xy, *prompt.negative_points_xy)]
    labels = [1] * len(prompt.positive_points_xy) + [0] * len(prompt.negative_points_xy)
    inputs: dict[str, Any] = {
        "images": image.convert("RGB"),
        "input_boxes": [[list(prompt.expanded_box_xyxy)]],
        "return_tensors": "pt",
    }
    if points:
        # batch -> prompted object -> points; labels use the same nesting.
        inputs["input_points"] = [[points]]
        inputs["input_labels"] = [[labels]]
    return inputs


def restore_tracker_hypotheses(restored_masks: Any, *, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Validate official post-processing output and return probabilities/masks."""

    array = restored_masks.detach().float().cpu().numpy() if hasattr(restored_masks, "detach") else np.asarray(restored_masks)
    if array.ndim != 4 or array.shape[0] != 1:
        raise Sam3RuntimeError(f"unexpected post-processed mask shape {array.shape}")
    logits = np.asarray(array[0], dtype=np.float32)
    if tuple(logits.shape[-2:]) != tuple(expected_shape):
        raise Sam3RuntimeError(
            f"SAM masks were restored to {logits.shape[-2:]}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(logits)):
        raise Sam3RuntimeError("SAM returned non-finite mask logits")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    return probabilities.astype(np.float32), probabilities >= 0.5


def _imports():
    try:
        import torch
        import transformers
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    except (ImportError, AttributeError) as error:
        raise Sam3RuntimeError(
            "Official SAM 3 Tracker requires an isolated modern environment with "
            "torch and transformers>=5.0; do not install it into .venv-gqcnn"
        ) from error
    return torch, transformers, Sam3TrackerModel, Sam3TrackerProcessor


def cuda_preflight() -> dict[str, Any]:
    torch, transformers, _, _ = _imports()
    available = bool(torch.cuda.is_available())
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "bf16_supported": bool(available and torch.cuda.is_bf16_supported()),
    }
    if not available:
        raise Sam3RuntimeError(
            "No compatible NVIDIA CUDA GPU is available. Full SAM 3 inference is intentionally "
            "not switched to CPU or MPS."
        )
    return result


class OfficialSam3Tracker:
    def __init__(
        self,
        model_path_or_id: str | Path,
        *,
        revision: str | None,
        local_files_only: bool,
        precision: str,
    ):
        torch, transformers, model_class, processor_class = _imports()
        runtime = cuda_preflight()
        if precision == "auto":
            precision = "bf16" if runtime["bf16_supported"] else "fp32"
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be auto, fp32, or bf16")
        if precision == "bf16" and not runtime["bf16_supported"]:
            raise Sam3RuntimeError("bf16 was requested but is not supported by the active GPU")
        kwargs = {"local_files_only": bool(local_files_only)}
        if revision:
            kwargs["revision"] = revision
        self.processor = processor_class.from_pretrained(str(model_path_or_id), **kwargs)
        self.model = model_class.from_pretrained(str(model_path_or_id), **kwargs).to("cuda")
        self.model.eval()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.torch = torch
        self.precision = precision
        self.runtime_metadata = {
            **runtime,
            "model_id_or_path": str(model_path_or_id),
            "model_revision": revision,
            "inference_precision": precision,
            "transformers_version": transformers.__version__,
        }

    def infer(self, image: Image.Image, prompt: VisualPrompt) -> Sam3InferenceResult:
        processor_kwargs = build_tracker_processor_inputs(image, prompt)
        inputs = self.processor(**processor_kwargs).to("cuda")
        dtype = self.torch.bfloat16 if self.precision == "bf16" else self.torch.float32
        started = time.perf_counter()
        with self.torch.inference_mode():
            with self.torch.autocast(device_type="cuda", dtype=dtype, enabled=self.precision == "bf16"):
                outputs = self.model(**inputs, multimask_output=True)
        self.torch.cuda.synchronize()
        runtime = time.perf_counter() - started
        restored_logits = self.processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            binarize=False,
        )[0]
        # Shape is [object, hypothesis, H, W] for one image and one prompted object.
        probabilities, masks = restore_tracker_hypotheses(
            restored_logits, expected_shape=(image.height, image.width)
        )
        qualities = outputs.iou_scores.detach().float().cpu().numpy()
        qualities = np.asarray(qualities).reshape(-1)
        if not np.all(np.isfinite(qualities)):
            raise Sam3RuntimeError("SAM returned non-finite quality scores")
        if qualities.size != masks.shape[0]:
            raise Sam3RuntimeError(
                f"SAM quality count {qualities.size} differs from mask count {masks.shape[0]}"
            )
        return Sam3InferenceResult(
            tuple(np.asarray(item, dtype=bool) for item in masks),
            tuple(np.asarray(item, dtype=np.float32) for item in probabilities),
            tuple(float(item) for item in qualities),
            float(runtime),
            dict(self.runtime_metadata),
        )
