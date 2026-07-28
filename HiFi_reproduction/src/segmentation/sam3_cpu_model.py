"""Strict CPU/float32 adapters for the official Transformers SAM 3 APIs.

Imports are lazy so prompt construction, tests, and dry-runs do not allocate
PyTorch or load the gated checkpoint.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import psutil
from PIL import Image

from .sam3_prompt_builder import VisualPrompt


class Sam3CpuRuntimeError(RuntimeError):
    """Raised when the strict CPU experiment contract cannot be met."""


@dataclass(frozen=True)
class Sam3CpuInferenceResult:
    masks: tuple[np.ndarray, ...]
    probabilities: tuple[np.ndarray, ...]
    qualities: tuple[float | None, ...]
    backend: str
    model_class: str
    processor_class: str
    timings: dict[str, float]
    memory: dict[str, int]
    output_schema: dict[str, Any]
    runtime_metadata: dict[str, Any]


class _PeakRssMonitor:
    def __init__(self, interval_seconds: float = 0.02):
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.start_rss = int(self.process.memory_info().rss)
        self.peak_rss = self.start_rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_rss = max(self.peak_rss, int(self.process.memory_info().rss))

    def __enter__(self) -> "_PeakRssMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak_rss = max(self.peak_rss, int(self.process.memory_info().rss))


def configure_cpu_runtime(
    *,
    num_threads: int,
    interop_threads: int,
    environment_threads: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Set CPU controls before model construction and return audited values."""

    if int(num_threads) <= 0 or int(interop_threads) <= 0:
        raise ValueError("CPU thread counts must be positive")
    values = {
        "OMP_NUM_THREADS": int(num_threads),
        "VECLIB_MAXIMUM_THREADS": int(num_threads),
        "OPENBLAS_NUM_THREADS": int(num_threads),
        "MKL_NUM_THREADS": int(num_threads),
    }
    if environment_threads:
        values.update({str(key): int(value) for key, value in environment_threads.items()})
    for key, value in values.items():
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        os.environ[key] = str(value)

    torch, _, *_ = import_sam3_classes()
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(int(interop_threads))
    except RuntimeError as error:
        # PyTorch permits this only before parallel work starts. A mismatch is
        # unsafe; an already-equal value is acceptable for reused test processes.
        if torch.get_num_interop_threads() != int(interop_threads):
            raise Sam3CpuRuntimeError(
                "torch interop threads were already initialized to a different value"
            ) from error
    return {
        "device": "cpu",
        "dtype": "float32",
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "environment_threads": {key: os.environ[key] for key in values},
    }


def import_sam3_classes():
    try:
        import torch
        import transformers
        from transformers import (
            Sam3Config,
            Sam3Model,
            Sam3Processor,
            Sam3TrackerModel,
            Sam3TrackerProcessor,
        )
    except (ImportError, AttributeError) as error:
        raise Sam3CpuRuntimeError(
            "The isolated environment must expose Sam3Model, Sam3Processor, "
            "Sam3TrackerModel, and Sam3TrackerProcessor"
        ) from error
    return (
        torch,
        transformers,
        Sam3Config,
        Sam3Model,
        Sam3Processor,
        Sam3TrackerModel,
        Sam3TrackerProcessor,
    )


def cpu_preflight() -> dict[str, Any]:
    torch, transformers, _, model, processor, tracker_model, tracker_processor = (
        import_sam3_classes()
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "requested_device": "cpu",
        "requested_dtype": "float32",
        "cuda_available_but_unused": bool(torch.cuda.is_available()),
        "mps_available_but_unused": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "classes": [
            model.__name__,
            processor.__name__,
            tracker_model.__name__,
            tracker_processor.__name__,
        ],
    }


def build_tracker_processor_inputs(
    image: Image.Image,
    prompt: VisualPrompt,
    prompt_mode: str,
) -> dict[str, Any]:
    """Build documented `[batch, object, point, xy]` Tracker inputs."""

    supported = {"point", "box", "box_point", "box_positive_negative_points"}
    if prompt_mode not in supported:
        raise ValueError(f"unsupported Tracker prompt mode: {prompt_mode}")
    result: dict[str, Any] = {"images": image.convert("RGB"), "return_tensors": "pt"}
    if prompt_mode in {"box", "box_point", "box_positive_negative_points"}:
        result["input_boxes"] = [[list(prompt.expanded_box_xyxy)]]
    if prompt_mode in {"point", "box_point", "box_positive_negative_points"}:
        positives = list(prompt.positive_points_xy)
        negatives = (
            list(prompt.negative_points_xy)
            if prompt_mode == "box_positive_negative_points"
            else []
        )
        points = [list(point) for point in positives + negatives]
        labels = [1] * len(positives) + [0] * len(negatives)
        if not points:
            raise ValueError("point prompt mode requires at least one point")
        result["input_points"] = [[points]]
        result["input_labels"] = [[labels]]
    return result


def build_pcs_processor_inputs(
    image: Image.Image,
    prompt: VisualPrompt,
    prompt_mode: str,
    *,
    short_text: str | None = None,
) -> dict[str, Any]:
    if prompt_mode not in {"pcs_positive_box", "pcs_text_box"}:
        raise ValueError(f"unsupported PCS prompt mode: {prompt_mode}")
    result: dict[str, Any] = {
        "images": image.convert("RGB"),
        "input_boxes": [[list(prompt.expanded_box_xyxy)]],
        "input_boxes_labels": [[1]],
        "return_tensors": "pt",
    }
    if prompt_mode == "pcs_text_box":
        if not short_text or not short_text.strip():
            raise ValueError("pcs_text_box requires a deterministic short noun phrase")
        result["text"] = short_text.strip()
    return result


def _tensor_schema(outputs: Any) -> dict[str, Any]:
    keys = list(outputs.keys()) if hasattr(outputs, "keys") else []
    tensors: dict[str, Any] = {}
    for key in keys:
        value = getattr(outputs, key, None)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            detached = value.detach()
            finite = (
                bool(detached.isfinite().all().item())
                if detached.is_floating_point() or detached.is_complex()
                else True
            )
            tensors[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device) if hasattr(value, "device") else None,
                "finite": finite,
            }
    return {"keys": keys, "tensors": tensors}


def _tracker_hypotheses(
    restored_logits: Any,
    qualities: Any | None,
    *,
    expected_shape: tuple[int, int],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[float | None, ...]]:
    array = (
        restored_logits.detach().float().cpu().numpy()
        if hasattr(restored_logits, "detach")
        else np.asarray(restored_logits)
    )
    if array.ndim != 4 or array.shape[0] != 1:
        raise Sam3CpuRuntimeError(
            f"Tracker post-processing must return [1,M,H,W], got {array.shape}"
        )
    logits = np.asarray(array[0], dtype=np.float32)
    if tuple(logits.shape[-2:]) != tuple(expected_shape):
        raise Sam3CpuRuntimeError(
            f"restored mask shape {logits.shape[-2:]} != {expected_shape}"
        )
    if not np.all(np.isfinite(logits)):
        raise Sam3CpuRuntimeError("Tracker returned non-finite mask logits")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    masks = probabilities >= 0.5
    if qualities is None:
        quality_values: tuple[float | None, ...] = (None,) * int(masks.shape[0])
    else:
        quality_array = qualities.detach().float().cpu().numpy().reshape(-1)
        if quality_array.size != masks.shape[0] or not np.isfinite(quality_array).all():
            raise Sam3CpuRuntimeError("Tracker quality output is invalid or misaligned")
        quality_values = tuple(float(value) for value in quality_array)
    return (
        tuple(np.asarray(mask, dtype=bool) for mask in masks),
        tuple(np.asarray(value, dtype=np.float32) for value in probabilities),
        quality_values,
    )


class TransformersSam3Cpu:
    """Load one official backend once and reuse it for batch-size-one inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        revision: str,
        backend: str = "tracker",
        processor_size: int = 1008,
        num_threads: int = 8,
        interop_threads: int = 1,
        environment_threads: Mapping[str, int] | None = None,
    ):
        if backend not in {"tracker", "pcs"}:
            raise ValueError("backend must be tracker or pcs")
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("revision must be an immutable 40-character lowercase commit SHA")
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"pinned local SAM 3 model directory is missing: {model_path}")
        manifest_path = model_path.parent / "model_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"pinned SAM 3 model manifest is missing: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("model_id") != "facebook/sam3"
            or manifest.get("resolved_revision_sha") != revision
            or Path(manifest.get("local_model_path", "")).expanduser().resolve()
            != model_path
        ):
            raise Sam3CpuRuntimeError(
                "local model path, official model ID, and pinned revision do not match"
            )
        if processor_size not in {1008, 560}:
            raise ValueError("processor_size must be 1008 or the documented PCS fallback 560")
        if backend == "tracker" and processor_size != 1008:
            raise Sam3CpuRuntimeError(
                "Transformers documents custom 560 resolution for Sam3Model only; "
                "Tracker 560 is not enabled without a successful official smoke test"
            )
        runtime = configure_cpu_runtime(
            num_threads=num_threads,
            interop_threads=interop_threads,
            environment_threads=environment_threads,
        )
        (
            torch,
            transformers,
            sam3_config_class,
            sam3_model_class,
            sam3_processor_class,
            tracker_model_class,
            tracker_processor_class,
        ) = import_sam3_classes()
        self.torch = torch
        self.backend = backend
        self.processor_size = int(processor_size)
        process = psutil.Process()
        rss_before = int(process.memory_info().rss)
        started = time.perf_counter()
        with _PeakRssMonitor() as load_monitor:
            processor_kwargs = {
                "local_files_only": True,
                "size": {"height": self.processor_size, "width": self.processor_size},
            }
            if backend == "tracker":
                model_class, processor_class = tracker_model_class, tracker_processor_class
                model_config = None
            else:
                model_class, processor_class = sam3_model_class, sam3_processor_class
                model_config = sam3_config_class.from_pretrained(
                    str(model_path), local_files_only=True
                )
                if self.processor_size != 1008:
                    model_config.image_size = self.processor_size
            self.processor = processor_class.from_pretrained(
                str(model_path), **processor_kwargs
            )
            model_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "use_safetensors": True,
                "dtype": torch.float32,
            }
            if model_config is not None:
                model_kwargs["config"] = model_config
            signature = inspect.signature(model_class.from_pretrained)
            # Do not claim `low_cpu_mem_usage`: Transformers 5.14 accepts then
            # discards it. The absence is recorded explicitly.
            self.model = model_class.from_pretrained(str(model_path), **model_kwargs)
            self.model.to(torch.device("cpu")).eval()
        self.load_time_seconds = float(time.perf_counter() - started)
        rss_after = int(process.memory_info().rss)
        devices = {str(parameter.device) for parameter in self.model.parameters()}
        dtypes = {str(parameter.dtype) for parameter in self.model.parameters()}
        if devices != {"cpu"} or dtypes != {"torch.float32"}:
            raise Sam3CpuRuntimeError(
                f"strict CPU/float32 placement failed: devices={devices}, dtypes={dtypes}"
            )
        self.runtime_metadata = {
            **cpu_preflight(),
            **runtime,
            "model_id": "facebook/sam3",
            "model_path": str(model_path),
            "model_revision": revision,
            "backend": backend,
            "model_class": model_class.__name__,
            "processor_class": processor_class.__name__,
            "processor_size": self.processor_size,
            "batch_size": 1,
            "dtype_keyword": "dtype",
            "low_cpu_mem_usage_effective": False,
            "use_safetensors": True,
            "model_load_time_seconds": self.load_time_seconds,
            "rss_before_model_load_bytes": rss_before,
            "rss_after_model_load_bytes": rss_after,
            "peak_rss_during_model_load_bytes": load_monitor.peak_rss,
            "from_pretrained_signature": str(signature),
            "transformers_version": transformers.__version__,
        }

    def infer(
        self,
        image: Image.Image,
        prompt: VisualPrompt,
        *,
        prompt_mode: str,
        short_text: str | None = None,
    ) -> Sam3CpuInferenceResult:
        image = image.convert("RGB")
        process = psutil.Process()
        preprocess_started = time.perf_counter()
        if self.backend == "tracker":
            processor_kwargs = build_tracker_processor_inputs(image, prompt, prompt_mode)
        else:
            processor_kwargs = build_pcs_processor_inputs(
                image, prompt, prompt_mode, short_text=short_text
            )
        inputs = self.processor(**processor_kwargs)
        inputs = inputs.to(device=torch_device(self.torch), dtype=self.torch.float32)
        for key, value in inputs.items():
            if hasattr(value, "device") and str(value.device) != "cpu":
                raise Sam3CpuRuntimeError(f"processor tensor {key} escaped CPU")
        preprocess_seconds = float(time.perf_counter() - preprocess_started)
        inference_started = time.perf_counter()
        with _PeakRssMonitor() as rss:
            with self.torch.inference_mode():
                if self.backend == "tracker":
                    outputs = self.model(**inputs, multimask_output=True)
                else:
                    outputs = self.model(**inputs)
        inference_seconds = float(time.perf_counter() - inference_started)
        schema = _tensor_schema(outputs)
        post_started = time.perf_counter()
        if self.backend == "tracker":
            restored = self.processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                binarize=False,
            )[0]
            masks, probabilities, qualities = _tracker_hypotheses(
                restored,
                getattr(outputs, "iou_scores", None),
                expected_shape=(image.height, image.width),
            )
        else:
            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                mask_threshold=0.5,
                target_sizes=inputs["original_sizes"].detach().cpu().tolist(),
            )[0]
            masks_array = result["masks"].detach().cpu().numpy().astype(bool)
            if tuple(masks_array.shape[-2:]) != (image.height, image.width):
                raise Sam3CpuRuntimeError("PCS masks were not restored to source resolution")
            score_array = result.get("scores")
            score_values = (
                score_array.detach().float().cpu().numpy().reshape(-1)
                if score_array is not None
                else np.asarray([], dtype=np.float32)
            )
            masks = tuple(np.asarray(mask, dtype=bool) for mask in masks_array)
            probabilities = tuple(mask.astype(np.float32) for mask in masks)
            qualities = (
                tuple(float(value) for value in score_values)
                if score_values.size == len(masks)
                else (None,) * len(masks)
            )
        postprocess_seconds = float(time.perf_counter() - post_started)
        result = Sam3CpuInferenceResult(
            masks=masks,
            probabilities=probabilities,
            qualities=qualities,
            backend=self.backend,
            model_class=self.runtime_metadata["model_class"],
            processor_class=self.runtime_metadata["processor_class"],
            timings={
                "preprocess_seconds": preprocess_seconds,
                "inference_seconds": inference_seconds,
                "postprocess_seconds": postprocess_seconds,
                "total_sample_seconds": preprocess_seconds
                + inference_seconds
                + postprocess_seconds,
                "model_load_seconds": self.load_time_seconds,
            },
            memory={
                "rss_before_inference_bytes": rss.start_rss,
                "peak_rss_during_inference_bytes": rss.peak_rss,
                "peak_rss_during_model_load_bytes": int(
                    self.runtime_metadata["peak_rss_during_model_load_bytes"]
                ),
                "peak_rss_bytes": max(
                    rss.peak_rss,
                    int(self.runtime_metadata["peak_rss_during_model_load_bytes"]),
                ),
                "rss_after_inference_bytes": int(process.memory_info().rss),
            },
            output_schema=schema,
            runtime_metadata=dict(self.runtime_metadata),
        )
        del outputs, inputs
        gc.collect()
        return result


def torch_device(torch_module: Any):
    """A single auditable device constructor used by every model/input path."""

    return torch_module.device("cpu")
