"""Resolution handling and reproducible binary-segmentation metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def _spatial_size(size: Sequence[int] | torch.Size) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"Expected (height, width), got {tuple(size)}.")
    height, width = (int(value) for value in size)
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}.")
    return height, width


def resize_logits(
    logits: torch.Tensor, size: Sequence[int] | torch.Size
) -> torch.Tensor:
    """Bilinearly resize two-class logits, preserving an optional batch axis."""

    output_size = _spatial_size(size)
    unbatched = logits.ndim == 3
    values = logits.unsqueeze(0) if unbatched else logits
    if values.ndim != 4:
        raise ValueError(
            f"Logits must be C×H×W or N×C×H×W, got {tuple(logits.shape)}."
        )
    resized = F.interpolate(
        values, size=output_size, mode="bilinear", align_corners=False
    )
    return resized[0] if unbatched else resized


def resize_probabilities(
    probabilities: torch.Tensor,
    size: Sequence[int] | torch.Size,
    *,
    channel_dim: bool | None = None,
) -> torch.Tensor:
    """Bilinearly resize class or foreground probabilities.

    Four-dimensional input is interpreted as ``N×C×H×W`` and two-dimensional
    input as ``H×W``.  For three dimensions, ``channel_dim`` can explicitly
    distinguish ``C×H×W`` from ``N×H×W``.  If omitted, a leading size of one or
    two is treated as a class dimension.
    """

    output_size = _spatial_size(size)
    original_ndim = probabilities.ndim
    interpretation: str
    if original_ndim == 4:
        values = probabilities
        interpretation = "nchw"
    elif original_ndim == 3:
        if channel_dim is None:
            channel_dim = probabilities.shape[0] in (1, 2)
        values = probabilities.unsqueeze(0 if channel_dim else 1)
        interpretation = "chw" if channel_dim else "nhw"
    elif original_ndim == 2:
        values = probabilities.unsqueeze(0).unsqueeze(0)
        interpretation = "hw"
    else:
        raise ValueError(
            "Probabilities must be H×W, C×H×W, N×H×W, or N×C×H×W; "
            f"got {tuple(probabilities.shape)}."
        )
    resized = F.interpolate(
        values, size=output_size, mode="bilinear", align_corners=False
    )
    if interpretation == "nchw":
        return resized
    if interpretation in {"chw", "nhw"}:
        return resized[0] if interpretation == "chw" else resized[:, 0]
    return resized[0, 0]


# Explicit names make the original-resolution evaluation operation auditable.
upsample_logits = resize_logits
upsample_probabilities = resize_probabilities


def logits_to_probabilities_and_mask(
    logits: torch.Tensor,
    *,
    output_size: Sequence[int] | torch.Size | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return foreground probability and argmax mask at the requested resolution.

    When ``output_size`` is provided, both class logits are first resized with
    bilinear interpolation.  Softmax and two-class argmax are then applied at
    that resolution, matching the declared primary evaluation protocol.
    """

    unbatched = logits.ndim == 3
    values = logits.unsqueeze(0) if unbatched else logits
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError(
            f"Expected N×2×H×W or 2×H×W logits, got {tuple(logits.shape)}."
        )
    if output_size is not None:
        values = resize_logits(values, output_size)
    class_probabilities = F.softmax(values, dim=1)
    foreground = class_probabilities[:, 1]
    prediction = class_probabilities.argmax(dim=1)
    if unbatched:
        return foreground[0], prediction[0]
    return foreground, prediction


def prediction_from_probabilities(
    probabilities: torch.Tensor,
    *,
    output_size: Sequence[int] | torch.Size | None = None,
) -> torch.Tensor:
    """Bilinearly resize two-class probabilities and return their argmax mask."""

    unbatched = probabilities.ndim == 3
    values = probabilities.unsqueeze(0) if unbatched else probabilities
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError(
            "Expected N×2×H×W or 2×H×W class probabilities, "
            f"got {tuple(probabilities.shape)}."
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("probabilities contain NaN or infinity.")
    if output_size is not None:
        values = resize_probabilities(values, output_size)
    prediction = values.argmax(dim=1)
    return prediction[0] if unbatched else prediction


def prediction_from_logits(
    logits: torch.Tensor,
    *,
    output_size: Sequence[int] | torch.Size | None = None,
) -> torch.Tensor:
    """Return the two-class argmax mask, optionally at original resolution."""

    return logits_to_probabilities_and_mask(
        logits, output_size=output_size
    )[1]


def foreground_probability_from_logits(
    logits: torch.Tensor,
    *,
    output_size: Sequence[int] | torch.Size | None = None,
) -> torch.Tensor:
    """Return class-one softmax probability, optionally at original resolution."""

    return logits_to_probabilities_and_mask(
        logits, output_size=output_size
    )[0]


def _two_dimensional_mask(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach()
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must describe one H×W mask, got {tuple(tensor.shape)}.")
    return tensor != 0


def _two_dimensional_probability(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(dtype=torch.float64)
    if tensor.ndim == 3 and tensor.shape[0] == 2:
        tensor = tensor[1]
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(
            "foreground_probability must describe one H×W field, "
            f"got {tuple(tensor.shape)}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("foreground_probability contains NaN or infinity.")
    return tensor


def compute_sample_metrics(
    prediction: Any | None,
    target: Any,
    *,
    foreground_probability: Any | None = None,
    inference_time_seconds: float | None = None,
    missing_prediction: bool = False,
) -> dict[str, Any]:
    """Compute one sample's overlap, probability, empty-mask, and timing fields.

    A genuinely missing prediction is scored as an empty mask instead of being
    omitted from the denominator.  Two empty masks receive IoU 1.0, matching
    the existing HiFi-CS comparison contract.
    """

    target_mask = _two_dimensional_mask(target, name="target")
    is_missing = bool(missing_prediction or prediction is None)
    if prediction is None:
        prediction_mask = torch.zeros_like(target_mask)
    else:
        prediction_mask = _two_dimensional_mask(prediction, name="prediction")
    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            f"Prediction shape {tuple(prediction_mask.shape)} does not match target "
            f"{tuple(target_mask.shape)}."
        )

    intersection = int(torch.logical_and(prediction_mask, target_mask).sum().item())
    union = int(torch.logical_or(prediction_mask, target_mask).sum().item())
    iou = 1.0 if union == 0 else intersection / union
    predicted_pixels = int(prediction_mask.sum().item())
    target_pixels = int(target_mask.sum().item())

    probability_sum = 0.0
    probability_pixels = 0
    mean_probability: float | None = None
    if foreground_probability is not None:
        probability = _two_dimensional_probability(foreground_probability)
        if probability.shape != target_mask.shape:
            raise ValueError(
                f"Probability shape {tuple(probability.shape)} does not match target "
                f"{tuple(target_mask.shape)}."
            )
        probability_sum = float(probability.sum().item())
        probability_pixels = int(probability.numel())
        mean_probability = probability_sum / probability_pixels

    if inference_time_seconds is not None:
        inference_time_seconds = float(inference_time_seconds)
        if not math.isfinite(inference_time_seconds) or inference_time_seconds < 0:
            raise ValueError("inference_time_seconds must be finite and non-negative.")

    return {
        "intersection": intersection,
        "union": union,
        "iou": float(iou),
        "predicted_foreground_pixels": predicted_pixels,
        "gt_foreground_pixels": target_pixels,
        "mean_foreground_probability": mean_probability,
        # These two additive fields prevent bias when aggregating mixed image sizes.
        "foreground_probability_sum": probability_sum,
        "probability_pixel_count": probability_pixels,
        "empty_prediction": predicted_pixels == 0,
        "empty_gt": target_pixels == 0,
        "missing_prediction": is_missing,
        "inference_time_seconds": inference_time_seconds,
    }


def precision_at_thresholds(
    ious: Sequence[float] | np.ndarray | torch.Tensor,
    thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> dict[float, float]:
    """Return strict P@X using float32 values, matching the HiFi-CS contract."""

    values = np.asarray(torch.as_tensor(ious).detach().cpu(), dtype=np.float32)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        return {float(threshold): 0.0 for threshold in thresholds}
    return {
        float(threshold): float(
            np.mean(values > np.float32(threshold), dtype=np.float64)
        )
        for threshold in thresholds
    }


def aggregate_metrics(
    samples: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """Aggregate per-sample records into mIoU, oIoU, P@50-90, and diagnostics."""

    sample_count = len(samples)
    if sample_count == 0:
        empty = {
            "test_sample_count": 0,
            "mean_iou": None,
            "overall_iou": None,
            "mean_foreground_probability": None,
            "empty_prediction_count": 0,
            "empty_gt_count": 0,
            "missing_prediction_count": 0,
            "inference_time_mean": None,
            "inference_time_median": None,
            "inference_time_p95": None,
            "precision_at_threshold_comparison": ">",
            "empty_union_iou": 1.0,
        }
        for threshold in thresholds:
            label = int(round(float(threshold) * 100))
            empty[f"p_at_{label}"] = 0.0
            empty[f"P@{label}"] = 0.0
        empty["mIoU"] = None
        empty["oIoU"] = None
        return empty

    ious = np.asarray([float(sample["iou"]) for sample in samples], dtype=np.float64)
    intersections = sum(int(sample["intersection"]) for sample in samples)
    unions = sum(int(sample["union"]) for sample in samples)
    overall_iou = 1.0 if unions == 0 else intersections / unions
    precisions = precision_at_thresholds(ious, thresholds=thresholds)

    probability_sum = sum(
        float(sample.get("foreground_probability_sum", 0.0)) for sample in samples
    )
    probability_pixels = sum(
        int(sample.get("probability_pixel_count", 0)) for sample in samples
    )
    mean_probability = (
        probability_sum / probability_pixels if probability_pixels else None
    )
    timings = np.asarray(
        [
            float(sample["inference_time_seconds"])
            for sample in samples
            if sample.get("inference_time_seconds") is not None
        ],
        dtype=np.float64,
    )

    result: dict[str, Any] = {
        "test_sample_count": sample_count,
        "mean_iou": float(ious.mean()),
        "overall_iou": float(overall_iou),
        "mean_foreground_probability": mean_probability,
        "empty_prediction_count": sum(
            bool(sample.get("empty_prediction")) for sample in samples
        ),
        "empty_gt_count": sum(bool(sample.get("empty_gt")) for sample in samples),
        "missing_prediction_count": sum(
            bool(sample.get("missing_prediction")) for sample in samples
        ),
        "inference_time_mean": float(timings.mean()) if timings.size else None,
        "inference_time_median": float(np.median(timings)) if timings.size else None,
        "inference_time_p95": (
            float(np.percentile(timings, 95)) if timings.size else None
        ),
        "precision_at_threshold_comparison": ">",
        "empty_union_iou": 1.0,
        # Explicit aliases make reports readable without changing numeric meaning.
        "mIoU": float(ious.mean()),
        "oIoU": float(overall_iou),
    }
    for threshold, precision in precisions.items():
        label = int(round(threshold * 100))
        result[f"p_at_{label}"] = precision
        result[f"P@{label}"] = precision
    return result


def _batched_masks(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(
            f"{name} must be H×W, N×H×W, or N×1×H×W; got {tuple(tensor.shape)}."
        )
    return tensor


class SegmentationMetrics:
    """Stateful metric accumulator for evaluation loops."""

    def __init__(
        self, thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS
    ) -> None:
        self.thresholds = tuple(float(value) for value in thresholds)
        self.samples: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.samples.clear()

    def update(
        self,
        prediction_or_logits: Any,
        target: Any,
        *,
        foreground_probability: Any | None = None,
        inference_time_seconds: float | Sequence[float] | None = None,
        missing_prediction: bool | Sequence[bool] = False,
    ) -> list[dict[str, Any]]:
        """Add a binary-mask batch, or dispatch ``N×2×H×W`` logits directly."""

        candidate = torch.as_tensor(prediction_or_logits)
        if candidate.ndim == 4 and candidate.shape[1] == 2:
            if foreground_probability is not None:
                raise ValueError(
                    "foreground_probability is derived automatically for logits."
                )
            return self.update_from_logits(
                candidate,
                target,
                inference_time_seconds=inference_time_seconds,
                missing_prediction=missing_prediction,
            )

        predictions = _batched_masks(candidate, name="prediction")
        targets = _batched_masks(target, name="target")
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction batch {tuple(predictions.shape)} does not match target "
                f"{tuple(targets.shape)}."
            )
        batch_size = predictions.shape[0]

        probabilities: torch.Tensor | None = None
        if foreground_probability is not None:
            probabilities = torch.as_tensor(foreground_probability).detach()
            if probabilities.ndim == 4 and probabilities.shape[1] in (1, 2):
                probabilities = probabilities[
                    :, 1 if probabilities.shape[1] == 2 else 0
                ]
            if probabilities.ndim == 2:
                probabilities = probabilities.unsqueeze(0)
            if probabilities.shape != targets.shape:
                raise ValueError(
                    f"Probability batch {tuple(probabilities.shape)} does not match "
                    f"target {tuple(targets.shape)}."
                )

        times = self._expand_per_sample(
            inference_time_seconds, batch_size, name="inference_time_seconds"
        )
        missing = self._expand_per_sample(
            missing_prediction, batch_size, name="missing_prediction"
        )
        new_samples = [
            compute_sample_metrics(
                predictions[index],
                targets[index],
                foreground_probability=(
                    probabilities[index] if probabilities is not None else None
                ),
                inference_time_seconds=(
                    None if times[index] is None else float(times[index])
                ),
                missing_prediction=bool(missing[index]),
            )
            for index in range(batch_size)
        ]
        self.samples.extend(new_samples)
        return new_samples

    def update_from_logits(
        self,
        logits: torch.Tensor,
        target: Any,
        *,
        output_size: Sequence[int] | torch.Size | None = None,
        inference_time_seconds: float | Sequence[float] | None = None,
        missing_prediction: bool | Sequence[bool] = False,
    ) -> list[dict[str, Any]]:
        """Resize logits if needed, then accumulate softmax/argmax metrics."""

        targets = _batched_masks(target, name="target")
        if output_size is None:
            output_size = targets.shape[-2:]
        probabilities, predictions = logits_to_probabilities_and_mask(
            logits, output_size=output_size
        )
        if predictions.ndim == 2:
            predictions = predictions.unsqueeze(0)
            probabilities = probabilities.unsqueeze(0)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Resized prediction {tuple(predictions.shape)} does not match target "
                f"{tuple(targets.shape)}."
            )
        return self.update(
            predictions,
            targets,
            foreground_probability=probabilities,
            inference_time_seconds=inference_time_seconds,
            missing_prediction=missing_prediction,
        )

    # Readable alias for engines that name operations rather than data formats.
    update_logits = update_from_logits

    def compute(self) -> dict[str, Any]:
        return aggregate_metrics(self.samples, thresholds=self.thresholds)

    @staticmethod
    def _expand_per_sample(
        value: Any, batch_size: int, *, name: str
    ) -> list[Any]:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [value.item()] * batch_size
            value = value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != batch_size:
                raise ValueError(
                    f"{name} has {len(value)} entries for batch size {batch_size}."
                )
            return list(value)
        return [value] * batch_size


__all__ = [
    "DEFAULT_IOU_THRESHOLDS",
    "SegmentationMetrics",
    "aggregate",
    "aggregate_metrics",
    "compute_sample_metrics",
    "foreground_probability_from_logits",
    "logits_to_probabilities_and_mask",
    "precision_at_thresholds",
    "prediction_from_logits",
    "prediction_from_probabilities",
    "resize_logits",
    "resize_probabilities",
    "upsample_logits",
    "upsample_probabilities",
]

# Compact alias used by some evaluation engines.
aggregate = aggregate_metrics
