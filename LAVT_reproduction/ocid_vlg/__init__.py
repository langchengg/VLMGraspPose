"""OCID-VLG adaptations for the LAVT referring-segmentation pipeline."""

from .checkpoint import (
    load_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
)
from .device import (
    cuda_device_ids,
    get_device,
    resolve_device,
    select_device,
    should_pin_memory,
    should_use_cuda_ddp,
)
from .losses import (
    MulticlassDiceLoss,
    WeightedCrossEntropyLoss,
    build_loss,
    multiclass_dice_loss,
    weighted_cross_entropy,
)
from .metrics import (
    SegmentationMetrics,
    aggregate,
    aggregate_metrics,
    compute_sample_metrics,
    foreground_probability_from_logits,
    logits_to_probabilities_and_mask,
    precision_at_thresholds,
    prediction_from_logits,
    prediction_from_probabilities,
    resize_logits,
    resize_probabilities,
)

__all__ = [
    "MulticlassDiceLoss",
    "SegmentationMetrics",
    "WeightedCrossEntropyLoss",
    "aggregate",
    "aggregate_metrics",
    "build_loss",
    "compute_sample_metrics",
    "cuda_device_ids",
    "foreground_probability_from_logits",
    "get_device",
    "load_checkpoint",
    "load_training_checkpoint",
    "logits_to_probabilities_and_mask",
    "multiclass_dice_loss",
    "precision_at_thresholds",
    "prediction_from_logits",
    "prediction_from_probabilities",
    "resize_logits",
    "resize_probabilities",
    "resolve_device",
    "save_checkpoint",
    "save_training_checkpoint",
    "select_device",
    "should_pin_memory",
    "should_use_cuda_ddp",
    "weighted_cross_entropy",
]
