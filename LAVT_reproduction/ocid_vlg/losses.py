"""Loss functions for two-class OCID-VLG target segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_CLASS_WEIGHTS = (0.9, 1.1)


def _class_index_target(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Normalize ``N×H×W`` or ``N×1×H×W`` labels for PyTorch CE/one-hot."""

    if target.ndim == logits.ndim and target.shape[1] == 1:
        target = target[:, 0]
    expected_ndim = logits.ndim - 1
    if target.ndim != expected_ndim:
        raise ValueError(
            f"Target must have {expected_ndim} dimensions (or a singleton "
            f"channel); got shape {tuple(target.shape)} for logits "
            f"{tuple(logits.shape)}."
        )
    if target.shape[0] != logits.shape[0] or target.shape[1:] != logits.shape[2:]:
        raise ValueError(
            f"Target shape {tuple(target.shape)} does not match logits "
            f"{tuple(logits.shape)}."
        )
    return target.to(device=logits.device, dtype=torch.long)


def weighted_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weights: Sequence[float] = DEFAULT_CLASS_WEIGHTS,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """LAVT's weighted cross entropy with weights allocated on ``logits.device``."""

    if logits.ndim < 3:
        raise ValueError(f"Expected N×C×... logits, got {tuple(logits.shape)}.")
    if len(class_weights) != logits.shape[1]:
        raise ValueError(
            f"Expected {logits.shape[1]} class weights, got {len(class_weights)}."
        )
    labels = _class_index_target(target, logits)
    weights = torch.as_tensor(
        class_weights, dtype=logits.dtype, device=logits.device
    )
    return F.cross_entropy(
        logits,
        labels,
        weight=weights,
        ignore_index=ignore_index,
        reduction=reduction,
    )


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    include_background: bool = True,
) -> torch.Tensor:
    """Compute stable soft Dice loss over foreground and background classes.

    Dice is calculated per sample and per class over all spatial dimensions,
    then averaged.  Adding ``epsilon`` to numerator and denominator gives a
    finite, well-defined value even when a class is absent from a target.
    """

    if logits.ndim < 3:
        raise ValueError(f"Expected N×C×... logits, got {tuple(logits.shape)}.")
    if logits.shape[1] != 2:
        raise ValueError(
            f"OCID-VLG Dice expects exactly two classes, got {logits.shape[1]}."
        )
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")

    labels = _class_index_target(target, logits)
    if torch.any((labels < 0) | (labels >= logits.shape[1])):
        raise ValueError("Dice targets must contain only class indices 0 and 1.")

    probabilities = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).movedim(-1, 1)
    one_hot = one_hot.to(dtype=probabilities.dtype)

    spatial_dims = tuple(range(2, logits.ndim))
    intersection = (probabilities * one_hot).sum(dim=spatial_dims)
    cardinality = (probabilities + one_hot).sum(dim=spatial_dims)
    scores = (2.0 * intersection + epsilon) / (cardinality + epsilon)
    if not include_background:
        scores = scores[:, 1:]
    return 1.0 - scores.mean()


# Short functional alias used in resolved configuration files.
dice_loss = multiclass_dice_loss


class MulticlassDiceLoss(nn.Module):
    """``nn.Module`` wrapper for :func:`multiclass_dice_loss`."""

    def __init__(
        self, *, epsilon: float = 1e-6, include_background: bool = True
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        self.epsilon = float(epsilon)
        self.include_background = bool(include_background)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return multiclass_dice_loss(
            logits,
            target,
            epsilon=self.epsilon,
            include_background=self.include_background,
        )


class WeightedCrossEntropyLoss(nn.Module):
    """Device-independent module wrapper for the original weighted CE."""

    def __init__(
        self,
        *,
        class_weights: Sequence[float] = DEFAULT_CLASS_WEIGHTS,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.class_weights = tuple(float(value) for value in class_weights)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return weighted_cross_entropy(
            logits,
            target,
            class_weights=self.class_weights,
            ignore_index=self.ignore_index,
        )


def build_loss(name: str, *, epsilon: float = 1e-6) -> nn.Module:
    """Build one explicitly selected loss; CE and Dice are never mixed."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"dice", "multiclass_dice"}:
        return MulticlassDiceLoss(epsilon=epsilon, include_background=True)
    if normalized in {"cross_entropy", "ce", "weighted_cross_entropy"}:
        return WeightedCrossEntropyLoss()
    raise ValueError(
        f"Unsupported loss {name!r}; use 'dice' or 'cross_entropy'."
    )


__all__ = [
    "DEFAULT_CLASS_WEIGHTS",
    "MulticlassDiceLoss",
    "WeightedCrossEntropyLoss",
    "build_loss",
    "dice_loss",
    "multiclass_dice_loss",
    "weighted_cross_entropy",
]
