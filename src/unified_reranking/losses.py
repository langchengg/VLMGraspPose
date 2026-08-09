"""Query-equal objectives for the unified reranking experiment."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _prepare(
    scores: Tensor, labels: Tensor, padding_mask: Tensor | None
) -> tuple[Tensor, Tensor, Tensor]:
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
        labels = labels.unsqueeze(0)
        if padding_mask is not None and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
    if scores.ndim != 2 or labels.shape != scores.shape:
        raise ValueError("scores and labels must have matching [B, N] shapes")
    labels = labels.to(device=scores.device, dtype=scores.dtype)
    if not bool(((labels == 0) | (labels == 1)).all().item()):
        raise ValueError("labels must be binary 0/1")
    if padding_mask is None:
        padding = torch.zeros_like(scores, dtype=torch.bool)
    else:
        if padding_mask.shape != scores.shape:
            raise ValueError("padding_mask must match scores")
        padding = padding_mask.to(device=scores.device, dtype=torch.bool)
    return scores, labels, padding


def _zero(scores: Tensor) -> Tensor:
    return scores.sum() * 0.0


def query_equal_bce_loss(
    scores: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Binary cross entropy with every non-empty query receiving equal mass."""

    scores, labels, padding = _prepare(scores, labels, padding_mask)
    valid = ~padding
    eligible = valid.any(dim=1)
    if not bool(eligible.any().item()):
        return _zero(scores)
    candidate = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    per_query = (candidate * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return per_query[eligible].mean()


def all_pairs_ranknet_loss(
    scores: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """RankNet over every positive-negative pair, with equal query weight."""

    scores, labels, padding = _prepare(scores, labels, padding_mask)
    valid = ~padding
    positive = valid & (labels == 1)
    negative = valid & (labels == 0)
    pair_mask = positive.unsqueeze(2) & negative.unsqueeze(1)
    pair_count = pair_mask.sum(dim=(1, 2))
    eligible = pair_count > 0
    if not bool(eligible.any().item()):
        return _zero(scores)
    pair_loss = F.softplus(-(scores.unsqueeze(2) - scores.unsqueeze(1)))
    per_query = (pair_loss * pair_mask).sum(dim=(1, 2)) / pair_count.clamp_min(1)
    return per_query[eligible].mean()


def multi_positive_listwise_loss(
    scores: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Negative log softmax mass on all positive candidates in each query."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scores, labels, padding = _prepare(scores, labels, padding_mask)
    valid = ~padding
    positive = valid & (labels == 1)
    valid_count = valid.sum(dim=1)
    positive_count = positive.sum(dim=1)
    eligible = (positive_count > 0) & (positive_count < valid_count)
    if not bool(eligible.any().item()):
        return _zero(scores)
    scaled = scores / float(temperature)
    negative_infinity = torch.full_like(scaled, -torch.inf)
    all_logits = torch.where(valid, scaled, negative_infinity)
    positive_logits = torch.where(positive, scaled, negative_infinity)
    # Replace ineligible rows before logsumexp to keep their gradients finite.
    all_logits = torch.where(eligible[:, None], all_logits, torch.zeros_like(all_logits))
    positive_logits = torch.where(
        eligible[:, None], positive_logits, torch.zeros_like(positive_logits)
    )
    per_query = torch.logsumexp(all_logits, dim=1) - torch.logsumexp(
        positive_logits, dim=1
    )
    return per_query[eligible].mean()


def jacquard_margin_ranknet_loss(
    scores: Tensor,
    labels: Tensor,
    jacquard_margins: Tensor,
    padding_mask: Tensor | None = None,
    *,
    beta: float = 1.0,
) -> Tensor:
    """Training-only margin-weighted all-positive/all-negative RankNet.

    The continuous margins must be evaluator-derived values clipped to [-1, 1].
    Each eligible query retains equal outer weight; its weighted pair sum is
    divided by pair count, so ``beta`` still changes a one-pair query's loss.
    """

    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    original_shape = scores.shape
    scores, labels, padding = _prepare(scores, labels, padding_mask)
    if len(original_shape) == 1 and jacquard_margins.ndim == 1:
        jacquard_margins = jacquard_margins.unsqueeze(0)
    if jacquard_margins.shape != scores.shape:
        raise ValueError("jacquard_margins must match scores")
    margins = jacquard_margins.to(device=scores.device, dtype=scores.dtype)
    if not bool(torch.isfinite(margins).all().item()):
        raise ValueError("jacquard_margins must be finite")
    if not bool(((margins >= -1) & (margins <= 1)).all().item()):
        raise ValueError("jacquard_margins must be clipped to [-1, 1]")
    valid = ~padding
    positive = valid & (labels == 1)
    negative = valid & (labels == 0)
    pair_mask = positive.unsqueeze(2) & negative.unsqueeze(1)
    eligible = pair_mask.any(dim=(1, 2))
    if not bool(eligible.any().item()):
        return _zero(scores)
    weights = 1.0 + float(beta) * torch.abs(
        margins.unsqueeze(2) - margins.unsqueeze(1)
    )
    pair_loss = F.softplus(-(scores.unsqueeze(2) - scores.unsqueeze(1)))
    masked_weights = weights * pair_mask
    pair_count = pair_mask.sum(dim=(1, 2))
    per_query = (masked_weights * pair_loss).sum(dim=(1, 2)) / pair_count.clamp_min(1)
    return per_query[eligible].mean()


__all__ = [
    "all_pairs_ranknet_loss",
    "jacquard_margin_ranknet_loss",
    "multi_positive_listwise_loss",
    "query_equal_bce_loss",
]
