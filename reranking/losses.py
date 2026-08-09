"""Query-normalized losses for padded candidate sets.

All ranking comparisons are confined to a single row/query.  A ``True`` value
in ``padding_mask`` marks an invalid padded candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


__all__ = [
    "CompositeRerankingLoss",
    "LossBreakdown",
    "QueryComposition",
    "combined_reranking_loss",
    "listwise_softmax_ce_loss",
    "query_composition",
    "query_normalized_bce_loss",
    "same_query_ranknet_loss",
]


@dataclass(frozen=True)
class QueryComposition:
    """Mutually exclusive query-label composition counts."""

    total: int
    empty: int
    no_positive: int
    all_positive: int
    mixed: int

    @property
    def non_empty(self) -> int:
        return self.total - self.empty

    def __add__(self, other: "QueryComposition") -> "QueryComposition":
        return QueryComposition(
            total=self.total + other.total,
            empty=self.empty + other.empty,
            no_positive=self.no_positive + other.no_positive,
            all_positive=self.all_positive + other.all_positive,
            mixed=self.mixed + other.mixed,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "non_empty": self.non_empty,
            "empty": self.empty,
            "no_positive": self.no_positive,
            "all_positive": self.all_positive,
            "mixed": self.mixed,
        }


@dataclass(frozen=True)
class LossBreakdown:
    """Differentiable loss terms plus explicit query eligibility metadata."""

    total: Tensor
    bce: Tensor
    ranknet: Tensor
    listwise: Tensor
    residual: Tensor
    composition: QueryComposition
    active_queries: int

    def detached_metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "loss": float(self.total.detach().cpu()),
            "bce": float(self.bce.detach().cpu()),
            "ranknet": float(self.ranknet.detach().cpu()),
            "listwise": float(self.listwise.detach().cpu()),
            "residual": float(self.residual.detach().cpu()),
            "active_queries": self.active_queries,
        }
        metrics.update(
            {
                f"queries_{key}": value
                for key, value in self.composition.as_dict().items()
            }
        )
        return metrics


def _prepare(
    scores: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None,
    *,
    validate_values: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if scores.ndim != 2 or labels.shape != scores.shape:
        raise ValueError("scores and labels must have matching shape [B, N]")
    if not scores.is_floating_point() or (
        validate_values and not torch.isfinite(scores).all()
    ):
        raise ValueError("scores must be finite floating-point logits")
    if padding_mask is None:
        padding = torch.zeros_like(scores, dtype=torch.bool)
    else:
        if padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if padding_mask.shape != scores.shape:
            raise ValueError("padding_mask must match scores")
        padding = padding_mask.to(device=scores.device, dtype=torch.bool)
    labels = labels.to(device=scores.device, dtype=scores.dtype)
    # Do not boolean-index labels merely to validate them.  On MPS that lowers
    # to ``nonzero_mps`` and forces an expensive synchronization on every loss
    # invocation.  Replacing padding with the valid value zero first preserves
    # the exact validation contract without materializing an index tensor.
    safe_labels = torch.where(padding, torch.zeros_like(labels), labels)
    if validate_values and (
        not torch.isfinite(safe_labels).all()
        or not bool(((safe_labels == 0) | (safe_labels == 1)).all().item())
    ):
        raise ValueError("non-padding labels must be finite binary values")
    return scores, safe_labels, padding


def _query_composition_from_prepared(
    labels: Tensor, padding: Tensor
) -> QueryComposition:
    valid = ~padding
    counts = valid.sum(dim=1)
    positives = (labels * valid).sum(dim=1)
    empty = counts == 0
    no_positive = (counts > 0) & (positives == 0)
    all_positive = (counts > 0) & (positives == counts)
    mixed = (positives > 0) & (positives < counts)
    return QueryComposition(
        total=int(labels.shape[0]),
        empty=int(empty.sum().item()),
        no_positive=int(no_positive.sum().item()),
        all_positive=int(all_positive.sum().item()),
        mixed=int(mixed.sum().item()),
    )


def query_composition(
    labels: Tensor, padding_mask: Tensor | None = None
) -> QueryComposition:
    """Classify queries as empty, no-positive, all-positive, or mixed."""

    dummy_scores = torch.zeros_like(labels, dtype=torch.float32)
    _, labels, padding = _prepare(dummy_scores, labels, padding_mask)
    return _query_composition_from_prepared(labels, padding)


def _zero(scores: Tensor) -> Tensor:
    return scores.sum() * 0.0


def _query_mean(
    values: Tensor,
    valid: Tensor,
    scores: Tensor,
    *,
    all_queries_nonempty: bool = False,
) -> Tensor:
    counts = valid.sum(dim=1)
    if all_queries_nonempty:
        per_query = (values * valid).sum(dim=1) / counts.to(values.dtype)
        return per_query.mean()
    eligible = counts > 0
    if not bool(eligible.any().item()):
        return _zero(scores)
    per_query = (values * valid).sum(dim=1) / counts.clamp_min(1).to(values.dtype)
    # Candidate batches normally contain no empty query.  Avoiding an all-True
    # boolean gather keeps this common path free of MPS ``nonzero`` while
    # retaining the original fallback and exact empty-query semantics.
    if bool(eligible.all().item()):
        return per_query.mean()
    return per_query[eligible].mean()


def query_normalized_bce_loss(
    scores: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
    *,
    pos_weight: float | None = None,
    _prepared: bool = False,
    _all_queries_nonempty: bool = False,
) -> Tensor:
    """BCE where each non-empty query contributes equal total weight."""

    if _prepared:
        if padding_mask is None:
            raise ValueError("prepared BCE inputs require a padding mask")
        padding = padding_mask
    else:
        scores, labels, padding = _prepare(scores, labels, padding_mask)
    if pos_weight is not None and pos_weight <= 0.0:
        raise ValueError("pos_weight must be positive")
    positive_weight = (
        scores.new_tensor(float(pos_weight)) if pos_weight is not None else None
    )
    candidate_loss = functional.binary_cross_entropy_with_logits(
        scores, labels, reduction="none", pos_weight=positive_weight
    )
    if positive_weight is not None:
        # BCE's pos_weight changes the total mass of positive-heavy queries.
        # Renormalize by the same per-candidate class multiplier so every
        # non-empty query still contributes one equal-weight mean.
        valid = ~padding
        class_multiplier = torch.where(
            labels == 1,
            positive_weight,
            torch.ones_like(labels),
        )
        denominator = (class_multiplier * valid).sum(dim=1)
        eligible = valid.any(dim=1)
        if not bool(eligible.any().item()):
            return _zero(scores)
        per_query = (candidate_loss * valid).sum(dim=1) / denominator.clamp_min(1e-9)
        return per_query.mean() if bool(eligible.all().item()) else per_query[eligible].mean()
    return _query_mean(
        candidate_loss,
        ~padding,
        scores,
        all_queries_nonempty=_all_queries_nonempty,
    )


def same_query_ranknet_loss(
    scores: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
    *,
    _prepared: bool = False,
) -> Tensor:
    """Mean RankNet loss over all positive/negative pairs within each query.

    Queries with no positive or no negative candidate have no valid ordering
    pair and are excluded rather than assigned a fabricated pair.
    """

    if _prepared:
        if padding_mask is None:
            raise ValueError("prepared RankNet inputs require a padding mask")
        padding = padding_mask
    else:
        scores, labels, padding = _prepare(scores, labels, padding_mask)
    valid = ~padding
    positive = valid & (labels == 1)
    negative = valid & (labels == 0)
    pair_mask = positive.unsqueeze(2) & negative.unsqueeze(1)
    pair_count = pair_mask.sum(dim=(1, 2))
    eligible = pair_count > 0
    if not bool(eligible.any().item()):
        return _zero(scores)
    margins = scores.unsqueeze(2) - scores.unsqueeze(1)
    pair_loss = functional.softplus(-margins)
    per_query = (pair_loss * pair_mask).sum(dim=(1, 2)) / pair_count.clamp_min(1)
    return (per_query * eligible).sum() / eligible.sum()


def listwise_softmax_ce_loss(
    scores: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
    *,
    temperature: float = 1.0,
    _prepared: bool = False,
) -> Tensor:
    """Multiple-positive listwise softmax cross entropy per query.

    The loss is ``logsumexp(all) - logsumexp(positives)``.  No-positive and
    all-positive queries are excluded: the former has no target mass and the
    latter has no ranking contrast (its mathematical loss is identically zero).
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if _prepared:
        if padding_mask is None:
            raise ValueError("prepared listwise inputs require a padding mask")
        padding = padding_mask
    else:
        scores, labels, padding = _prepare(scores, labels, padding_mask)
    valid = ~padding
    positive = valid & (labels == 1)
    valid_count = valid.sum(dim=1)
    positive_count = positive.sum(dim=1)
    eligible = (positive_count > 0) & (positive_count < valid_count)
    if not bool(eligible.any().item()):
        return _zero(scores)
    scaled = scores / temperature
    negative_infinity = torch.full_like(scaled, float("-inf"))
    all_logits = torch.where(valid, scaled, negative_infinity)
    positive_logits = torch.where(positive, scaled, negative_infinity)
    # Avoid all--inf logsumexp rows for ineligible queries. Their loss is
    # explicitly zeroed below, and the substitute prevents NaN gradients.
    all_logits = torch.where(eligible[:, None], all_logits, torch.zeros_like(all_logits))
    positive_logits = torch.where(
        eligible[:, None], positive_logits, torch.zeros_like(positive_logits)
    )
    per_query = torch.logsumexp(all_logits, dim=1) - torch.logsumexp(
        positive_logits, dim=1
    )
    return (per_query * eligible).sum() / eligible.sum()


def _query_normalized_residual_penalty(
    scores: Tensor,
    baseline_scores: Tensor,
    padding: Tensor,
    *,
    validate_values: bool = True,
    all_queries_nonempty: bool = False,
) -> Tensor:
    if baseline_scores.ndim == 1:
        baseline_scores = baseline_scores.unsqueeze(0)
    if baseline_scores.shape != scores.shape:
        raise ValueError("baseline_scores must match scores")
    baseline_scores = baseline_scores.to(device=scores.device, dtype=scores.dtype)
    if validate_values and not torch.isfinite(baseline_scores).all():
        raise ValueError("baseline_scores must be finite")
    return _query_mean(
        (scores - baseline_scores).square(),
        ~padding,
        scores,
        all_queries_nonempty=all_queries_nonempty,
    )


def combined_reranking_loss(
    scores: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
    *,
    baseline_scores: Tensor | None = None,
    bce_weight: float = 1.0,
    ranknet_weight: float = 0.0,
    listwise_weight: float = 0.0,
    residual_weight: float = 0.0,
    pos_weight: float | None = None,
    temperature: float = 1.0,
    compute_inactive_terms: bool = True,
    _validate_values: bool = True,
    _composition: QueryComposition | None = None,
) -> LossBreakdown:
    """Compute an independently weighted combination of reranking losses."""

    weights = (bce_weight, ranknet_weight, listwise_weight, residual_weight)
    if any(weight < 0.0 for weight in weights) or not any(
        weight > 0.0 for weight in weights
    ):
        raise ValueError("loss weights must be non-negative with at least one positive")
    prepared_scores, prepared_labels, padding = _prepare(
        scores,
        labels,
        padding_mask,
        validate_values=_validate_values,
    )
    composition = (
        _query_composition_from_prepared(prepared_labels, padding)
        if _composition is None
        else _composition
    )
    if composition.total != int(prepared_labels.shape[0]):
        raise ValueError("precomputed composition batch size does not match scores")
    all_queries_nonempty = composition.empty == 0
    bce = (
        query_normalized_bce_loss(
            prepared_scores,
            prepared_labels,
            padding,
            pos_weight=pos_weight,
            _prepared=True,
            _all_queries_nonempty=all_queries_nonempty,
        )
        if bce_weight > 0.0 or compute_inactive_terms
        else _zero(prepared_scores)
    )
    ranknet = (
        same_query_ranknet_loss(
            prepared_scores,
            prepared_labels,
            padding,
            _prepared=True,
        )
        if ranknet_weight > 0.0 or compute_inactive_terms
        else _zero(prepared_scores)
    )
    listwise = (
        listwise_softmax_ce_loss(
            prepared_scores,
            prepared_labels,
            padding,
            temperature=temperature,
            _prepared=True,
        )
        if listwise_weight > 0.0 or compute_inactive_terms
        else _zero(prepared_scores)
    )
    if residual_weight > 0.0:
        if baseline_scores is None:
            raise ValueError("baseline_scores are required for residual regularization")
        residual = _query_normalized_residual_penalty(
            prepared_scores,
            baseline_scores,
            padding,
            validate_values=_validate_values,
            all_queries_nonempty=all_queries_nonempty,
        )
    else:
        residual = _zero(prepared_scores)
    total = (
        bce_weight * bce
        + ranknet_weight * ranknet
        + listwise_weight * listwise
        + residual_weight * residual
    )
    active_queries = (
        composition.non_empty
        if bce_weight > 0.0 or residual_weight > 0.0
        else composition.mixed
    )
    return LossBreakdown(
        total=total,
        bce=bce,
        ranknet=ranknet,
        listwise=listwise,
        residual=residual,
        composition=composition,
        active_queries=active_queries,
    )


class CompositeRerankingLoss(nn.Module):
    """Module wrapper around :func:`combined_reranking_loss`."""

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        ranknet_weight: float = 0.0,
        listwise_weight: float = 0.0,
        residual_weight: float = 0.0,
        pos_weight: float | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        weights = (bce_weight, ranknet_weight, listwise_weight, residual_weight)
        if any(weight < 0.0 for weight in weights) or not any(
            weight > 0.0 for weight in weights
        ):
            raise ValueError(
                "loss weights must be non-negative with at least one positive"
            )
        if pos_weight is not None and pos_weight <= 0.0:
            raise ValueError("pos_weight must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.bce_weight = float(bce_weight)
        self.ranknet_weight = float(ranknet_weight)
        self.listwise_weight = float(listwise_weight)
        self.residual_weight = float(residual_weight)
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        self.temperature = float(temperature)

    @property
    def config(self) -> dict[str, float | None]:
        return {
            "bce_weight": self.bce_weight,
            "ranknet_weight": self.ranknet_weight,
            "listwise_weight": self.listwise_weight,
            "residual_weight": self.residual_weight,
            "pos_weight": self.pos_weight,
            "temperature": self.temperature,
        }

    def breakdown(
        self,
        scores: Tensor,
        labels: Tensor,
        padding_mask: Tensor | None = None,
        *,
        baseline_scores: Tensor | None = None,
    ) -> LossBreakdown:
        return combined_reranking_loss(
            scores,
            labels,
            padding_mask,
            baseline_scores=baseline_scores,
            **self.config,
        )

    def active_breakdown(
        self,
        scores: Tensor,
        labels: Tensor,
        padding_mask: Tensor | None = None,
        *,
        baseline_scores: Tensor | None = None,
        composition: QueryComposition | None = None,
    ) -> LossBreakdown:
        """Compute the training loss without evaluating zero-weight terms.

        ``breakdown`` intentionally retains all diagnostic component values.
        Epoch training only consumes the weighted total, composition, and
        active-query count, so evaluating inactive RankNet/Listwise terms there
        adds substantial MPS synchronization without changing optimization.
        """

        return combined_reranking_loss(
            scores,
            labels,
            padding_mask,
            baseline_scores=baseline_scores,
            compute_inactive_terms=False,
            _validate_values=composition is None,
            _composition=composition,
            **self.config,
        )

    def forward(
        self,
        scores: Tensor,
        labels: Tensor,
        padding_mask: Tensor | None = None,
        *,
        baseline_scores: Tensor | None = None,
    ) -> Tensor:
        return self.breakdown(
            scores,
            labels,
            padding_mask,
            baseline_scores=baseline_scores,
        ).total
