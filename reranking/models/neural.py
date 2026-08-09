"""Shared per-candidate MLP for direct and residual reranking experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn


ScoreMode = Literal["direct", "residual"]

__all__ = [
    "CandidateMLPScorer",
    "DirectResidualMLP",
    "SharedMLPScorer",
]


class SharedMLPScorer(nn.Module):
    """Score candidates independently with a shared MLP.

    ``mode='direct'`` returns the MLP logit.  ``mode='residual'`` returns
    ``baseline + residual_scale * tanh(MLP logit)``.  The forward interface and
    padding convention match the set-aware scorers in ``set_models.py``.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        mode: ScoreMode = "residual",
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(int(width) for width in hidden_dims)
        if input_dim <= 0 or not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("input_dim and every hidden dimension must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if mode not in {"direct", "residual"}:
            raise ValueError("mode must be 'direct' or 'residual'")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")

        self.input_dim = int(input_dim)
        self.hidden_dims = hidden_dims
        self.mode: ScoreMode = mode
        self.residual_scale = float(residual_scale)
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        previous = input_dim
        for index, width in enumerate(hidden_dims):
            layers.extend((nn.Linear(previous, width), nn.GELU()))
            if dropout and index == 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(
        self,
        features: Tensor,
        baseline_scores: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if features.ndim not in {2, 3}:
            raise ValueError("features must have shape [N, D] or [B, N, D]")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got {features.shape[-1]}"
            )
        unbatched = features.ndim == 2
        batched = features.unsqueeze(0) if unbatched else features
        batch_size, num_candidates, _ = batched.shape

        if padding_mask is None:
            padding = torch.zeros(
                (batch_size, num_candidates),
                dtype=torch.bool,
                device=batched.device,
            )
        else:
            if unbatched and padding_mask.ndim == 1:
                padding_mask = padding_mask.unsqueeze(0)
            if padding_mask.shape != (batch_size, num_candidates):
                raise ValueError("padding_mask must match [B, N]")
            padding = padding_mask.to(device=batched.device, dtype=torch.bool)

        baseline: Tensor | None = None
        if baseline_scores is not None:
            if unbatched and baseline_scores.ndim == 1:
                baseline_scores = baseline_scores.unsqueeze(0)
            if baseline_scores.ndim == 3 and baseline_scores.shape[-1] == 1:
                baseline_scores = baseline_scores.squeeze(-1)
            if baseline_scores.shape != (batch_size, num_candidates):
                raise ValueError("baseline_scores must match [B, N]")
            baseline = baseline_scores.to(device=batched.device, dtype=batched.dtype)

        if num_candidates == 0:
            raw_scores = batched.new_zeros((batch_size, 0))
        else:
            raw_scores = self.scorer(batched).squeeze(-1)
        if self.mode == "residual":
            if baseline is None:
                raise ValueError("baseline_scores are required in residual mode")
            scores = baseline + self.residual_scale * torch.tanh(raw_scores)
            scores = torch.where(padding, baseline, scores)
        else:
            scores = raw_scores.masked_fill(padding, 0.0)
        return scores.squeeze(0) if unbatched else scores


CandidateMLPScorer = SharedMLPScorer
DirectResidualMLP = SharedMLPScorer
