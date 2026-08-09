"""Controlled residual neural scorers for the unified reranking experiment.

Every scorer accepts padded candidate sets and implements
``native_score + alpha * tanh(residual)``.  Padding is represented by ``True``
in ``padding_mask``.  The set-aware models contain no positional encoding, so
their outputs are permutation equivariant.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn


def _prepare_inputs(
    features: Tensor,
    native_scores: Tensor,
    padding_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, bool]:
    if features.ndim not in {2, 3}:
        raise ValueError("features must have shape [N, D] or [B, N, D]")
    if not features.is_floating_point():
        raise TypeError("features must be floating point")
    unbatched = features.ndim == 2
    features = features.unsqueeze(0) if unbatched else features
    if unbatched and native_scores.ndim == 1:
        native_scores = native_scores.unsqueeze(0)
    if native_scores.ndim == 3 and native_scores.shape[-1] == 1:
        native_scores = native_scores.squeeze(-1)
    if native_scores.shape != features.shape[:2]:
        raise ValueError("native_scores must match features [B, N]")
    native_scores = native_scores.to(device=features.device, dtype=features.dtype)
    if padding_mask is None:
        padding = torch.zeros(
            features.shape[:2], dtype=torch.bool, device=features.device
        )
    else:
        if unbatched and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if padding_mask.shape != features.shape[:2]:
            raise ValueError("padding_mask must match features [B, N]")
        padding = padding_mask.to(device=features.device, dtype=torch.bool)
    return features, native_scores, padding, unbatched


class _ResidualScorer(nn.Module):
    def __init__(self, *, alpha: float) -> None:
        super().__init__()
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)

    def _forward_residual(
        self,
        features: Tensor,
        native_scores: Tensor,
        padding_mask: Tensor | None,
        residual_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        features, native, padding, unbatched = _prepare_inputs(
            features, native_scores, padding_mask
        )
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got {features.shape[-1]}"
            )
        # This explicit bypass makes alpha=0 a bit-exact native-order control.
        if self.alpha == 0.0:
            result = native.clone()
        elif features.shape[1] == 0:
            result = native
        else:
            residual = residual_fn(features, padding)
            if residual.shape != native.shape:
                raise RuntimeError("residual scorer returned an invalid shape")
            result = native + self.alpha * torch.tanh(residual)
            result = torch.where(padding, native, result)
        return result.squeeze(0) if unbatched else result


class ResidualMLPScorer(_ResidualScorer):
    """Exact LayerNorm-64-GELU-Dropout(0.1)-32-GELU-output MLP."""

    def __init__(self, input_dim: int, *, alpha: float = 0.5) -> None:
        super().__init__(alpha=alpha)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        features: Tensor,
        native_scores: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        return self._forward_residual(
            features,
            native_scores,
            padding_mask,
            lambda values, padding: self.network(values).squeeze(-1),
        )


class LinearResidualScorer(_ResidualScorer):
    """Single affine residual used for the controlled R2 baseline."""

    def __init__(self, input_dim: int, *, alpha: float = 0.5) -> None:
        super().__init__(alpha=alpha)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.linear = nn.Linear(input_dim, 1)

    def forward(
        self,
        features: Tensor,
        native_scores: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        return self._forward_residual(
            features,
            native_scores,
            padding_mask,
            lambda values, padding: self.linear(values).squeeze(-1),
        )


class DeepSetsResidualScorer(_ResidualScorer):
    """Hidden-64 DeepSets scorer with masked mean and max context."""

    def __init__(
        self,
        input_dim: int,
        *,
        alpha: float = 0.5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(alpha=alpha)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _residual(self, features: Tensor, padding: Tensor) -> Tensor:
        valid = ~padding
        hidden = self.phi(features).masked_fill(padding.unsqueeze(-1), 0.0)
        counts = valid.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        mean = hidden.sum(dim=1) / counts
        maximum = hidden.masked_fill(padding.unsqueeze(-1), -torch.inf).amax(dim=1)
        maximum = torch.where(
            valid.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
        )
        context = torch.cat((mean, maximum), dim=-1)
        context = context.unsqueeze(1).expand(-1, features.shape[1], -1)
        return self.head(torch.cat((hidden, context), dim=-1)).squeeze(-1)

    def forward(
        self,
        features: Tensor,
        native_scores: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        return self._forward_residual(
            features, native_scores, padding_mask, self._residual
        )


class _AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, padding: Tensor) -> Tensor:
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=padding,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + attended)
        hidden = self.feed_forward_norm(hidden + self.feed_forward(hidden))
        return hidden.masked_fill(padding.unsqueeze(-1), 0.0)


class SetTransformerResidualScorer(_ResidualScorer):
    """Hidden-64, four-head set transformer with one or two masked blocks."""

    def __init__(
        self,
        input_dim: int,
        *,
        alpha: float = 0.5,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
    ) -> None:
        super().__init__(alpha=alpha)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_blocks not in {1, 2}:
            raise ValueError("num_blocks must be 1 or 2")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.blocks = nn.ModuleList(
            _AttentionBlock(hidden_dim, num_heads) for _ in range(num_blocks)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def _residual(self, features: Tensor, padding: Tensor) -> Tensor:
        batch_size, num_candidates, _ = features.shape
        output = features.new_zeros((batch_size, num_candidates))
        # MultiheadAttention is undefined when every key in one batch row is masked.
        active = (~padding).any(dim=1)
        if bool(active.any().item()):
            local_padding = padding[active]
            hidden = self.input_encoder(features[active])
            hidden = hidden.masked_fill(local_padding.unsqueeze(-1), 0.0)
            for block in self.blocks:
                hidden = block(hidden, local_padding)
            output[active] = self.head(hidden).squeeze(-1).masked_fill(
                local_padding, 0.0
            )
        return output

    def forward(
        self,
        features: Tensor,
        native_scores: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        return self._forward_residual(
            features, native_scores, padding_mask, self._residual
        )


class _MessagePass(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        hidden: Tensor,
        edge_hidden: Tensor,
        edge_valid: Tensor,
    ) -> Tensor:
        batch_size, count, node_dim = hidden.shape
        source = hidden.unsqueeze(2).expand(batch_size, count, count, node_dim)
        destination = hidden.unsqueeze(1).expand(
            batch_size, count, count, node_dim
        )
        messages = self.message(torch.cat((source, destination, edge_hidden), dim=-1))
        messages = messages.masked_fill(~edge_valid.unsqueeze(-1), 0.0)
        denominator = edge_valid.sum(dim=1).clamp_min(1).unsqueeze(-1)
        aggregate = messages.sum(dim=1) / denominator.to(messages.dtype)
        update = self.update(torch.cat((hidden, aggregate), dim=-1))
        return self.norm(hidden + update)


class CompleteGraphGNNResidualScorer(_ResidualScorer):
    """Two-pass edge-conditioned complete-graph GNN implemented in PyTorch."""

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        *,
        alpha: float = 0.5,
        node_hidden_dim: int = 64,
        edge_hidden_dim: int = 32,
        num_message_passes: int = 2,
    ) -> None:
        super().__init__(alpha=alpha)
        if input_dim <= 0 or edge_dim <= 0:
            raise ValueError("input_dim and edge_dim must be positive")
        if node_hidden_dim <= 0 or edge_hidden_dim <= 0:
            raise ValueError("node and edge hidden dimensions must be positive")
        if num_message_passes != 2:
            raise ValueError("the controlled GNN requires exactly two message passes")
        self.input_dim = int(input_dim)
        self.edge_dim = int(edge_dim)
        self.node_hidden_dim = int(node_hidden_dim)
        self.edge_hidden_dim = int(edge_hidden_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, node_hidden_dim),
            nn.GELU(),
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.LayerNorm(node_hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.GELU(),
        )
        self.message_passes = nn.ModuleList(
            _MessagePass(node_hidden_dim, edge_hidden_dim)
            for _ in range(num_message_passes)
        )
        self.head = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.GELU(),
            nn.Linear(node_hidden_dim, 1),
        )

    def forward(
        self,
        features: Tensor,
        native_scores: Tensor,
        padding_mask: Tensor | None = None,
        *,
        edge_features: Tensor,
    ) -> Tensor:
        prepared_features, native, padding, unbatched = _prepare_inputs(
            features, native_scores, padding_mask
        )
        if prepared_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got {prepared_features.shape[-1]}"
            )
        if unbatched and edge_features.ndim == 3:
            edge_features = edge_features.unsqueeze(0)
        expected = (*prepared_features.shape[:2], prepared_features.shape[1], self.edge_dim)
        if edge_features.shape != expected:
            raise ValueError(f"edge_features must have shape {expected}")
        edge_features = edge_features.to(
            device=prepared_features.device, dtype=prepared_features.dtype
        )
        if self.alpha == 0.0:
            result = native.clone()
        elif prepared_features.shape[1] == 0:
            result = native
        else:
            valid = ~padding
            count = prepared_features.shape[1]
            off_diagonal = ~torch.eye(
                count, dtype=torch.bool, device=prepared_features.device
            )
            edge_valid = (
                valid.unsqueeze(2)
                & valid.unsqueeze(1)
                & off_diagonal.unsqueeze(0)
            )
            hidden = self.node_encoder(prepared_features).masked_fill(
                padding.unsqueeze(-1), 0.0
            )
            edge_hidden = self.edge_encoder(edge_features)
            for message_pass in self.message_passes:
                hidden = message_pass(hidden, edge_hidden, edge_valid)
                hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
            residual = self.head(hidden).squeeze(-1)
            result = native + self.alpha * torch.tanh(residual)
            result = torch.where(padding, native, result)
        return result.squeeze(0) if unbatched else result


__all__ = [
    "CompleteGraphGNNResidualScorer",
    "DeepSetsResidualScorer",
    "LinearResidualScorer",
    "ResidualMLPScorer",
    "SetTransformerResidualScorer",
]
