"""D1-only configurable residual MLP used by the fair R2/R3/R4/R6 grid."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class D1ResidualMLPScorer(nn.Module):
    """Residual scorer with a predeclared shared architecture grid."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or len(hidden_dims) != 2 or min(hidden_dims) <= 0:
            raise ValueError("D1 MLP dimensions must be positive")
        if not 0.0 <= dropout < 1.0 or alpha < 0:
            raise ValueError("D1 MLP dropout/alpha are invalid")
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(map(int, hidden_dims))
        self.dropout = float(dropout)
        self.alpha = float(alpha)
        first, second = self.hidden_dims
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, first),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Linear(second, 1),
        )

    def forward(
        self,
        features: Tensor,
        native_scores: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if features.ndim not in {2, 3} or not features.is_floating_point():
            raise ValueError("D1 MLP features must be floating [N,D] or [B,N,D]")
        unbatched = features.ndim == 2
        values = features.unsqueeze(0) if unbatched else features
        native = native_scores.unsqueeze(0) if unbatched else native_scores
        if native.ndim == 3 and native.shape[-1] == 1:
            native = native.squeeze(-1)
        if values.shape[-1] != self.input_dim or native.shape != values.shape[:2]:
            raise ValueError("D1 MLP input/native shapes differ from its contract")
        if padding_mask is None:
            padding = torch.zeros(
                values.shape[:2], dtype=torch.bool, device=values.device
            )
        else:
            padding = padding_mask.unsqueeze(0) if unbatched else padding_mask
            if padding.shape != values.shape[:2]:
                raise ValueError("D1 MLP padding shape differs from candidate tensor")
            padding = padding.to(device=values.device, dtype=torch.bool)
        native = native.to(device=values.device, dtype=values.dtype)
        if self.alpha == 0:
            score = native.clone()
        else:
            residual = self.network(values).squeeze(-1)
            score = native + self.alpha * torch.tanh(residual)
            score = torch.where(padding, native, score)
        return score.squeeze(0) if unbatched else score
