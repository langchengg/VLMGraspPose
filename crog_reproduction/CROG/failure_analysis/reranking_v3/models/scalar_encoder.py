from __future__ import annotations

import torch
from torch import nn


class ScalarHeadEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim), nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

