from __future__ import annotations

import torch
from torch import nn


class SetContextEncoder(nn.Module):
    """No positional encoding: candidate permutation produces output permutation."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, layers: int = 2, heads: int = 4, dropout: float = .1):
        super().__init__()
        self.input=nn.Linear(input_dim,hidden_dim)
        layer=nn.TransformerEncoderLayer(hidden_dim,heads,hidden_dim*2,dropout=dropout,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,layers)
        self.norm=nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor, candidate_mask: torch.Tensor | None=None) -> torch.Tensor:
        padding=None if candidate_mask is None else ~candidate_mask.bool()
        return self.norm(self.encoder(self.input(values),src_key_padding_mask=padding))

