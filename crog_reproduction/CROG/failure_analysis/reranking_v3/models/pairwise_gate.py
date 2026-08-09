from __future__ import annotations

import numpy as np
import torch
from torch import nn


def gate_outcomes(labels: torch.Tensor, baseline_indices: torch.Tensor) -> torch.Tensor:
    batch,candidates=labels.shape; baseline=labels.gather(1,baseline_indices[:,None]).bool()
    result=torch.full((batch,candidates),2,dtype=torch.long,device=labels.device)
    result[(~baseline)&labels.bool()]=0; result[baseline&(~labels.bool())]=1
    return result


class V2AnchoredGate(nn.Module):
    def __init__(self, embedding_dim: int=256, extra_dim: int=8, hidden_dim: int=128):
        super().__init__(); input_dim=embedding_dim*3+extra_dim
        self.network=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden_dim,64),nn.GELU(),nn.Linear(64,3))

    def forward(self, embeddings: torch.Tensor, baseline_indices: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        batch,candidates,dim=embeddings.shape
        baseline=embeddings.gather(1,baseline_indices[:,None,None].expand(-1,1,dim)).expand(-1,candidates,-1)
        return self.network(torch.cat((baseline,embeddings,embeddings-baseline,extras),-1))


def select_v2_anchored(probabilities: torch.Tensor, baseline_indices: torch.Tensor, *, harm_cost: float, threshold: float, uncertainty: torch.Tensor | None=None, kappa: float=0.0, coverage: torch.Tensor | None=None) -> tuple[torch.Tensor,torch.Tensor]:
    baseline_indices=baseline_indices.to(device=probabilities.device,dtype=torch.long)
    gain=probabilities[...,0]-float(harm_cost)*probabilities[...,1]
    if uncertainty is not None: gain=gain-float(kappa)*uncertainty
    if coverage is not None: gain=torch.where(coverage.bool(),gain,torch.full_like(gain,float("-inf")))
    gain.scatter_(1,baseline_indices[:,None],float("-inf"))
    best_gain,best=gain.max(1); selected=torch.where(best_gain>float(threshold),best,baseline_indices)
    return selected,best_gain
