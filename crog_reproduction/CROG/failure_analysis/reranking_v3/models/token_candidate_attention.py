from __future__ import annotations

import torch
from torch import nn


class TokenCandidateCrossAttention(nn.Module):
    def __init__(self, roi_dim: int = 224, token_dim: int = 512, sentence_dim: int = 1024, dynamic_dim: int = 2305, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.query = nn.Linear(roi_dim, hidden_dim)
        self.key = nn.Linear(token_dim, hidden_dim)
        self.value = nn.Linear(token_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.sentence = nn.Linear(sentence_dim,hidden_dim)
        self.dynamic = nn.Linear(dynamic_dim,hidden_dim)
        self.roi_dynamic = nn.Linear(roi_dim,hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim*4+1,output_dim),nn.LayerNorm(output_dim),nn.GELU())

    def forward(self, roi: torch.Tensor, tokens: torch.Tensor, sentence: torch.Tensor, dynamic: torch.Tensor, token_ids: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        batch,candidates,_=roi.shape; length=tokens.shape[1]
        query=self.query(roi).reshape(batch*candidates,1,-1)
        keys=self.key(tokens)[:,None].expand(-1,candidates,-1,-1).reshape(batch*candidates,length,-1)
        values=self.value(tokens)[:,None].expand(-1,candidates,-1,-1).reshape(batch*candidates,length,-1)
        pad=(token_ids==0)
        valid=(~pad).sum(-1)
        # Exclude SOT and the last non-padding token (EOT) from semantic attention.
        semantic_pad=pad.clone(); semantic_pad[:,0]=True
        semantic_pad[torch.arange(batch,device=token_ids.device),torch.clamp(valid-1,min=0)]=True
        # Degenerate one-token inputs fall back to EOT rather than all-masked attention.
        all_masked=semantic_pad.all(-1)
        semantic_pad[all_masked,torch.clamp(valid[all_masked]-1,min=0)]=False
        expanded_mask=semantic_pad[:,None].expand(-1,candidates,-1).reshape(batch*candidates,length)
        attended,weights=self.attention(query,keys,values,key_padding_mask=expanded_mask,need_weights=True)
        attended=attended.reshape(batch,candidates,-1); weights=weights.reshape(batch,candidates,length)
        probabilities=torch.clamp(weights,1e-8,1.0)
        entropy=-(probabilities*probabilities.log()).sum(-1,keepdim=True)
        sent=self.sentence(sentence)[:,None].expand(-1,candidates,-1)
        dyn=self.dynamic(dynamic)[:,None].expand(-1,candidates,-1)
        interaction=dyn*self.roi_dynamic(roi)
        return self.output(torch.cat((attended,sent,interaction,self.query(roi),entropy),-1)),weights

