from __future__ import annotations

import torch
from torch import nn

from .crop_encoder import RGBMapCropEncoder
from .latent_adapter import MultiScaleLatentAdapter
from .scalar_encoder import ScalarHeadEncoder
from .set_encoder import SetContextEncoder
from .token_candidate_attention import TokenCandidateCrossAttention


def bounded_q_scores(q: torch.Tensor, residual: torch.Tensor, alpha: float, epsilon: float=1e-6) -> torch.Tensor:
    base=torch.logit(torch.clamp(q,epsilon,1-epsilon))
    return base+float(alpha)*torch.tanh(residual)


def deterministic_order(scores: torch.Tensor, q_ranks: torch.Tensor, candidate_ids: list[list[str]]) -> list[list[int]]:
    result=[]
    for batch_index in range(scores.shape[0]):
        result.append(sorted(range(scores.shape[1]),key=lambda index:(-float(scores[batch_index,index]),int(q_ranks[batch_index,index]),str(candidate_ids[batch_index][index]))))
    return result


class FullChainRanker(nn.Module):
    def __init__(self, *, head_dim: int=146, depth_dim: int=13, prior_dim: int=80, crop_channels: int=17, hidden_dim: int=256, alpha: float=.5, use_depth: bool=False, use_prior: bool=True, use_crop: bool=True, use_latent: bool=True, text_mode: str="token", use_attention: bool=True, use_set: bool=True, latent_layers: list[int] | str="all", dropout: float=.1):
        super().__init__(); self.alpha=float(alpha); self.use_depth=bool(use_depth); self.use_prior=bool(use_prior); self.use_crop=bool(use_crop); self.use_latent=bool(use_latent); self.text_mode=str(text_mode); self.use_attention=bool(use_attention); self.use_set=bool(use_set)
        if self.text_mode not in {"none","sentence","token"}: raise ValueError("text_mode must be none, sentence or token")
        selected_layers=list(range(12)) if latent_layers=="all" else list(map(int,latent_layers))
        if not selected_layers or min(selected_layers)<0 or max(selected_layers)>=12: raise ValueError("invalid latent layer selection")
        self.register_buffer("latent_layer_mask",torch.tensor([index in selected_layers for index in range(12)],dtype=torch.float32).view(1,1,12,1),persistent=True)
        self.scalar=ScalarHeadEncoder(head_dim,128,64,dropout)
        native_channels=crop_channels if use_depth else crop_channels-2
        self.crop=RGBMapCropEncoder(native_channels,64) if use_crop else None
        self.latent=MultiScaleLatentAdapter(224,12,16,64) if use_latent else None
        self.text=TokenCandidateCrossAttention() if self.text_mode=="token" else None
        self.sentence_only=nn.Sequential(nn.Linear(1024,64),nn.LayerNorm(64),nn.GELU()) if self.text_mode=="sentence" else None
        self.attention=nn.Sequential(nn.Linear(3*17,32),nn.GELU()) if use_attention else None
        self.depth=nn.Sequential(nn.Linear(depth_dim,32),nn.GELU()) if use_depth else None
        self.prior=nn.Sequential(nn.Linear(prior_dim,64),nn.GELU()) if use_prior else None
        input_dim=64+(64 if use_crop else 0)+(64 if use_latent else 0)+(64 if self.text_mode!="none" else 0)+(32 if use_attention else 0)+(32 if use_depth else 0)+(64 if use_prior else 0)
        self.fuse=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout))
        self.set_encoder=SetContextEncoder(hidden_dim,hidden_dim,2,4,dropout) if use_set else None
        self.residual_head=nn.Linear(hidden_dim,1)
        self.absolute_head=nn.Linear(hidden_dim,1)
        self.any_head=nn.Sequential(nn.Linear(hidden_dim,64),nn.GELU(),nn.Linear(64,1))

    def forward(self, *, head: torch.Tensor, q: torch.Tensor, depth: torch.Tensor | None=None, latent: torch.Tensor | None=None, attention_roi: torch.Tensor | None=None, tokens: torch.Tensor | None=None, sentence: torch.Tensor | None=None, dynamic: torch.Tensor | None=None, token_ids: torch.Tensor | None=None, crops: torch.Tensor | None=None, prior: torch.Tensor | None=None, candidate_mask: torch.Tensor | None=None) -> dict[str,torch.Tensor]:
        batch,candidates=head.shape[:2]
        scalar=self.scalar(head); masked_latent=None
        if self.latent is not None or self.text is not None:
            if latent is None: raise ValueError("latent input required by configured model")
            masked_latent=latent*self.latent_layer_mask
        token_attention=head.new_zeros((batch,candidates,0 if token_ids is None else token_ids.shape[1]))
        parts=[scalar]
        if self.crop is not None:
            if crops is None: raise ValueError("crop input required by configured model")
            if self.use_depth:
                crop_input=crops
            else:
                # Drop relative_depth and depth_valid (channels 3 and 4).
                crop_input=torch.cat((crops[:,:,:3],crops[:,:,5:]),dim=2)
            parts.append(self.crop(crop_input))
        if self.latent is not None: parts.append(self.latent(masked_latent))
        if self.text is not None:
            if tokens is None or sentence is None or dynamic is None or token_ids is None or masked_latent is None: raise ValueError("token interaction inputs are incomplete")
            roi_query=masked_latent.sum(dim=-2)/self.latent_layer_mask.sum().clamp_min(1)
            text,token_attention=self.text(roi_query,tokens,sentence,dynamic,token_ids); parts.append(text)
        elif self.sentence_only is not None:
            if sentence is None: raise ValueError("sentence input required by configured model")
            parts.append(self.sentence_only(sentence)[:,None].expand(-1,candidates,-1))
        if self.attention is not None:
            if attention_roi is None: raise ValueError("decoder attention input required by configured model")
            parts.append(self.attention(attention_roi.flatten(-2)))
        if self.use_depth:
            if depth is None: raise ValueError("depth input required by configured model")
            parts.append(self.depth(depth))
        if self.use_prior:
            if prior is None: raise ValueError("V2 prior input required by configured model")
            parts.append(self.prior(prior))
        fused=self.fuse(torch.cat(parts,-1))
        embedding=self.set_encoder(fused,candidate_mask) if self.set_encoder is not None else fused
        residual=self.residual_head(embedding).squeeze(-1)
        probability=torch.sigmoid(self.absolute_head(embedding).squeeze(-1))
        if candidate_mask is None:
            context=embedding.mean(1)
        else:
            weight=candidate_mask.float().unsqueeze(-1); context=(embedding*weight).sum(1)/weight.sum(1).clamp_min(1)
        any_probability=torch.sigmoid(self.any_head(context).squeeze(-1))
        return {"scores":bounded_q_scores(q,residual,self.alpha),"residual":residual,"absolute_probability":probability,"any_probability":any_probability,"embedding":embedding,"token_attention":token_attention}
