from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .coordinate_mapping import candidate_feature_grid, sample_feature_roi


POOL_REGIONS = ("average", "maximum", "center", "contact", "left", "right", "axis")


def _compress_channels(vector: torch.Tensor, bins: int) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError("channel compression expects [C]")
    return F.adaptive_avg_pool1d(vector[None, None, :].float(), int(bins)).squeeze(0).squeeze(0)


def compressed_candidate_roi(
    feature: torch.Tensor,
    candidate: dict[str, Any],
    *,
    forward_affine: np.ndarray,
    roi_size: int = 7,
    channel_bins: int = 32,
) -> torch.Tensor:
    roi = sample_feature_roi(
        feature, candidate, forward_affine=forward_affine, output_size=roi_size
    ).squeeze(0)
    axis = torch.linspace(-1.0, 1.0, roi_size, device=roi.device, dtype=roi.dtype)
    vv, uu = torch.meshgrid(axis, axis, indexing="ij")
    masks = {
        "contact": (uu.abs() >= 0.35) & (uu.abs() <= 0.75) & (vv.abs() <= 0.25),
        "left": (uu <= -0.35) & (uu >= -0.75) & (vv.abs() <= 0.25),
        "right": (uu >= 0.35) & (uu <= 0.75) & (vv.abs() <= 0.25),
        "axis": vv.abs() <= 0.2,
    }
    vectors = [roi.mean((-2, -1)), roi.amax((-2, -1)), roi[:, roi_size // 2, roi_size // 2]]
    for name in ("contact", "left", "right", "axis"):
        vectors.append(roi[:, masks[name]].mean(-1))
    return torch.cat([_compress_channels(vector, channel_bins) for vector in vectors])


def pool_fullchain_features(
    feature_maps: dict[str, torch.Tensor],
    candidates_by_batch: list[list[dict[str, Any]]],
    forward_affines: list[np.ndarray],
    *,
    roi_size: int = 7,
    channel_bins: int = 32,
) -> tuple[list[str], torch.Tensor]:
    names = list(feature_maps)
    batch=len(candidates_by_batch)
    candidate_counts={len(values) for values in candidates_by_batch}
    if len(candidate_counts)!=1: raise ValueError("candidate mask required for variable K")
    candidates=candidate_counts.pop()
    first=next(iter(feature_maps.values()))
    grids=torch.cat([
        candidate_feature_grid(candidate,forward_affine=forward_affines[batch_index],output_size=roi_size,device=first.device,dtype=first.dtype)
        for batch_index,items in enumerate(candidates_by_batch) for candidate in items
    ],dim=0)
    axis=torch.linspace(-1.0,1.0,roi_size,device=first.device,dtype=first.dtype); vv,uu=torch.meshgrid(axis,axis,indexing="ij")
    masks=[(uu.abs()>=.35)&(uu.abs()<=.75)&(vv.abs()<=.25),(uu<=-.35)&(uu>=-.75)&(vv.abs()<=.25),(uu>=.35)&(uu<=.75)&(vv.abs()<=.25),vv.abs()<=.2]
    encoded=[]
    for name in names:
        value=feature_maps[name]; expanded=value[:,None].expand(-1,candidates,-1,-1,-1).reshape(batch*candidates,*value.shape[1:])
        try: roi=F.grid_sample(expanded,grids,mode="bilinear",padding_mode="zeros",align_corners=False)
        except RuntimeError: roi=F.grid_sample(expanded.cpu(),grids.cpu(),mode="bilinear",padding_mode="zeros",align_corners=False).to(value.device)
        vectors=[roi.mean((-2,-1)),roi.amax((-2,-1)),roi[:,:,roi_size//2,roi_size//2],*[roi[:,:,mask].mean(-1) for mask in masks]]
        compressed=[F.adaptive_avg_pool1d(vector[:,None].float(),channel_bins).squeeze(1) for vector in vectors]
        encoded.append(torch.cat(compressed,-1).reshape(batch,candidates,-1))
    return names,torch.stack(encoded,dim=2)


def pool_attention_features(
    attention_maps: dict[str, torch.Tensor],
    candidates_by_batch: list[list[dict[str, Any]]],
    forward_affines: list[np.ndarray],
    *,
    roi_size: int = 7,
) -> torch.Tensor:
    batch=len(candidates_by_batch); candidate_counts={len(values) for values in candidates_by_batch}
    if len(candidate_counts)!=1: raise ValueError("candidate mask required for variable K")
    candidates=candidate_counts.pop(); first=next(iter(attention_maps.values()))
    grids=torch.cat([candidate_feature_grid(candidate,forward_affine=forward_affines[batch_index],output_size=roi_size,device=first.device,dtype=first.dtype) for batch_index,items in enumerate(candidates_by_batch) for candidate in items],0)
    layers=[]
    for value in attention_maps.values():
        expanded=value[:,None].expand(-1,candidates,-1,-1,-1).reshape(batch*candidates,*value.shape[1:])
        try: roi=F.grid_sample(expanded,grids,mode="bilinear",padding_mode="zeros",align_corners=False)
        except RuntimeError: roi=F.grid_sample(expanded.cpu(),grids.cpu(),mode="bilinear",padding_mode="zeros",align_corners=False).to(value.device)
        layers.append(roi.mean((-2,-1)).reshape(batch,candidates,-1))
    return torch.stack(layers,dim=2)
