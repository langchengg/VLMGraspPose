from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .schema import read_jsonl, sha256_file
from .v2_prior import V2_PRIOR_DIM, load_v2_oof_prior


ARRAY_KEYS = ("head_features","depth_features","latent_rois","attention_rois","tokens","sentence","dynamic","token_ids","crops")


def load_fullchain_arrays(artifact_dir: str | Path, *, max_samples: int | None=None) -> dict[str,Any]:
    root=Path(artifact_dir); manifest=json.loads((root/"artifact_manifest.json").read_text())
    if manifest.get("status")!="complete": raise ValueError("full-chain artifact is incomplete")
    records=list(read_jsonl(root/"index.jsonl"));
    if max_samples is not None: records=records[:int(max_samples)]
    by_shard:dict[str,list[tuple[int,dict[str,Any]]]]={}
    for output_index,record in enumerate(records): by_shard.setdefault(record["shard"],[]).append((output_index,record))
    arrays={key:[None]*len(records) for key in ARRAY_KEYS}
    for shard_name,requested in by_shard.items():
        with np.load(root/"shards"/shard_name) as shard:
            for output_index,record in requested:
                offset=int(record["offset"])
                if str(shard["sample_ids"][offset])!=str(record["sample_id"]): raise ValueError("feature shard/index mismatch")
                for key in ARRAY_KEYS: arrays[key][output_index]=np.asarray(shard[key][offset])
    result={key:np.stack(values) for key,values in arrays.items()}
    result.update({"sample_ids":np.asarray([r["sample_id"] for r in records]),"records":records,"schema":json.loads((root/"feature_schema.json").read_text()),"artifact_sha256":manifest["content_sha256"]})
    return result


def load_labels_for_ids(path: str | Path, sample_ids: list[str]) -> np.ndarray:
    wanted=set(map(str,sample_ids)); lookup={}
    for record in read_jsonl(path):
        sample_id=str(record["sample_id"])
        if sample_id in wanted:
            lookup[sample_id]=np.asarray([float(item["candidate_correct"]) for item in record["candidate_labels"]],np.float32)
    missing=wanted-set(lookup)
    if missing: raise ValueError(f"missing corrected labels: {sorted(missing)[:5]}")
    return np.stack([lookup[str(sample_id)] for sample_id in sample_ids])


def fit_normalizer(values: np.ndarray, axes: tuple[int,...]=(0,1)) -> dict[str,np.ndarray]:
    array=np.asarray(values,dtype=np.float64); median=np.nanmedian(array,axis=axes); filled=np.where(np.isfinite(array),array,median); mean=filled.mean(axis=axes); scale=filled.std(axis=axes); scale=np.where(scale<1e-6,1.0,scale)
    return {"median":median.astype(np.float32),"mean":mean.astype(np.float32),"scale":scale.astype(np.float32)}


def apply_normalizer(values: np.ndarray, normalizer: dict[str,np.ndarray]) -> np.ndarray:
    array=np.asarray(values,dtype=np.float32); filled=np.where(np.isfinite(array),array,normalizer["median"]); return ((filled-normalizer["mean"])/normalizer["scale"]).astype(np.float32)


def prepare_training_arrays(artifact_dir: str | Path, labels_path: str | Path, v2_root: str | Path) -> dict[str,Any]:
    result=load_fullchain_arrays(artifact_dir); ids=list(map(str,result["sample_ids"])); result["labels"]=load_labels_for_ids(labels_path,ids); result["prior"],result["prior_valid"]=load_v2_oof_prior(v2_root,ids)
    q_index=result["schema"]["head_feature_names"].index("g0_q_raw"); result["q"]=result["head_features"][:,:,q_index].astype(np.float32)
    return result


def torch_batch(arrays: dict[str,Any], indices: np.ndarray, device: torch.device) -> dict[str,torch.Tensor]:
    mapping={
        "head":"head_features","depth":"depth_features","latent":"latent_rois","attention_roi":"attention_rois","tokens":"tokens","sentence":"sentence","dynamic":"dynamic","token_ids":"token_ids","crops":"crops","prior":"prior","q":"q",
    }
    result={}
    for target,source in mapping.items():
        value=torch.from_numpy(np.asarray(arrays[source][indices]))
        if target!="token_ids": value=value.float()
        result[target]=value.to(device)
    return result

