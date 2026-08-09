from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .models.pairwise_gate import V2AnchoredGate, gate_outcomes, select_v2_anchored
from .schema import artifact_identity, atomic_write_json


def gate_extras(
    *,
    scores: np.ndarray,
    probabilities: np.ndarray,
    q: np.ndarray,
    residual: np.ndarray,
    baseline_indices: np.ndarray,
    uncertainty: np.ndarray | None = None,
) -> np.ndarray:
    values=[np.asarray(value,dtype=np.float32) for value in (scores,probabilities,q,residual)]
    if any(value.shape!=values[0].shape for value in values) or values[0].ndim!=2 or values[0].shape[1]!=5: raise ValueError("gate score inputs must all be [N,5]")
    baseline=np.asarray(baseline_indices,dtype=np.int64); rows=np.arange(len(baseline))
    if baseline.shape!=(len(values[0]),): raise ValueError("gate baseline index shape mismatch")
    score,probability,q_value,residual_value=values
    u=np.zeros_like(score) if uncertainty is None else np.asarray(uncertainty,dtype=np.float32)
    if u.shape!=score.shape: raise ValueError("gate uncertainty must be [N,5]")
    return np.stack((score,probability,q_value,residual_value,score-score[rows,baseline,None],probability-probability[rows,baseline,None],q_value-q_value[rows,baseline,None],u),axis=-1).astype(np.float32)


def _atomic_torch_save(payload: dict[str,Any],path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try: torch.save(payload,temporary); os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def train_gate_ensemble(
    *,
    sample_ids: list[str],
    embeddings: np.ndarray,
    extras: np.ndarray,
    labels: np.ndarray,
    baseline_indices: np.ndarray,
    producing_folds: np.ndarray,
    producing_checkpoint_sha256: np.ndarray,
    output_dir: str|Path,
    seeds: tuple[int,int,int]=(20260801,20260802,20260803),
    device: str="auto",
    epochs: int=15,
    batch_size: int=512,
    resume: bool=False,
) -> dict[str,Any]:
    output=Path(output_dir); summary_path=output/"summary.json"
    if summary_path.exists():
        if not resume: raise FileExistsError(output)
        return __import__("json").loads(summary_path.read_text())
    output.mkdir(parents=True,exist_ok=resume); model_dir=output/"models"; model_dir.mkdir(exist_ok=resume)
    ids=list(map(str,sample_ids)); emb=np.asarray(embeddings,dtype=np.float32); x=np.asarray(extras,dtype=np.float32); y=np.asarray(labels,dtype=np.float32); baseline=np.asarray(baseline_indices,dtype=np.int64); folds=np.asarray(producing_folds,dtype=np.int64)
    if emb.shape[:2]!=(len(ids),5) or x.shape!=(len(ids),5,8) or y.shape!=(len(ids),5) or baseline.shape!=(len(ids),) or folds.shape!=(len(ids),): raise ValueError("OOF gate training input shape mismatch")
    if len(set(ids))!=len(ids) or set(folds.tolist())!={0,1,2}: raise ValueError("OOF gate coverage/folds invalid")
    checkpoint_values=np.asarray(producing_checkpoint_sha256).astype(str)
    if checkpoint_values.shape[0]!=len(ids) or np.any(checkpoint_values==""): raise ValueError("OOF producing checkpoint SHA coverage invalid")
    target=gate_outcomes(torch.from_numpy(y),torch.from_numpy(baseline)).numpy(); candidate=np.arange(5)[None,:]; challenger_mask=candidate!=baseline[:,None]
    counts=np.bincount(target[challenger_mask],minlength=3).astype(np.float64); class_weights=counts.sum()/np.maximum(3*counts,1)
    torch_device=torch.device("mps" if device=="auto" and torch.backends.mps.is_available() else ("cpu" if device=="auto" else device))
    paths=[]; histories=[]
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model=V2AnchoredGate(embedding_dim=emb.shape[-1],extra_dim=8,hidden_dim=128).to(torch_device); optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4); weight=torch.tensor(class_weights,dtype=torch.float32,device=torch_device); history=[]
        for epoch in range(int(epochs)):
            order=np.random.default_rng(seed+epoch).permutation(len(ids)); losses=[]; model.train()
            for start in range(0,len(order),int(batch_size)):
                index=order[start:start+int(batch_size)]; base=torch.from_numpy(baseline[index]).to(torch_device); truth=torch.from_numpy(target[index]).to(torch_device); mask=torch.from_numpy(challenger_mask[index]).to(torch_device)
                optimizer.zero_grad(set_to_none=True); logits=model(torch.from_numpy(emb[index]).to(torch_device),base,torch.from_numpy(x[index]).to(torch_device)); loss=F.cross_entropy(logits[mask],truth[mask],weight=weight); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            history.append(float(np.mean(losses)))
        path=model_dir/f"v2_anchored_gate_seed{seed}.pt"; _atomic_torch_save({"schema_version":"3.0.0","kind":"v3_v2_anchored_gate","status":"complete","state_dict":{key:value.detach().cpu() for key,value in model.state_dict().items()},"embedding_dim":emb.shape[-1],"extra_dim":8,"hidden_dim":128,"seed":seed,"epochs":epochs,"history":history,"class_counts":counts.astype(int),"class_weights":class_weights.astype(np.float32),"training_input":"scene-grouped FCER OOF predictions plus nested OOF V2-protocol anchor","sample_count":len(ids),"producing_folds":[0,1,2]},path); paths.append(path); histories.append(history)
    summary={"schema_version":"3.0.0","kind":"v3_v2_anchored_gate_ensemble","status":"complete","sample_count":len(ids),"unique_sample_count":len(set(ids)),"oof_folds":[0,1,2],"seeds":list(seeds),"models":[artifact_identity(path) for path in paths],"histories":histories,"class_counts":counts.astype(int).tolist(),"in_sample_base_predictions_rejected":True,"formal_test_read":False}
    atomic_write_json(summary_path,summary); return summary


@torch.no_grad()
def predict_gate_ensemble(
    *,
    embeddings: np.ndarray,
    extras: np.ndarray,
    baseline_indices: np.ndarray,
    checkpoint_paths: list[str|Path],
    device: str="auto",
    harm_cost: float=5.0,
    threshold: float=0.0,
) -> dict[str,np.ndarray]:
    emb=np.asarray(embeddings,dtype=np.float32); x=np.asarray(extras,dtype=np.float32); baseline=np.asarray(baseline_indices,dtype=np.int64); torch_device=torch.device("mps" if device=="auto" and torch.backends.mps.is_available() else ("cpu" if device=="auto" else device)); all_probabilities=[]; selections=[]
    for path in checkpoint_paths:
        artifact=torch.load(path,map_location="cpu",weights_only=False); model=V2AnchoredGate(embedding_dim=int(artifact["embedding_dim"]),extra_dim=int(artifact["extra_dim"]),hidden_dim=int(artifact["hidden_dim"])); model.load_state_dict(artifact["state_dict"],strict=True); model=model.to(torch_device).eval(); batches=[]
        for start in range(0,len(emb),1024): batches.append(model(torch.from_numpy(emb[start:start+1024]).to(torch_device),torch.from_numpy(baseline[start:start+1024]).to(torch_device),torch.from_numpy(x[start:start+1024]).to(torch_device)).softmax(-1).cpu())
        probabilities=torch.cat(batches); all_probabilities.append(probabilities.numpy()); selected,_=select_v2_anchored(probabilities,torch.from_numpy(baseline),harm_cost=harm_cost,threshold=threshold); selections.append(selected.numpy())
    values=np.stack(all_probabilities); seed_selected=np.stack(selections,axis=1); mean=values.mean(0); proposed,_=select_v2_anchored(torch.from_numpy(mean),torch.from_numpy(baseline),harm_cost=harm_cost,threshold=threshold); return {"probabilities":mean,"seed_probabilities":values,"seed_selected_indices":seed_selected,"mean_selected_indices":proposed.numpy()}
