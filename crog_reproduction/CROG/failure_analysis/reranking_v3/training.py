from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .feature_data import apply_normalizer, fit_normalizer, prepare_training_arrays, torch_batch
from .models.fullchain_ranker import FullChainRanker
from .models.pairwise_gate import V2AnchoredGate, gate_outcomes, select_v2_anchored
from .schema import artifact_identity, atomic_write_json, sha256_file


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def fullchain_loss(output: dict[str,torch.Tensor], labels: torch.Tensor, *, beta_abs: float=1.0,beta_pair: float=.25,beta_any: float=.5,beta_res: float=.01) -> tuple[torch.Tensor,dict[str,float]]:
    scores=output["scores"]; positive_count=labels.sum(1); valid=positive_count>0
    list_loss=scores.new_tensor(0.0)
    if valid.any():
        target=labels[valid]/positive_count[valid,None]
        list_loss=-(target*F.log_softmax(scores[valid],dim=1)).sum(1).mean()
    abs_loss=F.binary_cross_entropy(output["absolute_probability"],labels)
    any_loss=F.binary_cross_entropy(output["any_probability"],valid.float())
    pair_losses=[]
    for batch in range(labels.shape[0]):
        positives=torch.where(labels[batch]>0.5)[0]; negatives=torch.where(labels[batch]<0.5)[0]
        if len(positives) and len(negatives):
            differences=scores[batch,positives][:,None]-scores[batch,negatives][None,:]
            pair_losses.append(F.softplus(-differences).mean())
    pair_loss=torch.stack(pair_losses).mean() if pair_losses else scores.new_tensor(0.0)
    regularisation=output["residual"].square().mean()
    total=list_loss+beta_abs*abs_loss+beta_pair*pair_loss+beta_any*any_loss+beta_res*regularisation
    return total,{"list":float(list_loss.detach()),"absolute":float(abs_loss.detach()),"pair":float(pair_loss.detach()),"any":float(any_loss.detach()),"residual":float(regularisation.detach()),"total":float(total.detach())}


def train_smoke_suite(*, artifact_dir: str|Path, labels_path: str|Path, v2_root: str|Path, output_dir: str|Path, device: str="cpu", seed: int=20260801, epochs: int=3, batch_size: int=16) -> dict[str,Any]:
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=False); seed_everything(seed); torch_device=torch.device(device if device!="auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    arrays=prepare_training_arrays(artifact_dir,labels_path,v2_root)
    normalizers={name:fit_normalizer(arrays[name]) for name in ("head_features","depth_features","prior")}
    for name,norm in normalizers.items(): arrays[name]=apply_normalizer(arrays[name],norm)
    configs=[("native",False,True),("rgbd",True,True)]
    summaries={}; models={}
    for name,use_depth,use_prior in configs:
        seed_everything(seed); model=FullChainRanker(use_depth=use_depth,use_prior=use_prior,alpha=.5).to(torch_device); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        history=[]; indices=np.arange(len(arrays["sample_ids"]))
        for epoch in range(int(epochs)):
            rng=np.random.default_rng(seed+epoch); rng.shuffle(indices); epoch_losses=[]; model.train()
            for start in range(0,len(indices),batch_size):
                selected=indices[start:start+batch_size]; batch=torch_batch(arrays,selected,torch_device); labels=torch.from_numpy(arrays["labels"][selected]).float().to(torch_device)
                optimizer.zero_grad(set_to_none=True); result=model(**batch); loss,parts=fullchain_loss(result,labels); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); epoch_losses.append(parts["total"])
            history.append(float(np.mean(epoch_losses)))
        model.eval()
        with torch.no_grad():
            batch=torch_batch(arrays,np.arange(len(indices)),torch_device); result=model(**batch); top=result["scores"].argmax(1).cpu().numpy(); labels=arrays["labels"]
            selected=int(labels[np.arange(len(labels)),top].sum()); qtop=int(labels[:,0].sum()); oracle=int((labels.sum(1)>0).sum())
        model_path=output/f"fcer_{name}_seed{seed}.pt"; temporary=model_path.with_name(f".{model_path.name}.tmp-{os.getpid()}")
        torch.save({"state_dict":model.state_dict(),"config":{"use_depth":use_depth,"use_prior":use_prior,"alpha":.5},"normalizers":normalizers,"seed":seed},temporary); os.replace(temporary,model_path)
        models[name]=model; summaries[name]={"history":history,"q_success":qtop,"selected_success":selected,"oracle_success":oracle,"sample_count":len(labels),"checkpoint":artifact_identity(model_path),"parameters":sum(p.numel() for p in model.parameters())}
    # Functional V2-anchored gate smoke.  Baseline index 0 is explicit here;
    # full OOF training later replaces it with provenance-valid V2 selections.
    native=models["native"].eval();
    with torch.no_grad(): native_output=native(**torch_batch(arrays,np.arange(len(arrays["sample_ids"])),torch_device))
    baseline=torch.zeros(len(arrays["sample_ids"]),dtype=torch.long,device=torch_device); labels=torch.from_numpy(arrays["labels"]).float().to(torch_device)
    q=torch.from_numpy(arrays["q"]).float().to(torch_device); extras=torch.stack((native_output["scores"],native_output["absolute_probability"],q,native_output["residual"],native_output["scores"]-native_output["scores"][:,0,None],native_output["absolute_probability"]-native_output["absolute_probability"][:,0,None],torch.zeros_like(q),torch.ones_like(q)),dim=-1)
    gate=V2AnchoredGate().to(torch_device); optimizer=torch.optim.AdamW(gate.parameters(),lr=5e-4); target=gate_outcomes(labels,baseline)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); logits=gate(native_output["embedding"].detach(),baseline,extras.detach()); loss=F.cross_entropy(logits.flatten(0,1),target.flatten()); loss.backward(); optimizer.step()
    gate.eval(); probs=gate(native_output["embedding"],baseline,extras).softmax(-1); closed,_=select_v2_anchored(probs,baseline,harm_cost=5,threshold=2.0); opened,gain=select_v2_anchored(probs,baseline,harm_cost=5,threshold=0.0)
    if not torch.equal(closed,baseline): raise AssertionError("closed V2 gate did not preserve baseline exactly")
    gate_path=output/f"gate_seed{seed}.pt"; temp=gate_path.with_name(f".{gate_path.name}.tmp-{os.getpid()}"); torch.save({"state_dict":gate.state_dict(),"seed":seed},temp); os.replace(temp,gate_path)
    summary={"status":"complete","artifact_dir":str(Path(artifact_dir).resolve()),"labels_path":str(Path(labels_path).resolve()),"labels_path_sha256":sha256_file(labels_path),"v2_root":str(Path(v2_root).resolve()),"seed":seed,"epochs":epochs,"models":summaries,"gate":{"checkpoint":artifact_identity(gate_path),"closed_fallback_exact":True,"opened_switches":int((opened!=baseline).sum().cpu()),"loss":float(loss.detach().cpu())}}
    atomic_write_json(output/"summary.json",summary)
    return summary

