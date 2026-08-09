from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .schema import read_jsonl, sha256_file
from .models.pairwise_gate import V2AnchoredGate, gate_outcomes, select_v2_anchored
from .schema import artifact_identity, atomic_write_json


V2_PRIOR_DIM = 80


def verify_oof_provenance(v2_root: str | Path) -> dict[str, Any]:
    root = Path(v2_root)
    base_npz = root / "oof_base/oof_base_predictions.npz"
    setrank_npz = root / "oof_primary/oof_setrank_predictions.npz"
    with np.load(base_npz) as base, np.load(setrank_npz) as setrank:
        ids = list(map(str, base["sample_ids"])); folds = np.asarray(base["fold_ids"],dtype=np.int64)
        if ids != list(map(str,setrank["sample_ids"])):
            raise ValueError("V2 OOF base/SetRank sample order mismatch")
    fold_by_id=dict(zip(ids,folds.tolist(),strict=True)); seen_base=set(); seen_set=set(); checkpoint_hashes=set(); observed_hashes={}
    def checked(path, expected):
        key=str(path)
        if key not in observed_hashes: observed_hashes[key]=sha256_file(path)
        if observed_hashes[key]!=expected: raise ValueError(f"V2 OOF checkpoint changed: {path}")
    for record in read_jsonl(root/"oof_base/oof_provenance.jsonl"):
        sample_id=str(record["sample_id"]); heldout=int(record["heldout_fold"])
        if fold_by_id.get(sample_id)!=heldout: raise ValueError("V2 base OOF heldout fold mismatch")
        for checkpoint in record["base_checkpoints"]:
            if heldout in set(map(int,checkpoint["fit_folds"])): raise ValueError("V2 base OOF in-sample provenance")
            for name in ("critic","latent"):
                checked(checkpoint[name],checkpoint[f"{name}_sha256"])
                checkpoint_hashes.add(checkpoint[f"{name}_sha256"])
        seen_base.add(sample_id)
    for record in read_jsonl(root/"oof_primary/oof_setrank_provenance.jsonl"):
        sample_id=str(record["sample_id"]); heldout=int(record["heldout_fold"])
        if fold_by_id.get(sample_id)!=heldout: raise ValueError("V2 SetRank OOF heldout fold mismatch")
        for checkpoint in record["setrank_checkpoints"]:
            if heldout in set(map(int,checkpoint["fit_folds"])): raise ValueError("V2 SetRank OOF in-sample provenance")
            checked(checkpoint["path"],checkpoint["sha256"])
            checkpoint_hashes.add(checkpoint["sha256"])
        seen_set.add(sample_id)
    if seen_base!=set(ids) or seen_set!=set(ids): raise ValueError("V2 OOF provenance coverage incomplete")
    return {"sample_count":len(ids),"folds":sorted(set(folds.tolist())),"checkpoint_count":len(checkpoint_hashes),"base_sha256":sha256_file(base_npz),"setrank_sha256":sha256_file(setrank_npz),"passed":True}


def _load_prior(
    base_path: str | Path,
    setrank_path: str | Path,
    sample_ids: list[str],
) -> tuple[np.ndarray,np.ndarray]:
    with np.load(base_path) as base, np.load(setrank_path) as setrank:
        source_ids=list(map(str,base["sample_ids"])); lookup={sample_id:index for index,sample_id in enumerate(source_ids)}
        if source_ids != list(map(str,setrank["sample_ids"])):
            raise ValueError("V2 base/SetRank prior sample order mismatch")
        critic=np.asarray(base["critic_scores"]); embedding=np.asarray(base["critic_embeddings"],dtype=np.float32)
        latent=np.asarray(base["latent_scores"]); residual=np.asarray(base["latent_residuals"])
        scores=np.asarray(setrank["scores"]); probabilities=np.asarray(setrank["probabilities"])
        rows=[]; validity=[]
        for sample_id in sample_ids:
            index=lookup.get(str(sample_id))
            if index is None:
                rows.append(np.zeros((5,V2_PRIOR_DIM),np.float32)); validity.append(False); continue
            c=critic[:,index]; cp=1/(1+np.exp(-c)); e=embedding[:,index]; l=latent[:,index]; r=residual[:,index]; s=scores[:,index]; p=probabilities[:,index]
            top=np.argmax(s.mean(0)); candidate_rows=[]
            for candidate in range(5):
                rank=int(np.where(np.argsort(-s.mean(0),kind="stable")==candidate)[0][0])/4.0
                value=np.concatenate((
                    [c[:,candidate].mean(),c[:,candidate].std(),cp[:,candidate].mean(),cp[:,candidate].std()],
                    e[:,candidate].mean(0),
                    [l[:,candidate].mean(),l[:,candidate].std(),r[:,candidate].mean(),r[:,candidate].std(),s[:,candidate].mean(),s[:,candidate].std(),p[:,candidate].mean(),p[:,candidate].std(),p[:,candidate].mean()/max(p.mean(0).max(),1e-8),rank,float(candidate==top),1.0],
                )).astype(np.float32)
                if value.shape!=(V2_PRIOR_DIM,): raise AssertionError(value.shape)
                candidate_rows.append(value)
            rows.append(np.stack(candidate_rows)); validity.append(True)
    return np.stack(rows),np.asarray(validity,dtype=bool)


def assemble_v2_prior(
    *,
    critic_scores: np.ndarray,
    critic_embeddings: np.ndarray,
    latent_scores: np.ndarray,
    latent_residuals: np.ndarray,
    setrank_scores: np.ndarray,
    setrank_probabilities: np.ndarray,
) -> np.ndarray:
    critic=np.asarray(critic_scores); embedding=np.asarray(critic_embeddings,dtype=np.float32); latent=np.asarray(latent_scores); residual=np.asarray(latent_residuals); scores=np.asarray(setrank_scores); probabilities=np.asarray(setrank_probabilities)
    if critic.ndim!=3 or critic.shape[2]!=5 or embedding.shape[:3]!=critic.shape or embedding.shape[3]!=64: raise ValueError("V2 deployment prior base shapes invalid")
    if any(value.shape!=critic.shape for value in (latent,residual,scores,probabilities)): raise ValueError("V2 deployment prior score shapes differ")
    rows=[]
    for index in range(critic.shape[1]):
        c=critic[:,index]; cp=1/(1+np.exp(-np.clip(c,-40,40))); e=embedding[:,index]; l=latent[:,index]; r=residual[:,index]; s=scores[:,index]; p=probabilities[:,index]; top=np.argmax(s.mean(0)); candidate_rows=[]
        for candidate in range(5):
            rank=int(np.where(np.argsort(-s.mean(0),kind="stable")==candidate)[0][0])/4.0
            value=np.concatenate(([c[:,candidate].mean(),c[:,candidate].std(),cp[:,candidate].mean(),cp[:,candidate].std()],e[:,candidate].mean(0),[l[:,candidate].mean(),l[:,candidate].std(),r[:,candidate].mean(),r[:,candidate].std(),s[:,candidate].mean(),s[:,candidate].std(),p[:,candidate].mean(),p[:,candidate].std(),p[:,candidate].mean()/max(p.mean(0).max(),1e-8),rank,float(candidate==top),1.0])).astype(np.float32)
            if value.shape!=(V2_PRIOR_DIM,): raise AssertionError(value.shape)
            candidate_rows.append(value)
        rows.append(np.stack(candidate_rows))
    return np.stack(rows)


def build_deployment_v2_prior(
    *,
    frozen_features_path: str|Path,
    enhanced_dir: str|Path,
    allowed_ids: set[str],
    v2_manifest_path: str|Path,
    output_dir: str|Path,
    device: str="auto",
) -> dict[str,Any]:
    from failure_analysis.reranking_v2.datasets import load_inference_features
    from failure_analysis.reranking_v2.enhanced_data import load_enhanced_arrays
    from failure_analysis.reranking_v2.inference import predict_base_ensemble,predict_setrank_ensemble
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=False); manifest=json.loads(Path(v2_manifest_path).read_text()); primary=manifest["configs"]["primary"]
    samples=load_inference_features(frozen_features_path,allowed_sample_ids=allowed_ids)
    if {sample.sample_id for sample in samples}!=allowed_ids: raise ValueError("deployment V2 prior cohort coverage mismatch")
    arrays=load_enhanced_arrays(enhanced_dir,samples,include_crops=True,include_labels=False)
    base=predict_base_ensemble(arrays=arrays,critic_models=primary["critic_models"],latent_models=primary["latent_models"],device=device,alpha=primary["alpha"])
    setrank=predict_setrank_ensemble(arrays=arrays,base=base,setrank_models=primary["setrank_models"],device=device,alpha=primary["alpha"])
    values=assemble_v2_prior(critic_scores=base["critic_scores"],critic_embeddings=base["critic_embeddings"],latent_scores=base["latent_scores"],latent_residuals=base["latent_residuals"],setrank_scores=setrank["scores"],setrank_probabilities=setrank["probabilities"])
    path=output/"v2_prior.npz"; temporary=path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try: np.savez_compressed(temporary,sample_ids=np.asarray([sample.sample_id for sample in samples]),prior=values,valid=np.ones(len(samples),bool)); os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)
    result={"schema_version":"3.0.0","artifact_type":"deployment_v2_prior","status":"complete","row_count":len(samples),"unique_sample_count":len(samples),"unique_candidate_count":len(samples)*5,"missing_count":0,"fallback_count":0,"labels_read":False,"v2_manifest":{"path":str(Path(v2_manifest_path).resolve()),"sha256":sha256_file(v2_manifest_path)},"output":artifact_identity(path),"model_sha256":{"critic":[sha256_file(value) for value in primary["critic_models"]],"latent":[sha256_file(value) for value in primary["latent_models"]],"setrank":[sha256_file(value) for value in primary["setrank_models"]]}}
    atomic_write_json(output/"artifact_manifest.json",result); return result


def load_v2_oof_prior(v2_root: str | Path, sample_ids: list[str]) -> tuple[np.ndarray,np.ndarray]:
    root=Path(v2_root)
    return _load_prior(
        root/"oof_base/oof_base_predictions.npz",
        root/"oof_primary/oof_setrank_predictions.npz",
        sample_ids,
    )


def load_v2_validation_prior(v2_root: str | Path, sample_ids: list[str]) -> tuple[np.ndarray,np.ndarray]:
    root=Path(v2_root)
    return _load_prior(
        root/"oof_base/validation_base_predictions.npz",
        root/"oof_primary/validation_setrank_predictions.npz",
        sample_ids,
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload,temporary); os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def _nested_anchor_inputs(prior: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    value=np.asarray(prior,dtype=np.float32)
    if value.ndim!=3 or value.shape[1:]!=(5,V2_PRIOR_DIM): raise ValueError("nested V2 prior must be [N,5,80]")
    embedding=value[:,:,4:68]
    extra=value[:,:,[0,1,68,69,72,73,74,75]]
    return embedding,extra


def validate_nested_anchor_provenance(records: list[dict[str,Any]]) -> None:
    if not records: raise ValueError("nested V2 anchor provenance is empty")
    for record in records:
        heldout=int(record["heldout_fold"])
        checkpoints=record["checkpoints"]
        if len(checkpoints)!=3: raise ValueError("nested V2 anchor requires three seed checkpoints")
        for checkpoint in checkpoints:
            if heldout in set(map(int,checkpoint["fit_folds"])):
                raise ValueError(f"in-sample V2 anchor checkpoint for {record['sample_id']}")


def build_nested_oof_v2_anchor(
    *,
    sample_ids: list[str],
    prior: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    output_dir: str|Path,
    seeds: tuple[int,int,int]=(20260801,20260802,20260803),
    device: str="auto",
    epochs: int=10,
    batch_size: int=512,
    resume: bool=False,
) -> dict[str,Any]:
    output=Path(output_dir); prediction_path=output/"nested_v2_anchor.npz"; manifest_path=output/"artifact_manifest.json"
    if prediction_path.exists() or manifest_path.exists():
        if not resume: raise FileExistsError(output)
        manifest=json.loads(manifest_path.read_text())
        if sha256_file(prediction_path)!=manifest["prediction"]["sha256"]: raise ValueError("nested V2 anchor artifact changed")
        return manifest
    output.mkdir(parents=True,exist_ok=resume); model_dir=output/"models"; model_dir.mkdir(exist_ok=resume)
    ids=list(map(str,sample_ids)); y=np.asarray(labels,dtype=np.float32); folds=np.asarray(fold_ids,dtype=np.int64)
    if y.shape!=(len(ids),5) or folds.shape!=(len(ids),) or set(folds.tolist())!={0,1,2}: raise ValueError("nested V2 anchor cohort shape/folds invalid")
    embedding,extra=_nested_anchor_inputs(prior); baseline=torch.zeros(len(ids),dtype=torch.long)
    target=gate_outcomes(torch.from_numpy(y),baseline).numpy()
    probabilities=np.empty((len(ids),5,3),np.float32); seed_selections=np.empty((len(ids),len(seeds)),np.int64); final_selected=np.empty(len(ids),np.int64)
    provenance=[]; checkpoints_by_fold={}
    torch_device=torch.device("mps" if device=="auto" and torch.backends.mps.is_available() else ("cpu" if device=="auto" else device))
    for fold in range(3):
        fit=np.flatnonzero(folds!=fold); heldout=np.flatnonzero(folds==fold); fold_checkpoints=[]; fold_probabilities=[]
        for seed_index,seed in enumerate(seeds):
            random.seed(seed+fold*100); np.random.seed(seed+fold*100); torch.manual_seed(seed+fold*100)
            model=V2AnchoredGate(embedding_dim=64,extra_dim=8,hidden_dim=128).to(torch_device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
            counts=np.bincount(target[fit,1:].reshape(-1),minlength=3).astype(np.float64)
            weights=torch.tensor(counts.sum()/np.maximum(3*counts,1),dtype=torch.float32,device=torch_device)
            for epoch in range(int(epochs)):
                order=np.random.default_rng(seed+fold*100+epoch).permutation(fit)
                model.train()
                for start in range(0,len(order),int(batch_size)):
                    index=order[start:start+int(batch_size)]
                    emb=torch.from_numpy(embedding[index]).to(torch_device); ext=torch.from_numpy(extra[index]).to(torch_device)
                    base=torch.zeros(len(index),dtype=torch.long,device=torch_device); truth=torch.from_numpy(target[index,1:]).to(torch_device)
                    optimizer.zero_grad(set_to_none=True); logits=model(emb,base,ext)[:,1:]
                    loss=F.cross_entropy(logits.flatten(0,1),truth.flatten(),weight=weights); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                emb=torch.from_numpy(embedding[heldout]).to(torch_device); ext=torch.from_numpy(extra[heldout]).to(torch_device); base=torch.zeros(len(heldout),dtype=torch.long,device=torch_device)
                observed=model(emb,base,ext).softmax(-1)
                observed[:,0]=torch.tensor([0.,0.,1.],dtype=observed.dtype,device=observed.device)
                selected,_=select_v2_anchored(observed,base,harm_cost=5.0,threshold=0.0)
                fold_probabilities.append(observed.cpu().numpy()); seed_selections[heldout,seed_index]=selected.cpu().numpy()
            path=model_dir/f"v2_nested_fold{fold}_seed{seed}.pt"
            _atomic_torch_save({"schema_version":"3.0.0","kind":"nested_v2_protocol_gate","state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"heldout_fold":fold,"fit_folds":[value for value in range(3) if value!=fold],"seed":seed,"epochs":epochs,"prior_fields":{"embedding":[4,68],"extra":[0,1,68,69,72,73,74,75]}},path)
            fold_checkpoints.append({"path":str(path.resolve()),"sha256":sha256_file(path),"seed":seed,"fit_folds":[value for value in range(3) if value!=fold]})
        mean=np.mean(fold_probabilities,axis=0); probabilities[heldout]=mean
        mean_selected,_=select_v2_anchored(torch.from_numpy(mean),torch.zeros(len(heldout),dtype=torch.long),harm_cost=5.0,threshold=0.0)
        consensus=(seed_selections[heldout]==mean_selected.numpy()[:,None]).sum(1)
        final_selected[heldout]=np.where(consensus>=2,mean_selected.numpy(),0)
        checkpoints_by_fold[fold]=fold_checkpoints
    selected=final_selected
    for index,sample_id in enumerate(ids): provenance.append({"sample_id":sample_id,"heldout_fold":int(folds[index]),"checkpoints":checkpoints_by_fold[int(folds[index])],"selected_index":int(selected[index])})
    validate_nested_anchor_provenance(provenance)
    temporary=prediction_path.with_name(f".{prediction_path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary,sample_ids=np.asarray(ids),fold_ids=folds,probabilities=probabilities,seed_selected_indices=seed_selections,selected_indices=selected)
        os.replace(temporary,prediction_path)
    finally: temporary.unlink(missing_ok=True)
    from failure_analysis.reranking_v2.schema import atomic_write_jsonl
    provenance_path=output/"provenance.jsonl"; atomic_write_jsonl(provenance_path,provenance)
    manifest={"schema_version":"3.0.0","artifact_type":"nested_oof_v2_protocol_anchor","status":"complete","row_count":len(ids),"unique_sample_count":len(set(ids)),"folds":[0,1,2],"seeds":list(seeds),"harm_cost":5.0,"threshold":0.0,"required_consensus":2,"anchor_semantics":"OOF V2-protocol-equivalent gate for legal V3 gate training; final inference uses exact locked V2 selection","prediction":artifact_identity(prediction_path),"provenance":artifact_identity(provenance_path),"checkpoint_count":9,"in_sample_rejected":True}
    atomic_write_json(manifest_path,manifest); return manifest
