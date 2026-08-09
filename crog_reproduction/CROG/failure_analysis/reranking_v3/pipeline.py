from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from failure_analysis.reranking_v2.schema import atomic_write_jsonl

from .experiment_config import ENSEMBLE_SEEDS
from .feature_store import FeatureCatalog, fit_streaming_normalizers
from .gate_training import gate_extras, train_gate_ensemble
from .oof import fold_assignments, predict_ranker_streaming, train_ranker_streaming
from .artifacts import code_fingerprint
from .schema import artifact_identity, atomic_write_json, canonical_json, sha256_bytes, sha256_file
from .v2_prior import build_nested_oof_v2_anchor


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try: np.savez_compressed(temporary,**arrays); os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def run_grouped_oof_and_final_models(
    *,
    catalog: FeatureCatalog,
    train_ids: set[str],
    labels: dict[str,np.ndarray],
    priors: dict[str,np.ndarray],
    split_manifest: str|Path,
    selected_config: dict[str,Any],
    output_dir: str|Path,
    device: str="auto",
    oof_epochs: int=5,
    final_epochs: int=5,
    batch_size: int=16,
    resume: bool=False,
) -> dict[str,Any]:
    output=Path(output_dir); summary_path=output/"summary.json"
    run_spec={"schema_version":"3.0.0","kind":"v3_oof_final_run_spec","status":"frozen_before_fit","code_fingerprint":code_fingerprint(),"features":catalog.provenance(),"split":artifact_identity(split_manifest),"train_ids_sha256":hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),"selected_config":selected_config,"device":device,"oof_epochs":int(oof_epochs),"final_epochs":int(final_epochs),"batch_size":int(batch_size),"ensemble_seeds":list(ENSEMBLE_SEEDS)}
    run_spec["content_sha256"]=sha256_bytes(canonical_json(run_spec).encode()); run_spec_path=output/"RUN_SPEC.json"
    if summary_path.exists():
        if not resume: raise FileExistsError(output)
        if not run_spec_path.exists() or json.loads(run_spec_path.read_text())!=run_spec: raise ValueError("OOF/final resume run-spec mismatch")
        return json.loads(summary_path.read_text())
    output.mkdir(parents=True,exist_ok=resume); model_dir=output/"models"; model_dir.mkdir(exist_ok=resume)
    if run_spec_path.exists():
        if not resume or json.loads(run_spec_path.read_text())!=run_spec: raise ValueError("OOF/final run-spec mismatch")
    else: atomic_write_json(run_spec_path,run_spec)
    ids=sorted(train_ids); fold_by_id=fold_assignments(split_manifest,train_ids); index_by_id={value:index for index,value in enumerate(ids)}
    split_payload=json.loads(Path(split_manifest).read_text()); split_by_id={str(row["sample_id"]):row for row in split_payload["rows"] if str(row["sample_id"]) in train_ids}
    aligned_labels=np.stack([labels[value] for value in ids]).astype(np.float32); aligned_prior=np.stack([priors[value] for value in ids]).astype(np.float32); aligned_folds=np.asarray([fold_by_id[value] for value in ids],np.int64)
    oof_path=output/"oof/fcer_oof_predictions.npz"
    provenance_path=output/"oof/provenance.jsonl"
    if resume and oof_path.exists() and provenance_path.exists():
        with np.load(oof_path) as payload:
            if list(map(str,payload["sample_ids"]))!=ids: raise ValueError("resumed FCER OOF cohort changed")
            candidate_ids=np.asarray(payload["candidate_ids"]); candidate_checksums=np.asarray(payload["candidate_checksums"]); q_ranks=np.asarray(payload["q_ranks"]); checkpoint_sha=np.asarray(payload["producing_checkpoint_sha256"])
            oof_arrays={name:np.asarray(payload[name]) for name in ("scores","residual","probabilities","any_probability","embeddings","token_attention","q")}
        provenance=[json.loads(line) for line in provenance_path.read_text().splitlines() if line.strip()]
        fold_models=[artifact_identity(model_dir/f"fcer_oof_fold{fold}_seed{ENSEMBLE_SEEDS[fold]}.pt") for fold in range(3)]
    else:
        if oof_path.exists() or provenance_path.exists(): raise FileExistsError("partial FCER OOF artifacts require manual audit")
        oof_arrays={}; candidate_ids=np.empty((len(ids),5),dtype="U64"); candidate_checksums=np.empty((len(ids),5),dtype="U64"); q_ranks=np.empty((len(ids),5),dtype=np.int64); checkpoint_sha=np.empty(len(ids),dtype="U64"); provenance=[]; fold_models=[]
        for fold in range(3):
            fit={value for value in ids if fold_by_id[value]!=fold}; heldout={value for value in ids if fold_by_id[value]==fold}; seed=ENSEMBLE_SEEDS[fold]; path=model_dir/f"fcer_oof_fold{fold}_seed{seed}.pt"
            fit_groups={str(split_by_id[value]["sequence_id"]) for value in fit}; heldout_groups={str(split_by_id[value]["sequence_id"]) for value in heldout}
            if fit_groups & heldout_groups: raise AssertionError("FCER OOF fit/heldout group intersection is non-zero")
            result=train_ranker_streaming(catalog=catalog,train_ids=fit,labels={value:labels[value] for value in fit},priors={value:priors[value] for value in fit},config=selected_config,output_path=path,seed=seed,device=device,epochs=oof_epochs,batch_size=batch_size,resume=resume)
            prediction=predict_ranker_streaming(catalog=catalog,sample_ids=heldout,priors={value:priors[value] for value in heldout},checkpoint_path=path,device=device,batch_size=max(32,batch_size))
            for local,sample_id in enumerate(map(str,prediction["sample_ids"])):
                target=index_by_id[sample_id]; candidate_ids[target]=prediction["candidate_ids"][local]; candidate_checksums[target]=prediction["candidate_checksums"][local]; q_ranks[target]=prediction["q_ranks"][local]; checkpoint_sha[target]=result["checkpoint"]["sha256"]
                for name in ("scores","residual","probabilities","any_probability","embeddings","token_attention","q"):
                    value=prediction[name][local]
                    if name not in oof_arrays: oof_arrays[name]=np.empty((len(ids),*value.shape),dtype=np.float16 if name in {"embeddings","token_attention"} else np.float32)
                    oof_arrays[name][target]=value
                provenance.append({"sample_id":sample_id,"heldout_fold":fold,"heldout_group":str(split_by_id[sample_id]["sequence_id"]),"fit_group_count":len(fit_groups),"heldout_group_count":len(heldout_groups),"fit_groups_sha256":hashlib.sha256("\n".join(sorted(fit_groups)).encode()).hexdigest(),"heldout_groups_sha256":hashlib.sha256("\n".join(sorted(heldout_groups)).encode()).hexdigest(),"group_intersection_count":0,"checkpoint":{"path":str(path.resolve()),"sha256":result["checkpoint"]["sha256"],"fit_folds":[value for value in range(3) if value!=fold],"seed":seed}})
            fold_models.append(artifact_identity(path))
        _atomic_npz(oof_path,sample_ids=np.asarray(ids),candidate_ids=candidate_ids,candidate_checksums=candidate_checksums,q_ranks=q_ranks,fold_ids=aligned_folds,producing_checkpoint_sha256=checkpoint_sha,**oof_arrays)
        atomic_write_jsonl(provenance_path,sorted(provenance,key=lambda value:value["sample_id"]))
    if np.any(checkpoint_sha=="") or set(record["sample_id"] for record in provenance)!=train_ids: raise AssertionError("FCER OOF coverage incomplete")
    for record in provenance:
        if record["heldout_fold"] in record["checkpoint"]["fit_folds"]: raise AssertionError("FCER OOF in-sample checkpoint")
    nested_manifest=build_nested_oof_v2_anchor(sample_ids=ids,prior=aligned_prior,labels=aligned_labels,fold_ids=aligned_folds,output_dir=output/"nested_v2_anchor",device=device,resume=resume)
    with np.load(output/"nested_v2_anchor/nested_v2_anchor.npz") as nested: baseline=np.asarray(nested["selected_indices"],np.int64)
    extras=gate_extras(scores=oof_arrays["scores"],probabilities=oof_arrays["probabilities"],q=oof_arrays["q"],residual=oof_arrays["residual"],baseline_indices=baseline)
    gate_summary=train_gate_ensemble(sample_ids=ids,embeddings=oof_arrays["embeddings"].astype(np.float32),extras=extras,labels=aligned_labels,baseline_indices=baseline,producing_folds=aligned_folds,producing_checkpoint_sha256=checkpoint_sha,output_dir=output/"gate",device=device,resume=resume)
    full_normalizers=fit_streaming_normalizers(catalog,allowed_ids=train_ids,priors=priors); final_models=[]
    for seed in ENSEMBLE_SEEDS:
        path=model_dir/f"fcer_final_seed{seed}.pt"; result=train_ranker_streaming(catalog=catalog,train_ids=train_ids,labels=labels,priors=priors,config=selected_config,output_path=path,seed=seed,device=device,epochs=final_epochs,batch_size=batch_size,normalizers=full_normalizers,resume=resume); final_models.append(result["checkpoint"])
    summary={"schema_version":"3.0.0","kind":"v3_grouped_oof_and_final_ensemble","status":"complete","selected_config":selected_config,"sample_count":len(ids),"fold_count":3,"oof_epochs":int(oof_epochs),"final_epochs":int(final_epochs),"batch_size":int(batch_size),"device":device,"run_spec":artifact_identity(run_spec_path),"fold_models":fold_models,"oof_predictions":artifact_identity(oof_path),"oof_provenance":artifact_identity(provenance_path),"oof_contains_labels":False,"oof_candidate_checksums_persisted":True,"oof_q_ranks_persisted":True,"nested_v2_anchor_manifest":nested_manifest,"gate_summary":gate_summary,"final_models":final_models,"ensemble_seeds":list(ENSEMBLE_SEEDS),"in_sample_stacking_rejected":True,"formal_test_read":False}
    atomic_write_json(summary_path,summary); return summary


def predict_final_ensemble(
    *,
    catalog: FeatureCatalog,
    sample_ids: set[str],
    priors: dict[str,np.ndarray],
    checkpoint_paths: list[str|Path],
    output_path: str|Path,
    device: str="auto",
) -> dict[str,Any]:
    path=Path(output_path); all_predictions=[]
    if len(checkpoint_paths)!=len(ENSEMBLE_SEEDS): raise ValueError("final ensemble requires exactly three checkpoints")
    for expected_seed,checkpoint in zip(ENSEMBLE_SEEDS,checkpoint_paths,strict=True):
        artifact=__import__("torch").load(checkpoint,map_location="cpu",weights_only=False)
        if int(artifact.get("seed",-1))!=expected_seed: raise ValueError("final ensemble checkpoint seed/order mismatch")
        all_predictions.append(predict_ranker_streaming(catalog=catalog,sample_ids=sample_ids,priors=priors,checkpoint_path=checkpoint,device=device,batch_size=32))
    reference=list(map(str,all_predictions[0]["sample_ids"])); aligned=[]
    for prediction in all_predictions:
        lookup={str(value):index for index,value in enumerate(prediction["sample_ids"])}
        if set(lookup)!=sample_ids: raise ValueError("final ensemble prediction coverage mismatch")
        aligned.append({name:np.asarray(prediction[name])[[lookup[value] for value in reference]] for name in ("scores","residual","probabilities","any_probability","embeddings","token_attention","q","candidate_ids","candidate_checksums","q_ranks")})
    for value in aligned[1:]:
        if not np.array_equal(value["candidate_ids"].astype(str),aligned[0]["candidate_ids"].astype(str)): raise ValueError("final ensemble candidate identity differs across seeds")
        if not np.array_equal(value["candidate_checksums"].astype(str),aligned[0]["candidate_checksums"].astype(str)): raise ValueError("final ensemble candidate checksums differ across seeds")
        if not np.array_equal(value["q_ranks"],aligned[0]["q_ranks"]): raise ValueError("final ensemble q ranks differ across seeds")
        if not np.array_equal(value["q"],aligned[0]["q"]): raise ValueError("final ensemble q evidence differs across seeds")
    arrays={"sample_ids":np.asarray(reference),"candidate_ids":aligned[0]["candidate_ids"],"candidate_checksums":aligned[0]["candidate_checksums"],"q_ranks":aligned[0]["q_ranks"],"scores":np.stack([value["scores"] for value in aligned]),"residual":np.stack([value["residual"] for value in aligned]),"probabilities":np.stack([value["probabilities"] for value in aligned]),"any_probability":np.stack([value["any_probability"] for value in aligned]),"embeddings":np.stack([value["embeddings"] for value in aligned]).astype(np.float16),"token_attention":np.stack([value["token_attention"] for value in aligned]).astype(np.float16),"q":aligned[0]["q"],"checkpoint_sha256":np.asarray([sha256_file(value) for value in checkpoint_paths])}
    _atomic_npz(path,**arrays); manifest={"schema_version":"3.0.0","kind":"v3_final_fcer_ensemble_predictions","status":"complete","prediction":artifact_identity(path),"row_count":len(reference),"seeds":list(ENSEMBLE_SEEDS),"checkpoint_sha256":arrays["checkpoint_sha256"].tolist(),"candidate_identity_unchanged":True,"candidate_checksums_persisted":True,"q_ranks_persisted":True}; atomic_write_json(path.with_suffix(".manifest.json"),manifest); return manifest
