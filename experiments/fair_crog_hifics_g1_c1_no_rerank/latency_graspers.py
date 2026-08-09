#!/usr/bin/env python3
"""Batch-1 MPS latency for native no-rerank G1 or C1 phases."""

import argparse,json,sys
from pathlib import Path
import pandas as pd, torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"HiFi_reproduction"))
from experiments.fair_crog_hifics_g1_c1_no_rerank.native_inference import infer_one,load_config
from src.grasping.backends import BackendSample
from src.grasping.backends.training import load_finetuned_model
from src.grasping.common.sample_io import CompactSampleLoader,read_deployment_manifest

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--source-run",type=Path,required=True);p.add_argument("--method",choices=("g1","c1"),required=True);a=p.parse_args();run=a.run_dir.resolve();source=a.source_run.resolve()
 selected=json.loads((run/"05_statistics/latency_selection.json").read_text());by_id={r["sample_id"]:r for r in read_deployment_manifest(source/"manifests/test_samples.parquet")};cfg_path=source/"selected_configs"/f"{a.method.upper()}.json";cfg,raw=load_config(a.method,cfg_path,False);model,_,_=load_finetuned_model(raw["finetuned_checkpoint"],backend="grconvnet" if a.method=="g1" else "ggcnn2",device="mps");loader=CompactSampleLoader();out=[]
 for index,item in enumerate(selected):
  arrays=loader.load(by_id[item["sample_id"]]);sample=BackendSample(sample_id=arrays.sample_id,rgb=arrays.rgb,depth_m=arrays.depth_m,predicted_mask=arrays.binary_mask,probability_map=arrays.probability,mask_source="predicted")
  row,_=infer_one(method=a.method,model=model,device=torch.device("mps"),config=cfg,sample=sample,raw_path=None)
  if row["status"]=="technical_failure": raise RuntimeError(row["failure_reason"])
  out.append({"sample_id":item["sample_id"],"warmup":index<20,"grasper_preprocessing_seconds":row["conditioning_seconds"],"grasper_inference_seconds":row["model_seconds"],"candidate_decoding_seconds":row["decoder_seconds"],"grasper_total_seconds":row["total_seconds"]})
  if (index+1)%20==0: print(f"{a.method} latency {index+1}/{len(selected)}",flush=True)
 pd.DataFrame(out).to_parquet(run/f"05_statistics/latency_{a.method}.parquet",index=False)
if __name__=="__main__":main()

