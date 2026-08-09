#!/usr/bin/env python3
"""Merge phase-aligned latency runs, update Table 2, and replace figure 08."""

import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt,numpy as np,pandas as pd

def metrics(method,values):
 v=np.asarray(values,float);return {"Method":method,"N":len(v),"median_seconds":float(np.median(v)),"mean_seconds":float(np.mean(v)),"p90_seconds":float(np.quantile(v,.9)),"p95_seconds":float(np.quantile(v,.95)),"throughput_samples_per_second":float(1/np.mean(v))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();h=pd.DataFrame(json.loads((r/"05_statistics/latency_hifi.json").read_text()));g=pd.read_parquet(r/"05_statistics/latency_g1.parquet");c=pd.read_parquet(r/"05_statistics/latency_c1.parquet");x=pd.DataFrame(json.loads((r/"05_statistics/latency_crog.json").read_text()));assert all(len(v)==220 and v.sample_id.tolist()==h.sample_id.tolist() for v in (g,c,x));keep=~h.warmup
 gtot=h.hifi_total_seconds+g.grasper_total_seconds;ctot=h.hifi_total_seconds+c.grasper_total_seconds;rows=[metrics("CROG-native",x.loc[keep,"total_seconds"]),metrics("HiFi-CS→G1",gtot[keep]),metrics("HiFi-CS→C1",ctot[keep])];summary=pd.DataFrame(rows);summary.to_csv(r/"05_statistics/latency_summary.csv",index=False)
 phase=pd.DataFrame({"sample_id":h.sample_id,"warmup":h.warmup,"crog_total_seconds":x.total_seconds,"g1_total_seconds":gtot,"c1_total_seconds":ctot,"hifi_preprocessing_seconds":h.hifi_preprocessing_seconds,"hifi_inference_seconds":h.hifi_inference_seconds,"hifi_mask_postprocessing_seconds":h.hifi_mask_postprocessing_seconds,"g1_preprocessing_seconds":g.grasper_preprocessing_seconds,"g1_inference_seconds":g.grasper_inference_seconds,"g1_decoding_seconds":g.candidate_decoding_seconds,"c1_preprocessing_seconds":c.grasper_preprocessing_seconds,"c1_inference_seconds":c.grasper_inference_seconds,"c1_decoding_seconds":c.candidate_decoding_seconds});phase.to_parquet(r/"05_statistics/latency_per_sample.parquet",index=False)
 main=pd.read_csv(r/"04_metrics/main_results.csv");by=summary.set_index("Method");main["Median latency"]=main.Method.map(by.median_seconds);main["P95 latency"]=main.Method.map(by.p95_seconds);main.to_csv(r/"04_metrics/main_results.csv",index=False)
 fig,ax=plt.subplots(figsize=(5.6,3.2));ax.boxplot([x.loc[keep,"total_seconds"]*1000,gtot[keep]*1000,ctot[keep]*1000],tick_labels=["CROG-native","HiFi-CS→G1","HiFi-CS→C1"],showfliers=False);ax.set_ylabel("End-to-end latency (ms)");ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(r/"07_figures/08_latency_distribution.pdf",bbox_inches="tight");fig.savefig(r/"07_figures/08_latency_distribution.png",dpi=300,bbox_inches="tight");plt.close(fig)
 (r/"05_statistics/latency_protocol.json").write_text(json.dumps({"device":"MPS","batch_size":1,"model_load_excluded":True,"warmups":20,"measured":200,"selection":"first 220 by SHA-256(sample_id)","modular_total":"per-sample HiFi total + aligned grasper total","crog_preprocessing_note":"official OCIDVLGDataset item/collate includes annotation tensor preparation, so CROG preprocessing is conservatively overinclusive","peak_memory":"unavailable reliably on shared-memory MPS"},indent=2)+"\n")
 print(summary.to_string(index=False))
if __name__=="__main__":main()
