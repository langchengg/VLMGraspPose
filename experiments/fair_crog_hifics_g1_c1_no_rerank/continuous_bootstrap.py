#!/usr/bin/env python3
"""10,000-replicate scene-cluster paired-median difference intervals."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd

SEED=20260806;REPS=10000;METHODS=(("CROG","G1"),("CROG","C1"),("G1","C1"));METRICS=("rotated_iou","angle_error_deg","normalized_center_error","relative_width_error")
def weighted_bootstrap_median(values,cluster_ids,rng):
 labels,inverse=np.unique(cluster_ids,return_inverse=True);order=np.argsort(values);v=values[order];inv=inverse[order];out=np.empty(REPS);batch=200
 for start in range(0,REPS,batch):
  n=min(batch,REPS-start);draws=rng.integers(0,len(labels),size=(n,len(labels)));counts=np.zeros((n,len(labels)),dtype=np.int16)
  for i in range(n):counts[i]=np.bincount(draws[i],minlength=len(labels))
  weights=counts[:,inv];cum=np.cumsum(weights,axis=1);halves=weights.sum(axis=1)/2;indices=(cum>=halves[:,None]).argmax(axis=1);out[start:start+n]=v[indices]
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();s=pd.read_parquet(r/"04_metrics/per_sample_metrics.parquet");m=pd.read_parquet(r/"01_manifest/paired_manifest.parquet")[["sample_id","scene_family"]];s=s[s.method.isin(["CROG","G1","C1"])].merge(m,on="sample_id");rows=[]
 for pair_index,(aa,bb) in enumerate(METHODS):
  arow=s[s.method==aa].set_index("sample_id");brow=s[s.method==bb].set_index("sample_id");common=arow.index.intersection(brow.index)
  for metric_index,metric in enumerate(METRICS):
   av=arow.loc[common,metric].to_numpy(float);bv=brow.loc[common,metric].to_numpy(float);valid=np.isfinite(av)&np.isfinite(bv);delta=bv[valid]-av[valid];clusters=arow.loc[common[valid],"scene_family"].to_numpy(str);dist=weighted_bootstrap_median(delta,clusters,np.random.default_rng(SEED+100*pair_index+metric_index));low,high=np.quantile(dist,[.025,.975]);rows.append({"comparison":f"{bb}-{aa}","metric":metric,"paired_N":len(delta),"median_difference":float(np.median(delta)),"scene_cluster_CI_low":float(low),"scene_cluster_CI_high":float(high),"replicates":REPS,"seed":SEED})
 result=pd.DataFrame(rows);result.to_csv(r/"05_statistics/continuous_paired_median_bootstrap.csv",index=False);print(result.to_string(index=False))
if __name__=="__main__":main()
