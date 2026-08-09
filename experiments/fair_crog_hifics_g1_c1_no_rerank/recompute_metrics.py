#!/usr/bin/env python3
"""Independent aggregation from frozen canonical rows; never loads a model."""

import argparse,hashlib,json,math
from pathlib import Path
import cv2,numpy as np,pandas as pd
from scipy.stats import binomtest
from shapely.geometry import Polygon
from statsmodels.stats.contingency_tables import cochrans_q

METHODS=("CROG","G1","C1");DISPLAY={"CROG":"CROG-native","G1":"HiFi-CS→G1","C1":"HiFi-CS→C1"}
def angle_error(a,b):return abs(((float(a)-float(b)+90)%180)-90)
def gt_from(raw):
 x=np.asarray(raw,float);c=(x[0]+x[2])/2;j=x[3]-x[0];w=min(float(np.linalg.norm(j)),100.);v=math.degrees(math.atan2(float(j[0]),float(j[1])));theta=v-90 if v>0 else v+90;return [c[0],c[1],w,20.,theta]
def box(g):return cv2.boxPoints(((float(g[0]),float(g[1])),(float(g[2]),float(g[3])),-float(g[4]))).astype(float)
def cv_iou(a,b):
 pa,pb=box(a).astype(np.float32),box(b).astype(np.float32);inter,_=cv2.intersectConvexConvex(pa,pb);union=abs(cv2.contourArea(pa))+abs(cv2.contourArea(pb))-inter;return 0. if union<=0 else float(inter/union)
def shp_iou(a,b):
 pa,pb=Polygon(box(a)),Polygon(box(b));u=pa.union(pb).area;return 0. if u<=0 else float(pa.intersection(pb).area/u)
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();c=pd.read_parquet(r/"03_canonical/canonical_candidates.parquet");stored=pd.read_parquet(r/"04_metrics/per_sample_metrics.parquet");manifest=pd.read_parquet(r/"01_manifest/paired_manifest.parquet");official=pd.read_csv(r/"04_metrics/main_results.csv").set_index("Method");rows=[];per=[]
 for method in METHODS:
  cm=c[c.method==method];groups={sid:g.sort_values("native_rank") for sid,g in cm.groupby("sample_id")}
  sm=stored[stored.method==method].set_index("sample_id")
  for sid in manifest.sample_id:
   g=groups.get(sid);success=[] if g is None else g.candidate_success.astype(bool).tolist();first=next((i for i,v in enumerate(success,1) if v),None);per.append({"sample_id":sid,"method":method,"no_output":len(success)==0,**{f"j_at_{k}":first is not None and first<=k for k in range(1,6)},"oracle_all":first is not None,"mask_iou":float(sm.loc[sid,"mask_iou"])})
 frame=pd.DataFrame(per);comparisons=[]
 for method in METHODS:
  x=frame[frame.method==method];vals=x.mask_iou.to_numpy();row={"Method":DISPLAY[method],"N":len(x),"mask_miou":float(vals.mean()),**{f"p_at_{t}":float(np.mean(vals>t/100)) for t in (50,60,70,80,90)},**{f"J@{k}":float(x[f"j_at_{k}"].mean()) for k in range(1,6)},"Oracle@All (diagnostic)":float(x.oracle_all.mean()),"No-output":float(x.no_output.mean()),"J@1 numerator":int(x.j_at_1.sum())};rows.append(row)
  reference=official.loc[DISPLAY[method]]
  for key in ["N","mask_miou","p_at_50","p_at_60","p_at_70","p_at_80","p_at_90","J@1","J@2","J@3","J@4","J@5","Oracle@All (diagnostic)","No-output","J@1 numerator"]:
   if not np.isclose(float(row[key]),float(reference[key]),rtol=0,atol=1e-15):raise AssertionError(f"independent mismatch {method} {key}: {row[key]} vs {reference[key]}")
 pivot=frame.pivot(index="sample_id",columns="method",values="j_at_1").loc[:,METHODS].astype(bool);q=cochrans_q(pivot.to_numpy(int));
 for aa,bb in (("CROG","G1"),("CROG","C1"),("G1","C1")):
  av,bv=pivot[aa].to_numpy(),pivot[bb].to_numpy();b=int(np.sum(~av&bv));cc=int(np.sum(av&~bv));pval=1. if b+cc==0 else float(binomtest(min(b,cc),b+cc,.5).pvalue);comparisons.append({"comparison":f"{aa}-{bb}","a_wrong_b_correct":b,"a_correct_b_wrong":cc,"exact_p":pval})
 # Independent polygon/angle audit on 100 deterministically selected canonical candidates.
 mainc=c[c.method.isin(METHODS)].copy();mainc["key"]=(mainc.sample_id+"|"+mainc.method+"|"+mainc.candidate_id).map(lambda x:hashlib.sha256(x.encode()).hexdigest());chosen=mainc.sort_values("key").head(100);m=manifest.set_index("sample_id");checks=[]
 for row in chosen.itertuples(index=False):
  gts=[gt_from(x) for x in json.loads(m.loc[row.sample_id,"gt_grasp_list_json"])];gt=gts[int(row.diagnostic_gt_index)];pred=[row.cx_px,row.cy_px,row.jaw_width_px,row.rectangle_height_px,row.theta_deg];cv,sh=cv_iou(pred,gt),shp_iou(pred,gt);ang=angle_error(pred[4],gt[4]);
  if abs(cv-sh)>2e-6 or abs(ang-float(row.diagnostic_angle_error_deg))>1e-10:raise AssertionError("independent polygon/angle cross-check failed")
  checks.append({"sample_id":row.sample_id,"method":row.method,"candidate_id":row.candidate_id,"opencv_continuous_iou":cv,"shapely_continuous_iou":sh,"angle_error_deg":ang})
 result={"status":"PASS","source_only":["03_canonical/canonical_candidates.parquet","04_metrics/per_sample_metrics.parquet","01_manifest/paired_manifest.parquet"],"model_loaded":False,"main_metrics":rows,"cochran_q":{"statistic":float(q.statistic),"pvalue":float(q.pvalue)},"mcnemar":comparisons,"polygon_angle_cross_checks":100,"max_cv2_shapely_abs_difference":max(abs(x["opencv_continuous_iou"]-x["shapely_continuous_iou"]) for x in checks)}
 result_path=r/"09_reports/independent_recompute_results.json";result_path.write_text(json.dumps(result,indent=2)+"\n");pd.DataFrame(checks).to_csv(r/"09_reports/independent_polygon_checks.csv",index=False);digest=hashlib.sha256(result_path.read_bytes()).hexdigest();(r/"09_reports/INDEPENDENT_RECOMPUTATION.md").write_text(f"# Independent recomputation\n\nStatus: **PASS**. The script loaded no model and aggregated only frozen canonical/sample/manifest rows. All main metrics matched exactly (absolute tolerance 1e-15). Cochran Q and three exact McNemar disagreement tables were independently recomputed. One hundred deterministic candidate/GT pairs passed OpenCV-vs-Shapely continuous polygon IoU (max absolute difference {result['max_cv2_shapely_abs_difference']:.3g}) and periodic-angle checks.\n\nResult SHA-256: `{digest}`.\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
