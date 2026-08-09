#!/usr/bin/env python3
"""Deterministic qualitative galleries and 30-case manual-audit sheets."""

import argparse,hashlib,json,sys,textwrap
from pathlib import Path
import matplotlib.pyplot as plt,numpy as np,pandas as pd
from matplotlib.patches import Polygon
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.fair_crog_hifics_g1_c1_no_rerank.geometry import CanonicalGrasp,corners,gt_from_corners
from experiments.fair_crog_hifics_g1_c1_no_rerank.analyze import rle_decode

COL={"GT":"#F0E442","CROG":"#0072B2","G1":"#D55E00","C1":"#009E73"}
def key(value):return hashlib.sha256(str(value).encode()).hexdigest()
def add_grasp(ax,g,color,lw=2,ls="-"):
 if g is None:return
 ax.add_patch(Polygon(corners(g),closed=True,fill=False,edgecolor=color,linewidth=lw,linestyle=ls))
 ax.plot(g.cx_px,g.cy_px,"o",ms=3,color=color)
def from_row(row,method,sample_id):
 if pd.isna(row.top1_cx_px):return None
 return CanonicalGrasp(row.top1_cx_px,row.top1_cy_px,row.top1_theta_deg,row.top1_jaw_width_px,row.top1_rectangle_height_px,row.top1_native_score,1,method,sample_id)
def overlay(rgb,mask,color):
 out=rgb.astype(float)/255;shade=np.asarray(color);out[mask]=.58*out[mask]+.42*shade;return np.clip(out,0,1)
def render_case(axes,sid,meta,per,crog_mask):
 rgb=np.asarray(Image.open(meta.rgb_path).convert("RGB"));instance=np.asarray(Image.open(meta.gt_mask_path));gt=instance==int(meta.target_instance_id);hifi=np.asarray(Image.open(meta.predicted_hifics_mask_path))>0
 rows={m:per.loc[(sid,m)] for m in ("CROG","G1","C1")};gtgrasps=[gt_from_corners(x) for x in json.loads(meta.gt_grasp_list_json)]
 axes[0].imshow(overlay(rgb,gt,[.95,.8,.1]));[add_grasp(axes[0],g,COL["GT"],1) for g in gtgrasps];axes[0].set_title("RGB + GT mask/grasps",fontsize=7)
 axes[1].imshow(overlay(rgb,crog_mask,[0,.45,.7]));add_grasp(axes[1],from_row(rows["CROG"],"CROG",sid),COL["CROG"],2.5);axes[1].set_title(f"CROG Top-1: {'PASS' if rows['CROG'].j_at_1 else 'FAIL'}\nIoU {rows['CROG'].rotated_iou:.3f} | angle {rows['CROG'].angle_error_deg:.1f}°",fontsize=7)
 axes[2].imshow(overlay(rgb,hifi,[.2,.75,.5]));add_grasp(axes[2],from_row(rows["G1"],"G1",sid),COL["G1"],2.5);add_grasp(axes[2],from_row(rows["C1"],"C1",sid),COL["C1"],2.5,"--");axes[2].set_title(f"HiFi mask; G1 {'P' if rows['G1'].j_at_1 else 'F'} / C1 {'P' if rows['C1'].j_at_1 else 'F'}\nG1 IoU/ang {rows['G1'].rotated_iou:.2f}/{rows['G1'].angle_error_deg:.1f}; C1 {rows['C1'].rotated_iou:.2f}/{rows['C1'].angle_error_deg:.1f}",fontsize=7)
 earliest="none (all pass)" if all(rows[m].j_at_1 for m in rows) else ("mask/grounding association" if min(rows[m].mask_iou for m in rows)<.5 else "native grasp geometry/selection")
 axes[0].text(0,-.13,textwrap.fill(meta.expression,42),transform=axes[0].transAxes,fontsize=6,va="top");axes[1].text(0,-.13,f"scores: C {rows['CROG'].top1_native_score:.3f}; G {rows['G1'].top1_native_score if pd.notna(rows['G1'].top1_native_score) else float('nan'):.3f}; C1 {rows['C1'].top1_native_score if pd.notna(rows['C1'].top1_native_score) else float('nan'):.3f}",transform=axes[1].transAxes,fontsize=6,va="top");axes[2].text(0,-.13,f"earliest observable: {earliest}",transform=axes[2].transAxes,fontsize=6,va="top")
 for ax in axes:ax.axis("off")
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--crog-predictions",type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();manifest=pd.read_parquet(r/"01_manifest/paired_manifest.parquet").set_index("sample_id");samples=pd.read_parquet(r/"04_metrics/per_sample_metrics.parquet");per=samples[samples.method.isin(["CROG","G1","C1"])].set_index(["sample_id","method"]);outcomes=pd.read_csv(r/"06_failure_analysis/per_sample_outcome.csv");g1=pd.read_csv(r/"06_failure_analysis/g1_failure_taxonomy.csv");c1=pd.read_csv(r/"06_failure_analysis/c1_failure_taxonomy.csv")
 masks={};query_to_sid={int(sid[1:8]):sid for sid in manifest.index}
 with a.crog_predictions.resolve().open() as f:
  for line in f:
   x=json.loads(line);sid=query_to_sid.get(int(x["sample_index"]))
   if sid:masks[sid]=rle_decode(x["predicted_mask_rle"])
 # Avoid O(N^2) lookup in future render operations.
 if len(masks)!=len(manifest):raise RuntimeError("CROG qualitative masks incomplete")
 mainwide=per.reset_index().pivot(index="sample_id",columns="method",values=["j_at_1","mask_iou"])
 categories={
  "all_correct":set(outcomes.loc[outcomes.outcome=="all_correct","sample_id"]),"all_wrong":set(outcomes.loc[outcomes.outcome=="all_wrong","sample_id"]),"CROG_only_correct":set(outcomes.loc[outcomes.outcome=="CROG_only","sample_id"]),"G1_only_correct":set(outcomes.loc[outcomes.outcome=="G1_only","sample_id"]),"C1_only_correct":set(outcomes.loc[outcomes.outcome=="C1_only","sample_id"]),"both_modular_correct_CROG_wrong":set(outcomes.loc[outcomes.outcome=="G1_C1","sample_id"]),"CROG_correct_both_modular_wrong":set(outcomes.loc[outcomes.outcome=="CROG_only","sample_id"]),
  "high_mask_iou_grasp_fail":set(samples.loc[(samples.method.isin(["CROG","G1","C1"]))&(samples.mask_iou>=.9)&(~samples.j_at_1),"sample_id"]),"low_mask_iou_grasp_success":set(samples.loc[(samples.method.isin(["CROG","G1","C1"]))&(samples.mask_iou<.5)&samples.j_at_1,"sample_id"]),"G1_grounding_limited":set(g1.loc[g1.category=="grounding_limited","sample_id"]),"C1_grounding_limited":set(c1.loc[c1.category=="grounding_limited","sample_id"]),"G1_candidate_generation_limited":set(g1.loc[g1.category=="grasper_or_candidate_generation_limited","sample_id"]),"C1_candidate_generation_limited":set(c1.loc[c1.category=="grasper_or_candidate_generation_limited","sample_id"]),"native_top1_selection_failure":set(g1.loc[g1.category=="native_selection_within_top5","sample_id"])|set(c1.loc[c1.category=="native_selection_within_top5","sample_id"]),
 }
 selection=[];gallery=r/"08_qualitative/galleries";gallery.mkdir(parents=True,exist_ok=True)
 for category,ids in categories.items():
  chosen=sorted(ids,key=key)[:4]
  if not chosen:continue
  fig,axes=plt.subplots(len(chosen),3,figsize=(12,3.25*len(chosen)));axes=np.asarray(axes).reshape(len(chosen),3)
  for row,sid in enumerate(chosen):render_case(axes[row],sid,manifest.loc[sid],per,masks[sid]);selection.append({"category":category,"sample_id":sid,"selection_rule":"first four by SHA-256(sample_id)"})
  fig.suptitle(category.replace("_"," "),fontsize=12);fig.tight_layout();fig.savefig(gallery/f"{category}.png",dpi=180,bbox_inches="tight");fig.savefig(gallery/f"{category}.pdf",bbox_inches="tight");plt.close(fig)
 pd.DataFrame(selection).to_csv(r/"08_qualitative/gallery_selection_manifest.csv",index=False)
 # Stratified deterministic manual audit: round-robin outcome categories, then hash.
 audit=[]
 for outcome,group in outcomes.groupby("outcome"):
  for sid in sorted(group.sample_id,key=key)[:4]:audit.append({"sample_id":sid,"outcome":outcome})
 audit=sorted(audit,key=lambda x:key(x["sample_id"]))[:30]
 for sheet in range(3):
  chunk=audit[sheet*10:(sheet+1)*10];fig,axes=plt.subplots(len(chunk),3,figsize=(12,3.1*len(chunk)));axes=np.asarray(axes).reshape(len(chunk),3)
  for row,item in enumerate(chunk):render_case(axes[row],item["sample_id"],manifest.loc[item["sample_id"]],per,masks[item["sample_id"]])
  fig.suptitle(f"Manual audit sheet {sheet+1}: deterministic stratified sample",fontsize=12);fig.tight_layout();fig.savefig(r/f"08_qualitative/manual_audit_sheet_{sheet+1}.png",dpi=160,bbox_inches="tight");plt.close(fig)
 pd.DataFrame(audit).to_csv(r/"08_qualitative/manual_audit_selection.csv",index=False)
 (r/"08_qualitative/manual_audit_report.md").write_text("# Manual visual audit\n\nThirty cases were selected deterministically: up to four per outcome combination ordered by SHA-256(sample_id), then the first 30 by the same hash. The three rendered sheets are awaiting recorded visual observations.\n")
 print(f"galleries={len(selection)}, manual={len(audit)}")
if __name__=="__main__":main()
