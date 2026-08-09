#!/usr/bin/env python3
"""Batch-1 MPS latency for frozen CROG with its native decoder."""

import argparse,json,os,sys,time
from pathlib import Path
import cv2,numpy as np,torch,torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2];CROG=ROOT/"crog_reproduction/CROG";sys.path.insert(0,str(CROG))
import utils.config as config
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.dataset import OCIDVLGDataset
from utils.device import move_to_device
from utils.grasp_eval import detect_grasp_candidates

def sync(): torch.mps.synchronize()
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve();selected=json.loads((run/"05_statistics/latency_selection.json").read_text())
 os.chdir(CROG);cfg=config.load_cfg_from_cfg_file(str(CROG/"config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml"));root=(CROG/cfg.root_path).resolve();dataset=OCIDVLGDataset(root_dir=str(root),input_size=cfg.input_size,word_length=cfg.word_len,split="test",version=cfg.version);model,_=build_crog(cfg);model=model.to("mps").eval();load_checkpoint(str(CROG/"exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth"),model,torch.device("mps"));results=[]
 with torch.inference_mode():
  for order,row in enumerate(selected):
   index=int(dataset.get_index_from_sent(int(row["query_id"])));t=time.perf_counter();data=OCIDVLGDataset.collate_fn([dataset[index]]);prep=time.perf_counter()-t
   values=move_to_device((data["img"],data["word_vec"],data["mask"],data["grasp_masks"]["qua"],data["grasp_masks"]["sin"],data["grasp_masks"]["cos"],data["grasp_masks"]["wid"]),torch.device("mps"));image,text,ins,qua,sin,cos,wid=values
   sync();t=time.perf_counter();pred,_=model(image,text,ins.unsqueeze(1),qua.unsqueeze(1),sin.unsqueeze(1),cos.unsqueeze(1),wid.unsqueeze(1));sync();infer=time.perf_counter()-t
   t=time.perf_counter();ins_p,q,s,c,w=pred;ins_p=torch.sigmoid(ins_p);q=torch.sigmoid(q);w=torch.sigmoid(w)
   if q.shape[-2:]!=image.shape[-2:]:
    kw={"size":image.shape[-2:],"mode":"bicubic","align_corners":True};ins_p=F.interpolate(ins_p,**kw);q=F.interpolate(q,**kw);s=F.interpolate(s,**kw);c=F.interpolate(c,**kw);w=F.interpolate(w,**kw)
   inverse=data["inverse"][0];height,width=map(int,data["ori_size"][0]);restore=lambda x:cv2.warpAffine(x[0,0].float().cpu().numpy(),inverse,(width,height),flags=cv2.INTER_CUBIC);detect_grasp_candidates(restore(q),restore(s),restore(c),restore(w),5);post=time.perf_counter()-t
   results.append({"sample_id":row["sample_id"],"warmup":order<20,"preprocessing_seconds":prep,"model_inference_seconds":infer,"grasp_decoding_seconds":post,"total_seconds":prep+infer+post})
   if (order+1)%20==0: print(f"CROG latency {order+1}/{len(selected)}",flush=True)
 (run/"05_statistics/latency_crog.json").write_text(json.dumps(results,indent=2)+"\n")
if __name__=="__main__":main()
