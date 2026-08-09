#!/usr/bin/env python3
"""Batch-1 MPS latency for the frozen HiFi-CS frontend on 20+200 samples."""

import argparse, hashlib, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch, yaml
from PIL import Image
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[2]; HIFI=ROOT/"HiFi_reproduction/hifics"; sys.path.insert(0,str(HIFI))
from models.hifics import HierarchicalCLIPDensePredT  # noqa:E402

def sync(): torch.mps.synchronize()

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
 rows=json.loads((run/"05_statistics/latency_selection.json").read_text())
 source=ROOT/"HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615";cfg=yaml.safe_load((source/"config.yaml").read_text())
 model=HierarchicalCLIPDensePredT(version=cfg["clip_backbone"],extract_layers=tuple(cfg["projection_layers"]),reduce_dim=int(cfg["decoder_dimension"]),n_heads=int(cfg.get("decoder_heads",4)),cond_layer=None,extended_film=True,hierarchical_film=True)
 payload=torch.load(source/"checkpoints/best.pth",map_location="cpu",weights_only=False);state=payload["trainable_state"];complete=model.state_dict();complete.update(state);model.load_state_dict(complete,strict=True);model.to("mps").eval();model.clip_model.eval()
 transform=transforms.Compose([transforms.Resize((int(cfg["image_resolution"]),)*2),transforms.ToTensor()]); results=[]
 with torch.inference_mode():
  for index,row in enumerate(rows):
   t=time.perf_counter()
   with Image.open(row["rgb_path"]) as im: native_size=im.size; tensor=transform(im.convert("RGB"))
   prep=time.perf_counter()-t;t=time.perf_counter();sync();logit=model(tensor.unsqueeze(0).to("mps"),[row["expression"]])[0];sync();infer=time.perf_counter()-t
   t=time.perf_counter();prob=torch.sigmoid(-logit)[0,0].float().cpu().numpy();mask=prob>=float(cfg["validation_threshold"]);cv2.resize(mask.astype(np.uint8),native_size,interpolation=cv2.INTER_NEAREST);post=time.perf_counter()-t
   results.append({"sample_id":row["sample_id"],"warmup":index<20,"hifi_preprocessing_seconds":prep,"hifi_inference_seconds":infer,"hifi_mask_postprocessing_seconds":post,"hifi_total_seconds":prep+infer+post})
   if (index+1)%20==0: print(f"HiFi latency {index+1}/{len(rows)}",flush=True)
 (run/"05_statistics/latency_hifi.json").write_text(json.dumps(results,indent=2)+"\n")
if __name__=="__main__":main()
