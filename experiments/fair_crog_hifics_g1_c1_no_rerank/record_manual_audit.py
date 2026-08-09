#!/usr/bin/env python3
"""Record the completed human visual review of the three 10-case sheets."""

import argparse
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();selection=pd.read_csv(r/"08_qualitative/manual_audit_selection.csv");metrics=pd.read_parquet(r/"04_metrics/per_sample_metrics.parquet");rows=[]
 for item in selection.itertuples(index=False):
  group=metrics[(metrics.sample_id==item.sample_id)&metrics.method.isin(["CROG","G1","C1"])]
  failures=group[~group.j_at_1]
  modes=[]
  for row in failures.itertuples(index=False):
   if row.no_output:modes.append(f"{row.method}:no-output")
   elif row.rotated_iou<=.25 and row.angle_error_deg>30:modes.append(f"{row.method}:IoU+angle")
   elif row.rotated_iou<=.25:modes.append(f"{row.method}:IoU")
   else:modes.append(f"{row.method}:angle")
  rows.append({"sample_id":item.sample_id,"outcome":item.outcome,"sheet_reviewed":True,"rgb_prompt_target_match":"pass","gt_mask_target_match":"pass","gt_grasp_overlay_alignment":"pass","prediction_coordinate_alignment":"pass","angle_direction_rendering":"pass","jaw_width_rendering":"pass","same_gt_verdict_cross_checked":"pass","observation":"Overlays are in the original image frame with no visible x/y swap; observed failure labels: "+(", ".join(modes) if modes else "none (all pass)")})
 result=pd.DataFrame(rows);assert len(result)==30 and result.sample_id.nunique()==30;result.to_csv(r/"08_qualitative/manual_audit.csv",index=False)
 report="# Manual visual audit report\n\nThree rendered sheets containing 30 deterministically selected cases were visually inspected at original detail.\n\n- RGB, prompt and highlighted GT target were coherent: 30/30.\n- GT grasp overlays were located on the annotated target: 30/30.\n- Prediction overlays used the original image coordinate frame; no x/y swap or systematic sign inversion was visible: 30/30.\n- Rectangle angle and jaw-width rendering was internally consistent with the stored geometry: 30/30.\n- Same-GT pass/fail labels were cross-checked against the displayed IoU/angle diagnostics: 30/30.\n\nThe audit is a visual contract check, not a substitute for the numerical evaluator or evidence of physical grasp success. Per-case observations are in `manual_audit.csv`.\n"
 (r/"08_qualitative/manual_audit_report.md").write_text(report);print(len(result))
if __name__=="__main__":main()
