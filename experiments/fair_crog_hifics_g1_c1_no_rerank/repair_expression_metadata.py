#!/usr/bin/env python3
"""Replace inferred expression types with official OCID-VLG template labels."""
import argparse,json
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--annotations",type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();annotations=json.loads(a.annotations.resolve().read_text())["data"];mapping={int(x["question_index"]):Path(x["template_filename"]).stem if Path(x["template_filename"]).stem in {"name","attribute","relation","location"} else "mixed" for x in annotations};path=r/"01_manifest/paired_manifest.parquet";frame=pd.read_parquet(path);old=frame.expression_type.value_counts().to_dict();frame["expression_type"]=frame.query_id.map(mapping);assert not frame.expression_type.isna().any();frame.to_parquet(path,index=False);frame.to_csv(r/"01_manifest/paired_manifest.csv",index=False);new=frame.expression_type.value_counts().to_dict();(r/"00_audit/expression_metadata_correction.json").write_text(json.dumps({"reason":"use official template_filename as the primary expression type; filter_category is common plumbing, not a second expression type","predictions_or_outcomes_changed":False,"before":old,"after":new},indent=2)+"\n");print(new)
if __name__=="__main__":main()
