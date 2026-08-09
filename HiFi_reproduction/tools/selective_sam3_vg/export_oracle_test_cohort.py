#!/usr/bin/env python3
"""After formal evaluation, define the non-deployable GT-binned test cohort."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    formal_metrics = ROOT / "outputs/selective_sam3_vg/report/formal_metrics.json"
    if not formal_metrics.is_file():
        raise RuntimeError("formal deployable method must be evaluated before test oracle export")
    source = pd.read_parquet(
        ROOT / "outputs/selective_sam3_vg/diagnostics/baseline_formal_test_per_sample_metrics.parquet"
    )
    cohort = source[(source["iou"] >= 0.60) & (source["iou"] < 0.90)].copy()
    cohort.insert(0, "oracle_order", range(len(cohort)))
    cohort["iou_band"] = pd.cut(
        cohort["iou"], bins=[0.60, 0.70, 0.80, 0.90], right=False,
        labels=["60_70", "70_80", "80_90"],
    ).astype(str)
    cohort["target_area_band"] = pd.cut(
        cohort["target_area_fraction"], bins=[-1, 0.005, 0.015, 0.04, 1.0],
        labels=["tiny", "small", "medium", "large"],
    ).astype(str)
    output = ROOT / "outputs/selective_sam3_vg/oracle"
    output.mkdir(parents=True, exist_ok=True)
    stratification = output / "ORACLE_GT_SELECTED_cohort_stratification.parquet"
    cohort.to_parquet(stratification, index=False)
    cohort.to_csv(output / "ORACLE_GT_SELECTED_cohort_stratification.csv", index=False)
    inference_columns = {
        "sample_index": "sample_index", "sample_id": "sample_id",
        "question_index": "question_index", "scene_id": "scene_id", "query": "query",
        "rgb_path": "rgb_path", "depth_path": "depth_path",
        "probability_path": "probability_path", "native_mask_path": "native_mask_path",
    }
    with (output / "ORACLE_GT_SELECTED_inference_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for raw in cohort.to_dict(orient="records"):
            row = {destination: raw[source_name] for destination, source_name in inference_columns.items()}
            row["oracle"] = True
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "status": "COMPLETED", "oracle": True,
        "diagnostic_label": "Diagnostic upper bound; uses test ground truth; not deployable.",
        "cohort_definition": "0.60 <= baseline_iou < 0.90", "sample_count": len(cohort),
        "formal_evaluation_preceded_oracle_export": True,
    }
    (output / "ORACLE_GT_SELECTED_cohort_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
