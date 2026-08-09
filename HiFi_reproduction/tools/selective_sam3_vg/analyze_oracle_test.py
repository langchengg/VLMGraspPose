#!/usr/bin/env python3
"""Compute the GT-selected SAM ceiling without replacing the formal result."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.selective_sam3_vg.metrics import summarize_ious  # noqa: E402


PAPER = {"mean_iou": 0.8826, "p_at_50": 0.9268, "p_at_60": 0.9213, "p_at_70": 0.9153, "p_at_80": 0.8969, "p_at_90": 0.8321}


def main() -> None:
    root = ROOT / "outputs/selective_sam3_vg/oracle"
    candidates = pd.read_parquet(
        root / "ORACLE_GT_SELECTED_analysis/pilot_candidate_evaluation.parquet"
    )
    formal = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/report/formal_per_sample_metrics.parquet")
    cohort = candidates[candidates["candidate_id"] == "coarse_0"]["sample_id"].unique()
    best_sam = (
        candidates[candidates["candidate_id"] != "coarse_0"]
        .groupby("sample_id")["candidate_iou"].max()
    )
    baseline = formal.set_index("sample_id")["baseline_iou"].copy()
    oracle = baseline.copy()
    oracle.loc[cohort] = np.maximum(baseline.loc[cohort], best_sam.reindex(cohort))
    baseline_summary = summarize_ious(baseline.to_numpy())
    oracle_summary = summarize_ious(oracle.to_numpy())
    gain = best_sam.reindex(cohort) - baseline.reindex(cohort)
    gap = {
        "mean_iou": (oracle_summary["mean_iou"] - baseline_summary["mean_iou"]) / (PAPER["mean_iou"] - baseline_summary["mean_iou"])
    }
    for percent in (50, 60, 70, 80, 90):
        key = f"p_at_{percent}"
        gap[key] = (oracle_summary[key] - baseline_summary[key]) / (PAPER[key] - baseline_summary[key])
    per_sample = pd.DataFrame(
        {
            "sample_id": baseline.index,
            "baseline_iou": baseline.to_numpy(),
            "ORACLE_GT_SELECTED_iou": oracle.to_numpy(),
            "ORACLE_GT_SELECTED_delta_iou": oracle.to_numpy() - baseline.to_numpy(),
            "in_diagnostic_cohort": baseline.index.isin(cohort),
            "oracle": True,
        }
    )
    per_sample.to_parquet(root / "ORACLE_GT_SELECTED_per_sample.parquet", index=False)
    per_sample.to_csv(root / "ORACLE_GT_SELECTED_per_sample.csv", index=False)
    result = {
        "status": "COMPLETED", "oracle": True,
        "diagnostic_label": "Diagnostic upper bound; uses test ground truth; not deployable.",
        "cohort_definition": "0.60 <= baseline_iou < 0.90",
        "cohort_samples": int(len(cohort)),
        "samples_with_no_positive_sam_gain": int(np.count_nonzero(gain <= 0.0)),
        "fraction_with_any_positive_sam_gain": float(np.mean(gain > 0.0)),
        "baseline": baseline_summary,
        "ORACLE_GT_SELECTED": oracle_summary,
        "fraction_of_paper_gap_theoretically_recoverable": gap,
    }
    (root / "ORACLE_GT_SELECTED_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
