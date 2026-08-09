#!/usr/bin/env python3
"""Descriptive grouped analysis of locked formal outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def _aggregate(table: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in table.groupby(column, dropna=False, sort=True):
        row: dict[str, Any] = {
            "grouping": column, "group": str(value), "samples": len(group),
            "baseline_mean_iou": float(group["baseline_iou"].mean()),
            "hybrid_mean_iou": float(group["hybrid_iou"].mean()),
            "mean_delta_iou": float(group["delta_iou"].mean()),
            "trigger_rate": float(group["sam_invoked"].mean()),
            "sam_acceptance_rate": float((group["selected_source"] == "sam3").mean()),
            "accepted_harmful_count": int(np.count_nonzero((group["selected_source"] == "sam3") & (group["delta_iou"] < 0.0))),
        }
        for threshold in THRESHOLDS:
            percent = int(threshold * 100)
            row[f"baseline_p_at_{percent}"] = float(np.mean(group["baseline_iou"] > threshold))
            row[f"hybrid_p_at_{percent}"] = float(np.mean(group["hybrid_iou"] > threshold))
            row[f"delta_p_at_{percent}"] = row[f"hybrid_p_at_{percent}"] - row[f"baseline_p_at_{percent}"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    formal = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/report/formal_per_sample_metrics.parquet")
    diagnostics = pd.read_parquet(
        ROOT / "outputs/selective_sam3_vg/diagnostics/baseline_formal_test_per_sample_metrics.parquet"
    )[
        [
            "sample_id", "target_area_fraction", "error_type", "connected_component_count",
            "largest_component_ratio", "mask_precision", "mask_recall", "target_category",
        ]
    ]
    table = formal.merge(diagnostics, on="sample_id", how="left", validate="one_to_one", suffixes=("", "_baseline"))
    table["target_area_bin"] = pd.cut(
        table["target_area_fraction"], [-1.0, 0.005, 0.015, 0.04, 1.0],
        labels=["tiny", "small", "medium", "large"], right=False,
    )
    table["baseline_iou_bin"] = pd.cut(
        table["baseline_iou"], [-1.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0001],
        labels=["00_25", "25_50", "50_60", "60_70", "70_80", "80_90", "90_100"], right=False,
    )
    table["component_count_bin"] = table["connected_component_count"].clip(upper=4).map(
        {0: "0", 1: "1", 2: "2", 3: "3", 4: "4_plus"}
    )
    table["trigger_group"] = np.where(table["sam_invoked"], "triggered", "not_triggered")
    table["acceptance_group"] = np.where(table["selected_source"] == "sam3", "sam_accepted", "hifi_retained")
    group_columns = (
        "query_type", "target_category", "target_area_bin", "error_type", "component_count_bin",
        "baseline_iou_bin", "scene_id", "trigger_group", "acceptance_group",
    )
    output = ROOT / "outputs/selective_sam3_vg/report/grouped"
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for column in group_columns:
        result = _aggregate(table, column)
        result.to_csv(output / f"results_by_{column}.csv", index=False)
        all_rows.append(result)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_parquet(output / "all_grouped_results.parquet", index=False)
    accepted = table[table["selected_source"] == "sam3"]
    best_help = (
        _aggregate(table, "baseline_iou_bin")
        .sort_values("mean_delta_iou", ascending=False)
        .head(3)[["group", "mean_delta_iou", "samples"]]
        .to_dict(orient="records")
    )
    harmful = accepted[accepted["delta_iou"] < 0.0]
    summary = {
        "status": "COMPLETED",
        "analysis_type": "descriptive; no causal claim",
        "clutter_level_available": False,
        "clutter_level_note": "No authoritative clutter-level field was present in the frozen local manifest.",
        "accepted_samples": len(accepted),
        "accepted_improved_samples": int(np.count_nonzero(accepted["delta_iou"] > 0.0)),
        "accepted_harmful_samples": len(harmful),
        "largest_descriptive_help_by_baseline_iou_bin": best_help,
        "harmful_relation_query_count": int(np.count_nonzero(harmful["query_type"] == "relation")),
        "harmful_wrong_target_proxy_count_baseline_iou_lt_0_25": int(np.count_nonzero(harmful["baseline_iou"] < 0.25)),
        "harmful_already_correct_count_baseline_iou_ge_0_90": int(np.count_nonzero(harmful["baseline_iou"] >= 0.90)),
        "limitations": "Groups are observational diagnostics and do not identify causal effects.",
    }
    (output / "grouped_failure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
