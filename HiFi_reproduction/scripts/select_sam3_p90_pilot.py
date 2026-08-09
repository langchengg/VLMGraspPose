#!/usr/bin/env python3
"""Select a deterministic GT-stratified validation pilot by whole RGB frames."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/baseline_reproduction/per_sample_metrics.parquet"
OUTPUT = PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/pilot/pilot_manifest.parquet"


def stable_order(value: str) -> str:
    return hashlib.sha256(f"42|{value}".encode()).hexdigest()


def round_robin(strata: dict[tuple[str, ...], list[int]]) -> list[int]:
    keys = sorted(strata)
    offsets = {key: 0 for key in keys}
    result: list[int] = []
    while True:
        progress = False
        for key in keys:
            offset = offsets[key]
            if offset < len(strata[key]):
                result.append(strata[key][offset])
                offsets[key] += 1
                progress = True
        if not progress:
            return result


def main() -> int:
    frame = pd.read_parquet(BASELINE)
    frame = frame[frame["split"] == "val"].copy().reset_index(drop=True)
    frame["baseline_iou_band"] = pd.cut(
        frame["evaluator_iou_float32"],
        bins=[-np.inf, 0.25, 0.50, 0.70, 0.80, 0.90, np.inf],
        labels=["le25", "25_50", "50_70", "70_80", "80_90", "gt90"],
    ).astype(str)
    q25, q75 = frame["target_area_px"].quantile([0.25, 0.75])
    frame["target_area_bin"] = np.where(
        frame["target_area_px"] <= q25,
        "small",
        np.where(frame["target_area_px"] >= q75, "large", "medium"),
    )
    frame["error_balance"] = np.where(
        frame["false_negative_area_px"] > frame["false_positive_area_px"] * 1.1,
        "under_segmentation",
        np.where(
            frame["false_positive_area_px"] > frame["false_negative_area_px"] * 1.1,
            "over_segmentation",
            "balanced",
        ),
    )
    frame["stable_order"] = frame["sample_id"].map(stable_order)
    frame = frame.sort_values("stable_order", kind="stable").reset_index(drop=True)
    strata_columns = ["query_type", "baseline_iou_band", "target_area_bin", "error_balance"]
    strata: dict[tuple[str, ...], list[int]] = {}
    for index, row in frame.iterrows():
        key = tuple(str(row[column]) for column in strata_columns)
        strata.setdefault(key, []).append(int(index))
    ordered = round_robin(strata)

    selected_frames: set[str] = set()
    for index in ordered:
        selected_frames.add(str(frame.loc[index, "frame_id"]))
        if len(selected_frames) >= 100:
            break
    eligible = frame[frame["frame_id"].isin(selected_frames)]
    eligible_indices = set(int(value) for value in eligible.index)
    selected_indices = [index for index in ordered if index in eligible_indices][:500]
    selected = frame.loc[selected_indices].copy()
    if len(selected) != 500 or selected["frame_id"].nunique() < 100:
        raise RuntimeError("pilot selection failed its minimum frame/query contract")
    required_query_types = set(frame["query_type"])
    if set(selected["query_type"]) != required_query_types:
        raise RuntimeError("pilot does not cover all validation query types")
    selected["pilot_selection_uses_gt_for_stratification_only"] = True
    selected["pilot_rank"] = np.arange(len(selected), dtype=int)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(OUTPUT, index=False)
    (OUTPUT.parent / "pilot_sample_ids.txt").write_text(
        "\n".join(selected["sample_id"].astype(str)) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "status": "PILOT_SELECTION_LOCKED",
        "queries": int(len(selected)),
        "unique_rgb_frames": int(selected["frame_id"].nunique()),
        "query_types": selected["query_type"].value_counts().sort_index().to_dict(),
        "baseline_iou_bands": selected["baseline_iou_band"].value_counts().sort_index().to_dict(),
        "target_area_bins": selected["target_area_bin"].value_counts().sort_index().to_dict(),
        "error_balance": selected["error_balance"].value_counts().sort_index().to_dict(),
        "ground_truth_use": "validation stratification only; never passed to proposal generation",
        "manifest_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    (OUTPUT.parent / "pilot_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
