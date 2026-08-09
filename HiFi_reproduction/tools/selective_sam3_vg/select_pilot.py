#!/usr/bin/env python3
"""Select a deterministic, scene-diverse 300-sample validation pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS = ROOT / "outputs/selective_sam3_vg/diagnostics/baseline_validation_per_sample_metrics.parquet"
OUTPUT = ROOT / "outputs/selective_sam3_vg/validation_pilot"
SEED = 42
COUNT = 300


def _stable_random(sample_id: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{sample_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def _iou_band(value: float) -> str:
    edges = (0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0000001)
    labels = ("00_25", "25_50", "50_60", "60_70", "70_80", "80_90", "90_100")
    for lower, upper, label in zip(edges[:-1], edges[1:], labels, strict=True):
        if lower <= value < upper:
            return label
    raise ValueError(value)


def _area_band(value: float) -> str:
    if value < 0.005:
        return "tiny"
    if value < 0.015:
        return "small"
    if value < 0.04:
        return "medium"
    return "large"


def main() -> None:
    table = pd.read_parquet(DIAGNOSTICS).copy()
    if len(table) != 3778 or table["scene_id"].nunique() != 165:
        raise RuntimeError("unexpected authoritative validation diagnostics")
    table["iou_band"] = table["iou"].map(_iou_band)
    table["target_area_band"] = table["target_area_fraction"].map(_area_band)
    table["random_key"] = table["sample_id"].map(_stable_random)
    table["joint_stratum"] = (
        table["iou_band"].astype(str)
        + "|" + table["query_type"].astype(str)
        + "|" + table["target_area_band"].astype(str)
        + "|" + table["error_type"].astype(str)
    )
    stratum_count = table["joint_stratum"].value_counts().to_dict()
    category_count = table["target_category"].value_counts().to_dict()
    table["rarity"] = table.apply(
        lambda row: 1.0 / math.sqrt(stratum_count[row["joint_stratum"]])
        + 0.25 / math.sqrt(category_count[row["target_category"]]),
        axis=1,
    )

    selected: list[int] = []
    # First cover every validation scene. Within a scene prefer rare diagnostic
    # combinations, then use the fixed hash as a deterministic tie breaker.
    for _, group in table.groupby("scene_id", sort=True):
        chosen = group.sort_values(["rarity", "random_key"], ascending=[False, True]).index[0]
        selected.append(int(chosen))

    remaining = table.drop(index=selected).copy()
    represented = table.loc[selected, "joint_stratum"].value_counts().to_dict()
    category_represented = table.loc[selected, "target_category"].value_counts().to_dict()
    while len(selected) < COUNT:
        remaining["selection_score"] = remaining.apply(
            lambda row: (
                4.0 / (1.0 + represented.get(row["joint_stratum"], 0))
                + 1.0 / (1.0 + category_represented.get(row["target_category"], 0))
                + float(row["rarity"])
            ),
            axis=1,
        )
        chosen = remaining.sort_values(
            ["selection_score", "random_key"], ascending=[False, True]
        ).index[0]
        row = remaining.loc[chosen]
        selected.append(int(chosen))
        represented[row["joint_stratum"]] = represented.get(row["joint_stratum"], 0) + 1
        category_represented[row["target_category"]] = category_represented.get(row["target_category"], 0) + 1
        remaining = remaining.drop(index=chosen)

    pilot = table.loc[selected].copy().reset_index(drop=True)
    pilot.insert(0, "pilot_order", np.arange(len(pilot), dtype=np.int64))
    if len(pilot) != COUNT or pilot["scene_id"].nunique() != table["scene_id"].nunique():
        raise AssertionError("pilot selection did not preserve required size and scene coverage")
    for column in ("iou_band", "query_type", "target_area_band", "error_type"):
        missing = set(table[column].unique()) - set(pilot[column].unique())
        if missing:
            raise AssertionError(f"pilot misses {column} levels: {sorted(missing)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pilot.drop(columns=["selection_score"], errors="ignore").to_csv(
        OUTPUT / "pilot_stratification_manifest.csv", index=False
    )
    pilot.drop(columns=["selection_score"], errors="ignore").to_parquet(
        OUTPUT / "pilot_stratification_manifest.parquet", index=False
    )
    inference_columns = (
        "pilot_order", "sample_index", "sample_id", "question_index", "scene_id", "query",
        "rgb_path", "depth_path", "probability_path", "native_mask_path",
    )
    with (OUTPUT / "pilot_inference_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in pilot.loc[:, inference_columns].to_dict(orient="records"):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "status": "COMPLETED",
        "seed": SEED,
        "sample_count": len(pilot),
        "validation_sample_count": len(table),
        "validation_scene_count": int(table["scene_id"].nunique()),
        "pilot_scene_count": int(pilot["scene_id"].nunique()),
        "coverage": {
            column: pilot[column].value_counts().sort_index().to_dict()
            for column in ("iou_band", "query_type", "target_area_band", "target_category", "error_type")
        },
        "selection_uses_validation_gt_for_stratification": True,
        "inference_manifest_contains_gt": False,
    }
    (OUTPUT / "selection_provenance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
