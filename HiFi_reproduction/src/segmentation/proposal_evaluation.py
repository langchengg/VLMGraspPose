"""Deterministic Stage-1 candidate-selection baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .selective_sam3_vg.metrics import summarize_ious


def _select(
    frame: pd.DataFrame,
    mask: pd.Series,
    score: str,
    *,
    secondary: str = "hifi_candidate_iou",
) -> pd.DataFrame:
    fallback = frame[frame["source_family"] == "HIFI_ORIGINAL"]
    if fallback["sample_id"].nunique() != frame["sample_id"].nunique():
        raise ValueError("every sample must include exactly one original HiFi candidate")
    subset = frame[mask & frame["eligible_final"]].copy()
    subset = subset[np.isfinite(pd.to_numeric(subset[score], errors="coerce"))]
    selected = (
        subset.sort_values(
            ["sample_id", score, secondary, "candidate_id"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .drop_duplicates("sample_id", keep="first")
        .set_index("sample_id")
    )
    missing = fallback[~fallback["sample_id"].isin(selected.index)].set_index("sample_id")
    return pd.concat([selected, missing]).reset_index().sort_values("sample_id")


def deterministic_rule_score(frame: pd.DataFrame) -> np.ndarray:
    score = 0.15 * frame["hifi_candidate_iou"].fillna(0.0).to_numpy(float)
    clip = frame.get("clip_target_category_similarity", pd.Series(0.0, index=frame.index))
    score += 0.10 * clip.fillna(0.0).to_numpy(float)
    colour = frame.get("requested_colour_match_fraction", pd.Series(np.nan, index=frame.index))
    colour_active = frame.get("has_colour", pd.Series(False, index=frame.index)).to_numpy(bool)
    score += np.where(colour_active, 0.20 * colour.fillna(0.0).to_numpy(float), 0.0)
    relation_active = frame.get("has_relation", pd.Series(False, index=frame.index)).to_numpy(bool)
    relation = frame.get("relation_max_consistency", pd.Series(np.nan, index=frame.index))
    score += np.where(relation_active, 0.45 * relation.fillna(0.0).to_numpy(float), 0.0)
    locations = frame.get("absolute_location_type", pd.Series(None, index=frame.index)).fillna("")
    for location, column in (
        ("leftmost", "leftmost_score"),
        ("rightmost", "rightmost_score"),
        ("closest", "closest_score"),
        ("furthest", "furthest_score"),
    ):
        active = locations.to_numpy() == location
        values = frame.get(column, pd.Series(0.0, index=frame.index)).fillna(0.0).to_numpy(float)
        score += np.where(active, 0.45 * values, 0.0)
    score += 0.05 * frame["depth_reliability"].fillna(0.0).to_numpy(float)
    return score


def simple_baseline_decisions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"sample_id", "candidate_id", "candidate_iou", "source_family", "eligible_final"}
    if not required.issubset(frame.columns):
        raise ValueError(f"baseline table missing {sorted(required - set(frame.columns))}")
    data = frame.copy()
    data["deterministic_rule_score"] = deterministic_rule_score(data)
    all_true = pd.Series(True, index=data.index)
    source = data["source_family"]
    decisions = {
        "B0_hifi_original": _select(
            data, source == "HIFI_ORIGINAL", "hifi_candidate_iou"
        ),
        "B2_full_query_highest_sam": _select(
            data, source == "TEXT_FULL_QUERY", "sam_score"
        ),
        "B3_target_category_highest_sam": _select(
            data, source == "TEXT_TARGET_CATEGORY", "sam_score"
        ),
        "B4_target_attribute_highest_sam": _select(
            data, source == "TEXT_TARGET_ATTRIBUTE", "sam_score"
        ),
        "B5_bank_highest_sam": _select(data, all_true, "sam_score"),
        "B6_max_hifi_overlap": _select(data, all_true, "hifi_candidate_iou"),
        "B7_max_hifi_mass_precision": _select(
            data, all_true, "hifi_probability_mass_precision"
        ),
        "B8_max_clip_similarity": _select(
            data, data["clip_features_valid"], "clip_full_query_similarity"
        ),
        "B9_relation_aware_rules": _select(
            data, all_true, "deterministic_rule_score"
        ),
        "B12_gt_best_oracle": _select(data, all_true, "candidate_iou"),
    }
    metrics = {
        name: {
            **summarize_ious(selected["candidate_iou"].to_numpy()),
            "non_deployable_gt_oracle": name == "B12_gt_best_oracle",
        }
        for name, selected in decisions.items()
    }
    combined = pd.concat(
        [selected.assign(method=name) for name, selected in decisions.items()],
        ignore_index=True,
    )
    return combined, metrics


__all__ = ["deterministic_rule_score", "simple_baseline_decisions"]
