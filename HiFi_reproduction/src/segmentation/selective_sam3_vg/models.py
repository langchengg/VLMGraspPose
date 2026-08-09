"""Inference-only helpers for locked trigger and selector artifacts.

This module contains no ground-truth loader, metric, or validation target.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def trigger_probability(artifact: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    frame = pd.DataFrame([{key: features[key] for key in artifact["raw_feature_names"]}])
    score = artifact["estimator"].predict_proba(frame)[:, 1]
    return float(score[0])


def candidate_feature_frame(
    candidates: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append({name: float(candidate[name]) for name in feature_names})
    return pd.DataFrame(rows, columns=list(feature_names), dtype=np.float64)


def deterministic_candidate_score(candidate: Mapping[str, Any]) -> float:
    area_ratio = max(float(candidate["sam_to_coarse_area_ratio"]), 1.0e-6)
    return float(
        0.25 * float(candidate["sam_quality"])
        + 0.20 * float(candidate["coarse_sam_iou"])
        + 0.15 * min(float(candidate["hifi_probability_mass_recall"]), 1.0)
        + 0.10 * float(candidate["positive_point_inclusion_ratio"])
        + 0.10 * float(candidate["largest_component_ratio"])
        + 0.10 * float(candidate["rgb_edge_alignment"])
        + 0.05 * float(candidate["depth_consistency_with_positive_region"])
        - 0.15 * float(candidate["low_hifi_probability_expansion_fraction"])
        - 0.05 * float(candidate["fragmentation_penalty"])
        - 0.05 * abs(float(np.log(area_ratio)))
    )


def passes_conservative_gate(candidate: Mapping[str, Any], rules: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(candidate["positive_point_inclusion_ratio"]) < float(rules["minimum_positive_point_inclusion"]):
        reasons.append("positive_point_support")
    if float(candidate["hifi_probability_mass_recall"]) < float(rules["minimum_probability_mass_recall"]):
        reasons.append("probability_mass_recall")
    area = float(candidate["sam_to_coarse_area_ratio"])
    if area < float(rules["minimum_area_ratio"]):
        reasons.append("area_ratio_below_minimum")
    if area > float(rules["maximum_area_ratio"]):
        reasons.append("area_ratio_above_maximum")
    if float(candidate["low_hifi_probability_expansion_fraction"]) > float(rules["maximum_low_probability_expansion"]):
        reasons.append("low_probability_expansion")
    if float(candidate["fragmentation_penalty"]) > float(rules["maximum_fragmentation_penalty"]):
        reasons.append("fragmentation")
    if float(candidate["prompt_box_support"]) < float(rules["minimum_prompt_box_support"]):
        reasons.append("outside_prompt_box")
    return not reasons, reasons


def selector_scores(artifact: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if artifact["selector_type"] == "deterministic_rule":
        return np.asarray([deterministic_candidate_score(item) for item in candidates], dtype=np.float64)
    frame = candidate_feature_frame(candidates, artifact["feature_names"])
    return np.asarray(artifact["estimator"].predict(frame), dtype=np.float64)


def select_candidate(
    artifact: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not candidates or str(candidates[0]["candidate_id"]) != "coarse_0":
        raise ValueError("the original coarse mask must be candidate zero")
    scores = selector_scores(artifact, candidates)
    coarse_score = float(scores[0])
    eligible: list[tuple[float, str, int]] = []
    rejected: dict[str, list[str]] = {}
    for index, candidate in enumerate(candidates[1:], start=1):
        candidate_id = str(candidate["candidate_id"])
        valid, reasons = passes_conservative_gate(candidate, artifact["conservative_gate"])
        if valid:
            eligible.append((float(scores[index]), candidate_id, index))
        else:
            rejected[candidate_id] = reasons
    if not eligible:
        return {
            "selected_index": 0,
            "selected_candidate_id": "coarse_0",
            "selected_source": "hifics",
            "selector_score": coarse_score,
            "coarse_score": coarse_score,
            "predicted_gain": 0.0,
            "fallback_reason": "no_sam_candidate_passed_conservative_gate",
            "rejected_candidates": rejected,
        }
    best_score, candidate_id, index = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
    gain = float(best_score - coarse_score)
    if gain <= float(artifact["acceptance_margin"]):
        return {
            "selected_index": 0,
            "selected_candidate_id": "coarse_0",
            "selected_source": "hifics",
            "selector_score": best_score,
            "coarse_score": coarse_score,
            "predicted_gain": gain,
            "fallback_reason": "predicted_gain_not_above_locked_margin",
            "rejected_candidates": rejected,
        }
    return {
        "selected_index": index,
        "selected_candidate_id": candidate_id,
        "selected_source": "sam3",
        "selector_score": best_score,
        "coarse_score": coarse_score,
        "predicted_gain": gain,
        "fallback_reason": None,
        "rejected_candidates": rejected,
    }
