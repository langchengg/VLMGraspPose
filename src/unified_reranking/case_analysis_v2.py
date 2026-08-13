"""Deterministic post-formal case scoring and selection for v2 visuals."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .case_visuals_v2 import ROUTES


SCORE_SCALE = 1_000_000
SCORE_WEIGHTS = {
    "target_visibility": 150,
    "crop_resolution": 100,
    "native_final_geometric_separation": 150,
    "matched_gt_visibility": 100,
    "mask_difference_visibility": 100,
    "threshold_margin_clarity": 150,
    "candidate_overlay_legibility": 100,
    "prompt_legibility": 50,
    "mechanism_purity": 100,
}
AUDIT_CATEGORIES = (
    "recovered",
    "harmful",
    "gate_prevented_harmful",
    "gate_missed_recoverable",
    "wrong_to_wrong_solvable",
    "candidate_generation_irreparable",
)
SHORT_FAILURE = {
    "crog": "candidate_generation_irreparable",
    "g1": "candidate_generation_irreparable",
    "c1": "harmful",
}


def _clip(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(number, 0.0, 1.0)) if math.isfinite(number) else 0.0


def _periodic_angle(first: Any, second: Any) -> float:
    try:
        difference = float(first) - float(second)
    except (TypeError, ValueError):
        return 0.0
    return abs((difference + 90.0) % 180.0 - 90.0)


def clarity_components(row: Mapping[str, Any]) -> dict[str, float]:
    """Compute visual/geometric clarity only; outcome is not a score input."""

    bbox_w = max(float(row.get("target_bbox_width", 0.0)), 0.0)
    bbox_h = max(float(row.get("target_bbox_height", 0.0)), 0.0)
    crop_w = max(float(row.get("crop_width_px", 0.0)), 0.0)
    crop_h = max(float(row.get("crop_height_px", 0.0)), 0.0)
    target_diag = max(math.hypot(bbox_w, bbox_h), 1.0)
    centre_distance = math.hypot(
        float(row.get("native_cx_px", 0.0)) - float(row.get("final_cx_px", 0.0)),
        float(row.get("native_cy_px", 0.0)) - float(row.get("final_cy_px", 0.0)),
    )
    angle_distance = _periodic_angle(
        row.get("native_theta_deg", 0.0), row.get("final_theta_deg", 0.0)
    )
    width_distance = abs(
        float(row.get("native_width_px", 0.0))
        - float(row.get("final_width_px", 0.0))
    )
    geometric = _clip(
        0.50 * min(centre_distance / target_diag, 1.0)
        + 0.35 * min(angle_distance / 90.0, 1.0)
        + 0.15 * min(width_distance / target_diag, 1.0)
    )
    gt_width = max(float(row.get("final_matched_gt_width_px", 0.0)), 0.0)
    gt_height = max(float(row.get("final_matched_gt_height_px", 0.0)), 0.0)
    matched_gt = _clip(0.65 * min(gt_width / 60.0, 1.0) + 0.35 * min(gt_height / 20.0, 1.0))
    mask_iou = _clip(row.get("mask_iou", 0.0))
    # Large disagreement and near-complete overlap are both visually decisive;
    # the ambiguous middle around the 0.25/0.50 diagnostic band is downweighted.
    mask_visibility = _clip(abs(mask_iou - 0.375) / 0.625)
    margins = []
    for prefix in ("native", "final"):
        iou = float(row.get(f"{prefix}_diagnostic_iou", float("nan")))
        angle = float(row.get(f"{prefix}_diagnostic_angle_error_deg", float("nan")))
        if math.isfinite(iou) and math.isfinite(angle):
            margins.append(
                min(abs(iou - 0.25) / 0.25, abs(angle - 30.0) / 30.0, 1.0)
            )
    threshold = min(margins) if margins else 0.0
    count = max(int(row.get("candidate_count_all", 0)), 0)
    overlay = _clip(1.0 - max(count - 5, 0) / 15.0)
    prompt_length = len(str(row.get("language", "")))
    prompt = _clip((140.0 - prompt_length) / 80.0)
    purity = _clip(row.get("dominant_abs_share", 0.0))
    if str(row.get("candidate_generation_verdict", "")) in {"FAIL", "NO OUTPUT"}:
        purity = max(purity, 0.90)
    return {
        "target_visibility": _clip(math.sqrt(max(bbox_w * bbox_h, 0.0)) / 120.0),
        "crop_resolution": _clip(min(crop_w, crop_h) / 200.0),
        "native_final_geometric_separation": geometric,
        "matched_gt_visibility": matched_gt,
        "mask_difference_visibility": mask_visibility,
        "threshold_margin_clarity": _clip(threshold),
        "candidate_overlay_legibility": overlay,
        "prompt_legibility": prompt,
        "mechanism_purity": purity,
    }


def clarity_score(components: Mapping[str, Any]) -> tuple[int, float]:
    if set(components) != set(SCORE_WEIGHTS):
        raise ValueError("clarity component inventory differs from the fixed protocol")
    quantized = {
        name: int(round(_clip(value) * SCORE_SCALE))
        for name, value in components.items()
    }
    numerator = sum(SCORE_WEIGHTS[name] * quantized[name] for name in SCORE_WEIGHTS)
    return numerator, float(numerator / (1000 * SCORE_SCALE))


def presentation_categories(row: Mapping[str, Any]) -> tuple[str, ...]:
    categories: list[str] = []
    base = str(row.get("analysis_category", ""))
    mapping = {
        "recovered": "recovered",
        "harmful": "harmful",
        "gate_prevented_harmful": "gate_prevented_harmful",
        "gate_missed_recoverable": "gate_missed_recoverable",
        "wrong_to_wrong_solvable": "wrong_to_wrong_solvable",
        "no_positive_pool": "candidate_generation_irreparable",
        "correct_to_correct_switch": "correct_to_correct_large_rank_change",
    }
    if base in mapping:
        categories.append(mapping[base])
    if int(row.get("candidate_count_all", 0)) == 0:
        categories.append("no_output")
    if bool(row.get("full_pool_positive", False)) and not bool(
        row.get("top5_positive", False)
    ):
        categories.append("positive_only_below_top5")
    if bool(row.get("top5_positive", False)) and not bool(
        row.get("native_correct", False)
    ):
        categories.append("ranking_limited")
    route = str(row.get("route", ""))
    grounding = str(row.get("grounding_verdict", ""))
    if route in {"g1", "c1"}:
        if str(row.get("bridge_category", "")) == "grounding_limited":
            categories.append("grounding_supported")
        if grounding == "PASS" and not bool(row.get("full_pool_positive", False)):
            categories.append("good_grounding_no_candidate")
        if grounding == "PASS" and bool(row.get("top5_positive", False)) and not bool(row.get("native_correct", False)):
            categories.append("good_grounding_native_ranking_failure")
    elif route == "crog":
        if grounding == "PASS" and bool(row.get("top5_positive", False)) and base == "recovered":
            categories.append("crog_good_mask_solvable_recovered")
        if grounding == "FAIL" and not bool(row.get("full_pool_positive", False)):
            categories.append("crog_poor_mask_no_positive_association")
        if grounding == "PASS" and not bool(row.get("full_pool_positive", False)):
            categories.append("crog_good_mask_no_positive")
    return tuple(dict.fromkeys(categories))


def make_eligible_cases(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in cases.to_dict("records"):
        components = clarity_components(record)
        numerator, score = clarity_score(components)
        same_gt = (
            pd.notna(record.get("native_matched_gt_index"))
            and pd.notna(record.get("final_matched_gt_index"))
            and int(record["native_matched_gt_index"])
            == int(record["final_matched_gt_index"])
        )
        asset_ok = all(
            bool(record.get(name, False))
            for name in (
                "rgb_exists",
                "gt_mask_exists",
                "predicted_mask_available",
                "candidate_geometry_valid",
            )
        )
        mandatory = bool(
            asset_ok
            and same_gt
            and float(record.get("target_bbox_width", 0.0)) >= 28.0
            and float(record.get("target_bbox_height", 0.0)) >= 18.0
            and float(record.get("crop_width_px", 0.0)) >= 90.0
            and float(record.get("crop_height_px", 0.0)) >= 70.0
            and len(str(record.get("language", ""))) <= 140
            and float(record.get("image_quality_score", 0.0)) >= 0.15
            and float(record.get("highlight_clip_fraction", 1.0)) <= 0.20
        )
        for category in presentation_categories(record):
            tie = hashlib.sha256(
                f"{record['route']}|{category}|{record['sample_id']}".encode()
            ).hexdigest()
            rows.append(
                {
                    "route": record["route"],
                    "presentation_outcome": category,
                    "sample_id": record["sample_id"],
                    "scene_id": record["scene_id"],
                    "mandatory_eligible": mandatory,
                    "same_gt_native_final": same_gt,
                    "presentation_clarity_numerator": numerator,
                    "presentation_clarity_score": score,
                    "selection_tiebreak_sha256": tie,
                    **{f"clarity::{name}": value for name, value in components.items()},
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("no qualitative cases were classified")
    return result.sort_values(
        ["route", "presentation_outcome", "mandatory_eligible", "presentation_clarity_numerator", "selection_tiebreak_sha256", "sample_id"],
        ascending=[True, True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def select_audit_cases(eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected: list[pd.DataFrame] = []
    alternatives: list[pd.DataFrame] = []
    for route in ROUTES:
        for category in AUDIT_CATEGORIES:
            group = eligible[
                (eligible["route"] == route)
                & (eligible["presentation_outcome"] == category)
                & eligible["mandatory_eligible"].astype(bool)
            ].sort_values(
                ["presentation_clarity_numerator", "selection_tiebreak_sha256", "sample_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            if group.empty:
                alternatives.append(
                    pd.DataFrame(
                        [{"route": route, "presentation_outcome": category, "selection_status": "UNAVAILABLE"}]
                    )
                )
                continue
            head = group.head(6).copy()
            head["alternative_rank"] = np.arange(1, len(head) + 1)
            head["selection_status"] = np.where(head["alternative_rank"] == 1, "SELECTED", "ALTERNATIVE")
            alternatives.append(head)
            selected.append(head.head(1))
    if not selected:
        raise RuntimeError("no audit cases satisfy the mandatory filters")
    chosen = pd.concat(selected, ignore_index=True)
    if len(chosen) != len(ROUTES) * len(AUDIT_CATEGORIES):
        missing = [
            (route, category)
            for route in ROUTES
            for category in AUDIT_CATEGORIES
            if not ((chosen["route"] == route) & (chosen["presentation_outcome"] == category)).any()
        ]
        raise RuntimeError(f"required route/outcome case unavailable: {missing}")
    return chosen, pd.concat(alternatives, ignore_index=True, sort=False)


def short_deck_cases(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for route in ROUTES:
        for category in ("recovered", SHORT_FAILURE[route]):
            matched = audit[
                (audit["route"] == route)
                & (audit["presentation_outcome"] == category)
            ]
            if len(matched) != 1:
                raise RuntimeError(f"short-deck case missing for {route}/{category}")
            rows.append(matched)
    result = pd.concat(rows, ignore_index=True)
    result["short_deck_order"] = np.arange(1, len(result) + 1)
    return result


def cross_route_candidates(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_id, frame in cases.groupby("sample_id", sort=False):
        if set(frame["route"]) != set(ROUTES) or len(frame) != 3:
            continue
        if frame["final_matched_gt_index"].isna().any() or frame[
            "final_matched_gt_index"
        ].nunique() != 1:
            continue
        if not frame[["rgb_exists", "gt_mask_exists", "predicted_mask_available"]].all(axis=None):
            continue
        centres = frame[["final_cx_px", "final_cy_px"]].to_numpy(float)
        centre_sep = max(
            np.linalg.norm(centres[i] - centres[j])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        angles = frame["final_theta_deg"].to_numpy(float)
        angle_sep = max(
            _periodic_angle(angles[i], angles[j])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        diag = max(
            math.hypot(float(frame.iloc[0].target_bbox_width), float(frame.iloc[0].target_bbox_height)),
            1.0,
        )
        geometry = _clip(0.65 * min(centre_sep / diag, 1.0) + 0.35 * min(angle_sep / 90.0, 1.0))
        outcome_diversity = frame["outcome"].nunique() / 3.0
        issue_diversity = frame["earliest_observable_issue"].nunique() / 3.0
        clarity = float(frame["presentation_clarity_score"].mean())
        score = 0.45 * geometry + 0.25 * outcome_diversity + 0.20 * issue_diversity + 0.10 * clarity
        rows.append(
            {
                "sample_id": sample_id,
                "scene_id": frame.iloc[0].scene_id,
                "cross_route_gt_index": int(frame.iloc[0].final_matched_gt_index),
                "geometry_separation": geometry,
                "outcome_diversity": outcome_diversity,
                "issue_diversity": issue_diversity,
                "mean_route_clarity": clarity,
                "cross_route_score": score,
                "failed_routes": int((~frame["gated_correct"].astype(bool)).sum()),
                "selection_tiebreak_sha256": hashlib.sha256(f"cross|{sample_id}".encode()).hexdigest(),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["cross_route_score", "selection_tiebreak_sha256"],
        ascending=[False, True],
        kind="mergesort",
    )
    first = result.head(1)
    second_pool = result[
        (result["scene_id"] != first.iloc[0].scene_id)
        & (result["failed_routes"] >= 1)
        & (result["issue_diversity"] >= 2.0 / 3.0)
    ]
    second = second_pool.head(1)
    selected = pd.concat([first, second], ignore_index=True)
    if len(selected) != 2 or selected["sample_id"].nunique() != 2:
        raise RuntimeError("two distinct cross-route cases are unavailable")
    selected["cross_route_order"] = [1, 2]
    return selected


def expected_taxonomy_counts() -> dict[str, list[int]]:
    return {
        "crog": [0, 456, 0, 371, 6848, 278, 37, 170, 2145, 14, 16],
        "g1": [41, 3159, 6, 822, 3647, 729, 29, 1049, 1338, 14, 22],
        "c1": [13, 3143, 7, 1149, 3363, 1006, 52, 1506, 950, 12, 11],
    }


def reconcile_taxonomy(cases: pd.DataFrame) -> dict[str, Any]:
    observed: dict[str, list[int]] = {}
    mismatches = []
    for route in ROUTES:
        frame = cases[cases["route"] == route]
        counts = [int(frame[f"E{index}"].astype(bool).sum()) for index in range(11)]
        observed[route] = counts
        if counts != expected_taxonomy_counts()[route]:
            mismatches.append(
                {"route": route, "observed": counts, "expected": expected_taxonomy_counts()[route]}
            )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "denominator": int(cases["sample_id"].nunique()),
        "route_rows": {route: int((cases["route"] == route).sum()) for route in ROUTES},
        "taxonomy_order": [f"E{index}" for index in range(11)],
        "observed": observed,
        "expected": expected_taxonomy_counts(),
        "mismatches": mismatches,
    }


__all__ = [
    "AUDIT_CATEGORIES",
    "SCORE_SCALE",
    "SCORE_WEIGHTS",
    "SHORT_FAILURE",
    "clarity_components",
    "clarity_score",
    "cross_route_candidates",
    "expected_taxonomy_counts",
    "make_eligible_cases",
    "presentation_categories",
    "reconcile_taxonomy",
    "select_audit_cases",
    "short_deck_cases",
]
