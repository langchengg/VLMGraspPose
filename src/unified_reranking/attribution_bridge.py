"""Two-by-two fair-pool versus historical-selector attribution bridge."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import evaluate_order_only


def historical_selector_on_fair(frame: pd.DataFrame) -> np.ndarray:
    """Transfer the compatible historical ``q * centre_support`` selector."""

    required = {"native_score", "p_center"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fair bridge table misses compatible selector inputs: {missing}")
    quality = pd.to_numeric(frame["native_score"], errors="coerce").to_numpy(float)
    support = pd.to_numeric(frame["p_center"], errors="coerce").to_numpy(float)
    if not np.isfinite(quality).all() or not np.isfinite(support).all():
        raise ValueError("fair historical-selector inputs must be finite")
    return quality * np.clip(support, 0.0, 1.0)


def _evaluate_cell(
    sample_ids: Iterable[object],
    frame: pd.DataFrame,
    *,
    score_column: str,
    pool: str,
    selector: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    metrics, decisions = evaluate_order_only(sample_ids, frame, score_column=score_column)
    all_solvable = int(frame.loc[frame["candidate_success"].astype(bool), "sample_id"].nunique())
    denominator = int(metrics["sample_count"])
    row: dict[str, object] = {
        "candidate_pool": pool,
        "selector": selector,
        **metrics,
        "oracle_all_numerator": all_solvable,
        "oracle_all": all_solvable / denominator,
        "candidate_rows": len(frame),
        "candidate_bearing_samples": int(frame["sample_id"].nunique()),
        "no_output_samples": denominator - int(frame["sample_id"].nunique()),
    }
    decisions = decisions.assign(candidate_pool=pool, selector=selector)
    return row, decisions


def evaluate_attribution_bridge(
    sample_ids: Iterable[object],
    fair: pd.DataFrame,
    historical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all four pool/selector cells with a shared denominator."""

    universe = tuple(map(str, sample_ids))
    for name, frame in (("fair", fair), ("historical", historical)):
        required = {"sample_id", "candidate_id", "native_rank", "candidate_success"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} bridge table missing columns: {missing}")
        if frame[["sample_id", "candidate_id"]].duplicated().any():
            raise ValueError(f"{name} bridge table contains duplicate identities")
    fair_work = fair.copy()
    fair_work["fair_native_selector"] = pd.to_numeric(fair_work["native_score"], errors="raise")
    fair_work["historical_selector"] = historical_selector_on_fair(fair_work)
    historical_work = historical.copy()
    historical_work["fair_native_selector"] = pd.to_numeric(
        historical_work["raw_network_quality"], errors="raise"
    )
    historical_work["historical_selector"] = pd.to_numeric(
        historical_work["original_score"], errors="raise"
    )

    rows: list[dict[str, object]] = []
    decision_parts: list[pd.DataFrame] = []
    for pool, frame in (("fair_gaussian", fair_work), ("historical_nms", historical_work)):
        for selector in ("fair_native_selector", "historical_selector"):
            row, decisions = _evaluate_cell(
                universe, frame, score_column=selector, pool=pool, selector=selector
            )
            rows.append(row)
            decision_parts.append(decisions)
    return pd.DataFrame(rows), pd.concat(decision_parts, ignore_index=True)


def _geometry_keys(frame: pd.DataFrame, *, include_height: bool) -> set[tuple[object, ...]]:
    columns = ["sample_id", "cx_px", "cy_px", "theta_deg", "width_px"]
    if include_height:
        columns.append("height_px")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"geometry overlap table missing: {missing}")
    values = frame[columns].copy()
    values["sample_id"] = values["sample_id"].astype(str)
    values["theta_deg"] = np.mod(pd.to_numeric(values["theta_deg"], errors="raise"), 180.0)
    for column in columns[1:]:
        values[column] = np.round(pd.to_numeric(values[column], errors="raise"), 4)
    return set(map(tuple, values.to_numpy()))


def candidate_membership_overlap(fair: pd.DataFrame, historical: pd.DataFrame) -> dict[str, object]:
    """Report pose-only and full-rectangle membership overlap."""

    result: dict[str, object] = {}
    for name, include_height in (("pose", False), ("full_geometry", True)):
        left = _geometry_keys(fair, include_height=include_height)
        right = _geometry_keys(historical, include_height=include_height)
        intersection = len(left & right)
        union = len(left | right)
        result[name] = {
            "fair": len(left),
            "historical": len(right),
            "intersection": intersection,
            "union": union,
            "jaccard": None if union == 0 else intersection / union,
            "fair_coverage": None if not left else intersection / len(left),
            "historical_coverage": None if not right else intersection / len(right),
        }
    return result


__all__ = [
    "candidate_membership_overlap",
    "evaluate_attribution_bridge",
    "historical_selector_on_fair",
]
