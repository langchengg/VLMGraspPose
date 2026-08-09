from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np


CONFIDENCE_GRID = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
MARGIN_GRID = tuple(round(0.05 * index, 2) for index in range(11))
OVERALL_GRID = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


@dataclass(frozen=True)
class LockedSafeThreshold:
    status: str
    confidence: float | None
    margin: float | None
    overall: float | None

    @property
    def enabled(self) -> bool:
        return self.status == "selected"


def calibration_grid_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "confidence": list(CONFIDENCE_GRID),
        "score_margin_top1_top2": list(MARGIN_GRID),
        "minimum_selected_overall_score": list(OVERALL_GRID),
        "combination_count": len(CONFIDENCE_GRID) * len(MARGIN_GRID) * len(OVERALL_GRID),
        "legacy_harmful_rate_limit": 0.01,
        "optimization": "max Legacy Net under harmful cap",
        "tie_break": [
            "higher outcome-changing precision",
            "lower switch coverage",
            "higher Corrected Net",
            "lower model cost",
        ],
    }


def sweep_safe_thresholds(
    examples: Iterable[dict[str, Any]],
    *,
    model_id: str,
    harmful_rate_limit: float = 0.01,
    model_cost_usd: float = 0.0,
) -> tuple[LockedSafeThreshold, list[dict[str, Any]], dict[str, Any]]:
    rows = list(examples)
    if not rows:
        raise ValueError("calibration examples are empty")
    valid = np.asarray([bool(row["valid"]) for row in rows], dtype=bool)
    abstain = np.asarray([bool(row.get("abstain", False)) for row in rows], dtype=bool)
    wants_switch = np.asarray([str(row["decision"]) == "switch" for row in rows], dtype=bool)
    differs = np.asarray(
        [str(row["selected_candidate_id"]) != str(row["q_only_candidate_id"]) for row in rows],
        dtype=bool,
    )
    confidence = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)
    margin = np.asarray([float(row["score_margin_top1_top2"]) for row in rows], dtype=np.float64)
    overall = np.asarray([float(row["selected_overall_score"]) for row in rows], dtype=np.float64)
    legacy_q = np.asarray([bool(row["legacy_q_only_correct"]) for row in rows], dtype=bool)
    legacy_selected = np.asarray([bool(row["legacy_selected_correct"]) for row in rows], dtype=bool)
    corrected_q = np.asarray([bool(row["corrected_q_only_correct"]) for row in rows], dtype=bool)
    corrected_selected = np.asarray([bool(row["corrected_selected_correct"]) for row in rows], dtype=bool)
    base_accept = valid & ~abstain & wants_switch & differs
    sweep: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    total = len(rows)
    for tau, delta, gamma in product(CONFIDENCE_GRID, MARGIN_GRID, OVERALL_GRID):
        accept = base_accept & (confidence >= tau) & (margin >= delta) & (overall >= gamma)
        legacy_new = np.where(accept, legacy_selected, legacy_q)
        corrected_new = np.where(accept, corrected_selected, corrected_q)
        recovered = int((~legacy_q & legacy_new).sum())
        harmful = int((legacy_q & ~legacy_new).sum())
        corrected_recovered = int((~corrected_q & corrected_new).sum())
        corrected_harmful = int((corrected_q & ~corrected_new).sum())
        switches = int(accept.sum())
        outcome_count = recovered + harmful
        record = {
            "model_id": str(model_id),
            "confidence_threshold": tau,
            "margin_threshold": delta,
            "overall_threshold": gamma,
            "legacy_recovered": recovered,
            "legacy_harmful": harmful,
            "legacy_net": recovered - harmful,
            "legacy_harmful_rate": harmful / total,
            "outcome_changing_precision": recovered / outcome_count if outcome_count else None,
            "switch_count": switches,
            "switch_coverage": switches / total,
            "corrected_recovered": corrected_recovered,
            "corrected_harmful": corrected_harmful,
            "corrected_net": corrected_recovered - corrected_harmful,
            "eligible_harmful_cap": harmful / total <= harmful_rate_limit,
        }
        sweep.append(record)
        if not record["eligible_harmful_cap"]:
            continue
        precision = record["outcome_changing_precision"]
        score = (
            float(record["legacy_net"]),
            float(precision if precision is not None else 0.0),
            -float(record["switch_coverage"]),
            float(record["corrected_net"]),
            -float(model_cost_usd),
        )
        eligible.append((score, record))
    if not eligible:
        selected = LockedSafeThreshold("no-beneficial-switch", None, None, None)
        return selected, sweep, {
            "status": selected.status,
            "reason": "no_threshold_satisfies_harmful_cap",
            "threshold": asdict(selected),
        }
    best = max(eligible, key=lambda item: item[0])[1]
    if int(best["legacy_net"]) <= 0:
        selected = LockedSafeThreshold("no-beneficial-switch", None, None, None)
        return selected, sweep, {
            "status": selected.status,
            "reason": "no_positive_legacy_net",
            "best_diagnostic": best,
            "threshold": asdict(selected),
        }
    selected = LockedSafeThreshold(
        "selected",
        float(best["confidence_threshold"]),
        float(best["margin_threshold"]),
        float(best["overall_threshold"]),
    )
    return selected, sweep, {
        "status": "selected",
        "metrics": best,
        "threshold": asdict(selected),
        "harmful_rate_limit": harmful_rate_limit,
    }


def apply_locked_threshold(example: dict[str, Any], threshold: LockedSafeThreshold) -> str:
    q_only = str(example["q_only_candidate_id"])
    if not threshold.enabled:
        return q_only
    accept = (
        bool(example["valid"])
        and not bool(example.get("abstain", False))
        and str(example["decision"]) == "switch"
        and str(example["selected_candidate_id"]) != q_only
        and float(example["confidence"]) >= float(threshold.confidence)
        and float(example["score_margin_top1_top2"]) >= float(threshold.margin)
        and float(example["selected_overall_score"]) >= float(threshold.overall)
    )
    return str(example["selected_candidate_id"]) if accept else q_only
