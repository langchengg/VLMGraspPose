"""Validation-locked conservative switch gate with HiFi as the default mask."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .proposal_statistics import paired_transitions
from .selective_sam3_vg.metrics import summarize_ious


@dataclass(frozen=True)
class GateThresholds:
    minimum_p90_margin: float = 0.0
    minimum_iou_margin: float = 0.0
    minimum_source_consensus: float = 1.0
    minimum_probability_mass_precision: float = 0.0
    minimum_depth_reliability: float = 0.0
    maximum_low_probability_expansion: float = 1.0
    maximum_connected_components: float = 1.0e9
    minimum_selector_stability: float = 0.0
    minimum_relation_consistency: float = 0.0
    require_reliable_features: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_gate_evidence(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach margins and model-agreement evidence without using ground truth."""

    frame = predictions.copy()
    required = {
        "sample_id",
        "candidate_id",
        "source_family",
        "m0_p90_calibrated",
        "m1_p90_calibrated",
        "m2_predicted_iou",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"gate predictions missing {sorted(required - set(frame.columns))}")
    hifi = (
        frame[frame["source_family"].isin({"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"})]
        .sort_values(["sample_id", "candidate_id"], kind="stable")
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )
    if len(hifi) != frame["sample_id"].nunique():
        raise ValueError("every sample requires a HiFi fallback candidate")
    frame["hifi_p90_prediction"] = frame["sample_id"].map(hifi["m1_p90_calibrated"])
    frame["hifi_iou_prediction"] = frame["sample_id"].map(hifi["m2_predicted_iou"])
    frame["predicted_p90_margin_over_hifi"] = (
        frame["m1_p90_calibrated"] - frame["hifi_p90_prediction"]
    )
    frame["predicted_iou_margin_over_hifi"] = (
        frame["m2_predicted_iou"] - frame["hifi_iou_prediction"]
    )
    agreement = frame[
        ["m0_p90_calibrated", "m1_p90_calibrated", "m2_predicted_iou"]
    ].to_numpy(dtype=np.float64)
    frame["selector_stability"] = np.clip(1.0 - np.std(agreement, axis=1), 0.0, 1.0)
    return frame


def gate_acceptance(frame: pd.DataFrame, thresholds: GateThresholds) -> np.ndarray:
    """Return GT-free switch decisions for proposed non-HiFi candidates."""

    def values(name: str, default: float) -> np.ndarray:
        return pd.to_numeric(
            frame.get(name, pd.Series(default, index=frame.index)), errors="coerce"
        ).fillna(default).to_numpy(dtype=np.float64)

    accept = ~frame["source_family"].isin(
        {"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"}
    ).to_numpy()
    accept &= values("predicted_p90_margin_over_hifi", -np.inf) >= thresholds.minimum_p90_margin
    accept &= values("predicted_iou_margin_over_hifi", -np.inf) >= thresholds.minimum_iou_margin
    accept &= values("source_consensus_count", 0.0) >= thresholds.minimum_source_consensus
    accept &= values("hifi_probability_mass_precision", 0.0) >= thresholds.minimum_probability_mass_precision
    accept &= values("depth_reliability", 0.0) >= thresholds.minimum_depth_reliability
    accept &= values("low_probability_expansion_fraction", 1.0) <= thresholds.maximum_low_probability_expansion
    accept &= values("connected_component_count", np.inf) <= thresholds.maximum_connected_components
    accept &= values("selector_stability", 0.0) >= thresholds.minimum_selector_stability
    has_relation = frame.get("has_relation", pd.Series(False, index=frame.index)).fillna(False).to_numpy(bool)
    relation = values("relation_max_consistency", -np.inf)
    accept &= ~has_relation | (relation >= thresholds.minimum_relation_consistency)
    if thresholds.require_reliable_features:
        appearance = frame.get(
            "appearance_features_valid", pd.Series(False, index=frame.index)
        ).fillna(False).to_numpy(bool)
        depth = frame.get("depth_features_valid", pd.Series(False, index=frame.index)).fillna(False).to_numpy(bool)
        accept &= appearance & depth
        relation_valid = frame.get(
            "relation_features_valid", pd.Series(False, index=frame.index)
        ).fillna(False).to_numpy(bool)
        accept &= ~has_relation | relation_valid
    return accept


def proposed_alternatives(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    """Select exactly one pre-gate alternative per sample with stable tie breaks."""

    score_columns = {
        "F1_hgb_classifier": ["m1_p90_calibrated"],
        "F2_classifier_regressor": ["m1_p90_calibrated", "m2_predicted_iou"],
        "F0_deterministic": ["deterministic_rule_score", "hifi_candidate_iou"],
    }
    if method not in score_columns:
        raise ValueError(f"unknown final selector {method}")
    data = frame[frame["eligible_final"]].copy()
    columns = ["sample_id", *score_columns[method], "candidate_id"]
    ascending = [True, *([False] * len(score_columns[method])), True]
    return (
        data.sort_values(columns, ascending=ascending, kind="stable")
        .drop_duplicates("sample_id", keep="first")
        .sort_values("sample_id")
    )


def apply_gate(
    frame: pd.DataFrame,
    proposed: pd.DataFrame,
    thresholds: GateThresholds,
) -> pd.DataFrame:
    """Choose proposed candidates only when all locked switch conditions pass."""

    enriched = (
        frame.copy()
        if {
            "predicted_p90_margin_over_hifi",
            "predicted_iou_margin_over_hifi",
            "selector_stability",
        }.issubset(frame.columns)
        else add_gate_evidence(frame)
    )
    proposal = proposed[["sample_id", "candidate_id"]].merge(
        enriched, on=["sample_id", "candidate_id"], how="left", validate="one_to_one"
    )
    proposal["gate_accept"] = gate_acceptance(proposal, thresholds)
    hifi = (
        enriched[enriched["source_family"].isin({"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"})]
        .sort_values(["sample_id", "candidate_id"], kind="stable")
        .drop_duplicates("sample_id")
    )
    if hifi["sample_id"].nunique() != proposal["sample_id"].nunique():
        raise ValueError("every proposed sample requires a HiFi fallback candidate")
    accepted = proposal[proposal["gate_accept"]].copy()
    accepted["gate_reason"] = "ALL_LOCKED_CONDITIONS_PASS"
    rejected_ids = proposal.loc[~proposal["gate_accept"], ["sample_id"]]
    rejected = rejected_ids.merge(
        hifi,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    rejected["gate_accept"] = False
    rejected["gate_reason"] = "KEEP_HIFI_CONSERVATIVE_GATE"
    return (
        pd.concat([accepted, rejected], ignore_index=True, sort=False)
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )


def _gate_objective(
    chosen: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    noninferiority_margin: float,
) -> tuple[tuple[float, ...], dict[str, Any]] | None:
    chosen_iou = chosen["candidate_iou"].to_numpy(dtype=np.float64)
    baseline_iou = baseline["candidate_iou"].to_numpy(dtype=np.float64)
    metrics = summarize_ious(chosen_iou)
    base_metrics = summarize_ious(baseline_iou)
    if (
        metrics["p_at_50"] < base_metrics["p_at_50"] - noninferiority_margin
        or metrics["p_at_60"] < base_metrics["p_at_60"] - noninferiority_margin
    ):
        return None
    p90 = paired_transitions(baseline_iou, chosen_iou, 0.90)
    harmful_baseline_successes = p90["harmed"]
    accepted = int(chosen.get("gate_accept", pd.Series(False, index=chosen.index)).sum())
    boundary_value = (
        float(chosen["validation_boundary_fscore"].mean())
        if "validation_boundary_fscore" in chosen
        else float("nan")
    )
    mean_boundary_fscore = boundary_value if np.isfinite(boundary_value) else None
    objective = (
        metrics["p_at_90"],
        -float(harmful_baseline_successes),
        metrics["mean_iou"],
        mean_boundary_fscore if mean_boundary_fscore is not None else -1.0,
        metrics["p_at_80"],
        -float(accepted),
    )
    return objective, {
        "metrics": metrics,
        "mean_validation_boundary_fscore": mean_boundary_fscore,
        "transitions_p90": p90,
        "accepted": accepted,
    }


def tune_gate(
    frame: pd.DataFrame,
    proposed: pd.DataFrame,
    *,
    grid: dict[str, Iterable[float]],
    noninferiority_margin: float = 0.001,
) -> tuple[GateThresholds, pd.DataFrame, dict[str, Any]]:
    """Exhaustively tune a small predefined validation grid, with HiFi fallback."""

    enriched = add_gate_evidence(frame)
    baseline = (
        enriched[enriched["source_family"].isin({"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"})]
        .sort_values(["sample_id", "candidate_id"], kind="stable")
        .drop_duplicates("sample_id")
        .sort_values("sample_id")
    )
    names = list(grid)
    best: tuple[tuple[float, ...], GateThresholds, pd.DataFrame, dict[str, Any]] | None = None
    candidates_evaluated = 0
    for combination in product(*(list(grid[name]) for name in names)):
        thresholds = GateThresholds(**dict(zip(names, combination, strict=True)))
        chosen = apply_gate(enriched, proposed, thresholds)
        evaluated = _gate_objective(
            chosen, baseline, noninferiority_margin=noninferiority_margin
        )
        candidates_evaluated += 1
        if evaluated is None:
            continue
        objective, details = evaluated
        if best is None or objective > best[0]:
            best = (objective, thresholds, chosen, details)
    if best is None:
        fallback = baseline.copy()
        fallback["gate_accept"] = False
        fallback["gate_reason"] = "KEEP_HIFI_NO_NONINFERIOR_GATE"
        thresholds = GateThresholds(minimum_p90_margin=2.0)
        details = {
            "metrics": summarize_ious(fallback["candidate_iou"]),
            "selected_fallback_only": True,
        }
        return thresholds, fallback, details
    _, thresholds, chosen, details = best
    details = {
        **details,
        "grid_candidates_evaluated": candidates_evaluated,
        "selected_fallback_only": False,
    }
    return thresholds, chosen, details


__all__ = [
    "GateThresholds",
    "add_gate_evidence",
    "apply_gate",
    "gate_acceptance",
    "proposed_alternatives",
    "tune_gate",
]
