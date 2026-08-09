from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .schema import PairwiseDecision


class PairLabel(str, Enum):
    BENEFICIAL = "BENEFICIAL"
    HARMFUL = "HARMFUL"
    NEUTRAL_BOTH_CORRECT = "NEUTRAL_BOTH_CORRECT"
    NEUTRAL_BOTH_WRONG = "NEUTRAL_BOTH_WRONG"


def pair_label(*, baseline_correct: bool, challenger_correct: bool) -> PairLabel:
    if not baseline_correct and challenger_correct:
        return PairLabel.BENEFICIAL
    if baseline_correct and not challenger_correct:
        return PairLabel.HARMFUL
    if baseline_correct:
        return PairLabel.NEUTRAL_BOTH_CORRECT
    return PairLabel.NEUTRAL_BOTH_WRONG


@dataclass(frozen=True)
class GateThresholds:
    tau: float
    eta: float
    reliability_min: float = 0.5
    require_confirmation: bool = True

    def __post_init__(self) -> None:
        for name in ("tau", "eta", "reliability_min"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class PairGateEvidence:
    challenger_id: str
    p_benefit: float
    p_harm: float
    reliability: float
    challenger_hard_valid: bool
    critic_decision: PairwiseDecision | None
    confirmation_decision: PairwiseDecision | None = None
    confirmation_valid: bool = False
    hard_veto: bool = False
    terminal_failure: bool = False


@dataclass(frozen=True)
class ProtectedDecision:
    baseline_id: str
    selected_id: str
    switched: bool
    reason: str
    challenger_id: str | None = None
    p_benefit: float | None = None
    p_harm: float | None = None


def protected_select(
    baseline_id: str,
    evidence: Iterable[PairGateEvidence],
    thresholds: GateThresholds,
    *,
    query_gate_passed: bool,
) -> ProtectedDecision:
    """Select at most one frozen challenger; every uncertainty path keeps c0."""

    if not query_gate_passed:
        return ProtectedDecision(baseline_id, baseline_id, False, "QUERY_GATE_KEEP")
    rows = list(evidence)
    if not rows:
        return ProtectedDecision(baseline_id, baseline_id, False, "NO_EVIDENCE")
    eligible = []
    for row in rows:
        if row.terminal_failure or row.hard_veto or not row.challenger_hard_valid:
            continue
        if row.critic_decision is not PairwiseDecision.PREFER_CHALLENGER:
            continue
        if float(row.reliability) < thresholds.reliability_min:
            continue
        if float(row.p_benefit) < thresholds.tau or float(row.p_harm) > thresholds.eta:
            continue
        if thresholds.require_confirmation and not (
            row.confirmation_valid
            and row.confirmation_decision is PairwiseDecision.PREFER_CHALLENGER
        ):
            continue
        eligible.append(row)
    if not eligible:
        return ProtectedDecision(baseline_id, baseline_id, False, "SAFE_GATE_KEEP")
    best = sorted(
        eligible,
        key=lambda row: (-float(row.p_benefit), float(row.p_harm), row.challenger_id),
    )[0]
    if best.challenger_id == baseline_id:
        return ProtectedDecision(baseline_id, baseline_id, False, "IDENTITY_KEEP")
    return ProtectedDecision(
        baseline_id,
        best.challenger_id,
        True,
        "SAFE_GATE_SWITCH",
        challenger_id=best.challenger_id,
        p_benefit=float(best.p_benefit),
        p_harm=float(best.p_harm),
    )


def full_denominator_metrics(
    rows: Iterable[Mapping[str, object]], *, baseline_key: str, selected_key: str
) -> dict[str, float | int]:
    records = list(rows)
    recovered = sum(not bool(row[baseline_key]) and bool(row[selected_key]) for row in records)
    harmful = sum(bool(row[baseline_key]) and not bool(row[selected_key]) for row in records)
    baseline = sum(bool(row[baseline_key]) for row in records)
    final = sum(bool(row[selected_key]) for row in records)
    switches = sum(bool(row.get("switched", False)) for row in records)
    denominator = len(records)
    return {
        "total": denominator,
        "baseline_successes": baseline,
        "final_successes": final,
        "baseline_j1": baseline / denominator if denominator else 0.0,
        "final_j1": final / denominator if denominator else 0.0,
        "recovered": recovered,
        "harmful": harmful,
        "net": recovered - harmful,
        "switches": switches,
        "switch_rate": switches / denominator if denominator else 0.0,
        "outcome_changing_precision": recovered / (recovered + harmful)
        if recovered + harmful
        else 0.0,
        "harm_rate": harmful / baseline if baseline else 0.0,
    }

