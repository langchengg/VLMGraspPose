from __future__ import annotations

from failure_analysis.gemini_crog_evidence_v1.calibration import (
    LockedSafeThreshold,
    apply_locked_threshold,
    calibration_grid_payload,
    sweep_safe_thresholds,
)


def _row(*, q=False, selected=True, confidence=0.9, margin=0.4, overall=0.9):
    return {
        "valid": True,
        "abstain": False,
        "decision": "switch",
        "selected_candidate_id": "candidate_1",
        "q_only_candidate_id": "candidate_0",
        "confidence": confidence,
        "score_margin_top1_top2": margin,
        "selected_overall_score": overall,
        "legacy_q_only_correct": q,
        "legacy_selected_correct": selected,
        "corrected_q_only_correct": q,
        "corrected_selected_correct": selected,
    }


def test_registered_grid_has_1100_combinations():
    assert calibration_grid_payload()["combination_count"] == 1100


def test_sweep_selects_positive_safe_switch():
    threshold, sweep, summary = sweep_safe_thresholds(
        [_row() for _ in range(99)] + [_row(q=True, selected=True)],
        model_id="model",
    )
    assert len(sweep) == 1100
    assert threshold.enabled
    assert summary["metrics"]["legacy_net"] == 99


def test_no_beneficial_switch_is_explicitly_disabled():
    threshold, _, summary = sweep_safe_thresholds(
        [_row(q=True, selected=False) for _ in range(100)],
        model_id="model",
    )
    assert threshold.status == "no-beneficial-switch"
    assert threshold.confidence is None
    assert apply_locked_threshold(_row(), threshold) == "candidate_0"
    assert summary["status"] == "no-beneficial-switch"


def test_apply_threshold_never_switches_below_any_gate():
    threshold = LockedSafeThreshold("selected", 0.8, 0.2, 0.8)
    assert apply_locked_threshold(_row(confidence=0.7), threshold) == "candidate_0"
    assert apply_locked_threshold(_row(), threshold) == "candidate_1"
