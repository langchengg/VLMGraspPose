from __future__ import annotations

from failure_analysis.gemini_crog_evidence_v1.stability import compute_stability


def test_stability_metrics_cover_exact_and_rank_agreement():
    rows = []
    for replicate in (1, 2, 3):
        rows.append(
            {
                "model_id": "flash",
                "sample_id": "sample",
                "replicate_id": replicate,
                "selected_candidate_id": "A",
                "ranking": ["A", "B", "C", "D", "E"],
                "confidence": 0.8,
                "score_margin_top1_top2": 0.2,
                "decision": "keep_original",
                "valid": True,
            }
        )
    per_sample, summary = compute_stability(rows)
    assert per_sample[0]["selected_candidate_exact_agreement"]
    assert per_sample[0]["complete_ranking_exact_agreement"]
    assert per_sample[0]["kendall_tau_mean"] == 1.0
    assert summary["flash"]["decision_consistency_rate"] == 1.0


def test_stability_rejects_missing_replicate():
    rows = [
        {
            "model_id": "flash",
            "sample_id": "sample",
            "replicate_id": replicate,
            "selected_candidate_id": "A",
            "ranking": ["A", "B", "C", "D", "E"],
            "confidence": 0.8,
            "score_margin_top1_top2": 0.2,
            "decision": "switch",
        }
        for replicate in (1, 3)
    ]
    try:
        compute_stability(rows)
    except ValueError as exc:
        assert "replicate_id 1,2,3" in str(exc)
    else:
        raise AssertionError("missing replicate was accepted")
