from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.reranking_opportunity import (
    OPPORTUNITY_CLASSES,
    build_opportunity_table,
    first_positive_recall,
    opportunity_counts,
    quality_discrimination_metrics,
    sample_cluster_bootstrap_quality_auc,
)


def _candidate(
    sample_id: str,
    index: int,
    quality: float,
    *,
    filtered: bool,
    positive: bool,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "candidate_index_original": index,
        "vgn_quality": quality,
        "pred_filter_pass": filtered,
        "gt_target_positive_primary": positive,
        "rank_vgn_all": 0,  # overwritten below by the fixture builder
    }


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.DataFrame(
        [
            {"sample_id": "technical", "pred_status": "support_plane_failed"},
            {"sample_id": "none", "pred_status": "no_official_grasp"},
            {"sample_id": "generation", "pred_status": "no_target_grasp"},
            {"sample_id": "filter", "pred_status": "no_target_grasp"},
            {"sample_id": "ranking", "pred_status": "ok"},
            {"sample_id": "correct", "pred_status": "ok"},
        ]
    )
    candidates = pd.DataFrame(
        [
            _candidate("generation", 0, 0.99, filtered=True, positive=False),
            _candidate("filter", 0, 0.99, filtered=True, positive=False),
            _candidate("filter", 1, 0.95, filtered=False, positive=True),
            _candidate("ranking", 0, 0.99, filtered=True, positive=False),
            _candidate("ranking", 1, 0.98, filtered=True, positive=True),
            _candidate("correct", 0, 0.99, filtered=True, positive=True),
            _candidate("correct", 1, 0.90, filtered=True, positive=False),
        ]
    )
    candidates["rank_vgn_all"] = candidates.groupby("sample_id").cumcount() + 1
    return samples, candidates


def test_opportunity_classes_are_mutually_exclusive() -> None:
    samples, candidates = _fixture()
    result = build_opportunity_table(samples, candidates)
    assert result["opportunity_class"].tolist() == list(OPPORTUNITY_CLASSES)
    assert result["sample_id"].nunique() == len(result)


def test_opportunity_classes_cover_all_samples() -> None:
    samples, candidates = _fixture()
    result = build_opportunity_table(samples, candidates)
    counts = opportunity_counts(result)
    assert int(counts["numerator"].sum()) == len(samples)
    assert set(counts["opportunity_class"]) == set(OPPORTUNITY_CLASSES)


def test_post_filter_recoverable_definition() -> None:
    samples, candidates = _fixture()
    rows = build_opportunity_table(samples, candidates).set_index("sample_id")
    assert bool(rows.loc["ranking", "has_gt_positive_after_pred_filter"])
    assert not bool(rows.loc["ranking", "hard_filter_top1_is_gt_positive"])
    assert bool(rows.loc["ranking", "post_filter_recoverable"])
    assert rows.loc["ranking", "rank_first_gt_positive"] == 2
    assert np.isclose(rows.loc["ranking", "quality_gap"], 0.01)


def test_filter_recoverable_definition() -> None:
    samples, candidates = _fixture()
    rows = build_opportunity_table(samples, candidates).set_index("sample_id")
    assert bool(rows.loc["filter", "has_gt_positive_anywhere"])
    assert not bool(rows.loc["filter", "has_gt_positive_after_pred_filter"])
    assert bool(rows.loc["filter", "filter_recoverable"])
    assert bool(rows.loc["filter", "pre_filter_recoverable"])
    assert not bool(rows.loc["filter", "post_filter_recoverable"])


def test_generation_limited_definition() -> None:
    samples, candidates = _fixture()
    rows = build_opportunity_table(samples, candidates).set_index("sample_id")
    assert bool(rows.loc["generation", "generation_limited"])
    assert bool(rows.loc["none", "generation_limited"])
    assert rows.loc["none", "opportunity_class"] == "no_official_candidate"


def test_first_positive_recall_and_quality_metrics() -> None:
    samples, candidates = _fixture()
    rows = build_opportunity_table(samples, candidates)
    recall = first_positive_recall(rows, ks=(1, 2)).set_index("metric")
    assert recall.loc["Recall@1", "numerator"] == 1
    assert recall.loc["Recall@2", "numerator"] == 3
    assert recall.loc["Recall@Any", "numerator"] == 3

    metrics = quality_discrimination_metrics(candidates)
    assert metrics["positive_candidate_count"] == 3
    assert metrics["negative_candidate_count"] == 4
    assert 0.0 <= float(metrics["roc_auc"]) <= 1.0
    assert 0.0 <= float(metrics["pr_auc"]) <= 1.0


def test_rank_vgn_all_is_stable() -> None:
    samples = pd.DataFrame([{"sample_id": "tie", "pred_status": "ok"}])
    candidates = pd.DataFrame(
        [
            _candidate("tie", 8, 0.95, filtered=True, positive=True),
            _candidate("tie", 3, 0.95, filtered=True, positive=False),
        ]
    )
    # Deliberately provide rows in the reverse of the specified tie-break.
    candidates["rank_vgn_all"] = [2, 1]
    row = build_opportunity_table(samples, candidates).iloc[0]
    assert row["baseline_vgn_all_candidate_index"] == 3
    assert row["baseline_hard_filter_candidate_index"] == 3
    assert row["rank_first_gt_positive"] == 2
    assert not bool(row["hard_filter_top1_is_gt_positive"])


def test_candidate_auc_sample_cluster_bootstrap_is_deterministic() -> None:
    _, candidates = _fixture()
    first = sample_cluster_bootstrap_quality_auc(candidates, replicates=50, seed=7)
    second = sample_cluster_bootstrap_quality_auc(candidates, replicates=50, seed=7)
    assert first == second
    assert first["cluster_unit"] == "sample_id"
    assert len(first["roc_auc_ci_95"]) == 2
    assert len(first["pr_auc_ci_95"]) == 2
