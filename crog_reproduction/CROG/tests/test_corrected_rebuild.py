from pathlib import Path

import numpy as np

from failure_analysis.rebuild_corrected_evaluation import (
    DEFAULT_EXPRESSIONS,
    EXPECTED_SAMPLE_COUNT,
    _bootstrap_delta,
    _holm_adjust,
    load_expressions,
)
from failure_analysis.reranking.labels import build_label_record
from utils.grasp_metrics import CORRECTED_EVALUATOR_VERSION


def test_raw_expressions_rebuild_complete_refer_partition():
    records, counts = load_expressions(DEFAULT_EXPRESSIONS)
    assert len(records) == EXPECTED_SAMPLE_COUNT
    assert counts == {
        "name": 5809,
        "location": 2672,
        "attribute": 781,
        "pure_relation": 5769,
        "mixed_relation": 2718,
    }


def test_holm_adjustment_is_monotonic_in_sorted_pvalues():
    adjusted = _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def test_paired_cluster_bootstrap_is_deterministic_and_zero_for_identical_outcomes():
    outcomes = np.asarray([True, False, True, False])
    clusters = ["frame_a", "frame_a", "frame_b", "frame_b"]
    first = _bootstrap_delta(outcomes, outcomes, clusters, seed=17, iterations=100)
    second = _bootstrap_delta(outcomes, outcomes, clusters, seed=17, iterations=100)
    assert first == second
    assert first["low_pp"] == 0.0
    assert first["high_pp"] == 0.0
    assert first["cluster_count"] == 2


def test_reranking_label_record_carries_version_and_pairwise_matrix():
    candidate = {
        "candidate_id": "candidate_0",
        "candidate_checksum": "checksum",
        "cx": 550.0,
        "cy": 240.0,
        "width_px": 80.0,
        "height_px": 20.0,
        "angle_deg": 0.0,
    }
    record = build_label_record(
        {"sample_id": 1, "candidates": [candidate]},
        [[550.0, 240.0, 80.0, 20.0, 0.0, 1.0]],
    )
    assert record["evaluator_version"] == CORRECTED_EVALUATOR_VERSION
    assert record["candidate_labels"][0]["candidate_valid"]
    assert len(record["candidate_labels"][0]["pairwise"]) == 1
