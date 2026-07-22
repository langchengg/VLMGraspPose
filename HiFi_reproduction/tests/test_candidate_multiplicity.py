from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.multiplicity import (
    build_multiplicity_table,
    candidate_count_bin,
    cluster_pose_modes,
    rankability_table,
    rotation_geodesic_degrees,
)


def _rotation_z(degrees: float) -> list[list[float]]:
    radians = np.radians(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _candidate(
    sample_id: str,
    index: int,
    x: float,
    *,
    rotation_degrees: float = 0.0,
    width: float = 0.04,
    filtered: bool = False,
    positive: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "candidate_index_original": index,
        "position_camera_x": x,
        "position_camera_y": 0.0,
        "position_camera_z": 1.0,
        "rotation_camera_3x3": _rotation_z(rotation_degrees),
        "width_m": width,
        "pred_filter_pass": filtered,
        "gt_target_positive_primary": positive,
        "gt_inside_raw_mask": positive,
        "gt_inside_dilated_mask_5px": positive,
        "gt_inside_dilated_mask_10px": positive,
    }


def test_zero_one_multiple_candidate_bins() -> None:
    assert [candidate_count_bin(value) for value in (0, 1, 2, 3, 4)] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    assert candidate_count_bin(5) == "5–9"
    assert candidate_count_bin(19) == "10–19"
    assert candidate_count_bin(49) == "20–49"
    assert candidate_count_bin(50) == "50+"


def test_pose_mode_clustering() -> None:
    # 0--1 and 1--2 are connected; union-find must make the chain transitive
    # even though 0--2 exceeds the translation threshold.
    candidates = pd.DataFrame(
        [
            _candidate("a", 2, 0.018),
            _candidate("a", 0, 0.000),
            _candidate("a", 1, 0.009),
            _candidate("a", 3, 0.100),
        ]
    )
    labels = cluster_pose_modes(candidates)
    by_index = dict(zip(candidates["candidate_index_original"], labels, strict=True))
    assert by_index[0] == by_index[1] == by_index[2]
    assert by_index[3] != by_index[0]
    assert labels.nunique() == 2

    # A permutation cannot change candidate-index-to-mode assignments.
    shuffled = candidates.sample(frac=1.0, random_state=42)
    shuffled_labels = cluster_pose_modes(shuffled)
    assert dict(zip(shuffled["candidate_index_original"], shuffled_labels, strict=True)) == by_index


def test_pose_mode_clustering_checks_rotation_and_width() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("a", 0, 0.0),
            _candidate("a", 1, 0.001, rotation_degrees=16.0),
            _candidate("a", 2, 0.001, width=0.051),
        ]
    )
    labels = cluster_pose_modes(candidates)
    assert labels.nunique() == 3
    assert np.isclose(
        rotation_geodesic_degrees(np.eye(3), np.asarray(_rotation_z(16.0))), 16.0
    )


def test_rankability_definitions() -> None:
    samples = pd.DataFrame(
        [
            {"sample_id": "zero"},
            {"sample_id": "one"},
            {"sample_id": "two_filtered"},
            {"sample_id": "two_one_filtered"},
        ]
    )
    candidates = pd.DataFrame(
        [
            _candidate("one", 0, 0.0, filtered=True, positive=True),
            _candidate("two_filtered", 0, 0.0, filtered=True, positive=False),
            _candidate("two_filtered", 1, 0.1, filtered=True, positive=True),
            _candidate("two_one_filtered", 0, 0.0, filtered=True, positive=False),
            _candidate("two_one_filtered", 1, 0.1, filtered=False, positive=True),
        ]
    )
    table = build_multiplicity_table(samples, candidates, include_pose_sensitivity=False)
    rows = table.set_index("sample_id")
    assert not bool(rows.loc["zero", "pre_filter_rankable"])
    assert not bool(rows.loc["one", "pre_filter_rankable"])
    assert bool(rows.loc["two_filtered", "pre_filter_rankable"])
    assert bool(rows.loc["two_filtered", "post_filter_rankable"])
    assert bool(rows.loc["two_filtered", "gt_rankable"])
    assert bool(rows.loc["two_one_filtered", "pre_filter_rankable"])
    assert not bool(rows.loc["two_one_filtered", "post_filter_rankable"])
    assert rows.loc["zero", "gt_positive_multiplicity_class"] == "zero_candidate"
    assert rows.loc["one", "gt_positive_multiplicity_class"] == "single_gt_positive"
    assert rows.loc["two_filtered", "gt_positive_multiplicity_class"] == "multiple_with_one_gt_positive"

    rankability = rankability_table(table).set_index("metric")
    assert rankability.loc["pre_filter_rankable_all", "numerator"] == 2
    assert rankability.loc["post_filter_rankable_all", "numerator"] == 1
    assert rankability.loc[
        "post_filter_rankable_given_pred_filtered", "denominator"
    ] == 3
