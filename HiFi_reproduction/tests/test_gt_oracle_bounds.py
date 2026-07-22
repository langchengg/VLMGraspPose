from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.oracle_bounds import (
    LABEL_TOLERANCES,
    build_oracle_analysis,
    compute_sample_oracles,
    deduplicate_union_pool,
    pose_rotation_geodesic_deg,
    pose_translation_distance_m,
    poses_match,
    summarize_oracle_bounds,
)


def _samples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "a", "scene_id": "s1", "dataset_index": 0},
            {"sample_id": "b", "scene_id": "s1", "dataset_index": 1},
            {"sample_id": "c", "scene_id": "s2", "dataset_index": 2},
            {"sample_id": "d", "scene_id": "s3", "dataset_index": 3},
        ]
    )


def _candidate(
    sample_id: str,
    index: int,
    quality: float,
    *,
    positive: bool,
    pred_filter: bool,
    baseline: bool = False,
    pool: str = "predicted_mask",
    xyz: tuple[float, float, float] = (0.0, 0.0, 1.0),
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    width: float = 0.05,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "pool_source": pool,
        "candidate_index_original": index,
        "vgn_quality": quality,
        "pred_filter_pass": pred_filter,
        "is_baseline_top1": baseline,
        "position_camera_x": xyz[0],
        "position_camera_y": xyz[1],
        "position_camera_z": xyz[2],
        "quaternion_camera_xyzw": list(quaternion),
        "width_m": width,
        "gt_target_positive_primary": positive,
    }
    for column in LABEL_TOLERANCES.values():
        row[column] = positive
    return row


def _candidate_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    predicted = pd.DataFrame(
        [
            # The current selection is target-inconsistent.  A lower-quality
            # positive remains after the predicted-mask filter.
            _candidate("a", 0, 0.99, positive=False, pred_filter=True, baseline=True),
            _candidate("a", 1, 0.95, positive=True, pred_filter=True),
            # A positive exists only before the predicted-mask filter.
            _candidate("b", 0, 0.98, positive=False, pred_filter=True, baseline=True),
            _candidate("b", 1, 0.94, positive=True, pred_filter=False),
            # Current selection is already target-consistent.
            _candidate("c", 0, 0.97, positive=True, pred_filter=True, baseline=True),
        ]
    )
    regenerated = pd.DataFrame(
        [
            _candidate(
                "a", 100, 0.93, positive=True, pred_filter=True, pool="gt_regenerated"
            ),
            _candidate(
                "b", 101, 0.92, positive=True, pred_filter=True, pool="gt_regenerated"
            ),
            _candidate(
                "c", 102, 0.91, positive=False, pred_filter=False, pool="gt_regenerated"
            ),
            _candidate(
                "d", 103, 0.90, positive=True, pred_filter=True, pool="gt_regenerated"
            ),
        ]
    )
    return predicted, regenerated


def test_same_pool_oracle_never_regenerates_candidates() -> None:
    predicted, regenerated = _candidate_tables()
    outcomes = compute_sample_oracles(_samples(), predicted, regenerated)
    predicted_ids = {
        sample_id: set(group["candidate_index_original"])
        for sample_id, group in predicted.groupby("sample_id")
    }
    for row in outcomes.itertuples(index=False):
        if row.same_pool_pre_filter_candidate_index is not None and not pd.isna(
            row.same_pool_pre_filter_candidate_index
        ):
            assert int(row.same_pool_pre_filter_candidate_index) in predicted_ids[row.sample_id]
        if row.same_pool_post_filter_candidate_index is not None and not pd.isna(
            row.same_pool_post_filter_candidate_index
        ):
            assert int(row.same_pool_post_filter_candidate_index) in predicted_ids[row.sample_id]
    indexed = outcomes.set_index("sample_id")
    assert not bool(indexed.loc["d", "oracle_same_pool_pre_filter"])
    assert bool(indexed.loc["d", "oracle_gt_regenerated_pool"])


def test_post_filter_oracle_is_not_above_pre_filter_oracle() -> None:
    predicted, regenerated = _candidate_tables()
    outcomes = compute_sample_oracles(_samples(), predicted, regenerated)
    assert (
        outcomes["oracle_same_pool_post_filter"]
        <= outcomes["oracle_same_pool_pre_filter"]
    ).all()
    row_b = outcomes.set_index("sample_id").loc["b"]
    assert bool(row_b.oracle_same_pool_pre_filter)
    assert not bool(row_b.oracle_same_pool_post_filter)


def test_baseline_not_above_oracle() -> None:
    predicted, regenerated = _candidate_tables()
    outcomes = compute_sample_oracles(_samples(), predicted, regenerated)
    assert (
        outcomes["baseline_target_consistent"]
        <= outcomes["oracle_same_pool_post_filter"]
    ).all()


def test_oracle_denominators() -> None:
    predicted, regenerated = _candidate_tables()
    outcomes = compute_sample_oracles(_samples(), predicted, regenerated)
    summary = summarize_oracle_bounds(outcomes, bootstrap_replicates=30, seed=42)
    denominators = (
        summary.groupby("denominator_scope")["denominator"].unique().map(list).to_dict()
    )
    assert denominators == {
        "all_samples": [4],
        "samples_with_baseline_selection": [3],
        "samples_with_official_candidates": [3],
    }
    main = summary.loc[
        (summary["metric"] == "same_pool_pre_filter_oracle")
        & (summary["denominator_scope"] == "all_samples")
    ].iloc[0]
    assert main.numerator == 3
    assert main.denominator == 4
    assert main.rate == pytest.approx(0.75)
    assert 0.0 <= main.wilson_ci_lower <= main.rate <= main.wilson_ci_upper <= 1.0


def test_pose_matching_translation() -> None:
    left = _candidate("a", 0, 0.9, positive=True, pred_filter=True)
    within = _candidate(
        "a", 1, 0.8, positive=True, pred_filter=True, xyz=(0.01, 0.0, 1.0)
    )
    outside = _candidate(
        "a", 2, 0.7, positive=True, pred_filter=True, xyz=(0.01001, 0.0, 1.0)
    )
    assert pose_translation_distance_m(left, within) == pytest.approx(0.01)
    assert poses_match(left, within)
    assert not poses_match(left, outside)


def test_pose_matching_rotation() -> None:
    left = _candidate("a", 0, 0.9, positive=True, pred_filter=True)
    angle = math.radians(15.0)
    within = _candidate(
        "a",
        1,
        0.8,
        positive=True,
        pred_filter=True,
        quaternion=(0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)),
    )
    outside_angle = math.radians(15.01)
    outside = _candidate(
        "a",
        2,
        0.7,
        positive=True,
        pred_filter=True,
        quaternion=(
            0.0,
            0.0,
            math.sin(outside_angle / 2.0),
            math.cos(outside_angle / 2.0),
        ),
    )
    assert pose_rotation_geodesic_deg(left, within) == pytest.approx(15.0)
    assert poses_match(left, within)
    assert not poses_match(left, outside)
    sign_flipped = dict(within)
    sign_flipped["quaternion_camera_xyzw"] = list(
        -np.asarray(within["quaternion_camera_xyzw"])
    )
    assert pose_rotation_geodesic_deg(within, sign_flipped) == pytest.approx(0.0)


def test_union_pool_deduplication() -> None:
    predicted = pd.DataFrame(
        [
            _candidate("a", 0, 0.95, positive=True, pred_filter=True),
            _candidate(
                "a", 1, 0.80, positive=False, pred_filter=False, xyz=(0.05, 0.0, 1.0)
            ),
        ]
    )
    regenerated = pd.DataFrame(
        [
            _candidate(
                "a",
                10,
                0.90,
                positive=True,
                pred_filter=True,
                pool="gt_regenerated",
                xyz=(0.005, 0.0, 1.0),
            )
        ]
    )
    union = deduplicate_union_pool(predicted, regenerated)
    assert len(union) == 2
    retained = union.loc[union["candidate_index_original"] == 0].iloc[0]
    assert retained.union_member_count == 2
    assert retained.union_member_sources == ["gt_regenerated", "predicted_mask"]
    assert retained.vgn_quality == pytest.approx(0.95)


def test_build_oracle_analysis_reports_all_tolerances() -> None:
    predicted, regenerated = _candidate_tables()
    analysis = build_oracle_analysis(
        _samples(),
        predicted,
        regenerated,
        include_union=False,
        bootstrap_replicates=10,
        seed=42,
    )
    assert set(analysis.oracle_sensitivity["label_tolerance"]) == set(LABEL_TOLERANCES)
    assert "union_diagnostic_ceiling" not in set(analysis.oracle_upper_bounds["metric"])
    assert len(analysis.samples) == 4


def test_scene_cluster_bootstrap_deterministic() -> None:
    predicted, regenerated = _candidate_tables()
    outcomes = compute_sample_oracles(_samples(), predicted, regenerated)
    first = summarize_oracle_bounds(outcomes, bootstrap_replicates=50, seed=7)
    second = summarize_oracle_bounds(outcomes, bootstrap_replicates=50, seed=7)
    pd.testing.assert_frame_equal(first, second)
