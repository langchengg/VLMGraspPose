from __future__ import annotations

import numpy as np
import pandas as pd

from unified_reranking.case_analysis import (
    GALLERY_QUOTAS,
    deterministic_case_selection,
    pair_contribution_analysis,
)
from unified_reranking.case_visuals import rectangle_points


def _samples() -> pd.DataFrame:
    rows = []
    categories = list(GALLERY_QUOTAS)
    for route in ("crog", "g1", "c1"):
        for index in range(18):
            rows.append(
                {
                    "route": route,
                    "sample_id": f"{route}-{index}",
                    "scene_id": f"scene-{index // 3}",
                    "frame_id": f"frame-{index // 2}",
                    "analysis_category": categories[index % len(categories)],
                    "mechanism": "M2 soft target support",
                }
            )
    return pd.DataFrame(rows)


def test_deterministic_case_selection_is_order_invariant() -> None:
    samples = _samples()
    first = deterministic_case_selection(samples)
    second = deterministic_case_selection(samples.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )
    assert not first.duplicated(["route", "category", "sample_id"]).any()


def test_pair_contributions_are_challenger_minus_native_and_family_additive() -> None:
    samples = pd.DataFrame(
        [
            {
                "route": "crog",
                "sample_id": "s",
                "outcome": "recovered",
                "native_candidate_id": "a",
                "ungated_candidate_id": "b",
                "gated_candidate_id": "b",
                "score_margin": 1.0,
                "candidate_count_top5": 2,
            }
        ]
    )
    contributions = pd.DataFrame(
        [
            {
                "route": "crog",
                "sample_id": "s",
                "candidate_id": "a",
                "expected_value": 0.5,
                "contrib::base_logit": 1.0,
                "contrib::p_center": 0.2,
            },
            {
                "route": "crog",
                "sample_id": "s",
                "candidate_id": "b",
                "expected_value": 0.5,
                "contrib::base_logit": 1.4,
                "contrib::p_center": 0.8,
            },
        ]
    )
    pairs, long = pair_contribution_analysis(samples, contributions)
    assert np.isclose(pairs.loc[0, "family_delta::native_calibration"], 0.4)
    assert np.isclose(pairs.loc[0, "family_delta::soft_target_support"], 0.6)
    assert pairs.loc[0, "expected_value_delta"] == 0
    assert np.isclose(long["challenger_minus_native"].sum(), 1.0)


def test_rectangle_geometry_uses_physical_negative_theta() -> None:
    points = rectangle_points(
        {"cx_px": 100, "cy_px": 80, "width_px": 40, "height_px": 20, "theta_deg": 30}
    )
    assert points.shape == (4, 2)
    assert np.allclose(points.mean(axis=0), [100, 80])
    edges = np.roll(points, -1, axis=0) - points
    lengths = sorted(np.linalg.norm(edges, axis=1))
    assert np.allclose(lengths, [20, 20, 40, 40], atol=1e-4)


def test_no_output_pair_is_explicitly_not_applicable() -> None:
    samples = pd.DataFrame(
        [
            {
                "route": "g1",
                "sample_id": "empty",
                "outcome": "wrong_retained",
                "native_candidate_id": None,
                "ungated_candidate_id": None,
                "gated_candidate_id": None,
                "score_margin": np.nan,
                "candidate_count_top5": 0,
            }
        ]
    )
    pairs, long = pair_contribution_analysis(
        samples,
        pd.DataFrame(
            columns=[
                "route",
                "sample_id",
                "candidate_id",
                "expected_value",
                "contrib::base_logit",
            ]
        ),
    )
    assert pairs.loc[0, "mechanism"] == "not_applicable_no_output"
    assert long.empty
