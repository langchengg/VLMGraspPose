from __future__ import annotations

import pandas as pd

from unified_reranking.attribution_bridge import (
    candidate_membership_overlap,
    evaluate_attribution_bridge,
)


def _fair() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "a", "b"],
            "candidate_id": ["fa1", "fa2", "fb1"],
            "native_rank": [1, 2, 1],
            "native_score": [0.9, 0.8, 0.7],
            "p_center": [0.1, 1.0, 1.0],
            "candidate_success": [0, 1, 1],
            "cx_px": [1.0, 2.0, 3.0],
            "cy_px": [1.0, 2.0, 3.0],
            "theta_deg": [0.0, 10.0, 20.0],
            "width_px": [4.0, 5.0, 6.0],
            "height_px": [2.0, 2.5, 3.0],
        }
    )


def _historical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "candidate_id": ["ha", "hb"],
            "native_rank": [1, 1],
            "raw_network_quality": [0.9, 0.7],
            "original_score": [0.8, 0.6],
            "candidate_success": [0, 1],
            "cx_px": [1.0, 3.0],
            "cy_px": [1.0, 3.0],
            "theta_deg": [180.0, 20.0],
            "width_px": [4.0, 6.0],
            "height_px": [20.0, 3.0],
        }
    )


def test_bridge_has_exact_four_cells_and_common_denominator() -> None:
    table, decisions = evaluate_attribution_bridge(["a", "b", "c"], _fair(), _historical())
    assert len(table) == 4
    assert set(table["sample_count"]) == {3}
    fair_native = table.loc[
        (table["candidate_pool"] == "fair_gaussian")
        & (table["selector"] == "fair_native_selector")
    ].iloc[0]
    fair_historical = table.loc[
        (table["candidate_pool"] == "fair_gaussian")
        & (table["selector"] == "historical_selector")
    ].iloc[0]
    assert fair_native["j_at_1_numerator"] == 1
    assert fair_historical["j_at_1_numerator"] == 2
    assert len(decisions) == 12


def test_membership_overlap_distinguishes_pose_from_height_contract() -> None:
    overlap = candidate_membership_overlap(_fair(), _historical())
    assert overlap["pose"]["intersection"] == 2
    assert overlap["full_geometry"]["intersection"] == 1

