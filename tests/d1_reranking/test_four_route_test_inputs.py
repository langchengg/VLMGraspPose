from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from d1_reranking.four_route import FOUR_ROUTES
from d1_reranking.four_route_test_inputs import (
    build_router_test_frame,
    build_union_test_frame,
)
from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS


SAMPLES = pd.Series(["s0", "s1"])


def _candidates(route: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": SAMPLES,
            "candidate_id": [f"{route.lower()}0", f"{route.lower()}1"],
            "native_rank": [1, 1],
            "native_score": [0.9, 0.8],
            "candidate_identity_sha256": [f"identity-{route}-0", f"identity-{route}-1"],
            "candidate_geometry_sha256": [f"geometry-{route}-0", f"geometry-{route}-1"],
            "cx_px": [10.0, 20.0],
            "cy_px": [11.0, 21.0],
            "theta_deg": [0.0, 10.0],
            "width_px": [30.0, 31.0],
            "height_px": [20.0, 20.0],
        }
    )


def _three_router() -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "sample_id": SAMPLES,
            "scene_id": ["a", "b"],
            "prediction_source": "test_label_free",
        }
    )
    for route in ("CROG", "G1", "C1"):
        prefix = route.lower()
        candidates = _candidates(route)
        result[f"{prefix}_candidate_id"] = candidates["candidate_id"]
        result[f"{prefix}_selected_candidate_geometry_sha256"] = candidates[
            "candidate_geometry_sha256"
        ]
        result[f"{prefix}_feature"] = [0.1, 0.2]
    return result


def _d1_gate_inputs() -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "sample_id": SAMPLES,
            "prediction_source": "test_label_free",
            "score_margin": [0.2, 0.3],
            "challenger_reliability": [0.8, 0.7],
            "perturbation_stability": [0.9, 0.8],
        }
    )
    for index, column in enumerate(SAFE_GATE_FEATURE_COLUMNS):
        result[column] = float(index + 1)
    return result


def _d1_decisions() -> pd.DataFrame:
    candidates = _candidates("D1")
    return pd.DataFrame(
        {
            "sample_id": SAMPLES,
            "prediction_source": "test_label_free",
            "candidate_count": [1, 1],
            "selected_candidate_id": candidates["candidate_id"],
            "selected_geometry_sha256": candidates["candidate_geometry_sha256"],
        }
    )


def test_router_test_frame_binds_all_selected_candidates_to_top5() -> None:
    candidates = {route: _candidates(route) for route in FOUR_ROUTES}
    result = build_router_test_frame(
        three_route_frame=_three_router(),
        d1_gate_inputs=_d1_gate_inputs(),
        d1_gate_decisions=_d1_decisions(),
        candidates_by_route=candidates,
        denominator_ids=SAMPLES,
        feature_columns={
            "G1": ["g1_feature"],
            "C1": ["c1_feature"],
            "D1": ["d1_candidate_exists", "d1_ranker_score_margin"],
        },
    )

    assert len(result) == 2
    assert set(result["prediction_source"]) == {"test_label_free"}
    for route in FOUR_ROUTES:
        prefix = route.lower()
        assert result[f"{prefix}_candidate_geometry_sha256"].tolist() == candidates[
            route
        ]["candidate_geometry_sha256"].tolist()
        assert result[f"{prefix}_native_rank"].tolist() == [1, 1]


def test_router_test_frame_rejects_selected_geometry_drift() -> None:
    decisions = _d1_decisions()
    decisions.loc[0, "selected_geometry_sha256"] = "tampered"
    with pytest.raises(RuntimeError, match="D1 selected Test geometry differs"):
        build_router_test_frame(
            three_route_frame=_three_router(),
            d1_gate_inputs=_d1_gate_inputs(),
            d1_gate_decisions=decisions,
            candidates_by_route={route: _candidates(route) for route in FOUR_ROUTES},
            denominator_ids=SAMPLES,
            feature_columns={
                "G1": ["g1_feature"],
                "C1": ["c1_feature"],
                "D1": ["d1_candidate_exists"],
            },
        )


def _union_features(route: str, *, qualified: bool) -> pd.DataFrame:
    candidates = _candidates(route)
    result = pd.DataFrame(
        {
            "sample_id": SAMPLES,
            "candidate_id": candidates["candidate_id"],
            "native_rank": [1, 1],
            "native_score_raw": candidates["native_score"],
            "base_logit": [0.4, 0.5],
        }
    )
    if qualified:
        result["source_route"] = route
        result["source_candidate_id"] = result["candidate_id"]
        result["route_native_rank"] = result["native_rank"]
        result["candidate_id"] = route + ":" + result["candidate_id"]
    return result


def test_union_test_frame_is_exact_route_qualified_top20_membership() -> None:
    candidates = {route: _candidates(route) for route in FOUR_ROUTES}
    three = pd.concat(
        [_union_features(route, qualified=True) for route in ("CROG", "G1", "C1")],
        ignore_index=True,
    )
    result = build_union_test_frame(
        three_route_union=three,
        d1_features=_union_features("D1", qualified=False),
        candidates_by_route=candidates,
        denominator_ids=SAMPLES,
        feature_columns=[
            "native_score_raw",
            "base_logit",
            *[f"union_route_{route.lower()}" for route in FOUR_ROUTES],
        ],
    )

    assert len(result) == 8
    assert result.groupby("sample_id").size().tolist() == [4, 4]
    assert set(result["source_route"]) == set(FOUR_ROUTES)
    assert np.allclose(
        result[[f"union_route_{route.lower()}" for route in FOUR_ROUTES]].sum(axis=1),
        1.0,
    )


def test_union_test_frame_rejects_missing_d1_candidate() -> None:
    candidates = {route: _candidates(route) for route in FOUR_ROUTES}
    three = pd.concat(
        [_union_features(route, qualified=True) for route in ("CROG", "G1", "C1")],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="D1 Test feature membership differs"):
        build_union_test_frame(
            three_route_union=three,
            d1_features=_union_features("D1", qualified=False).iloc[:1],
            candidates_by_route=candidates,
            denominator_ids=SAMPLES,
            feature_columns=[
                "native_score_raw",
                "base_logit",
                *[f"union_route_{route.lower()}" for route in FOUR_ROUTES],
            ],
        )
