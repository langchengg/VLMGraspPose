from __future__ import annotations

from pathlib import Path

import pandas as pd

from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS

from d1_reranking import four_route_inputs
from d1_reranking.four_route_t4 import _read_candidates, build_t4_frame


def test_d1_router_schema_includes_route_candidate_existence(
    monkeypatch, tmp_path: Path
) -> None:
    sample_ids = ["bearing", "empty"]
    three = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "prediction_source": ["validation", "validation"],
            **{
                f"{route}_{suffix}": ["g0", ""]
                for route in ("crog", "g1", "c1")
                for suffix in ("candidate_id", "selected_candidate_geometry_sha256")
            },
        }
    )
    d1 = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "prediction_source": ["validation", "validation"],
            "native_correct": [True, False],
            "challenger_correct": [False, False],
            "score_margin": [0.25, 0.0],
            "challenger_reliability": [0.8, 0.0],
            "perturbation_stability": [0.9, 0.0],
            **{column: [0.5, 0.0] for column in SAFE_GATE_FEATURE_COLUMNS},
        }
    )
    decisions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "switch": [False, False],
            "candidate_count": [1, 0],
            "selected_candidate_id": ["d1-0", ""],
            "selected_geometry_sha256": ["d1-g0", ""],
        }
    )

    monkeypatch.setattr(
        four_route_inputs,
        "_d1_gate_decisions",
        lambda _inputs, *, gate_manifest: (
            decisions,
            tuple(SAFE_GATE_FEATURE_COLUMNS),
        ),
    )
    frame, columns = four_route_inputs._router_frame(
        three=three,
        d1=d1,
        gate_manifest=tmp_path / "unused.json",
        split="validation",
    )

    assert columns[-1] == "d1_candidate_exists"
    assert frame["d1_candidate_exists"].tolist() == [True, False]
    assert "d1_challenger_exists_numeric" in columns


def test_enriched_top5_keeps_canonical_geometry_names(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "candidate_id": ["c0"],
            "native_rank": [1],
            "candidate_geometry_sha256": ["geometry"],
            "native_score": [0.7],
            "cx_px": [10.0],
            "cy_px": [20.0],
            "theta_deg": [30.0],
            "width_px": [40.0],
            "height_px": [20.0],
        }
    )
    # T2 repeats geometry among its model columns.  The frozen candidate
    # values remain authoritative even if a feature copy differs.
    features = candidates[
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "cx_px",
            "cy_px",
            "theta_deg",
            "width_px",
            "height_px",
        ]
    ].copy()
    features["cx_px"] = 999.0
    features["base_logit"] = 0.25
    labels = candidates[["sample_id", "candidate_id", "native_rank"]].copy()
    labels["candidate_success"] = True
    labels["jacquard_margin"] = 0.5
    paths = {
        "candidates": tmp_path / "candidates.parquet",
        "features": tmp_path / "features.parquet",
        "labels": tmp_path / "labels.parquet",
    }
    candidates.to_parquet(paths["candidates"], index=False)
    features.to_parquet(paths["features"], index=False)
    labels.to_parquet(paths["labels"], index=False)

    enriched = four_route_inputs._enriched_route_top5(
        candidates_path=paths["candidates"],
        features_path=paths["features"],
        labels_path=paths["labels"],
        feature_columns=(
            "cx_px",
            "cy_px",
            "theta_deg",
            "width_px",
            "height_px",
            "base_logit",
        ),
        route="D1",
    )

    assert enriched["cx_px"].tolist() == [10.0]
    assert enriched["base_logit"].tolist() == [0.25]
    assert not any(column.endswith(("_x", "_y")) for column in enriched.columns)


def test_t4_derives_extension_identity_for_frozen_three_route_pool(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "candidate_id": ["raw-0"],
            "native_rank": [1],
            "native_score": [0.7],
            "candidate_geometry_sha256": ["geometry"],
            "cx_px": [10.0],
            "cy_px": [20.0],
            "theta_deg": [30.0],
            "width_px": [40.0],
            "height_px": [20.0],
        }
    )
    path = tmp_path / "legacy_three_route_top5.parquet"
    source.to_parquet(path, index=False)

    first = _read_candidates(path, route="CROG", split="validation")
    second = _read_candidates(path, route="CROG", split="validation")

    assert first["candidate_identity_sha256"].tolist() == second[
        "candidate_identity_sha256"
    ].tolist()
    assert len(first.loc[0, "candidate_identity_sha256"]) == 64
    assert first.loc[0, "native_score_raw"] == 0.7


def test_t4_injects_candidate_hashes_absent_from_t3() -> None:
    def candidate(route: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": ["s0"],
                "candidate_id": [f"{route}-0"],
                "native_rank": [1],
                "native_score_raw": [0.7],
                "candidate_identity_sha256": [f"{route}-identity"],
                "candidate_geometry_sha256": [f"{route}-geometry"],
                "cx_px": [10.0],
                "cy_px": [20.0],
                "theta_deg": [30.0],
                "width_px": [40.0],
                "height_px": [20.0],
            }
        )

    d1 = candidate("d1")
    t3 = d1.drop(
        columns=["candidate_identity_sha256", "candidate_geometry_sha256"]
    ).copy()
    t3["t3_signal"] = 0.5
    frame, columns = build_t4_frame(
        d1,
        {route: candidate(route.lower()) for route in ("CROG", "G1", "C1")},
        t3,
        ("t3_signal",),
    )

    assert frame["candidate_identity_sha256"].tolist() == ["d1-identity"]
    assert frame["candidate_geometry_sha256"].tolist() == ["d1-geometry"]
    assert columns[0] == "t3_signal"


def test_validation_references_replay_router_and_top15_oracle(
    tmp_path: Path,
) -> None:
    router_path = (
        tmp_path / "08_lock/route_router/route_router_validation_decisions.parquet"
    )
    labels_path = (
        tmp_path
        / "03_features/tracks/T5_cross_route/union_validation/candidate_labels.parquet"
    )
    router_path.parent.mkdir(parents=True)
    labels_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"sample_id": ["s0", "s1"], "selected_correct": [True, False]}
    ).to_parquet(router_path, index=False)
    pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_success": [False, True],
        }
    ).to_parquet(labels_path, index=False)

    references, sources = four_route_inputs._validation_reference_columns(
        three_route_run=tmp_path, sample_ids=pd.Series(["s0", "s1"])
    )

    assert references["three_route_router_correct"].tolist() == [True, False]
    assert references["existing_top15_oracle"].tolist() == [True, False]
    assert set(sources) == {
        "three_route_validation_router_decisions",
        "three_route_validation_top15_labels",
    }
