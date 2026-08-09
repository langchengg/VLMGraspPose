import pandas as pd

from unified_reranking.feature_extractors.consensus import tri_backend_consensus_features


def _route(route, x):
    candidate = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "candidate_id": [f"{route}0", f"{route}1"],
            "native_rank": [1, 2],
            "cx_px": [x, 100.0],
            "cy_px": [10.0, 100.0],
            "theta_deg": [0.0, 45.0],
            "width_px": [20.0, 30.0],
        }
    )
    calibration = candidate[["sample_id", "candidate_id"]].assign(
        calibrated_native_probability=[0.8, 0.2]
    )
    return candidate, calibration


def test_consensus_is_invariant_to_other_backend_row_order():
    pairs = {route: _route(route, 10.0 + index) for index, route in enumerate(("crog", "g1", "c1"))}
    candidates = {route: pair[0] for route, pair in pairs.items()}
    calibration = {route: pair[1] for route, pair in pairs.items()}
    first = tri_backend_consensus_features(
        anchor_route="crog", candidates=candidates, calibrations=calibration
    ).sort_values("candidate_id").reset_index(drop=True)
    shuffled = dict(candidates)
    shuffled["g1"] = shuffled["g1"].iloc[::-1].reset_index(drop=True)
    second = tri_backend_consensus_features(
        anchor_route="crog", candidates=shuffled, calibrations=calibration
    ).sort_values("candidate_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "number_of_backends_with_nearby_peak"] == 3.0
