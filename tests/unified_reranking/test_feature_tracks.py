import numpy as np
import pandas as pd
import pytest

from unified_reranking.feature_tracks import (
    assemble_common_track,
    merge_calibration_features,
    select_crog_native_reference_columns,
)


def test_common_track_keeps_labels_out_of_model_schema():
    candidates = pd.DataFrame(
        {"sample_id": ["s"], "candidate_id": ["c"], "route": ["G1"]}
    )
    features = candidates.assign(native_score_raw=[0.8], p_center=[0.5])
    calibration = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["c"],
            "candidate_success": [True],
            "calibrated_native_probability": [0.7],
            "base_logit": [0.847],
        }
    )
    track = assemble_common_track(candidates, features, calibration)
    assert "candidate_success" not in track.frame
    assert set(track.model_columns) == {
        "base_logit",
        "calibrated_native_probability",
        "native_score_raw",
        "p_center",
    }


def test_calibration_requires_exact_candidate_coverage():
    features = pd.DataFrame(
        {"sample_id": ["s"], "candidate_id": ["c"], "route": ["C1"], "x": [1.0]}
    )
    calibration = pd.DataFrame(
        {
            "sample_id": ["other"],
            "candidate_id": ["c"],
            "calibrated_native_probability": [0.5],
            "base_logit": [0.0],
        }
    )
    with pytest.raises(ValueError, match="cover every candidate"):
        merge_calibration_features(features, calibration)


def test_crog_native_selector_excludes_depth_and_collision_proxies():
    reference = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["c"],
            "q_raw": [0.8],
            "center_prob": [0.7],
            "depth_mad_m": [0.1],
            "collision_proxy": [0.2],
            "z_reference": [1.0],
            "width_m": [0.05],
        }
    )
    selected = select_crog_native_reference_columns(reference)
    assert selected == ("center_prob", "q_raw")
    assert all(np.issubdtype(reference[name].dtype, np.number) for name in selected)
