import numpy as np
import pandas as pd

from unified_reranking.feature_extractors.rgb import candidate_rgb_features


def test_rgb_features_are_candidate_aligned_and_finite():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :16, 0] = 255
    candidates = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["c"],
            "cx_px": [16.0],
            "cy_px": [16.0],
            "theta_deg": [0.0],
            "width_px": [12.0],
            "height_px": [6.0],
        }
    )
    result = candidate_rgb_features(candidates, rgb)
    assert result.loc[0, "sample_id"] == "s"
    assert result.loc[0, "candidate_id"] == "c"
    assert np.isfinite(result.drop(columns=["sample_id", "candidate_id"]).to_numpy(float)).all()
    assert result.loc[0, "local_rgb_red_mean"] > result.loc[0, "local_rgb_blue_mean"]
