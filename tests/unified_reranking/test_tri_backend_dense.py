import numpy as np
import pandas as pd

from unified_reranking.feature_extractors.tri_backend_dense import (
    tri_backend_dense_features,
)


class _IdentityTransform:
    def native_to_model_point(self, x, y):
        return x, y

    def model_to_native_point(self, x, y, *, clip=False):
        return x, y

    def model_to_native_pose(self, x, y, angle, width, *, clip=False):
        return x, y, angle, width


def _pool(route: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": [f"{route}-0"],
            "route": [route.upper()],
            "native_rank": [1],
            "native_score": [0.8],
            "cx_px": [4.0],
            "cy_px": [4.0],
            "theta_deg": [10.0],
            "width_px": [20.0],
            "height_px": [10.0],
        }
    )


def test_tri_backend_dense_samples_every_map_at_same_original_coordinate():
    pools = {route: _pool(route) for route in ("crog", "g1", "c1")}
    calibrations = {
        route: pd.DataFrame(
            {
                "sample_id": ["s"],
                "candidate_id": [f"{route}-0"],
                "calibrated_native_probability": [0.7],
            }
        )
        for route in pools
    }
    maps = {}
    for index, route in enumerate(pools):
        quality = np.zeros((9, 9), dtype=np.float32)
        quality[4, 4] = 0.6 + index * 0.1
        maps[route] = {
            "quality_post": quality,
            "cos_2theta_post": np.full_like(quality, np.cos(np.deg2rad(20.0))),
            "sin_2theta_post": np.full_like(quality, np.sin(np.deg2rad(20.0))),
            "width_px_post": np.full_like(quality, 20.0),
        }
    result = tri_backend_dense_features(
        anchor_route="crog",
        anchor_candidates=pools["crog"],
        candidate_pools=pools,
        calibrations=calibrations,
        dense_maps=maps,
        transforms={route: _IdentityTransform() for route in pools},
    )
    assert len(result) == 1
    assert result.loc[0, "dense_crog_backend_quality_at_candidate"] == np.float32(0.6)
    assert result.loc[0, "dense_g1_backend_quality_at_candidate"] == np.float32(0.7)
    assert result.loc[0, "dense_c1_backend_quality_at_candidate"] == np.float32(0.8)
    assert result.loc[0, "number_of_backends_with_nearby_peak"] == 3.0
    assert result.loc[0, "dense_consensus_score"] > 0.0
