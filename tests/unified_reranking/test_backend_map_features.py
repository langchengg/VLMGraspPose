import numpy as np
import pandas as pd

from unified_reranking.feature_extractors.backend_maps import candidate_backend_map_features


class _ScaledCropTransform:
    def native_to_model_point(self, x, y):
        return (x - 10.0) / 10.0, (y - 20.0) / 10.0

    def model_to_native_point(self, x, y, *, clip=False):
        return 10.0 + 10.0 * x, 20.0 + 10.0 * y

    def model_to_native_pose(self, x, y, angle, width, *, clip=False):
        native_x, native_y = self.model_to_native_point(x, y, clip=clip)
        return native_x, native_y, angle, width * 10.0


def test_backend_feature_sampling_uses_stored_coordinate_without_new_peak_search():
    quality = np.zeros((9, 9), dtype=np.float32)
    quality[4, 5] = 0.9
    maps = {
        "quality_post": quality,
        "cos_2theta_post": np.ones_like(quality),
        "sin_2theta_post": np.zeros_like(quality),
        "width_px_post": np.full_like(quality, 20.0),
    }
    candidates = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["c"],
            "native_score": [0.9],
            "theta_deg": [0.0],
            "width_px": [20.0],
            "source_row": [4],
            "source_column": [5],
        }
    )
    result = candidate_backend_map_features(candidates, maps)
    assert result.loc[0, "backend_quality_at_candidate"] == np.float32(0.9)
    assert result.loc[0, "absolute_periodic_angle_difference_to_backend"] == 0.0
    assert result.loc[0, "log_width_ratio_to_backend"] == 0.0
    assert result.loc[0, "local_peak_prominence"] > 0


def test_backend_feature_sampling_round_trips_crop_coordinates_and_width():
    quality = np.zeros((9, 9), dtype=np.float32)
    quality[4, 5] = 0.9
    maps = {
        "quality_post": quality,
        "cos_2theta_post": np.ones_like(quality),
        "sin_2theta_post": np.zeros_like(quality),
        "width_px_post": np.full_like(quality, 2.0),
    }
    candidates = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["c"],
            "native_score": [0.9],
            "cx_px": [60.0],
            "cy_px": [60.0],
            "theta_deg": [0.0],
            "width_px": [20.0],
            "source_row": [4],
            "source_column": [5],
        }
    )
    result = candidate_backend_map_features(
        candidates, maps, transform=_ScaledCropTransform()
    )
    assert result.loc[0, "backend_quality_at_candidate"] == np.float32(0.9)
    assert result.loc[0, "backend_width_at_candidate_px"] == 20.0
    assert result.loc[0, "backend_transform_roundtrip_error_px"] == 0.0


def test_backend_feature_sampling_marks_fractional_out_of_crop_coordinate_missing():
    quality = np.zeros((9, 9), dtype=np.float32)
    maps = {
        "quality_post": quality,
        "cos_2theta_post": np.ones_like(quality),
        "sin_2theta_post": np.zeros_like(quality),
        "width_px_post": np.full_like(quality, 2.0),
    }
    candidates = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["outside"],
            "native_score": [0.5],
            "cx_px": [8.5],
            "cy_px": [60.0],
            "theta_deg": [0.0],
            "width_px": [20.0],
        }
    )
    result = candidate_backend_map_features(
        candidates,
        maps,
        transform=_ScaledCropTransform(),
        allow_missing=True,
    )
    assert np.isclose(result.loc[0, "backend_model_x"], -0.15)
    assert result.loc[0, "backend_map_missing"] == 1.0
    assert result.loc[0, "backend_quality_at_candidate"] == 0.0


def test_backend_feature_sampling_marks_audited_whole_map_absence_missing():
    candidates = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["missing"],
            "native_score": [0.5],
            "cx_px": [60.0],
            "cy_px": [60.0],
            "theta_deg": [0.0],
            "width_px": [20.0],
        }
    )
    result = candidate_backend_map_features(
        candidates,
        None,
        transform=_ScaledCropTransform(),
        allow_missing=True,
    )
    assert result.loc[0, "backend_map_missing"] == 1.0
    assert result.loc[0, "backend_quality_at_candidate"] == 0.0
    assert np.isfinite(result.select_dtypes(include=[float, int])).all().all()
