from __future__ import annotations

import numpy as np
import pandas as pd
import cv2

from src.unified_reranking.feature_extractors.common import _corners, extract_common_evidence


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "q", "candidate_id": "a", "route": "G1", "native_rank": 1, "native_score": 0.8, "cx_px": 32.0, "cy_px": 32.0, "theta_deg": 0.0, "width_px": 20.0, "height_px": 10.0},
            {"sample_id": "q", "candidate_id": "b", "route": "G1", "native_rank": 2, "native_score": 0.6, "cx_px": 36.0, "cy_px": 32.0, "theta_deg": 10.0, "width_px": 18.0, "height_px": 9.0},
        ]
    )


def test_common_features_preserve_candidates_and_separate_relations() -> None:
    yy, xx = np.mgrid[:64, :64]
    mask = (xx - 32) ** 2 + (yy - 32) ** 2 <= 18**2
    probability = np.where(mask, 0.9, 0.1).astype(np.float32)
    depth = (0.5 + 0.0005 * xx).astype(np.float32)
    features, relations = extract_common_evidence(
        _candidates(), probability=probability, binary_mask=mask, depth_m=depth
    )
    assert features[["sample_id", "candidate_id"]].to_records(index=False).tolist() == [("q", "a"), ("q", "b")]
    assert len(relations) == 2
    assert not any("success" in column or column.startswith("gt_") for column in features.columns)
    assert features["rectangle_probability_mean"].between(0, 1).all()
    assert features["overall_feature_reliability"].between(0, 1).all()
    assert (relations["sweep_corridor_conflict"] <= relations["rotated_rectangle_iou"] + 1e-12).all()


def test_candidate_permutation_does_not_change_keyed_features() -> None:
    mask = np.ones((64, 64), dtype=bool)
    probability = np.full((64, 64), 0.8, dtype=np.float32)
    depth = np.full((64, 64), 0.6, dtype=np.float32)
    left, _ = extract_common_evidence(
        _candidates(), probability=probability, binary_mask=mask, depth_m=depth
    )
    right, _ = extract_common_evidence(
        _candidates().iloc[::-1].reset_index(drop=True), probability=probability, binary_mask=mask, depth_m=depth
    )
    columns = ["candidate_id", "rectangle_probability_mean", "nearest_candidate_distance"]
    pd.testing.assert_frame_equal(
        left[columns].sort_values("candidate_id").reset_index(drop=True),
        right[columns].sort_values("candidate_id").reset_index(drop=True),
    )


def test_feature_vertices_match_frozen_fair_opencv_angle_convention() -> None:
    center = np.asarray([37.25, 22.75], dtype=float)
    observed = _corners(center, 31.0, 24.0, 8.0)
    expected = cv2.boxPoints(((37.25, 22.75), (24.0, 8.0), -31.0)).astype(float)
    # Vertex starting points differ, so compare the unordered point set.
    observed = observed[np.lexsort((observed[:, 1], observed[:, 0]))]
    expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    np.testing.assert_allclose(observed, expected, atol=3e-6, rtol=0.0)


def test_directed_relation_angle_uses_physical_opencv_axes() -> None:
    candidates = pd.DataFrame(
        [
            {"sample_id": "q", "candidate_id": "source", "route": "G1", "native_rank": 1, "native_score": 0.8, "cx_px": 32.0, "cy_px": 32.0, "theta_deg": -30.0, "width_px": 20.0, "height_px": 10.0},
            {"sample_id": "q", "candidate_id": "target", "route": "G1", "native_rank": 2, "native_score": 0.6, "cx_px": 36.0, "cy_px": 32.0, "theta_deg": 30.0, "width_px": 18.0, "height_px": 9.0},
        ]
    )
    _, relations = extract_common_evidence(
        candidates,
        probability=np.ones((64, 64), dtype=np.float32),
        binary_mask=np.ones((64, 64), dtype=bool),
        depth_m=np.ones((64, 64), dtype=np.float32),
    )
    forward = relations.loc[
        (relations["source_candidate_id"] == "source")
        & (relations["target_candidate_id"] == "target")
    ].iloc[0]
    reverse = relations.loc[
        (relations["source_candidate_id"] == "target")
        & (relations["target_candidate_id"] == "source")
    ].iloc[0]
    # OpenCV physical angles are -theta, hence +30 -> -30 is a -60 degree
    # directed change and sin(2 * -60 degrees) is negative.
    assert np.isclose(forward["sin_2_delta_angle"], np.sin(np.deg2rad(-120.0)))
    assert np.isclose(reverse["sin_2_delta_angle"], -forward["sin_2_delta_angle"])
