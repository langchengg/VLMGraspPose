from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from d1_reranking.candidates import (
    canonicalise_candidate_rows,
    source_pose_identity_sha256,
)
from d1_reranking.features import (
    assemble_route_rich_features,
    project_native_available_features,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "sample_index": [0, 1],
            "scene_id": ["scene0", "scene1"],
            "frame_id": ["frame0", "frame1"],
        }
    )
    candidates = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "sample_index": [0, 0],
            "scene_id": ["scene0", "scene0"],
            "candidate_id": ["a", "b"],
            "center_u_px": [10.0, 12.0],
            "center_v_px": [20.0, 21.0],
            "center_depth_m": [0.8, 0.9],
            "angle_rad": [0.0, np.pi / 2],
            "width_m": [0.05, 0.06],
            "width_px": [30.0, 31.0],
            "valid": [True, True],
            "candidate_json": ["{}", "{}"],
            "endpoints_uv_json": ["[[0, 0], [1, 1]]", "[[2, 2], [3, 3]]"],
            "center_camera_xyz_m_json": ["[0, 0, 0.8]", "[0, 0, 0.9]"],
            "pose_matrix_json": [
                "[[1,0,0,0],[0,1,0,0],[0,0,1,0.8],[0,0,0,1]]",
                "[[1,0,0,0],[0,1,0,0],[0,0,1,0.9],[0,0,0,1]]",
            ],
        }
    )
    scores = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "sample_index": [0, 0],
            "scene_id": ["scene0", "scene0"],
            "candidate_id": ["a", "b"],
            "candidate_identity_sha256": ["", ""],
            "gqcnn_q_value": [0.9, 0.8],
            "gqcnn_rank": [1, 2],
            "source_candidate_index": [0, 1],
            "source_candidates_npz_sha256": ["c" * 64, "c" * 64],
            "scored_candidates_npz_sha256": ["d" * 64, "d" * 64],
            "model_name": ["GQCNN-4.0-PJ", "GQCNN-4.0-PJ"],
            "model_commit": ["e" * 40, "e" * 40],
            "model_config_sha256": ["f" * 64, "f" * 64],
        }
    )
    scores["candidate_identity_sha256"] = [
        source_pose_identity_sha256(
            {
                "sample_id": row.sample_id,
                "candidate_id": row.candidate_id,
                "center_u_px": row.center_u_px,
                "center_v_px": row.center_v_px,
                "center_depth_m": row.center_depth_m,
                "center_camera_xyz_m": [0, 0, row.center_depth_m],
                "angle_rad": row.angle_rad,
                "width_m": row.width_m,
                "width_px": row.width_px,
                "endpoints_uv": [[0, 0], [1, 1]]
                if row.candidate_id == "a"
                else [[2, 2], [3, 3]],
                "pose_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, row.center_depth_m],
                    [0, 0, 0, 1],
                ],
            }
        )
        for row in candidates.itertuples(index=False)
    ]
    return candidates, scores, paired


def test_native_feature_projection_uses_only_explicit_allowlist() -> None:
    candidates, scores, paired = _frames()
    canonical = canonicalise_candidate_rows(candidates, scores, paired, split="test")
    source = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "original_gqcnn_rank": [1, 2],
            "q_raw": [0.9, 0.8],
            "p_center": [0.7, 0.6],
            "candidate_success": [1, 0],
        }
    )
    projected, model_columns = project_native_available_features(
        canonical, source, ("p_center",), split="test"
    )
    assert "candidate_success" not in projected.columns
    assert "p_center" in model_columns


def test_native_feature_projection_rejects_q_drift() -> None:
    candidates, scores, paired = _frames()
    canonical = canonicalise_candidate_rows(candidates, scores, paired, split="train")
    source = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "original_gqcnn_rank": [1, 2],
            "q_raw": [0.91, 0.8],
            "p_center": [0.7, 0.6],
        }
    )
    with pytest.raises(ValueError, match="q differs"):
        project_native_available_features(
            canonical, source, ("p_center",), split="train"
        )


def test_route_rich_track_is_exact_prefixed_t2_plus_t1_superset() -> None:
    t2 = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "native_rank": [1, 2],
            "p_center": [0.8, 0.4],
        }
    )
    t1 = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "candidate_id": ["a", "b"],
            "native_rank": [1, 2],
            "gqcnn_crop_mean": [0.1, 0.2],
        }
    )
    frame, columns = assemble_route_rich_features(t2, t1)
    assert len(frame) == 2
    assert "p_center" in columns
    assert "d1_route_gqcnn_crop_mean" in columns


def test_route_rich_track_rejects_membership_drift() -> None:
    t2 = pd.DataFrame({"sample_id": ["s0"], "candidate_id": ["a"], "native_rank": [1]})
    t1 = pd.DataFrame({"sample_id": ["s0"], "candidate_id": ["b"], "native_rank": [1]})
    with pytest.raises(ValueError, match="membership mismatch"):
        assemble_route_rich_features(t2, t1)
