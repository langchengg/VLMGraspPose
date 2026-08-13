from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from d1_reranking.candidates import (
    canonicalise_candidate_rows,
    pool_hash_rows,
    select_pool,
    source_pose_identity_sha256,
    verify_canonical_candidate_frame,
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
            # A legacy label is deliberately present and must be stripped.
            "candidate_success": [True, False],
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


def test_canonical_projection_is_label_free_and_preserves_no_output_denominator() -> (
    None
):
    candidates, scores, paired = _frames()
    result = canonicalise_candidate_rows(candidates, scores, paired, split="test")
    assert "candidate_success" not in result.columns
    assert result["native_rank"].tolist() == [1, 2]
    assert result["theta_deg"].tolist() == pytest.approx([0.0, 90.0])
    assert result["height_px"].tolist() == [20.0, 20.0]
    pools = {name: select_pool(result, name) for name in ("top5", "top10", "allnms")}
    hashes = pool_hash_rows(pools, paired, split="test")
    assert set(hashes.loc[hashes["sample_id"] == "s1", "candidate_count"]) == {0}


def test_exact_q_ties_use_candidate_id_order() -> None:
    candidates, scores, paired = _frames()
    scores["gqcnn_q_value"] = 0.5
    scores["gqcnn_rank"] = [2, 1]
    with pytest.raises(ValueError, match="candidate-id tie order"):
        canonicalise_candidate_rows(candidates, scores, paired, split="validation")


def test_full_pose_identity_rejects_geometry_mutation() -> None:
    candidates, scores, paired = _frames()
    result = canonicalise_candidate_rows(candidates, scores, paired, split="train")
    result.loc[0, "center_depth_m"] += 0.01
    with pytest.raises(ValueError, match="full-pose identity mismatch"):
        verify_canonical_candidate_frame(result, split="train")


def test_full_pose_identity_rejects_pose_metadata_mutation() -> None:
    candidates, scores, paired = _frames()
    result = canonicalise_candidate_rows(candidates, scores, paired, split="train")
    result.loc[0, "endpoints_uv_json"] = "[[0, 0], [2, 2]]"
    with pytest.raises(ValueError, match="full-pose identity mismatch"):
        verify_canonical_candidate_frame(result, split="train")


def test_geometry_hash_does_not_depend_on_native_rank() -> None:
    candidates, scores, paired = _frames()
    result = canonicalise_candidate_rows(candidates, scores, paired, split="train")
    original = result["candidate_geometry_sha256"].copy()
    result["native_rank"] = [2, 1]
    assert result["candidate_geometry_sha256"].equals(original)


def test_candidate_score_membership_must_match_exactly() -> None:
    candidates, scores, paired = _frames()
    scores = scores.iloc[:1].copy()
    with pytest.raises(ValueError, match="candidate/score key mismatch"):
        canonicalise_candidate_rows(candidates, scores, paired, split="test")


def test_npz_float32_score_identity_contract_is_replayed() -> None:
    candidates, scores, paired = _frames()
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
                "endpoints_uv": (
                    [[0, 0], [1, 1]] if row.candidate_id == "a" else [[2, 2], [3, 3]]
                ),
                "pose_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, row.center_depth_m],
                    [0, 0, 0, 1],
                ],
            },
            precision_contract="scorer_npz_float32",
        )
        for row in candidates.itertuples(index=False)
    ]
    result = canonicalise_candidate_rows(candidates, scores, paired, split="train")
    assert set(result["candidate_identity_precision_contract"]) == {
        "scorer_npz_float32"
    }


def test_compact_test_candidate_json_expands_full_pose() -> None:
    candidates, scores, paired = _frames()
    candidates["candidate_json"] = [
        (
            '{"endpoint_1_uv":[0,0],"endpoint_2_uv":[1,1],'
            '"center_camera_xyz_m":[0,0,0.8],'
            '"T_camera_grasp_fixed_approach":'
            "[[1,0,0,0],[0,1,0,0],[0,0,1,0.8],[0,0,0,1]]}"
        ),
        (
            '{"endpoint_1_uv":[2,2],"endpoint_2_uv":[3,3],'
            '"center_camera_xyz_m":[0,0,0.9],'
            '"T_camera_grasp_fixed_approach":'
            "[[1,0,0,0],[0,1,0,0],[0,0,1,0.9],[0,0,0,1]]}"
        ),
    ]
    candidates = candidates.drop(
        columns=[
            "endpoints_uv_json",
            "center_camera_xyz_m_json",
            "pose_matrix_json",
        ]
    )
    result = canonicalise_candidate_rows(candidates, scores, paired, split="test")
    assert result["endpoints_uv_json"].str.len().gt(0).all()
    assert result["center_camera_xyz_m_json"].str.len().gt(0).all()
    assert result["pose_matrix_json"].str.len().gt(0).all()
