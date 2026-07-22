from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from src.grasping.camera_geometry import CameraIntrinsicsData
from src.grasping.geometric_ranker import (
    _robust_normalize,
    compute_raw_features,
    load_frozen_candidates,
    rank_candidates,
    save_ranked_npz,
    save_strict_json,
    sha256_file,
)
from src.grasping.mask_processing import binary_dilate
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    yaml = YAML(typ="safe")
    return yaml.load((ROOT / "configs" / "dexnet_geometric_ranker.yaml").read_text())


def _intrinsics(shape=(48, 64)) -> CameraIntrinsicsData:
    return CameraIntrinsicsData(
        frame="test_camera",
        fx=100.0,
        fy=100.0,
        cx=shape[1] / 2,
        cy=shape[0] / 2,
        skew=0.0,
        height=shape[0],
        width=shape[1],
    )


def _record(candidate_id: str, center=(32.0, 24.0)) -> dict:
    return {
        "candidate_id": candidate_id,
        "sample_id": "synthetic",
        "query": "test",
        "center_uv": list(center),
        "center_u_px": center[0],
        "center_v_px": center[1],
        "center_depth_m": 1.0,
        "center_camera_xyz_m": [0.0, 0.0, 1.0],
        "angle_rad": 0.0,
        "angle_deg": 0.0,
        "width_m": 0.05,
        "width_px": 30.0,
        "endpoints_uv": [[17.0, 24.0], [47.0, 24.0]],
        "endpoint_1_uv": [17.0, 24.0],
        "endpoint_2_uv": [47.0, 24.0],
        "contact_points_uv": [[27.0, 24.0], [37.0, 24.0]],
        "contact_normals": [[-1.0, 0.0], [1.0, 0.0]],
        "T_camera_grasp_fixed_approach": np.eye(4).tolist(),
        "gqcnn_q_value": None,
        "rejection_reason": None,
    }


def _scene():
    shape = (48, 64)
    depth = np.ones(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    mask[16:33, 25:40] = True
    return depth, mask


def test_bounded_degenerate_normalization_preserves_zero_and_one():
    zeros, zero_stats = _robust_normalize(
        np.zeros(3), bounded=True, lower_quantile=0.1, upper_quantile=0.9, epsilon=1e-12
    )
    ones, one_stats = _robust_normalize(
        np.ones(3), bounded=True, lower_quantile=0.1, upper_quantile=0.9, epsilon=1e-12
    )
    assert np.array_equal(zeros, np.zeros(3))
    assert np.array_equal(ones, np.ones(3))
    assert zero_stats["fallback"] == one_stats["fallback"] == "preserve_raw_bounded_0_1"


def test_unbounded_degenerate_normalization_is_neutral():
    values, stats = _robust_normalize(
        np.full(4, 7.0), bounded=False, lower_quantile=0.1, upper_quantile=0.9, epsilon=1e-12
    )
    assert np.array_equal(values, np.full(4, 0.5))
    assert stats["fallback"] == "neutral_0.5"


def test_opposed_contact_normals_score_above_same_direction():
    depth, mask = _scene()
    config = _config()
    common = dict(
        depth_m=depth,
        target_mask=mask,
        dilated_mask=binary_dilate(mask, 3),
        boundary_distance=ndimage.distance_transform_edt(mask),
        centroid_uv=np.array([32.0, 24.0]),
        intrinsics=_intrinsics(),
        config=config,
    )
    opposed = compute_raw_features(_record("opposed"), **common)
    same_record = _record("same")
    same_record["contact_normals"] = [[-1.0, 0.0], [-1.0, 0.0]]
    same = compute_raw_features(same_record, **common)
    assert opposed["local_antipodal_score_raw"] == 1.0
    assert same["local_antipodal_score_raw"] == 0.0


def test_nearer_off_target_depth_creates_interference_penalty():
    depth, mask = _scene()
    depth[21:28, 18:24] = 0.8
    config = _config()
    feature = compute_raw_features(
        _record("hazard"),
        depth_m=depth,
        target_mask=mask,
        dilated_mask=binary_dilate(mask, 3),
        boundary_distance=ndimage.distance_transform_edt(mask),
        centroid_uv=np.array([32.0, 24.0]),
        intrinsics=_intrinsics(),
        config=config,
    )
    assert feature["interference_penalty_raw"] > 0.0


def test_deterministic_tie_break_uses_candidate_id():
    depth, mask = _scene()
    ranked, _ = rank_candidates(
        [_record("g0002"), _record("g0001")],
        depth_m=depth,
        target_mask=mask,
        intrinsics=_intrinsics(),
        config=_config(),
    )
    assert [item["candidate_id"] for item in ranked] == ["g0001", "g0002"]


def test_strict_json_converts_nonfinite_scores_to_null(tmp_path: Path):
    destination = save_strict_json(tmp_path / "strict.json", {"q": float("nan")})
    text = destination.read_text()
    assert "NaN" not in text
    assert json.loads(text) == {"q": None}


def test_ranked_npz_is_deterministic_and_pickle_free(tmp_path: Path):
    depth, mask = _scene()
    ranked, _ = rank_candidates(
        [_record("g0001"), _record("g0002", center=(33.0, 24.0))],
        depth_m=depth,
        target_mask=mask,
        intrinsics=_intrinsics(),
        config=_config(),
    )
    first = save_ranked_npz(tmp_path / "first.npz", ranked)
    second = save_ranked_npz(tmp_path / "second.npz", ranked)
    assert sha256_file(first) == sha256_file(second)
    with np.load(first, allow_pickle=False) as archive:
        assert archive["candidate_id"].dtype.kind == "U"
        assert archive["geometric_score"].shape == (2,)


def test_real_frozen_npz_json_are_cross_checked_without_mutation(tmp_path: Path):
    sample = ROOT / "outputs" / "dexnet_candidates_one_sample" / "q0000000_b32eb3299dcd3ae9"
    source_npz = sample / "candidates.npz"
    source_json = sample / "candidates.json"
    before = sha256_file(source_npz)
    records, _, hashes = load_frozen_candidates(source_npz, source_json)
    assert records[0]["candidate_id"].startswith("g")
    assert hashes["candidates_npz_sha256"] == before == sha256_file(source_npz)

    payload = json.loads(source_json.read_text())
    payload["candidates"][0]["center_u_px"] += 1.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload, allow_nan=True))
    try:
        load_frozen_candidates(source_npz, changed)
    except ValueError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("mismatched JSON sidecar must fail closed")
