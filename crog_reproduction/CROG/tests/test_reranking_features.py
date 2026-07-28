from copy import deepcopy

import cv2
import numpy as np
import pytest

from failure_analysis.reranking.feature_extraction import (
    DEFAULT_FEATURE_CONFIG,
    extract_features_for_candidates,
    load_pcd_xyz,
    mask_span_width,
)
from failure_analysis.reranking.geometry import grasp_polygon
from failure_analysis.reranking.rankers import RANKER_NAMES, rank_candidates, score_candidate
from failure_analysis.reranking.schema import assert_no_forbidden_feature_keys
from utils.grasp_eval import detect_grasp_candidates


def _candidate(shape=(100, 120), center=(60, 50), width=40, angle=0):
    quality = np.zeros(shape, dtype=np.float64)
    col, row = center
    quality[row, col] = 0.9
    radians = np.deg2rad(angle)
    sin_map = np.full(shape, np.sin(2 * radians), dtype=np.float64)
    cos_map = np.full(shape, np.cos(2 * radians), dtype=np.float64)
    width_map = np.full(shape, width / 100.0, dtype=np.float64)
    candidates, _ = detect_grasp_candidates(quality, sin_map, cos_map, width_map, 1)
    return candidates, quality, sin_map, cos_map


def _extract(mask_probability, *, center=(60, 50), width=40, angle=0, depth=None):
    candidates, quality, sin_map, cos_map = _candidate(
        mask_probability.shape, center=center, width=width, angle=angle
    )
    if depth is None:
        depth = np.ones(mask_probability.shape, dtype=np.float64)
    return extract_features_for_candidates(
        candidates,
        mask_probability=mask_probability,
        quality=quality,
        sin_map=sin_map,
        cos_map=cos_map,
        depth_m=depth,
    )[0]


def test_coverage_one_zero_and_partial_border():
    mask = np.ones((100, 120), dtype=np.float64)
    full = _extract(mask)
    assert full["features"]["soft_coverage"]["value"] == 1
    assert full["features"]["binary_coverage"]["value"] == 1

    empty = _extract(np.zeros_like(mask))
    assert empty["features"]["soft_coverage"]["value"] == 0
    assert empty["features"]["binary_coverage"]["value"] == 0
    assert empty["features"]["width_compatibility"]["reliability"] == 0

    border = _extract(mask, center=(4, 4), width=40)
    assert 0 < border["features"]["image_support"]["value"] < 1
    assert border["features"]["soft_coverage"]["reliability"] < 1


@pytest.mark.parametrize("angle", [0, 45, 90])
def test_mask_span_width_at_canonical_angles(angle):
    shape = (120, 140)
    candidates, _, _, _ = _candidate(shape, center=(70, 60), width=40, angle=angle)
    polygon = grasp_polygon(70, 60, 40, 9, angle)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.intp)], 1)
    span, reason = mask_span_width(candidates[0], mask.astype(bool), DEFAULT_FEATURE_CONFIG)
    assert reason is None
    assert 35 <= span["object_width_px"] <= 44


def test_empty_mask_and_invalid_depth_produce_neutral_missing_features():
    shape = (100, 120)
    result = _extract(
        np.zeros(shape, dtype=np.float64),
        depth=np.full(shape, np.nan, dtype=np.float64),
    )
    assert result["features"]["center_margin"]["reliability"] == 0
    assert result["features"]["depth_geometry"]["reliability"] == 0
    assert result["features"]["safety"]["reliability"] == 0
    for feature in result["features"].values():
        assert np.isfinite(feature["reliability"])
        assert feature["value"] is None or np.isfinite(feature["value"])


def test_binary_pcd_reader_copies_xyz_and_drops_gt_label(tmp_path):
    path = tmp_path / "sample.pcd"
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgba label\n"
        "SIZE 4 4 4 4 4\nTYPE F F F U U\nCOUNT 1 1 1 1 1\n"
        "WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n"
    ).encode("ascii")
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgba", "<u4"), ("label", "<u4")]
    )
    records = np.zeros(2, dtype=dtype)
    records["x"] = [1, 2]
    records["z"] = [np.nan, 0.7]
    records["label"] = [999, 123]
    path.write_bytes(header + records.tobytes())
    xyz, reason = load_pcd_xyz(path)
    assert reason is None
    assert xyz.shape == (1, 2, 3)
    assert np.isnan(xyz[0, 0, 2])
    assert xyz[0, 1, 2] == pytest.approx(0.7)
    assert xyz.dtype.fields is None


def test_ranker_scores_ignore_all_non_allowlisted_gt_fields_and_ties_are_stable():
    mask = np.ones((100, 120), dtype=np.float64)
    candidate = _extract(mask)
    contaminated = deepcopy(candidate)
    contaminated.update(
        {
            "gt_grasps": [[1, 2, 3, 4, 5]],
            "j1_success": True,
            "candidate_validity": True,
            "iou_with_gt": 1.0,
        }
    )
    for ranker in ("q_only", "rule_2d_equal", "rule_fixed_v1"):
        assert score_candidate(candidate, ranker)[0] == score_candidate(contaminated, ranker)[0]

    duplicate = deepcopy(candidate)
    duplicate["candidate_id"] = "candidate_1"
    duplicate["legacy_rank"] = 1
    ranked = rank_candidates([candidate, duplicate], "rule_fixed_v1")
    assert [item["candidate_id"] for item in ranked] == ["candidate_0", "candidate_1"]


def test_features_schema_rejects_gt_leakage_keys():
    feature_record = {"sample_id": 1, "candidates": [_extract(np.ones((100, 120)))]}
    assert_no_forbidden_feature_keys(feature_record)
    feature_record["gt_grasps"] = []
    with pytest.raises(ValueError):
        assert_no_forbidden_feature_keys(feature_record)
