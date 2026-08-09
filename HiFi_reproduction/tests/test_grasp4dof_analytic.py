from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from src.grasping.backends.mask_depth_analytic import (
    ANALYTIC_FEATURE_NAMES,
    CLEARANCE_PROXY_SEMANTICS,
    AnalyticGraspConfig,
    MaskDepthAnalyticBackend,
    clean_predicted_mask,
    estimate_inward_normals,
    generate_antipodal_pairs,
    sample_contour_by_arclength,
    score_candidate_features,
    visible_surface_occupancy,
)
from src.grasping.common.candidate_decoder import NMSConfig


def _config(**changes) -> AnalyticGraspConfig:
    values = {
        "min_component_area_px": 20,
        "max_hole_area_px": 20,
        "opening_radius_px": 0,
        "closing_radius_px": 1,
        "contour_spacing_px": 2.0,
        "max_contour_points": 192,
        "min_width_px": 8.0,
        "max_width_px": 55.0,
        "min_width_m": 0.005,
        "max_width_m": 0.15,
        "antipodal_alignment_min": 0.45,
        "normal_axis_alignment_min": 0.35,
        "min_axis_mask_support": 0.45,
        "max_jaw_depth_difference_m": 0.08,
        "max_axis_depth_jump_m": 0.08,
        "max_raw_candidates": 100,
        "nms": NMSConfig(
            center_distance_px=1.0,
            angle_distance_deg=1.0,
            width_distance_px=1.0,
            rectangle_iou_threshold=0.99,
            max_output=50,
        ),
    }
    values.update(changes)
    return AnalyticGraspConfig(**values)


def _scene(*, depth_delta: float = 0.0):
    shape = (96, 128)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[28:68, 48:78] = 1
    probability = np.full(shape, 0.02, dtype=np.float32)
    probability[mask.astype(bool)] = 0.95
    depth = np.full(shape, 1.2, dtype=np.float32)
    depth[mask.astype(bool)] = 1.0
    if depth_delta:
        depth[28:68, 63:78] += float(depth_delta)
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 63.5, "cy": 47.5}
    return probability, mask, depth, intrinsics


def test_mask_cleanup_removes_speck_fills_hole_and_closes() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 18:46] = 1
    mask[30:33, 31:34] = 0
    mask[3, 3] = 1
    cleaned, stats = clean_predicted_mask(mask, _config())
    assert not cleaned[3, 3]
    assert cleaned[31, 32]
    assert stats["input_component_count"] == 2
    assert stats["cleaned_component_count"] == 1


def test_arclength_sampling_and_normals_point_inward() -> None:
    mask = np.zeros((60, 80), dtype=bool)
    mask[15:45, 20:60] = True
    contour_rc = __import__("skimage").measure.find_contours(mask.astype(float), 0.5)[0]
    contour_xy = np.column_stack((contour_rc[:, 1], contour_rc[:, 0]))
    points = sample_contour_by_arclength(contour_xy, spacing_px=4.0, max_points=100)
    normals = estimate_inward_normals(points, mask)
    steps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    assert float(np.std(steps) / np.mean(steps)) < 0.2
    inward = points + 2.0 * normals
    x = np.clip(np.rint(inward[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(inward[:, 1]).astype(int), 0, mask.shape[0] - 1)
    assert float(np.mean(mask[y, x])) > 0.9


def test_ckdtree_antipodal_pair_filter_uses_opposed_inward_normals() -> None:
    points = np.asarray([[10.0, 20.0], [30.0, 20.0], [20.0, 10.0]])
    normals = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    pairs = generate_antipodal_pairs(
        points,
        normals,
        min_width_px=15.0,
        max_width_px=25.0,
        antipodal_alignment_min=0.9,
        normal_axis_alignment_min=0.9,
    )
    assert len(pairs) == 1
    assert pairs[0]["left_index"] == 0
    assert pairs[0]["right_index"] == 1
    assert pairs[0]["antipodal_normal_alignment"] == 1.0


def test_visible_surface_occupancy_detects_closer_non_target_depth() -> None:
    depth = np.full((40, 40), 1.2, dtype=np.float32)
    mask = np.zeros((40, 40), dtype=bool)
    points = np.asarray([[x, y] for y in range(10, 20) for x in range(10, 20)])
    clear = visible_surface_occupancy(
        depth, mask, points, reference_depth_m=1.0, closer_margin_m=0.01
    )
    depth[10:20, 10:20] = 0.8
    blocked = visible_surface_occupancy(
        depth, mask, points, reference_depth_m=1.0, closer_margin_m=0.01
    )
    assert clear == 0.0
    assert blocked == 1.0


def test_candidate_features_include_depth_difference_occupancy_and_finite_score() -> None:
    probability, mask, depth, intrinsics = _scene(depth_delta=0.01)
    backend = MaskDepthAnalyticBackend(_config())
    candidates, stats = backend.generate_candidates(
        sample_id="synthetic",
        probability=probability,
        binary_mask=mask,
        depth_m=depth,
        intrinsics=intrinsics,
    )
    assert candidates, stats
    candidate = candidates[0]
    features = candidate.metadata["features"]
    assert set(ANALYTIC_FEATURE_NAMES) == set(features)
    assert any(
        item.metadata["features"]["jaw_depth_difference"] > 0.005
        for item in candidates
    )
    assert features["local_depth_variance"] >= 0.0
    assert 0.0 <= features["visible_clearance"] <= 1.0
    assert math.isfinite(candidate.score)
    assert 0.0 <= candidate.score <= 1.0
    assert candidate.metadata["clearance_proxy_semantics"] == CLEARANCE_PROXY_SEMANTICS
    rescored, terms = score_candidate_features(features, backend.config)
    assert math.isfinite(rescored)
    assert terms["standardized_score"] == rescored


def test_predict_empty_mask_records_reason_and_cpu() -> None:
    probability, mask, depth, intrinsics = _scene()
    prediction = MaskDepthAnalyticBackend(_config()).predict(
        {
            "sample_id": "empty",
            "predicted_probability": probability,
            "predicted_mask": np.zeros_like(mask),
            "depth": depth,
            "intrinsics": intrinsics,
        }
    )
    assert prediction.top1 is None
    assert prediction.top5 == ()
    assert prediction.empty_reason == "empty_mask"
    assert prediction.device == "cpu"


def test_predict_duck_typed_sample_returns_stable_unpadded_top5() -> None:
    probability, mask, depth, intrinsics = _scene()
    backend = MaskDepthAnalyticBackend(_config())
    sample = SimpleNamespace(
        sample_id="duck-sample",
        predicted_probability=probability,
        predicted_mask=mask,
        depth_m=depth,
        intrinsics=intrinsics,
    )
    first = backend.predict(sample)
    second = backend.predict(sample)
    assert first.device == "cpu"
    assert len(first.top5) == 5
    assert first.top1 == first.top5[0]
    assert len({item.candidate_id for item in first.top5}) == len(first.top5)
    assert [item.candidate_id for item in first.top5] == [
        item.candidate_id for item in second.top5
    ]
    assert first.nms_candidate_count <= first.raw_candidate_count
