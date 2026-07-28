import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.grasp_metrics import (
    CORRECTED_EVALUATOR_VERSION,
    binary_mask_iou,
    evaluate_candidate,
    joint_success,
    legacy_angle_compatible,
    legacy_rectangle_iou,
    periodic_angle_difference_deg,
    periodic_angle_difference_rad,
    rasterize_rectangle,
    rectangle_iou,
    validate_binary_mask,
)
from utils.grasp_eval import calculate_jacquard_index


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FEATURES = (
    REPO_ROOT
    / "failure_analysis"
    / "reranking_outputs"
    / "full_test_17749_v1"
    / "features.jsonl"
)
FROZEN_PREDICTIONS = (
    REPO_ROOT / "failure_analysis" / "predictions" / "test_predictions.jsonl"
)


def _candidate(cx, cy, width=80.0, height=20.0, angle=0.0):
    return {
        "candidate_id": "candidate_test",
        "cx": float(cx),
        "cy": float(cy),
        "width_px": float(width),
        "height_px": float(height),
        "angle_deg": float(angle),
    }


def _gt(cx, cy, width=80.0, height=20.0, angle=0.0, target=1.0):
    return [float(cx), float(cy), float(width), float(height), float(angle), float(target)]


def test_corrected_evaluator_version_is_explicit():
    assert CORRECTED_EVALUATOR_VERSION == "corrected_geometric_v2"


def test_non_square_right_side_rectangle_is_not_clipped_on_x():
    rect = [550.0, 240.0, 80.0, 20.0, 0.0]
    mask = rasterize_rectangle(rect, shape=(480, 640))
    assert mask.any()
    assert mask[240, 550]
    assert rectangle_iou(rect, rect, shape=(480, 640), normalize_gt=False) == 1.0


def test_legacy_raster_reproduces_the_right_side_xy_failure_for_audit():
    rect = [550.0, 240.0, 80.0, 20.0, 0.0]
    assert legacy_rectangle_iou(rect, rect, shape=(480, 640), normalize_gt=False) == 0.0
    assert rectangle_iou(rect, rect, shape=(480, 640), normalize_gt=False) == 1.0


def test_legacy_angle_rule_reproduces_false_success_for_audit():
    assert legacy_angle_compatible(60.0, -60.0)
    assert periodic_angle_difference_deg(60.0, -60.0) == 60.0
    assert not joint_success(1.0, periodic_angle_difference_deg(60.0, -60.0))


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([100, 100, 40, 20, 0], [100, 100, 40, 20, 0], 1.0),
        ([100, 100, 40, 20, 0], [300, 300, 40, 20, 0], 0.0),
        ([-50, -50, 10, 10, 0], [-50, -50, 10, 10, 0], 0.0),
    ],
)
def test_rectangle_iou_basic_invariants(a, b, expected):
    actual = rectangle_iou(a, b, shape=(480, 640), normalize_gt=False)
    reverse = rectangle_iou(b, a, shape=(480, 640), normalize_gt=False)
    assert np.isfinite(actual)
    assert 0.0 <= actual <= 1.0
    assert actual == pytest.approx(reverse)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "rect",
    [
        [5, 5, 30, 20, 25],
        [635, 475, 30, 20, -25],
        [-2, 240, 30, 20, 0],
        [642, 240, 30, 20, 0],
    ],
)
def test_rectangle_raster_clips_at_all_canvas_boundaries(rect):
    mask = rasterize_rectangle(rect, shape=(480, 640))
    assert mask.shape == (480, 640)
    assert mask.dtype == np.bool_


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (60, -60, 60),
        (89, -89, 2),
        (90, -90, 0),
        (0, 180, 0),
        (12, 12 + 7 * 180, 0),
    ],
)
def test_parallel_jaw_angle_difference(a, b, expected):
    assert periodic_angle_difference_deg(a, b) == pytest.approx(expected)
    assert periodic_angle_difference_deg(b, a) == pytest.approx(expected)
    assert 0 <= periodic_angle_difference_deg(a, b) <= 90


def test_parallel_jaw_radian_difference_matches_degrees():
    for a, b in [(0, np.pi), (np.pi / 3, -np.pi / 3), (1.2, -1.1)]:
        radians = periodic_angle_difference_rad(a, b)
        degrees = periodic_angle_difference_deg(np.degrees(a), np.degrees(b))
        assert np.degrees(radians) == pytest.approx(degrees)
        assert 0 <= radians <= np.pi / 2


def test_success_thresholds_are_not_rounded():
    iou_above = np.nextafter(0.25, np.inf)
    angle_above = np.nextafter(30.0, np.inf)
    assert joint_success(iou_above, 30.0)
    assert not joint_success(0.25, 30.0)
    assert not joint_success(iou_above, angle_above)


def test_multi_gt_uses_one_successful_pair_and_is_order_invariant():
    candidate = _candidate(100, 100, angle=0)
    closest_but_wrong_angle = _gt(100, 100, angle=60)
    farther_but_valid = _gt(105, 100, angle=0)
    forward = evaluate_candidate(candidate, [closest_but_wrong_angle, farther_but_valid])
    reverse = evaluate_candidate(candidate, [farther_but_valid, closest_but_wrong_angle])
    assert forward["candidate_success"]
    assert reverse["candidate_success"]
    assert forward["best_gt"]["gt_id"] == reverse["best_gt"]["gt_id"]
    assert forward["best_gt"]["joint_success"]


def test_multi_gt_does_not_mix_iou_and_angle_from_different_targets():
    candidate = _candidate(100, 100, width=30, angle=0)
    iou_only = _gt(100, 100, width=30, angle=31)
    angle_only = _gt(300, 300, width=30, angle=0)
    result = evaluate_candidate(candidate, [iou_only, angle_only])
    assert result["any_iou_compatible"]
    assert result["any_angle_compatible"]
    assert not result["candidate_success"]
    assert result["failure_mode"] == "joint_mismatch"
    assert result["best_gt"] in result["pairwise"]


def test_compatibility_wrapper_uses_corrected_kernel_without_mutating_gt():
    prediction = [_candidate(550, 240)[key] for key in ("cx", "cy", "width_px", "height_px", "angle_deg")]
    targets = np.asarray([_gt(550, 240)], dtype=np.float64)
    before = targets.copy()
    assert calculate_jacquard_index([prediction], targets) == 1
    np.testing.assert_array_equal(targets, before)


def test_corrected_gt_mask_does_not_keep_cubic_halo():
    raw = np.zeros((32, 48), dtype=np.uint8)
    raw[8:24, 12:36] = 1
    enlarged = cv2.resize(raw.astype(np.float32), (96, 64), interpolation=cv2.INTER_CUBIC)
    restored = cv2.resize(enlarged, (48, 32), interpolation=cv2.INTER_CUBIC)
    assert np.count_nonzero(restored != 0) > int(raw.sum())
    corrected = validate_binary_mask(raw)
    assert np.count_nonzero(np.logical_xor(corrected, raw.astype(bool))) == 0
    assert corrected.sum() == raw.sum()
    assert binary_mask_iou(corrected, raw) == 1.0


def _load_golden_records(sample_ids):
    predictions = {}
    with FROZEN_PREDICTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["sample_id"]) in sample_ids:
                predictions[int(row["sample_id"])] = row
    features = {}
    with FROZEN_FEATURES.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["sample_id"]) in sample_ids:
                features[int(row["sample_id"])] = row
            if len(features) == len(sample_ids):
                break
    assert set(predictions) == set(sample_ids)
    assert set(features) == set(sample_ids)
    return predictions, features


def test_frozen_prediction_golden_samples():
    expected = {
        11246: True,
        15383: True,
        17490: True,
        12328: False,
        293: False,
        7011: False,
    }
    predictions, features = _load_golden_records(set(expected))
    for sample_id, success in expected.items():
        result = evaluate_candidate(
            features[sample_id]["candidates"][0],
            predictions[sample_id]["gt_grasps"],
        )
        assert result["candidate_success"] is success
        if success:
            assert result["best_gt"]["rectangle_iou"] > 0.25
        elif sample_id == 12328:
            assert 0.20 < result["best_gt"]["rectangle_iou"] < 0.25
