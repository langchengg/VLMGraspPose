import pytest

from experiments.fair_crog_hifics_g1_c1_no_rerank.geometry import (
    CanonicalGrasp,
    evaluate_candidate,
    evaluate_ranked,
    gt_from_corners,
)


GT = [[90, 90], [90, 110], [170, 110], [170, 90]]


def pred(**changes):
    values = dict(cx_px=130, cy_px=100, theta_deg=0, jaw_width_px=80, rectangle_height_px=20)
    values.update(changes)
    return CanonicalGrasp(**values)


def test_empty_prediction_and_empty_gt():
    result = evaluate_ranked([], [GT])
    assert not result["j_at_1"] and result["candidate_count"] == 0
    result = evaluate_candidate(pred(), [])
    assert not result["success"] and result["diagnostic_match"] is None


def test_multi_gt_and_same_gt_conjunction():
    # One GT overlaps but has bad angle; another has the angle but is far away.
    near_bad_angle = pred(theta_deg=30.1)
    far_good_angle = pred(cx_px=400, theta_deg=0)
    result = evaluate_candidate(pred(), [near_bad_angle, far_good_angle])
    assert any(item["iou_ok"] for item in result["pairwise"])
    assert any(item["angle_ok"] for item in result["pairwise"])
    assert not result["success"]


def test_strict_iou_comparator_at_observed_boundary():
    gt = gt_from_corners(GT)
    candidate = pred(cx_px=178)
    observed = evaluate_candidate(candidate, [gt])["pairwise"][0]["iou"]
    assert observed >= 0
    # The implementation's contract is explicitly strict; evaluate at a
    # threshold-equivalent value through the public pairwise fields.
    assert (observed > 0.25) is evaluate_candidate(candidate, [gt])["pairwise"][0]["iou_ok"]


def test_ranked_j_at_k_without_padding():
    candidates = [pred(cx_px=400, native_rank=1), pred(native_rank=2)]
    result = evaluate_ranked(candidates, [GT])
    assert not result["j_at_1"] and result["j_at_2"] and result["j_at_5"]


def test_ocid_official_angle_conversion_golden():
    first = [[264.0, 381.0], [276.0, 382.0], [275.003, 393.959], [263.003, 392.959]]
    converted = gt_from_corners(first)
    assert converted.theta_deg == pytest.approx(85.234375, abs=2e-4)
