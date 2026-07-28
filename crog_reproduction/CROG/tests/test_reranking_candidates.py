import math

import cv2
import numpy as np
from skimage.feature import peak_local_max

from failure_analysis.reranking.geometry import angle_difference_deg, rasterize_candidate
from utils.grasp_eval import detect_grasp_candidates, detect_grasps


def _maps(shape=(60, 80)):
    quality = np.zeros(shape, dtype=np.float64)
    quality[20, 30] = 0.9
    quality[35, 50] = 0.8
    sin_map = np.zeros(shape, dtype=np.float64)
    cos_map = np.ones(shape, dtype=np.float64)
    width_map = np.full(shape, 0.4, dtype=np.float64)
    return quality, sin_map, cos_map, width_map


def _old_detect(quality, sin_map, cos_map, width_map, count):
    peaks = peak_local_max(
        quality, min_distance=2, threshold_abs=0.4, num_peaks=count
    )
    angle_map = np.arctan2(sin_map, cos_map) / 2.0
    grasps = []
    for peak in peaks:
        point = tuple(peak)
        grasps.append(
            [
                float(point[1]),
                float(point[0]),
                width_map[point] * 100,
                20,
                angle_map[point] / np.pi * 180,
            ]
        )
    return grasps, angle_map


def test_detect_grasps_is_exactly_backward_compatible():
    maps = _maps()
    expected, expected_angle = _old_detect(*maps, 5)
    actual, actual_angle = detect_grasps(*maps, 5)
    assert actual == expected
    assert np.array_equal(actual_angle, expected_angle)


def test_candidate_q_is_actual_peak_value_and_coordinates_are_explicit():
    quality, sin_map, cos_map, width_map = _maps()
    candidates, _ = detect_grasp_candidates(quality, sin_map, cos_map, width_map, 5)
    assert [(item["row"], item["col"]) for item in candidates] == [(20, 30), (35, 50)]
    for candidate in candidates:
        assert candidate["q_raw"] == quality[candidate["row"], candidate["col"]]
        assert candidate["cx"] == candidate["col"]
        assert candidate["cy"] == candidate["row"]
        assert candidate["legacy_rank"] == candidate["q_rank"]


def test_zero_one_and_fewer_than_five_candidates_are_not_padded():
    quality, sin_map, cos_map, width_map = _maps()
    quality[:] = 0
    assert detect_grasps(quality, sin_map, cos_map, width_map, 5)[0] == []
    quality[10, 10] = 0.7
    assert len(detect_grasps(quality, sin_map, cos_map, width_map, 5)[0]) == 1
    quality[40, 50] = 0.6
    assert len(detect_grasps(quality, sin_map, cos_map, width_map, 5)[0]) == 2


def test_parallel_jaw_angle_symmetry():
    assert angle_difference_deg(0, 180) == 0
    assert angle_difference_deg(179, 1) == 2


def test_rotated_rectangle_rasterization_matches_repo_opencv_geometry():
    quality, sin_map, cos_map, width_map = _maps()
    candidates, _ = detect_grasp_candidates(quality, sin_map, cos_map, width_map, 1)
    candidate = candidates[0]
    mask, full_count, polygon = rasterize_candidate(candidate, quality.shape)
    expected = np.zeros(quality.shape, dtype=np.uint8)
    cv2.fillPoly(expected, [np.asarray(polygon, dtype=np.intp)], 1)
    assert np.array_equal(mask, expected.astype(bool))
    assert mask.sum() == full_count
    assert candidate["candidate_checksum"]
