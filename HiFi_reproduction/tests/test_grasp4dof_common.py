"""Focused contract tests for the shared 4-DoF grasp protocol."""

from __future__ import annotations

import numpy as np
import pytest
import cv2
from skimage.draw import polygon as draw_polygon

from src.grasping.common import (
    CropTransform,
    EvaluatorConfig,
    Grasp4DoF,
    NMSConfig,
    build_prediction,
    decode_quality_maps,
    evaluate_ocid_predictions,
    extract_quality_peaks,
    grasp_from_ocid_corners,
    non_maximum_suppression,
    normalize_angle_deg,
    periodic_angle_difference_deg,
    rasterized_rectangle_iou,
    serialize_top5,
    stable_candidate_id,
    transform_oriented_length,
)
from src.grasping.common.geometry import _rectangle_pixels


def _grasp(
    x: float,
    y: float,
    angle: float = 0.0,
    width: float = 60.0,
    score: float = 1.0,
    candidate_id: str = "candidate",
    height: float = 20.0,
) -> Grasp4DoF:
    return Grasp4DoF(
        center_x=x,
        center_y=y,
        angle_deg=angle,
        width_px=width,
        height_px=height,
        score=score,
        candidate_id=candidate_id,
    )


def _ocid_corners(
    x: float,
    y: float,
    angle: float,
    width: float,
    source_height: float = 20.0,
) -> np.ndarray:
    """Corners whose frozen OCID axis is corner[3] - corner[0]."""

    radians = np.deg2rad(angle)
    axis = np.asarray([np.cos(radians), np.sin(radians)])
    normal = np.asarray([-axis[1], axis[0]])
    center = np.asarray([x, y])
    return np.stack(
        (
            center - 0.5 * width * axis - 0.5 * source_height * normal,
            center - 0.5 * width * axis + 0.5 * source_height * normal,
            center + 0.5 * width * axis + 0.5 * source_height * normal,
            center + 0.5 * width * axis - 0.5 * source_height * normal,
        )
    )


def test_x_is_column_y_is_row_and_rasterization_clips_to_image() -> None:
    inside_right_edge = _grasp(95.0, 20.0, width=20.0)
    swapped_outside_height = _grasp(20.0, 95.0, width=20.0)

    assert rasterized_rectangle_iou(
        inside_right_edge, inside_right_edge, shape=(40, 100)
    ) == pytest.approx(1.0)
    assert (
        rasterized_rectangle_iou(
            swapped_outside_height, swapped_outside_height, shape=(40, 100)
        )
        == 0.0
    )


def test_physical_angle_matches_crog_corrected_internal_sign_convention() -> None:
    """CROG stores the negative of our image-space physical grasp angle."""

    random = np.random.default_rng(20260803)
    for index in range(200):
        grasp = _grasp(
            float(random.uniform(-20.0, 660.0)),
            float(random.uniform(-20.0, 500.0)),
            angle=float(random.uniform(-90.0, 90.0)),
            width=float(random.uniform(1.0, 120.0)),
            height=float(random.uniform(1.0, 50.0)),
            candidate_id=f"parity-{index}",
        )
        # CROG rectangle_vertices calls cv2.boxPoints(..., -internal_angle).
        crog_internal_angle = -grasp.angle_deg
        box = cv2.boxPoints(
            (
                (grasp.center_x, grasp.center_y),
                (grasp.width_px, grasp.height_px),
                -crog_internal_angle,
            )
        ).astype(np.intp)
        rows, columns = draw_polygon(box[:, 1], box[:, 0], shape=(480, 640))
        crog_pixels = np.unique(rows.astype(np.int64) * 640 + columns.astype(np.int64))

        assert np.array_equal(_rectangle_pixels(grasp, (480, 640)), crog_pixels)


def test_crop_pose_mapping_is_reversible_and_clip_uses_native_bounds() -> None:
    transform = CropTransform(
        crop_x=100,
        crop_y=50,
        crop_width=320,
        crop_height=240,
        model_width=160,
        model_height=80,
        native_width=640,
        native_height=480,
    )
    native = (260.0, 170.0, 37.0, 84.0)

    model = transform.native_to_model_pose(*native)
    restored = transform.model_to_native_pose(*model)

    assert restored[:2] == pytest.approx(native[:2])
    assert periodic_angle_difference_deg(restored[2], native[2]) == pytest.approx(0.0)
    assert restored[3] == pytest.approx(native[3])
    assert transform.model_to_native_point(999, -999, clip=True) == (639.0, 0.0)


def test_angle_is_180_degree_periodic() -> None:
    assert normalize_angle_deg(179.0) == -1.0
    assert normalize_angle_deg(90.0) == -90.0
    assert periodic_angle_difference_deg(0.0, 179.0) == 1.0
    assert periodic_angle_difference_deg(-89.0, 89.0) == 2.0


def test_width_scaling_follows_grasp_axis_under_anisotropic_resize() -> None:
    horizontal_width, horizontal_angle = transform_oriented_length(
        10.0, 0.0, scale_x=2.0, scale_y=3.0
    )
    vertical_width, vertical_angle = transform_oriented_length(
        10.0, 90.0, scale_x=2.0, scale_y=3.0
    )

    assert (horizontal_width, horizontal_angle) == pytest.approx((20.0, 0.0))
    assert vertical_width == pytest.approx(30.0)
    assert periodic_angle_difference_deg(vertical_angle, 90.0) == pytest.approx(0.0)


def test_ocid_conversion_clips_width_and_uses_configurable_fixed_height() -> None:
    grasp = grasp_from_ocid_corners(
        _ocid_corners(100.0, 120.0, 20.0, 140.0, source_height=55.0),
        fixed_height_px=24.0,
        width_clip_px=100.0,
    )

    assert (grasp.center_x, grasp.center_y) == pytest.approx((100.0, 120.0))
    assert grasp.width_px == pytest.approx(100.0)
    assert grasp.height_px == pytest.approx(24.0)
    assert periodic_angle_difference_deg(grasp.angle_deg, 20.0) == pytest.approx(0.0)


def test_peak_extraction_and_map_decoder_preserve_x_column_y_row() -> None:
    quality = np.zeros((5, 7), dtype=np.float32)
    quality[1, 5] = 0.9
    quality[4, 0] = 0.8
    cos_2theta = np.ones_like(quality)
    sin_2theta = np.zeros_like(quality)
    sin_2theta[1, 5] = 1.0
    cos_2theta[1, 5] = 0.0
    width = np.full_like(quality, 4.0)

    peaks = extract_quality_peaks(
        quality, threshold=0.5, min_distance_px=1, max_peaks=5
    )
    decoded = decode_quality_maps(
        quality,
        cos_2theta,
        sin_2theta,
        width,
        sample_id="sample",
        backend="backend",
        quality_threshold=0.5,
        min_peak_distance_px=1,
        width_scale=2.5,
    )

    assert peaks == [(1, 5, pytest.approx(0.9)), (4, 0, pytest.approx(0.8))]
    assert (decoded[0].center_x, decoded[0].center_y) == (5.0, 1.0)
    assert decoded[0].angle_deg == pytest.approx(45.0)
    assert decoded[0].width_px == pytest.approx(10.0)


def test_peak_extraction_returns_empty_for_flat_threshold_map() -> None:
    quality = np.zeros((3, 3), dtype=np.float32)
    quality[1, 1] = 0.9
    quality[1, 2] = 0.8

    assert (
        extract_quality_peaks(np.zeros((3, 3)), threshold=0.0, min_distance_px=0) == []
    )
    assert extract_quality_peaks(
        quality, threshold=0.0, min_distance_px=0, max_peaks=2
    ) == [(1, 1, pytest.approx(0.9)), (1, 2, pytest.approx(0.8))]


def test_nms_uses_center_angle_width_and_rotated_iou() -> None:
    candidates = [
        _grasp(50, 50, angle=0, width=60, score=0.9, candidate_id="best"),
        _grasp(52, 51, angle=3, width=62, score=0.8, candidate_id="duplicate"),
        _grasp(50, 50, angle=70, width=60, score=0.7, candidate_id="different-angle"),
        _grasp(140, 140, angle=0, width=60, score=0.6, candidate_id="far"),
    ]

    kept = non_maximum_suppression(candidates, NMSConfig())

    assert [item.candidate_id for item in kept] == ["best", "different-angle", "far"]


def test_candidate_ids_are_deterministic_and_score_independent() -> None:
    kwargs = dict(
        sample_id="sample-7",
        backend="grconvnet",
        center_x=12.5,
        center_y=30.25,
        angle_deg=179.0,
        width_px=42.0,
        source_key="peak:3:2",
    )

    first = stable_candidate_id(**kwargs)
    second = stable_candidate_id(**kwargs)
    equivalent_angle = stable_candidate_id(**{**kwargs, "angle_deg": -1.0})

    assert first == second == equivalent_angle
    assert first.startswith("c_") and len(first) == 22


def test_same_gt_must_satisfy_iou_and_angle_jointly() -> None:
    prediction = _grasp(100, 100, angle=0, width=80, candidate_id="joint-check")
    gt_high_iou_wrong_angle = _ocid_corners(100, 100, angle=31, width=80)
    gt_right_angle_no_iou = _ocid_corners(250, 200, angle=0, width=80)

    result = evaluate_ocid_predictions(
        [prediction],
        [gt_high_iou_wrong_angle, gt_right_angle_no_iou],
        EvaluatorConfig(image_shape=(240, 320)),
    )

    assert result.candidates[0].pairwise[0].iou_ok
    assert not result.candidates[0].pairwise[0].angle_ok
    assert not result.candidates[0].pairwise[1].iou_ok
    assert result.candidates[0].pairwise[1].angle_ok
    assert not result.j_at_1
    assert not result.j_at_5


def test_angle_threshold_is_inclusive_but_iou_threshold_is_strict() -> None:
    prediction = _grasp(100, 100, angle=0, width=80)
    gt_at_angle_boundary = _ocid_corners(100, 100, angle=30, width=80)

    inclusive_angle = evaluate_ocid_predictions(
        [prediction],
        [gt_at_angle_boundary],
        EvaluatorConfig(image_shape=(240, 320), iou_threshold=0.0),
    )
    strict_iou = evaluate_ocid_predictions(
        [prediction],
        [_ocid_corners(100, 100, angle=0, width=80)],
        EvaluatorConfig(image_shape=(240, 320), iou_threshold=1.0),
    )

    assert inclusive_angle.j_at_1
    assert inclusive_angle.candidates[0].pairwise[
        0
    ].angle_difference_deg == pytest.approx(30)
    assert not strict_iou.j_at_1


def test_top5_and_empty_predictions_do_not_pad_or_crash() -> None:
    candidates = [
        _grasp(1, 1, score=0.2, candidate_id="low"),
        _grasp(2, 2, score=0.9, candidate_id="high"),
    ]
    serialized = serialize_top5(candidates)
    prediction = build_prediction(
        sample_id="s",
        backend="b",
        conditioning_variant="standard",
        raw_candidates=candidates,
        nms_candidates=candidates,
    )
    empty = build_prediction(
        sample_id="empty",
        backend="b",
        conditioning_variant="standard",
        raw_candidates=[],
        nms_candidates=[],
    )
    evaluation = evaluate_ocid_predictions([], [_ocid_corners(20, 20, 0, 20)])

    assert [item["candidate_id"] for item in serialized] == ["high", "low"]
    assert [item["rank"] for item in serialized] == [1, 2]
    assert len(prediction.top5) == 2 and prediction.top1.candidate_id == "high"
    assert empty.top1 is None and empty.top5 == ()
    assert empty.empty_reason == "no_candidate_generated"
    assert evaluation.empty_prediction
    assert not evaluation.j_at_1 and not evaluation.j_at_5
    assert evaluation.first_valid_rank is None and evaluation.reciprocal_rank == 0.0
