from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.grasping.geometric_ranker import make_ocid_vlg_evaluation_rectangles
from src.grasping.reranking_v1.baseline_audit import classify_funnel_stage
from src.grasping.reranking_v1.identity import (
    assert_candidate_identity_invariant,
    candidate_identity_sha256,
    stable_sample_id,
)
from src.grasping.reranking_v1.labels import (
    evaluate_candidate_label,
    is_positive_pair,
    periodic_angle_difference_deg,
    polygon_iou,
)


def test_full_repeatedfilm_baseline_regression_artifact() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "runs/modular_reranking_repeatedfilm_v1_20260729_203147"
        / "reports/baseline_test_audit_v2"
    )
    audit_path = root / "repeatedfilm_baseline_oracle.json"
    assert audit_path.is_file()
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == (
        "e387ea541db9196ced454123e9d9f398edde77e2a070f48bc0a2bffc454f866d"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["counts"] == {
        "empty_masks": 1,
        "empty_nms_samples": 108,
        "execution_failures": 0,
        "finite_q_values": 187077,
        "mask_validated_candidates": 1449011,
        "nms_candidates": 187077,
        "raw_candidates": 1466046,
        "samples": 7675,
        "scenes": 325,
        "scored_candidates": 187077,
    }
    assert audit["lineage"] == "hierarchical_repeated_film_only"
    assert audit["single_film_used"] is False
    assert audit["retained_anchor_check"]["all_equal"] is True
    assert audit["metrics"]["j_at_1"]["numerator"] == 2934
    assert audit["metrics"]["recall_at_5"]["numerator"] == 4655
    assert audit["metrics"]["recall_at_10"]["numerator"] == 5201
    assert audit["metrics"]["j_at_any"]["numerator"] == 5799
    assert audit["funnel"] == {
        "already_correct": 2934,
        "generation_limited": 1553,
        "mask_filter_loss": 37,
        "nms_loss": 178,
        "ranking_loss_beyond_top5": 1144,
        "ranking_loss_top5": 1721,
        "valid_empty": 108,
    }
    forbidden = {
        "candidate_success",
        "candidate_positive",
        "best_gt_id",
        "best_gt_index",
        "rectangle_iou",
        "angle_difference_deg",
        "iou_ok",
        "angle_ok",
        "joint_success",
        "failure_mode",
        "evaluator_version",
    }
    deployment_columns = set(
        pq.read_schema(root / "test_deployment_candidates.parquet").names
    )
    assert deployment_columns.isdisjoint(forbidden)


def _candidate(candidate_id: str = "g0001") -> dict:
    return {
        "sample_id": "sample",
        "candidate_id": candidate_id,
        "center_uv": [500.0, 240.0],
        "center_depth_m": 1.0,
        "center_camera_xyz_m": [0.1, 0.0, 1.0],
        "angle_rad": 0.0,
        "width_m": 0.05,
        "width_px": 30.0,
        "contact_points_uv": [[485.0, 240.0], [515.0, 240.0]],
        "endpoints_uv": [[485.0, 240.0], [515.0, 240.0]],
        "T_camera_grasp_fixed_approach": [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def test_strict_iou_boundary_and_angle_boundary() -> None:
    assert not is_positive_pair(0.25, 0.0)
    assert is_positive_pair(math.nextafter(0.25, 1.0), 30.0)
    assert not is_positive_pair(1.0, math.nextafter(30.0, 31.0))


def test_parallel_jaw_angle_is_180_degree_periodic() -> None:
    assert periodic_angle_difference_deg(0.0, math.pi) == pytest.approx(0.0)
    assert periodic_angle_difference_deg(
        math.radians(179.0), math.radians(1.0)
    ) == pytest.approx(2.0)


def test_candidate_label_never_mixes_iou_and_angle_from_different_gt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    far = [[30.0, 0.0], [40.0, 0.0], [40.0, 10.0], [30.0, 10.0]]
    monkeypatch.setattr(
        "src.grasping.reranking_v1.labels.make_candidate_evaluation_rectangle",
        lambda *_: {
            "center_uv": np.asarray([100.0, 100.0]),
            "width_px": 40.0,
            "height_px": 20.0,
            "angle_rad": 0.0,
            "polygon": square,
        },
    )
    monkeypatch.setattr(
        "src.grasping.reranking_v1.labels.make_ocid_vlg_evaluation_rectangles",
        lambda *_: [
            {
                "center_uv": np.asarray([100.0, 100.0]),
                "width_px": 40.0,
                "height_px": 20.0,
                "angle_rad": math.radians(60.0),
                "polygon": square,
            },
            {
                "center_uv": np.asarray([200.0, 100.0]),
                "width_px": 40.0,
                "height_px": 20.0,
                "angle_rad": 0.0,
                "polygon": far,
            },
        ],
    )
    label = evaluate_candidate_label(
        _candidate(),
        [square, far],
        {"iou_threshold": 0.25, "angle_threshold_deg": 30.0},
    )
    assert label.candidate_positive is False
    assert label.maximum_rectangle_iou_with_angle_gate == pytest.approx(0.0)


def _evaluation_config() -> dict:
    return {
        "iou_threshold": 0.25,
        "angle_threshold_deg": 30.0,
        "predicted_rectangle_height_px": 20.0,
        "ground_truth_rectangle_height_px": 20.0,
        "ground_truth_width_clip_px": 100.0,
    }


def _canonical_grasp_metrics():
    path = (
        Path(__file__).resolve().parents[2]
        / "crog_reproduction/CROG/utils/grasp_metrics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_canonical_crog_grasp_metrics_for_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("center", "contacts", "gt", "angle_rad"),
    [
        (
            [550.0, 100.0],
            [[520.0, 100.0], [580.0, 100.0]],
            [[520.0, 90.0], [520.0, 110.0], [580.0, 110.0], [580.0, 90.0]],
            0.0,
        ),
        (
            [320.0, 240.0],
            [[290.0, 240.0], [350.0, 240.0]],
            [[300.0, 230.0], [300.0, 250.0], [360.0, 250.0], [360.0, 230.0]],
            0.0,
        ),
    ],
)
def test_raster_label_matches_canonical_crog_evaluator(
    center: list[float],
    contacts: list[list[float]],
    gt: list[list[float]],
    angle_rad: float,
) -> None:
    candidate = _candidate()
    candidate["center_uv"] = center
    candidate["contact_points_uv"] = contacts
    candidate["width_px"] = 999.0
    label = evaluate_candidate_label(candidate, [gt], _evaluation_config())

    canonical = _canonical_grasp_metrics()
    gt_geometry = make_ocid_vlg_evaluation_rectangles([gt], _evaluation_config())[0]
    contact_array = np.asarray(contacts, dtype=np.float64)
    predicted_width = float(np.linalg.norm(contact_array[1] - contact_array[0]))
    canonical_iou = canonical.rectangle_iou(
        [
            center[0],
            center[1],
            predicted_width,
            20.0,
            math.degrees(angle_rad),
        ],
        [
            gt_geometry["center_uv"][0],
            gt_geometry["center_uv"][1],
            gt_geometry["width_px"],
            gt_geometry["height_px"],
            math.degrees(gt_geometry["angle_rad"]),
        ],
    )
    canonical_angle = canonical.periodic_angle_difference_deg(
        math.degrees(angle_rad),
        math.degrees(gt_geometry["angle_rad"]),
    )
    assert label.candidate_gt_iou == pytest.approx(canonical_iou)
    assert label.candidate_gt_angle_error_deg == pytest.approx(canonical_angle)
    assert label.candidate_positive is canonical.joint_success(
        canonical_iou, canonical_angle
    )


def test_candidate_metric_uses_contact_span_not_configured_width() -> None:
    gt = [
        [520.0, 90.0],
        [520.0, 110.0],
        [580.0, 110.0],
        [580.0, 90.0],
    ]
    candidate = _candidate()
    candidate["center_uv"] = [550.0, 100.0]
    candidate["contact_points_uv"] = [[520.0, 100.0], [580.0, 100.0]]
    candidate["width_px"] = 600.0
    label = evaluate_candidate_label(candidate, [gt], _evaluation_config())
    assert label.candidate_positive is True
    assert label.candidate_gt_iou == pytest.approx(1.0)


def test_polygon_coordinates_are_xy_unclipped_and_empty_gt_fails() -> None:
    polygon = [
        [500.0, 20.0],
        [620.0, 20.0],
        [620.0, 40.0],
        [500.0, 40.0],
    ]
    assert polygon_iou(polygon, polygon) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least one GT"):
        evaluate_candidate_label(
            _candidate(),
            [],
            {"iou_threshold": 0.25, "angle_threshold_deg": 30.0},
        )


def test_gt_rectangle_construction_preserves_xy_and_clips_only_width() -> None:
    rectangles = make_ocid_vlg_evaluation_rectangles(
        [
            [
                [500.0, 100.0],
                [500.0, 180.0],
                [620.0, 180.0],
                [650.0, 100.0],
            ]
        ],
        {
            "ground_truth_rectangle_height_px": 20.0,
            "ground_truth_width_clip_px": 100.0,
        },
    )
    rectangle = rectangles[0]
    assert rectangle["center_uv"].tolist() == [560.0, 140.0]
    assert rectangle["width_px"] == pytest.approx(100.0)
    assert rectangle["angle_rad"] == pytest.approx(0.0)
    polygon = rectangle["polygon"]
    assert polygon[:, 0].min() == pytest.approx(510.0)
    assert polygon[:, 0].max() == pytest.approx(610.0)
    assert polygon[:, 1].min() == pytest.approx(130.0)
    assert polygon[:, 1].max() == pytest.approx(150.0)


def test_stable_sample_id_matches_frozen_first_test_sample() -> None:
    assert (
        stable_sample_id(
            "ARID10/floor/top/non-fruits/seq09,"
            "result_2018-08-27-16-13-28.png",
            0,
        )
        == "q0000000_b32eb3299dcd3ae9"
    )


def test_identity_allows_only_permutation() -> None:
    left = _candidate("g0001")
    right = _candidate("g0002")
    assert_candidate_identity_invariant([left, right], [right, left])
    assert candidate_identity_sha256(left) == candidate_identity_sha256(
        copy.deepcopy(left)
    )


def test_identity_rejects_pose_change() -> None:
    before = _candidate()
    after = copy.deepcopy(before)
    after["center_uv"][0] += 1.0
    with pytest.raises(AssertionError, match="candidate pose changed"):
        assert_candidate_identity_invariant([before], [after])


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "valid_empty": True,
                "raw_oracle": False,
                "mask_validated_oracle": False,
                "nms_oracle": False,
                "gqcnn_top1": False,
                "gqcnn_top5": False,
            },
            "valid_empty",
        ),
        (
            {
                "valid_empty": False,
                "raw_oracle": False,
                "mask_validated_oracle": False,
                "nms_oracle": False,
                "gqcnn_top1": False,
                "gqcnn_top5": False,
            },
            "generation_limited",
        ),
        (
            {
                "valid_empty": False,
                "raw_oracle": True,
                "mask_validated_oracle": False,
                "nms_oracle": False,
                "gqcnn_top1": False,
                "gqcnn_top5": False,
            },
            "mask_filter_loss",
        ),
        (
            {
                "valid_empty": False,
                "raw_oracle": True,
                "mask_validated_oracle": True,
                "nms_oracle": False,
                "gqcnn_top1": False,
                "gqcnn_top5": False,
            },
            "nms_loss",
        ),
        (
            {
                "valid_empty": False,
                "raw_oracle": True,
                "mask_validated_oracle": True,
                "nms_oracle": True,
                "gqcnn_top1": False,
                "gqcnn_top5": False,
            },
            "ranking_loss_beyond_top5",
        ),
        (
            {
                "valid_empty": False,
                "raw_oracle": True,
                "mask_validated_oracle": True,
                "nms_oracle": True,
                "gqcnn_top1": False,
                "gqcnn_top5": True,
            },
            "ranking_loss_top5",
        ),
        (
            {
                "valid_empty": False,
                "raw_oracle": True,
                "mask_validated_oracle": True,
                "nms_oracle": True,
                "gqcnn_top1": True,
                "gqcnn_top5": True,
            },
            "already_correct",
        ),
    ],
)
def test_funnel_categories_are_ordered_and_exclusive(
    kwargs: dict, expected: str
) -> None:
    assert classify_funnel_stage(**kwargs) == expected
