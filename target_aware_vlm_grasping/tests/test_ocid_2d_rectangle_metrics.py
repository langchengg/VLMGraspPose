from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import rectangle_iou, topk_2d_grasp_rectangle_match_rate


def test_rectangle_iou_returns_one_for_identical_rectangles() -> None:
    rect = [[10, 10], [30, 10], [30, 20], [10, 20]]

    assert rectangle_iou(rect, rect) == 1.0


def test_topk_rectangle_match_uses_iou_and_angle_thresholds() -> None:
    good = [[10, 10], [30, 10], [30, 20], [10, 20]]
    bad = [[60, 60], [80, 60], [80, 70], [60, 70]]
    records = [{
        "best_grasp_rectangle_2d": bad,
        "top_k_grasp_rectangles_2d": [bad, good],
        "gt_grasp_rectangles": [good],
    }]

    assert topk_2d_grasp_rectangle_match_rate(
        records,
        k=1,
        iou_threshold=0.25,
        angle_threshold_deg=30.0,
    ) == 0.0
    assert topk_2d_grasp_rectangle_match_rate(
        records,
        k=2,
        iou_threshold=0.25,
        angle_threshold_deg=30.0,
    ) == 1.0

