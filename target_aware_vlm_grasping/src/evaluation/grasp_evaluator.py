from __future__ import annotations

from evaluation.evaluator import OutputEvaluator
from evaluation.metrics import (
    grasp_rectangle_matches,
    rectangle_angle_difference_deg,
    rectangle_iou,
    topk_2d_grasp_center_hit_rate,
    topk_2d_grasp_rectangle_match_rate,
)


class Grasp2DEvaluator(OutputEvaluator):
    def evaluate_records(self, records: list[dict], mode: str = "ocid_2d") -> dict:
        return super().evaluate_records(records, mode="ocid_2d")


__all__ = [
    "Grasp2DEvaluator",
    "grasp_rectangle_matches",
    "rectangle_angle_difference_deg",
    "rectangle_iou",
    "topk_2d_grasp_center_hit_rate",
    "topk_2d_grasp_rectangle_match_rate",
]
