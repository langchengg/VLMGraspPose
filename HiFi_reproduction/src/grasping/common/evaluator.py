"""Corrected OCID-VLG offline 2D grasp-rectangle evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .candidate_decoder import rank_candidates
from .geometry import normalize_angle_deg, periodic_angle_difference_deg
from .geometry import rasterized_rectangle_iou
from .types import Grasp4DoF


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    image_shape: tuple[int, int] = (480, 640)
    fixed_height_px: float = 20.0
    gt_width_clip_px: float = 100.0
    iou_threshold: float = 0.25
    angle_threshold_deg: float = 30.0

    def __post_init__(self) -> None:
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("image_shape must be positive (height, width)")
        if self.fixed_height_px <= 0.0 or self.gt_width_clip_px <= 0.0:
            raise ValueError("rectangle dimensions must be positive")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0.0 <= self.angle_threshold_deg <= 90.0:
            raise ValueError("angle threshold must be in [0, 90]")


@dataclass(frozen=True, slots=True)
class PairwiseMatch:
    gt_index: int
    rectangle_iou: float
    angle_difference_deg: float
    iou_ok: bool
    angle_ok: bool
    joint_success: bool


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_id: str
    candidate_success: bool
    best_gt_index: int | None
    best_rectangle_iou: float | None
    best_angle_difference_deg: float | None
    pairwise: tuple[PairwiseMatch, ...]


@dataclass(frozen=True, slots=True)
class SampleEvaluation:
    sample_count: int
    candidate_count: int
    j_at_1: bool
    j_at_5: bool
    first_valid_rank: int | None
    reciprocal_rank: float
    empty_prediction: bool
    candidates: tuple[CandidateEvaluation, ...]


def grasp_from_ocid_corners(
    corners: Sequence[Sequence[float]],
    *,
    fixed_height_px: float = 20.0,
    width_clip_px: float = 100.0,
) -> Grasp4DoF:
    """Apply the frozen OCID convention: axis is ``corner[3]-corner[0]``."""

    array = np.asarray(corners, dtype=np.float64)
    if array.shape != (4, 2) or not np.all(np.isfinite(array)):
        raise ValueError("OCID grasp corners must be finite with shape (4, 2)")
    center = (array[0] + array[2]) * 0.5
    opening = array[3] - array[0]
    width = min(float(np.linalg.norm(opening)), float(width_clip_px))
    if width <= 0.0:
        raise ValueError("OCID grasp width must be positive")
    angle = normalize_angle_deg(math.degrees(math.atan2(opening[1], opening[0])))
    return Grasp4DoF(
        center_x=float(center[0]),
        center_y=float(center[1]),
        angle_deg=angle,
        width_px=width,
        height_px=fixed_height_px,
        score=0.0,
    )


def _fixed_height(candidate: Grasp4DoF, height_px: float) -> Grasp4DoF:
    return Grasp4DoF(
        center_x=candidate.center_x,
        center_y=candidate.center_y,
        angle_deg=normalize_angle_deg(candidate.angle_deg),
        width_px=candidate.width_px,
        height_px=height_px,
        score=candidate.score,
        candidate_id=candidate.candidate_id,
        metadata=candidate.metadata,
    )


def evaluate_candidate(
    candidate: Grasp4DoF,
    ground_truth: Sequence[Grasp4DoF],
    config: EvaluatorConfig = EvaluatorConfig(),
) -> CandidateEvaluation:
    prediction = _fixed_height(candidate, config.fixed_height_px)
    pairwise: list[PairwiseMatch] = []
    for gt_index, gt in enumerate(ground_truth):
        normalized_gt = _fixed_height(gt, config.fixed_height_px)
        iou = rasterized_rectangle_iou(
            prediction, normalized_gt, shape=config.image_shape
        )
        angle_difference = periodic_angle_difference_deg(
            prediction.angle_deg, normalized_gt.angle_deg
        )
        iou_ok = iou > config.iou_threshold
        angle_ok = angle_difference <= config.angle_threshold_deg
        pairwise.append(
            PairwiseMatch(
                gt_index=gt_index,
                rectangle_iou=iou,
                angle_difference_deg=angle_difference,
                iou_ok=iou_ok,
                angle_ok=angle_ok,
                joint_success=bool(iou_ok and angle_ok),
            )
        )
    successes = [item for item in pairwise if item.joint_success]
    selectable = successes or pairwise
    best = min(
        selectable,
        key=lambda item: (-item.rectangle_iou, item.angle_difference_deg, item.gt_index),
        default=None,
    )
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        candidate_success=bool(successes),
        best_gt_index=None if best is None else best.gt_index,
        best_rectangle_iou=None if best is None else best.rectangle_iou,
        best_angle_difference_deg=None if best is None else best.angle_difference_deg,
        pairwise=tuple(pairwise),
    )


def evaluate_ocid_predictions(
    candidates: Sequence[Grasp4DoF],
    gt_corners: Sequence[Sequence[Sequence[float]]],
    config: EvaluatorConfig = EvaluatorConfig(),
) -> SampleEvaluation:
    """Evaluate ranked candidates; empty and fewer-than-five inputs are valid."""

    ground_truth = tuple(
        grasp_from_ocid_corners(
            corners,
            fixed_height_px=config.fixed_height_px,
            width_clip_px=config.gt_width_clip_px,
        )
        for corners in gt_corners
    )
    ranked = rank_candidates(candidates)
    evaluations = tuple(evaluate_candidate(item, ground_truth, config) for item in ranked)
    first_rank = next(
        (index for index, item in enumerate(evaluations, 1) if item.candidate_success),
        None,
    )
    return SampleEvaluation(
        sample_count=1,
        candidate_count=len(ranked),
        j_at_1=bool(evaluations and evaluations[0].candidate_success),
        j_at_5=any(item.candidate_success for item in evaluations[:5]),
        first_valid_rank=first_rank,
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        empty_prediction=not evaluations,
        candidates=evaluations,
    )
