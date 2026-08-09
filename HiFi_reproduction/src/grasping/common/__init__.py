"""Unified 4-DoF grasp protocol shared by all modular backends."""

from .candidate_decoder import (
    NMSConfig,
    build_prediction,
    decode_quality_maps,
    extract_quality_peaks,
    non_maximum_suppression,
    rank_candidates,
    serialize_top5,
    stable_candidate_id,
)
from .evaluator import (
    CandidateEvaluation,
    EvaluatorConfig,
    PairwiseMatch,
    SampleEvaluation,
    evaluate_candidate,
    evaluate_ocid_predictions,
    grasp_from_ocid_corners,
)
from .geometry import (
    CropTransform,
    normalize_angle_deg,
    periodic_angle_difference_deg,
    rasterized_rectangle_iou,
    rectangle_corners,
    rotated_rectangle_iou,
    transform_oriented_length,
)
from .types import Grasp4DoF, GraspPrediction

__all__ = [
    "CandidateEvaluation",
    "CropTransform",
    "EvaluatorConfig",
    "Grasp4DoF",
    "GraspPrediction",
    "NMSConfig",
    "PairwiseMatch",
    "SampleEvaluation",
    "build_prediction",
    "decode_quality_maps",
    "evaluate_candidate",
    "evaluate_ocid_predictions",
    "extract_quality_peaks",
    "grasp_from_ocid_corners",
    "non_maximum_suppression",
    "normalize_angle_deg",
    "periodic_angle_difference_deg",
    "rank_candidates",
    "rasterized_rectangle_iou",
    "rectangle_corners",
    "rotated_rectangle_iou",
    "serialize_top5",
    "stable_candidate_id",
    "transform_oriented_length",
]
