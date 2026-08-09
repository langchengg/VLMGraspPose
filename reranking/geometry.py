"""Thin adapter over the repository's canonical HiFi grasp evaluator.

No geometry formula lives here.  Every predicate, rasterization, IoU and
candidate-label decision delegates directly to
``HiFi_reproduction/src/grasping/reranking_v1/labels.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from reranking.data_contracts import HIFI_CANONICAL_LABELS, HIFI_ROOT


# The canonical module uses the package name ``src`` with HiFi_reproduction as
# its import root.  Add that repository-local root to this process only; no
# filesystem or source artifact is modified.
_hifi_import_root = str(HIFI_ROOT)
if _hifi_import_root not in sys.path:
    sys.path.insert(0, _hifi_import_root)

from src.grasping.reranking_v1 import labels as _canonical  # noqa: E402


CANONICAL_EVALUATOR_PATH = HIFI_CANONICAL_LABELS
CandidateLabel = _canonical.CandidateLabel


def canonical_evaluator_path() -> Path:
    """Return the exact local source file used for every geometry decision."""

    return CANONICAL_EVALUATOR_PATH


def is_positive_pair(
    iou: float,
    angle_error_deg: float,
    *,
    iou_threshold: float = 0.25,
    angle_threshold_deg: float = 30.0,
) -> bool:
    return _canonical.is_positive_pair(
        iou,
        angle_error_deg,
        iou_threshold=iou_threshold,
        angle_threshold_deg=angle_threshold_deg,
    )


def periodic_angle_difference_deg(first_rad: float, second_rad: float) -> float:
    return _canonical.periodic_angle_difference_deg(first_rad, second_rad)


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    return _canonical.polygon_iou(first, second)


def raster_pixels(
    *,
    center_uv: Sequence[float],
    width_px: float,
    height_px: float,
    angle_rad: float,
    shape: tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Delegate canonical row=y/column=x rectangle rasterization."""

    return _canonical._raster_pixels(
        center_uv=center_uv,
        width_px=width_px,
        height_px=height_px,
        angle_rad=angle_rad,
        shape=shape,
    )


def raster_pixel_iou(first: np.ndarray, second: np.ndarray) -> float:
    return _canonical._raster_pixel_iou(first, second)


def evaluate_candidate_label(
    record: Mapping[str, Any],
    grasp_rectangles: Sequence[Sequence[Sequence[float]]],
    evaluation_config: Mapping[str, Any],
) -> CandidateLabel:
    return _canonical.evaluate_candidate_label(
        record, grasp_rectangles, evaluation_config
    )


__all__ = [
    "CANONICAL_EVALUATOR_PATH",
    "CandidateLabel",
    "canonical_evaluator_path",
    "evaluate_candidate_label",
    "is_positive_pair",
    "periodic_angle_difference_deg",
    "polygon_iou",
    "raster_pixel_iou",
    "raster_pixels",
]
