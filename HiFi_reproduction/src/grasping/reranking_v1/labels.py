"""Strict OCID-VLG rectangle labels for frozen candidate pools."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from skimage.draw import polygon as draw_polygon

from src.grasping.geometric_ranker import (
    make_candidate_evaluation_rectangle,
    make_ocid_vlg_evaluation_rectangles,
)


@dataclass(frozen=True)
class CandidateLabel:
    """GT-only label payload, kept outside the inference feature table."""

    candidate_positive: bool
    best_gt_id: int | None
    candidate_gt_iou: float | None
    candidate_gt_angle_error_deg: float | None
    maximum_rectangle_iou_with_angle_gate: float
    legacy_geq_positive: bool
    exact_iou_threshold_pair_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def periodic_angle_difference_deg(first_rad: float, second_rad: float) -> float:
    """Smallest parallel-jaw angle difference with 180-degree periodicity."""

    delta = abs(
        (float(first_rad) - float(second_rad) + math.pi / 2.0) % math.pi
        - math.pi / 2.0
    )
    return float(math.degrees(delta))


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Analytic IoU for candidate-relation features, not the GT metric."""

    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    if left.shape != (4, 2) or right.shape != (4, 2):
        raise ValueError("rectangle polygons must have shape (4, 2)")
    area_left = float(abs(cv2.contourArea(left)))
    area_right = float(abs(cv2.contourArea(right)))
    intersection, _ = cv2.intersectConvexConvex(left, right)
    union = area_left + area_right - float(intersection)
    return 0.0 if union <= 0.0 else float(intersection / union)


def _raster_pixels(
    *,
    center_uv: Sequence[float],
    width_px: float,
    height_px: float,
    angle_rad: float,
    shape: tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Rasterize the frozen corrected_geometric_v2 rectangle convention."""

    center = np.asarray(center_uv, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("rectangle center must be a finite length-2 vector")
    values = np.asarray(
        [width_px, height_px, angle_rad], dtype=np.float64
    )
    if not np.all(np.isfinite(values)) or width_px <= 0 or height_px <= 0:
        raise ValueError("rectangle width/height/angle must be finite and positive")
    vertices = np.asarray(
        cv2.boxPoints(
            (
                (float(center[0]), float(center[1])),
                (float(width_px), float(height_px)),
                -float(math.degrees(angle_rad)),
            )
        ),
        dtype=np.intp,
    )
    rows, columns = draw_polygon(
        vertices[:, 1], vertices[:, 0], shape=shape
    )
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(
        rows.astype(np.int64) * int(shape[1])
        + columns.astype(np.int64)
    )


def _raster_pixel_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 and second.size == 0:
        return 0.0
    intersection = np.intersect1d(
        first, second, assume_unique=True
    ).size
    union = int(first.size + second.size - intersection)
    return float(intersection / union) if union else 0.0


def is_positive_pair(
    iou: float,
    angle_error_deg: float,
    *,
    iou_threshold: float = 0.25,
    angle_threshold_deg: float = 30.0,
) -> bool:
    """Apply the preregistered strict same-GT predicate."""

    return bool(
        float(angle_error_deg) <= float(angle_threshold_deg)
        and float(iou) > float(iou_threshold)
    )


def evaluate_candidate_label(
    record: Mapping[str, Any],
    grasp_rectangles: Sequence[Sequence[Sequence[float]]],
    evaluation_config: Mapping[str, Any],
) -> CandidateLabel:
    """Label with frozen 480x640 raster IoU and same-GT joint matching."""

    if not grasp_rectangles:
        raise ValueError("at least one GT grasp rectangle is required")
    candidate = dict(record)
    candidate.setdefault(
        "center_uv",
        [candidate.get("center_u_px"), candidate.get("center_v_px")],
    )
    predicted = make_candidate_evaluation_rectangle(candidate, evaluation_config)
    ground_truth = make_ocid_vlg_evaluation_rectangles(
        grasp_rectangles, evaluation_config
    )
    angle_threshold = float(evaluation_config["angle_threshold_deg"])
    iou_threshold = float(evaluation_config["iou_threshold"])

    predicted_pixels = _raster_pixels(
        center_uv=predicted["center_uv"],
        width_px=predicted["width_px"],
        height_px=predicted["height_px"],
        angle_rad=predicted["angle_rad"],
    )
    comparisons: list[dict[str, Any]] = []
    exact_count = 0
    for gt_id, target in enumerate(ground_truth):
        angle_error = periodic_angle_difference_deg(
            predicted["angle_rad"], target["angle_rad"]
        )
        target_pixels = _raster_pixels(
            center_uv=target["center_uv"],
            width_px=target["width_px"],
            height_px=target["height_px"],
            angle_rad=target["angle_rad"],
        )
        iou = _raster_pixel_iou(predicted_pixels, target_pixels)
        angle_eligible = angle_error <= angle_threshold
        strict_positive = is_positive_pair(
            iou,
            angle_error,
            iou_threshold=iou_threshold,
            angle_threshold_deg=angle_threshold,
        )
        legacy_positive = angle_eligible and iou >= iou_threshold
        if angle_eligible and math.isclose(
            iou, iou_threshold, rel_tol=0.0, abs_tol=1e-12
        ):
            exact_count += 1
        comparisons.append(
            {
                "gt_id": gt_id,
                "angle_error": angle_error,
                "iou": iou,
                "strict_positive": strict_positive,
                "legacy_positive": legacy_positive,
                "angle_eligible": angle_eligible,
            }
        )

    # Match corrected_geometric_v2: prefer a successful same-GT pair, then
    # highest raster IoU, lowest periodic angle error, and stable GT index.
    successes = [item for item in comparisons if item["strict_positive"]]
    best = min(
        successes or comparisons,
        key=lambda item: (
            -float(item["iou"]),
            float(item["angle_error"]),
            int(item["gt_id"]),
        ),
    )
    eligible_ious = [
        float(item["iou"]) if item["angle_eligible"] else 0.0
        for item in comparisons
    ]
    return CandidateLabel(
        candidate_positive=any(item["strict_positive"] for item in comparisons),
        best_gt_id=int(best["gt_id"]),
        candidate_gt_iou=float(best["iou"]),
        candidate_gt_angle_error_deg=float(best["angle_error"]),
        maximum_rectangle_iou_with_angle_gate=max(eligible_ious, default=0.0),
        legacy_geq_positive=any(item["legacy_positive"] for item in comparisons),
        exact_iou_threshold_pair_count=exact_count,
    )
