"""Frozen canonical geometry and unified OCID-VLG evaluator.

Prediction rectangle geometry is never rewritten here.  OCID-VLG GT corners are
converted to the corrected CROG convention (jaw axis corner[3]-corner[0]), with
the historical 100 px jaw-width clip and a 20 px evaluation height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon as draw_polygon


IMAGE_SHAPE = (480, 640)
IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEG = 30.0
GT_HEIGHT_PX = 20.0
GT_WIDTH_CLIP_PX = 100.0


def normalize_angle_deg(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    result = (value + 90.0) % 180.0 - 90.0
    return 0.0 if result == 0.0 else float(result)


def periodic_angle_error_deg(first: float, second: float) -> float:
    return abs(normalize_angle_deg(float(first) - float(second)))


@dataclass(frozen=True, slots=True)
class CanonicalGrasp:
    cx_px: float
    cy_px: float
    theta_deg: float
    jaw_width_px: float
    rectangle_height_px: float
    native_score: float = 0.0
    native_rank: int = 1
    source_method: str = ""
    sample_id: str = ""
    status: str = "ok"

    def __post_init__(self) -> None:
        finite = (
            self.cx_px,
            self.cy_px,
            self.theta_deg,
            self.jaw_width_px,
            self.rectangle_height_px,
            self.native_score,
        )
        if not all(math.isfinite(float(item)) for item in finite):
            raise ValueError("canonical grasp values must be finite")
        if self.jaw_width_px <= 0.0 or self.rectangle_height_px <= 0.0:
            raise ValueError("canonical rectangle dimensions must be positive")
        if self.native_rank <= 0:
            raise ValueError("native_rank must be positive")
        object.__setattr__(self, "theta_deg", normalize_angle_deg(self.theta_deg))


def corners(grasp: CanonicalGrasp) -> np.ndarray:
    # CROG/OCID angles are stored in image coordinates and passed to OpenCV
    # with a negative sign.  This exact convention is also what the upstream
    # row/column Grasp representation produces after conversion to x/y.
    return cv2.boxPoints(
        (
            (float(grasp.cx_px), float(grasp.cy_px)),
            (float(grasp.jaw_width_px), float(grasp.rectangle_height_px)),
            -float(grasp.theta_deg),
        )
    ).astype(np.float64)


def _pixels(grasp: CanonicalGrasp, shape: tuple[int, int] = IMAGE_SHAPE) -> np.ndarray:
    height, width = map(int, shape)
    if height <= 0 or width <= 0:
        raise ValueError("image shape must be positive")
    box = corners(grasp).astype(np.intp)
    rows, columns = draw_polygon(box[:, 1], box[:, 0], shape=(height, width))
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(rows.astype(np.int64) * width + columns.astype(np.int64))


def raster_iou(
    first: CanonicalGrasp,
    second: CanonicalGrasp,
    shape: tuple[int, int] = IMAGE_SHAPE,
) -> float:
    """Corrected CROG-style, clipped, rasterized rectangle IoU."""

    a, b = _pixels(first, shape), _pixels(second, shape)
    if a.size == 0 and b.size == 0:
        return 0.0
    intersection = np.intersect1d(a, b, assume_unique=True).size
    union = int(a.size + b.size - intersection)
    return 0.0 if union <= 0 else float(intersection / union)


def continuous_iou_cv2(first: CanonicalGrasp, second: CanonicalGrasp) -> float:
    a, b = corners(first).astype(np.float32), corners(second).astype(np.float32)
    area_a, area_b = abs(cv2.contourArea(a)), abs(cv2.contourArea(b))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = float(area_a + area_b - intersection)
    return 0.0 if union <= 0 else float(intersection / union)


def continuous_iou_shapely(first: CanonicalGrasp, second: CanonicalGrasp) -> float:
    a, b = Polygon(corners(first)), Polygon(corners(second))
    union = a.union(b).area
    return 0.0 if union <= 0 else float(a.intersection(b).area / union)


def gt_from_corners(values: Sequence[Sequence[float]]) -> CanonicalGrasp:
    value = np.asarray(values, dtype=np.float64)
    if value.shape != (4, 2) or not np.all(np.isfinite(value)):
        raise ValueError("GT corners must be finite 4x2 x/y coordinates")
    centre = 0.5 * (value[0] + value[2])
    jaw = value[3] - value[0]
    jaw_width = min(float(np.linalg.norm(jaw)), GT_WIDTH_CLIP_PX)
    if jaw_width <= 0:
        raise ValueError("GT jaw width must be positive")
    raw = math.degrees(math.atan2(float(jaw[0]), float(jaw[1])))
    official_angle = raw - 90.0 if raw > 0.0 else raw + 90.0
    return CanonicalGrasp(
        cx_px=float(centre[0]),
        cy_px=float(centre[1]),
        theta_deg=official_angle,
        jaw_width_px=jaw_width,
        rectangle_height_px=GT_HEIGHT_PX,
    )


def candidate_row_to_grasp(row: dict[str, object]) -> CanonicalGrasp:
    return CanonicalGrasp(
        cx_px=float(row["cx_px"]),
        cy_px=float(row["cy_px"]),
        theta_deg=float(row["theta_deg"]),
        jaw_width_px=float(row["jaw_width_px"]),
        rectangle_height_px=float(row["rectangle_height_px"]),
        native_score=float(row["native_score"]),
        native_rank=int(row["native_rank"]),
        source_method=str(row["method"]),
        sample_id=str(row["sample_id"]),
        status=str(row.get("status", "ok")),
    )


def evaluate_candidate(
    prediction: CanonicalGrasp,
    ground_truth: Sequence[CanonicalGrasp],
) -> dict[str, object]:
    pairwise = []
    for index, gt in enumerate(ground_truth):
        iou = raster_iou(prediction, gt)
        angle = periodic_angle_error_deg(prediction.theta_deg, gt.theta_deg)
        pairwise.append(
            {
                "gt_index": index,
                "iou": iou,
                "angle_error_deg": angle,
                "iou_ok": bool(iou > IOU_THRESHOLD),
                "angle_ok": bool(angle <= ANGLE_THRESHOLD_DEG),
                "success": bool(iou > IOU_THRESHOLD and angle <= ANGLE_THRESHOLD_DEG),
            }
        )
    successes = [item for item in pairwise if item["success"]]
    ranked = sorted(
        successes or pairwise,
        key=lambda item: (-float(item["iou"]), float(item["angle_error_deg"]), int(item["gt_index"])),
    )
    best = ranked[0] if ranked else None
    # The fixed continuous-diagnostic match is max raster IoU, then min angle.
    diagnostic = min(
        pairwise,
        key=lambda item: (-float(item["iou"]), float(item["angle_error_deg"]), int(item["gt_index"])),
        default=None,
    )
    return {
        "success": bool(successes),
        "best_success_or_fallback": best,
        "diagnostic_match": diagnostic,
        "pairwise": pairwise,
    }


def evaluate_ranked(
    predictions: Iterable[CanonicalGrasp],
    gt_corner_sets: Sequence[Sequence[Sequence[float]]],
) -> dict[str, object]:
    ground_truth = tuple(gt_from_corners(item) for item in gt_corner_sets)
    ranked = sorted(predictions, key=lambda item: item.native_rank)
    evaluated = [evaluate_candidate(item, ground_truth) for item in ranked]
    first = next((index for index, item in enumerate(evaluated, 1) if item["success"]), None)
    return {
        "candidate_count": len(ranked),
        "first_success_rank": first,
        **{f"j_at_{k}": bool(first is not None and first <= k) for k in range(1, 6)},
        "oracle_all": bool(first is not None),
        "top1": evaluated[0] if evaluated else None,
        "evaluated": evaluated,
        "ground_truth": ground_truth,
    }


def rectangle_mask(grasp: CanonicalGrasp, shape: tuple[int, int] = IMAGE_SHAPE) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    flat = _pixels(grasp, shape)
    result.reshape(-1)[flat] = True
    return result
