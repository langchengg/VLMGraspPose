"""Deterministic quality-map decoding and shared planar-grasp NMS."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

from .geometry import CropTransform, normalize_angle_deg, periodic_angle_difference_deg
from .geometry import rotated_rectangle_iou
from .types import Grasp4DoF, GraspPrediction


def stable_candidate_id(
    *,
    sample_id: str,
    backend: str,
    center_x: float,
    center_y: float,
    angle_deg: float,
    width_px: float,
    source_key: str = "",
) -> str:
    """Hash canonical geometry and source identity without score-dependent drift."""

    values = (center_x, center_y, angle_deg, width_px)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("candidate geometry must be finite")
    payload = {
        "sample_id": str(sample_id),
        "backend": str(backend),
        "center_x": float(center_x).hex(),
        "center_y": float(center_y).hex(),
        "angle_deg": normalize_angle_deg(angle_deg).hex(),
        "width_px": float(width_px).hex(),
        "source_key": str(source_key),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "c_" + hashlib.sha256(encoded).hexdigest()[:20]


def candidate_sort_key(candidate: Grasp4DoF) -> tuple[float, float, float, float, str]:
    return (
        -candidate.score,
        candidate.center_y,
        candidate.center_x,
        normalize_angle_deg(candidate.angle_deg),
        candidate.candidate_id,
    )


def rank_candidates(candidates: Sequence[Grasp4DoF]) -> list[Grasp4DoF]:
    return sorted(candidates, key=candidate_sort_key)


def extract_quality_peaks(
    quality_map: np.ndarray,
    *,
    threshold: float = 0.0,
    min_distance_px: int = 1,
    max_peaks: int = 100,
) -> list[tuple[int, int, float]]:
    """Extract deterministic ``(row, column, score)`` local maxima."""

    quality = np.asarray(quality_map, dtype=np.float64)
    if quality.ndim != 2:
        raise ValueError("quality_map must be 2D")
    if min_distance_px < 0 or max_peaks < 0:
        raise ValueError("peak limits must be non-negative")
    if max_peaks == 0:
        return []
    finite_quality = np.where(np.isfinite(quality), quality, -np.inf)
    radius = int(min_distance_px)
    local_maximum = ndimage.maximum_filter(
        finite_quality, size=2 * radius + 1, mode="constant", cval=-np.inf
    )
    rows, columns = np.nonzero(
        np.isfinite(finite_quality)
        & (finite_quality > float(threshold))
        & (finite_quality == local_maximum)
    )
    ordered = sorted(
        ((int(row), int(column), float(finite_quality[row, column])) for row, column in zip(rows, columns)),
        key=lambda item: (-item[2], item[0], item[1]),
    )
    selected: list[tuple[int, int, float]] = []
    minimum_squared = float(min_distance_px * min_distance_px)
    for peak in ordered:
        if min_distance_px and any(
            (peak[0] - kept[0]) ** 2 + (peak[1] - kept[1]) ** 2 <= minimum_squared
            for kept in selected
        ):
            continue
        selected.append(peak)
        if len(selected) == max_peaks:
            break
    return selected


def decode_quality_maps(
    quality_map: np.ndarray,
    cos_2theta_map: np.ndarray,
    sin_2theta_map: np.ndarray,
    width_map: np.ndarray,
    *,
    sample_id: str,
    backend: str,
    transform: CropTransform | None = None,
    quality_threshold: float = 0.0,
    min_peak_distance_px: int = 1,
    max_peaks: int = 100,
    width_scale: float = 1.0,
    fixed_height_px: float = 20.0,
) -> list[Grasp4DoF]:
    """Decode network maps, restoring x/y, angle and width to native pixels."""

    maps = [np.asarray(value) for value in (quality_map, cos_2theta_map, sin_2theta_map, width_map)]
    if any(value.ndim != 2 or value.shape != maps[0].shape for value in maps):
        raise ValueError("all decoder maps must have the same 2D shape")
    if not math.isfinite(float(width_scale)) or width_scale <= 0.0:
        raise ValueError("width_scale must be finite and positive")
    candidates: list[Grasp4DoF] = []
    for row, column, score in extract_quality_peaks(
        maps[0],
        threshold=quality_threshold,
        min_distance_px=min_peak_distance_px,
        max_peaks=max_peaks,
    ):
        cos_value = float(maps[1][row, column])
        sin_value = float(maps[2][row, column])
        width = float(maps[3][row, column]) * float(width_scale)
        if not all(math.isfinite(value) for value in (cos_value, sin_value, width)) or width <= 0.0:
            continue
        angle = normalize_angle_deg(0.5 * math.degrees(math.atan2(sin_value, cos_value)))
        center_x, center_y = float(column), float(row)
        if transform is not None:
            center_x, center_y, angle, width = transform.model_to_native_pose(
                center_x, center_y, angle, width, clip=True
            )
        candidate_id = stable_candidate_id(
            sample_id=sample_id,
            backend=backend,
            center_x=center_x,
            center_y=center_y,
            angle_deg=angle,
            width_px=width,
            source_key=f"peak:{row}:{column}",
        )
        candidates.append(
            Grasp4DoF(
                center_x=center_x,
                center_y=center_y,
                angle_deg=angle,
                width_px=width,
                height_px=fixed_height_px,
                score=score,
                candidate_id=candidate_id,
                metadata={"source_row": row, "source_column": column},
            )
        )
    return rank_candidates(candidates)


@dataclass(frozen=True, slots=True)
class NMSConfig:
    center_distance_px: float = 8.0
    angle_distance_deg: float = 15.0
    width_distance_px: float = 10.0
    rectangle_iou_threshold: float = 0.25
    max_output: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.center_distance_px,
            self.angle_distance_deg,
            self.width_distance_px,
            self.rectangle_iou_threshold,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("NMS thresholds must be finite and non-negative")
        if not 0.0 <= self.rectangle_iou_threshold <= 1.0:
            raise ValueError("rectangle_iou_threshold must be in [0, 1]")
        if self.max_output is not None and self.max_output <= 0:
            raise ValueError("max_output must be positive")


def _duplicates(first: Grasp4DoF, second: Grasp4DoF, config: NMSConfig) -> bool:
    center_close = math.hypot(
        first.center_x - second.center_x, first.center_y - second.center_y
    ) <= config.center_distance_px
    angle_close = (
        periodic_angle_difference_deg(first.angle_deg, second.angle_deg)
        <= config.angle_distance_deg
    )
    width_close = abs(first.width_px - second.width_px) <= config.width_distance_px
    iou_close = rotated_rectangle_iou(first, second) >= config.rectangle_iou_threshold
    return bool(angle_close and ((center_close and width_close) or iou_close))


def non_maximum_suppression(
    candidates: Sequence[Grasp4DoF], config: NMSConfig = NMSConfig()
) -> list[Grasp4DoF]:
    kept: list[Grasp4DoF] = []
    for candidate in rank_candidates(candidates):
        if any(_duplicates(candidate, previous, config) for previous in kept):
            continue
        kept.append(candidate)
        if config.max_output is not None and len(kept) >= config.max_output:
            break
    return kept


def serialize_top5(candidates: Sequence[Grasp4DoF]) -> list[dict[str, object]]:
    """Serialize up to five real candidates without padding or duplication."""

    return [candidate.to_dict(rank=rank) for rank, candidate in enumerate(rank_candidates(candidates)[:5], 1)]


def build_prediction(
    *,
    sample_id: str,
    backend: str,
    conditioning_variant: str,
    raw_candidates: Sequence[Grasp4DoF],
    nms_candidates: Sequence[Grasp4DoF],
    empty_reason: str | None = None,
    runtime_seconds: float = 0.0,
    device: str = "cpu",
) -> GraspPrediction:
    ranked = tuple(rank_candidates(nms_candidates)[:5])
    if not ranked and not empty_reason:
        empty_reason = "no_candidate_generated"
    return GraspPrediction(
        sample_id=sample_id,
        backend=backend,
        conditioning_variant=conditioning_variant,
        raw_candidate_count=len(raw_candidates),
        nms_candidate_count=len(nms_candidates),
        top1=ranked[0] if ranked else None,
        top5=ranked,
        candidates=tuple(rank_candidates(nms_candidates)),
        empty_reason=empty_reason,
        runtime_seconds=runtime_seconds,
        device=device,
    )
