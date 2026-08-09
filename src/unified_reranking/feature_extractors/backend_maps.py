"""Candidate-aligned features from frozen G1/C1 backend maps.

Map coordinates are the exact decoder peak coordinates stored with each frozen
candidate.  This module never finds new peaks and cannot alter candidate
membership or geometry.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from .common import periodic_angle_error


REQUIRED_MAPS = (
    "quality_post",
    "cos_2theta_post",
    "sin_2theta_post",
    "width_px_post",
)


def _patch(array: np.ndarray, row: int, column: int, radius: int = 3) -> np.ndarray:
    y0, y1 = max(0, row - radius), min(array.shape[0], row + radius + 1)
    x0, x1 = max(0, column - radius), min(array.shape[1], column + radius + 1)
    return np.asarray(array[y0:y1, x0:x1], dtype=np.float64)


def _bilinear(array: np.ndarray, x: float, y: float) -> float:
    value = np.asarray(array, dtype=np.float64)
    height, width = value.shape
    if not (0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0):
        raise ValueError(f"candidate coordinate outside backend map: ({x}, {y}) / {value.shape}")
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    return float(
        value[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + value[y0, x1] * dx * (1.0 - dy)
        + value[y1, x0] * (1.0 - dx) * dy
        + value[y1, x1] * dx * dy
    )


def _plateau_area(quality: np.ndarray, row: int, column: int) -> int:
    center = float(quality[row, column])
    tolerance = max(1e-4, 0.01 * max(abs(center), 1e-3))
    plateau = np.abs(np.asarray(quality, dtype=float) - center) <= tolerance
    labels, _ = ndimage.label(plateau)
    label = int(labels[row, column])
    return 0 if label == 0 else int(np.sum(labels == label))


def candidate_backend_map_features(
    candidates: pd.DataFrame,
    maps: Mapping[str, np.ndarray] | None,
    *,
    transform: Any | None = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Extract F2 evidence at immutable native candidate coordinates.

    When a ``CropTransform`` is supplied, every candidate is mapped from the
    original image into the backend map and the sampled angle/width are mapped
    back to original coordinates.  This deliberately avoids the invalid
    ``native_size / model_size`` shortcut for cropped backend inputs.
    """

    required_columns = {
        "sample_id",
        "candidate_id",
        "native_score",
        "theta_deg",
        "width_px",
    }
    if maps is not None:
        if transform is None:
            required_columns.update({"source_row", "source_column"})
        else:
            required_columns.update({"cx_px", "cy_px"})
    missing = sorted(required_columns.difference(candidates.columns))
    if missing:
        raise ValueError(f"backend candidate table missing columns: {missing}")
    def missing_row(candidate: Any, model_x: float, model_y: float, roundtrip_error: float) -> dict[str, object]:
        return {
            "sample_id": str(candidate.sample_id),
            "candidate_id": str(candidate.candidate_id),
            "backend_quality_at_candidate": 0.0,
            "local_peak_prominence": 0.0,
            "local_q_mean": 0.0,
            "local_q_std": 0.0,
            "local_q_min": 0.0,
            "local_q_max": 0.0,
            "peak_laplacian": 0.0,
            "peak_plateau_area": 0.0,
            "angle_coherence": 0.0,
            "backend_angle_at_candidate_deg": 0.0,
            "absolute_periodic_angle_difference_to_backend": 90.0,
            "cos_2_angle_difference_to_backend": -1.0,
            "sin_2_angle_difference_to_backend": 0.0,
            "width_local_mean": 0.0,
            "width_local_std": 0.0,
            "backend_width_at_candidate_px": 0.0,
            "log_width_ratio_to_backend": 0.0,
            "backend_map_missing": 1.0,
            "backend_model_x": model_x,
            "backend_model_y": model_y,
            "backend_transform_roundtrip_error_px": roundtrip_error,
        }

    if maps is None:
        if not allow_missing:
            raise ValueError("backend map archive is unavailable")
        return pd.DataFrame(
            [missing_row(candidate, 0.0, 0.0, 0.0) for candidate in candidates.itertuples(index=False)]
        )
    missing_maps = sorted(set(REQUIRED_MAPS).difference(maps))
    if missing_maps:
        raise ValueError(f"backend map archive missing arrays: {missing_maps}")
    prepared = {name: np.asarray(maps[name], dtype=np.float64) for name in REQUIRED_MAPS}
    shape = prepared[REQUIRED_MAPS[0]].shape
    if len(shape) != 2 or any(array.shape != shape for array in prepared.values()):
        raise ValueError("backend maps must share a two-dimensional shape")
    if any(not np.isfinite(array).all() for array in prepared.values()):
        raise ValueError("backend maps must be finite")

    quality = prepared["quality_post"]
    cosine = prepared["cos_2theta_post"]
    sine = prepared["sin_2theta_post"]
    width = prepared["width_px_post"]
    laplacian = cv2.Laplacian(quality.astype(np.float32), cv2.CV_32F)
    rows: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        if transform is None:
            model_x, model_y = float(candidate.source_column), float(candidate.source_row)
            roundtrip_error = 0.0
        else:
            model_x, model_y = transform.native_to_model_point(
                float(candidate.cx_px), float(candidate.cy_px)
            )
            native_x, native_y = transform.model_to_native_point(model_x, model_y)
            roundtrip_error = math.hypot(
                native_x - float(candidate.cx_px), native_y - float(candidate.cy_px)
            )
            if roundtrip_error > 1e-6:
                raise ValueError(
                    f"backend CropTransform round-trip failed: {candidate.sample_id}/{candidate.candidate_id}"
                )
        row, column = int(round(model_y)), int(round(model_x))
        coordinate_outside = not (
            0.0 <= model_x <= shape[1] - 1.0
            and 0.0 <= model_y <= shape[0] - 1.0
        )
        if (
            coordinate_outside
            or row < 0
            or row >= shape[0]
            or column < 0
            or column >= shape[1]
        ):
            if not allow_missing:
                raise ValueError(f"stored peak coordinate outside map: {candidate.sample_id}/{candidate.candidate_id}")
            rows.append(missing_row(candidate, model_x, model_y, roundtrip_error))
            continue
        q_patch = _patch(quality, row, column)
        cos_patch = _patch(cosine, row, column)
        sin_patch = _patch(sine, row, column)
        width_patch = _patch(width, row, column)
        q_center = _bilinear(quality, model_x, model_y)
        neighbour = q_patch.reshape(-1)
        # Exclude exactly one centre element from the local background mean.
        centre_index = (row - max(0, row - 3)) * q_patch.shape[1] + (column - max(0, column - 3))
        background = np.delete(neighbour, centre_index)
        backend_model_angle = math.degrees(
            0.5
            * math.atan2(
                _bilinear(sine, model_x, model_y),
                _bilinear(cosine, model_x, model_y),
            )
        )
        backend_model_width = abs(_bilinear(width, model_x, model_y))
        if transform is None:
            backend_angle, backend_width = backend_model_angle, backend_model_width
        else:
            _, _, backend_angle, backend_width = transform.model_to_native_pose(
                model_x,
                model_y,
                backend_model_angle,
                max(backend_model_width, 1e-6),
                clip=False,
            )
        rows.append(
            {
                "sample_id": str(candidate.sample_id),
                "candidate_id": str(candidate.candidate_id),
                "backend_quality_at_candidate": q_center,
                "local_peak_prominence": q_center - (float(background.mean()) if len(background) else q_center),
                "local_q_mean": float(q_patch.mean()),
                "local_q_std": float(q_patch.std()),
                "local_q_min": float(q_patch.min()),
                "local_q_max": float(q_patch.max()),
                "peak_laplacian": float(laplacian[row, column]),
                "peak_plateau_area": _plateau_area(quality, row, column),
                "angle_coherence": float(np.hypot(cos_patch.mean(), sin_patch.mean())),
                "backend_angle_at_candidate_deg": backend_angle,
                "absolute_periodic_angle_difference_to_backend": periodic_angle_error(
                    float(candidate.theta_deg), backend_angle
                ),
                "cos_2_angle_difference_to_backend": math.cos(
                    math.radians(2.0 * (float(candidate.theta_deg) - backend_angle))
                ),
                "sin_2_angle_difference_to_backend": math.sin(
                    math.radians(2.0 * (float(candidate.theta_deg) - backend_angle))
                ),
                "width_local_mean": float(width_patch.mean()),
                "width_local_std": float(width_patch.std()),
                "backend_width_at_candidate_px": backend_width,
                "log_width_ratio_to_backend": math.log(
                    max(float(candidate.width_px), 1e-6) / max(abs(backend_width), 1e-6)
                ),
                "backend_map_missing": 0.0,
                "backend_model_x": model_x,
                "backend_model_y": model_y,
                "backend_transform_roundtrip_error_px": roundtrip_error,
            }
        )
    return pd.DataFrame(rows)


def load_candidate_backend_map_features(
    candidates: pd.DataFrame,
    archive_path: str | Path,
    *,
    transform: Any | None = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    with np.load(Path(archive_path), allow_pickle=False) as archive:
        return candidate_backend_map_features(
            candidates, archive, transform=transform, allow_missing=allow_missing
        )
