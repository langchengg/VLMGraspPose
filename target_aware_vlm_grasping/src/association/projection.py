from __future__ import annotations

import numpy as np

from utils.geometry import grasp_rectangle_2d, project_points


def project_point(point_3d: np.ndarray, intrinsics: np.ndarray) -> tuple[float, float]:
    point = np.asarray(point_3d, dtype=float).reshape(1, 3)
    uv = project_points(point, np.asarray(intrinsics, dtype=float))[0]
    return float(uv[0]), float(uv[1])


def project_grasp_rectangle(position: np.ndarray, closing_direction: np.ndarray, width: float, intrinsics: np.ndarray) -> list[list[float]]:
    return grasp_rectangle_2d(position, closing_direction, width, intrinsics)
