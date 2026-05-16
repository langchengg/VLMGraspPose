from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        if fallback is None:
            return np.zeros_like(v, dtype=float)
        return normalize(fallback)
    return v / n


def rotation_from_approach_closing(
    approach: np.ndarray,
    closing: np.ndarray,
) -> np.ndarray:
    x_axis = normalize(approach, np.array([0.0, 0.0, -1.0]))
    y_axis = np.asarray(closing, dtype=float)
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = normalize(y_axis, np.array([1.0, 0.0, 0.0]))
    z_axis = normalize(np.cross(x_axis, y_axis), np.array([0.0, 1.0, 0.0]))
    y_axis = normalize(np.cross(z_axis, x_axis), np.array([1.0, 0.0, 0.0]))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rotation_matrix).as_quat()


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quaternion).as_matrix()


def project_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    z = np.clip(points[:, 2], 1e-8, None)
    u = points[:, 0] * intrinsics[0, 0] / z + intrinsics[0, 2]
    v = points[:, 1] * intrinsics[1, 1] / z + intrinsics[1, 2]
    return np.stack([u, v], axis=1)


def distance_to_plane(points: np.ndarray, plane: np.ndarray | None) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if plane is None:
        return np.full(len(points), np.inf)
    plane = np.asarray(plane, dtype=float)
    normal = plane[:3]
    denom = max(np.linalg.norm(normal), 1e-8)
    return (points @ normal + plane[3]) / denom


def clamp01(x: float | np.ndarray) -> float | np.ndarray:
    return np.clip(x, 0.0, 1.0)
