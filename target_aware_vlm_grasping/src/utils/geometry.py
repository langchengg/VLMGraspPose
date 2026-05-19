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


def grasp_rectangle_2d(
    position: np.ndarray,
    closing_direction: np.ndarray,
    gripper_width: float,
    intrinsics: np.ndarray,
    height_px: float = 18.0,
    min_width_px: float = 8.0,
    max_width_px: float = 220.0,
) -> list[list[float]]:
    position = np.asarray(position, dtype=float)
    closing = normalize(closing_direction, np.array([1.0, 0.0, 0.0]))
    K = np.asarray(intrinsics, dtype=float)
    center = project_points(position.reshape(1, 3), K)[0]
    half_width_3d = max(float(gripper_width), 1e-4) * 0.5
    edge_points = np.stack([
        position - closing * half_width_3d,
        position + closing * half_width_3d,
    ])
    edge_uv = project_points(edge_points, K)
    axis = edge_uv[1] - edge_uv[0]
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        axis = np.array([1.0, 0.0])
        axis_norm = 1.0
    direction = axis / axis_norm
    width_px = float(np.clip(axis_norm, min_width_px, max_width_px))
    height_px = float(max(height_px, 4.0))
    normal = np.array([-direction[1], direction[0]])
    corners = np.stack([
        center - 0.5 * width_px * direction - 0.5 * height_px * normal,
        center + 0.5 * width_px * direction - 0.5 * height_px * normal,
        center + 0.5 * width_px * direction + 0.5 * height_px * normal,
        center - 0.5 * width_px * direction + 0.5 * height_px * normal,
    ])
    return corners.tolist()


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
