from __future__ import annotations

import numpy as np


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homo @ transform.T)[:, :3]
