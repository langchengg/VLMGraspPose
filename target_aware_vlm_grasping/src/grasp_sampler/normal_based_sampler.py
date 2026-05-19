from __future__ import annotations

import numpy as np

from utils.data_types import GraspCandidate, PointCloudRepresentation
from utils.geometry import matrix_to_quaternion, normalize, rotation_from_approach_closing


def sample_normal_based_grasps(
    pcr: PointCloudRepresentation,
    config: dict,
) -> list[GraspCandidate]:
    pcd = pcr.clean_target_pcd
    if pcd is None or len(pcd.points) < 8:
        return []
    points = np.asarray(pcd.points, dtype=float)
    normals = np.asarray(pcd.normals, dtype=float) if len(pcd.normals) else None
    if normals is None or normals.shape != points.shape:
        return []
    max_points = min(config.get("normal_pair_max_points", 1500), len(points))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(points), size=max_points, replace=False)
    points_s = points[sample_idx]
    normals_s = normals[sample_idx]
    min_width = config.get("min_width", 0.02)
    max_width = config.get("max_width", 0.10)
    candidates = []
    for i in sample_idx[:config.get("normal_samples", 32)]:
        p = points[i]
        n = normalize(normals[i], np.array([0.0, 0.0, 1.0]))
        diffs = points_s - p
        dists = np.linalg.norm(diffs, axis=1)
        width_mask = (dists >= min_width) & (dists <= max_width)
        opposite = -normals_s @ n
        valid = np.where(width_mask & (opposite > 0.4))[0]
        if len(valid) == 0:
            continue
        j = valid[np.argmax(opposite[valid])]
        q = points_s[j]
        center = 0.5 * (p + q)
        closing = normalize(q - p, np.array([1.0, 0.0, 0.0]))
        approach = normalize(-(n + normals_s[j]), np.array([0.0, 0.0, -1.0]))
        if np.linalg.norm(approach) < 1e-6:
            approach = np.array([0.0, 0.0, -1.0])
        R = rotation_from_approach_closing(approach, closing)
        width = float(np.clip(np.linalg.norm(q - p) + config.get("width_margin", 0.01), min_width, max_width))
        candidates.append(GraspCandidate(
            position=center,
            orientation=matrix_to_quaternion(R),
            approach_vector=approach,
            closing_direction=closing,
            gripper_width=width,
            grasp_type="normal_based",
            initial_geometric_score=float(np.clip(opposite[j], 0.0, 1.0)),
            metadata={"sampler": "normal_based"},
        ))
    return candidates
