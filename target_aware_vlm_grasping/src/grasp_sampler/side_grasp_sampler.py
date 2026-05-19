from __future__ import annotations

import numpy as np

from utils.data_types import GraspCandidate, PointCloudRepresentation
from utils.geometry import distance_to_plane, matrix_to_quaternion, normalize, rotation_from_approach_closing


def sample_side_grasps(
    pcr: PointCloudRepresentation,
    config: dict,
) -> list[GraspCandidate]:
    center = pcr.target_center_3d
    obb = pcr.target_obb
    if center is None or obb is None:
        return []
    axes = np.asarray(obb.R, dtype=float)
    extent = np.asarray(obb.extent, dtype=float)
    vertical = np.array([0.0, 0.0, 1.0])
    candidates = []
    clearance = config.get("min_table_clearance", 0.015)
    if pcr.table_plane is not None:
        if distance_to_plane(np.asarray(center).reshape(1, 3), pcr.table_plane)[0] < clearance:
            return []
    for axis_idx in range(3):
        axis = normalize(axes[:, axis_idx])
        if abs(axis[2]) > 0.5:
            continue
        for sign in [-1.0, 1.0]:
            approach = normalize(sign * axis)
            closing = normalize(np.cross(vertical, approach), np.array([1.0, 0.0, 0.0]))
            width_dim = sorted(extent)[1] if len(extent) else config.get("min_width", 0.02)
            width = float(np.clip(width_dim + config.get("width_margin", 0.01),
                                  config.get("min_width", 0.02),
                                  config.get("max_width", 0.10)))
            pos = np.asarray(center, dtype=float) - approach * min(0.01, width * 0.25)
            R = rotation_from_approach_closing(approach, closing)
            candidates.append(GraspCandidate(
                position=pos,
                orientation=matrix_to_quaternion(R),
                approach_vector=approach,
                closing_direction=closing,
                gripper_width=width,
                grasp_type="side",
                initial_geometric_score=0.55,
                metadata={"sampler": "side", "axis": int(axis_idx), "sign": float(sign)},
            ))
    return candidates
