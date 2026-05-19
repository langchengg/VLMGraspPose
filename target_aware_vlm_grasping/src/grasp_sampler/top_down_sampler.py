from __future__ import annotations

import numpy as np

from utils.data_types import GraspCandidate, PointCloudRepresentation
from utils.geometry import matrix_to_quaternion, normalize, rotation_from_approach_closing


def sample_top_down_grasps(
    pcr: PointCloudRepresentation,
    config: dict,
) -> list[GraspCandidate]:
    center = pcr.target_center_3d
    obb = pcr.target_obb
    if center is None or obb is None:
        return []
    axes = np.asarray(obb.R, dtype=float)
    extent = np.asarray(obb.extent, dtype=float)
    horizontal_axes = sorted(
        [axes[:, 0], axes[:, 1], axes[:, 2]],
        key=lambda a: abs(a[2]),
    )[:2]
    approach = np.array([0.0, 0.0, -1.0])
    width = float(np.clip(min(extent[:2]) + config.get("width_margin", 0.01),
                          config.get("min_width", 0.02),
                          config.get("max_width", 0.10)))
    z_offset = max(0.5 * min(extent) if len(extent) else 0.0, 0.0)
    base_pos = np.asarray(center, dtype=float) + np.array([0.0, 0.0, z_offset])
    candidates = []
    for i, closing in enumerate(horizontal_axes):
        closing = normalize(closing, np.array([1.0, 0.0, 0.0]))
        R = rotation_from_approach_closing(approach, closing)
        score = 0.75 if width <= config.get("max_width", 0.10) else 0.3
        candidates.append(GraspCandidate(
            position=base_pos.copy(),
            orientation=matrix_to_quaternion(R),
            approach_vector=approach.copy(),
            closing_direction=closing,
            gripper_width=width,
            grasp_type="top_down",
            initial_geometric_score=score,
            metadata={"sampler": "top_down", "axis_index": i},
        ))
    return candidates
