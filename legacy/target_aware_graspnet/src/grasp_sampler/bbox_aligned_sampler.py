from __future__ import annotations

import numpy as np

from utils.data_types import GraspCandidate, PointCloudRepresentation
from utils.geometry import matrix_to_quaternion, normalize, rotation_from_approach_closing


def sample_bbox_aligned_grasps(
    pcr: PointCloudRepresentation,
    config: dict,
) -> list[GraspCandidate]:
    center = pcr.target_center_3d
    obb = pcr.target_obb
    if center is None or obb is None:
        return []
    axes = np.asarray(obb.R, dtype=float)
    extent = np.asarray(obb.extent, dtype=float)
    order = np.argsort(extent)
    candidates = []
    for rank, axis_idx in enumerate(order[:2]):
        closing = normalize(axes[:, axis_idx])
        approach = np.array([0.0, 0.0, -1.0])
        if abs(np.dot(closing, approach)) > 0.85:
            continue
        width = float(np.clip(extent[axis_idx] + config.get("width_margin", 0.01),
                              config.get("min_width", 0.02),
                              config.get("max_width", 0.10)))
        R = rotation_from_approach_closing(approach, closing)
        width_score = 1.0 - abs(width - (extent[axis_idx] + config.get("width_margin", 0.01))) / max(config.get("max_width", 0.10), 1e-6)
        candidates.append(GraspCandidate(
            position=np.asarray(center, dtype=float).copy(),
            orientation=matrix_to_quaternion(R),
            approach_vector=approach.copy(),
            closing_direction=closing,
            gripper_width=width,
            grasp_type="bbox_aligned",
            initial_geometric_score=float(np.clip(0.55 + 0.35 * width_score, 0.0, 1.0)),
            metadata={"sampler": "bbox_aligned", "obb_axis": int(axis_idx), "axis_rank": int(rank)},
        ))
    return candidates
