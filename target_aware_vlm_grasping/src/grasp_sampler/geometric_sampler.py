from __future__ import annotations

import numpy as np

from grasp_sampler.bbox_aligned_sampler import sample_bbox_aligned_grasps
from grasp_sampler.normal_based_sampler import sample_normal_based_grasps
from grasp_sampler.side_grasp_sampler import sample_side_grasps
from grasp_sampler.top_down_sampler import sample_top_down_grasps
from utils.data_types import GraspCandidate, PointCloudRepresentation


class GeometricGraspSampler:
    def __init__(self, config: dict):
        self.config = config

    def sample(self, pcr: PointCloudRepresentation, top_k: int | None = None) -> list[GraspCandidate]:
        candidates: list[GraspCandidate] = []
        candidates.extend(sample_top_down_grasps(pcr, self.config))
        candidates.extend(sample_bbox_aligned_grasps(pcr, self.config))
        candidates.extend(sample_side_grasps(pcr, self.config))
        candidates.extend(sample_normal_based_grasps(pcr, self.config))
        candidates = self._deduplicate(candidates)
        candidates = [c for c in candidates if self._valid(c)]
        candidates.sort(key=lambda c: c.initial_geometric_score, reverse=True)
        k = top_k if top_k is not None else self.config.get("top_k", 5)
        return candidates[:k]

    def _valid(self, candidate: GraspCandidate) -> bool:
        width = candidate.gripper_width
        quat_norm = np.linalg.norm(candidate.orientation)
        approach_norm = np.linalg.norm(candidate.approach_vector)
        closing_norm = np.linalg.norm(candidate.closing_direction)
        return (
            np.all(np.isfinite(candidate.position))
            and np.all(np.isfinite(candidate.orientation))
            and 0.95 <= quat_norm <= 1.05
            and 0.95 <= approach_norm <= 1.05
            and 0.95 <= closing_norm <= 1.05
            and self.config.get("min_width", 0.02) <= width <= self.config.get("max_width", 0.10)
            and 0.0 <= candidate.initial_geometric_score <= 1.0
        )

    def _deduplicate(self, candidates: list[GraspCandidate], min_dist: float = 0.004) -> list[GraspCandidate]:
        kept: list[GraspCandidate] = []
        for c in candidates:
            if all(np.linalg.norm(c.position - k.position) >= min_dist for k in kept):
                kept.append(c)
        return kept
