from __future__ import annotations

import cv2
import numpy as np

from association.collision_features import collision_penalty
from association.depth_features import depth_stability
from association.target_overlap import target_overlap
from association.width_features import gripper_width_match
from utils.data_types import CandidateFeatureVector, GraspCandidate, PointCloudRepresentation, TargetRegion
from utils.geometry import normalize, project_points


class CandidateFeatureExtractor:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def extract(
        self,
        candidates: list[GraspCandidate],
        target: TargetRegion,
        pcr: PointCloudRepresentation,
        depth: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> list[CandidateFeatureVector]:
        return [
            self.extract_one(c, target, pcr, depth=depth, intrinsics=intrinsics)
            for c in candidates
        ]

    def extract_one(
        self,
        candidate: GraspCandidate,
        target: TargetRegion,
        pcr: PointCloudRepresentation,
        depth: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> CandidateFeatureVector:
        center = pcr.target_center_3d
        if center is None:
            distance = 1.0
            center_alignment = 0.0
        else:
            distance = float(np.linalg.norm(candidate.position - center))
            scale = self._target_scale(pcr)
            center_alignment = float(np.exp(-distance / max(scale, 1e-4)))
        return CandidateFeatureVector(
            target_overlap=target_overlap(candidate, target, intrinsics),
            center_alignment=center_alignment,
            distance_to_target_center=distance,
            gripper_width_match=gripper_width_match(candidate, pcr, self.config.get("max_width", 0.10)),
            approach_direction_score=self._approach_score(candidate, pcr),
            depth_stability=depth_stability(candidate, depth, intrinsics),
            collision_penalty=collision_penalty(candidate, pcr),
            boundary_penalty=self._boundary_penalty(candidate, target, intrinsics),
            initial_geometric_score=float(candidate.initial_geometric_score),
            grounding_score=float(target.grounding_score),
        )

    def _target_scale(self, pcr: PointCloudRepresentation) -> float:
        if pcr.target_obb is not None:
            return float(max(np.mean(pcr.target_obb.extent), 1e-3))
        pts = np.asarray(pcr.clean_target_pcd.points) if pcr.clean_target_pcd is not None else np.zeros((0, 3))
        return float(max(np.linalg.norm(np.ptp(pts, axis=0)), 1e-3)) if len(pts) else 0.05

    def _approach_score(self, candidate: GraspCandidate, pcr: PointCloudRepresentation) -> float:
        approach = normalize(candidate.approach_vector, np.array([0.0, 0.0, -1.0]))
        if candidate.grasp_type == "top_down":
            return float(np.clip(np.dot(approach, np.array([0.0, 0.0, -1.0])), 0.0, 1.0))
        horizontal = 1.0 - abs(float(approach[2]))
        return float(np.clip(horizontal, 0.0, 1.0))

    def _boundary_penalty(self, candidate: GraspCandidate, target: TargetRegion, intrinsics: np.ndarray | None) -> float:
        if target.mask is None or intrinsics is None:
            return 0.2
        H, W = target.mask.shape[:2]
        uv = project_points(candidate.position.reshape(1, 3), intrinsics)[0]
        u, v = int(round(uv[0])), int(round(uv[1]))
        if not (0 <= u < W and 0 <= v < H):
            return 1.0
        dist = cv2.distanceTransform(target.mask.astype(np.uint8), cv2.DIST_L2, 3)
        max_dist = max(float(dist.max()), 1.0)
        return float(1.0 - np.clip(dist[v, u] / max_dist, 0.0, 1.0))
