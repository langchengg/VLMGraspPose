"""
stage2/grasp_generator.py — Grasp Candidate Generation
========================================================
Generates 6-DoF grasp candidates from point clouds.

Two implementations:
  • AntipodalGraspSampler  — target-region local generator (main)
  • (Future) GraspNetBaseline — wraps pre-trained GraspNet checkpoint

All grasp poses are in **camera frame**.
"""

import abc
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ── Grasp Candidate ──────────────────────────────────────────────────

@dataclass
class GraspCandidate:
    candidate_id: int
    position: List[float]        # [x, y, z] in camera frame
    orientation: List[float]     # [qx, qy, qz, qw] quaternion
    width: float                 # gripper opening in metres
    score: float                 # quality score 0–1
    source: str                  # e.g. "antipodal", "graspnet"
    frame: str = "camera"

    def to_dict(self):
        return asdict(self)


# ── Base class ───────────────────────────────────────────────────────

class GraspGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        point_cloud: np.ndarray,
        normals: Optional[np.ndarray] = None,
        target_region: Optional[dict] = None,
    ) -> List[GraspCandidate]:
        ...


# ── Antipodal Grasp Sampler ──────────────────────────────────────────

class AntipodalGraspSampler(GraspGenerator):
    """Generate grasp candidates via antipodal sampling on local point cloud.

    Strategy
    --------
    1. Sub-sample *num_contact_samples* points on the target surface.
    2. For each sample, pick a nearby point that forms an approximate
       antipodal pair (opposing normals, within gripper width).
    3. Compute grasp centre, approach direction, and opening width.
    4. Score by normal-alignment quality and contact stability.
    5. Return top-K candidates.
    """

    def __init__(
        self,
        top_k: int = config.GRASP_TOP_K,
        num_contact_samples: int = 200,
        min_width: float = config.GRASP_MIN_WIDTH,
        max_width: float = config.GRASP_MAX_WIDTH,
        antipodal_thresh: float = 0.3,
    ):
        self.top_k = top_k
        self.num_contact_samples = num_contact_samples
        self.min_width = min_width
        self.max_width = max_width
        self.antipodal_thresh = antipodal_thresh

    def generate(
        self,
        point_cloud: np.ndarray,
        normals: Optional[np.ndarray] = None,
        target_region: Optional[dict] = None,
    ) -> List[GraspCandidate]:
        """
        Parameters
        ----------
        point_cloud : (N, 3) target object points in camera frame
        normals : (N, 3) surface normals (estimated if None)
        target_region : unused here (already cropped)
        """
        if len(point_cloud) < 10:
            return []

        # Estimate normals if not provided
        if normals is None:
            from data.point_cloud import estimate_normals_pca
            normals = estimate_normals_pca(point_cloud, k=min(30, len(point_cloud)))

        N = len(point_cloud)
        num_samples = min(self.num_contact_samples, N)

        # Random contact point indices
        rng = np.random.RandomState(42)
        idx1 = rng.choice(N, size=num_samples, replace=(num_samples > N))

        candidates = []

        for i in idx1:
            p1 = point_cloud[i]
            n1 = normals[i]

            # Find points that could form antipodal pair
            diffs = point_cloud - p1
            dists = np.linalg.norm(diffs, axis=1)

            # Within gripper width
            width_mask = (dists >= self.min_width) & (dists <= self.max_width)
            # Opposing normals
            if np.linalg.norm(n1) < 1e-6:
                continue
            cos_angle = -np.sum(normals * n1, axis=1)  # n1 · (-n2)
            antipodal_mask = cos_angle > self.antipodal_thresh

            valid = width_mask & antipodal_mask
            valid_idx = np.where(valid)[0]

            if len(valid_idx) == 0:
                continue

            # Pick best antipodal partner
            best_j = valid_idx[np.argmax(cos_angle[valid_idx])]
            p2 = point_cloud[best_j]
            n2 = normals[best_j]

            # Grasp parameters
            center = (p1 + p2) / 2.0
            width = float(np.linalg.norm(p2 - p1))
            approach = p2 - p1
            approach = approach / (np.linalg.norm(approach) + 1e-8)

            # Grasp frame: approach direction + up direction
            orientation = _approach_to_quaternion(approach)

            # Quality score: how antipodal is the pair
            quality = float(cos_angle[best_j])
            quality = np.clip(quality, 0.0, 1.0)

            candidates.append(GraspCandidate(
                candidate_id=len(candidates),
                position=center.tolist(),
                orientation=orientation.tolist(),
                width=width,
                score=quality,
                source="antipodal",
            ))

        # De-duplicate close candidates
        candidates = _dedup_candidates(candidates, min_dist=0.005)

        # Sort by score and return top-K
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:self.top_k]

        # Re-assign IDs
        for i, c in enumerate(candidates):
            c.candidate_id = i

        return candidates


# ── Helpers ──────────────────────────────────────────────────────────

def _approach_to_quaternion(approach: np.ndarray) -> np.ndarray:
    """Convert an approach vector to a quaternion [qx,qy,qz,qw].

    The approach vector defines the gripper closing direction.
    We build a full rotation matrix and convert to quaternion.
    """
    # Approach = x-axis of gripper
    ax = approach / (np.linalg.norm(approach) + 1e-8)

    # Choose an up vector not parallel to ax
    up = np.array([0.0, 0.0, -1.0])
    if abs(np.dot(ax, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])

    # Gram-Schmidt
    az = np.cross(ax, up)
    az = az / (np.linalg.norm(az) + 1e-8)
    ay = np.cross(az, ax)
    ay = ay / (np.linalg.norm(ay) + 1e-8)

    R = np.stack([ax, ay, az], axis=1)  # 3×3 rotation

    return _rotation_to_quaternion(R)


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) + 1e-8)


def _dedup_candidates(
    candidates: List[GraspCandidate],
    min_dist: float = 0.005,
) -> List[GraspCandidate]:
    """Remove near-duplicate candidates (centres closer than min_dist)."""
    if not candidates:
        return candidates

    kept = [candidates[0]]
    for c in candidates[1:]:
        pos = np.array(c.position)
        too_close = False
        for k in kept:
            if np.linalg.norm(pos - np.array(k.position)) < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append(c)
    return kept
