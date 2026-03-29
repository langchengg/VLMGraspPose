"""
stage3/feature_extractor.py — Compute per-candidate feature vectors
====================================================================
Core features (f1–f5):
    f1  raw grasp score
    f2  is grasp centre inside target region (0/1)
    f3  normalised distance to target centre
    f4  IoU of grasp projection with target region
    f5  VLM confidence

Extended features (f6–f9):
    f6  depth consistency
    f7  collision risk
    f8  distance to mask boundary
    f9  grasp axis vs surface normal alignment

All spatial computations use **camera frame**.
Projection to image frame uses the intrinsic matrix.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.point_cloud import project_to_image, compute_target_center
from stage2.grasp_generator import GraspCandidate


class FeatureExtractor:
    """Compute feature vectors for a batch of grasp candidates."""

    def __init__(self, use_extended: bool = False):
        self.use_extended = use_extended
        self.feature_dim = (config.FEATURE_DIM_EXTENDED if use_extended
                            else config.FEATURE_DIM_CORE)

    def extract(
        self,
        candidates: List[GraspCandidate],
        target_bbox: List[int],
        target_mask: Optional[np.ndarray],
        target_points: np.ndarray,
        vlm_confidence: float,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        scene_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute features for all candidates.

        Returns (num_candidates, feature_dim) float32 array.
        """
        if not candidates:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        target_center_3d = compute_target_center(target_points)

        # Pre-compute max distance for normalisation
        if len(target_points) > 0:
            dists_to_center = np.linalg.norm(
                target_points - target_center_3d, axis=1
            )
            d_max = max(dists_to_center.max() * 3.0, 0.1)
        else:
            d_max = 1.0

        # Target depth stats
        x1, y1, x2, y2 = target_bbox
        target_depth_region = depth[y1:y2+1, x1:x2+1]
        valid_depth = target_depth_region[target_depth_region > 0]
        target_depth_mean = float(valid_depth.mean()) if len(valid_depth) > 0 else 0.5
        target_depth_max = float(valid_depth.max()) if len(valid_depth) > 0 else 1.0

        features = []

        for c in candidates:
            feat = self._compute_single(
                c, target_bbox, target_mask, target_center_3d,
                vlm_confidence, depth, intrinsics,
                d_max, target_depth_mean, target_depth_max,
                target_points, scene_points,
            )
            features.append(feat)

        return np.array(features, dtype=np.float32)

    def _compute_single(
        self,
        c: GraspCandidate,
        bbox: List[int],
        mask: Optional[np.ndarray],
        target_center: np.ndarray,
        vlm_conf: float,
        depth: np.ndarray,
        K: np.ndarray,
        d_max: float,
        target_depth_mean: float,
        target_depth_max: float,
        target_points: np.ndarray,
        scene_points: Optional[np.ndarray],
    ) -> List[float]:
        pos_3d = np.array(c.position)

        # ── f1: raw grasp score ──
        f1 = c.score

        # ── f2: is centre in target region ──
        f2 = self._is_in_target(pos_3d, K, bbox, mask)

        # ── f3: normalised distance to target centre ──
        dist = float(np.linalg.norm(pos_3d - target_center))
        f3 = min(dist / d_max, 1.0)

        # ── f4: IoU of grasp projection with target ──
        f4 = self._compute_iou(pos_3d, c.width, K, bbox, mask)

        # ── f5: VLM confidence ──
        f5 = vlm_conf

        feat = [f1, f2, f3, f4, f5]

        if self.use_extended:
            # ── f6: depth consistency ──
            d_candidate = pos_3d[2]  # z in camera frame
            f6 = 1.0 - min(abs(d_candidate - target_depth_mean) / max(target_depth_max, 0.1), 1.0)

            # ── f7: collision risk ──
            f7 = self._collision_risk(pos_3d, c.width, scene_points)

            # ── f8: distance to mask boundary ──
            f8 = self._boundary_distance(pos_3d, K, mask)

            # ── f9: grasp axis vs surface normal alignment ──
            f9 = self._normal_alignment(pos_3d, c.orientation, target_points)

            feat.extend([f6, f7, f8, f9])

        return feat

    # ── Feature helpers ──────────────────────────────────────────────

    def _is_in_target(
        self, pos_3d: np.ndarray, K: np.ndarray,
        bbox: List[int], mask: Optional[np.ndarray],
    ) -> float:
        """Project grasp centre to image; check if inside target region."""
        uv = project_to_image(pos_3d.reshape(1, 3), K)[0]
        u, v = int(round(uv[0])), int(round(uv[1]))

        if mask is not None:
            H, W = mask.shape
            if 0 <= u < W and 0 <= v < H:
                return 1.0 if mask[v, u] else 0.0
            return 0.0
        else:
            x1, y1, x2, y2 = bbox
            return 1.0 if (x1 <= u <= x2 and y1 <= v <= y2) else 0.0

    def _compute_iou(
        self, pos_3d: np.ndarray, width: float,
        K: np.ndarray, bbox: List[int], mask: Optional[np.ndarray],
    ) -> float:
        """Approximate IoU between grasp footprint and target region.

        The grasp footprint is estimated as a small box around the
        projected grasp centre with radius proportional to gripper width.
        """
        uv = project_to_image(pos_3d.reshape(1, 3), K)[0]
        u, v = uv[0], uv[1]

        # Approximate grasp footprint radius in pixels
        fx = K[0, 0]
        z = max(pos_3d[2], 0.01)
        radius_px = (width / 2.0) * fx / z

        gx1 = u - radius_px
        gy1 = v - radius_px
        gx2 = u + radius_px
        gy2 = v + radius_px

        x1, y1, x2, y2 = bbox

        # Intersection
        ix1 = max(gx1, x1)
        iy1 = max(gy1, y1)
        ix2 = min(gx2, x2)
        iy2 = min(gy2, y2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_g = max(1, (gx2 - gx1) * (gy2 - gy1))
        area_t = max(1, (x2 - x1) * (y2 - y1))
        union = area_g + area_t - inter

        return float(inter / max(union, 1e-6))

    def _collision_risk(
        self, pos_3d: np.ndarray, width: float,
        scene_points: Optional[np.ndarray],
    ) -> float:
        """Estimate collision risk: fraction of scene points within
        the gripper opening volume (simplified as a sphere)."""
        if scene_points is None or len(scene_points) == 0:
            return 0.0

        dists = np.linalg.norm(scene_points - pos_3d, axis=1)
        # Points within gripper opening radius but NOT on target
        collision_radius = width * 0.6
        nearby = np.sum(dists < collision_radius)
        # Normalise by expected empty space
        risk = min(float(nearby) / max(len(scene_points) * 0.01, 1), 1.0)
        return risk

    def _boundary_distance(
        self, pos_3d: np.ndarray, K: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> float:
        """Distance from projected grasp centre to nearest mask boundary.
        Normalised by mask diagonal. Higher = more stable."""
        if mask is None:
            return 0.5

        uv = project_to_image(pos_3d.reshape(1, 3), K)[0]
        u, v = int(round(uv[0])), int(round(uv[1]))

        H, W = mask.shape
        if not (0 <= u < W and 0 <= v < H):
            return 0.0

        # Simple distance transform approximation
        from scipy.ndimage import distance_transform_edt
        dist_map = distance_transform_edt(mask.astype(np.uint8))
        diag = np.sqrt(H**2 + W**2)

        d = dist_map[v, u]
        return float(min(d / (diag * 0.1), 1.0))

    def _normal_alignment(
        self, pos_3d: np.ndarray, orientation: List[float],
        target_points: np.ndarray,
    ) -> float:
        """Cosine similarity between grasp approach axis and local
        surface normal at the nearest target point."""
        if len(target_points) == 0:
            return 0.0

        # Find nearest target point
        dists = np.linalg.norm(target_points - pos_3d, axis=1)
        nearest_idx = np.argmin(dists)

        # Approximate normal from local neighbourhood
        from data.point_cloud import estimate_normals_pca
        k = min(10, len(target_points))
        # Just estimate for the nearest point
        _, idx = np.argsort(dists)[:k], np.argsort(dists)[:k]
        local_pts = target_points[idx]
        if len(local_pts) < 3:
            return 0.0

        cov = np.cov(local_pts.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
        except np.linalg.LinAlgError:
            return 0.0

        # Grasp approach axis from quaternion (x-axis of gripper frame)
        qx, qy, qz, qw = orientation
        # Rotation matrix column 0 = approach direction
        approach = np.array([
            1 - 2*(qy**2 + qz**2),
            2*(qx*qy + qw*qz),
            2*(qx*qz - qw*qy),
        ])

        cos_sim = abs(float(np.dot(approach, normal)))
        return min(cos_sim, 1.0)


# ── Persistence ──────────────────────────────────────────────────────

def save_features(
    sample_id: str,
    features: np.ndarray,
    candidate_ids: List[int],
    output_dir: Path = config.FEATURES_DIR,
) -> Tuple[Path, Path]:
    """Save feature matrix and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / f"{sample_id}.npy"
    meta_path = output_dir / f"{sample_id}_meta.json"

    np.save(str(npy_path), features)

    meta = {
        "sample_id": sample_id,
        "num_candidates": len(candidate_ids),
        "feature_dim": features.shape[1] if len(features) > 0 else 0,
        "candidate_ids": candidate_ids,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return npy_path, meta_path
