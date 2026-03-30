"""
src/feature_extractor.py — Per-candidate semantic-geometric features (Step 8)
===============================================================================
Migrated + rewritten from stage3/feature_extractor.py.

9 candidate-specific features for ranking:
    f1  detector_score         raw grasp quality
    f2  dist_target_3d         3D distance: grasp centre → target centroid
    f3  proj_dist_2d           2D distance: grasp projection → target mask centre
    f4  proj_overlap           projected overlap of grasp region with target mask
    f5  target_points_ratio    fraction of target points inside gripper region
    f6  nontarget_points_ratio fraction of non-target points inside gripper region
    f7  collision_risk         collision indicator from labels or heuristic
    f8  depth_consistency      how close grasp depth matches target depth
    f9  florence_conf          VLM grounding confidence (auxiliary)

All spatial computations use **camera frame**.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.point_cloud import project_to_image, compute_target_center
from src.grasp_detector import GraspCandidate


class FeatureExtractor:
    """Compute the 9-dim feature vector for each grasp candidate."""

    def __init__(self):
        self.feature_dim = config.FEATURE_DIM  # 9
        self.feature_names = config.FEATURE_NAMES

    def extract_batch(
        self,
        candidates: List[GraspCandidate],
        target_bbox: List[int],
        target_mask: Optional[np.ndarray],
        target_points: np.ndarray,
        scene_points: np.ndarray,
        scene_pixel_coords: np.ndarray,
        label: np.ndarray,
        target_mask_val: int,
        florence_conf: float,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        collision_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute features for all candidates.

        Returns (num_candidates, 9) float32 array.
        """
        if not candidates:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        target_center_3d = compute_target_center(target_points)

        # Pre-compute normalisation scales
        if len(target_points) > 0:
            dists_to_center = np.linalg.norm(
                target_points - target_center_3d, axis=1
            )
            d_max_3d = max(float(dists_to_center.max()) * 3.0, 0.1)
        else:
            d_max_3d = 1.0

        # Target depth stats
        x1, y1, x2, y2 = target_bbox
        target_depth_region = depth[
            max(0, y1):min(depth.shape[0], y2 + 1),
            max(0, x1):min(depth.shape[1], x2 + 1),
        ]
        valid_depth = target_depth_region[target_depth_region > 0]
        target_depth_mean = (
            float(valid_depth.mean()) if len(valid_depth) > 0 else 0.5
        )
        target_depth_max = (
            float(valid_depth.max()) if len(valid_depth) > 0 else 1.0
        )

        # Target mask centre in 2D
        if target_mask is not None:
            tys, txs = np.where(target_mask)
            if len(tys) > 0:
                target_center_2d = np.array(
                    [float(txs.mean()), float(tys.mean())]
                )
            else:
                target_center_2d = np.array(
                    [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                )
        else:
            target_center_2d = np.array(
                [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            )

        # Image diagonal for 2D distance normalisation
        img_diag = np.sqrt(
            config.IMAGE_HEIGHT ** 2 + config.IMAGE_WIDTH ** 2
        )

        features = []
        for c in candidates:
            feat = self._compute_single(
                c,
                target_bbox=target_bbox,
                target_mask=target_mask,
                target_center_3d=target_center_3d,
                target_center_2d=target_center_2d,
                target_points=target_points,
                scene_points=scene_points,
                scene_pixel_coords=scene_pixel_coords,
                label=label,
                target_mask_val=target_mask_val,
                florence_conf=florence_conf,
                depth=depth,
                intrinsics=intrinsics,
                d_max_3d=d_max_3d,
                target_depth_mean=target_depth_mean,
                target_depth_max=target_depth_max,
                img_diag=img_diag,
                collision_labels=collision_labels,
            )
            features.append(feat)

        return np.array(features, dtype=np.float32)

    def _compute_single(
        self,
        c: GraspCandidate,
        *,
        target_bbox,
        target_mask,
        target_center_3d,
        target_center_2d,
        target_points,
        scene_points,
        scene_pixel_coords,
        label,
        target_mask_val,
        florence_conf,
        depth,
        intrinsics,
        d_max_3d,
        target_depth_mean,
        target_depth_max,
        img_diag,
        collision_labels,
    ) -> List[float]:
        pos_3d = np.array(c.position)

        # ── f1: detector_score ───────────────────────────────────────
        f1 = c.detector_score

        # ── f2: dist_target_3d ───────────────────────────────────────
        dist_3d = float(np.linalg.norm(pos_3d - target_center_3d))
        f2 = min(dist_3d / d_max_3d, 1.0)

        # ── f3: proj_dist_2d ─────────────────────────────────────────
        uv = project_to_image(pos_3d.reshape(1, 3), intrinsics)[0]
        dist_2d = float(np.linalg.norm(uv - target_center_2d))
        f3 = min(dist_2d / (img_diag * 0.3), 1.0)

        # ── f4: proj_overlap ─────────────────────────────────────────
        f4 = self._compute_proj_overlap(
            pos_3d, c.width, intrinsics, target_bbox, target_mask,
        )

        # ── f5: target_points_ratio ──────────────────────────────────
        f5 = self._target_points_in_gripper(
            pos_3d, c.width, scene_points, scene_pixel_coords,
            label, target_mask_val, is_target=True,
        )

        # ── f6: nontarget_points_ratio ───────────────────────────────
        f6 = self._target_points_in_gripper(
            pos_3d, c.width, scene_points, scene_pixel_coords,
            label, target_mask_val, is_target=False,
        )

        # ── f7: collision_risk ───────────────────────────────────────
        if collision_labels is not None:
            f7 = self._collision_from_labels(c.candidate_id, collision_labels)
        else:
            f7 = self._collision_heuristic(
                pos_3d, c.width, scene_points,
            )

        # ── f8: depth_consistency ────────────────────────────────────
        d_cand = pos_3d[2]
        f8 = 1.0 - min(
            abs(d_cand - target_depth_mean) / max(target_depth_max, 0.1),
            1.0,
        )

        # ── f9: florence_conf ────────────────────────────────────────
        f9 = florence_conf

        return [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    # ── Feature helpers ──────────────────────────────────────────────

    def _compute_proj_overlap(
        self, pos_3d, width, intrinsics, bbox, mask,
    ) -> float:
        """Approximate overlap of gripper projection with target region."""
        uv = project_to_image(pos_3d.reshape(1, 3), intrinsics)[0]
        u, v = uv[0], uv[1]

        fx = intrinsics[0, 0]
        z = max(pos_3d[2], 0.01)
        radius_px = (width / 2.0) * fx / z

        gx1, gy1 = u - radius_px, v - radius_px
        gx2, gy2 = u + radius_px, v + radius_px

        x1, y1, x2, y2 = bbox
        ix1 = max(gx1, x1)
        iy1 = max(gy1, y1)
        ix2 = min(gx2, x2)
        iy2 = min(gy2, y2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_g = max(1, (gx2 - gx1) * (gy2 - gy1))
        area_t = max(1, (x2 - x1) * (y2 - y1))
        union = area_g + area_t - inter

        return float(inter / max(union, 1e-6))

    def _target_points_in_gripper(
        self, pos_3d, width, scene_points, scene_pixel_coords,
        label, target_mask_val, is_target: bool,
    ) -> float:
        """Fraction of target (or non-target) points inside gripper."""
        if len(scene_points) == 0:
            return 0.0

        gripper_radius = width * 0.6
        dists = np.linalg.norm(scene_points - pos_3d, axis=1)
        inside = dists < gripper_radius

        if not np.any(inside):
            return 0.0

        inside_idx = np.where(inside)[0]
        inside_px = scene_pixel_coords[inside_idx]

        H, W = label.shape[:2]
        u = np.clip(inside_px[:, 0], 0, W - 1)
        v = np.clip(inside_px[:, 1], 0, H - 1)

        on_target = label[v, u] == target_mask_val

        if is_target:
            return float(np.sum(on_target)) / max(float(np.sum(inside)), 1.0)
        else:
            return float(np.sum(~on_target)) / max(float(np.sum(inside)), 1.0)

    def _collision_heuristic(
        self, pos_3d, width, scene_points,
    ) -> float:
        """Estimate collision risk from non-target points near gripper."""
        if len(scene_points) == 0:
            return 0.0
        dists = np.linalg.norm(scene_points - pos_3d, axis=1)
        collision_radius = width * 0.6
        nearby = np.sum(dists < collision_radius)
        return min(float(nearby) / max(len(scene_points) * 0.01, 1), 1.0)

    def _collision_from_labels(
        self, candidate_id: int, collision_labels,
    ) -> float:
        """Extract collision flag from GraspNet collision labels."""
        # collision_labels format depends on GraspNet version
        try:
            if isinstance(collision_labels, np.ndarray):
                if candidate_id < len(collision_labels):
                    return float(collision_labels[candidate_id])
            return 0.0
        except (IndexError, TypeError):
            return 0.0
