"""
src/feature_extractor.py — Per-candidate semantic-geometric features (Step 8)
===============================================================================
Migrated + rewritten from stage3/feature_extractor.py.

9 candidate-specific features for ranking:
    f1  detector_score         raw grasp quality
    f2  dist_target_3d         3D distance: grasp centre → target centroid
    f3  proj_dist_2d           2D distance: grasp projection → target centre
    f4  proj_overlap           projected overlap (IoU) with target region
    f5  target_points_ratio    fraction of target points inside gripper
    f6  nontarget_points_ratio fraction of non-target points inside gripper
    f7  collision_risk         collision indicator (detector score proxy)
    f8  depth_consistency      how close grasp depth matches target depth
    f9  florence_conf          VLM grounding confidence (placeholder)

Informative features by grounding mode:
    Mode           Informative         Constant / degraded
    oracle (GT)    f1–f8               f9=1.0
    seg (mask)     f1–f8               f9=1.0
    phrase (bbox)  f1–f4, f7, f8       f5=f6=0.5, f9=1.0

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

    def __init__(self, max_scene_points: int = config.FEATURE_MAX_SCENE_POINTS):
        self.feature_dim = config.FEATURE_DIM  # 9
        self.feature_names = config.FEATURE_NAMES
        self.max_scene_points = max_scene_points

    def _downsample_scene_context(
        self,
        scene_points: np.ndarray,
        scene_pixel_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound scene-point count before repeated per-candidate geometry."""
        if len(scene_points) <= self.max_scene_points:
            return scene_points, scene_pixel_coords

        rng = np.random.RandomState(42)
        idx = rng.choice(
            len(scene_points),
            size=self.max_scene_points,
            replace=False,
        )
        idx = np.sort(idx)
        return scene_points[idx], scene_pixel_coords[idx]

    def extract_batch(
        self,
        candidates: List[GraspCandidate],
        target_bbox: List[int],
        target_mask: Optional[np.ndarray],
        target_points: np.ndarray,
        scene_points: np.ndarray,
        scene_pixel_coords: np.ndarray,
        florence_conf: float,
        depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """Compute features for all candidates.

        Returns (num_candidates, 9) float32 array.
        """
        if not candidates:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        scene_points, scene_pixel_coords = self._downsample_scene_context(
            scene_points, scene_pixel_coords,
        )
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
                florence_conf=florence_conf,
                depth=depth,
                intrinsics=intrinsics,
                d_max_3d=d_max_3d,
                target_depth_mean=target_depth_mean,
                target_depth_max=target_depth_max,
                img_diag=img_diag,
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
        florence_conf,
        depth,
        intrinsics,
        d_max_3d,
        target_depth_mean,
        target_depth_max,
        img_diag,
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
        # Uses target_mask (GT in oracle mode, predicted in VLM mode)
        f5 = self._target_points_in_gripper(
            pos_3d, c.width, scene_points, scene_pixel_coords,
            target_mask, is_target=True,
        )

        # ── f6: nontarget_points_ratio ───────────────────────────────
        f6 = self._target_points_in_gripper(
            pos_3d, c.width, scene_points, scene_pixel_coords,
            target_mask, is_target=False,
        )

        # ── f7: collision_risk (always geometry heuristic) ───────────
        # Note: official collision labels are per-grasp-configuration
        # and can't be indexed by candidate_id. Use heuristic instead.
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
        # NOTE: Currently non-informative (constant 1.0).
        # Florence-2 does not output per-prediction confidence scores.
        # This feature channel is kept as a placeholder for future
        # grounding models that do provide confidence.  It currently
        # acts as a bias term and has zero predictive contribution.
        f9 = florence_conf

        return [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    # ── Feature helpers ──────────────────────────────────────────────

    def _compute_proj_overlap(
        self, pos_3d, width, intrinsics, bbox, mask,
    ) -> float:
        """Approximate overlap of gripper projection with target region.

        When a binary mask is available, computes the fraction of the
        gripper's circular footprint that overlaps with the mask pixels.
        Falls back to bbox-vs-bbox IoU when no mask is provided.
        """
        uv = project_to_image(pos_3d.reshape(1, 3), intrinsics)[0]
        u, v = uv[0], uv[1]

        fx = intrinsics[0, 0]
        z = max(pos_3d[2], 0.01)
        radius_px = (width / 2.0) * fx / z

        # ── Mask-based overlap (when available) ─────────────────────
        if mask is not None:
            H, W = mask.shape[:2]
            # Create gripper footprint region
            gx1 = int(max(0, u - radius_px))
            gy1 = int(max(0, v - radius_px))
            gx2 = int(min(W, u + radius_px))
            gy2 = int(min(H, v + radius_px))

            if gx2 <= gx1 or gy2 <= gy1:
                return 0.0

            gripper_patch = mask[gy1:gy2, gx1:gx2]
            mask_area = float(gripper_patch.sum())
            gripper_area = max(1.0, (gx2 - gx1) * (gy2 - gy1))
            total_mask = float(mask.sum())

            if total_mask < 1:
                return 0.0

            # IoU: intersection / union
            inter = mask_area
            union = gripper_area + total_mask - inter
            return float(inter / max(union, 1e-6))

        # ── Bbox-based overlap (fallback) ───────────────────────────
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
        target_mask: Optional[np.ndarray], is_target: bool,
    ) -> float:
        """Fraction of target (or non-target) points inside gripper.

        Uses `target_mask` (HxW bool) which can be either:
          - GT mask (oracle mode)
          - Florence-2 predicted mask (predicted mode)
          - None → fall back to 0.0 / 1.0 defaults
        """
        if len(scene_points) == 0:
            return 0.0

        gripper_radius = width * 0.6
        dists = np.linalg.norm(scene_points - pos_3d, axis=1)
        inside = dists < gripper_radius

        if not np.any(inside):
            return 0.0

        # If no mask available, we can't distinguish target/non-target
        if target_mask is None:
            return 0.5 if is_target else 0.5

        inside_idx = np.where(inside)[0]
        inside_px = scene_pixel_coords[inside_idx]

        H, W = target_mask.shape[:2]
        u = np.clip(inside_px[:, 0], 0, W - 1)
        v = np.clip(inside_px[:, 1], 0, H - 1)

        on_target = target_mask[v, u]

        if is_target:
            return float(np.sum(on_target)) / max(float(np.sum(inside)), 1.0)
        else:
            return float(np.sum(~on_target)) / max(float(np.sum(inside)), 1.0)

    def _collision_heuristic(
        self, pos_3d, width, scene_points,
    ) -> float:
        """Estimate collision risk from point density near gripper."""
        if len(scene_points) == 0:
            return 0.0
        dists = np.linalg.norm(scene_points - pos_3d, axis=1)
        collision_radius = width * 0.6
        nearby = np.sum(dists < collision_radius)
        return min(float(nearby) / max(len(scene_points) * 0.01, 1), 1.0)
