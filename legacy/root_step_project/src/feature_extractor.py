"""
src/feature_extractor.py — Per-candidate semantic-geometric features (Step 8)
===============================================================================
Migrated + rewritten from stage3/feature_extractor.py.

10 candidate-specific features for target-conditioned ranking:
    f1   target_overlap
    f2   center_alignment
    f3   distance_to_target_center
    f4   gripper_width_match
    f5   approach_direction_score
    f6   depth_stability
    f7   collision_penalty
    f8   boundary_penalty
    f9   initial_geometric_score
    f10  grounding_score

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
    """Compute the target-conditioned feature vector for each candidate."""

    def __init__(self, max_scene_points: int = config.FEATURE_MAX_SCENE_POINTS):
        self.feature_dim = config.FEATURE_DIM
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
        grounding_score: float,
        depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """Compute features for all candidates.

        Returns (num_candidates, config.FEATURE_DIM) float32 array.
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
            target_extent = np.ptp(target_points, axis=0)
            target_width_scale = max(float(np.median(target_extent[:2])), 1e-3)
        else:
            d_max_3d = 1.0
            target_width_scale = config.GRASP_MAX_WIDTH

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
                grounding_score=grounding_score,
                depth=depth,
                intrinsics=intrinsics,
                d_max_3d=d_max_3d,
                target_depth_mean=target_depth_mean,
                target_depth_max=target_depth_max,
                img_diag=img_diag,
                target_width_scale=target_width_scale,
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
        grounding_score,
        depth,
        intrinsics,
        d_max_3d,
        target_depth_mean,
        target_depth_max,
        img_diag,
        target_width_scale,
    ) -> List[float]:
        pos_3d = np.array(c.position)
        dist_3d = float(np.linalg.norm(pos_3d - target_center_3d))
        uv = project_to_image(pos_3d.reshape(1, 3), intrinsics)[0]

        # f1: target_overlap
        target_overlap = self._compute_proj_overlap(
            pos_3d, c.width, intrinsics, target_bbox, target_mask,
        )

        # f2/f3: center alignment and normalized 3D distance
        distance_to_target_center = min(dist_3d / d_max_3d, 1.0)
        center_alignment = 1.0 - distance_to_target_center

        # f4: gripper width compatibility with target size
        gripper_width_match = self._gripper_width_match(c.width, target_width_scale)

        # f5: approach direction consistency
        approach_direction_score = self._approach_direction_score(c, pos_3d)

        # f6: depth stability
        d_cand = pos_3d[2]
        depth_stability = 1.0 - min(
            abs(d_cand - target_depth_mean) / max(target_depth_max, 0.1),
            1.0,
        )

        # f7: collision penalty
        collision_penalty = self._collision_heuristic(
            pos_3d, c.width, scene_points,
        )

        # f8: boundary penalty
        boundary_penalty = self._boundary_penalty(
            uv, target_bbox, target_mask,
        )

        # f9/f10: sampler score and grounding confidence
        initial_geometric_score = float(c.detector_score)
        grounding_score = float(grounding_score)

        return [
            target_overlap,
            center_alignment,
            distance_to_target_center,
            gripper_width_match,
            approach_direction_score,
            depth_stability,
            collision_penalty,
            boundary_penalty,
            initial_geometric_score,
            grounding_score,
        ]

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

    def _gripper_width_match(self, width: float, target_width_scale: float) -> float:
        desired = np.clip(target_width_scale * 1.15, self._min_width(), self._max_width())
        denom = max(self._max_width() - self._min_width(), 1e-6)
        return float(np.clip(1.0 - abs(width - desired) / denom, 0.0, 1.0))

    def _approach_direction_score(self, candidate: GraspCandidate, pos_3d: np.ndarray) -> float:
        approach = np.asarray(candidate.approach_vector, dtype=np.float32)
        if np.linalg.norm(approach) < 1e-6:
            approach = candidate.rotation_matrix[:, 0]
        approach = approach / (np.linalg.norm(approach) + 1e-8)
        view_dir = -pos_3d / (np.linalg.norm(pos_3d) + 1e-8)
        return float(np.clip(np.dot(-approach, view_dir), 0.0, 1.0))

    def _boundary_penalty(
        self,
        uv: np.ndarray,
        bbox: list,
        mask: Optional[np.ndarray],
    ) -> float:
        u, v = float(uv[0]), float(uv[1])
        if mask is not None and mask.any():
            try:
                import cv2

                dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
                H, W = mask.shape[:2]
                ui = int(np.clip(round(u), 0, W - 1))
                vi = int(np.clip(round(v), 0, H - 1))
                max_dist = max(float(dist.max()), 1.0)
                return float(1.0 - np.clip(dist[vi, ui] / max_dist, 0.0, 1.0))
            except Exception:
                pass

        x1, y1, x2, y2 = bbox
        if u < x1 or u > x2 or v < y1 or v > y2:
            return 1.0
        dist_to_edge = min(u - x1, x2 - u, v - y1, y2 - v)
        norm = max(min(x2 - x1, y2 - y1) * 0.5, 1.0)
        return float(1.0 - np.clip(dist_to_edge / norm, 0.0, 1.0))

    @staticmethod
    def _min_width() -> float:
        return config.GRASP_MIN_WIDTH

    @staticmethod
    def _max_width() -> float:
        return config.GRASP_MAX_WIDTH
