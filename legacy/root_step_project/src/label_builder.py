"""
src/label_builder.py — Target-aware training label generation (Step 7)
========================================================================
Rewritten from stage4/label_generator.py.

A candidate is POSITIVE only if:
  1. It is associated with the TARGET object, AND
  2. Its detector score >= 0.3 (proxy for grasp quality).

NOTE on collision labels: Official GraspNet collision labels exist but
are indexed by (object × angle × depth) grasp configurations, NOT by
detector candidate_id.  Implementing a matching step would require
nearest-neighbour search in SE(3) grasp space.  Until that is done,
detector score serves as a quality proxy. This means the supervision
signal captures "target + high detector confidence", not "target +
physically collision-free".

Object association uses the official GraspNet evaluator logic:
identify which object occupies the points inside the gripper.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.grasp_detector import GraspCandidate


# ═════════════════════════════════════════════════════════════════════
#  Object association
# ═════════════════════════════════════════════════════════════════════

def associate_grasp_to_object(
    candidate: GraspCandidate,
    scene_points: np.ndarray,
    scene_pixel_coords: np.ndarray,
    label: np.ndarray,
) -> Optional[int]:
    """Determine which object a grasp candidate is associated with.

    Identifies the object that has the most points inside the gripper
    closing region.  Returns the object mask_val (= obj_id + 1), or
    None if no object is dominant.
    """
    pos_3d = np.array(candidate.position)
    gripper_radius = candidate.width * 0.6

    dists = np.linalg.norm(scene_points - pos_3d, axis=1)
    inside = dists < gripper_radius

    if not np.any(inside):
        return None

    inside_idx = np.where(inside)[0]
    inside_px = scene_pixel_coords[inside_idx]

    H, W = label.shape[:2]
    u = np.clip(inside_px[:, 0], 0, W - 1)
    v = np.clip(inside_px[:, 1], 0, H - 1)

    labels_inside = label[v, u]

    # Exclude background (0) and table
    object_labels = labels_inside[labels_inside > 0]
    if len(object_labels) == 0:
        return None

    # Most frequent object
    vals, counts = np.unique(object_labels, return_counts=True)
    dominant_val = int(vals[np.argmax(counts)])
    dominant_ratio = float(counts.max()) / float(len(object_labels))

    # Require at least 30% dominance
    if dominant_ratio < 0.3:
        return None

    return dominant_val


# ═════════════════════════════════════════════════════════════════════
#  Label generation
# ═════════════════════════════════════════════════════════════════════

def generate_candidate_label(
    candidate: GraspCandidate,
    target_mask_val: int,
    scene_points: np.ndarray,
    scene_pixel_coords: np.ndarray,
    label: np.ndarray,
    collision_label: Optional[float] = None,
    collision_thresh: float = config.LABEL_COLLISION_THRESH,
) -> dict:
    """Generate a training label for one candidate.

    Returns
    -------
    dict with keys:
        candidate_id, target_mask_val, associated_object_val,
        is_collision_free, label (0 or 1)
    """
    # Step 1: Object association
    associated_val = associate_grasp_to_object(
        candidate, scene_points, scene_pixel_coords, label,
    )

    is_on_target = (associated_val is not None and
                    associated_val == target_mask_val)

    # Step 2: Collision check
    if collision_label is not None:
        is_collision_free = collision_label < collision_thresh
    else:
        # If no collision labels, assume valid if detector score > 0.3
        is_collision_free = candidate.detector_score >= 0.3

    # Step 3: Final label
    is_positive = is_on_target and is_collision_free

    return {
        "candidate_id": candidate.candidate_id,
        "target_mask_val": int(target_mask_val),
        "associated_object_val": int(associated_val) if associated_val else -1,
        "is_collision_free": int(is_collision_free),
        "label": int(is_positive),
    }


def generate_labels_for_sample(
    candidates: List[GraspCandidate],
    target_mask_val: int,
    scene_points: np.ndarray,
    scene_pixel_coords: np.ndarray,
    label: np.ndarray,
    collision_labels: Optional[np.ndarray] = None,
) -> List[dict]:
    """Generate labels for all candidates in a sample.

    NOTE on collision_labels: Official GraspNet collision labels are indexed
    by pre-defined grasp configurations (object × angle × depth), NOT by
    detector candidate_id.  We cannot directly index them by candidate_id.

    Two approaches:
      1. If candidates come from the official grasp set with matching indices,
         collision_labels can be used (not implemented — needs grasp matching).
      2. Otherwise (local geometric sampler output),
         we use the detector score as a quality proxy instead.

    Returns list of label dicts, one per candidate.
    """
    results = []
    for i, c in enumerate(candidates):
        # Collision labels cannot be naively indexed by candidate_id.
        # Use detector score as proxy for grasp quality / collision-safety.
        lbl = generate_candidate_label(
            c, target_mask_val,
            scene_points, scene_pixel_coords, label,
            collision_label=None,  # don't index by candidate_id
        )
        results.append(lbl)

    return results


# ═════════════════════════════════════════════════════════════════════
#  Pseudo-label fallback (when no official labels available)
# ═════════════════════════════════════════════════════════════════════

def generate_pseudo_labels(
    features: np.ndarray,
    candidates: List[GraspCandidate],
    target_mask_val: int,
    scene_points: np.ndarray,
    scene_pixel_coords: np.ndarray,
    label: np.ndarray,
) -> np.ndarray:
    """Generate pseudo-labels using object association + detector score.

    Positive if:
      • Candidate is associated with the target object
      • Detector score >= 0.3

    Returns (N,) int32 array of labels.
    """
    N = len(candidates)
    labels = np.zeros(N, dtype=np.int32)

    for i, c in enumerate(candidates):
        associated_val = associate_grasp_to_object(
            c, scene_points, scene_pixel_coords, label,
        )
        is_on_target = (associated_val is not None and
                        associated_val == target_mask_val)
        is_good_score = c.detector_score >= 0.3

        labels[i] = 1 if (is_on_target and is_good_score) else 0

    return labels
