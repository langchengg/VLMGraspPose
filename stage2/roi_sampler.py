"""
stage2/roi_sampler.py — Target-Region Local Sampler
====================================================
A thin wrapper that:
  1. Crops the scene point cloud to the target region
  2. Feeds the cropped cloud into the grasp generator
  3. Returns filtered candidates

This is the "target-aware" approach (方式3) from the task spec.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.point_cloud import (
    backproject_depth,
    crop_point_cloud_by_bbox,
    crop_point_cloud_by_mask,
    voxel_downsample,
    estimate_normals_pca,
)
from stage2.grasp_generator import AntipodalGraspSampler, GraspCandidate


def generate_target_grasps(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    bbox: List[int],
    label: Optional[np.ndarray] = None,
    instance_id: Optional[int] = None,
    factor_depth: float = 1.0,
    top_k: int = config.GRASP_TOP_K,
    voxel_size: float = config.VOXEL_SIZE,
) -> List[GraspCandidate]:
    """End-to-end: depth + target region → grasp candidates.

    Parameters
    ----------
    depth : (H, W) float32 depth in metres (already divided by factor)
    intrinsics : (3, 3)
    bbox : [x1, y1, x2, y2]
    label : optional instance segmentation mask
    instance_id : pixel value for target in label mask
    factor_depth : if depth is still raw uint16, divide first
    top_k : number of candidates to return
    voxel_size : for point cloud down-sampling

    Returns
    -------
    List of GraspCandidate, sorted by score descending.
    """
    # 1. Back-project full depth to point cloud
    points, pixel_coords = backproject_depth(depth, intrinsics)

    if len(points) == 0:
        return []

    # 2. Crop to target region
    if label is not None and instance_id is not None:
        target_pts, target_px = crop_point_cloud_by_mask(
            points, pixel_coords, label, instance_id
        )
        # Fallback to bbox if mask yields too few points
        if len(target_pts) < 20:
            target_pts, target_px = crop_point_cloud_by_bbox(
                points, pixel_coords, bbox
            )
    else:
        target_pts, target_px = crop_point_cloud_by_bbox(
            points, pixel_coords, bbox
        )

    if len(target_pts) < 10:
        return []

    # 3. Down-sample
    target_pts = voxel_downsample(target_pts, voxel_size)

    # 4. Estimate normals
    normals = estimate_normals_pca(
        target_pts,
        k=min(config.NORMAL_MAX_NN, len(target_pts)),
    )

    # 5. Generate grasp candidates
    sampler = AntipodalGraspSampler(top_k=top_k)
    candidates = sampler.generate(target_pts, normals)

    return candidates


# ── Persistence ──────────────────────────────────────────────────────

def save_stage2_output(
    sample_id: str,
    candidates: List[GraspCandidate],
    output_dir: Path = config.STAGE2_OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sample_id}.json"

    record = {
        "sample_id": sample_id,
        "num_candidates": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
    }

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return out_path


def load_stage2_output(
    sample_id: str,
    output_dir: Path = config.STAGE2_OUTPUT_DIR,
) -> List[GraspCandidate]:
    path = output_dir / f"{sample_id}.json"
    with open(path) as f:
        record = json.load(f)

    candidates = []
    for c in record["candidates"]:
        candidates.append(GraspCandidate(**c))
    return candidates
