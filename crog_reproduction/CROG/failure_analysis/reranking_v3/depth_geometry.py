from __future__ import annotations

import numpy as np

from .aligned_crops import CROP_CHANNELS


DEPTH_FEATURE_NAMES = (
    "g9_valid_fraction", "g9_center_relative_depth_m", "g9_left_contact_relative_depth_m",
    "g9_right_contact_relative_depth_m", "g9_contact_depth_difference_m",
    "g9_contact_surface_symmetry", "g9_local_depth_gradient", "g9_surface_flatness",
    "g9_normal_proxy", "g9_axis_clearance", "g9_background_clearance",
    "g9_missing_fraction", "g9_mask_inside_outside_depth_separation",
)

DERIVED_DEPTH_FEATURE_NAMES = (
    "g9_left_endpoint_relative_depth_m",
    "g9_right_endpoint_relative_depth_m",
    "g9_finger_swept_volume_collision_proxy",
)


def derived_depth_features(crop: np.ndarray) -> np.ndarray:
    depth=np.asarray(crop[CROP_CHANNELS.index("relative_depth_m")],dtype=np.float64)
    valid=np.asarray(crop[CROP_CHANNELS.index("depth_valid")],dtype=np.float64)>.5
    left_finger=np.asarray(crop[CROP_CHANNELS.index("left_finger_template")],dtype=np.float64)>.5
    right_finger=np.asarray(crop[CROP_CHANNELS.index("right_finger_template")],dtype=np.float64)>.5
    size=depth.shape[-1]; center=size//2
    left_column=max(0,int(round((size-1)*.25))); right_column=min(size-1,int(round((size-1)*.75)))
    def local_endpoint(column: int) -> float:
        rows=slice(max(0,center-1),min(size,center+2)); selected=valid[rows,column]
        return float(depth[rows,column][selected].mean()) if selected.any() else 0.0
    finger=valid & (left_finger|right_finger)
    collision=float((np.abs(depth[finger])<=.02).mean()) if finger.any() else 0.0
    result=np.asarray((local_endpoint(left_column),local_endpoint(right_column),collision),dtype=np.float32)
    if not np.isfinite(result).all(): raise FloatingPointError("non-finite derived depth feature")
    return result


def depth_geometry_features(crop: np.ndarray) -> np.ndarray:
    depth = np.asarray(crop[CROP_CHANNELS.index("relative_depth_m")], dtype=np.float64)
    valid = np.asarray(crop[CROP_CHANNELS.index("depth_valid")], dtype=np.float64) > 0.5
    mask = np.asarray(crop[CROP_CHANNELS.index("mask_probability")], dtype=np.float64)
    size = depth.shape[-1]; center = size // 2
    axis = np.linspace(-1.0, 1.0, size); vv, uu = np.meshgrid(axis, axis, indexing="ij")
    left = valid & (uu < -0.35) & (uu > -0.75) & (np.abs(vv) < .25)
    right = valid & (uu > .35) & (uu < .75) & (np.abs(vv) < .25)
    def mean(where): return float(depth[where].mean()) if where.any() else 0.0
    left_mean, right_mean = mean(left), mean(right)
    gy, gx = np.gradient(np.where(valid, depth, 0.0))
    valid_values = depth[valid]
    inside = valid & (mask >= .5); outside = valid & (mask < .5)
    values = [
        valid.mean(), depth[center,center] if valid[center,center] else 0.0,
        left_mean, right_mean, abs(left_mean-right_mean),
        1.0-abs(left_mean-right_mean)/(np.std(valid_values)+1e-6) if len(valid_values) else 0.0,
        np.hypot(gx,gy)[valid].mean() if valid.any() else 0.0,
        np.std(valid_values) if len(valid_values) else 0.0,
        np.median(np.hypot(gx,gy)[valid]) if valid.any() else 0.0,
        np.quantile(valid_values,.9)-np.quantile(valid_values,.1) if len(valid_values) else 0.0,
        mean(valid & ((np.abs(uu)>.8)|(np.abs(vv)>.8))),
        1.0-valid.mean(),
        abs(mean(inside)-mean(outside)) if inside.any() and outside.any() else 0.0,
    ]
    result = np.asarray(values, dtype=np.float32)
    result[~np.isfinite(result)] = 0.0
    return result
