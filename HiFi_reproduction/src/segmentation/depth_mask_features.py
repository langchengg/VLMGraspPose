"""Depth and lightweight 3-D consistency features for binary proposals."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _valid_depth(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    return np.isfinite(value) & (value > 0.0)


def _seed_connected_fraction(mask: np.ndarray, depth_m: np.ndarray, median: float) -> float:
    """Fraction connected to an interior seed under a robust local-depth band."""

    distance = ndimage.distance_transform_edt(mask)
    if not np.any(distance > 0.0):
        return 0.0
    seed = tuple(int(value) for value in np.unravel_index(np.argmax(distance), mask.shape))
    seed_depth = float(depth_m[seed])
    if not np.isfinite(seed_depth) or seed_depth <= 0.0:
        seed_depth = median
    values = depth_m[mask & _valid_depth(depth_m)]
    mad = float(np.median(np.abs(values - median))) if len(values) else 0.0
    tolerance = max(0.015, 3.0 * 1.4826 * mad)
    compatible = mask & _valid_depth(depth_m) & (np.abs(depth_m - seed_depth) <= tolerance)
    labels, count = ndimage.label(compatible, structure=np.ones((3, 3), dtype=bool))
    if count == 0 or labels[seed] == 0:
        return 0.0
    connected = labels == labels[seed]
    return float(np.count_nonzero(connected) / max(np.count_nonzero(mask), 1))


def depth_mask_features(
    mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    intrinsics: dict[str, float] | None = None,
    reference_mask: np.ndarray | None = None,
) -> dict[str, float | bool]:
    mask = np.asarray(mask, dtype=bool)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if mask.shape != depth_m.shape:
        raise ValueError("mask and depth must be aligned")
    valid = _valid_depth(depth_m)
    inside = mask & valid
    values = depth_m[inside]
    validity = bool(len(values))
    if not validity:
        return {
            "depth_features_valid": False,
            "depth_features_missing": True,
            "depth_reliability": 0.0,
            "valid_depth_fraction": 0.0,
            "median_depth": float("nan"),
            "depth_iqr": float("nan"),
            "depth_std": float("nan"),
            "robust_depth_range": float("nan"),
            "seed_depth_connected_fraction": float("nan"),
            "low_depth_contamination_fraction": float("nan"),
            "depth_boundary_discontinuity": float("nan"),
            "foreground_background_depth_separation": float("nan"),
            "major_depth_cluster_count": 0.0,
            "point_cloud_compactness": float("nan"),
            "point_cloud_centroid_x": float("nan"),
            "point_cloud_centroid_y": float("nan"),
            "point_cloud_centroid_z": float("nan"),
            "point_cloud_extent_x": float("nan"),
            "point_cloud_extent_y": float("nan"),
            "point_cloud_extent_z": float("nan"),
            "reference_centroid_3d_distance": float("nan"),
        }
    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    median = float(np.median(values))
    depth_band = max(0.02, float(q90 - q10))
    low_depth_contamination = float(np.mean(values < median - depth_band))
    boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    outer = ndimage.binary_dilation(mask, iterations=2) & ~mask & valid
    inner_values = depth_m[boundary & valid]
    outer_values = depth_m[outer]
    separation = (
        float(abs(np.median(outer_values) - median)) if len(outer_values) else float("nan")
    )
    discontinuity = (
        float(abs(np.median(inner_values) - np.median(outer_values)))
        if len(inner_values) and len(outer_values)
        else float("nan")
    )
    histogram, _ = np.histogram(values, bins=min(16, max(4, int(np.sqrt(len(values))))))
    peaks = ndimage.maximum_filter1d(histogram.astype(np.float32), size=3) == histogram
    major_clusters = int(np.count_nonzero(peaks & (histogram >= max(3, 0.1 * histogram.max()))))

    yy, xx = np.nonzero(inside)
    fx = float((intrinsics or {}).get("fx", (intrinsics or {}).get("focal_x", 525.0)))
    fy = float((intrinsics or {}).get("fy", (intrinsics or {}).get("focal_y", 525.0)))
    cx = float((intrinsics or {}).get("cx", (depth_m.shape[1] - 1) / 2.0))
    cy = float((intrinsics or {}).get("cy", (depth_m.shape[0] - 1) / 2.0))
    z = values.astype(np.float64)
    x = (xx.astype(np.float64) - cx) * z / max(fx, 1e-6)
    y = (yy.astype(np.float64) - cy) * z / max(fy, 1e-6)
    points = np.column_stack((x, y, z))
    centre = np.median(points, axis=0)
    radii = np.linalg.norm(points - centre, axis=1)
    extent = np.quantile(points, 0.95, axis=0) - np.quantile(points, 0.05, axis=0)

    reference_distance = float("nan")
    if reference_mask is not None:
        ref = np.asarray(reference_mask, dtype=bool) & valid
        ryy, rxx = np.nonzero(ref)
        if len(rxx):
            rz = depth_m[ref].astype(np.float64)
            rpoints = np.column_stack(
                ((rxx - cx) * rz / max(fx, 1e-6), (ryy - cy) * rz / max(fy, 1e-6), rz)
            )
            reference_distance = float(np.linalg.norm(centre - np.median(rpoints, axis=0)))

    return {
        "depth_features_valid": True,
        "depth_features_missing": False,
        "depth_reliability": float(len(values) / max(int(np.count_nonzero(mask)), 1)),
        "valid_depth_fraction": float(len(values) / max(int(np.count_nonzero(mask)), 1)),
        "median_depth": median,
        "depth_iqr": float(q75 - q25),
        "depth_std": float(np.std(values)),
        "robust_depth_range": float(q90 - q10),
        "seed_depth_connected_fraction": _seed_connected_fraction(mask, depth_m, median),
        "low_depth_contamination_fraction": low_depth_contamination,
        "depth_boundary_discontinuity": discontinuity,
        "foreground_background_depth_separation": separation,
        "major_depth_cluster_count": float(major_clusters),
        "point_cloud_compactness": float(np.quantile(radii, 0.90)),
        "point_cloud_centroid_x": float(centre[0]),
        "point_cloud_centroid_y": float(centre[1]),
        "point_cloud_centroid_z": float(centre[2]),
        "point_cloud_extent_x": float(extent[0]),
        "point_cloud_extent_y": float(extent[1]),
        "point_cloud_extent_z": float(extent[2]),
        "reference_centroid_3d_distance": reference_distance,
    }


__all__ = ["depth_mask_features"]
