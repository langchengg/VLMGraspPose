"""Leak-free GT target-consistency labels for frozen VGN candidates."""

from __future__ import annotations

import json
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from src.grasping.vgn_geometry import dilate_mask


class GTLabelError(RuntimeError):
    pass


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GTContext:
    masks: dict[int, np.ndarray]
    depth_m: np.ndarray
    signed_distance: np.ndarray
    target_points_tree: cKDTree
    intrinsics: Mapping[str, Any]


def _binary_image(path: str | Path) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise GTLabelError(f"missing mask: {source}")
    with Image.open(source) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise GTLabelError(f"mask is not 2-D: {source} -> {array.shape}")
    return np.asarray(array > 0, dtype=bool)


def _depth_m(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise GTLabelError(f"missing depth: {source}")
    with Image.open(source) as image:
        depth = np.asarray(image)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape != shape:
        raise GTLabelError(f"depth/mask shape mismatch: {depth.shape} != {shape}")
    # The frozen run metadata explicitly records raw OCID depth as millimetres.
    return np.asarray(depth, dtype=np.float32) / 1000.0


def project_position(
    position_camera_m: np.ndarray, intrinsics: Mapping[str, Any]
) -> tuple[float, float] | None:
    position = np.asarray(position_camera_m, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)) or position[2] <= 0:
        return None
    u = float(intrinsics["fx"]) * position[0] / position[2] + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * position[1] / position[2] + float(intrinsics["cy"])
    return (float(u), float(v)) if np.isfinite(u) and np.isfinite(v) else None


def _sample_mask(mask: np.ndarray, u: float, v: float) -> bool:
    pixel_u, pixel_v = int(np.rint(u)), int(np.rint(v))
    return bool(
        0 <= pixel_u < mask.shape[1]
        and 0 <= pixel_v < mask.shape[0]
        and mask[pixel_v, pixel_u]
    )


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    # Positive inside, negative outside. Pixel centres on either side of a
    # boundary have magnitude one under scipy's Euclidean distance transform.
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    return np.asarray(inside - outside, dtype=np.float32)


def _gt_points(
    mask: np.ndarray, depth_m: np.ndarray, intrinsics: Mapping[str, Any]
) -> np.ndarray:
    valid = mask & np.isfinite(depth_m) & (depth_m > 0)
    v, u = np.nonzero(valid)
    if not len(u):
        raise GTLabelError("GT target mask contains no valid metric depth")
    z = depth_m[v, u]
    return np.column_stack(
        (
            (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"]),
            (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"]),
            z,
        )
    )


def _build_context(sample: Mapping[str, Any], gt: np.ndarray | None = None) -> _GTContext:
    gt = _binary_image(str(sample["gt_mask_path"])) if gt is None else gt
    depth = _depth_m(str(sample["depth_path"]), gt.shape)
    intrinsics = sample["intrinsics"]
    if isinstance(intrinsics, str):
        intrinsics = json.loads(intrinsics)
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        if key not in intrinsics:
            raise GTLabelError(f"sample {sample['sample_id']} lacks intrinsics {key}")
    if (int(intrinsics["height"]), int(intrinsics["width"])) != gt.shape:
        raise GTLabelError(f"sample {sample['sample_id']} intrinsics/mask shape mismatch")
    masks = {
        0: gt,
        3: dilate_mask(gt, 3),
        5: dilate_mask(gt, 5),
        10: dilate_mask(gt, 10),
    }
    points = _gt_points(gt, depth, intrinsics)
    return _GTContext(
        masks=masks,
        depth_m=depth,
        signed_distance=_signed_distance(gt),
        target_points_tree=cKDTree(points),
        intrinsics=intrinsics,
    )


def label_candidate_group(
    sample: Mapping[str, Any], candidates: pd.DataFrame, *, _context: _GTContext | None = None
) -> pd.DataFrame:
    """Label one candidate group using only pose, GT mask, depth, intrinsics.

    Predicted masks are intentionally not accepted by this API, which prevents
    the predicted hard-filter decision from leaking into GT labels.
    """

    result = candidates.copy()
    if result.empty:
        for name, dtype in (
            ("projected_u", float),
            ("projected_v", float),
            ("gt_inside_raw_mask", bool),
            ("gt_inside_dilated_mask_3px", bool),
            ("gt_inside_dilated_mask_5px", bool),
            ("gt_inside_dilated_mask_10px", bool),
            ("gt_signed_distance_px", float),
            ("nearest_gt_target_point_distance_m", float),
            ("projected_depth_difference_m", float),
        ):
            result[name] = pd.Series(dtype=dtype)
        return result
    context = _context or _build_context(sample)
    masks = context.masks
    gt = masks[0]
    depth = context.depth_m
    intrinsics = context.intrinsics
    positions = result[
        ["position_camera_x", "position_camera_y", "position_camera_z"]
    ].to_numpy(dtype=np.float64)
    nearest = context.target_points_tree.query(positions, k=1)[0]

    projected_u: list[float] = []
    projected_v: list[float] = []
    inside: dict[int, list[bool]] = {radius: [] for radius in masks}
    signed: list[float] = []
    depth_difference: list[float] = []
    for position in positions:
        uv = project_position(position, intrinsics)
        if uv is None:
            projected_u.append(float("nan"))
            projected_v.append(float("nan"))
            for radius in masks:
                inside[radius].append(False)
            signed.append(float("nan"))
            depth_difference.append(float("nan"))
            continue
        u, v = uv
        projected_u.append(u)
        projected_v.append(v)
        for radius, mask in masks.items():
            inside[radius].append(_sample_mask(mask, u, v))
        pixel_u, pixel_v = int(np.rint(u)), int(np.rint(v))
        in_image = 0 <= pixel_u < gt.shape[1] and 0 <= pixel_v < gt.shape[0]
        signed.append(
            float(context.signed_distance[pixel_v, pixel_u])
            if in_image
            else float("nan")
        )
        observed = float(depth[pixel_v, pixel_u]) if in_image else float("nan")
        depth_difference.append(
            float(position[2] - observed)
            if np.isfinite(observed) and observed > 0
            else float("nan")
        )
    result["projected_u"] = projected_u
    result["projected_v"] = projected_v
    result["gt_inside_raw_mask"] = inside[0]
    result["gt_inside_dilated_mask_3px"] = inside[3]
    result["gt_inside_dilated_mask_5px"] = inside[5]
    result["gt_inside_dilated_mask_10px"] = inside[10]
    result["gt_target_positive_primary"] = result["gt_inside_dilated_mask_3px"]
    result["gt_signed_distance_px"] = signed
    result["nearest_gt_target_point_distance_m"] = nearest.astype(float)
    result["projected_depth_difference_m"] = depth_difference
    for millimetres in (10, 20, 30):
        result[f"gt_3d_near_{millimetres}mm"] = (
            result["nearest_gt_target_point_distance_m"] <= millimetres / 1000.0
        )
    saved_u = result["projected_u_saved"].to_numpy(dtype=float)
    saved_v = result["projected_v_saved"].to_numpy(dtype=float)
    valid_saved = np.isfinite(saved_u) & np.isfinite(saved_v)
    if np.any(valid_saved):
        error = np.maximum(
            np.abs(saved_u[valid_saved] - result.loc[valid_saved, "projected_u"].to_numpy()),
            np.abs(saved_v[valid_saved] - result.loc[valid_saved, "projected_v"].to_numpy()),
        )
        if np.nanmax(error) > 1e-5:
            raise GTLabelError(
                f"sample {sample['sample_id']} candidate projection mismatch: {np.nanmax(error)}"
            )
    return result


def label_all_candidates(
    samples: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Apply leak-free GT labels to every candidate without dropping samples."""

    return label_candidate_pools(samples, candidates)[0]


def label_candidate_pools(
    samples: pd.DataFrame,
    *candidate_pools: pd.DataFrame,
    cache_size: int = 32,
) -> tuple[pd.DataFrame, ...]:
    """Label several pools while sharing GT geometry for the same sample.

    A small LRU additionally reuses repeated scene/object masks across adjacent
    referring expressions. The key includes raw depth path and mask bytes, so
    geometry is never reused across different observations or targets.
    """

    if cache_size < 1:
        raise ValueError("cache_size must be positive")
    known = set(samples["sample_id"].astype(str))
    for pool in candidate_pools:
        unknown = set(pool["sample_id"].astype(str)) - known
        if unknown:
            raise GTLabelError(f"candidates refer to unknown samples: {sorted(unknown)[:5]}")
    grouped = [
        {str(key): value for key, value in pool.groupby("sample_id", sort=False)}
        for pool in candidate_pools
    ]
    outputs = list(candidate_pools)
    label_columns = (
        "projected_u",
        "projected_v",
        "gt_inside_raw_mask",
        "gt_inside_dilated_mask_3px",
        "gt_inside_dilated_mask_5px",
        "gt_inside_dilated_mask_10px",
        "gt_target_positive_primary",
        "gt_signed_distance_px",
        "nearest_gt_target_point_distance_m",
        "projected_depth_difference_m",
        "gt_3d_near_10mm",
        "gt_3d_near_20mm",
        "gt_3d_near_30mm",
    )
    boolean_label_columns = {
        "gt_inside_raw_mask",
        "gt_inside_dilated_mask_3px",
        "gt_inside_dilated_mask_5px",
        "gt_inside_dilated_mask_10px",
        "gt_target_positive_primary",
        "gt_3d_near_10mm",
        "gt_3d_near_20mm",
        "gt_3d_near_30mm",
    }
    for output in outputs:
        for column in label_columns:
            output[column] = False if column in boolean_label_columns else np.nan
    cache: OrderedDict[tuple[str, bytes], _GTContext] = OrderedDict()
    started = time.perf_counter()
    processed = 0
    for sample in samples.sort_values("dataset_index").to_dict(orient="records"):
        sample_id = str(sample["sample_id"])
        groups = [mapping.get(sample_id) for mapping in grouped]
        if not any(group is not None and not group.empty for group in groups):
            continue
        processed += 1
        gt = _binary_image(str(sample["gt_mask_path"]))
        digest = hashlib.blake2b(gt.view(np.uint8), digest_size=16).digest()
        key = (str(Path(str(sample["depth_path"])).resolve()), digest)
        context = cache.pop(key, None)
        if context is None:
            context = _build_context(sample, gt)
        cache[key] = context
        while len(cache) > cache_size:
            cache.popitem(last=False)
        for index, group in enumerate(groups):
            if group is not None and not group.empty:
                labeled = label_candidate_group(sample, group, _context=context)
                for column in label_columns:
                    outputs[index].loc[labeled.index, column] = labeled[column].to_numpy()
        if processed % 500 == 0:
            LOGGER.info(
                "GT geometry labeling: %d candidate-bearing samples in %.1f s",
                processed,
                time.perf_counter() - started,
            )
    results = []
    for pool, result in zip(candidate_pools, outputs, strict=True):
        if len(result) != len(pool):
            raise GTLabelError("candidate count changed during shared GT labeling")
        results.append(
            result.sort_values(["dataset_index", "candidate_index_original"]).reset_index(drop=True)
        )
    return tuple(results)


__all__ = [
    "GTLabelError",
    "label_all_candidates",
    "label_candidate_pools",
    "label_candidate_group",
    "project_position",
]
