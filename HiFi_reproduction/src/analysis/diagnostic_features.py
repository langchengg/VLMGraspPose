"""Analysis-only secondary flags and official VGN stage diagnostics.

The functions in this module never change a candidate score, acceptance
decision, or model parameter.  In particular, the no-candidate diagnostic
reconstructs the frozen sample TSDF and records scalar counts while applying
the pinned CoRL 2020 post-processing constants in their official order.
Volumes are deliberately not retained by the streaming API.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from src.grasping.vgn_adapter import (
    OFFICIAL_DEPTH_TRUNC_M,
    OFFICIAL_GAUSSIAN_FILTER_SIGMA,
    OFFICIAL_MAX_FILTER_SIZE,
    OFFICIAL_MAX_WIDTH_VOXELS,
    OFFICIAL_MIN_WIDTH_VOXELS,
    OFFICIAL_QUALITY_THRESHOLD,
    OFFICIAL_RESOLUTION,
    OFFICIAL_VOXEL_SIZE_M,
    OFFICIAL_VGN_ROOT,
    OFFICIAL_WORKSPACE_SIZE_M,
    PredictionResult,
    build_tsdf_grid,
    load_official_network,
    predict_official,
    resolve_device_info,
)
from src.grasping.vgn_geometry import (
    CameraIntrinsics,
    resolve_depth_m,
    transform_points,
)


class DiagnosticFeatureError(RuntimeError):
    """Raised when a frozen artifact cannot support a requested diagnostic."""


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Pre-registered physical and image-space secondary-flag thresholds.

    These values are analysis configuration, not learned parameters.  Reports
    must serialize :meth:`to_dict` and should additionally repeat analyses at
    empirical quantiles where requested by the study protocol.
    """

    undersegmented_area_ratio: float = 0.5
    oversegmented_area_ratio: float = 1.5
    fragmented_largest_component_ratio: float = 0.9
    centroid_far_gt_bbox_diagonal_fraction: float = 0.25
    small_target_area_fraction: float = 0.005
    large_target_area_fraction: float = 0.20
    low_valid_target_depth_points: int = 100
    high_intrinsics_fit_rmse_px: float = 1.0
    high_support_plane_rmse_m: float = 0.01
    workspace_boundary_margin_m: float = 2.0 * OFFICIAL_VOXEL_SIZE_M
    sparse_tsdf_nonzero_fraction: float = 0.05
    gripper_width_limit_margin_m: float = OFFICIAL_VOXEL_SIZE_M
    large_projected_depth_residual_m: float = 0.03
    many_near_duplicate_fraction: float = 0.25
    many_near_duplicate_minimum: int = 2
    small_quality_gap: float = 0.02
    large_quality_gap: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PostprocessingStageDiagnostics:
    """Scalar audit trail for one official VGN forward/post-processing pass."""

    sample_id: str
    raw_quality_max: float
    count_raw_quality_above_0_90: int
    count_after_gaussian: int
    count_after_surface_filter: int
    count_after_width_filter: int
    count_after_threshold: int
    count_after_3d_local_maximum: int
    first_zero_stage: str | None
    candidate_generation_secondary_flag: str | None
    tsdf_nonzero_count: int
    tsdf_nonzero_fraction: float
    S_empty_or_sparse_tsdf: bool
    device_requested: str | None = None
    device_used: str | None = None
    mps_fallback_reason: str | None = None
    processing_time_tsdf_s: float | None = None
    processing_time_vgn_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _binary_mask(source: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise DiagnosticFeatureError(f"missing mask: {path}")
        with Image.open(path) as image:
            value = np.asarray(image)
    else:
        value = np.asarray(source)
    if value.ndim == 3:
        value = value[..., 0]
    if value.ndim != 2:
        raise DiagnosticFeatureError(f"mask must be 2-D, got {value.shape}")
    return np.asarray(value > 0, dtype=bool)


def _component_metrics(mask: np.ndarray) -> tuple[int, float]:
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0, 0.0
    areas = np.bincount(labels.ravel())[1:]
    return int(count), float(areas.max() / max(int(mask.sum()), 1))


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    row, column = np.nonzero(mask)
    if not len(row):
        return None
    return np.asarray([column.mean(), row.mean()], dtype=np.float64)


def grounding_diagnostic_features(
    predicted_mask: np.ndarray | str | Path,
    gt_mask: np.ndarray | str | Path,
    *,
    thresholds: DiagnosticThresholds | None = None,
) -> dict[str, Any]:
    """Return registered grounding metrics and heuristic secondary flags."""

    config = thresholds or DiagnosticThresholds()
    predicted = _binary_mask(predicted_mask)
    target = _binary_mask(gt_mask)
    if predicted.shape != target.shape:
        raise DiagnosticFeatureError(
            f"predicted/GT mask shape mismatch: {predicted.shape} != {target.shape}"
        )
    predicted_area = int(predicted.sum())
    target_area = int(target.sum())
    if target_area == 0:
        raise DiagnosticFeatureError("GT target mask is empty")
    intersection = int(np.count_nonzero(predicted & target))
    union = int(np.count_nonzero(predicted | target))
    iou = float(intersection / union) if union else 1.0
    area_ratio = float(predicted_area / target_area)
    component_count, largest_ratio = _component_metrics(predicted)

    gt_rows, gt_columns = np.nonzero(target)
    bbox_diagonal = math.hypot(
        float(gt_columns.max() - gt_columns.min() + 1),
        float(gt_rows.max() - gt_rows.min() + 1),
    )
    pred_centroid = _centroid(predicted)
    gt_centroid = _centroid(target)
    centroid_distance = (
        float("inf")
        if pred_centroid is None or gt_centroid is None
        else float(np.linalg.norm(pred_centroid - gt_centroid))
    )
    centroid_far = centroid_distance > (
        config.centroid_far_gt_bbox_diagonal_fraction * bbox_diagonal
    )
    target_fraction = float(target_area / target.size)
    zero_overlap = intersection == 0
    wrong_object_likely = zero_overlap or (iou < 0.25 and centroid_far)
    return {
        "pred_mask_area_px_diagnostic": predicted_area,
        "gt_mask_area_px_diagnostic": target_area,
        "mask_intersection_px": intersection,
        "mask_iou_diagnostic": iou,
        "mask_area_ratio_pred_to_gt": area_ratio,
        "pred_mask_connected_components": component_count,
        "pred_mask_largest_component_ratio": largest_ratio,
        "mask_centroid_distance_px": centroid_distance,
        "gt_bbox_diagonal_px": bbox_diagonal,
        "gt_target_area_fraction": target_fraction,
        "S_mask_zero_overlap": zero_overlap,
        "S_mask_iou_lt_025": iou < 0.25,
        "S_mask_iou_025_050": 0.25 <= iou < 0.50,
        "S_mask_undersegmented": area_ratio < config.undersegmented_area_ratio,
        "S_mask_oversegmented": area_ratio > config.oversegmented_area_ratio,
        "S_mask_fragmented": (
            component_count > 1
            and largest_ratio < config.fragmented_largest_component_ratio
        ),
        "S_mask_centroid_far": centroid_far,
        # This is deliberately heuristic and must never be reported as a
        # human-confirmed wrong-object annotation.
        "S_mask_wrong_object_likely": wrong_object_likely,
        "S_small_target": target_fraction < config.small_target_area_fraction,
        "S_large_target": target_fraction > config.large_target_area_fraction,
    }


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _target_near_workspace_boundary(
    sample: Mapping[str, Any], margin_m: float
) -> bool | None:
    cloud = sample.get("target_cloud")
    transform = sample.get("T_camera_task")
    if isinstance(cloud, str):
        cloud = json.loads(cloud)
    if isinstance(transform, str):
        transform = json.loads(transform)
    if not isinstance(cloud, Mapping) or transform is None:
        return None
    minimum = np.asarray(cloud.get("aabb_min_camera_m"), dtype=np.float64)
    maximum = np.asarray(cloud.get("aabb_max_camera_m"), dtype=np.float64)
    camera_from_task = np.asarray(transform, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or camera_from_task.shape != (4, 4):
        return None
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )
    task_from_camera = np.linalg.inv(camera_from_task)
    task_corners = transform_points(task_from_camera, corners)
    size = float(sample.get("workspace_size_m", OFFICIAL_WORKSPACE_SIZE_M))
    return bool(
        np.any(task_corners < margin_m)
        or np.any(task_corners > size - margin_m)
    )


def geometry_diagnostic_features(
    sample: Mapping[str, Any],
    candidates: pd.DataFrame,
    *,
    thresholds: DiagnosticThresholds | None = None,
) -> dict[str, Any]:
    """Compute per-sample geometry flags without changing candidate validity."""

    config = thresholds or DiagnosticThresholds()
    valid_points = _finite(sample.get("valid_target_depth_points"))
    fit_rmse = _finite(sample.get("fit_rmse_px"))
    plane_rmse = _finite(sample.get("support_plane_residual"))
    tsdf_fraction = _finite(sample.get("tsdf_nonzero_fraction"))

    selected = candidates[candidates.get("is_baseline_top1", False).astype(bool)] if (
        not candidates.empty and "is_baseline_top1" in candidates
    ) else candidates.iloc[0:0]
    width_near_limit = False
    large_depth_residual = False
    if len(selected) == 1:
        width = _finite(selected.iloc[0].get("width_m"))
        if width is not None:
            minimum = OFFICIAL_MIN_WIDTH_VOXELS * OFFICIAL_VOXEL_SIZE_M
            maximum = OFFICIAL_MAX_WIDTH_VOXELS * OFFICIAL_VOXEL_SIZE_M
            width_near_limit = bool(
                width - minimum <= config.gripper_width_limit_margin_m
                or maximum - width <= config.gripper_width_limit_margin_m
            )
        depth_residual = _finite(
            selected.iloc[0].get("projected_depth_difference_m")
        )
        large_depth_residual = bool(
            depth_residual is not None
            and abs(depth_residual) > config.large_projected_depth_residual_m
        )
    elif len(selected) > 1:
        raise DiagnosticFeatureError(
            f"sample {sample.get('sample_id')} has multiple baseline top-1 candidates"
        )

    boundary = _target_near_workspace_boundary(
        sample, config.workspace_boundary_margin_m
    )
    return {
        "S_low_valid_target_depth": bool(
            valid_points is not None
            and valid_points < config.low_valid_target_depth_points
        ),
        "S_high_intrinsics_fit_rmse": bool(
            fit_rmse is not None and fit_rmse > config.high_intrinsics_fit_rmse_px
        ),
        "S_high_support_plane_rmse": bool(
            plane_rmse is not None
            and plane_rmse > config.high_support_plane_rmse_m
        ),
        "S_target_near_workspace_boundary": boundary,
        "S_sparse_observed_tsdf": (
            None
            if tsdf_fraction is None
            else tsdf_fraction < config.sparse_tsdf_nonzero_fraction
        ),
        "S_single_view_curvature_risk": str(
            sample.get("tsdf_mode", "single_view_adaptation")
        )
        == "single_view_adaptation",
        "S_gripper_width_near_limit": width_near_limit,
        "S_large_projected_depth_residual": large_depth_residual,
    }


def ranking_diagnostic_features(
    sample: Mapping[str, Any],
    candidates: pd.DataFrame,
    *,
    positive_column: str = "gt_target_positive_primary",
    thresholds: DiagnosticThresholds | None = None,
) -> dict[str, Any]:
    """Compute candidate/ranking flags for one frozen official pool."""

    config = thresholds or DiagnosticThresholds()
    if not candidates.empty and positive_column not in candidates:
        raise DiagnosticFeatureError(f"candidate table lacks {positive_column}")
    official_count = len(candidates)
    positive = (
        candidates[positive_column].fillna(False).astype(bool)
        if official_count
        else pd.Series([], dtype=bool)
    )
    passed = (
        candidates["pred_filter_pass"].fillna(False).astype(bool)
        if official_count and "pred_filter_pass" in candidates
        else pd.Series([False] * official_count, index=candidates.index, dtype=bool)
    )
    positive_ranks = (
        candidates.loc[positive, "rank_vgn_all"].astype(int).tolist()
        if official_count and "rank_vgn_all" in candidates
        else []
    )
    first_positive_rank = min(positive_ranks) if positive_ranks else None
    baseline = (
        candidates[candidates["is_baseline_top1"].fillna(False).astype(bool)]
        if official_count and "is_baseline_top1" in candidates
        else candidates.iloc[0:0]
    )
    if len(baseline) > 1:
        raise DiagnosticFeatureError(
            f"sample {sample.get('sample_id')} has multiple baseline top-1 candidates"
        )
    best_positive_quality = (
        float(candidates.loc[positive, "vgn_quality"].max()) if positive.any() else None
    )
    baseline_quality = (
        _finite(baseline.iloc[0].get("vgn_quality")) if len(baseline) else None
    )
    quality_gap = (
        None
        if best_positive_quality is None or baseline_quality is None
        else float(baseline_quality - best_positive_quality)
    )
    distinct = _finite(sample.get("n_distinct_pose_modes"))
    duplicate_count = None if distinct is None else max(0, official_count - int(distinct))
    many_duplicates = (
        None
        if duplicate_count is None or official_count == 0
        else duplicate_count >= config.many_near_duplicate_minimum
        and duplicate_count / official_count >= config.many_near_duplicate_fraction
    )
    return {
        "rank_first_gt_positive": first_positive_rank,
        "best_gt_positive_quality": best_positive_quality,
        "baseline_top1_quality_diagnostic": baseline_quality,
        "quality_gap_to_best_gt_positive": quality_gap,
        "S_single_official_candidate": official_count == 1,
        "S_multiple_official_candidates": official_count >= 2,
        "S_many_near_duplicate_candidates": many_duplicates,
        "S_gt_positive_rank_2_3": first_positive_rank in {2, 3},
        "S_gt_positive_rank_4_5": first_positive_rank in {4, 5},
        "S_gt_positive_rank_gt5": bool(
            first_positive_rank is not None and first_positive_rank > 5
        ),
        "S_small_quality_gap": bool(
            quality_gap is not None and 0.0 < quality_gap <= config.small_quality_gap
        ),
        "S_large_quality_gap": bool(
            quality_gap is not None and quality_gap >= config.large_quality_gap
        ),
        "S_pred_filter_false_negative": bool(positive.any() and not (positive & passed).any()),
        "S_pred_filter_false_positive": bool(((~positive) & passed).any()),
    }


def secondary_diagnostic_features(
    sample: Mapping[str, Any],
    candidates: pd.DataFrame,
    *,
    thresholds: DiagnosticThresholds | None = None,
    positive_column: str = "gt_target_positive_primary",
    include_grounding: bool = True,
) -> dict[str, Any]:
    """Combine grounding, geometry, and ranking diagnostics for one sample."""

    config = thresholds or DiagnosticThresholds()
    result: dict[str, Any] = {}
    if include_grounding:
        result.update(
            grounding_diagnostic_features(
                sample["pred_mask_path"], sample["gt_mask_path"], thresholds=config
            )
        )
    result.update(geometry_diagnostic_features(sample, candidates, thresholds=config))
    result.update(
        ranking_diagnostic_features(
            sample,
            candidates,
            positive_column=positive_column,
            thresholds=config,
        )
    )
    return result


def add_secondary_diagnostic_features(
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    thresholds: DiagnosticThresholds | None = None,
    positive_column: str = "gt_target_positive_primary",
    include_grounding: bool = True,
) -> pd.DataFrame:
    """Return a sample table augmented with registered secondary flags."""

    if "sample_id" not in samples or "sample_id" not in candidates:
        raise DiagnosticFeatureError("sample and candidate tables require sample_id")
    groups = {
        str(sample_id): group
        for sample_id, group in candidates.groupby("sample_id", sort=False)
    }
    features = []
    for row in samples.to_dict(orient="records"):
        group = groups.get(str(row["sample_id"]), candidates.iloc[0:0])
        features.append(
            secondary_diagnostic_features(
                row,
                group,
                thresholds=thresholds,
                positive_column=positive_column,
                include_grounding=include_grounding,
            )
        )
    addition = pd.DataFrame(features, index=samples.index)
    overlapping = sorted(set(addition) & set(samples))
    if overlapping:
        raise DiagnosticFeatureError(
            f"secondary diagnostic columns already exist: {overlapping}"
        )
    return pd.concat([samples.copy(), addition], axis=1)


def _first_zero_stage(counts: Mapping[str, int]) -> tuple[str | None, str | None]:
    stages = (
        ("raw_quality_threshold", "S_no_quality_above_threshold"),
        ("gaussian_smoothing", "S_no_quality_above_threshold"),
        ("surface_filter", "S_removed_by_surface_filter"),
        ("width_filter", "S_removed_by_width_filter"),
        ("quality_threshold", "S_no_quality_above_threshold"),
        ("3d_local_maximum", "S_removed_by_nms"),
    )
    for stage, flag in stages:
        if counts[stage] == 0:
            return stage, flag
    return None, "S_unknown_candidate_generation_failure"


def official_postprocessing_stage_diagnostics(
    tsdf_grid: np.ndarray,
    quality_volume: np.ndarray,
    width_volume: np.ndarray,
    *,
    sample_id: str = "",
    sparse_tsdf_nonzero_fraction: float = 0.05,
) -> PostprocessingStageDiagnostics:
    """Instrument official process/select without changing their parameters."""

    tsdf = np.asarray(tsdf_grid)
    quality = np.asarray(quality_volume)
    width = np.asarray(width_volume)
    if tsdf.shape != (1, OFFICIAL_RESOLUTION, OFFICIAL_RESOLUTION, OFFICIAL_RESOLUTION):
        raise DiagnosticFeatureError(f"invalid TSDF shape: {tsdf.shape}")
    expected = (OFFICIAL_RESOLUTION, OFFICIAL_RESOLUTION, OFFICIAL_RESOLUTION)
    if quality.shape != expected or width.shape != expected:
        raise DiagnosticFeatureError(
            f"invalid VGN output shapes: quality={quality.shape}, width={width.shape}"
        )
    if not np.all(np.isfinite(tsdf)) or not np.all(np.isfinite(quality)) or not np.all(np.isfinite(width)):
        raise DiagnosticFeatureError("stage diagnostics received NaN or Inf")

    threshold = OFFICIAL_QUALITY_THRESHOLD
    smoothed = ndimage.gaussian_filter(
        quality, sigma=OFFICIAL_GAUSSIAN_FILTER_SIGMA, mode="nearest"
    )
    squeezed = tsdf.squeeze()
    outside_voxels = squeezed > 0.5
    inside_voxels = np.logical_and(1e-3 < squeezed, squeezed < 0.5)
    valid_voxels = ndimage.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    after_surface = smoothed.copy()
    after_surface[valid_voxels == False] = 0.0  # noqa: E712 - upstream expression.
    after_width = after_surface.copy()
    after_width[
        np.logical_or(
            width < OFFICIAL_MIN_WIDTH_VOXELS,
            width > OFFICIAL_MAX_WIDTH_VOXELS,
        )
    ] = 0.0
    after_threshold = after_width.copy()
    after_threshold[after_threshold < threshold] = 0.0
    maximum = ndimage.maximum_filter(after_threshold, size=OFFICIAL_MAX_FILTER_SIZE)
    selected = np.where(after_threshold == maximum, after_threshold, 0.0)

    counts = {
        "raw_quality_threshold": int(np.count_nonzero(quality >= threshold)),
        "gaussian_smoothing": int(np.count_nonzero(smoothed >= threshold)),
        "surface_filter": int(np.count_nonzero(after_surface >= threshold)),
        "width_filter": int(np.count_nonzero(after_width >= threshold)),
        "quality_threshold": int(np.count_nonzero(after_threshold)),
        "3d_local_maximum": int(np.count_nonzero(selected)),
    }
    first_zero, flag = _first_zero_stage(counts)
    nonzero = int(np.count_nonzero(tsdf))
    fraction = float(nonzero / tsdf.size)
    return PostprocessingStageDiagnostics(
        sample_id=str(sample_id),
        raw_quality_max=float(np.max(quality)),
        count_raw_quality_above_0_90=counts["raw_quality_threshold"],
        count_after_gaussian=counts["gaussian_smoothing"],
        count_after_surface_filter=counts["surface_filter"],
        count_after_width_filter=counts["width_filter"],
        count_after_threshold=counts["quality_threshold"],
        count_after_3d_local_maximum=counts["3d_local_maximum"],
        first_zero_stage=first_zero,
        candidate_generation_secondary_flag=flag,
        tsdf_nonzero_count=nonzero,
        tsdf_nonzero_fraction=fraction,
        S_empty_or_sparse_tsdf=(
            nonzero == 0 or fraction < float(sparse_tsdf_nonzero_fraction)
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticFeatureError(f"missing diagnostic artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DiagnosticFeatureError(f"cannot parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise DiagnosticFeatureError(f"expected JSON object: {path}")
    return value


def diagnose_no_official_sample(
    sample: Mapping[str, Any],
    net: Any,
    device: str,
    *,
    vgn_root: Path | str = OFFICIAL_VGN_ROOT,
    tsdf_builder: Callable[..., Any] = build_tsdf_grid,
    predictor: Callable[..., PredictionResult] = predict_official,
    thresholds: DiagnosticThresholds | None = None,
) -> PostprocessingStageDiagnostics:
    """Rebuild one frozen TSDF and instrument one pinned official VGN pass.

    The saved task frame is used directly.  Support-plane estimation and mask
    processing are intentionally not repeated, which removes an unnecessary
    source of diagnostic drift.
    """

    if str(sample.get("pred_status")) != "no_official_grasp":
        raise DiagnosticFeatureError(
            f"sample {sample.get('sample_id')} is not no_official_grasp"
        )
    sample_id = str(sample.get("sample_id", ""))
    workspace_path = Path(str(sample["workspace_frame_path"])).expanduser().resolve()
    workspace = _read_json(workspace_path)
    depth_path = Path(str(sample["depth_path"])).expanduser().resolve()
    if not depth_path.is_file():
        raise DiagnosticFeatureError(f"missing depth: {depth_path}")
    with Image.open(depth_path) as image:
        raw_depth = np.asarray(image)
    if raw_depth.ndim == 3:
        raw_depth = raw_depth[..., 0]
    depth_metadata = workspace.get("depth")
    if not isinstance(depth_metadata, Mapping):
        raise DiagnosticFeatureError(f"{workspace_path} lacks depth provenance")
    unit = str(depth_metadata.get("depth_unit", ""))
    scale = _finite(depth_metadata.get("depth_scale"))
    if unit not in {"m", "mm"} or scale is None:
        raise DiagnosticFeatureError(
            f"{workspace_path} lacks an explicit resolved depth unit/scale"
        )
    depth_m = resolve_depth_m(
        raw_depth,
        unit=unit,
        depth_scale=scale,
        min_depth_m=0.05,
        max_depth_m=2.0,
    ).depth_m
    intrinsics_payload = workspace.get("intrinsics")
    if not isinstance(intrinsics_payload, Mapping):
        raise DiagnosticFeatureError(f"{workspace_path} lacks intrinsics")
    intrinsics = CameraIntrinsics.from_mapping(
        intrinsics_payload, image_shape=raw_depth.shape
    )
    camera_from_task = np.asarray(workspace.get("T_camera_task"), dtype=np.float64)
    if camera_from_task.shape != (4, 4) or not np.all(np.isfinite(camera_from_task)):
        raise DiagnosticFeatureError(f"invalid T_camera_task in {workspace_path}")

    tsdf_started = time.perf_counter()
    built = tsdf_builder(
        depth_m,
        intrinsics,
        camera_from_task,
        vgn_root=Path(vgn_root),
        workspace_size_m=OFFICIAL_WORKSPACE_SIZE_M,
        resolution=OFFICIAL_RESOLUTION,
        depth_trunc_m=OFFICIAL_DEPTH_TRUNC_M,
        preset="official",
    )
    tsdf_time = time.perf_counter() - tsdf_started
    grid = np.asarray(built.grid)
    vgn_started = time.perf_counter()
    prediction = predictor(grid, net, device)
    diagnostic = official_postprocessing_stage_diagnostics(
        grid,
        prediction.qual_vol,
        prediction.width_vol,
        sample_id=sample_id,
        sparse_tsdf_nonzero_fraction=(
            thresholds or DiagnosticThresholds()
        ).sparse_tsdf_nonzero_fraction,
    )
    vgn_time = time.perf_counter() - vgn_started
    values = diagnostic.to_dict()
    values.update(
        device_requested=str(prediction.requested_device),
        device_used=str(prediction.used_device),
        mps_fallback_reason=prediction.mps_fallback_reason,
        processing_time_tsdf_s=float(tsdf_time),
        processing_time_vgn_s=float(vgn_time),
    )
    return PostprocessingStageDiagnostics(**values)


def iter_no_official_stage_diagnostics(
    samples: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    vgn_weights: Path | str,
    vgn_root: Path | str = OFFICIAL_VGN_ROOT,
    device: str = "auto",
    thresholds: DiagnosticThresholds | None = None,
) -> Iterator[PostprocessingStageDiagnostics]:
    """Yield failure diagnostics one at a time while loading VGN only once."""

    selection = resolve_device_info(device)
    net = load_official_network(
        vgn_weights, device=selection.resolved, vgn_root=vgn_root
    )
    rows = (
        samples.to_dict(orient="records")
        if isinstance(samples, pd.DataFrame)
        else samples
    )
    for sample in rows:
        if str(sample.get("pred_status")) != "no_official_grasp":
            continue
        # The returned object is scalar-only; the TSDF and network outputs go
        # out of scope before the generator advances to the next sample.
        yield diagnose_no_official_sample(
            sample,
            net,
            selection.resolved,
            vgn_root=vgn_root,
            thresholds=thresholds,
        )


__all__ = [
    "DiagnosticFeatureError",
    "DiagnosticThresholds",
    "PostprocessingStageDiagnostics",
    "add_secondary_diagnostic_features",
    "diagnose_no_official_sample",
    "geometry_diagnostic_features",
    "grounding_diagnostic_features",
    "iter_no_official_stage_diagnostics",
    "official_postprocessing_stage_diagnostics",
    "ranking_diagnostic_features",
    "secondary_diagnostic_features",
]
