"""Offline HiFi-CS predicted target mask -> local TSDF -> official VGN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.grasping.vgn_adapter import (
    OFFICIAL_DEPTH_TRUNC_M,
    OFFICIAL_MAX_FILTER_SIZE,
    OFFICIAL_MAX_WIDTH_VOXELS,
    OFFICIAL_MIN_WIDTH_VOXELS,
    OFFICIAL_QUALITY_THRESHOLD,
    OFFICIAL_RESOLUTION,
    OFFICIAL_TABLE_HEIGHT_M,
    OFFICIAL_VOXEL_SIZE_M,
    OFFICIAL_WORKSPACE_SIZE_M,
    VGNAdapterError,
    build_tsdf_grid,
    filter_target_candidates,
    load_official_network,
    predict_official,
    resolve_device_info,
    run_official_postprocessing,
    runtime_metadata,
    select_candidate,
    sort_candidates_by_quality,
)
from src.grasping.vgn_geometry import (
    GeometryError,
    SupportPlaneError,
    backproject_depth,
    build_task_frame,
    dilate_mask,
    estimate_support_plane,
    invert_transform,
    load_intrinsics_config,
    prepare_target_mask,
    resize_mask_nearest,
    resolve_depth_m,
    robust_target_cloud,
    transform_points,
)
from src.grasping.vgn_pipeline import (
    LIMITATIONS,
    SCORE_SOURCE,
    SUMMARY_FIELDS,
    TSDF_MODE,
    ManifestSample,
    PipelineError,
    atomic_write_candidates_npz,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    build_stem_manifest,
    candidate_status,
    candidate_records,
    elapsed,
    empty_summary,
    environment_versions,
    load_manifest_samples,
    load_sample_arrays,
    sha256_file,
)
from src.grasping.vgn_visualization import (
    atomic_write_point_cloud,
    save_candidates_overlay,
    save_grasps_3d_ply,
    save_quality_max_projection,
    save_rgb_mask_overlay,
    save_top1_overlay,
)


LOGGER = logging.getLogger("hifics_vgn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and rank target-filtered official VGN 6-DoF candidates."
    )
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--hifi-root", type=Path, required=True)
    parser.add_argument("--vgn-root", type=Path, default=Path("third_party/vgn"))
    parser.add_argument(
        "--vgn-weights", type=Path, default=Path("third_party/vgn/data/models/vgn_conv.pth")
    )
    parser.add_argument(
        "--intrinsics",
        required=True,
        help=(
            "JSON/YAML calibration, or 'per-sample-bundle' to use each bundle's "
            "explicit PCD-derived intrinsics.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/hifics_vgn"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--vgn-preset", choices=("official",), default="official")
    parser.add_argument(
        "--selection-policy",
        choices=("highest_vgn_quality", "official_sim_random", "official_panda_highest_z"),
        default="highest_vgn_quality",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--save-tsdf", action="store_true")
    parser.add_argument("--save-pointclouds", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    parser.add_argument(
        "--mask-cleanup", choices=("none", "largest-component", "close"), default="none"
    )
    parser.add_argument("--target-mask-dilation-px", type=int, default=3)
    parser.add_argument("--mask-min-area-px", type=int, default=25)
    parser.add_argument("--min-masked-depth-points", type=int, default=20)
    parser.add_argument("--depth-unit", choices=("auto", "m", "mm"), default="auto")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-min-m", type=float, default=0.05)
    parser.add_argument("--depth-max-m", type=float, default=2.0)
    parser.add_argument("--depth-trunc-m", type=float, default=OFFICIAL_DEPTH_TRUNC_M)
    parser.add_argument("--workspace-size-m", type=float, default=OFFICIAL_WORKSPACE_SIZE_M)
    parser.add_argument("--resolution", type=int, default=OFFICIAL_RESOLUTION)
    parser.add_argument("--table-height-m", type=float, default=OFFICIAL_TABLE_HEIGHT_M)
    parser.add_argument("--allow-camera-aligned-fallback", action="store_true")
    parser.add_argument(
        "--multi-view-manifest",
        type=Path,
        help="Reserved for calibrated multi-view frames; uncalibrated top/bottom fusion is rejected.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.start_index is not None and args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must not be smaller than --start-index")
    if args.target_mask_dilation_px < 0:
        raise ValueError("--target-mask-dilation-px cannot be negative")
    if args.multi_view_manifest is not None:
        raise PipelineError(
            "uncalibrated_multi_view_forbidden",
            "Multi-view integration requires explicit frame extrinsics in a common coordinate system; "
            "the current OCID top/bottom images cannot be fused safely.",
        )
    if (
        args.workspace_size_m != OFFICIAL_WORKSPACE_SIZE_M
        or args.resolution != OFFICIAL_RESOLUTION
    ):
        LOGGER.warning(
            "changing physical scale invalidates strict pretrained-model comparability"
        )


def _select_samples(samples: list[ManifestSample], args: argparse.Namespace) -> list[ManifestSample]:
    selected = samples
    if args.sample_id is not None:
        selected = [sample for sample in selected if sample.sample_id == args.sample_id]
        if not selected:
            raise KeyError(f"Unknown --sample-id: {args.sample_id}")
    if args.start_index is not None:
        selected = [sample for sample in selected if sample.dataset_index >= args.start_index]
    if args.end_index is not None:
        selected = [sample for sample in selected if sample.dataset_index < args.end_index]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    return selected


def _load_existing_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {str(row["sample_id"]): dict(row) for row in csv.DictReader(stream)}


def _existing_status(sample_dir: Path) -> str | None:
    path = sample_dir / "top1.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = str(value.get("status")) if value.get("status") else None
    # An `ok` top1 may have been atomically written just before an abrupt
    # interruption during visualization. Only the final marker means every
    # configured artifact completed.
    if status == "ok" and not (sample_dir / "_SUCCESS.json").is_file():
        return None
    return status


def _run_signature(args: argparse.Namespace, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Fingerprint every option that can change inference or saved artifacts."""

    excluded = {
        "overwrite",
        "retry_failures",
        "sample_id",
        "start_index",
        "end_index",
        "max_samples",
        "log_level",
        "output",
    }
    options = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in excluded
    }
    inputs: dict[str, Any] = {
        "options": options,
        "repository_commit": metadata["repository_commit"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
    }
    implementation_root = Path(__file__).resolve().parents[1]
    implementation_files = (
        Path(__file__).resolve(),
        implementation_root / "src/grasping/vgn_adapter.py",
        implementation_root / "src/grasping/vgn_geometry.py",
        implementation_root / "src/grasping/vgn_pipeline.py",
        implementation_root / "src/grasping/vgn_visualization.py",
    )
    inputs["implementation_sha256"] = {
        str(path.relative_to(implementation_root)): sha256_file(path)
        for path in implementation_files
    }
    if args.manifest is not None and args.manifest.is_file():
        inputs["manifest_sha256"] = sha256_file(args.manifest)
    intrinsics_path = Path(args.intrinsics).expanduser()
    if args.intrinsics != "per-sample-bundle" and intrinsics_path.is_file():
        inputs["intrinsics_config_sha256"] = sha256_file(intrinsics_path)
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), inputs


def _prepare_run_output(
    output: Path,
    metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Prevent a resumable output tree from mixing incompatible run configs."""

    config_path = output / "run_config.json"
    existing: dict[str, Any] | None = None
    if config_path.is_file():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise PipelineError("run_config_corrupt", f"Invalid {config_path}") from error
    has_results = (output / "samples").exists() or (output / "summary.csv").exists()
    prior_signature = None if existing is None else existing.get("run_signature_sha256")
    new_signature = metadata["run_signature_sha256"]
    incompatible = has_results and prior_signature != new_signature
    if incompatible and not overwrite:
        raise PipelineError(
            "run_config_mismatch",
            "Output contains results from a different or legacy run configuration; "
            "use a new --output or pass --overwrite to start a clean run.",
        )
    if incompatible:
        LOGGER.warning("Run signature changed; --overwrite resets prior generated samples")
        samples_dir = output / "samples"
        if samples_dir.exists():
            shutil.rmtree(samples_dir)
        for name in ("summary.csv", "failures.csv"):
            path = output / name
            if path.exists():
                path.unlink()
    atomic_write_json(config_path, metadata)


def _intrinsics_for_sample(
    source: str, sample: ManifestSample, image_shape: tuple[int, int]
) -> Any:
    if source == "per-sample-bundle":
        if sample.intrinsics_path is None:
            raise GeometryError(
                "missing_camera_intrinsics",
                f"No per-sample intrinsics.json for {sample.sample_id}",
            )
        selected: Path | str = sample.intrinsics_path
    else:
        selected = source
    intrinsics = load_intrinsics_config(selected, view=sample.view, image_shape=image_shape)
    LOGGER.info(
        "[%s] intrinsics source=%s view=%s fx=%.9g fy=%.9g cx=%.9g cy=%.9g",
        sample.sample_id,
        intrinsics.source,
        sample.view,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
    )
    return intrinsics


def _failure_top1(args: argparse.Namespace, status: str, reason: str) -> dict[str, Any]:
    return {
        "selection_policy": args.selection_policy,
        "score_source": SCORE_SOURCE,
        "custom_reranking": False,
        "candidate_count_before_target_filter": 0,
        "candidate_count_after_target_filter": 0,
        "status": status,
        "failure_reason": reason,
        "limitations": LIMITATIONS,
    }


def _sample_failure(
    sample: ManifestSample,
    sample_dir: Path,
    args: argparse.Namespace,
    status: str,
    reason: str,
    *,
    partial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    payload = _failure_top1(args, status, reason)
    if partial:
        payload.update(partial)
    atomic_write_json(sample_dir / "top1.json", payload)
    summary = empty_summary(sample, status=status, reason=reason)
    if partial:
        for key in SUMMARY_FIELDS:
            if key in partial:
                summary[key] = partial[key]
    return summary


def _save_sample_visuals(
    sample_dir: Path,
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: Any,
    all_candidates: list[Any],
    target_candidates: list[Any],
    top1: Any | None,
    processed_quality: np.ndarray,
    local_scene_points: np.ndarray,
    target_points: np.ndarray,
    top_k: int,
    T_camera_task: np.ndarray,
    workspace_size_m: float,
    table_height_m: float,
) -> None:
    save_rgb_mask_overlay(rgb, mask, sample_dir / "rgb_mask_overlay.png")
    save_candidates_overlay(
        rgb,
        mask,
        all_candidates,
        sample_dir / "candidates_2d_overlay.png",
        title="Official VGN centres: green=target, red=off-target",
    )
    save_candidates_overlay(
        rgb,
        mask,
        target_candidates,
        sample_dir / "target_topk_2d_overlay.png",
        top_k=top_k,
        target_only=True,
        title=f"Target-filtered top-{top_k} by official VGN quality",
    )
    save_quality_max_projection(processed_quality, sample_dir / "quality_max_projection.png")
    if top1 is not None:
        save_top1_overlay(
            rgb,
            mask,
            top1,
            intrinsics,
            sample_dir / "top1_2d_overlay.png",
            title="Selected grasp: approach (blue), opening width (orange)",
        )
    save_grasps_3d_ply(
        sample_dir / "grasps_3d.ply",
        local_scene_points,
        target_points,
        target_candidates,
        top_k=top_k,
        T_camera_task=T_camera_task,
        workspace_size_m=workspace_size_m,
        table_height_m=table_height_m,
        selected_official_index=(
            None if top1 is None else top1.official_selection_index
        ),
    )


def process_sample(
    sample: ManifestSample,
    *,
    args: argparse.Namespace,
    net: Any,
    device: str,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Process one sample, optionally using arrays validated by a scene cache.

    ``arrays`` is intentionally optional so the existing CLI/API remains
    unchanged.  The full-dataset runner uses it to avoid decoding the same RGB
    and depth for every referring expression in one scene.
    """

    sample_dir = args.output / "samples" / sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    partial: dict[str, Any] = {}
    depth_start = time.perf_counter()
    if arrays is None:
        rgb, raw_depth, raw_mask = load_sample_arrays(sample)
    else:
        rgb, raw_depth, raw_mask = arrays
    intrinsics = _intrinsics_for_sample(args.intrinsics, sample, raw_depth.shape)

    metadata_scale = sample.metadata.get("depth_scale")
    if args.depth_unit == "auto" and metadata_scale is not None and not np.isclose(
        float(metadata_scale), float(args.depth_scale)
    ):
        raise GeometryError(
            "ambiguous_depth_unit",
            f"manifest depth_scale={metadata_scale} conflicts with --depth-scale={args.depth_scale}",
        )
    mask_for_depth_stats = resize_mask_nearest(raw_mask, raw_depth.shape)
    depth_result = resolve_depth_m(
        raw_depth,
        unit=args.depth_unit,
        depth_scale=args.depth_scale,
        min_depth_m=args.depth_min_m,
        max_depth_m=args.depth_max_m,
        target_mask=mask_for_depth_stats,
        metadata=sample.metadata,
    )
    LOGGER.info("[%s] depth decision: %s", sample.sample_id, depth_result.log_dict())
    mask_result = prepare_target_mask(
        raw_mask,
        depth_result.depth_m,
        cleanup=args.mask_cleanup,
        min_area_px=args.mask_min_area_px,
        min_valid_depth_points=args.min_masked_depth_points,
    )
    target_cloud = robust_target_cloud(
        depth_result.depth_m,
        mask_result.mask,
        intrinsics,
        min_points=args.min_masked_depth_points,
    )
    partial.update(
        mask_area=mask_result.diagnostics.area_px,
        valid_target_depth_points=mask_result.diagnostics.valid_depth_points,
        processing_time_depth=elapsed(depth_start),
    )

    support_payload: dict[str, Any]
    try:
        support_plane = estimate_support_plane(
            depth_result.depth_m,
            mask_result.mask,
            intrinsics,
            target_cloud,
            seed=args.seed,
        )
        support_payload = support_plane.to_dict()
        partial["support_plane_residual"] = support_plane.residual_rmse_m
    except SupportPlaneError as error:
        if not args.allow_camera_aligned_fallback:
            raise
        LOGGER.warning("[%s] support plane failed; explicit fallback enabled: %s", sample.sample_id, error)
        support_plane = None
        support_payload = {
            "status": "support_plane_failed",
            "failure_reason": str(error),
            "non_official_geometry_fallback": True,
        }
    task_frame = build_task_frame(
        target_cloud.centroid_camera_m,
        support_plane,
        workspace_size_m=args.workspace_size_m,
        table_height_m=args.table_height_m,
        allow_camera_aligned_fallback=args.allow_camera_aligned_fallback,
    )
    atomic_write_json(sample_dir / "support_plane.json", support_payload)
    atomic_write_json(
        sample_dir / "workspace_frame.json",
        {
            **task_frame.to_dict(),
            "target_cloud": target_cloud.to_dict(),
            "intrinsics": intrinsics.to_dict(),
            "depth": depth_result.log_dict(),
            "mask": mask_result.diagnostics.to_dict(),
            "tsdf_mode": TSDF_MODE,
        },
    )

    scene_points = backproject_depth(depth_result.depth_m, intrinsics)
    scene_task = transform_points(invert_transform(task_frame.T_camera_task), scene_points)
    local = np.all(
        (scene_task >= 0.0) & (scene_task < float(args.workspace_size_m)), axis=1
    )
    local_scene_points = scene_points[local]

    tsdf_start = time.perf_counter()
    tsdf = build_tsdf_grid(
        depth_result.depth_m,
        intrinsics,
        task_frame.T_camera_task,
        vgn_root=args.vgn_root,
        workspace_size_m=args.workspace_size_m,
        resolution=args.resolution,
        depth_trunc_m=args.depth_trunc_m,
        preset=args.vgn_preset,
        logger=LOGGER,
    )
    partial["processing_time_tsdf"] = elapsed(tsdf_start)
    if args.save_tsdf:
        atomic_write_npz(
            sample_dir / "tsdf_grid.npz",
            tsdf_grid=tsdf.grid,
            workspace_size_m=np.asarray(args.workspace_size_m),
            voxel_size_m=np.asarray(tsdf.voxel_size_m),
        )
    if args.save_pointclouds:
        atomic_write_point_cloud(sample_dir / "local_scene_point_cloud.ply", local_scene_points)
        atomic_write_point_cloud(
            sample_dir / "target_point_cloud.ply", target_cloud.points_camera_m
        )

    vgn_start = time.perf_counter()
    prediction = predict_official(tsdf.grid, net, device, logger=LOGGER)
    post = run_official_postprocessing(
        tsdf.grid,
        prediction.qual_vol,
        prediction.rot_vol,
        prediction.width_vol,
        voxel_size_m=tsdf.voxel_size_m,
    )
    dilated = dilate_mask(mask_result.mask, args.target_mask_dilation_px)
    all_candidates, target_candidates_raw = filter_target_candidates(
        post.candidates,
        intrinsics=intrinsics,
        raw_target_mask=mask_result.raw_resized_mask,
        dilated_target_mask=dilated,
        target_mask_dilation_px=args.target_mask_dilation_px,
        depth_m=depth_result.depth_m,
        target_points_camera=target_cloud.points_camera_m,
        T_camera_task=task_frame.T_camera_task,
    )
    # Preserve the original upstream selection order for the all-candidate
    # collection, but annotate every record with its deterministic quality
    # rank.  The target collection itself is emitted in that ranked order.
    all_quality_ranked = sort_candidates_by_quality(all_candidates)
    all_score_ranks = {
        candidate.official_selection_index: candidate.score_rank
        for candidate in all_quality_ranked
    }
    all_candidates = [
        replace(
            candidate,
            score_rank=all_score_ranks[candidate.official_selection_index],
        )
        for candidate in all_candidates
    ]
    target_candidates = sort_candidates_by_quality(target_candidates_raw)
    partial["processing_time_vgn"] = elapsed(vgn_start)
    partial["official_candidate_count"] = len(all_candidates)
    partial["target_candidate_count"] = len(target_candidates)

    atomic_write_candidates_npz(sample_dir / "candidates_all.npz", all_candidates)
    atomic_write_candidates_npz(sample_dir / "candidates_target.npz", target_candidates)
    all_records = candidate_records(all_candidates)
    target_records = candidate_records(target_candidates)
    atomic_write_json(
        sample_dir / "candidates.json",
        {
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "instruction": sample.instruction,
            "coordinate_frames": {
                "task": "local VGN workspace",
                "camera": "OCID camera optical frame",
            },
            "score_source": SCORE_SOURCE,
            "custom_reranking": False,
            "official_candidate_count": len(all_records),
            "target_filtered_candidate_count": len(target_records),
            "off_target_rejection_count": len(all_records) - len(target_records),
            "inference_device_requested": prediction.requested_device,
            "inference_device_used": prediction.used_device,
            "mps_fallback_reason": prediction.mps_fallback_reason,
            "all_official_vgn_candidates": all_records,
            "target_filtered_vgn_candidates": target_records,
            "deterministic_top_k": target_records[: args.top_k],
        },
    )

    status = candidate_status(len(all_candidates), len(target_candidates))
    top1 = None
    if status == "ok":
        selected = select_candidate(
            target_candidates_raw, policy=args.selection_policy, seed=args.seed
        )
        assert selected is not None
        top1 = next(
            candidate
            for candidate in target_candidates
            if candidate.official_selection_index == selected.official_selection_index
        )
        partial.update(
            top1_vgn_quality=top1.vgn_quality,
            top1_width_m=top1.width_m,
            top1_x_task=top1.position_task_m[0],
            top1_y_task=top1.position_task_m[1],
            top1_z_task=top1.position_task_m[2],
        )
    top1_payload = {
        "sample_id": sample.sample_id,
        "selection_policy": args.selection_policy,
        "score_source": SCORE_SOURCE,
        "custom_reranking": False,
        "candidate_count_before_target_filter": len(all_candidates),
        "candidate_count_after_target_filter": len(target_candidates),
        "off_target_rejection_count": len(all_candidates) - len(target_candidates),
        "status": status,
        "limitations": LIMITATIONS,
        "tsdf_mode": TSDF_MODE,
        "non_official_geometry_fallback": task_frame.non_official_geometry_fallback,
        "inference_device_requested": prediction.requested_device,
        "inference_device_used": prediction.used_device,
        "mps_fallback_reason": prediction.mps_fallback_reason,
        "candidate": None if top1 is None else top1.to_record(),
    }
    atomic_write_json(sample_dir / "top1.json", top1_payload)
    if args.visualize:
        _save_sample_visuals(
            sample_dir,
            rgb=rgb,
            mask=mask_result.mask,
            intrinsics=intrinsics,
            all_candidates=all_candidates,
            target_candidates=target_candidates,
            top1=top1,
            processed_quality=post.processed_quality,
            local_scene_points=local_scene_points,
            target_points=target_cloud.points_camera_m,
            top_k=args.top_k,
            T_camera_task=task_frame.T_camera_task,
            workspace_size_m=args.workspace_size_m,
            table_height_m=args.table_height_m,
        )
    if status == "ok":
        atomic_write_json(
            sample_dir / "_SUCCESS.json",
            {"sample_id": sample.sample_id, "status": "ok"},
        )
    return {
        **empty_summary(sample, status=status, reason=""),
        **partial,
        "status": status,
        "failure_reason": "" if status == "ok" else status,
    }


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.ocid_root = args.ocid_root.expanduser().resolve()
    args.hifi_root = args.hifi_root.expanduser().resolve()
    args.vgn_root = args.vgn_root.expanduser().resolve()
    args.vgn_weights = args.vgn_weights.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.manifest is not None:
        args.manifest = args.manifest.expanduser().resolve()
    _validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None and args.manifest.is_file():
        samples, mapping = load_manifest_samples(
            args.manifest,
            ocid_root=args.ocid_root,
            hifi_root=args.hifi_root,
            logger=LOGGER,
        )
    else:
        if args.manifest is not None:
            LOGGER.warning("Manifest missing; invoking strict stem fallback: %s", args.manifest)
        samples = build_stem_manifest(
            ocid_root=args.ocid_root,
            hifi_root=args.hifi_root,
            report_path=args.output / "unmatched_files.json",
        )
        mapping = {"fallback": "strict_file_stem"}
    selected_samples = _select_samples(samples, args)

    device_info = resolve_device_info(args.device, logger=LOGGER)
    metadata = runtime_metadata(vgn_root=args.vgn_root, weights_path=args.vgn_weights)
    metadata.update(
        environment=environment_versions(),
        manifest_path=None if args.manifest is None else str(args.manifest),
        manifest_field_mapping=mapping,
        ocid_root=str(args.ocid_root),
        hifi_root=str(args.hifi_root),
        output=str(args.output),
        intrinsics_argument=args.intrinsics,
        depth_unit_argument=args.depth_unit,
        depth_scale=args.depth_scale,
        tsdf_mode=TSDF_MODE,
        vgn_preset=args.vgn_preset,
        selection_policy=args.selection_policy,
        score_source=SCORE_SOURCE,
        custom_reranking=False,
        device_requested=args.device,
        device_resolved=device_info.resolved,
        device_fallback_reason=device_info.fallback_reason,
        official_postprocessing={
            "gaussian_filter_sigma": 1.0,
            "min_width_voxels": OFFICIAL_MIN_WIDTH_VOXELS,
            "max_width_voxels": OFFICIAL_MAX_WIDTH_VOXELS,
            "quality_threshold": OFFICIAL_QUALITY_THRESHOLD,
            "maximum_filter_size": OFFICIAL_MAX_FILTER_SIZE,
        },
        limitations=LIMITATIONS,
        cli_arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    signature, signature_inputs = _run_signature(args, metadata)
    metadata["run_signature_sha256"] = signature
    metadata["run_signature_inputs"] = signature_inputs
    _prepare_run_output(args.output, metadata, overwrite=args.overwrite)
    LOGGER.info(
        "Official VGN: threshold=%.2f workspace=%.2fm voxel=%.4fm commit=%s checkpoint_sha256=%s",
        OFFICIAL_QUALITY_THRESHOLD,
        OFFICIAL_WORKSPACE_SIZE_M,
        OFFICIAL_VOXEL_SIZE_M,
        metadata["repository_commit"],
        metadata["checkpoint_sha256"],
    )
    net = load_official_network(
        args.vgn_weights,
        device=device_info.resolved,
        vgn_root=args.vgn_root,
        logger=LOGGER,
    )

    summaries = _load_existing_summary(args.output / "summary.csv")
    processed = 0
    for sample in selected_samples:
        sample_dir = args.output / "samples" / sample.sample_id
        prior_status = _existing_status(sample_dir)
        if not args.overwrite and prior_status == "ok":
            LOGGER.info("[%s] skip completed sample", sample.sample_id)
            continue
        if (
            not args.overwrite
            and prior_status is not None
            and prior_status != "ok"
            and not args.retry_failures
        ):
            LOGGER.info("[%s] skip prior failure (use --retry-failures)", sample.sample_id)
            continue
        if (args.overwrite or args.retry_failures) and sample_dir.exists():
            shutil.rmtree(sample_dir)
        LOGGER.info("[%s] processing scene=%s instruction=%r", sample.sample_id, sample.scene_id, sample.instruction)
        try:
            summary = process_sample(
                sample,
                args=args,
                net=net,
                device=device_info.resolved,
            )
        except GeometryError as error:
            LOGGER.error("[%s] %s: %s", sample.sample_id, error.status, error)
            summary = _sample_failure(sample, sample_dir, args, error.status, str(error))
        except (PipelineError, VGNAdapterError) as error:
            status = getattr(error, "status", "vgn_inference_failed")
            LOGGER.exception("[%s] %s", sample.sample_id, error)
            summary = _sample_failure(sample, sample_dir, args, status, str(error))
        except Exception as error:  # Batch isolation is intentional and recorded.
            LOGGER.exception("[%s] unexpected processing failure", sample.sample_id)
            summary = _sample_failure(
                sample, sample_dir, args, "processing_error", f"{type(error).__name__}: {error}"
            )
        summaries[sample.sample_id] = summary
        ordered = [summaries[key] for key in sorted(summaries)]
        atomic_write_csv(args.output / "summary.csv", ordered, SUMMARY_FIELDS)
        failures = [row for row in ordered if row.get("status") != "ok"]
        atomic_write_csv(args.output / "failures.csv", failures, SUMMARY_FIELDS)
        processed += 1
    LOGGER.info("Finished %d samples; summary=%s", processed, args.output / "summary.csv")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (PipelineError, VGNAdapterError, GeometryError, FileNotFoundError, ValueError) as error:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
