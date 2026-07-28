#!/usr/bin/env python3
"""Generate target-aware planar 4-DoF grasps from completed HiFi masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "gqcnn-official"))

from autolab_core import YamlConfig  # noqa: E402

from src.grasping.dexnet_adapter import (  # noqa: E402
    export_intrinsics_file,
    gqcnn_runtime,
    make_camera_intrinsics,
    make_rgbd_and_segmask,
)
from src.grasping.dexnet_candidate_generator import (  # noqa: E402
    CandidateGenerationResult,
    generate_candidates,
)
from src.grasping.dexnet_scoring import (  # noqa: E402
    GQCNNScoringUnavailable,
    resolve_model_directory,
    score_fixed_candidates,
)
from src.grasping.dexnet_run_reliability import (  # noqa: E402
    FAILED,
    SUCCESS_EMPTY,
    SUCCESS_NONEMPTY,
    SUCCESS_REQUIRED_FILES,
    append_jsonl,
    atomic_commit_sample,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_json_hash,
    decide_sample_action,
    has_identity_mismatch,
    make_staging_directory,
    recover_interrupted_backup,
    remove_staging_directory,
    select_sample_ids,
    sha256_file,
    utc_timestamp,
    validate_sample_output,
    write_completion_marker,
)
from src.grasping.grasp_serialization import (  # noqa: E402
    save_candidate_bundle,
    save_candidates_json,
)
from src.grasping.grasp_visualization import (  # noqa: E402
    save_candidate_overlay,
    save_depth_visualization,
    save_mask_overlay,
)
from src.grasping.ocid_vlg_grasp_adapter import (  # noqa: E402
    OcidVlgBundleIndex,
    OcidVlgGraspSample,
)


SUMMARY_FIELDS = (
    "sample_id",
    "query",
    "mask_area_px",
    "valid_target_depth_px",
    "requested_candidate_count",
    "raw_candidate_count",
    "mask_validated_count",
    "post_nms_count",
    "scored_candidate_count",
    "best_gqcnn_q",
    "median_gqcnn_q",
    "generation_time_ms",
    "scoring_time_ms",
    "total_time_ms",
    "failure_reason",
    "status",
    "question_index",
    "scene_id",
)


def parse_args() -> argparse.Namespace:
    run = REPO_ROOT / "runs" / "hifics_ocidvlg_20260711_112921"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT.parent / "crog_reproduction" / "OCID-VLG",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=run / "anygrasp_input_predicted_mask",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "dexnet_candidates"
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "dexnet_candidates.yaml"
    )
    parser.add_argument("--sample-id")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--mode", choices=("candidate-only", "ranking", "cem"), default="candidate-only"
    )
    parser.add_argument(
        "--score-with-gqcnn",
        action="store_true",
        help="Alias for --mode ranking; scoring remains an optional stage",
    )
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--sample-seed-mode",
        choices=("fixed", "stable-sha256"),
        default="fixed",
        help="Use one fixed seed or derive a deterministic seed for each stable sample ID.",
    )
    parser.add_argument(
        "--seed-namespace",
        default="hifics-dexnet-v1",
        help="Domain separator used by stable-sha256 sample seed derivation.",
    )
    parser.add_argument("--mask-threshold", type=float)
    parser.add_argument("--mask-erode-px", type=int)
    parser.add_argument("--mask-dilate-px", type=int)
    parser.add_argument("--min-boundary-distance-px", type=float)
    parser.add_argument("--gripper-width-m", type=float)
    parser.add_argument("--gqcnn-model")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--overwrite",
        dest="overwrite_existing",
        action="store_true",
        help="Backward-compatible alias for --overwrite-existing",
    )
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--status-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--visualize-policy",
        choices=("all", "none", "failures", "sampled"),
        default="all",
    )
    parser.add_argument("--visualize-every", type=int, default=100)
    parser.add_argument("--max-failures", type=int)
    parser.add_argument("--stop-after-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def derive_sample_seed(
    sample_id: str,
    *,
    base_seed: int,
    mode: str,
    namespace: str,
) -> int:
    """Return a sampler-compatible deterministic seed for one stable sample ID."""
    if mode == "fixed":
        return int(base_seed)
    if mode != "stable-sha256":
        raise ValueError(f"unsupported sample seed mode: {mode}")
    payload = (
        f"{namespace}\0{int(base_seed)}\0{sample_id}".encode("utf-8")
    )
    # NumPy's legacy RandomState accepts unsigned 32-bit seeds. Avoid 2**32-1
    # so every derived value is accepted consistently by old and new runtimes.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def _finite_q_values(candidates: list[dict[str, Any]]) -> np.ndarray:
    """Return only real scores; candidate-only mode deliberately stores ``None``."""
    values = [
        np.nan if candidate.get("gqcnn_q_value") is None else candidate["gqcnn_q_value"]
        for candidate in candidates
    ]
    numeric = np.asarray(values, dtype=np.float64)
    return numeric[np.isfinite(numeric)]


def _write_downstream_inputs(sample: OcidVlgGraspSample, output: Path) -> None:
    """Persist non-visual inputs required by the existing downstream rankers."""
    np.save(output / "depth_m.npy", sample.depth_m, allow_pickle=False)
    _save_mask(output / "hifics_mask_processed.png", sample.target_mask_processed)


def _write_diagnostics(
    sample: OcidVlgGraspSample,
    result: CandidateGenerationResult,
    output: Path,
) -> None:
    Image.fromarray(sample.rgb, mode="RGB").save(output / "rgb.png")
    _save_mask(output / "hifics_mask_original.png", sample.target_mask_original)
    _save_mask(output / "valid_depth_mask.png", sample.valid_depth_mask)

    save_candidate_overlay(
        sample.rgb,
        result.raw_candidates,
        output / "raw_candidates_overlay.png",
        mask=sample.target_mask_processed,
        title=f"{sample.sample_id}: raw official antipodal candidates",
        show_scores=False,
    )
    save_candidate_overlay(
        sample.rgb,
        result.deduplicated_candidates,
        output / "filtered_candidates_overlay.png",
        mask=sample.target_mask_processed,
        title=f"{sample.sample_id}: target-validated + NMS",
        show_scores=False,
    )
    save_candidate_overlay(
        sample.rgb,
        result.topk_candidates,
        output / "topk_candidates_overlay.png",
        mask=sample.target_mask_processed,
        title=f"{sample.sample_id}: top-K planar 4-DoF candidates",
        show_scores=True,
    )
    save_candidate_overlay(
        sample.rgb,
        result.rejected_candidates,
        output / "rejected_candidates_overlay.png",
        mask=sample.target_mask_processed,
        title=f"{sample.sample_id}: rejected candidates",
        show_scores=False,
    )
    for reason in result.rejection_summary:
        selected = [
            candidate
            for candidate in result.rejected_candidates
            if reason in candidate.get("rejection_reasons", [])
        ]
        save_candidate_overlay(
            sample.rgb,
            selected,
            output / f"rejected_{reason}_overlay.png",
            mask=sample.target_mask_processed,
            title=f"Rejected: {reason}",
            show_scores=False,
        )
    save_depth_visualization(
        sample.depth_m,
        output / "depth_visualization.png",
        candidates=result.topk_candidates,
        mask=sample.target_mask_processed,
        title="Metric full-scene depth + top-K",
    )
    save_mask_overlay(
        sample.rgb,
        sample.target_mask_processed,
        output / "mask_overlay.png",
        candidates=result.topk_candidates,
        title="RGB + processed HiFi target mask + top-K",
    )


def _write_result(
    result: CandidateGenerationResult,
    output: Path,
    *,
    config: Mapping[str, Any],
    args: argparse.Namespace,
    model_name: str | None,
    model_dir: Path | None,
    sample_seed: int,
    scoring_failure_reason: str = "",
    write_visualizations: bool = True,
) -> dict[str, Any]:
    sample = result.sample
    finite_q = _finite_q_values(result.deduplicated_candidates)
    metadata = {
        "schema_version": 1,
        "sample_id": sample.sample_id,
        "sample_index": sample.sample_index,
        "question_index": sample.question_index,
        "scene_id": sample.scene_id,
        "query": sample.query,
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "camera_frame": sample.intrinsics.frame,
        "coordinate_units": {
            "image": "pixels; u right, v down; origin top-left",
            "depth": "metres",
            "angle": "radians counter-clockwise in image (u,v) coordinates",
            "width": "metres; configured maximum jaw opening",
        },
        "pose": {
            "name": "T_camera_grasp_fixed_approach",
            "source": "official gqcnn Grasp2D.pose()",
            "transform": "from grasp frame to camera frame",
            "fixed_approach_direction_camera": [0.0, 0.0, 1.0],
            "is_freely_predicted_6dof": False,
        },
        "input_bundle": str(sample.bundle_dir),
        "depth_source": str(sample.metadata["source_depth"]),
        "mask_source": str(sample.metadata["prediction_mask"]),
        "intrinsics_source": sample.intrinsics_metadata["source"],
        "factory_calibration": False,
        "depth_scale": 1000.0,
        "counts": {
            "requested": result.requested_candidate_count,
            "raw": len(result.raw_candidates),
            "mask_validated": len(result.mask_validated_candidates),
            "post_nms": len(result.deduplicated_candidates),
            "top_k": len(result.topk_candidates),
            "scored": int(finite_q.size),
        },
        "mask_area_px": int(np.count_nonzero(sample.target_mask_original)),
        "valid_target_depth_px": int(np.count_nonzero(sample.target_mask_processed)),
        "timing_ms": {
            "generation": result.generation_time_ms,
            "scoring": result.scoring_time_ms,
            "total": result.generation_time_ms + result.scoring_time_ms,
        },
        "rejection_summary": result.rejection_summary,
        "mode": args.mode,
        "model_name": model_name,
        "model_dir": None if model_dir is None else str(model_dir),
        "scoring_status": (
            "not_requested"
            if args.mode == "candidate-only"
            else ("failed" if scoring_failure_reason else "completed")
        ),
        "scoring_failure_reason": scoring_failure_reason or None,
        "seed": int(sample_seed),
        "seed_mode": args.sample_seed_mode,
        "seed_namespace": args.seed_namespace,
        "empty_category": (
            None
            if result.deduplicated_candidates
            else (
                "valid_empty_mask"
                if not np.any(sample.target_mask_original)
                else "valid_empty_candidates"
            )
        ),
        "failure_reason": (
            None
            if result.deduplicated_candidates
            else (
                "predicted_mask_empty"
                if not np.any(sample.target_mask_original)
                else (
                    "no_valid_depth_in_predicted_mask"
                    if not np.any(sample.target_mask_processed)
                    else (
                        "official_sampler_returned_no_candidates"
                        if not result.raw_candidates
                        else "no_candidates_survived_target_filtering_and_nms"
                    )
                )
            )
        ),
        "config": _plain(config),
        "gqcnn_runtime": gqcnn_runtime(),
        "gqcnn_score_is_calibrated_success_probability": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    save_candidates_json(output / "raw_candidates.json", result.raw_candidates)
    save_candidates_json(
        output / "mask_validated_candidates.json", result.mask_validated_candidates
    )
    save_candidates_json(
        output / "filtered_candidates.json", result.deduplicated_candidates
    )
    save_candidates_json(output / "topk_candidates.json", result.topk_candidates)
    save_candidate_bundle(
        result.deduplicated_candidates,
        json_path=output / "candidates.json",
        npz_path=output / "candidates.npz",
        csv_path=output / "candidates.csv",
        metadata=metadata,
    )
    (output / "rejection_summary.json").write_text(
        json.dumps(result.rejection_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    intrinsics = make_camera_intrinsics(
        {
            "fx": sample.intrinsics.fx,
            "fy": sample.intrinsics.fy,
            "cx": sample.intrinsics.cx,
            "cy": sample.intrinsics.cy,
            "skew": sample.intrinsics.skew,
            "height": sample.intrinsics.height,
            "width": sample.intrinsics.width,
        },
        frame=sample.intrinsics.frame,
    )
    export_intrinsics_file(intrinsics, output / "camera.intr")
    _write_downstream_inputs(sample, output)
    if write_visualizations:
        _write_diagnostics(sample, result, output)
    status = SUCCESS_NONEMPTY if result.deduplicated_candidates else SUCCESS_EMPTY
    return {
        "sample_id": sample.sample_id,
        "query": sample.query,
        "mask_area_px": int(np.count_nonzero(sample.target_mask_original)),
        "valid_target_depth_px": int(np.count_nonzero(sample.target_mask_processed)),
        "requested_candidate_count": result.requested_candidate_count,
        "raw_candidate_count": len(result.raw_candidates),
        "mask_validated_count": len(result.mask_validated_candidates),
        "post_nms_count": len(result.deduplicated_candidates),
        "scored_candidate_count": int(finite_q.size),
        "best_gqcnn_q": float(np.max(finite_q)) if finite_q.size else "",
        "median_gqcnn_q": float(np.median(finite_q)) if finite_q.size else "",
        "generation_time_ms": result.generation_time_ms,
        "scoring_time_ms": result.scoring_time_ms,
        "total_time_ms": result.generation_time_ms + result.scoring_time_ms,
        "failure_reason": (
            scoring_failure_reason
            or (
                ""
                if result.deduplicated_candidates
                else (
                    "official_sampler_returned_no_candidates"
                    if not result.raw_candidates
                    else "no_candidates_survived_target_filtering_and_nms"
                )
            )
        ),
        "status": status,
        "question_index": sample.question_index,
        "scene_id": sample.scene_id,
    }


def _rank_with_gqcnn(
    result: CandidateGenerationResult,
    *,
    model_name: str | None,
    model_dir: Path | None,
    scoring_config: Mapping[str, Any],
    top_k: int,
) -> None:
    """Score the frozen raw candidate list through the official bulk API."""
    runtime = gqcnn_runtime()
    if not runtime["scoring_import_available"]:
        raise GQCNNScoringUnavailable(
            "official GQ-CNN v1.3.0 scoring requires TensorFlow<=1.15, which "
            "has no Python 3.11/macOS arm64 build"
        )
    if model_name is None or model_dir is None:
        raise GQCNNScoringUnavailable(
            "ranking requires --gqcnn-model or an explicit --model-dir"
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"GQ-CNN model directory missing: {model_dir}; use the official "
            "scripts/downloads/models/download_models.sh in a compatible environment"
        )

    sample = result.sample
    intrinsics = make_camera_intrinsics(
        {
            "fx": sample.intrinsics.fx,
            "fy": sample.intrinsics.fy,
            "cx": sample.intrinsics.cx,
            "cy": sample.intrinsics.cy,
            "skew": sample.intrinsics.skew,
            "height": sample.intrinsics.height,
            "width": sample.intrinsics.width,
        },
        frame=sample.intrinsics.frame,
    )
    rgbd, segmask = make_rgbd_and_segmask(
        sample.rgb,
        sample.depth_m,
        sample.target_mask_processed,
        frame=sample.intrinsics.frame,
    )
    # Match the official saved-image policy example before neural crop
    # transformation. The original metric depth array remains unchanged.
    inpainted_depth = rgbd.depth.inpaint(
        rescale_factor=float(scoring_config.get("inpaint_rescale_factor", 0.5))
    )
    from autolab_core import RgbdImage
    from gqcnn.grasping import RgbdImageState

    state = RgbdImageState(
        RgbdImage.from_color_and_depth(rgbd.color, inpainted_depth),
        intrinsics,
        segmask=segmask,
    )
    started = time.perf_counter()
    q_values = score_fixed_candidates(
        state,
        result.official_grasps,
        model_name=model_name,
        model_dir=model_dir,
        scoring_config={
            "crop_height": int(scoring_config.get("crop_height", 96)),
            "crop_width": int(scoring_config.get("crop_width", 96)),
        },
        policy_config={"vis": {"tf_images": False, "k": int(top_k)}},
    )
    result.scoring_time_ms = (time.perf_counter() - started) * 1000.0
    order = np.argsort(-q_values, kind="stable")
    rank_by_index = {
        int(raw_index): rank + 1 for rank, raw_index in enumerate(order)
    }
    for raw_index, (candidate, value) in enumerate(
        zip(result.raw_candidates, q_values, strict=True)
    ):
        candidate["gqcnn_q_value"] = float(value)
        candidate["gqcnn_rank"] = rank_by_index[raw_index]
        candidate["model_name"] = model_name
    result.deduplicated_candidates.sort(
        key=lambda candidate: float(candidate["gqcnn_q_value"]), reverse=True
    )
    result.topk_candidates = result.deduplicated_candidates[: max(0, int(top_k))]


def _failure_row(sample_id: str, query: str, error: Exception) -> dict[str, Any]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "sample_id": sample_id,
            "query": query,
            "failure_reason": f"{type(error).__name__}: {error}",
            "status": FAILED,
        }
    )
    return row


def _write_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_csv(output / "summary.partial.csv", rows, SUMMARY_FIELDS)


def _write_completed_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    """Publish a stable final summary only after the full selection is clean."""

    atomic_write_csv(output / "summary.csv", rows, SUMMARY_FIELDS)
    atomic_write_text(output / "failures.jsonl", "")


def _log(output: Path, message: str, *, error: bool = False) -> None:
    line = f"{utc_timestamp()} {message}"
    print(line, file=sys.stderr if error else sys.stdout, flush=True)
    with (output / "run.log").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _visualizations_requested(
    policy: str, *, selected_position: int, visualize_every: int, is_empty: bool
) -> bool:
    if policy == "all":
        return True
    if policy == "none":
        return False
    if policy == "failures":
        return is_empty
    return selected_position % visualize_every == 0


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=True) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _checkpoint(
    output: Path,
    *,
    sample_ids: list[str],
    rows_by_id: Mapping[str, dict[str, Any]],
    attempted: int,
    skipped_valid: int,
    current_sample_id: str | None,
    started: float,
    configuration_hash: str,
) -> dict[str, Any]:
    rows = [rows_by_id[sample_id] for sample_id in sample_ids if sample_id in rows_by_id]
    _write_summary(output, rows)
    failed_rows = [row for row in rows if row.get("status") == FAILED]
    atomic_write_text(
        output / "failures.partial.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=True) + "\n"
            for row in failed_rows
        ),
    )
    counts = {
        status: sum(row.get("status") == status for row in rows)
        for status in (SUCCESS_NONEMPTY, SUCCESS_EMPTY, FAILED)
    }
    elapsed = max(0.0, time.time() - started)
    remaining = len(sample_ids) - len(rows)
    mean = elapsed / attempted if attempted else 0.0
    progress = {
        "total_expected": len(sample_ids),
        "attempted": attempted,
        "skipped_valid": skipped_valid,
        "success_nonempty": counts[SUCCESS_NONEMPTY],
        "success_empty": counts[SUCCESS_EMPTY],
        "failed": counts[FAILED],
        "remaining": remaining,
        "current_sample_id": current_sample_id,
        "elapsed_seconds": elapsed,
        "mean_time_per_attempted_sample_seconds": mean,
        "estimated_remaining_seconds": mean * remaining if attempted else None,
        "start_timestamp": utc_timestamp(started),
        "last_update_timestamp": utc_timestamp(),
        "process_id": os.getpid(),
        "configuration_hash": configuration_hash,
        "command_line": sys.argv,
    }
    atomic_write_json(output / "progress.json", progress)
    return progress


def main() -> int:
    args = parse_args()
    if args.retry_failures and not args.resume:
        raise ValueError("--retry-failures requires --resume")
    if args.verify_existing and not args.resume:
        raise ValueError("--verify-existing requires --resume")
    if args.status_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("status/checkpoint intervals must be positive")
    if args.visualize_every <= 0:
        raise ValueError("--visualize-every must be positive")
    if args.max_failures is not None and args.max_failures < 0:
        raise ValueError("--max-failures must be non-negative")
    if args.stop_after_seconds is not None and args.stop_after_seconds < 0:
        raise ValueError("--stop-after-seconds must be non-negative")
    if args.visualize:
        args.visualize_policy = "all"
    if args.score_with_gqcnn:
        if args.mode == "cem":
            raise ValueError("--score-with-gqcnn cannot be combined with --mode cem")
        args.mode = "ranking"
    config = _plain(dict(YamlConfig(str(args.config))))
    input_config = dict(config["input"])
    sampling_config = dict(config["sampling"])
    generation_config = dict(config["generation"])
    filtering_config = dict(config["filtering"])
    scoring_config = dict(config["scoring"])

    args.num_candidates = int(
        args.num_candidates or generation_config["num_grasp_samples"]
    )
    args.top_k = int(args.top_k or generation_config["top_k"])
    args.seed = int(args.seed if args.seed is not None else generation_config["seed"])
    if args.mask_threshold is not None:
        input_config["mask_threshold"] = args.mask_threshold
        input_config["mask_source"] = "probability"
    if args.mask_erode_px is not None:
        input_config["mask_erode_px"] = args.mask_erode_px
    if args.mask_dilate_px is not None:
        input_config["mask_dilate_px"] = args.mask_dilate_px
    if args.min_boundary_distance_px is not None:
        filtering_config["min_center_boundary_distance_px"] = args.min_boundary_distance_px
    if args.gripper_width_m is not None:
        sampling_config["gripper_width"] = args.gripper_width_m
    config["input"] = input_config
    config["sampling"] = sampling_config
    config["generation"] = {
        **generation_config,
        "num_grasp_samples": args.num_candidates,
        "top_k": args.top_k,
        "seed": args.seed,
        "sample_seed_mode": args.sample_seed_mode,
        "seed_namespace": args.seed_namespace,
        "sample_seed_derivation": (
            "fixed base seed"
            if args.sample_seed_mode == "fixed"
            else "uint64_be(sha256(namespace\\0base_seed\\0stable_sample_id)[:8]) mod (2**32-1)"
        ),
    }
    config["filtering"] = filtering_config

    index = OcidVlgBundleIndex(args.dataset_root, args.mask_root, split=args.split)
    canonical_ids = [str(row["sample_id"]) for row in index.rows]
    sample_ids = select_sample_ids(
        canonical_ids,
        sample_id=args.sample_id,
        sample_limit=args.sample_limit,
        start_index=args.start_index,
        end_index=args.end_index,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_OK",
                    "selected_count": len(sample_ids),
                    "samples": sample_ids,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_name, model_dir = resolve_model_directory(args.gqcnn_model, args.model_dir)
    runtime = gqcnn_runtime()
    configuration_hash = canonical_json_hash(config)
    config_file_hash = sha256_file(args.config.expanduser().resolve())
    manifest_hash = sha256_file(index.manifest_path)
    identity = {
        "dataset_root": str(index.dataset_root),
        "mask_root": str(index.mask_root),
        "manifest": str(index.manifest_path),
        "manifest_sha256": manifest_hash,
        "configuration_hash": configuration_hash,
        "config_file_sha256": config_file_hash,
        "seed": args.seed,
        "sample_seed_mode": args.sample_seed_mode,
        "seed_namespace": args.seed_namespace,
        "num_candidates": args.num_candidates,
        "top_k": args.top_k,
        "mode": args.mode,
        "sampler_commit": runtime.get("commit"),
        "sampler_version": runtime.get("version"),
    }
    run_config = {
        "schema_version": 1,
        "created_timestamp": utc_timestamp(),
        "identity": identity,
        "config_path": str(args.config.expanduser().resolve()),
        "resolved_config": config,
        "initial_command_line": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model_name": model_name,
        "model_dir": None if model_dir is None else str(model_dir),
        "gqcnn_runtime": runtime,
        "execution": {
            "visualize_policy": args.visualize_policy,
            "visualize_every": args.visualize_every,
            "status_every": args.status_every,
            "checkpoint_every": args.checkpoint_every,
        },
        "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "PATH")},
    }
    run_config_path = output / "run_config.json"
    if run_config_path.is_file():
        existing_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_run_config.get("identity") != identity:
            raise ValueError("existing run_config identity differs; use a new output root")
    else:
        atomic_write_json(run_config_path, run_config)

    selected_set = set(sample_ids)
    manifest_rows = [
        {"manifest_index": manifest_index, **dict(row)}
        for manifest_index, row in enumerate(index.rows)
        if str(row["sample_id"]) in selected_set
    ]
    run_manifest_path = output / "run_manifest.jsonl"
    if run_manifest_path.is_file():
        existing_ids = [str(row["sample_id"]) for row in _read_jsonl(run_manifest_path)]
        if existing_ids != sample_ids:
            raise ValueError("existing run_manifest selection differs; use a new output root")
    else:
        _write_manifest(run_manifest_path, manifest_rows)

    started = time.time()
    append_jsonl(
        output / "attempt_history.jsonl",
        {
            "event": "invocation_started",
            "timestamp": utc_timestamp(started),
            "process_id": os.getpid(),
            "resume": args.resume,
            "retry_failures": args.retry_failures,
            "command_line": sys.argv,
        },
    )
    _log(output, f"START selected={len(sample_ids)} configuration={configuration_hash}")
    rows_by_id: dict[str, dict[str, Any]] = {}
    skipped_valid = 0
    attempted = 0
    invocation_failures = 0
    decisions = 0
    stop_requested = False

    for selected_position, sample_id in enumerate(sample_ids):
        if args.stop_after_seconds is not None and time.time() - started >= args.stop_after_seconds:
            stop_requested = True
            _log(output, f"STOP_BUDGET reached before {sample_id}")
            break
        query = str(index.by_id[sample_id]["query"])
        question_index = int(index.by_id[sample_id]["question_index"])
        scene_id = str(index.by_id[sample_id]["scene_id"])
        sample_seed = derive_sample_seed(
            sample_id,
            base_seed=args.seed,
            mode=args.sample_seed_mode,
            namespace=args.seed_namespace,
        )
        recover_interrupted_backup(output, sample_id)
        sample_output = output / sample_id
        existing = None
        if sample_output.is_dir():
            existing = validate_sample_output(
                sample_output,
                expected_sample_id=sample_id,
                expected_configuration_hash=configuration_hash,
                expected_config_file_sha256=config_file_hash,
                expected_seed=sample_seed,
                expected_sampler_runtime=runtime,
                verify_hashes=args.verify_existing,
                allow_legacy=args.resume,
            )
            if existing.valid and existing.legacy:
                visual_files = sorted(
                    path.name
                    for path in sample_output.glob("*.png")
                    if path.name != "hifics_mask_processed.png"
                )
                write_completion_marker(
                    sample_output,
                    sample_id=sample_id,
                    question_index=question_index,
                    configuration_hash=configuration_hash,
                    config_file_sha256=config_file_hash,
                    seed=sample_seed,
                    sampler_runtime=runtime,
                    counts={
                        "requested": existing.summary_row["requested_candidate_count"],
                        "raw": existing.summary_row["raw_candidate_count"],
                        "mask_validated": existing.summary_row["mask_validated_count"],
                        "post_nms": existing.summary_row["post_nms_count"],
                        "top_k": len(
                            json.loads(
                                (sample_output / "topk_candidates.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                        ),
                    },
                    status=str(existing.status),
                    required_files=SUCCESS_REQUIRED_FILES,
                    visualization_files=visual_files,
                    summary_row=existing.summary_row,
                    failure_reason=existing.summary_row.get("failure_reason") or None,
                )
                existing = validate_sample_output(
                    sample_output,
                    expected_sample_id=sample_id,
                    expected_configuration_hash=configuration_hash,
                    expected_config_file_sha256=config_file_hash,
                    expected_seed=sample_seed,
                    expected_sampler_runtime=runtime,
                    verify_hashes=True,
                )
                _log(output, f"MIGRATED legacy marker {sample_id}")

        if (
            existing is not None
            and not existing.valid
            and has_identity_mismatch(existing)
            and not args.overwrite_existing
        ):
            raise ValueError(
                f"existing sample {sample_id} has a different run identity; "
                "use a new output root or pass --overwrite-existing explicitly: "
                f"{existing.errors}"
            )
        should_process = decide_sample_action(
            output_exists=sample_output.exists(),
            validation=existing,
            resume=args.resume,
            overwrite_existing=args.overwrite_existing,
            retry_failures=args.retry_failures,
        ) == "process"
        if not should_process:
            if existing is not None and existing.summary_row is not None:
                rows_by_id[sample_id] = dict(existing.summary_row)
            skipped_valid += 1
        elif sample_output.exists() and existing is not None and not existing.valid:
            _log(output, f"REGENERATE invalid {sample_id}: {existing.errors}")

        if should_process:
            attempted += 1
            previous_attempt = 0
            previous_failure = None
            if existing is not None and existing.marker is not None:
                previous_attempt = int(existing.marker.get("attempt", 0))
                previous_failure = existing.marker.get("failure_reason")
            attempt = previous_attempt + 1
            staging = make_staging_directory(output, sample_id)
            sample: OcidVlgGraspSample | None = None
            last_successful_stage = "attempt_started"
            append_jsonl(
                output / "attempt_history.jsonl",
                {
                    "event": "attempt_started",
                    "timestamp": utc_timestamp(),
                    "sample_id": sample_id,
                    "attempt": attempt,
                    "previous_failure": previous_failure,
                },
            )
            try:
                sample = index.load_sample(
                    sample_id,
                    camera_frame=config["camera_frame"],
                    mask_source=input_config["mask_source"],
                    mask_threshold=float(input_config["mask_threshold"]),
                    min_component_area_px=int(input_config["min_component_area_px"]),
                    retain_largest_component=bool(input_config["retain_largest_component"]),
                    mask_erode_px=int(input_config["mask_erode_px"]),
                    mask_dilate_px=int(input_config["mask_dilate_px"]),
                    allow_empty_mask=True,
                )
                last_successful_stage = "input_loaded"
                if not np.any(sample.target_mask_processed):
                    result = CandidateGenerationResult(
                        sample=sample,
                        official_grasps=[],
                        raw_candidates=[],
                        mask_validated_candidates=[],
                        deduplicated_candidates=[],
                        topk_candidates=[],
                        rejected_candidates=[],
                        rejection_summary={
                            (
                                "predicted_mask_empty"
                                if not np.any(sample.target_mask_original)
                                else "no_valid_depth_in_predicted_mask"
                            ): 1
                        },
                        requested_candidate_count=0,
                        generation_time_ms=0.0,
                    )
                else:
                    result = generate_candidates(
                        sample,
                        sampling_config,
                        filtering_config,
                        num_samples=args.num_candidates,
                        top_k=args.top_k,
                        seed=sample_seed,
                        visualize_sampler=False,
                    )
                last_successful_stage = "candidate_generation"
                scoring_failure_reason = ""
                if args.mode == "ranking":
                    try:
                        _rank_with_gqcnn(
                            result,
                            model_name=model_name,
                            model_dir=model_dir,
                            scoring_config=scoring_config,
                            top_k=args.top_k,
                        )
                    except Exception as error:
                        scoring_failure_reason = (
                            f"{type(error).__name__}: {error}; candidate-only outputs were preserved"
                        )
                elif args.mode == "cem":
                    scoring_failure_reason = (
                        "GQCNNScoringUnavailable: CEM requires the unavailable legacy "
                        "TensorFlow<=1.15 scoring runtime; candidate-only outputs were preserved"
                    )
                write_visualizations = _visualizations_requested(
                    args.visualize_policy,
                    selected_position=selected_position,
                    visualize_every=args.visualize_every,
                    is_empty=not result.deduplicated_candidates,
                )
                row = _write_result(
                    result,
                    staging,
                    config=config,
                    args=args,
                    model_name=model_name,
                    model_dir=model_dir,
                    sample_seed=sample_seed,
                    scoring_failure_reason=scoring_failure_reason,
                    write_visualizations=write_visualizations,
                )
                last_successful_stage = "result_serialized"
                visual_files = sorted(
                    path.name
                    for path in staging.glob("*.png")
                    if path.name != "hifics_mask_processed.png"
                )
                write_completion_marker(
                    staging,
                    sample_id=sample_id,
                    question_index=sample.question_index,
                    configuration_hash=configuration_hash,
                    config_file_sha256=config_file_hash,
                    seed=sample_seed,
                    sampler_runtime=runtime,
                    counts={
                        "requested": result.requested_candidate_count,
                        "raw": len(result.raw_candidates),
                        "mask_validated": len(result.mask_validated_candidates),
                        "post_nms": len(result.deduplicated_candidates),
                        "top_k": len(result.topk_candidates),
                    },
                    status=row["status"],
                    required_files=(*SUCCESS_REQUIRED_FILES, *visual_files),
                    visualization_files=visual_files,
                    summary_row=row,
                    failure_reason=row["failure_reason"] or None,
                    attempt=attempt,
                )
                staged_validation = validate_sample_output(
                    staging,
                    expected_sample_id=sample_id,
                    expected_configuration_hash=configuration_hash,
                    expected_config_file_sha256=config_file_hash,
                    expected_seed=sample_seed,
                    expected_sampler_runtime=runtime,
                    verify_hashes=True,
                )
                if not staged_validation.valid:
                    raise ValueError(f"staged output verification failed: {staged_validation.errors}")
                last_successful_stage = "staged_output_validation"
                atomic_commit_sample(staging, sample_output, output)
                rows_by_id[sample_id] = row
                append_jsonl(
                    output / "attempt_history.jsonl",
                    {
                        "event": "attempt_finished",
                        "timestamp": utc_timestamp(),
                        "sample_id": sample_id,
                        "attempt": attempt,
                        "status": row["status"],
                        "candidate_counts": {
                            "raw": row["raw_candidate_count"],
                            "mask_validated": row["mask_validated_count"],
                            "post_nms": row["post_nms_count"],
                        },
                    },
                )
                _log(output, json.dumps(row, ensure_ascii=False, sort_keys=True))
            except Exception as error:
                failure_text = f"{type(error).__name__}: {error}"
                failure_traceback = traceback.format_exc()
                failure_row = _failure_row(sample_id, query, error)
                failure_row.update(
                    {"question_index": question_index, "scene_id": scene_id}
                )
                if staging.exists():
                    remove_staging_directory(staging, output)
                staging = make_staging_directory(output, sample_id)
                if last_successful_stage == "attempt_started":
                    failure_category = "input_related"
                elif last_successful_stage == "input_loaded":
                    failure_category = "sampler_related"
                else:
                    failure_category = "serialization_or_validation_related"
                try:
                    atomic_write_json(
                        staging / "failure.json",
                        {
                            "sample_id": sample_id,
                            "query": query,
                            "question_index": question_index,
                            "scene_id": scene_id,
                            "failure_reason": failure_text,
                            "traceback": failure_traceback,
                            "last_successful_pipeline_stage": last_successful_stage,
                            "failure_category": failure_category,
                            "mask_area_px": (
                                None
                                if sample is None
                                else int(np.count_nonzero(sample.target_mask_original))
                            ),
                            "valid_target_depth_px": (
                                None
                                if sample is None
                                else int(np.count_nonzero(sample.target_mask_processed))
                            ),
                            "timestamp": utc_timestamp(),
                        },
                    )
                    write_completion_marker(
                        staging,
                        sample_id=sample_id,
                        question_index=question_index,
                        configuration_hash=configuration_hash,
                        config_file_sha256=config_file_hash,
                        seed=sample_seed,
                        sampler_runtime=runtime,
                        counts={},
                        status=FAILED,
                        required_files=("failure.json",),
                        summary_row=failure_row,
                        failure_reason=failure_text,
                        attempt=attempt,
                    )
                    atomic_commit_sample(staging, sample_output, output)
                finally:
                    if staging.exists():
                        remove_staging_directory(staging, output)
                rows_by_id[sample_id] = failure_row
                invocation_failures += 1
                append_jsonl(
                    output / "attempt_history.jsonl",
                    {
                        "event": "attempt_finished",
                        "timestamp": utc_timestamp(),
                        "sample_id": sample_id,
                        "attempt": attempt,
                        "status": FAILED,
                        "failure_reason": failure_text,
                    },
                )
                _log(output, f"ERROR {sample_id}: {failure_text}", error=True)

        decisions += 1
        if decisions % args.checkpoint_every == 0:
            _checkpoint(
                output,
                sample_ids=sample_ids,
                rows_by_id=rows_by_id,
                attempted=attempted,
                skipped_valid=skipped_valid,
                current_sample_id=sample_id,
                started=started,
                configuration_hash=configuration_hash,
            )
        if decisions % args.status_every == 0:
            _log(
                output,
                f"STATUS decisions={decisions}/{len(sample_ids)} attempted={attempted} "
                f"skipped={skipped_valid} terminal={len(rows_by_id)} failures={invocation_failures}",
            )
        if (
            args.max_failures is not None
            and invocation_failures > 0
            and invocation_failures >= args.max_failures
        ):
            stop_requested = True
            _log(output, f"STOP_MAX_FAILURES count={invocation_failures}", error=True)
            break

    progress = _checkpoint(
        output,
        sample_ids=sample_ids,
        rows_by_id=rows_by_id,
        attempted=attempted,
        skipped_valid=skipped_valid,
        current_sample_id=None,
        started=started,
        configuration_hash=configuration_hash,
    )
    if (
        not stop_requested
        and progress["remaining"] == 0
        and progress["failed"] == 0
        and len(rows_by_id) == len(sample_ids)
    ):
        _write_completed_summary(
            output,
            [rows_by_id[sample_id] for sample_id in sample_ids],
        )
    append_jsonl(
        output / "attempt_history.jsonl",
        {
            "event": "invocation_finished",
            "timestamp": utc_timestamp(),
            "process_id": os.getpid(),
            "stop_requested": stop_requested,
            "progress": progress,
        },
    )
    final = {
        "status": "STOPPED" if stop_requested else "DONE",
        "samples": len(rows_by_id),
        "failures": progress["failed"],
        "remaining": progress["remaining"],
        "output_dir": str(output),
    }
    _log(output, json.dumps(final, sort_keys=True))
    if stop_requested:
        return 75
    return 1 if progress["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
