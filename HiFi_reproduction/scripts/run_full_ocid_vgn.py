#!/usr/bin/env python3
"""Run the immutable 7,675-row OCID-VLG predicted-mask VGN experiment."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from scripts.run_vgn_on_hifics import _validate_args
from src.experiments.bootstrap import bootstrap_experiment
from src.experiments.experiment_store import ExperimentStore
from src.experiments.full_vgn_runner import (
    FullVGNRunner,
    benchmark_summary,
    manifest_registration_rows,
)
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
    load_official_network,
    resolve_device_info,
    runtime_metadata,
)
from src.grasping.vgn_pipeline import (
    LIMITATIONS,
    SCORE_SOURCE,
    TSDF_MODE,
    atomic_write_json,
    environment_versions,
    load_manifest_samples,
    sha256_file,
)


LOGGER = logging.getLogger("full_ocid_vgn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hifi-root", type=Path, required=True)
    parser.add_argument("--vgn-root", type=Path, default=Path("third_party/vgn"))
    parser.add_argument(
        "--vgn-weights",
        type=Path,
        default=Path("third_party/vgn/data/models/vgn_conv.pth"),
    )
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/hifics_vgn_full"))
    parser.add_argument("--expected-count", type=int, default=7675)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--vgn-preset", choices=("official",), default="official")
    parser.add_argument(
        "--selection-policy",
        choices=("highest_vgn_quality", "official_sim_random", "official_panda_highest_z"),
        default="highest_vgn_quality",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-technical-failures", action="store_true")
    parser.add_argument("--retry-model-outcomes", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--preprocess-workers", type=int, default=1)
    parser.add_argument("--vgn-batch-size", type=int, default=1)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--scene-cache-size", type=int, default=4)
    parser.add_argument("--lease-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--save-pointclouds", choices=("none", "all", "representative"), default="representative"
    )
    parser.add_argument(
        "--save-tsdf",
        choices=("none", "all", "failures-and-representative"),
        default="failures-and-representative",
    )
    parser.add_argument("--render-all-2d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-3d", choices=("none", "all", "representative"), default="representative")
    parser.add_argument("--mask-source", choices=("predicted", "gt-oracle"), default="predicted")
    parser.add_argument("--mask-cleanup", choices=("none", "largest-component", "close"), default="none")
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
    parser.add_argument("--log-level", default="INFO")
    return parser


def _resolve(args: argparse.Namespace) -> None:
    for name in ("ocid_root", "manifest", "hifi_root", "vgn_root", "vgn_weights", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if args.start_index is not None and args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < 0:
        raise ValueError("--end-index must be non-negative")
    if args.start_index is not None and args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must not be smaller than --start-index")
    for name in ("preprocess_workers", "vgn_batch_size", "render_workers", "scene_cache_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    # Existing processor options that are intentionally fixed by the full runner.
    args.max_samples = None
    args.overwrite = False
    args.retry_failures = False
    args.visualize = False
    args.multi_view_manifest = None
    _validate_args(args)


def _signature_payload(args: argparse.Namespace, mapping: dict[str, Any]) -> dict[str, Any]:
    stable_options = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key
        not in {
            "start_index",
            "end_index",
            "sample_id",
            "resume",
            "retry_technical_failures",
            "retry_model_outcomes",
            "force",
            "log_level",
            "preprocess_workers",
            "render_workers",
            "vgn_batch_size",
        }
    }
    return {
        "stable_options": stable_options,
        "manifest_field_mapping": mapping,
        "manifest_sha256": sha256_file(args.manifest),
        "implementation_sha256": {
            path: sha256_file(Path(path))
            for path in (
                "src/grasping/vgn_adapter.py",
                "src/grasping/vgn_geometry.py",
                "src/grasping/vgn_pipeline.py",
                "scripts/run_vgn_on_hifics.py",
                "src/experiments/full_vgn_runner.py",
            )
        },
    }


def _selected(samples: list[Any], args: argparse.Namespace) -> list[Any]:
    result = samples
    if args.sample_id:
        result = [sample for sample in result if sample.sample_id == args.sample_id]
        if not result:
            raise KeyError(f"unknown --sample-id {args.sample_id}")
    if args.start_index is not None:
        result = [sample for sample in result if sample.dataset_index >= args.start_index]
    if args.end_index is not None:
        result = [sample for sample in result if sample.dataset_index < args.end_index]
    return result


def run(args: argparse.Namespace) -> int:
    _resolve(args)
    args.output.mkdir(parents=True, exist_ok=True)
    samples, mapping = load_manifest_samples(
        args.manifest,
        ocid_root=args.ocid_root,
        hifi_root=args.hifi_root,
        logger=LOGGER,
    )
    if len(samples) != int(args.expected_count):
        raise ValueError(
            f"manifest count guard failed: expected {args.expected_count}, found {len(samples)}"
        )

    device_info = resolve_device_info(args.device, logger=LOGGER)
    metadata = runtime_metadata(vgn_root=args.vgn_root, weights_path=args.vgn_weights)
    signature_payload = _signature_payload(args, mapping)
    metadata.update(
        tsdf_mode=TSDF_MODE,
        score_source=SCORE_SOURCE,
        custom_reranking=False,
        limitations=LIMITATIONS,
        selection_policy=args.selection_policy,
        mask_source=args.mask_source,
        manifest_path=str(args.manifest),
        manifest_sha256=signature_payload["manifest_sha256"],
        manifest_field_mapping=mapping,
        run_signature_sha256=__import__("hashlib").sha256(
            json.dumps(signature_payload, sort_keys=True).encode()
        ).hexdigest(),
        environment=environment_versions(),
        device_requested=args.device,
        device_resolved=device_info.resolved,
        device_fallback_reason=device_info.fallback_reason,
        execution_engine={
            "model_load_count": 1,
            "preprocess_workers_requested": args.preprocess_workers,
            "preprocess_workers_effective": 1,
            "vgn_batch_size_requested": args.vgn_batch_size,
            "vgn_batch_size_effective": 1,
            "render_workers_requested": args.render_workers,
            "render_workers_effective": 1,
            "reason": "single-process deterministic SQLite runner; model remains loaded once",
        },
        official_postprocessing={
            "gaussian_filter_sigma": 1.0,
            "min_width_voxels": OFFICIAL_MIN_WIDTH_VOXELS,
            "max_width_voxels": OFFICIAL_MAX_WIDTH_VOXELS,
            "quality_threshold": OFFICIAL_QUALITY_THRESHOLD,
            "maximum_filter_size": OFFICIAL_MAX_FILTER_SIZE,
        },
    )
    run_id = f"ocid_vgn_{args.mask_source.replace('-', '_')}"
    database = args.output / "experiment.sqlite3"
    with ExperimentStore(database, run_id) as store:
        bootstrap = bootstrap_experiment(
            store,
            manifest_registration_rows(samples),
            metadata,
            manifest_count=args.expected_count,
        )
        chosen = _selected(samples, args)
        chosen_ids = [sample.sample_id for sample in chosen]
        if args.force:
            statuses = {
                str(row["outcome_status"])
                for row in store.sample_rows()
                if row.get("outcome_status")
            }
            reset = store.requeue_statuses(statuses, sample_ids=chosen_ids)
            for sample in chosen:
                directory = args.output / "samples" / sample.sample_id
                if directory.exists():
                    shutil.rmtree(directory)
            LOGGER.warning("--force reset %d selected records", reset)
        elif args.retry_technical_failures:
            reset = store.requeue_retryable()
            LOGGER.info("requeued %d retryable technical failures", reset)
        if args.retry_model_outcomes:
            reset = store.requeue_statuses(
                {"no_official_grasp", "no_target_grasp"}, sample_ids=chosen_ids
            )
            LOGGER.info("requeued %d model outcomes", reset)

        atomic_write_json(
            args.output / "run_config.json",
            {
                **metadata,
                "run_id": run_id,
                "database": str(database),
                "expected_manifest_count": args.expected_count,
                "bootstrap": bootstrap,
                "cli_arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "truthful_scope": {
                    "offline_metric": "candidate coverage, not physical grasp success",
                    "simulation_metric": "not computed by this command",
                    "real_robot_grasp_success_rate": None,
                },
            },
        )
        LOGGER.info(
            "Official VGN threshold=%.2f workspace=%.2fm voxel=%.4fm commit=%s checkpoint_sha256=%s",
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
        # Map retention policies to the existing per-sample booleans.  The
        # representative subset is selected after metrics, never for all rows.
        requested_pointcloud_policy = args.save_pointclouds
        requested_tsdf_policy = args.save_tsdf
        args.save_pointclouds = requested_pointcloud_policy == "all"
        args.save_tsdf = requested_tsdf_policy == "all"
        started = time.perf_counter()
        runner = FullVGNRunner(
            samples=samples,
            store=store,
            args=args,
            net=net,
            device=device_info.resolved,
            run_metadata=metadata,
        )
        progress = runner.run_pending(chosen)
        wall = time.perf_counter() - started
        rows = [row for row in store.sample_rows() if row["sample_id"] in set(chosen_ids)]
        benchmark = benchmark_summary(
            rows,
            manifest_count=args.expected_count,
            wall_time_s=wall,
            output_dir=args.output,
        )
        progress.update(
            wall_time_s=wall,
            benchmark=benchmark,
            save_pointclouds_policy=requested_pointcloud_policy,
            save_tsdf_policy=requested_tsdf_policy,
        )
        atomic_write_json(args.output / "run_progress.json", progress)
        if len(chosen) == 100:
            atomic_write_json(args.output / "benchmark_100.json", benchmark)
        store.checkpoint(truncate=True)
    LOGGER.info("finished selected=%d output=%s", len(chosen), args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, str(args.log_level).upper()))
    try:
        return run(args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
