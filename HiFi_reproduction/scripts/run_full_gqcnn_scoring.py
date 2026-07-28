#!/usr/bin/env python3
"""Score and rank immutable full-dataset Dex-Net candidates with GQCNN-2.1."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.score_existing_dexnet_candidates import make_official_state_and_grasps  # noqa: E402
from src.grasping.gqcnn_full_scoring import (  # noqa: E402
    GQCNN_COMMIT,
    MARKER_NAME,
    SCORING_FAILED,
    build_source_manifest,
    model_file_manifest,
    select_entries,
    make_staging_directory,
    remove_staging,
    quarantine_output,
    atomic_commit_sample,
    atomic_write_json,
    assert_disjoint_roots,
    cleanup_stale_staging,
    validate_scored_output,
    write_empty_sample,
    write_failed_sample,
    score_and_write_sample,
    write_root_state,
    write_run_statistics,
    sha256_file,
    utc_timestamp,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="GQCNN-2.1")
    parser.add_argument("--docker-image", default="vlmgrasp/gqcnn-score:1.3.0")
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--expected-model-config-hash", required=True)
    parser.add_argument("--crop-height", type=int, default=96)
    parser.add_argument("--crop-width", type=int, default=96)
    parser.add_argument("--inpaint-rescale-factor", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=25, help="root checkpoint interval in samples")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--source-manifest-only", action="store_true")
    parser.add_argument("--expected-samples", type=int, default=7675)
    parser.add_argument("--expected-nonempty", type=int, default=7620)
    parser.add_argument("--expected-empty", type=int, default=55)
    parser.add_argument("--expected-candidates", type=int, default=206538)
    parser.add_argument("--nearly-identical-atol", type=float, default=1e-12)
    parser.add_argument("--low-q-threshold", type=float, action="append")
    return parser.parse_args()


def verify_runtime():
    import tensorflow as tf
    from gqcnn.version import __version__ as gqcnn_version

    runtime = {
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "tensorflow_gpu_available": bool(tf.test.is_gpu_available()),
        "gqcnn_version": gqcnn_version,
        "gqcnn_commit": GQCNN_COMMIT,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if runtime["machine"] != "x86_64":
        raise RuntimeError("linux/amd64 runtime required")
    if runtime["python_version"] != "3.7.17":
        raise RuntimeError("Python 3.7.17 required")
    if runtime["tensorflow_version"] != "1.15.0":
        raise RuntimeError("TensorFlow 1.15.0 required")
    if runtime["tensorflow_gpu_available"]:
        raise RuntimeError("CPU-only scoring required")
    if runtime["gqcnn_version"] != "1.3.0":
        raise RuntimeError("GQ-CNN 1.3.0 required")
    return runtime


def emit(status, **values):
    payload = {"status": status, "timestamp": utc_timestamp()}
    payload.update(values)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("--batch-size and --log-every must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    candidate_root, output_root = assert_disjoint_roots(
        args.candidate_root, args.output_root
    )
    if not candidate_root.is_dir():
        raise FileNotFoundError("candidate root missing: %s" % candidate_root)
    model_dir = args.model_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest_path = output_root / "source_candidate_manifest.jsonl"
    emit("SOURCE_MANIFEST_START", candidate_root=str(candidate_root))
    entries, source_identity = build_source_manifest(
        candidate_root,
        manifest_path,
        expected_samples=args.expected_samples,
        expected_nonempty=args.expected_nonempty,
        expected_empty=args.expected_empty,
        expected_candidates=args.expected_candidates,
        verify_hashes=True,
    )
    emit("SOURCE_MANIFEST_OK", **source_identity)
    files, model_manifest_hash = model_file_manifest(model_dir)
    model_config_hash = sha256_file(model_dir / "config.json")
    if args.expected_model_config_hash and model_config_hash != args.expected_model_config_hash:
        raise RuntimeError("model config hash mismatch")
    model_info = {
        "model_name": args.model_name,
        "model_commit": GQCNN_COMMIT,
        "model_config_hash": model_config_hash,
        "model_file_manifest_hash": model_manifest_hash,
        "model_file_manifest": files,
        "docker_image": args.docker_image,
        "docker_image_id": args.docker_image_id,
    }
    selected = select_entries(
        entries,
        sample_ids=args.sample_id,
        sample_limit=args.sample_limit,
        start_index=args.start_index,
        end_index=args.end_index,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    stale_staging = cleanup_stale_staging(
        output_root, [entry["sample_id"] for entry in selected]
    )
    if stale_staging:
        emit(
            "STALE_STAGING_REMOVED",
            count=len(stale_staging),
            directories=stale_staging,
        )
    run_config = {
        "schema_version": 1,
        "candidate_root": str(candidate_root),
        "output_root": str(output_root),
        "source_identity": source_identity,
        "model": model_info,
        "crop_height": args.crop_height,
        "crop_width": args.crop_width,
        "inpaint_rescale_factor": args.inpaint_rescale_factor,
        "seed": args.seed,
        "selected_samples": len(selected),
        "selection": {
            "sample_ids": args.sample_id,
            "sample_limit": args.sample_limit,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "ranking": "raw q descending; exact ties candidate_id ascending",
        "created_at": utc_timestamp(),
    }
    atomic_write_json(output_root / "run_config.json", run_config)
    if args.source_manifest_only:
        emit("SOURCE_MANIFEST_ONLY_DONE", selected_samples=len(selected))
        return 0
    np.random.seed(args.seed)
    runtime = verify_runtime()
    run_config["runtime"] = runtime
    atomic_write_json(output_root / "run_config.json", run_config)

    actions = []
    for entry in selected:
        final = output_root / entry["sample_id"]
        if not final.exists():
            actions.append((entry, "process"))
            continue
        valid, status, marker, errors = validate_scored_output(
            final,
            candidate_root / entry["sample_id"],
            entry,
            model_info,
            args.seed,
            verify_hashes=args.verify_existing,
        )
        if valid and not args.overwrite:
            if status == SCORING_FAILED and args.retry_failed:
                actions.append((entry, "process"))
            elif args.resume:
                actions.append((entry, "skip"))
            else:
                raise FileExistsError("valid output exists: %s" % final)
        elif args.overwrite:
            actions.append((entry, "process"))
        elif args.resume:
            quarantine_output(final, output_root, label="corrupt")
            emit("CORRUPT_OUTPUT_QUARANTINED", sample_id=entry["sample_id"], errors=errors)
            actions.append((entry, "process"))
        else:
            raise RuntimeError("invalid output exists; pass --resume or --overwrite: %s" % final)

    to_process = [entry for entry, action in actions if action == "process"]
    skipped = sum(action == "skip" for _, action in actions)
    nonempty_to_process = [entry for entry in to_process if int(entry["candidate_count"]) > 0]
    quality_fn = None
    if nonempty_to_process:
        from gqcnn.grasping import GraspQualityFunctionFactory

        emit(
            "MODEL_LOAD_START",
            model_path=str(model_dir),
            tensorflow_version=runtime["tensorflow_version"],
            gqcnn_commit=GQCNN_COMMIT,
        )
        load_started = time.perf_counter()
        quality_fn = GraspQualityFunctionFactory.quality_function(
            "gqcnn",
            {
                "gqcnn_model": str(model_dir),
                "crop_height": int(args.crop_height),
                "crop_width": int(args.crop_width),
            },
        )
        emit(
            "MODEL_LOAD_OK",
            model_path=str(model_dir),
            model_load_seconds=time.perf_counter() - load_started,
            model_im_height=int(quality_fn.gqcnn.im_height),
            model_im_width=int(quality_fn.gqcnn.im_width),
            model_pose_dim=int(quality_fn.gqcnn.pose_dim),
        )
    emit(
        "SCORING_START",
        selected_samples=len(selected),
        process_samples=len(to_process),
        skipped_existing=skipped,
        process_candidates=sum(int(entry["candidate_count"]) for entry in to_process),
    )
    processed = 0
    processed_candidates = 0
    processed_empty = 0
    last_completed = None
    failures = 0
    controlled_stop = False
    try:
        for entry, action in actions:
            if action == "skip":
                last_completed = entry["sample_id"]
                continue
            if args.stop_after is not None and processed >= int(args.stop_after):
                controlled_stop = True
                emit("CONTROLLED_STOP", processed_samples=processed, last_completed_sample_id=last_completed)
                break
            sample_started = time.perf_counter()
            final = output_root / entry["sample_id"]
            staging = make_staging_directory(output_root, entry["sample_id"])
            stage = "initialize"
            committed_status = None
            try:
                if final.exists():
                    quarantine_output(final, output_root, label="replaced")
                if int(entry["candidate_count"]) == 0:
                    stage = "valid_empty"
                    write_empty_sample(staging, entry, model_info, args.seed)
                else:
                    stage = "official_gqcnn_scoring"
                    score_and_write_sample(
                        staging,
                        candidate_root / entry["sample_id"],
                        entry,
                        quality_fn,
                        make_official_state_and_grasps,
                        model_info,
                        args.seed,
                        inpaint_rescale_factor=args.inpaint_rescale_factor,
                    )
                stage = "validate_staging"
                valid, status, marker, errors = validate_scored_output(
                    staging,
                    candidate_root / entry["sample_id"],
                    entry,
                    model_info,
                    args.seed,
                    verify_hashes=True,
                )
                if not valid:
                    raise RuntimeError("staging validation failed: %s" % errors)
                committed_status = status
                stage = "atomic_commit"
                atomic_commit_sample(staging, final, output_root)
            except Exception as error:
                failures += 1
                if staging.exists():
                    remove_staging(staging, output_root)
                staging = make_staging_directory(output_root, entry["sample_id"])
                write_failed_sample(staging, entry, model_info, args.seed, error, stage)
                atomic_commit_sample(staging, final, output_root)
                committed_status = SCORING_FAILED
                emit(
                    "SAMPLE_FAILED",
                    sample_id=entry["sample_id"],
                    failure_type=type(error).__name__,
                    failure_reason=str(error),
                    failure_stage=stage,
                )
            processed += 1
            if committed_status == "scored_nonempty":
                processed_candidates += int(entry["candidate_count"])
            elif committed_status == "skipped_valid_empty":
                processed_empty += 1
            last_completed = entry["sample_id"]
            if processed % args.log_every == 0 or processed == len(to_process):
                elapsed = max(time.time() - started, 1e-12)
                rate = processed_candidates / elapsed
                remaining = sum(int(item["candidate_count"]) for item, selected_action in actions if selected_action == "process") - processed_candidates
                emit(
                    "PROGRESS",
                    processed_sample_count=processed,
                    processed_candidate_count=processed_candidates,
                    skipped_empty_count=processed_empty,
                    elapsed_seconds=elapsed,
                    throughput_candidates_per_second=rate,
                    estimated_remaining_seconds=None if rate <= 0 else remaining / rate,
                    last_successfully_committed_sample=last_completed,
                    sample_wall_seconds=time.perf_counter() - sample_started,
                )
            if processed % args.batch_size == 0 or processed == len(to_process):
                write_root_state(
                    output_root,
                    entries,
                    candidate_root,
                    model_info,
                    args.seed,
                    started,
                    last_completed_sample_id=last_completed,
                    verify_hashes=False,
                )
    finally:
        if quality_fn is not None:
            quality_fn.gqcnn.close_session()
    progress = write_root_state(
        output_root,
        entries,
        candidate_root,
        model_info,
        args.seed,
        started,
        last_completed_sample_id=last_completed,
        verify_hashes=False,
    )
    thresholds = tuple(args.low_q_threshold or (1e-6, 1e-4, 1e-2))
    statistics = write_run_statistics(
        output_root,
        entries,
        candidate_root,
        model_info,
        args.seed,
        started,
        nearly_identical_atol=args.nearly_identical_atol,
        low_q_thresholds=thresholds,
    )
    terminal_failures = int(progress["failed_samples"])
    final_status = (
        "INCOMPLETE_CONTROLLED_STOP"
        if controlled_stop
        else ("DONE" if terminal_failures == 0 else "PARTIAL")
    )
    emit(
        final_status,
        processed_samples=processed,
        skipped_existing=skipped,
        failures=failures,
        progress=progress,
        statistics={
            "terminal_samples": statistics["terminal_samples"],
            "scored_candidates": statistics["scored_candidates"],
            "finite_q_values": statistics["finite_q_values"],
        },
    )
    if controlled_stop:
        return 3
    return 0 if terminal_failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
