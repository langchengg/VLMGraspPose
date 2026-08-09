#!/usr/bin/env python3
"""Generate a resumable GT-free SAM 3 Stage-1 proposal bank."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.proposal_deduplication import deduplicate_candidates  # noqa: E402
from src.segmentation.proposal_types import (  # noqa: E402
    ProposalCandidate,
    load_candidate_masks_npz,
)
from src.segmentation.query_semantics import parse_query  # noqa: E402
from src.segmentation.sam3_automatic_proposals import (  # noqa: E402
    OfficialSam3AutomaticProposalGenerator,
)
from src.segmentation.sam3_embedding_cache import Sam3EmbeddingCache  # noqa: E402
from src.segmentation.sam3_proposal_generator import (  # noqa: E402
    build_component_prompt_specs,
    build_hifi_candidates,
    build_text_prompt_specs,
    build_visual_prompt_specs,
    clone_frame_candidates,
    morphological_variants,
    write_proposal_bundle,
)
from src.segmentation.sam3_text_proposals import (  # noqa: E402
    OfficialSam3TextProposalGenerator,
    TextPromptSpec,
)
from src.segmentation.sam3_visual_proposals import (  # noqa: E402
    OfficialSam3VisualProposalGenerator,
)
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_compact_manifest,
    load_probability,
    sha256_file,
    stable_json_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/sam3_proposal_bank_p90_v1/proposal_generation.yaml",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--pilot-manifest", type=Path)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--group-list", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument(
        "--force-recompute-sample",
        action="store_true",
        help="Recompute explicitly selected samples without deleting prior bundles.",
    )
    parser.add_argument("--save-all-candidates", action="store_true")
    parser.add_argument("--max-proposals-per-image", type=int)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument(
        "--bundle-workers",
        type=int,
        help="Bounded workers for per-query deduplication and atomic bundle writes.",
    )
    parser.add_argument(
        "--embedding-cache-max-gib",
        type=float,
        help="Optional LRU bound, intended for the 1,104-frame training split.",
    )
    parser.add_argument(
        "--embedding-cache-root",
        type=Path,
        help="Optional process-local cache root for safe frame-sharded execution.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Verify all selected bundles and rebuild the canonical split manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _depth_metres(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"depth shape {value.shape} != RGB shape {shape}")
    value[~np.isfinite(value)] = 0.0
    if float(value.max(initial=0.0)) > 20.0:
        value /= 1000.0
    return value


def _bundle_complete(directory: Path, verify: bool) -> bool:
    terminal = directory / "terminal_status.json"
    if not terminal.is_file():
        return False
    try:
        payload = json.loads(terminal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "COMPLETE":
        return False
    if verify:
        for name, expected in payload.get("checksums", {}).items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != expected:
                return False
    return True


def _write_frame_cache(
    directory: Path,
    candidates: list[ProposalCandidate],
    metadata: dict[str, Any],
    *,
    mask_shape: tuple[int, int],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    unique = deduplicate_candidates(candidates, iou_threshold=0.995)
    unique.sort(key=lambda item: str(item.candidate_id))
    masks = directory / "candidate_masks.npz"
    index = directory / "candidate_index.parquet"
    masks_tmp = masks.with_name(f".{masks.name}.{os.getpid()}.tmp.npz")
    index_tmp = index.with_name(f".{index.name}.{os.getpid()}.tmp")
    try:
        np.savez_compressed(
            masks_tmp,
            __mask_shape__=np.asarray(mask_shape, dtype=np.int32),
            __storage_schema__=np.asarray([3], dtype=np.int32),
            __candidate_ids__=np.asarray([str(item.candidate_id) for item in unique]),
            __packed_masks__=(
                np.stack(
                    [
                        np.packbits(item.mask.reshape(-1), bitorder="little")
                        for item in unique
                    ]
                )
                if unique
                else np.empty(
                    (0, (int(np.prod(mask_shape)) + 7) // 8), dtype=np.uint8
                )
            ),
        )
        template = ProposalCandidate(
            "__schema__",
            "__schema__",
            "__schema__",
            np.zeros(mask_shape, dtype=bool),
        ).to_index_record()
        pd.DataFrame(
            [item.to_index_record() for item in unique], columns=list(template)
        ).to_parquet(index_tmp, index=False)
        masks_tmp.replace(masks)
        index_tmp.replace(index)
    finally:
        masks_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)
    _atomic_json(directory / "provenance.json", {str(x.candidate_id): x.provenance for x in unique})
    _atomic_json(directory / "metadata.json", metadata)
    _atomic_json(
        directory / "terminal_status.json",
        {
            "status": "COMPLETE",
            "candidate_count": len(unique),
            "checksums": {
                "candidate_masks.npz": sha256_file(masks),
                "candidate_index.parquet": sha256_file(index),
                "provenance.json": sha256_file(directory / "provenance.json"),
                "metadata.json": sha256_file(directory / "metadata.json"),
            },
        },
    )


def _load_frame_cache(directory: Path) -> list[ProposalCandidate]:
    terminal = json.loads((directory / "terminal_status.json").read_text(encoding="utf-8"))
    for name, expected in terminal["checksums"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"frame-cache integrity failure: {directory / name}")
    frame = pd.read_parquet(directory / "candidate_index.parquet")
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    masks = load_candidate_masks_npz(directory / "candidate_masks.npz")
    try:
        candidates = []
        for row in frame.to_dict(orient="records"):
            candidate_id = str(row["candidate_id"])
            candidates.append(
                ProposalCandidate(
                    sample_id=str(row["sample_id"]),
                    candidate_id=candidate_id,
                    source_family=str(row["source_family"]),
                    source_variant=str(row["source_variant"]),
                    mask=masks[candidate_id],
                    sam_score=None if pd.isna(row["sam_score"]) else float(row["sam_score"]),
                    presence_score=(
                        None if pd.isna(row["presence_score"]) else float(row["presence_score"])
                    ),
                    mask_quality_score=(
                        None
                        if pd.isna(row["mask_quality_score"])
                        else float(row["mask_quality_score"])
                    ),
                    canonical_text_prompt=(
                        None
                        if pd.isna(row["canonical_text_prompt"])
                        else str(row["canonical_text_prompt"])
                    ),
                    mask_threshold=(
                        None if pd.isna(row["mask_threshold"]) else float(row["mask_threshold"])
                    ),
                    instance_threshold=(
                        None
                        if pd.isna(row["instance_threshold"])
                        else float(row["instance_threshold"])
                    ),
                    model_revision=(
                        None if pd.isna(row["model_revision"]) else str(row["model_revision"])
                    ),
                    rgb_checksum=(
                        None if pd.isna(row["rgb_checksum"]) else str(row["rgb_checksum"])
                    ),
                    eligible_final=bool(row["eligible_final"]),
                    source_rank=(None if pd.isna(row["source_rank"]) else int(row["source_rank"])),
                    provenance=provenance[candidate_id],
                )
            )
        return candidates
    finally:
        masks.clear()


def _rebuild_split_manifest(
    rows: list[Any], proposal_root: Path, manifest_path: Path
) -> pd.DataFrame:
    # Preserve the end-to-end timing captured by completed sharded runs.  The
    # per-bundle runtime file historically recorded only the inference work
    # before compression/checksumming/visualisation, so silently replacing a
    # shard manifest with that partial number would under-report efficiency.
    recorded_runtime: dict[str, float] = {}
    timing_manifests = sorted(
        manifest_path.parent.glob(f"{manifest_path.stem}*.parquet")
    )
    selected_ids = {str(row.sample_id) for row in rows}
    for timing_path in timing_manifests:
        try:
            timing = pd.read_parquet(
                timing_path, columns=["sample_id", "runtime_seconds"]
            )
        except (OSError, ValueError, KeyError):
            continue
        for item in timing.itertuples(index=False):
            sample_id = str(item.sample_id)
            value = float(item.runtime_seconds)
            if sample_id in selected_ids and np.isfinite(value) and value >= 0.0:
                recorded_runtime[sample_id] = value
    records: list[dict[str, Any]] = []
    for row in rows:
        directory = proposal_root / row.sample_id
        if not _bundle_complete(directory, True):
            raise ValueError(f"proposal bundle is not verified complete: {row.sample_id}")
        terminal = json.loads(
            (directory / "terminal_status.json").read_text(encoding="utf-8")
        )
        runtime = json.loads((directory / "runtime.json").read_text(encoding="utf-8"))
        records.append(
            {
                "sample_id": row.sample_id,
                "scene_id": row.scene_id,
                "rgb_sha256": str(row.raw["source_rgb_sha256"]),
                "candidate_count": int(terminal["candidate_count"]),
                "runtime_seconds": float(
                    recorded_runtime.get(
                        str(row.sample_id),
                        runtime.get(
                            "sample_seconds_total",
                            runtime.get("sample_seconds_before_write", 0.0),
                        ),
                    )
                ),
                "terminal_status": "COMPLETE",
            }
        )
    frame = pd.DataFrame(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return frame


def _select_frame_shard(
    rows_by_frame: dict[str, list[Any]], num_shards: int, shard_index: int
) -> dict[str, list[Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    return {
        frame: values
        for index, (frame, values) in enumerate(rows_by_frame.items())
        if index % num_shards == shard_index
    }


def _rebuild_embedding_cache_manifest(output_root: Path) -> int:
    cache_root = output_root / "cache"
    records: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(cache_root.glob("image_embeddings*/**/*.json")):
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        digest = str(record.get("cache_digest", ""))
        tensor_path = metadata_path.with_suffix(".safetensors")
        if not digest or not tensor_path.is_file():
            continue
        if record.get("tensor_sha256") != sha256_file(tensor_path):
            raise ValueError(f"embedding-cache checksum failure: {tensor_path}")
        record["metadata_path"] = str(metadata_path)
        records.setdefault(digest, record)
    frame = pd.DataFrame(list(records.values()))
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "cache_manifest.parquet"
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_json(
        cache_root / "cache_integrity.json",
        {
            "embedding_records": len(frame),
            "invalid_records": 0,
            "rebuilt_from_process_local_caches": True,
        },
    )
    return len(frame)


def _materialize_text_candidates(
    sample_id: str,
    specification: TextPromptSpec,
    generic_candidates: list[ProposalCandidate],
) -> list[ProposalCandidate]:
    result: list[ProposalCandidate] = []
    for item in generic_candidates:
        if item.canonical_text_prompt != specification.text:
            continue
        suffix = item.source_variant.split("|", 1)[1] if "|" in item.source_variant else item.source_variant
        provenance = [
            {
                **dict(value),
                "sample_prompt_id": specification.prompt_id,
                "sample_source_variant": specification.source_variant,
                "frame_text_cache_reused": True,
            }
            for value in item.provenance
        ]
        result.append(
            ProposalCandidate(
                sample_id=sample_id,
                source_family=specification.source_family,
                source_variant=f"{specification.source_variant}|{suffix}",
                mask=item.mask,
                probability=item.probability,
                sam_score=item.sam_score,
                presence_score=item.presence_score,
                mask_quality_score=item.mask_quality_score,
                box_xyxy=item.box_xyxy,
                canonical_text_prompt=specification.text,
                mask_threshold=item.mask_threshold,
                instance_threshold=item.instance_threshold,
                model_revision=item.model_revision,
                rgb_checksum=item.rgb_checksum,
                eligible_final=specification.eligible_final,
                source_rank=item.source_rank,
                provenance=provenance,
            )
        )
    return result


def _finish_sample_bundle(
    *,
    row: Any,
    scene_id: str,
    output_directory: Path,
    image: Image.Image,
    rgb_sha: str,
    hifi_candidates: list[ProposalCandidate],
    hifi_mask: np.ndarray,
    text_candidates: list[ProposalCandidate],
    automatic: list[ProposalCandidate],
    visual_candidates: list[ProposalCandidate],
    prompt_metadata: dict[str, Any],
    automatic_runtime: dict[str, Any],
    text_runtime: dict[str, Any],
    visual_runtime: dict[str, Any],
    model_load_seconds: float,
    sample_started: float,
    submitted_at: float,
    process: psutil.Process,
    config: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Finish one independent query after all SAM model calls are complete."""

    finish_started = time.perf_counter()
    automatic_candidates = clone_frame_candidates(row.sample_id, automatic)
    initial = deduplicate_candidates(
        hifi_candidates
        + text_candidates
        + automatic_candidates
        + visual_candidates,
        iou_threshold=float(config["deduplication"]["iou_threshold"]),
    )
    depth_m = _depth_metres(row.depth_path, hifi_mask.shape)
    morphology = morphological_variants(
        initial, hifi_mask, depth_m, config["morphology"]
    )
    rss_bytes = int(process.memory_info().rss)
    active_before_write = (
        submitted_at - sample_started + time.perf_counter() - finish_started
    )
    runtime = {
        "sample_seconds_before_write": time.perf_counter() - sample_started,
        "sample_active_seconds_before_write": active_before_write,
        "bundle_queue_seconds": finish_started - submitted_at,
        "bundle_worker_count": int(config["bundle_write_workers"]),
        "model_load_seconds_run_level": model_load_seconds,
        "automatic": automatic_runtime,
        "text": text_runtime,
        "visual": visual_runtime,
        "rss_bytes": rss_bytes,
        "peak_rss_bytes_so_far": rss_bytes,
    }
    unique = write_proposal_bundle(
        output_directory,
        image,
        initial + morphology,
        hifi_mask,
        prompt_metadata,
        runtime,
        deduplication_iou=float(config["deduplication"]["iou_threshold"]),
    )
    return (
        {
            "sample_id": row.sample_id,
            "scene_id": scene_id,
            "rgb_sha256": rgb_sha,
            "candidate_count": len(unique),
            "runtime_seconds": time.perf_counter() - sample_started,
            "terminal_status": "COMPLETE",
        },
        max(rss_bytes, int(process.memory_info().rss)),
    )


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["device"] != "cpu" or config["dtype"] != "float32":
        raise ValueError("proposal generation must use strict CPU/float32")
    np.random.seed(int(args.seed))
    model_path = (PROJECT_ROOT / config["model"]["local_path"]).resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1"
    )
    manifest = PROJECT_ROOT / (
        f"runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/"
        f"{args.split}/manifest.jsonl"
    )
    rows = load_compact_manifest(manifest, expected_split=args.split)
    if args.pilot_manifest:
        pilot = pd.read_parquet(args.pilot_manifest.expanduser().resolve())
        allowed = set(pilot["sample_id"].astype(str))
        rows = [row for row in rows if row.sample_id in allowed]
    if args.sample_id:
        allowed = set(args.sample_id)
        rows = [row for row in rows if row.sample_id in allowed]
    if args.group_list:
        groups = set(args.group_list.read_text(encoding="utf-8").splitlines())
        rows = [row for row in rows if row.scene_id in groups]
    if args.frame_limit is not None:
        retained: set[str] = set()
        for row in rows:
            if row.scene_id not in retained and len(retained) >= int(args.frame_limit):
                continue
            retained.add(row.scene_id)
        rows = [row for row in rows if row.scene_id in retained]
    if args.sample_limit is not None:
        rows = rows[: int(args.sample_limit)]
    if not rows:
        raise ValueError("no proposal-generation rows selected")

    proposal_root = output_root / "proposals"
    canonical_manifest_path = (
        output_root / f"proposal_generation_{args.split}_manifest.parquet"
    )
    if args.rebuild_manifest:
        if args.num_shards != 1 or args.shard_index != 0:
            raise ValueError("manifest rebuilding must run without sharding")
        rebuilt = _rebuild_split_manifest(
            rows, proposal_root, canonical_manifest_path
        )
        embedding_records = _rebuild_embedding_cache_manifest(output_root)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "verified_samples": len(rebuilt),
                    "retained_embedding_records": embedding_records,
                    "manifest": str(canonical_manifest_path),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    rows_by_frame: dict[str, list[Any]] = {}
    for row in rows:
        rows_by_frame.setdefault(row.scene_id, []).append(row)
    rows_by_frame = _select_frame_shard(
        rows_by_frame, args.num_shards, args.shard_index
    )
    if args.num_shards > 1:
        rows = [row for values in rows_by_frame.values() for row in values]
        if not rows:
            raise ValueError("selected proposal-generation shard is empty")
    pending_frames = {
        frame: values
        for frame, values in rows_by_frame.items()
        if args.force_recompute_sample
        or not all(
            _bundle_complete(proposal_root / item.sample_id, args.verify_existing)
            for item in values
        )
    }
    if not pending_frames:
        print("all selected proposal bundles are already verified complete", flush=True)
        return 0

    threads = int(args.num_threads or config["num_threads"])
    bundle_workers = int(
        args.bundle_workers
        if args.bundle_workers is not None
        else config.get("bundle_write_workers", 1)
    )
    if not 1 <= bundle_workers <= 4:
        raise ValueError("bundle-workers must be within [1,4]")
    config["bundle_write_workers"] = bundle_workers
    cache_root = (
        args.embedding_cache_root.expanduser().resolve()
        if args.embedding_cache_root is not None
        else output_root / "cache/image_embeddings"
    )
    cache = Sam3EmbeddingCache(cache_root)
    load_started = time.perf_counter()
    text_generator = OfficialSam3TextProposalGenerator(
        model_path,
        revision=config["model"]["revision"],
        cache=cache,
        processor_sha256=config["model"]["pcs_processor_sha256"],
        num_threads=threads,
        micro_batch_size=int(config["pcs_micro_batch_size"]),
    )
    visual_generator = OfficialSam3VisualProposalGenerator(
        model_path,
        revision=config["model"]["revision"],
        cache=cache,
        processor_sha256=config["model"]["tracker_processor_sha256"],
        num_threads=threads,
    )
    share_tracker = bool(config.get("reuse_tracker_model_for_automatic", False))
    automatic_generator = OfficialSam3AutomaticProposalGenerator(
        model_path,
        revision=config["model"]["revision"],
        num_threads=threads,
        shared_tracker_model=(visual_generator.model if share_tracker else None),
        shared_tracker_processor=(visual_generator.processor if share_tracker else None),
    )
    model_load_seconds = time.perf_counter() - load_started
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    run_rows: list[dict[str, Any]] = []
    bundle_pool = ThreadPoolExecutor(
        max_workers=bundle_workers,
        thread_name_prefix="proposal-bundle",
    )

    for frame_index, (scene_id, frame_rows) in enumerate(pending_frames.items(), start=1):
        frame_started = time.perf_counter()
        pending_bundle_futures: list[Future[tuple[dict[str, Any], int]]] = []
        first = frame_rows[0]
        image = Image.open(first.rgb_path).convert("RGB")
        rgb_sha = str(first.raw["source_rgb_sha256"])
        if any(str(item.raw["source_rgb_sha256"]) != rgb_sha for item in frame_rows):
            raise ValueError("frame group contains conflicting RGB checksums")

        automatic_configuration = {
            **config["automatic"],
            "maximum_proposals_per_image": int(
                args.max_proposals_per_image
                or config["automatic"]["maximum_proposals_per_image"]
            ),
            "model_revision": config["model"]["revision"],
        }
        automatic_key = stable_json_sha256(automatic_configuration)
        automatic_directory = output_root / "cache/automatic_instance_proposals" / rgb_sha / automatic_key
        if _bundle_complete(automatic_directory, True):
            automatic = _load_frame_cache(automatic_directory)
            automatic_metadata = json.loads(
                (automatic_directory / "metadata.json").read_text(encoding="utf-8")
            )
            automatic_runtime = {
                **automatic_metadata.get("runtime", {}),
                "cache_hit": True,
            }
        else:
            automatic, automatic_runtime = automatic_generator.generate(
                f"frame_{rgb_sha[:20]}",
                image,
                rgb_sha,
                points_per_side=int(config["automatic"]["points_per_side"]),
                points_per_batch=int(config["automatic"]["points_per_batch"]),
                score_threshold=float(config["automatic"]["score_threshold"]),
                stability_threshold=float(config["automatic"]["stability_threshold"]),
                nms_threshold=float(config["automatic"]["nms_threshold"]),
                minimum_mask_area=int(config["automatic"]["minimum_mask_area_px"]),
                maximum_proposals=int(automatic_configuration["maximum_proposals_per_image"]),
            )
            automatic_runtime.update(
                {
                    "cache_hit": False,
                    "cache_key": automatic_key,
                    "shared_frame_inference": True,
                    "shared_runtime_seconds": float(
                        automatic_runtime.get("runtime_seconds", 0.0)
                    ),
                }
            )
            _write_frame_cache(
                automatic_directory,
                automatic,
                {"configuration": automatic_configuration, "runtime": automatic_runtime},
                mask_shape=(image.height, image.width),
            )
        automatic_runtime.update(
            {
                "cache_key": automatic_key,
                "shared_frame_inference": True,
                "shared_runtime_seconds": float(
                    automatic_runtime.get(
                        "shared_runtime_seconds",
                        automatic_runtime.get("runtime_seconds", 0.0),
                    )
                ),
            }
        )

        specifications = {
            item.sample_id: build_text_prompt_specs(parse_query(item.query)) for item in frame_rows
        }
        unique_texts = sorted(
            {spec.text for values in specifications.values() for spec in values},
            key=lambda value: (value.lower(), value),
        )
        text_configuration = {
            "texts": unique_texts,
            "instance_thresholds": config["pcs"]["instance_thresholds"],
            "mask_thresholds": config["pcs"]["mask_thresholds"],
            "maximum_instances_per_prompt": config["pcs"]["maximum_instances_per_prompt"],
            "model_revision": config["model"]["revision"],
        }
        text_key = stable_json_sha256(text_configuration)
        text_directory = output_root / "cache/text_proposals" / rgb_sha / text_key
        if _bundle_complete(text_directory, True):
            frame_text = _load_frame_cache(text_directory)
            text_metadata = json.loads(
                (text_directory / "metadata.json").read_text(encoding="utf-8")
            )
            text_runtime = {**text_metadata.get("runtime", {}), "cache_hit": True}
        else:
            generic_specs = [
                TextPromptSpec(
                    prompt_id=f"FRAME_TEXT_{index:04d}",
                    source_family="FRAME_TEXT_CACHE",
                    source_variant="frame_text",
                    text=text,
                )
                for index, text in enumerate(unique_texts)
            ]
            frame_text, text_runtime = text_generator.generate(
                f"frame_{rgb_sha[:20]}",
                image,
                rgb_sha,
                generic_specs,
                instance_thresholds=tuple(float(x) for x in config["pcs"]["instance_thresholds"]),
                mask_thresholds=tuple(float(x) for x in config["pcs"]["mask_thresholds"]),
                maximum_instances_per_prompt=int(config["pcs"]["maximum_instances_per_prompt"]),
            )
            text_runtime.update(
                {
                    "cache_hit": False,
                    "cache_key": text_key,
                    "shared_frame_inference": True,
                    "shared_runtime_seconds": float(
                        text_runtime.get("vision_seconds", 0.0)
                        + text_runtime.get("decoder_seconds", 0.0)
                    ),
                }
            )
            _write_frame_cache(
                text_directory,
                frame_text,
                {"configuration": text_configuration, "runtime": text_runtime},
                mask_shape=(image.height, image.width),
            )
        text_runtime.update(
            {
                "cache_key": text_key,
                "shared_frame_inference": True,
                "shared_runtime_seconds": float(
                    text_runtime.get(
                        "shared_runtime_seconds",
                        text_runtime.get("vision_seconds", 0.0)
                        + text_runtime.get("decoder_seconds", 0.0),
                    )
                ),
            }
        )

        for row in frame_rows:
            output_directory = proposal_root / row.sample_id
            if (
                not args.force_recompute_sample
                and (args.resume or args.skip_existing)
                and _bundle_complete(
                output_directory, args.verify_existing
                )
            ):
                continue
            sample_started = time.perf_counter()
            probability = load_probability(row.probability_path)
            hifi_candidates, hifi_mask, native_probability = build_hifi_candidates(
                row.sample_id,
                probability,
                (image.height, image.width),
                rgb_sha,
                [float(value) for value in config["hifi_probability_thresholds"]],
            )
            semantics = parse_query(row.query)
            text_candidates: list[ProposalCandidate] = []
            for specification in specifications[row.sample_id]:
                text_candidates.extend(
                    _materialize_text_candidates(row.sample_id, specification, frame_text)
                )
            visual_specs, visual_metadata = build_visual_prompt_specs(
                hifi_mask, native_probability, config["tracker"]
            )
            component_specs, component_metadata = build_component_prompt_specs(
                hifi_mask, native_probability, config["components"]
            )
            visual_candidates, visual_runtime = visual_generator.generate(
                row.sample_id,
                image,
                rgb_sha,
                visual_specs + component_specs,
                mask_thresholds=tuple(float(x) for x in config["tracker"]["mask_thresholds"]),
            )
            prompt_metadata = {
                "query": row.query,
                "query_parse": semantics.to_dict(),
                "text_prompts": [item.__dict__ for item in specifications[row.sample_id]],
                "visual": visual_metadata,
                "components": component_metadata,
                "uses_ground_truth": False,
            }
            submitted_at = time.perf_counter()
            pending_bundle_futures.append(
                bundle_pool.submit(
                    _finish_sample_bundle,
                    row=row,
                    scene_id=scene_id,
                    output_directory=output_directory,
                    image=image,
                    rgb_sha=rgb_sha,
                    hifi_candidates=hifi_candidates,
                    hifi_mask=hifi_mask,
                    text_candidates=text_candidates,
                    automatic=automatic,
                    visual_candidates=visual_candidates,
                    prompt_metadata=prompt_metadata,
                    automatic_runtime=dict(automatic_runtime),
                    text_runtime=dict(text_runtime),
                    visual_runtime=dict(visual_runtime),
                    model_load_seconds=model_load_seconds,
                    sample_started=sample_started,
                    submitted_at=submitted_at,
                    process=process,
                    config=config,
                )
            )
            if len(pending_bundle_futures) >= bundle_workers:
                completed_row, observed_rss = pending_bundle_futures.pop(0).result()
                run_rows.append(completed_row)
                peak_rss = max(peak_rss, observed_rss)
        for future in pending_bundle_futures:
            completed_row, observed_rss = future.result()
            run_rows.append(completed_row)
            peak_rss = max(peak_rss, observed_rss)
        print(
            f"proposal frame {frame_index}/{len(pending_frames)}: "
            f"{len(frame_rows)} queries, {time.perf_counter() - frame_started:.2f}s",
            flush=True,
        )
        gc.collect()
        cache_limit_gib = (
            args.embedding_cache_max_gib
            if args.embedding_cache_max_gib is not None
            else config.get("embedding_cache_max_gib")
        )
        if cache_limit_gib is not None:
            cache.prune_to_max_bytes(
                int(float(cache_limit_gib) * 1024**3),
                protected_rgb_sha256=rgb_sha,
            )

    bundle_pool.shutdown(wait=True)

    manifest_frame = pd.DataFrame(run_rows)
    manifest_path = (
        output_root
        / (
            f"proposal_generation_{args.split}_manifest."
            f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.parquet"
        )
        if args.num_shards > 1
        else canonical_manifest_path
    )
    if manifest_path.is_file() and (args.resume or args.skip_existing):
        manifest_frame = pd.concat([pd.read_parquet(manifest_path), manifest_frame], ignore_index=True)
        manifest_frame = manifest_frame.drop_duplicates("sample_id", keep="last")
    manifest_frame.to_parquet(manifest_path, index=False)
    cache_records = cache.manifest_records()
    cache_suffix = (
        f".shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
        if args.num_shards > 1
        else ""
    )
    (output_root / "cache").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cache_records).to_parquet(
        output_root / f"cache/cache_manifest{cache_suffix}.parquet", index=False
    )
    _atomic_json(
        output_root / f"cache/cache_integrity{cache_suffix}.json",
        {
            "embedding_records": len(cache_records),
            "invalid_records": sum(item.get("status") == "INVALID" for item in cache_records),
            "model_revision": config["model"]["revision"],
            "peak_rss_bytes": peak_rss,
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "processed_samples": len(run_rows),
                "selected_samples": len(rows),
                "selected_frames": len(rows_by_frame),
                "peak_rss_bytes": peak_rss,
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
