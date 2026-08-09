#!/usr/bin/env python3
"""Generate and freeze development candidates with the already locked backend.

This is read-only model inference.  It never trains an upstream model and it
stores inference candidates separately from offline labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.backends import BackendSample  # noqa: E402
from src.grasping.common.results import evaluate_prediction_records  # noqa: E402
from src.grasping.common.sample_io import (  # noqa: E402
    CompactSampleLoader,
    aligned_labels,
    read_deployment_manifest,
    read_label_manifest,
)
from src.grasping.g1_c1_safe_rerank.contracts import (  # noqa: E402
    canonical_sha256,
    identity_table_sha256,
    sha256_file,
)
from src.grasping.g1_c1_safe_rerank.pools import (  # noqa: E402
    adapt_source_candidates,
    adapt_source_labels,
)
from tools.grasp4dof.run_method import _build_backend, _load_json  # noqa: E402


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("G1", "C1"), required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_prefix = (PROJECT_ROOT / ".venv-grasp4dof").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(f"candidate generation requires {expected_prefix}")
    if args.chunk_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("chunk size/limit must be positive")
    base = args.base_run.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    debug = args.limit is not None
    namespace = (
        Path("debug") / f"limit_{int(args.limit)}" / args.backend.lower()
        if debug
        else Path("development_backend") / args.backend.lower()
    )
    output = run / "data" / namespace
    output.mkdir(parents=True, exist_ok=True)
    final_candidates = (
        output / "debug_candidates.parquet"
        if debug
        else run / "data" / f"frozen_{args.backend.lower()}_train_candidates.parquet"
    )
    final_labels = (
        output / "debug_labels.parquet"
        if debug
        else run / "data" / f"{args.backend.lower()}_train_candidate_labels.parquet"
    )
    final_manifest = output / "COMPLETE.json"
    config_path = base / "selected_configs" / f"{args.backend}.json"
    samples_path = base / "manifests/train_samples.parquet"
    labels_path = base / "manifests/train_labels.parquet"
    config = _load_json(config_path)
    checkpoint_path = Path(str(config["finetuned_checkpoint"])).expanduser().resolve()
    if final_manifest.is_file():
        manifest = json.loads(final_manifest.read_text())
        if (
            manifest.get("status") == ("DEBUG_COMPLETE" if debug else "COMPLETE")
            and final_candidates.is_file()
            and final_labels.is_file()
            and sha256_file(final_candidates) == manifest.get("candidate_sha256")
            and sha256_file(final_labels) == manifest.get("label_sha256")
            and sha256_file(samples_path) == manifest.get("source_samples_sha256")
            and sha256_file(labels_path) == manifest.get("source_labels_sha256")
            and sha256_file(config_path) == manifest.get("config_sha256")
            and sha256_file(checkpoint_path) == manifest.get("checkpoint_sha256")
            and int(manifest.get("sample_count", -1))
            == (int(args.limit) if debug else 26_295)
        ):
            print(json.dumps({"status": "CACHE_HIT", "backend": args.backend}))
            return 0
        raise RuntimeError("completed candidate artifact is inconsistent")
    backend, backend_manifest = _build_backend(
        method_id=args.backend, config=config, oracle=False
    )
    deployment = read_deployment_manifest(samples_path)
    labels = list(aligned_labels(deployment, read_label_manifest(labels_path)))
    if args.limit is not None:
        deployment, labels = deployment[: args.limit], labels[: args.limit]
    loader = CompactSampleLoader()
    chunk_count = (len(deployment) + args.chunk_size - 1) // args.chunk_size
    started = time.perf_counter()
    for chunk in range(chunk_count):
        start = chunk * args.chunk_size
        stop = min(len(deployment), start + args.chunk_size)
        candidate_path = output / "candidate_shards" / f"part-{chunk:05d}.parquet"
        label_path = output / "label_shards" / f"part-{chunk:05d}.parquet"
        marker_path = output / "markers" / f"part-{chunk:05d}.json"
        if args.resume and marker_path.is_file():
            marker = json.loads(marker_path.read_text())
            if (
                candidate_path.is_file()
                and label_path.is_file()
                and sha256_file(candidate_path) == marker.get("candidate_sha256")
                and sha256_file(label_path) == marker.get("label_sha256")
                and marker.get("start") == start
                and marker.get("stop") == stop
                and marker.get("sample_ids_sha256") == canonical_sha256(
                    [str(row["sample_id"]) for row in deployment[start:stop]]
                )
            ):
                continue
            raise RuntimeError(f"stale or corrupt completed shard {chunk}")
        source_candidates: list[dict[str, Any]] = []
        for sample, label in zip(deployment[start:stop], labels[start:stop], strict=True):
            arrays = loader.load(sample, mask_source="predicted", load_intrinsics=False)
            prediction = backend.predict(
                BackendSample(
                    sample_id=arrays.sample_id,
                    rgb=arrays.rgb,
                    depth_m=arrays.depth_m,
                    predicted_mask=arrays.binary_mask,
                    probability_map=arrays.probability,
                    mask_source="predicted",
                    metadata={"scene_id": arrays.scene_id},
                )
            )
            _, candidates = evaluate_prediction_records(
                method=str(backend_manifest["method"]), prediction=prediction, label=label
            )
            source_candidates.extend(candidates)
        if source_candidates:
            raw = pd.DataFrame(source_candidates)
            frozen = adapt_source_candidates(raw, backend=args.backend, split="train")
            labels_frame = adapt_source_labels(raw, backend=args.backend)
        else:
            frozen = pd.DataFrame(
                columns=[
                    "sample_id", "scene_id", "split", "backend", "source_candidate_id",
                    "stable_candidate_id", "original_rank", "original_score", "center_x",
                    "center_y", "angle_deg", "width_px", "height_px", "raw_network_quality",
                    "stored_center_mask_support", "stored_jaw_mask_support", "source_row",
                    "source_column", "candidate_identity_sha256",
                ]
            )
            labels_frame = pd.DataFrame(
                columns=[
                    "sample_id", "candidate_correct", "best_rectangle_iou",
                    "best_angle_difference_deg", "stable_candidate_id",
                ]
            )
        _atomic_parquet(candidate_path, frozen)
        _atomic_parquet(label_path, labels_frame)
        _atomic_json(
            marker_path,
            {
                "chunk": chunk,
                "start": start,
                "stop": stop,
                "sample_ids_sha256": canonical_sha256(
                    [str(row["sample_id"]) for row in deployment[start:stop]]
                ),
                "candidate_rows": len(frozen),
                "candidate_sha256": sha256_file(candidate_path),
                "label_sha256": sha256_file(label_path),
            },
        )
        _atomic_json(
            output / "progress.json",
            {
                "status": "RUNNING",
                "backend": args.backend,
                "completed_samples": stop,
                "total_samples": len(deployment),
                "completed_chunks": chunk + 1,
                "total_chunks": chunk_count,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps({"backend": args.backend, "completed": stop, "total": len(deployment)}), flush=True)
    candidate_parts = [pd.read_parquet(output / "candidate_shards" / f"part-{chunk:05d}.parquet") for chunk in range(chunk_count)]
    label_parts = [pd.read_parquet(output / "label_shards" / f"part-{chunk:05d}.parquet") for chunk in range(chunk_count)]
    candidates = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    candidate_labels = pd.concat(label_parts, ignore_index=True) if label_parts else pd.DataFrame()
    _atomic_parquet(final_candidates, candidates)
    _atomic_parquet(final_labels, candidate_labels)
    top5_candidates = candidates.loc[
        candidates["original_rank"].astype(int) <= 5
    ].copy()
    top5_keys = top5_candidates[["sample_id", "stable_candidate_id"]].copy()
    top5_labels = candidate_labels.merge(
        top5_keys,
        on=["sample_id", "stable_candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    top5_candidate_path = (
        output / "debug_top5_candidates.parquet"
        if debug
        else run / "data" / f"frozen_{args.backend.lower()}_train_top5_candidates.parquet"
    )
    top5_label_path = (
        output / "debug_top5_labels.parquet"
        if debug
        else run / "data" / f"{args.backend.lower()}_train_top5_candidate_labels.parquet"
    )
    _atomic_parquet(top5_candidate_path, top5_candidates)
    _atomic_parquet(top5_label_path, top5_labels)
    _atomic_json(
        final_manifest,
        {
            "status": "DEBUG_COMPLETE" if debug else "COMPLETE",
            "debug_limit": args.limit,
            "backend": args.backend,
            "sample_count": len(deployment),
            "candidate_rows": len(candidates),
            "candidate_identity_sha256": identity_table_sha256(candidates),
            "candidate_path": str(final_candidates),
            "candidate_sha256": sha256_file(final_candidates),
            "top5_candidate_path": str(top5_candidate_path),
            "top5_candidate_rows": len(top5_candidates),
            "top5_candidate_sha256": sha256_file(top5_candidate_path),
            "label_path": str(final_labels),
            "label_sha256": sha256_file(final_labels),
            "top5_label_path": str(top5_label_path),
            "top5_label_sha256": sha256_file(top5_label_path),
            "source_samples_sha256": sha256_file(samples_path),
            "source_labels_sha256": sha256_file(labels_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_sha256": backend_manifest["finetuned_checkpoint_sha256"],
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _atomic_json(output / "progress.json", {"status": "COMPLETE", "completed_samples": len(deployment), "total_samples": len(deployment)})
    print(json.dumps({"status": "COMPLETE", "backend": args.backend, "candidate_rows": len(candidates)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
