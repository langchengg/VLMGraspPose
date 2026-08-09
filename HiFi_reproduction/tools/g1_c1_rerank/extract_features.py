#!/usr/bin/env python3
"""Shardable feature extraction over frozen candidate pools."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.sample_io import read_deployment_manifest  # noqa: E402
from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json, atomic_parquet  # noqa: E402
from src.grasping.g1_c1_safe_rerank.contracts import sha256_file  # noqa: E402
from src.grasping.g1_c1_safe_rerank.features import extract_feature_table, feature_schema  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--track", choices=("g1", "c1", "union"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _candidate_path(run: Path, track: str, split: str) -> Path:
    if track == "union":
        return run / "data" / f"dedup_union_{split}_candidates.parquet"
    return run / "data" / f"calibrated_{track}_{split}_candidates.parquet"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    run = args.run_dir.expanduser().resolve()
    base = args.base_run.expanduser().resolve()
    source_split = "validation" if args.split == "validation" else args.split
    candidates_path = _candidate_path(run, args.track, args.split)
    candidates = pd.read_parquet(candidates_path)
    deployment = read_deployment_manifest(base / "manifests" / f"{source_split}_samples.parquet")
    output = run / "data" / "feature_shards" / args.track / args.split
    output.mkdir(parents=True, exist_ok=True)
    chunks = (len(deployment) + args.chunk_size - 1) // args.chunk_size
    started = time.perf_counter()
    for chunk in range(chunks):
        start, stop = chunk * args.chunk_size, min(len(deployment), (chunk + 1) * args.chunk_size)
        feature_path = output / f"part-{chunk:05d}.parquet"
        sample_path = output / f"samples-{chunk:05d}.parquet"
        marker_path = output / f"part-{chunk:05d}.json"
        if args.resume and marker_path.is_file():
            marker = json.loads(marker_path.read_text())
            if (
                feature_path.is_file()
                and sample_path.is_file()
                and marker.get("start") == start
                and marker.get("stop") == stop
                and sha256_file(feature_path) == marker.get("feature_sha256")
                and sha256_file(sample_path) == marker.get("sample_sha256")
            ):
                continue
            raise RuntimeError(f"corrupt feature shard {args.track}/{args.split}/{chunk}")
        rows = deployment[start:stop]
        ids = {str(row["sample_id"]) for row in rows}
        local = candidates.loc[candidates["sample_id"].astype(str).isin(ids)].copy()
        features, samples = extract_feature_table(local, rows)
        atomic_parquet(feature_path, features)
        atomic_parquet(sample_path, samples)
        atomic_json(marker_path, {
            "track": args.track, "split": args.split, "chunk": chunk,
            "start": start, "stop": stop, "feature_rows": len(features),
            "feature_sha256": sha256_file(feature_path),
            "sample_sha256": sha256_file(sample_path),
        })
        atomic_json(output / "progress.json", {
            "status": "RUNNING", "completed_samples": stop, "total_samples": len(deployment),
            "elapsed_seconds": time.perf_counter() - started,
        })
        print(json.dumps({"track": args.track, "split": args.split, "completed": stop, "total": len(deployment)}), flush=True)
    feature_parts = [pd.read_parquet(output / f"part-{chunk:05d}.parquet") for chunk in range(chunks)]
    sample_parts = [pd.read_parquet(output / f"samples-{chunk:05d}.parquet") for chunk in range(chunks)]
    features = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    samples = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    final_features = run / "data" / f"feature_table_{args.track}_{args.split}.parquet"
    final_samples = run / "data" / f"sample_table_{args.track}_{args.split}.parquet"
    atomic_parquet(final_features, features)
    atomic_parquet(final_samples, samples)
    schema = feature_schema()
    atomic_json(run / "data/feature_schema.json", schema)
    complete = {
        "status": "COMPLETE", "track": args.track, "split": args.split,
        "sample_count": len(samples), "candidate_rows": len(features),
        "feature_sha256": sha256_file(final_features), "sample_sha256": sha256_file(final_samples),
        "schema_sha256": schema["schema_sha256"],
    }
    atomic_json(output / "COMPLETE.json", complete)
    atomic_json(output / "progress.json", complete)
    print(json.dumps(complete))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
