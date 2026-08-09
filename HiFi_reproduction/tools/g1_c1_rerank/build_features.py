#!/usr/bin/env python3
"""Build resumable, label-free scalar features for a frozen G1/C1 pool.

This module deliberately imports only the deployment manifest reader.  It can
therefore be used for the sealed test split without making a label loader
reachable from the feature-preparation process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.sample_io import read_deployment_manifest  # noqa: E402
from src.grasping.g1_c1_safe_rerank.contracts import sha256_file  # noqa: E402
from src.grasping.g1_c1_safe_rerank.contracts import canonical_sha256  # noqa: E402
from src.grasping.g1_c1_safe_rerank.features import (  # noqa: E402
    extract_feature_table,
    feature_schema,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("G1", "C1"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    if args.chunk_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("chunk-size and limit must be positive")
    base = args.base_run.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    backend = args.backend.lower()
    split = str(args.split)
    candidate_path = run / "data" / f"frozen_{backend}_{split}_candidates.parquet"
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    sample_path = base / "manifests" / f"{split}_samples.parquet"
    deployment = list(read_deployment_manifest(sample_path))
    if args.limit is not None:
        deployment = deployment[: args.limit]
    sample_ids = {str(row["sample_id"]) for row in deployment}
    candidates = pd.read_parquet(candidate_path)
    candidate_sample_ids = set(candidates["sample_id"].astype(str))
    manifest_sample_ids = {str(row["sample_id"]) for row in deployment}
    unexpected = candidate_sample_ids - manifest_sample_ids
    if unexpected:
        raise AssertionError(f"candidate pool contains {len(unexpected)} samples outside deployment manifest")
    candidates = candidates.loc[candidates["sample_id"].astype(str).isin(sample_ids)].copy()
    current_candidate_sha256 = sha256_file(candidate_path)
    current_sample_sha256 = sha256_file(sample_path)

    output = run / "02_features" / split / backend
    output.mkdir(parents=True, exist_ok=True)
    final_features = output / "allnms_features.parquet"
    final_samples = output / "sample_context.parquet"
    final_manifest = output / "COMPLETE.json"
    if final_manifest.is_file():
        manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
        valid = (
            manifest.get("status") == "COMPLETE"
            and final_features.is_file()
            and final_samples.is_file()
            and sha256_file(final_features) == manifest.get("features_sha256")
            and sha256_file(final_samples) == manifest.get("samples_sha256")
            and int(manifest.get("sample_count", -1)) == len(deployment)
            and manifest.get("source_candidate_sha256") == current_candidate_sha256
            and manifest.get("source_sample_manifest_sha256") == current_sample_sha256
        )
        if valid:
            print(json.dumps({"status": "CACHE_HIT", "backend": args.backend, "split": split}))
            return 0
        raise RuntimeError("completed feature artifact is inconsistent")

    chunks = (len(deployment) + args.chunk_size - 1) // args.chunk_size
    started = time.perf_counter()
    for chunk in range(chunks):
        start = chunk * args.chunk_size
        stop = min(len(deployment), start + args.chunk_size)
        feature_part = output / "feature_shards" / f"part-{chunk:05d}.parquet"
        sample_part = output / "sample_shards" / f"part-{chunk:05d}.parquet"
        marker = output / "markers" / f"part-{chunk:05d}.json"
        if args.resume and marker.is_file():
            record = json.loads(marker.read_text(encoding="utf-8"))
            local_ids = {str(row["sample_id"]) for row in deployment[start:stop]}
            local_candidates = candidates.loc[
                candidates["sample_id"].astype(str).isin(local_ids)
            ]
            if (
                feature_part.is_file()
                and sample_part.is_file()
                and sha256_file(feature_part) == record.get("features_sha256")
                and sha256_file(sample_part) == record.get("samples_sha256")
                and record.get("start") == start
                and record.get("stop") == stop
                and record.get("sample_ids_sha256")
                == canonical_sha256(sorted(local_ids))
                and record.get("candidate_identities_sha256")
                == canonical_sha256(
                    sorted(local_candidates["candidate_identity_sha256"].astype(str))
                )
            ):
                continue
            raise RuntimeError(f"stale or corrupt feature shard {chunk}")
        rows = deployment[start:stop]
        local_ids = {str(row["sample_id"]) for row in rows}
        local_candidates = candidates.loc[
            candidates["sample_id"].astype(str).isin(local_ids)
        ].copy()
        features, sample_context = extract_feature_table(local_candidates, rows)
        _atomic_parquet(feature_part, features)
        _atomic_parquet(sample_part, sample_context)
        _atomic_json(
            marker,
            {
                "chunk": chunk,
                "start": start,
                "stop": stop,
                "candidate_rows": len(features),
                "sample_ids_sha256": canonical_sha256(sorted(local_ids)),
                "candidate_identities_sha256": canonical_sha256(
                    sorted(local_candidates["candidate_identity_sha256"].astype(str))
                ),
                "features_sha256": sha256_file(feature_part),
                "samples_sha256": sha256_file(sample_part),
            },
        )
        _atomic_json(
            output / "progress.json",
            {
                "status": "RUNNING",
                "backend": args.backend,
                "split": split,
                "completed_samples": stop,
                "total_samples": len(deployment),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps({"backend": args.backend, "split": split, "completed": stop, "total": len(deployment)}), flush=True)

    feature_parts = [pd.read_parquet(output / "feature_shards" / f"part-{chunk:05d}.parquet") for chunk in range(chunks)]
    sample_parts = [pd.read_parquet(output / "sample_shards" / f"part-{chunk:05d}.parquet") for chunk in range(chunks)]
    features = pd.concat(feature_parts, ignore_index=True) if feature_parts else candidates.iloc[:0].copy()
    sample_context = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    if len(features) != len(candidates):
        raise AssertionError(f"feature coverage drift: {len(features)} != {len(candidates)}")
    if len(sample_context) != len(deployment):
        raise AssertionError("sample context does not cover the complete denominator")
    _atomic_parquet(final_features, features)
    _atomic_parquet(final_samples, sample_context)
    top5 = features.loc[features["original_rank"].astype(int) <= 5].copy()
    top5_path = output / "top5_features.parquet"
    _atomic_parquet(top5_path, top5)
    _atomic_json(output / "feature_schema.json", feature_schema())
    _atomic_json(
        final_manifest,
        {
            "status": "COMPLETE",
            "backend": args.backend,
            "split": split,
            "sample_count": len(deployment),
            "candidate_rows": len(features),
            "top5_candidate_rows": len(top5),
            "source_candidate_sha256": current_candidate_sha256,
            "source_sample_manifest_sha256": current_sample_sha256,
            "features_sha256": sha256_file(final_features),
            "top5_features_sha256": sha256_file(top5_path),
            "samples_sha256": sha256_file(final_samples),
            "label_loader_imported": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _atomic_json(output / "progress.json", {"status": "COMPLETE", "completed_samples": len(deployment), "total_samples": len(deployment)})
    print(json.dumps({"status": "COMPLETE", "backend": args.backend, "split": split, "candidate_rows": len(features)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
