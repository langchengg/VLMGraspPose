"""Chunked, resumable extraction of matched HiFi/RGB-D scalar evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HIFI_ROOT = ROOT / "HiFi_reproduction"
for item in (SRC, HIFI_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.feature_extractors.common import extract_common_evidence
from unified_reranking.feature_cache import (
    common_asset_records,
    valid_common_shard_manifest,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.telemetry import (
    flatten_telemetry,
    missing_feature_rate,
    telemetry_payload,
)
from unified_reranking.test_access_guard import append_access_log

from src.grasping.backends.conditioning import resize_probability_to_native
from src.grasping.common.sample_io import CompactSampleLoader


IDENTITY_COLUMNS = {"sample_id", "candidate_id", "route"}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("crog", "g1", "c1"))
    parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    parser.add_argument("--pool", choices=("top5", "all"), default="top5")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="formal")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if args.tag == "formal" and args.limit is not None:
        raise ValueError("limited extraction requires a non-formal tag")
    run_dir = args.run_dir.resolve()
    candidates_path = (
        run_dir / "02_candidates" / f"{args.route}_{args.split}_{args.pool}.parquet"
    )
    manifest_path = run_dir / "01_manifests" / f"paired_{args.split}.parquet"
    candidates = pd.read_parquet(candidates_path)
    deployment = pq.read_table(manifest_path).to_pylist()
    if args.limit is not None:
        deployment = deployment[: args.limit]
        allowed = {str(row["sample_id"]) for row in deployment}
        candidates = candidates.loc[candidates["sample_id"].isin(allowed)].copy()
    groups = {
        str(sample_id): part.copy()
        for sample_id, part in candidates.groupby("sample_id", sort=False)
    }
    pool_suffix = "" if args.pool == "top5" else f"_{args.pool}"
    output_name = (
        f"{args.route}_{args.split}{pool_suffix}"
        if args.tag == "formal"
        else f"{args.route}_{args.split}{pool_suffix}_{args.tag}"
    )
    output = run_dir / "03_features" / "common" / output_name
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    candidates_sha = sha256_file(candidates_path)
    manifest_sha = sha256_file(manifest_path)
    extractor_sha = sha256_file(
        ROOT / "src" / "unified_reranking" / "feature_extractors" / "common.py"
    )
    tool_sha = sha256_file(Path(__file__))
    asset_hash_cache: dict[str, str] = {}
    all_asset_records: list[dict[str, Any]] = []
    loader = CompactSampleLoader()

    for start in range(0, len(deployment), args.chunk_size):
        stop = min(start + args.chunk_size, len(deployment))
        chunk = deployment[start:stop]
        chunk_dir = shards / f"{start:08d}_{stop:08d}"
        assets = common_asset_records(chunk, asset_hash_cache)
        all_asset_records.extend(assets)
        expected = {
            "start": start,
            "stop": stop,
            "sample_identity_sha256": canonical_sha256(
                [str(row["sample_id"]) for row in chunk]
            ),
            "candidate_manifest_sha256": candidates_sha,
            "paired_manifest_sha256": manifest_sha,
            "feature_extractor_sha256": extractor_sha,
            "tool_sha256": tool_sha,
            "asset_identity_sha256": canonical_sha256(assets),
        }
        if args.pool != "top5":
            expected["pool"] = args.pool
        marker = chunk_dir / "manifest.json"
        if valid_common_shard_manifest(marker, expected):
            continue
        feature_parts: list[pd.DataFrame] = []
        relation_parts: list[pd.DataFrame] = []
        context_rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for row in chunk:
            sample_id = str(row["sample_id"])
            group = groups.get(sample_id)
            context_rows.append(
                {
                    "sample_id": sample_id,
                    "frame_id": str(row["frame_id"]),
                    "scene_id": str(row["scene_id"]),
                    "split": args.split,
                    "language": str(row["language"]),
                    "source_rgb_path": str(row["source_rgb_path"]),
                    "source_rgb_sha256": str(row["source_rgb_sha256"]),
                    "source_depth_path": str(row["source_depth_path"]),
                    "source_depth_sha256": str(row["source_depth_sha256"]),
                    "predicted_mask_path": str(row["predicted_mask_path"]),
                    "predicted_mask_sha256": str(row["predicted_mask_sha256"]),
                    "predicted_probability_path": str(
                        row["predicted_probability_path"]
                    ),
                    "predicted_probability_sha256": str(
                        row["predicted_probability_sha256"]
                    ),
                    "candidate_count": 0 if group is None else len(group),
                }
            )
            if group is None or group.empty:
                continue
            arrays = loader.load(
                row, mask_source="predicted", labels=None, load_intrinsics=False
            )
            probability = resize_probability_to_native(
                arrays.probability, arrays.depth_m.shape
            )
            feature, relation = extract_common_evidence(
                group,
                probability=probability,
                binary_mask=arrays.binary_mask,
                depth_m=arrays.depth_m,
            )
            feature_parts.append(feature)
            relation_parts.append(relation)
        feature_frame = (
            pd.concat(feature_parts, ignore_index=True)
            if feature_parts
            else pd.DataFrame()
        )
        relation_frame = (
            pd.concat(relation_parts, ignore_index=True)
            if relation_parts
            else pd.DataFrame()
        )
        context_frame = pd.DataFrame(context_rows)
        _atomic_parquet(chunk_dir / "candidate_features.parquet", feature_frame)
        _atomic_parquet(chunk_dir / "candidate_relations.parquet", relation_frame)
        _atomic_parquet(chunk_dir / "sample_context.parquet", context_frame)
        artifact_records: dict[str, Any] = {}
        for kind, frame, key_columns in (
            ("candidate_features", feature_frame, ["sample_id", "candidate_id"]),
            (
                "candidate_relations",
                relation_frame,
                ["sample_id", "source_candidate_id", "target_candidate_id"],
            ),
            ("sample_context", context_frame, ["sample_id"]),
        ):
            artifact_path = chunk_dir / f"{kind}.parquet"
            keys = (
                frame[key_columns]
                .astype(str)
                .sort_values(key_columns)
                .to_dict("records")
                if len(frame)
                else []
            )
            artifact_records[kind] = {
                "sha256": sha256_file(artifact_path),
                "rows": len(frame),
                "key_sha256": canonical_sha256(keys),
            }
        atomic_json(
            marker,
            {
                "status": "COMPLETE",
                **expected,
                "candidate_feature_rows": len(feature_frame),
                "candidate_relation_rows": len(relation_frame),
                "sample_context_rows": len(context_frame),
                "elapsed_seconds": time.perf_counter() - started,
                "artifacts": artifact_records,
            },
        )
        print(
            f"[{output_name}] rows {start}:{stop} features={len(feature_frame)}",
            flush=True,
        )

    kinds = ("candidate_features", "candidate_relations", "sample_context")
    merged: dict[str, pd.DataFrame] = {}
    for kind in kinds:
        parts = []
        for start in range(0, len(deployment), args.chunk_size):
            stop = min(start + args.chunk_size, len(deployment))
            path = shards / f"{start:08d}_{stop:08d}" / f"{kind}.parquet"
            part = pd.read_parquet(path)
            if len(part):
                parts.append(part)
        merged[kind] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        _atomic_parquet(output / f"{kind}.parquet", merged[kind])
    feature_keys = set(
        map(
            tuple,
            merged["candidate_features"][["sample_id", "candidate_id"]].to_numpy(),
        )
    )
    candidate_keys = set(
        map(tuple, candidates[["sample_id", "candidate_id"]].to_numpy())
    )
    if feature_keys != candidate_keys:
        raise RuntimeError("feature rows do not match exact candidate membership")
    if len(merged["sample_context"]) != len(deployment):
        raise RuntimeError("sample context does not preserve full denominator")
    numeric = [
        column
        for column in merged["candidate_features"].columns
        if column not in IDENTITY_COLUMNS
        and pd.api.types.is_numeric_dtype(merged["candidate_features"][column])
    ]
    assert_model_feature_columns(numeric)
    if len(merged["candidate_features"]) <= 0:
        raise RuntimeError("common feature extraction produced no candidate rows")
    persisted_extraction_seconds = 0.0
    for start in range(0, len(deployment), args.chunk_size):
        stop = min(start + args.chunk_size, len(deployment))
        shard_manifest = json.loads(
            (shards / f"{start:08d}_{stop:08d}" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        elapsed = float(shard_manifest.get("elapsed_seconds", -1.0))
        if not np.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("common feature shard has invalid elapsed telemetry")
        persisted_extraction_seconds += elapsed
    telemetry = telemetry_payload(
        phase="common_feature_extraction",
        parameter_count=None,
        ranker_latency_ms=None,
        feature_latency_ms=persisted_extraction_seconds
        * 1000.0
        / len(merged["candidate_features"]),
        missing_feature_rate_value=missing_feature_rate(
            merged["candidate_features"], numeric
        ),
        not_applicable_fields=("parameter_count", "ranker_latency_ms"),
    )
    result = {
        "status": "COMPLETE",
        "route": args.route,
        "split": args.split,
        "pool": args.pool,
        "tag": args.tag,
        "samples": len(deployment),
        "candidate_features": len(merged["candidate_features"]),
        "candidate_relations": len(merged["candidate_relations"]),
        "model_feature_columns": numeric,
        "model_feature_schema_sha256": canonical_sha256(numeric),
        "candidate_manifest": str(candidates_path),
        "candidate_manifest_sha256": candidates_sha,
        "paired_manifest": str(manifest_path),
        "paired_manifest_sha256": manifest_sha,
        "feature_extractor_sha256": extractor_sha,
        "tool_sha256": tool_sha,
        "asset_identity_sha256": canonical_sha256(all_asset_records),
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "feature_extraction_elapsed_seconds": persisted_extraction_seconds,
        "feature_extraction_latency_ms": telemetry["feature_latency_ms"],
        "feature_extraction_latency_protocol": (
            "sum of persisted shard extractor wall times divided by candidate rows"
        ),
        "artifacts": {
            kind: {
                "path": str((output / f"{kind}.parquet").resolve()),
                "sha256": sha256_file(output / f"{kind}.parquet"),
            }
            for kind in kinds
        },
    }
    atomic_json(output / "feature_manifest.json", result)
    if args.split == "test" and not getattr(args, "suppress_access_log", False):
        append_access_log(
            run_dir,
            {
                "event": "prelock_label_free_test_stage",
                "stage": f"common_features_{args.route}_test_{args.pool}_{args.tag}",
                "inputs": [
                    "candidate_geometry",
                    "RGB",
                    "depth",
                    "predicted_mask",
                    "predicted_probability",
                    "language",
                ],
                "output_manifest": str((output / "feature_manifest.json").resolve()),
                "output_manifest_sha256": sha256_file(output / "feature_manifest.json"),
                "candidate_labels_opened_as_table": False,
            },
        )
    return result


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    substage = f"common_features_{args.route}_{args.split}_{args.pool}_{args.tag}"
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P3_P4",
        substage=substage,
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(args)
        pool_suffix = "" if args.pool == "top5" else f"_{args.pool}"
        output_name = (
            f"{args.route}_{args.split}{pool_suffix}"
            if args.tag == "formal"
            else f"{args.route}_{args.split}{pool_suffix}_{args.tag}"
        )
        artifact = (
            run_dir / "03_features" / "common" / output_name / "feature_manifest.json"
        )
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
