"""Extract resumable local RGB features with identical schema for all routes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HIFI = ROOT / "HiFi_reproduction"
for path in (SRC, HIFI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.feature_extractors.rgb import candidate_rgb_features
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log
from unified_reranking.telemetry import (
    peak_memory_mb,
    per_candidate_extraction_latency_ms,
)
from src.grasping.common.sample_io import CompactSampleLoader


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("crog", "g1", "c1"))
    parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    parser.add_argument("--chunk-size", type=int, default=250)
    return parser.parse_args()


def _rgb_asset_records(
    rows: list[dict[str, Any]], cache: dict[str, str]
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in rows:
        path = str(Path(str(row["source_rgb_path"])).resolve())
        observed = cache.get(path)
        if observed is None:
            observed = sha256_file(path)
            cache[path] = observed
        recorded = str(row["source_rgb_sha256"])
        if observed != recorded:
            raise RuntimeError(f"paired manifest RGB hash drift: {row['sample_id']}")
        result.append(
            {"sample_id": str(row["sample_id"]), "path": path, "sha256": observed}
        )
    return result


def _completed_shard_is_valid(
    marker: Path,
    shard_path: Path,
    expected: dict[str, Any],
) -> bool:
    if not marker.is_file() or not shard_path.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("status") != "COMPLETE" or any(
        value.get(k) != v for k, v in expected.items()
    ):
        return False
    if sha256_file(shard_path) != value.get("artifact_sha256"):
        return False
    elapsed = value.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed < 0
    ):
        return False
    frame = pd.read_parquet(shard_path, columns=["sample_id", "candidate_id"])
    keys = (
        frame[["sample_id", "candidate_id"]]
        .astype(str)
        .sort_values(["sample_id", "candidate_id"])
        .to_dict("records")
    )
    return canonical_sha256(keys) == value.get("candidate_keys_sha256")


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.resolve()
    candidate_path = (
        run_dir / "02_candidates" / f"{args.route}_{args.split}_top5.parquet"
    )
    manifest_path = run_dir / "01_manifests" / f"paired_{args.split}.parquet"
    candidates = pd.read_parquet(candidate_path)
    deployment = pd.read_parquet(manifest_path).to_dict(orient="records")
    groups = {
        str(key): value for key, value in candidates.groupby("sample_id", sort=False)
    }
    output = run_dir / "03_features" / "rgb" / f"{args.route}_{args.split}"
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    loader = CompactSampleLoader()
    geometry_extractor_sha = sha256_file(
        ROOT / "src" / "unified_reranking" / "feature_extractors" / "common.py"
    )
    rgb_extractor_sha = sha256_file(
        ROOT / "src" / "unified_reranking" / "feature_extractors" / "rgb.py"
    )
    tool_sha = sha256_file(Path(__file__))
    candidate_manifest_sha = sha256_file(candidate_path)
    paired_manifest_sha = sha256_file(manifest_path)
    rgb_hash_cache: dict[str, str] = {}
    all_rgb_assets: list[dict[str, str]] = []
    expected_shards: list[Path] = []
    for start in range(0, len(deployment), args.chunk_size):
        stop = min(start + args.chunk_size, len(deployment))
        rows = deployment[start:stop]
        marker = shards / f"{start:08d}_{stop:08d}.json"
        shard_path = marker.with_suffix(".parquet")
        expected_shards.append(shard_path)
        rgb_assets = _rgb_asset_records(rows, rgb_hash_cache)
        all_rgb_assets.extend(rgb_assets)
        candidate_keys = []
        for row in rows:
            group = groups.get(str(row["sample_id"]))
            if group is not None:
                candidate_keys.extend(
                    group[["sample_id", "candidate_id"]].astype(str).to_dict("records")
                )
        candidate_keys = sorted(
            candidate_keys, key=lambda item: (item["sample_id"], item["candidate_id"])
        )
        expected = {
            "start": start,
            "stop": stop,
            "sample_identity_sha256": canonical_sha256(
                [str(row["sample_id"]) for row in rows]
            ),
            "geometry_extractor_sha256": geometry_extractor_sha,
            "rgb_extractor_sha256": rgb_extractor_sha,
            "tool_sha256": tool_sha,
            "candidate_manifest_sha256": candidate_manifest_sha,
            "paired_manifest_sha256": paired_manifest_sha,
            "rgb_assets_identity_sha256": canonical_sha256(rgb_assets),
            "candidate_keys_sha256": canonical_sha256(candidate_keys),
        }
        if _completed_shard_is_valid(marker, shard_path, expected):
            continue
        extraction_started = time.perf_counter()
        pieces = []
        for row in rows:
            group = groups.get(str(row["sample_id"]))
            if group is None or group.empty:
                continue
            arrays = loader.load(
                row, mask_source="predicted", labels=None, load_intrinsics=False
            )
            pieces.append(candidate_rgb_features(group, arrays.rgb))
        shard = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        _atomic_parquet(shard_path, shard)
        atomic_json(
            marker,
            {
                "status": "COMPLETE",
                **expected,
                "candidate_rows": len(shard),
                "elapsed_seconds": time.perf_counter() - extraction_started,
                "artifact_sha256": sha256_file(shard_path),
            },
        )
        print(
            f"[{args.route}_{args.split}] RGB samples {start}:{stop} features={len(shard)}",
            flush=True,
        )
    parts = [pd.read_parquet(path) for path in expected_shards]
    features = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    expected_keys = set(
        map(tuple, candidates[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    actual_keys = set(
        map(tuple, features[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    if expected_keys != actual_keys:
        raise RuntimeError("RGB features do not preserve frozen candidate membership")
    feature_columns = tuple(
        column for column in features if column not in {"sample_id", "candidate_id"}
    )
    assert_model_feature_columns(feature_columns)
    artifact = output / "candidate_features.parquet"
    _atomic_parquet(artifact, features)
    extraction_seconds = sum(
        float(json.loads(marker.read_text(encoding="utf-8"))["elapsed_seconds"])
        for marker in (path.with_suffix(".json") for path in expected_shards)
    )
    extraction_latency = per_candidate_extraction_latency_ms(
        extraction_seconds, len(features)
    )
    result: dict[str, object] = {
        "status": "COMPLETE",
        "route": args.route,
        "split": args.split,
        "candidate_rows": len(features),
        "model_feature_columns": list(feature_columns),
        "model_feature_schema_sha256": canonical_sha256(feature_columns),
        "candidate_manifest_sha256": candidate_manifest_sha,
        "paired_manifest_sha256": paired_manifest_sha,
        "rgb_assets_identity_sha256": canonical_sha256(all_rgb_assets),
        "geometry_extractor_sha256": geometry_extractor_sha,
        "rgb_extractor_sha256": rgb_extractor_sha,
        "tool_sha256": tool_sha,
        "feature_extraction_elapsed_seconds": extraction_seconds,
        "feature_extraction_latency_ms": extraction_latency,
        "feature_extraction_peak_memory_mb": peak_memory_mb(),
        "feature_extraction_latency_protocol": (
            "sum of persisted shard extractor wall times divided by candidate rows"
        ),
        "artifact": {"path": str(artifact.resolve()), "sha256": sha256_file(artifact)},
    }
    atomic_json(output / "feature_manifest.json", result)
    if args.split == "test":
        append_access_log(
            run_dir,
            {
                "event": "prelock_label_free_test_stage",
                "stage": f"rgb_features_{args.route}_test",
                "inputs": ["candidate_geometry", "RGB"],
                "output_manifest": str((output / "feature_manifest.json").resolve()),
                "output_manifest_sha256": sha256_file(output / "feature_manifest.json"),
                "candidate_labels_opened_as_table": False,
            },
        )
    return result


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P3_P4",
        substage=f"rgb_features_{args.route}_{args.split}",
        route=args.route,
        evidence_track="T2_matched_common",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(args)
        artifact = (
            run_dir
            / "03_features"
            / "rgb"
            / f"{args.route}_{args.split}"
            / "feature_manifest.json"
        )
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
