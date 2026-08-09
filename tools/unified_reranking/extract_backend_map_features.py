"""Extract resumable G1/C1 candidate-aligned dense-map evidence."""

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
HIFI_ROOT = ROOT / "HiFi_reproduction"
for item in (SRC, HIFI_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.feature_extractors.backend_maps import (
    load_candidate_backend_map_features,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log
from unified_reranking.telemetry import (
    peak_memory_mb,
    per_candidate_extraction_latency_ms,
)

from src.grasping.common.geometry import CropTransform


ID_COLUMNS = ("sample_id", "candidate_id")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fair-test-source", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("g1", "c1"))
    parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    parser.add_argument("--chunk-size", type=int, default=250)
    return parser.parse_args()


def _source_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.split == "test":
        work = (
            args.fair_test_source.resolve()
            / "02_predictions"
            / "native_work"
            / args.route
        )
    else:
        work = (
            args.run_dir.resolve()
            / "02_candidates"
            / "native_work"
            / f"{args.route}_{args.split}"
        )
    return work / "candidates.parquet", work / "per_sample.parquet"


def _crop_transform(group: pd.DataFrame) -> CropTransform:
    values = group["transform_json"].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise RuntimeError(
            "backend sample does not have one exact persisted CropTransform"
        )
    payload = json.loads(values[0])
    required = {
        "crop_x",
        "crop_y",
        "crop_width",
        "crop_height",
        "model_width",
        "model_height",
        "native_width",
        "native_height",
    }
    if set(payload) != required:
        raise RuntimeError("persisted backend CropTransform schema changed")
    return CropTransform(**payload)


def _raw_map_records(
    sample_ids: list[str], paths: dict[str, Any]
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for sample_id in sample_ids:
        raw = paths.get(sample_id)
        if raw is None or pd.isna(raw):
            raise RuntimeError(
                f"missing raw backend map for candidate-bearing sample {sample_id}"
            )
        path = Path(str(raw)).resolve()
        if not path.is_file():
            raise RuntimeError(
                f"missing raw backend map for candidate-bearing sample {sample_id}"
            )
        records.append(
            {"sample_id": sample_id, "path": str(path), "sha256": sha256_file(path)}
        )
    return records


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
    frame = pd.read_parquet(shard_path, columns=list(ID_COLUMNS))
    keys = (
        frame[list(ID_COLUMNS)]
        .astype(str)
        .sort_values(list(ID_COLUMNS))
        .to_dict("records")
    )
    return canonical_sha256(keys) == value.get("candidate_keys_sha256")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    run_dir = args.run_dir.resolve()
    frozen_path = run_dir / "02_candidates" / f"{args.route}_{args.split}_top5.parquet"
    source_candidates_path, per_sample_path = _source_paths(args)
    frozen = pd.read_parquet(frozen_path)
    source = pd.read_parquet(source_candidates_path)
    per_sample = pd.read_parquet(per_sample_path)
    source = source.rename(columns={"jaw_width_px": "width_px"})
    source_columns = [*ID_COLUMNS, "source_row", "source_column", "transform_json"]
    source_join = source[source_columns]
    work = frozen.merge(
        source_join, on=list(ID_COLUMNS), how="left", validate="one_to_one"
    )
    if work[["source_row", "source_column"]].isna().any().any():
        raise RuntimeError("source peak coordinates do not cover frozen candidates")
    paths = per_sample.set_index("sample_id")["raw_maps_path"].to_dict()
    output = run_dir / "03_features" / "backend_maps" / f"{args.route}_{args.split}"
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    sample_ids = work["sample_id"].drop_duplicates().astype(str).tolist()
    candidate_manifest_sha = sha256_file(frozen_path)
    source_candidates_sha = sha256_file(source_candidates_path)
    per_sample_sha = sha256_file(per_sample_path)
    extractor_sha = sha256_file(
        ROOT / "src" / "unified_reranking" / "feature_extractors" / "backend_maps.py"
    )
    tool_sha = sha256_file(Path(__file__))
    raw_map_records: list[dict[str, str]] = []
    expected_shards: list[Path] = []
    for start in range(0, len(sample_ids), args.chunk_size):
        stop = min(start + args.chunk_size, len(sample_ids))
        selected = sample_ids[start:stop]
        shard_path = shards / f"{start:08d}_{stop:08d}.parquet"
        expected_shards.append(shard_path)
        marker = shard_path.with_suffix(".json")
        selected_raw_maps = _raw_map_records(selected, paths)
        raw_map_records.extend(selected_raw_maps)
        selected_work = work.loc[work["sample_id"].astype(str).isin(selected)]
        selected_keys = (
            selected_work[list(ID_COLUMNS)]
            .astype(str)
            .sort_values(list(ID_COLUMNS))
            .to_dict("records")
        )
        expected = {
            "start": start,
            "stop": stop,
            "sample_identity_sha256": canonical_sha256(selected),
            "candidate_keys_sha256": canonical_sha256(selected_keys),
            "candidate_manifest_sha256": candidate_manifest_sha,
            "source_candidates_sha256": source_candidates_sha,
            "per_sample_sha256": per_sample_sha,
            "raw_maps_identity_sha256": canonical_sha256(selected_raw_maps),
            "feature_extractor_sha256": extractor_sha,
            "tool_sha256": tool_sha,
        }
        if _completed_shard_is_valid(marker, shard_path, expected):
            continue
        extraction_started = time.perf_counter()
        pieces = []
        for sample_id in selected:
            archive = paths.get(sample_id)
            group = work.loc[work["sample_id"].astype(str) == sample_id]
            pieces.append(
                load_candidate_backend_map_features(
                    group,
                    str(archive),
                    transform=_crop_transform(group),
                )
            )
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
            f"[{args.route}_{args.split}] samples {start}:{stop} features={len(shard)}",
            flush=True,
        )
    parts = [pd.read_parquet(path) for path in expected_shards]
    features = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    expected_keys = set(map(tuple, frozen[list(ID_COLUMNS)].astype(str).to_numpy()))
    actual_keys = set(map(tuple, features[list(ID_COLUMNS)].astype(str).to_numpy()))
    if expected_keys != actual_keys:
        raise RuntimeError("dense features do not preserve frozen candidate membership")
    feature_columns = tuple(
        column for column in features.columns if column not in ID_COLUMNS
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
        "source_candidates_sha256": source_candidates_sha,
        "per_sample_sha256": per_sample_sha,
        "raw_maps_identity_sha256": canonical_sha256(raw_map_records),
        "feature_extractor_sha256": extractor_sha,
        "tool_sha256": tool_sha,
        "feature_extraction_elapsed_seconds": extraction_seconds,
        "feature_extraction_latency_ms": extraction_latency,
        "feature_extraction_peak_memory_mb": peak_memory_mb(),
        "feature_extraction_latency_protocol": (
            "sum of persisted shard extractor wall times divided by candidate rows"
        ),
        "artifact": {"path": str(artifact.resolve()), "sha256": sha256_file(artifact)},
        "new_peak_search": False,
        "candidate_geometry_changed": False,
    }
    atomic_json(output / "feature_manifest.json", result)
    if args.split == "test":
        append_access_log(
            run_dir,
            {
                "event": "prelock_label_free_test_stage",
                "stage": f"backend_maps_{args.route}_test",
                "inputs": ["candidate_geometry", f"{args.route.upper()} dense maps"],
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
        substage=f"backend_maps_{args.route}_{args.split}",
        route=args.route,
        evidence_track="T1_native",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(args)
        artifact = (
            run_dir
            / "03_features"
            / "backend_maps"
            / f"{args.route}_{args.split}"
            / "feature_manifest.json"
        )
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
