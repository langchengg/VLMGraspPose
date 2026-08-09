#!/usr/bin/env python3
"""Extract resumable GT-free Stage-1 candidate features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.proposal_features import extract_candidate_features  # noqa: E402
from src.segmentation.proposal_types import (  # noqa: E402
    load_candidate_masks_npz,
    reference_candidate_ids_from_provenance,
)
from src.segmentation.query_semantics import parse_query  # noqa: E402
from src.segmentation.camera_intrinsics import resolve_camera_intrinsics  # noqa: E402
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_binary_mask,
    load_compact_manifest,
    load_probability,
    resize_probability,
    sha256_file,
    stable_json_sha256,
)


FEATURE_SCHEMA_VERSION = 2
# The sample-level feature semantics were frozen at this source identity.  The
# code below may evolve in ways that affect only shard consolidation (for
# example, accepting sparse relation-specific columns) without invalidating
# already verified per-sample feature files.
FEATURE_EXTRACTOR_SEMANTIC_SHA256 = (
    "575287c08985eefcf0a0a07bc91bc16f52f17a68bd9bde10fe5fc6ef9f2fb3a4"
)
STRING_COLUMNS = {
    "sample_id",
    "candidate_id",
    "split",
    "scene_id",
    "frame_id",
    "rgb_sha256",
    "query",
    "source_family",
    "source_variant",
    "query_type",
    "target_category",
    "absolute_location_type",
    "relation_type",
    "best_reference_candidate_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--proposal-stage",
        choices=("stage1", "stage2"),
        default="stage1",
        help="Candidate bank to featurize; Stage 2 uses the same GT-free schema.",
    )
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument(
        "--skip-consolidation",
        action="store_true",
        help=(
            "Publish verified per-sample feature artifacts only. This is intended "
            "for disjoint helper manifests; a later unsharded invocation must "
            "build the canonical feature table."
        ),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1",
    )
    return parser.parse_args()


def _depth_metres(path: Path, shape: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.shape != shape:
        raise ValueError(f"depth shape mismatch: {depth.shape} != {shape}")
    depth[~np.isfinite(depth)] = 0.0
    if float(depth.max(initial=0.0)) > 20.0:
        depth /= 1000.0
    return depth


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    for column in STRING_COLUMNS & set(value.columns):
        value[column] = value[column].astype("string")
    return value


def _stream_feature_table(
    paths: list[Path],
    output: Path,
) -> tuple[int, dict[str, str], dict[str, dict[str, float | int | None]]]:
    if not paths:
        raise ValueError("no completed feature rows to combine")
    source_schemas = [pq.read_schema(path) for path in paths]
    # Relation-specific numeric columns are intentionally absent for queries
    # without a parsed relation.  A full split therefore has a sparse union of
    # columns rather than one identical physical Parquet schema per sample.
    unified = pa.unify_schemas(source_schemas, promote_options="permissive")
    arrow_schema = pa.schema(
        [pa.field(field.name, field.type, nullable=True) for field in unified]
    )
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for path in paths:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    arrow_schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            for field in arrow_schema:
                if field.name not in table.column_names:
                    table = table.append_column(
                        field.name,
                        pa.nulls(table.num_rows, type=field.type),
                    )
            table = table.select(arrow_schema.names)
            if table.schema != arrow_schema:
                table = table.cast(arrow_schema, safe=True)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output)

    accumulators: dict[str, dict[str, float | int]] = {}
    parquet = pq.ParquetFile(output)
    for batch in parquet.iter_batches(batch_size=65536):
        frame = batch.to_pandas()
        numeric = frame.select_dtypes(include=[np.number, "bool"])
        for column in numeric:
            values = pd.to_numeric(numeric[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            accumulator = accumulators.setdefault(
                column,
                {"count": 0, "missing": 0, "sum": 0.0, "sum_squares": 0.0},
            )
            accumulator["count"] += int(len(finite))
            accumulator["missing"] += int(len(values) - len(finite))
            accumulator["sum"] += float(finite.sum(dtype=np.float64))
            accumulator["sum_squares"] += float(
                np.square(finite).sum(dtype=np.float64)
            )
    statistics: dict[str, dict[str, float | int | None]] = {}
    for column, accumulator in accumulators.items():
        count = int(accumulator["count"])
        total = float(accumulator["sum"])
        sum_squares = float(accumulator["sum_squares"])
        variance = (
            max(0.0, (sum_squares - total * total / count) / (count - 1))
            if count > 1
            else None
        )
        statistics[column] = {
            "count": count,
            "missing": int(accumulator["missing"]),
            "mean": total / count if count else None,
            "std": float(np.sqrt(variance)) if variance is not None else None,
        }
    columns = {field.name: str(field.type) for field in arrow_schema}
    return rows, columns, statistics


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.expanduser().resolve()
    manifest_path = PROJECT_ROOT / (
        "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/"
        f"{args.split}/manifest.jsonl"
    )
    compact = load_compact_manifest(manifest_path, expected_split=args.split)
    by_id = {row.sample_id: row for row in compact}
    if args.sample_manifest:
        selection = pd.read_parquet(args.sample_manifest.expanduser().resolve())
        sample_ids = list(selection["sample_id"].astype(str))
    else:
        bank_name = "proposals" if args.proposal_stage == "stage1" else "stage2"
        sample_ids = list(by_id)
        missing = [
            sample_id
            for sample_id in sample_ids
            if not (
                experiment_root / bank_name / sample_id / "terminal_status.json"
            ).is_file()
        ]
        if missing and args.sample_limit is None:
            raise RuntimeError(
                f"feature extraction requires the complete {args.proposal_stage} "
                f"{args.split} bank; missing {len(missing)} samples"
            )
        missing_set = set(missing)
        sample_ids = [
            sample_id for sample_id in sample_ids if sample_id not in missing_set
        ]
    if args.sample_limit is not None:
        sample_ids = sample_ids[: int(args.sample_limit)]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.num_shards > 1:
        frame_indices: dict[str, int] = {}
        for sample_id in sample_ids:
            scene_id = by_id[sample_id].scene_id
            frame_indices.setdefault(scene_id, len(frame_indices))
        sample_ids = [
            sample_id
            for sample_id in sample_ids
            if frame_indices[by_id[sample_id].scene_id] % args.num_shards
            == args.shard_index
        ]
        if not sample_ids:
            raise ValueError("selected feature-extraction shard is empty")
    feature_namespace = "features" if args.proposal_stage == "stage1" else "features_stage2"
    sample_root = experiment_root / feature_namespace / f"{args.split}_samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    implementation_contract = stable_json_sha256(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "proposal_stage": args.proposal_stage,
            "implementation_sha256": {
                name: sha256_file(PROJECT_ROOT / "src/segmentation" / name)
                for name in (
                    "camera_intrinsics.py",
                    "proposal_features.py",
                    "depth_mask_features.py",
                    "spatial_relation_features.py",
                    "query_semantics.py",
                )
            },
            "extractor_sha256": FEATURE_EXTRACTOR_SEMANTIC_SHA256,
            "pcd_intrinsics_derivation_sha256": sha256_file(
                PROJECT_ROOT / "tools/export_anygrasp_inputs.py"
            ),
            "intrinsics_policy": "per-sample fitted fx/fy/cx/cy from frozen bundle",
        }
    )
    completed_paths: list[Path] = []
    invariant_cache: dict[str, dict] = {}
    cached_rgb_sha256: str | None = None
    for number, sample_id in enumerate(sample_ids, start=1):
        bank_name = "proposals" if args.proposal_stage == "stage1" else "stage2"
        source = experiment_root / bank_name / sample_id
        destination = sample_root / f"{sample_id}.parquet"
        status_path = sample_root / f"{sample_id}.json"
        row = by_id.get(sample_id)
        if row is None:
            raise ValueError(f"sample is not in {args.split}: {sample_id}")
        rgb_sha256 = str(row.raw["source_rgb_sha256"])
        if rgb_sha256 != cached_rgb_sha256:
            invariant_cache.clear()
            cached_rgb_sha256 = rgb_sha256
        intrinsics, intrinsics_path, _ = resolve_camera_intrinsics(
            row,
            cache_root=experiment_root / "cache/intrinsics",
            image_shape=(480, 640),
        )
        clip_path = (
            experiment_root
            / "cache/clip_embeddings"
            / (
                "dense_candidates"
                if args.proposal_stage == "stage1"
                else "dense_candidates_stage2"
            )
            / f"{sample_id}.npz"
        )
        if (
            not clip_path.is_file()
            and args.sample_manifest is None
            and args.sample_limit is None
        ):
            raise FileNotFoundError(
                f"formal feature extraction requires dense CLIP features: {clip_path}"
            )
        sample_contract = stable_json_sha256(
            {
                "implementation_contract": implementation_contract,
                "proposal_terminal_sha256": sha256_file(source / "terminal_status.json"),
                "rgb_sha256": str(row.raw["source_rgb_sha256"]),
                "depth_sha256": str(row.raw["source_depth_sha256"]),
                "hifi_mask_sha256": row.native_mask_sha256,
                "hifi_probability_sha256": row.probability_sha256,
                "intrinsics_sha256": sha256_file(intrinsics_path),
                "clip_candidate_features_sha256": (
                    sha256_file(clip_path) if clip_path.is_file() else None
                ),
            }
        )
        if args.resume and destination.is_file() and status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            verified = (
                status.get("status") == "COMPLETE"
                and status.get("sample_contract_sha256") == sample_contract
            )
            if args.verify_existing:
                verified = verified and status.get("sha256") == sha256_file(destination)
            if verified:
                completed_paths.append(destination)
                continue
        started = time.perf_counter()
        index = pd.read_parquet(source / "candidate_index.parquet")
        masks = load_candidate_masks_npz(source / "candidate_masks.npz")
        provenance = json.loads(
            (source / "candidate_provenance.json").read_text(encoding="utf-8")
        )
        hifi_mask = load_binary_mask(row.native_mask_path, expected_shape=(480, 640))
        probability = resize_probability(load_probability(row.probability_path), hifi_mask.shape)
        rgb = np.asarray(Image.open(row.rgb_path).convert("RGB"), dtype=np.uint8)
        depth = _depth_metres(row.depth_path, hifi_mask.shape)
        features = extract_candidate_features(
            index,
            masks,
            hifi_mask=hifi_mask,
            hifi_probability=probability,
            rgb=rgb,
            depth_m=depth,
            intrinsics=intrinsics,
            semantics=parse_query(row.query),
            reference_candidate_ids=reference_candidate_ids_from_provenance(
                index, provenance
            ),
            candidate_provenance=provenance,
            invariant_cache=invariant_cache,
        )
        if clip_path.is_file():
            archive = np.load(clip_path, allow_pickle=False)
            try:
                clip_frame = pd.DataFrame(
                    {
                        (
                            key.removeprefix("feature_")
                            if key.startswith("feature_")
                            else key
                        ): np.asarray(archive[key])
                        for key in archive.files
                        if key == "candidate_id" or key.startswith("feature_")
                    }
                )
            finally:
                archive.close()
            replacement = [
                column for column in clip_frame.columns if column != "candidate_id"
            ]
            features = features.drop(columns=replacement, errors="ignore").merge(
                clip_frame, on="candidate_id", how="left", validate="one_to_one"
            )
        features.insert(2, "split", args.split)
        features.insert(3, "scene_id", row.scene_id)
        features.insert(4, "frame_id", row.scene_id)
        features.insert(5, "rgb_sha256", str(row.raw["source_rgb_sha256"]))
        features.insert(6, "query", row.query)
        identity_columns = [
            "sample_id",
            "candidate_id",
            "split",
            "scene_id",
            "frame_id",
            "rgb_sha256",
            "query",
            "source_family",
            "source_variant",
            "eligible_final",
        ]
        features = features[
            identity_columns
            + sorted(set(features.columns) - set(identity_columns))
        ]
        features = _normalize_feature_frame(features)
        temporary = destination.with_name(f".{destination.name}.tmp")
        features.to_parquet(temporary, index=False)
        temporary.replace(destination)
        status = {
            "status": "COMPLETE",
            "sample_id": sample_id,
            "candidate_rows": len(features),
            "runtime_seconds": time.perf_counter() - started,
            "sha256": sha256_file(destination),
            "sample_contract_sha256": sample_contract,
            "implementation_contract_sha256": implementation_contract,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "uses_ground_truth": False,
        }
        _atomic_text(
            status_path,
            json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        completed_paths.append(destination)
        if number % 10 == 0:
            print(f"features: {number}/{len(sample_ids)}", flush=True)

    if args.skip_consolidation:
        print(
            json.dumps(
                {
                    "status": "SAMPLE_ARTIFACTS_COMPLETE",
                    "samples": len(completed_paths),
                    "proposal_stage": args.proposal_stage,
                    "split": args.split,
                    "implementation_contract_sha256": implementation_contract,
                    "consolidated": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    output_stem = (
        f"candidate_features_{'validation' if args.split == 'val' else args.split}"
    )
    output_name = (
        f"{output_stem}.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.parquet"
        if args.num_shards > 1
        else f"{output_stem}.parquet"
    )
    output = (
        experiment_root
        / feature_namespace
        / output_name
    )
    row_count, column_types, statistics = _stream_feature_table(
        completed_paths, output
    )
    schema = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "implementation_contract_sha256": implementation_contract,
        "rows": row_count,
        "samples": len(completed_paths),
        "columns": column_types,
        "ground_truth_columns": [],
        "inference_time_only": True,
        "proposal_stage": args.proposal_stage,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "optional_feature_policy": "NaN plus explicit valid/missing flags",
        "sha256": sha256_file(output),
    }
    schema_text = json.dumps(schema, indent=2, sort_keys=True, allow_nan=False) + "\n"
    statistics_text = (
        json.dumps(statistics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    schema_names = (
        [
            f"feature_schema_{args.split}.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
        ]
        if args.num_shards > 1
        else ["feature_schema.json", f"feature_schema_{args.split}.json"]
    )
    for name in schema_names:
        _atomic_text(experiment_root / feature_namespace / name, schema_text)
    statistics_names = (
        [
            f"feature_statistics_{args.split}.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
        ]
        if args.num_shards > 1
        else ["feature_statistics.json", f"feature_statistics_{args.split}.json"]
    )
    for name in statistics_names:
        _atomic_text(experiment_root / feature_namespace / name, statistics_text)
    leakage = {
        "feature_file_sha256": sha256_file(output),
        "feature_columns_sha256": hashlib.sha256(
            "\n".join(column_types).encode()
        ).hexdigest(),
        "forbidden_columns_present": sorted(
            set(column_types)
            & {"gt_mask", "target_instance_id", "answer_instance_value", "candidate_iou", "y90"}
        ),
        "uses_ground_truth": False,
    }
    if leakage["forbidden_columns_present"]:
        raise RuntimeError("GT leakage column appeared in inference-time features")
    leakage_text = json.dumps(
        leakage, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    leakage_names = (
        [
            f"leakage_audit_{args.split}.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
        ]
        if args.num_shards > 1
        else ["leakage_audit.json", f"leakage_audit_{args.split}.json"]
    )
    for name in leakage_names:
        _atomic_text(experiment_root / feature_namespace / name, leakage_text)
    print(json.dumps(schema, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
