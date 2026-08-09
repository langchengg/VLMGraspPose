"""Generate fair G1/C1 development candidates with the byte-locked decoder.

The locked fair implementation is imported only after its SHA-256 is verified.
This wrapper changes orchestration (split selection and atomic chunking), not model,
conditioning, decoder, score, candidate ID, or geometry semantics.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HIFI_ROOT = ROOT / "HiFi_reproduction"
for item in (SRC, HIFI_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.feature_cache import common_asset_records
from unified_reranking.ledger import ledger_stage


LOCKED_INFERENCE = ROOT / "experiments/fair_crog_hifics_g1_c1_no_rerank/native_inference.py"
LOCKED_INFERENCE_SHA256 = "5bbae71830db8264494b20651ff6c62fd87210011614b159c48f11bfd4586544"


def _load_locked_symbols() -> tuple[Any, ...]:
    if sha256_file(LOCKED_INFERENCE) != LOCKED_INFERENCE_SHA256:
        raise ValueError("fair native-inference source hash mismatch")
    from experiments.fair_crog_hifics_g1_c1_no_rerank.native_inference import (
        DECODER,
        infer_one,
        load_config,
    )
    from src.grasping.backends import BackendSample
    from src.grasping.backends.training import load_finetuned_model
    from src.grasping.common.sample_io import CompactSampleLoader, read_deployment_manifest

    return DECODER, infer_one, load_config, BackendSample, load_finetuned_model, CompactSampleLoader, read_deployment_manifest


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    os.replace(temporary, path)


def _chunk_manifest_valid(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if observed.get("status") != "COMPLETE" or not all(
        observed.get(key) == value for key, value in expected.items()
    ):
        return False
    artifacts = observed.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "per_sample",
        "candidates",
    }:
        return False
    expected_rows = {
        "per_sample": int(expected["stop"]) - int(expected["start"]),
        "candidates": int(observed.get("candidate_rows", -1)),
    }
    for name, filename in (
        ("per_sample", "per_sample.parquet"),
        ("candidates", "candidates.parquet"),
    ):
        record = artifacts.get(name)
        artifact_path = (path.parent / filename).resolve()
        if (
            not isinstance(record, dict)
            or Path(str(record.get("path", ""))).resolve() != artifact_path
            or not artifact_path.is_file()
            or artifact_path.is_symlink()
            or record.get("sha256") != sha256_file(artifact_path)
            or record.get("bytes") != artifact_path.stat().st_size
        ):
            return False
        try:
            rows = pq.read_metadata(artifact_path).num_rows
        except (OSError, pa.ArrowException):
            return False
        if record.get("rows") != rows or rows != expected_rows[name]:
            return False
    return observed.get("sample_rows") == expected_rows["per_sample"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--method", choices=("g1", "c1"), required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--save-raw-maps", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="formal")
    parser.add_argument(
        "--attest-existing-shards",
        action="store_true",
        help=(
            "Upgrade legacy pre-lock shard manifests after exact row/key/lineage "
            "verification; never runs model inference or changes Parquet outputs."
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if args.tag == "formal" and args.limit is not None:
        raise ValueError("a limited run must use a non-formal tag")
    attest_existing = bool(getattr(args, "attest_existing_shards", False))
    DECODER, infer_one, load_config, BackendSample, load_finetuned_model, CompactSampleLoader, read_deployment_manifest = _load_locked_symbols()
    source = args.source_run.resolve()
    run_dir = args.run_dir.resolve()
    source_split = "validation" if args.split == "validation" else "train"
    samples_file = source / "manifests" / f"{source_split}_samples.parquet"
    deployment = read_deployment_manifest(samples_file)
    if args.limit is not None:
        deployment = deployment[: args.limit]
    output_name = f"{args.method}_{args.split}" if args.tag == "formal" else f"{args.method}_{args.split}_{args.tag}"
    output = run_dir / "02_candidates" / "native_work" / output_name
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    selected_path = source / "selected_configs" / f"{args.method.upper()}.json"
    config, selected = load_config(args.method, selected_path, False)
    checkpoint = Path(selected["finetuned_checkpoint"]).resolve()
    expected_checkpoint_sha = str(selected["finetuned_checkpoint_sha256"])
    if sha256_file(checkpoint) != expected_checkpoint_sha:
        raise ValueError("fine-tuned checkpoint hash mismatch")
    source_hash = sha256_file(samples_file)
    selected_hash = sha256_file(selected_path)
    decoder_hash = canonical_sha256(DECODER)
    # These are the semantic inference dependencies.  The wrapper itself only
    # orchestrates chunks and writes provenance, so provenance-only changes do
    # not falsely invalidate otherwise identical model outputs.
    code_paths = {LOCKED_INFERENCE.resolve()}
    for symbol in (
        infer_one,
        load_config,
        BackendSample,
        load_finetuned_model,
        CompactSampleLoader,
        read_deployment_manifest,
    ):
        source_file = inspect.getsourcefile(symbol)
        if source_file is None:
            raise RuntimeError(f"cannot resolve inference dependency source: {symbol}")
        code_paths.add(Path(source_file).resolve())
    code_records = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(code_paths, key=str)
    ]
    code_bundle_sha = canonical_sha256(code_records)
    asset_cache: dict[str, str] = {}

    if attest_existing:
        if (run_dir / "08_lock" / "FORMAL_TEST_LOCK.json").exists():
            raise PermissionError("legacy candidate shards cannot be attested after formal lock")
        previous_run_path = output / "run_manifest.json"
        if not previous_run_path.is_file():
            raise FileNotFoundError("legacy candidate run manifest is missing")
        previous_run = json.loads(previous_run_path.read_text(encoding="utf-8"))
        expected_run = {
            "status": "COMPLETE",
            "method": args.method.upper(),
            "split": args.split,
            "tag": args.tag,
            "device": args.device,
            "sample_count": len(deployment),
            "source_samples_sha256": source_hash,
            "checkpoint_sha256": expected_checkpoint_sha,
            "selected_config_sha256": selected_hash,
            "native_decoder_config_sha256": decoder_hash,
            "locked_inference_source_sha256": LOCKED_INFERENCE_SHA256,
            "save_raw_maps": bool(args.save_raw_maps),
        }
        mismatch = {
            key: (previous_run.get(key), value)
            for key, value in expected_run.items()
            if previous_run.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"legacy candidate run identity mismatch: {mismatch}")

    pending: list[tuple[int, int, list[dict[str, Any]], dict[str, Any], Path]] = []
    for start in range(0, len(deployment), args.chunk_size):
        stop = min(start + args.chunk_size, len(deployment))
        chunk_rows = deployment[start:stop]
        identity = canonical_sha256([str(row["sample_id"]) for row in chunk_rows])
        asset_records = common_asset_records(chunk_rows, asset_cache)
        chunk_dir = shards / f"{start:08d}_{stop:08d}"
        expected = {
            "start": start,
            "stop": stop,
            "sample_identity_sha256": identity,
            "source_samples_sha256": source_hash,
            "checkpoint_sha256": expected_checkpoint_sha,
            "selected_config_sha256": selected_hash,
            "native_decoder_config_sha256": decoder_hash,
            "method": args.method.upper(),
            "split": args.split,
            "tag": args.tag,
            "device": args.device,
            "locked_inference_source_sha256": LOCKED_INFERENCE_SHA256,
            "inference_code_bundle_sha256": code_bundle_sha,
            "input_asset_identity_sha256": canonical_sha256(asset_records),
            "save_raw_maps": bool(args.save_raw_maps),
        }
        manifest = chunk_dir / "manifest.json"
        if attest_existing and not _chunk_manifest_valid(manifest, expected):
            per_sample_path = chunk_dir / "per_sample.parquet"
            candidate_path = chunk_dir / "candidates.parquet"
            if not per_sample_path.is_file() or not candidate_path.is_file() or not manifest.is_file():
                raise RuntimeError(f"legacy candidate shard is incomplete: {chunk_dir}")
            legacy = json.loads(manifest.read_text(encoding="utf-8"))
            stable_expected = {
                key: value
                for key, value in expected.items()
                if key
                in {
                    "start",
                    "stop",
                    "sample_identity_sha256",
                    "source_samples_sha256",
                    "checkpoint_sha256",
                    "selected_config_sha256",
                    "native_decoder_config_sha256",
                    "save_raw_maps",
                }
            }
            if legacy.get("status") != "COMPLETE" or any(
                legacy.get(key) != value for key, value in stable_expected.items()
            ):
                raise RuntimeError(f"legacy candidate shard identity mismatch: {chunk_dir}")
            samples = pq.read_table(per_sample_path).to_pandas()
            candidates = pq.read_table(candidate_path).to_pandas()
            expected_ids = [str(row["sample_id"]) for row in chunk_rows]
            if (
                samples["sample_id"].astype(str).tolist() != expected_ids
                or samples["sample_id"].astype(str).duplicated().any()
                or not set(candidates["sample_id"].astype(str)).issubset(expected_ids)
                or candidates[["sample_id", "candidate_id"]].astype(str).duplicated().any()
                or int(samples["candidate_count"].sum()) != len(candidates)
            ):
                raise RuntimeError(f"legacy candidate shard row/key audit failed: {chunk_dir}")
            row_lineage = {
                "checkpoint_sha256": expected_checkpoint_sha,
                "selected_config_sha256": selected_hash,
                "native_decoder_config_sha256": decoder_hash,
            }
            for name, value in row_lineage.items():
                if set(candidates[name].astype(str)) - {value} or set(samples[name].astype(str)) - {value}:
                    raise RuntimeError(f"legacy candidate shard lineage mismatch: {chunk_dir}/{name}")
            atomic_json(
                manifest,
                {
                    **legacy,
                    **expected,
                    "sample_rows": len(samples),
                    "candidate_rows": len(candidates),
                    "artifacts": {
                        "per_sample": {
                            "path": str(per_sample_path.resolve()),
                            "sha256": sha256_file(per_sample_path),
                            "bytes": per_sample_path.stat().st_size,
                            "rows": len(samples),
                        },
                        "candidates": {
                            "path": str(candidate_path.resolve()),
                            "sha256": sha256_file(candidate_path),
                            "bytes": candidate_path.stat().st_size,
                            "rows": len(candidates),
                        },
                    },
                    "provenance_upgrade": {
                        "kind": "PRELOCK_EXACT_EXISTING_ARTIFACT_ATTESTATION",
                        "model_inference_rerun": False,
                        "parquet_outputs_changed": False,
                    },
                },
            )
        if _chunk_manifest_valid(manifest, expected):
            continue
        pending.append((start, stop, chunk_rows, expected, chunk_dir))

    model = None
    if pending:
        if attest_existing:
            raise RuntimeError("legacy attestation left an unverifiable shard")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        device = torch.device(args.device)
        model, observed_sha, _ = load_finetuned_model(
            checkpoint,
            backend="grconvnet" if args.method == "g1" else "ggcnn2",
            device=args.device,
        )
        if observed_sha != expected_checkpoint_sha:
            raise ValueError("loaded fine-tuned checkpoint hash mismatch")
        loader = CompactSampleLoader()
        for chunk_number, (start, stop, chunk_rows, expected, chunk_dir) in enumerate(pending, 1):
            chunk_dir.mkdir(parents=True, exist_ok=True)
            sample_results: list[dict[str, Any]] = []
            candidates: list[dict[str, Any]] = []
            started = time.perf_counter()
            for row in chunk_rows:
                arrays = loader.load(row, mask_source="predicted", labels=None)
                sample = BackendSample(
                    sample_id=str(row["sample_id"]),
                    rgb=arrays.rgb,
                    depth_m=arrays.depth_m,
                    predicted_mask=arrays.binary_mask,
                    probability_map=arrays.probability,
                    mask_source="predicted",
                    metadata={"scene_id": arrays.scene_id},
                )
                raw_path = None
                if args.save_raw_maps:
                    raw_path = output / "raw_maps" / sample.sample_id[:3] / f"{sample.sample_id}.npz"
                sample_result, sample_candidates = infer_one(
                    method=args.method,
                    model=model,
                    device=device,
                    config=config,
                    sample=sample,
                    raw_path=raw_path,
                )
                lineage = {
                    "checkpoint_sha256": expected_checkpoint_sha,
                    "selected_config_sha256": selected_hash,
                    "native_decoder_config_sha256": decoder_hash,
                }
                sample_result.update(lineage)
                for candidate in sample_candidates:
                    candidate.update(lineage)
                sample_results.append(sample_result)
                candidates.extend(sample_candidates)
            _atomic_parquet(chunk_dir / "per_sample.parquet", sample_results)
            _atomic_parquet(chunk_dir / "candidates.parquet", candidates)
            per_sample_path = chunk_dir / "per_sample.parquet"
            candidate_path = chunk_dir / "candidates.parquet"
            atomic_json(
                chunk_dir / "manifest.json",
                {
                    "status": "COMPLETE",
                    **expected,
                    "sample_rows": len(sample_results),
                    "candidate_rows": len(candidates),
                    "technical_failures": sum(row["status"] == "technical_failure" for row in sample_results),
                    "elapsed_seconds": time.perf_counter() - started,
                    "artifacts": {
                        "per_sample": {
                            "path": str(per_sample_path.resolve()),
                            "sha256": sha256_file(per_sample_path),
                            "bytes": per_sample_path.stat().st_size,
                            "rows": len(sample_results),
                        },
                        "candidates": {
                            "path": str(candidate_path.resolve()),
                            "sha256": sha256_file(candidate_path),
                            "bytes": candidate_path.stat().st_size,
                            "rows": len(candidates),
                        },
                    },
                },
            )
            print(
                f"[{output_name}] chunk {chunk_number}/{len(pending)} rows {start}:{stop} candidates={len(candidates)}",
                flush=True,
            )

    sample_tables: list[pa.Table] = []
    candidate_tables: list[pa.Table] = []
    manifests: list[dict[str, Any]] = []
    for start in range(0, len(deployment), args.chunk_size):
        stop = min(start + args.chunk_size, len(deployment))
        chunk_dir = shards / f"{start:08d}_{stop:08d}"
        chunk_rows = deployment[start:stop]
        asset_records = common_asset_records(chunk_rows, asset_cache)
        expected = {
            "start": start,
            "stop": stop,
            "sample_identity_sha256": canonical_sha256(
                [str(row["sample_id"]) for row in chunk_rows]
            ),
            "source_samples_sha256": source_hash,
            "checkpoint_sha256": expected_checkpoint_sha,
            "selected_config_sha256": selected_hash,
            "native_decoder_config_sha256": decoder_hash,
            "method": args.method.upper(),
            "split": args.split,
            "tag": args.tag,
            "device": args.device,
            "locked_inference_source_sha256": LOCKED_INFERENCE_SHA256,
            "inference_code_bundle_sha256": code_bundle_sha,
            "input_asset_identity_sha256": canonical_sha256(asset_records),
            "save_raw_maps": bool(args.save_raw_maps),
        }
        chunk_manifest_path = chunk_dir / "manifest.json"
        if not _chunk_manifest_valid(chunk_manifest_path, expected):
            raise RuntimeError(f"incomplete or drifted inference shard: {chunk_dir}")
        manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
        manifests.append(
            {
                "path": str(chunk_manifest_path.resolve()),
                "sha256": sha256_file(chunk_manifest_path),
                "start": start,
                "stop": stop,
                "sample_identity_sha256": expected["sample_identity_sha256"],
                "artifacts": manifest["artifacts"],
            }
        )
        sample_tables.append(pq.read_table(chunk_dir / "per_sample.parquet"))
        candidate_shard = pq.read_table(chunk_dir / "candidates.parquet")
        if candidate_shard.num_rows:
            candidate_tables.append(candidate_shard)
    sample_table = pa.concat_tables(sample_tables, promote_options="default")
    candidate_table = pa.concat_tables(candidate_tables, promote_options="default") if candidate_tables else pa.table({})
    expected_ids = [str(row["sample_id"]) for row in deployment]
    observed_ids = sample_table.column("sample_id").to_pylist()
    if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("merged inference output does not preserve exact sample identity/order")
    _atomic_parquet(output / "per_sample.parquet", sample_table.to_pylist())
    _atomic_parquet(output / "candidates.parquet", candidate_table.to_pylist())
    merged_sample_path = output / "per_sample.parquet"
    merged_candidate_path = output / "candidates.parquet"
    result = {
        "status": "COMPLETE",
        "method": args.method.upper(),
        "split": args.split,
        "tag": args.tag,
        "device": args.device,
        "sample_count": len(observed_ids),
        "candidate_count": candidate_table.num_rows,
        "no_output_count": sum(row["status"] == "no_output" for row in sample_table.to_pylist()),
        "technical_failure_count": sum(row["status"] == "technical_failure" for row in sample_table.to_pylist()),
        "source_samples": str(samples_file),
        "source_samples_sha256": source_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha,
        "selected_config": str(selected_path),
        "selected_config_sha256": selected_hash,
        "native_decoder": DECODER,
        "native_decoder_config_sha256": decoder_hash,
        "locked_inference_source": str(LOCKED_INFERENCE),
        "locked_inference_source_sha256": LOCKED_INFERENCE_SHA256,
        "inference_code_records": code_records,
        "inference_code_bundle_sha256": code_bundle_sha,
        "save_raw_maps": bool(args.save_raw_maps),
        "shards": len(manifests),
        "chunk_manifests": manifests,
        "artifacts": {
            "per_sample": {
                "path": str(merged_sample_path.resolve()),
                "sha256": sha256_file(merged_sample_path),
                "bytes": merged_sample_path.stat().st_size,
                "rows": sample_table.num_rows,
            },
            "candidates": {
                "path": str(merged_candidate_path.resolve()),
                "sha256": sha256_file(merged_candidate_path),
                "bytes": merged_candidate_path.stat().st_size,
                "rows": candidate_table.num_rows,
            },
        },
    }
    atomic_json(output / "run_manifest.json", result)
    return result


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    substage = f"fair_candidate_inference_{args.method}_{args.split}_{args.tag}"
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P1",
        substage=substage,
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(args)
        artifact = run_dir / "02_candidates" / "native_work" / (
            f"{args.method}_{args.split}" if args.tag == "formal" else f"{args.method}_{args.split}_{args.tag}"
        ) / "run_manifest.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
