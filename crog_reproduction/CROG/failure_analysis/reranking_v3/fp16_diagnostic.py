from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .feature_data import load_fullchain_arrays
from .feature_store import FeatureCatalog, prior_array_to_lookup
from .fullchain_extractor import FLOAT16_STORAGE_FIELDS, extract_fullchain_features
from .oof import predict_ranker_streaming
from .precision_efficiency import (
    compare_feature_quantized_predictions,
    compare_precision_arrays,
    directory_statistics,
    write_precision_efficiency_report,
)
from .schema import (
    artifact_identity,
    atomic_write_json,
    canonical_json,
    read_jsonl,
    sha256_bytes,
    stable_sample_id,
)
from .v2_prior import load_v2_oof_prior


PERSISTED_FLOAT32_FIELDS = ("head_features", "depth_features")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _label_like_paths(value: Any, *, path: str = "root") -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            lowered = str(key).lower()
            if "label" in lowered or "correct" in lowered:
                result.append(child_path)
            result.extend(_label_like_paths(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.extend(_label_like_paths(child, path=f"{path}[{index}]"))
    return result


def _write_candidate_subset(
    source: Path, destination: Path, *, sample_ids: set[str],
) -> dict[str, Any]:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    seen: set[str] = set()
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for record in read_jsonl(source):
                sample_id = stable_sample_id(str(record["split"]), record["sample_id"])
                if sample_id not in sample_ids:
                    continue
                forbidden = _label_like_paths(record)
                if forbidden:
                    raise ValueError(f"candidate source unexpectedly contains label-like fields: {sorted(forbidden)}")
                handle.write(canonical_json(record) + "\n")
                seen.add(sample_id)
                if seen == sample_ids:
                    break
            handle.flush()
            os.fsync(handle.fileno())
        if seen != sample_ids:
            raise ValueError(f"candidate subset coverage mismatch: missing={sorted(sample_ids-seen)[:5]}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return artifact_identity(destination)


def _stored_float32_checks(
    reference: dict[str, Any], cached: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in PERSISTED_FLOAT32_FIELDS:
        left = np.asarray(reference[name])
        right = np.asarray(cached[name])
        result[name] = {
            "reference_dtype": str(left.dtype),
            "cache_dtype": str(right.dtype),
            "shape": list(left.shape),
            "bitwise_equal": bool(np.array_equal(left, right)),
            "max_abs_error": float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))),
            "cast_to_float16": False,
        }
    return result


def run_fp16_diagnostic(
    *,
    frozen_features_path: str | Path,
    split_manifest_path: str | Path,
    output_dir: str | Path,
    head_override_dir: str | Path,
    checkpoint_path: str | Path,
    v2_root: str | Path,
    extraction_device: str = "auto",
    inference_device: str = "cpu",
    max_samples: int = 200,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run a label-free pre-cast-f32 versus persisted-f16 cache audit."""
    sample_limit = int(max_samples)
    if sample_limit <= 0 or sample_limit > 200:
        raise ValueError("fp16 diagnostic requires 1..200 development expressions")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = Path(frozen_features_path).resolve()
    split_manifest = Path(split_manifest_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    override = Path(head_override_dir).resolve()
    v2 = Path(v2_root).resolve()
    cache_dir = output/"persisted_float16_cache"
    reference_dir = output/"precast_float32_reference"
    extraction_manifest = extract_fullchain_features(
        frozen_features_path=source,
        split_manifest_path=split_manifest,
        output_dir=cache_dir,
        device=extraction_device,
        batch_size=int(batch_size),
        shard_samples=64,
        max_samples=sample_limit,
        precast_reference_dir=reference_dir,
    )
    reference_arrays = load_fullchain_arrays(reference_dir)
    cached_arrays = load_fullchain_arrays(cache_dir)
    reference_ids = list(map(str, reference_arrays["sample_ids"]))
    cached_ids = list(map(str, cached_arrays["sample_ids"]))
    if reference_ids != cached_ids or len(set(reference_ids)) != sample_limit:
        raise AssertionError("pre-cast/cache sample identity mismatch")
    for name in FLOAT16_STORAGE_FIELDS:
        if np.asarray(reference_arrays[name]).dtype != np.float32:
            raise AssertionError(f"{name} pre-cast reference is not float32")
        if np.asarray(cached_arrays[name]).dtype != np.float16:
            raise AssertionError(f"{name} persisted cache is not float16")
    precision = compare_precision_arrays(
        {name: np.asarray(reference_arrays[name]) for name in FLOAT16_STORAGE_FIELDS},
        {name: np.asarray(cached_arrays[name]) for name in FLOAT16_STORAGE_FIELDS},
    )
    float32_storage = _stored_float32_checks(reference_arrays, cached_arrays)

    candidate_subset = output/"candidate_features_label_free_subset.jsonl"
    candidate_identity = _write_candidate_subset(
        source, candidate_subset, sample_ids=set(reference_ids),
    )
    reference_catalog = FeatureCatalog(
        [reference_dir], head_override_dirs=[override],
        candidate_feature_paths=[candidate_subset],
    )
    cached_catalog = FeatureCatalog(
        [cache_dir], head_override_dirs=[override],
        candidate_feature_paths=[candidate_subset],
    )
    prior, prior_valid = load_v2_oof_prior(v2, reference_ids)
    if not bool(prior_valid.all()):
        raise ValueError("V2 OOF prior is incomplete for the diagnostic cohort")
    prior_lookup = prior_array_to_lookup(np.asarray(reference_ids), prior)
    reference_prediction = predict_ranker_streaming(
        catalog=reference_catalog,
        sample_ids=set(reference_ids),
        priors=prior_lookup,
        checkpoint_path=checkpoint,
        device=inference_device,
        batch_size=32,
    )
    cached_prediction = predict_ranker_streaming(
        catalog=cached_catalog,
        sample_ids=set(reference_ids),
        priors=prior_lookup,
        checkpoint_path=checkpoint,
        device=inference_device,
        batch_size=32,
    )
    if not np.array_equal(reference_prediction["sample_ids"], cached_prediction["sample_ids"]):
        raise AssertionError("reference/cache prediction order differs")
    if not np.array_equal(reference_prediction["candidate_ids"], cached_prediction["candidate_ids"]):
        raise AssertionError("reference/cache prediction candidate identity differs")
    prediction_effect = compare_feature_quantized_predictions(
        reference_scores=np.asarray(reference_prediction["scores"], dtype=np.float32),
        quantized_input_scores=np.asarray(cached_prediction["scores"], dtype=np.float32),
        reference_probabilities=np.asarray(reference_prediction["probabilities"], dtype=np.float32),
        quantized_input_probabilities=np.asarray(cached_prediction["probabilities"], dtype=np.float32),
    )
    reference_prediction_path = output/"reference_predictions.npz"
    cached_prediction_path = output/"cached_predictions.npz"
    prediction_fields = ("sample_ids", "candidate_ids", "scores", "probabilities")
    _atomic_npz(
        reference_prediction_path,
        **{name: np.asarray(reference_prediction[name]) for name in prediction_fields},
    )
    _atomic_npz(
        cached_prediction_path,
        **{name: np.asarray(cached_prediction[name]) for name in prediction_fields},
    )
    precision_payload = {
        "kind":"v3_real_precast_float32_vs_persisted_float16_fullchain_audit",
        "feature_precision":precision,
        "persisted_float32_fields":float32_storage,
        "prediction_effect":prediction_effect,
        "top_rank_parity":prediction_effect["top_rank_parity"],
        "full_ranking_parity":{
            "scores":prediction_effect["scores"]["ranking"]["full_order_consistency"],
            "probabilities":prediction_effect["probabilities"]["ranking"]["full_order_consistency"],
        },
        "labels_read":False,
    }
    report = write_precision_efficiency_report(
        output/"precision_report.json",
        precision=precision_payload,
        disk=directory_statistics(output, sample_count=sample_limit),
        metadata={
            "cohort":"first development expressions in frozen base_train source",
            "sample_limit":sample_limit,
            "capture":"float32 arrays captured before first float16 astype call",
            "inference":"same smoke FCER checkpoint, float32 model, CPU by default",
            "float16_storage_fields":list(FLOAT16_STORAGE_FIELDS),
            "float32_storage_fields":list(PERSISTED_FLOAT32_FIELDS),
            "candidate_source_forbidden_label_fields_checked":True,
            "formal_lockcheck_read":False,
            "formal_test_read":False,
        },
    )
    manifest = {
        "schema_version":"3.0.0",
        "artifact_type":"v3_fp16_diagnostic_bundle",
        "status":"complete",
        "created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "row_count":sample_limit,
        "unique_sample_count":len(set(reference_ids)),
        "unique_candidate_count":sample_limit*5,
        "labels_read":False,
        "formal_lockcheck_read":False,
        "formal_test_read":False,
        "source":artifact_identity(source),
        "split_manifest":artifact_identity(split_manifest),
        "checkpoint":artifact_identity(checkpoint),
        "head_override_manifest":artifact_identity(override/"artifact_manifest.json"),
        "candidate_subset":candidate_identity,
        "v2_oof_base":artifact_identity(v2/"oof_base/oof_base_predictions.npz"),
        "v2_oof_setrank":artifact_identity(v2/"oof_primary/oof_setrank_predictions.npz"),
        "extraction_content_sha256":extraction_manifest["content_sha256"],
        "precast_reference_manifest":artifact_identity(reference_dir/"artifact_manifest.json"),
        "persisted_cache_manifest":artifact_identity(cache_dir/"artifact_manifest.json"),
        "reference_predictions":artifact_identity(reference_prediction_path),
        "cached_predictions":artifact_identity(cached_prediction_path),
        "precision_report":artifact_identity(output/"precision_report.json"),
        "precision_report_content_sha256":report["content_sha256"],
    }
    manifest["content_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    atomic_write_json(output/"artifact_manifest.json",manifest)
    return manifest


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real label-free V3 float16 cache audit.")
    parser.add_argument("--frozen-features", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head-override", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--extraction-device", default="auto")
    parser.add_argument("--inference-device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    manifest = run_fp16_diagnostic(
        frozen_features_path=args.frozen_features,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        head_override_dir=args.head_override,
        checkpoint_path=args.checkpoint,
        v2_root=args.v2_root,
        extraction_device=args.extraction_device,
        inference_device=args.inference_device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "output":str(Path(args.output_dir).resolve()),
        "content_sha256":manifest["content_sha256"],
        "labels_read":manifest["labels_read"],
    },sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
