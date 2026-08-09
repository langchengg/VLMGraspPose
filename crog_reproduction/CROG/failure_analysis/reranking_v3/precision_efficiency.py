from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .schema import artifact_identity, atomic_write_json, canonical_json, sha256_bytes


_ERROR_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("precision quantiles require non-empty finite values")
    return {
        f"p{str(100 * quantile).rstrip('0').rstrip('.').replace('.', '_')}": float(np.quantile(array, quantile))
        for quantile in _ERROR_QUANTILES
    }


def _precision_error(
    reference: np.ndarray,
    cached: np.ndarray,
    *,
    name: str,
    relative_floor: float,
) -> dict[str, Any]:
    f32 = np.asarray(reference)
    f16 = np.asarray(cached)
    if f32.dtype != np.dtype(np.float32) or f16.dtype != np.dtype(np.float16):
        raise ValueError(f"{name} must compare float32 reference with float16 cache")
    if f32.shape != f16.shape:
        raise ValueError(f"{name} precision arrays differ in shape: {f32.shape} != {f16.shape}")
    if not f32.size:
        raise ValueError(f"{name} precision arrays are empty")
    if not np.isfinite(f32).all() or not np.isfinite(f16).all():
        raise ValueError(f"{name} precision arrays contain non-finite values")
    reference64 = f32.astype(np.float64)
    cached64 = f16.astype(np.float64)
    absolute = np.abs(cached64 - reference64)
    relative = absolute / np.maximum(np.abs(reference64), relative_floor)
    reference_l2 = float(np.linalg.norm(reference64.ravel()))
    return {
        "shape": list(f32.shape),
        "element_count": int(f32.size),
        "reference_dtype": str(f32.dtype),
        "cache_dtype": str(f16.dtype),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "abs_error_quantiles": _quantiles(absolute),
        "max_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
        "relative_error_quantiles": _quantiles(relative),
        "relative_l2_error": float(
            np.linalg.norm(absolute.ravel()) / max(reference_l2, relative_floor)
        ),
    }


def compare_precision_arrays(
    reference_arrays: Mapping[str, np.ndarray],
    cached_arrays: Mapping[str, np.ndarray],
    *,
    relative_floor: float = 1e-8,
) -> dict[str, Any]:
    """Compare exact-key float32 references with shape-identical float16 caches."""
    floor = float(relative_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive")
    if any(not isinstance(value, str) for value in reference_arrays) or any(
        not isinstance(value, str) for value in cached_arrays
    ):
        raise ValueError("precision array names must be strings")
    reference_keys = set(reference_arrays)
    cached_keys = set(cached_arrays)
    if reference_keys != cached_keys:
        raise ValueError(
            "precision array keys differ: "
            f"missing={sorted(reference_keys - cached_keys)}, "
            f"extra={sorted(cached_keys - reference_keys)}"
        )
    if not reference_keys:
        raise ValueError("precision comparison requires at least one array")
    per_array: dict[str, dict[str, Any]] = {}
    total_elements = 0
    absolute_sum = 0.0
    relative_sum = 0.0
    absolute_square_sum = 0.0
    reference_square_sum = 0.0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    all_absolute: list[np.ndarray] = []
    all_relative: list[np.ndarray] = []
    for name in sorted(reference_keys):
        reference = np.asarray(reference_arrays[name])
        cached = np.asarray(cached_arrays[name])
        metrics = _precision_error(
            reference, cached, name=name, relative_floor=floor,
        )
        per_array[name] = metrics
        reference64 = reference.astype(np.float64)
        absolute = np.abs(cached.astype(np.float64) - reference64)
        relative = absolute / np.maximum(np.abs(reference64), floor)
        total_elements += int(reference.size)
        absolute_sum += float(absolute.sum())
        relative_sum += float(relative.sum())
        absolute_square_sum += float(np.square(absolute).sum())
        reference_square_sum += float(np.square(reference64).sum())
        maximum_absolute = max(maximum_absolute, float(absolute.max()))
        maximum_relative = max(maximum_relative, float(relative.max()))
        all_absolute.append(absolute.reshape(-1))
        all_relative.append(relative.reshape(-1))
    absolute_distribution = np.concatenate(all_absolute)
    relative_distribution = np.concatenate(all_relative)
    return {
        "array_count": len(per_array),
        "element_count": total_elements,
        "relative_floor": floor,
        "per_array": per_array,
        "overall": {
            "max_abs_error": maximum_absolute,
            "mean_abs_error": absolute_sum / total_elements,
            "abs_error_quantiles": _quantiles(absolute_distribution),
            "max_relative_error": maximum_relative,
            "mean_relative_error": relative_sum / total_elements,
            "relative_error_quantiles": _quantiles(relative_distribution),
            "relative_l2_error": math.sqrt(absolute_square_sum)
            / max(math.sqrt(reference_square_sum), floor),
        },
    }


def _ranking_consistency(reference: np.ndarray, cached: np.ndarray) -> dict[str, Any]:
    reference_order = np.argsort(-reference.astype(np.float64), axis=1, kind="stable")
    cached_order = np.argsort(-cached.astype(np.float64), axis=1, kind="stable")
    top1 = reference_order[:, 0] == cached_order[:, 0]
    full = np.all(reference_order == cached_order, axis=1)
    return {
        "sample_count": int(len(reference)),
        "candidate_count": int(reference.shape[1]),
        "top1_consistent_count": int(top1.sum()),
        "top1_consistency": float(top1.mean()),
        "full_order_consistent_count": int(full.sum()),
        "full_order_consistency": float(full.mean()),
    }


def compare_prediction_precision(
    *,
    reference_scores: np.ndarray,
    cached_scores: np.ndarray,
    reference_probabilities: np.ndarray,
    cached_probabilities: np.ndarray,
    relative_floor: float = 1e-8,
) -> dict[str, Any]:
    """Compare f32/f16 prediction values and their deterministic rankings."""
    precision = compare_precision_arrays(
        {"scores": reference_scores, "probabilities": reference_probabilities},
        {"scores": cached_scores, "probabilities": cached_probabilities},
        relative_floor=relative_floor,
    )
    scores32 = np.asarray(reference_scores)
    scores16 = np.asarray(cached_scores)
    probabilities32 = np.asarray(reference_probabilities)
    probabilities16 = np.asarray(cached_probabilities)
    if scores32.ndim != 2 or scores32.shape[1] < 2:
        raise ValueError("prediction scores must be [samples,candidates] with at least two candidates")
    if probabilities32.shape != scores32.shape:
        raise ValueError("prediction scores/probabilities differ in shape")
    if np.any(probabilities32 < 0.0) or np.any(probabilities32 > 1.0):
        raise ValueError("reference probabilities must be in [0,1]")
    if np.any(probabilities16 < 0.0) or np.any(probabilities16 > 1.0):
        raise ValueError("cached probabilities must be in [0,1]")
    return {
        "precision": precision,
        "ranking": {
            "scores": _ranking_consistency(scores32, scores16),
            "probabilities": _ranking_consistency(probabilities32, probabilities16),
        },
    }


def _prediction_effect_error(
    reference: np.ndarray, observed: np.ndarray, *, name: str, relative_floor: float,
) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(observed)
    if left.dtype != np.float32 or right.dtype != np.float32:
        raise ValueError(f"{name} inference outputs must both be float32")
    if left.ndim != 2 or left.shape[1] < 2 or right.shape != left.shape:
        raise ValueError(f"{name} inference outputs must be matching [samples,candidates] arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{name} inference outputs contain non-finite values")
    absolute = np.abs(right.astype(np.float64) - left.astype(np.float64))
    relative = absolute / np.maximum(np.abs(left.astype(np.float64)), float(relative_floor))
    return {
        "shape": list(left.shape),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "abs_error_quantiles": _quantiles(absolute),
        "max_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
        "relative_error_quantiles": _quantiles(relative),
        "ranking": _ranking_consistency(left, right),
    }


def compare_feature_quantized_predictions(
    *,
    reference_scores: np.ndarray,
    quantized_input_scores: np.ndarray,
    reference_probabilities: np.ndarray,
    quantized_input_probabilities: np.ndarray,
    relative_floor: float = 1e-8,
) -> dict[str, Any]:
    """Measure fp16 *feature-storage* effects on float32 model outputs."""
    floor = float(relative_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive")
    reference_probability = np.asarray(reference_probabilities)
    quantized_probability = np.asarray(quantized_input_probabilities)
    if np.any(reference_probability < 0.0) or np.any(reference_probability > 1.0):
        raise ValueError("reference probabilities must be in [0,1]")
    if np.any(quantized_probability < 0.0) or np.any(quantized_probability > 1.0):
        raise ValueError("quantized-input probabilities must be in [0,1]")
    scores = _prediction_effect_error(
        reference_scores, quantized_input_scores, name="scores", relative_floor=floor,
    )
    probabilities = _prediction_effect_error(
        reference_probability, quantized_probability,
        name="probabilities", relative_floor=floor,
    )
    return {
        "comparison": "float32 inference from float32 features versus float32 inference after float16 storage round-trip",
        "scores": scores,
        "probabilities": probabilities,
        "top_rank_parity": {
            "scores": scores["ranking"]["top1_consistency"],
            "probabilities": probabilities["ranking"]["top1_consistency"],
        },
    }


def audit_float16_storage_roundtrip(
    reference_arrays: Mapping[str, np.ndarray],
    *,
    predictor: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
    relative_floor: float = 1e-8,
) -> dict[str, Any]:
    """Execute a real f32→f16 cache round-trip and compare model predictions.

    The callback must return float32 ``scores`` and ``probabilities``.  The
    quantized call receives float16-cached arrays converted back to float32,
    matching the V3 loader/model boundary.
    """
    references = {name: np.asarray(value) for name, value in reference_arrays.items()}
    if not references or any(value.dtype != np.float32 for value in references.values()):
        raise ValueError("storage audit requires non-empty float32 reference arrays")
    cached = {name: value.astype(np.float16) for name, value in references.items()}
    precision = compare_precision_arrays(references, cached, relative_floor=relative_floor)
    reference_prediction = dict(predictor(references))
    quantized_prediction = dict(predictor({
        name: value.astype(np.float32) for name, value in cached.items()
    }))
    for name, prediction in (
        ("reference", reference_prediction), ("quantized", quantized_prediction),
    ):
        if set(prediction) != {"scores", "probabilities"}:
            raise ValueError(f"{name} predictor output must contain exactly scores and probabilities")
    prediction = compare_feature_quantized_predictions(
        reference_scores=np.asarray(reference_prediction["scores"]),
        quantized_input_scores=np.asarray(quantized_prediction["scores"]),
        reference_probabilities=np.asarray(reference_prediction["probabilities"]),
        quantized_input_probabilities=np.asarray(quantized_prediction["probabilities"]),
        relative_floor=relative_floor,
    )
    return {
        "kind": "v3_float16_storage_roundtrip_audit",
        "reference_dtype": "float32",
        "cache_dtype": "float16",
        "model_input_dtype_after_load": "float32",
        "feature_precision": precision,
        "prediction_effect": prediction,
        "labels_read": False,
    }


def _load_npz_fields(path: str | Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    requested = tuple(dict.fromkeys(map(str, fields)))
    if not requested:
        raise ValueError("at least one NPZ field is required")
    with np.load(source, allow_pickle=False) as payload:
        missing = set(requested) - set(payload.files)
        if missing:
            raise ValueError(f"NPZ is missing fields {sorted(missing)}: {source}")
        return {name: np.asarray(payload[name]) for name in requested}


def audit_npz_float16_storage(
    *,
    reference_features_path: str | Path,
    cached_features_path: str | Path,
    feature_keys: Sequence[str],
    reference_predictions_path: str | Path,
    cached_predictions_path: str | Path,
    score_key: str = "scores",
    probability_key: str = "probabilities",
    relative_floor: float = 1e-8,
) -> dict[str, Any]:
    """Audit independent pre-cast f32 and post-cast f16 NPZ artifacts."""
    reference_features = _load_npz_fields(reference_features_path, feature_keys)
    cached_features = _load_npz_fields(cached_features_path, feature_keys)
    feature_precision = compare_precision_arrays(
        reference_features, cached_features, relative_floor=relative_floor,
    )
    reference_prediction = _load_npz_fields(
        reference_predictions_path, (score_key, probability_key),
    )
    cached_prediction = _load_npz_fields(
        cached_predictions_path, (score_key, probability_key),
    )
    prediction_precision = compare_prediction_precision(
        reference_scores=reference_prediction[score_key],
        cached_scores=cached_prediction[score_key],
        reference_probabilities=reference_prediction[probability_key],
        cached_probabilities=cached_prediction[probability_key],
        relative_floor=relative_floor,
    )
    return {
        "kind": "v3_independent_npz_float16_storage_audit",
        "feature_precision": feature_precision,
        "prediction_precision": prediction_precision,
        "top_rank_parity": {
            "scores": prediction_precision["ranking"]["scores"]["top1_consistency"],
            "probabilities": prediction_precision["ranking"]["probabilities"]["top1_consistency"],
        },
        "sources": {
            "reference_features": artifact_identity(reference_features_path),
            "cached_features": artifact_identity(cached_features_path),
            "reference_predictions": artifact_identity(reference_predictions_path),
            "cached_predictions": artifact_identity(cached_predictions_path),
        },
        "labels_read": False,
    }


def array_storage_statistics(
    arrays: Mapping[str, np.ndarray], *, sample_count: int,
) -> dict[str, Any]:
    samples = int(sample_count)
    if samples <= 0:
        raise ValueError("sample_count must be positive")
    if not arrays:
        raise ValueError("array storage statistics require at least one array")
    if any(not isinstance(value, str) for value in arrays):
        raise ValueError("array storage names must be strings")
    per_array: dict[str, dict[str, Any]] = {}
    total = 0
    for name in sorted(map(str, arrays)):
        value = np.asarray(arrays[name])
        if value.ndim == 0 or value.shape[0] != samples:
            raise ValueError(f"{name} first dimension must equal sample_count")
        size = int(value.nbytes)
        total += size
        per_array[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": size,
            "bytes_per_sample": size / samples,
        }
    return {
        "sample_count": samples,
        "array_count": len(per_array),
        "total_bytes": total,
        "bytes_per_sample": total / samples,
        "per_array": per_array,
    }


def directory_statistics(
    directory: str | Path, *, sample_count: int | None = None,
) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"disk statistics path is not a directory: {root}")
    if sample_count is not None and int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    file_count = 0
    directory_count = 0
    symlink_count = 0
    total_bytes = 0
    bytes_by_suffix: dict[str, int] = {}
    largest_path: str | None = None
    largest_bytes = -1
    for value in sorted(root.rglob("*")):
        if value.is_symlink():
            symlink_count += 1
            continue
        if value.is_dir():
            directory_count += 1
            continue
        if not value.is_file():
            continue
        size = int(value.stat().st_size)
        relative = value.relative_to(root).as_posix()
        suffix = value.suffix.lower() or "<none>"
        file_count += 1
        total_bytes += size
        bytes_by_suffix[suffix] = bytes_by_suffix.get(suffix, 0) + size
        if size > largest_bytes or (size == largest_bytes and (largest_path is None or relative < largest_path)):
            largest_path = relative
            largest_bytes = size
    return {
        "directory": str(root),
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "total_bytes": total_bytes,
        "bytes_per_sample": None if sample_count is None else total_bytes / int(sample_count),
        "bytes_by_suffix": dict(sorted(bytes_by_suffix.items())),
        "largest_file": None if largest_path is None else {
            "path": largest_path, "bytes": largest_bytes,
        },
    }


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return observed if sys.platform == "darwin" else observed * 1024


def measure_efficiency(
    operation: Callable[[], Any],
    *,
    sample_count: int,
    warmup: int = 3,
    repeat: int = 10,
    storage_bytes: int | None = None,
    disk_directory: str | Path | None = None,
    synchronize: Callable[[], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    rss_reader: Callable[[], int | None] = _peak_rss_bytes,
) -> dict[str, Any]:
    """Measure a fixed operation with an explicit, reproducible warmup/repeat protocol."""
    samples = int(sample_count)
    warmups = int(warmup)
    repeats = int(repeat)
    if samples <= 0 or warmups < 0 or repeats <= 0:
        raise ValueError("sample_count/repeat must be positive and warmup non-negative")
    if storage_bytes is not None and int(storage_bytes) < 0:
        raise ValueError("storage_bytes must be non-negative")
    sync = (lambda: None) if synchronize is None else synchronize
    def read_rss() -> int | None:
        observed = rss_reader()
        if observed is None:
            return None
        value = int(observed)
        if value < 0:
            raise ValueError("RSS measurements must be non-negative")
        return value

    rss_before = read_rss()
    for _ in range(warmups):
        operation()
        sync()
    latencies = []
    for _ in range(repeats):
        sync()
        started = float(clock())
        operation()
        sync()
        elapsed = float(clock()) - started
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("measured latency must be finite and positive")
        latencies.append(elapsed)
    rss_after = read_rss()
    ordered = sorted(latencies)
    mean_latency = sum(latencies) / repeats
    variance = sum((value - mean_latency) ** 2 for value in latencies) / repeats
    percentile95 = ordered[max(0, math.ceil(0.95 * repeats) - 1)]
    rss_values = [value for value in (rss_before, rss_after) if value is not None]
    result: dict[str, Any] = {
        "sample_count_per_repeat": samples,
        "warmup": warmups,
        "repeat": repeats,
        "total_measured_samples": samples * repeats,
        "latency_seconds": {
            "values": latencies,
            "minimum": ordered[0],
            "maximum": ordered[-1],
            "mean": mean_latency,
            "median": (
                ordered[repeats // 2]
                if repeats % 2 else (ordered[repeats // 2 - 1] + ordered[repeats // 2]) / 2
            ),
            "p95_nearest_rank": percentile95,
            "population_std": math.sqrt(variance),
            "mean_per_sample": mean_latency / samples,
        },
        "throughput_samples_per_second": {
            "mean": samples / mean_latency,
            "minimum": samples / ordered[-1],
            "maximum": samples / ordered[0],
        },
        "rss": {
            "measurement": "process peak RSS from resource.getrusage when available",
            "before_bytes": rss_before,
            "after_bytes": rss_after,
            "maximum_observed_bytes": None if not rss_values else max(rss_values),
            "increase_bytes": None if rss_before is None or rss_after is None else max(0, rss_after - rss_before),
        },
        "storage_bytes": None if storage_bytes is None else int(storage_bytes),
        "storage_bytes_per_sample": None if storage_bytes is None else int(storage_bytes) / samples,
    }
    if disk_directory is not None:
        result["disk"] = directory_statistics(disk_directory, sample_count=samples)
    return result


def write_precision_efficiency_report(
    output_path: str | Path,
    *,
    precision: Mapping[str, Any] | None = None,
    efficiency: Mapping[str, Any] | None = None,
    storage: Mapping[str, Any] | None = None,
    disk: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if precision is None and efficiency is None and storage is None and disk is None:
        raise ValueError("precision/efficiency report requires at least one measurement")
    value: dict[str, Any] = {
        "schema_version": "3.0.0",
        "kind": "v3_fp16_precision_efficiency_acceptance",
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "labels_read": False,
        "metadata": {} if metadata is None else dict(metadata),
    }
    for name, payload in (
        ("precision", precision), ("efficiency", efficiency),
        ("storage", storage), ("disk", disk),
    ):
        if payload is not None:
            value[name] = dict(payload)
    value["content_sha256"] = sha256_bytes(canonical_json(value).encode("utf-8"))
    atomic_write_json(output_path, value)
    return value


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit independent float32 reference and float16 cached V3 NPZ artifacts.",
    )
    parser.add_argument("--reference-features", required=True)
    parser.add_argument("--cached-features", required=True)
    parser.add_argument("--feature-key", action="append", required=True)
    parser.add_argument("--reference-predictions", required=True)
    parser.add_argument("--cached-predictions", required=True)
    parser.add_argument("--score-key", default="scores")
    parser.add_argument("--probability-key", default="probabilities")
    parser.add_argument("--relative-floor", type=float, default=1e-8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    audit = audit_npz_float16_storage(
        reference_features_path=args.reference_features,
        cached_features_path=args.cached_features,
        feature_keys=args.feature_key,
        reference_predictions_path=args.reference_predictions,
        cached_predictions_path=args.cached_predictions,
        score_key=args.score_key,
        probability_key=args.probability_key,
        relative_floor=args.relative_floor,
    )
    report = write_precision_efficiency_report(
        args.output,
        precision=audit,
        metadata={
            "invocation": "python -m failure_analysis.reranking_v3.precision_efficiency",
            "precast_reference_required": True,
        },
    )
    print(json.dumps({
        "output": str(Path(args.output).expanduser().resolve()),
        "content_sha256": report["content_sha256"],
        "top_rank_parity": audit["top_rank_parity"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
