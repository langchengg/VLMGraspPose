"""Evaluation-only interpretation helpers for CROG reranking V3.

This module deliberately separates prediction from evaluation:

* subgroup functions receive analysis metadata and already-computed correctness
  masks;
* permutation sensitivity passes only named feature arrays and a candidate mask
  to the prediction callback;
* its metric callback receives only predictions.

Consequently analysis metadata, labels, correctness, and other evaluation-only
fields cannot enter the prediction callback through this API.  Nothing here is
an inference feature extractor.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType
from typing import Any

import numpy as np

from .schema import atomic_write_json, find_forbidden_token


MISSING_CATEGORY = "__missing__"
OUTCOME_ORDER = ("recovered", "harmful", "stable_correct", "stable_incorrect")
NATIVE_COMPARISON_VARIANTS = ("native", "rgbd", "latent", "text", "gate", "uncertainty")


def _binary_mask(values: Sequence[bool] | np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype != np.bool_:
        if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
            raise ValueError(f"{name} must contain only booleans or finite 0/1 values")
        if not np.isin(raw, (0, 1)).all():
            raise ValueError(f"{name} must contain only booleans or 0/1 values")
    return raw.astype(bool, copy=False)


def _candidate_mask(values: np.ndarray | Sequence[Sequence[bool]], sample_count: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] != sample_count or raw.shape[1] < 1:
        raise ValueError("candidate_mask must have shape (N,K) with K >= 1")
    if raw.dtype != np.bool_:
        if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all() or not np.isin(raw, (0, 1)).all():
            raise ValueError("candidate_mask must contain only booleans or 0/1 values")
    return raw.astype(bool, copy=False)


def _category(value: Any) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return MISSING_CATEGORY
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return MISSING_CATEGORY
        if not math.isfinite(value):
            raise ValueError("analysis metadata contains infinity")
        return value
    if isinstance(value, str):
        return value
    raise ValueError(f"analysis metadata values must be scalar, received {type(value).__name__}")


def _metadata_columns(
    analysis_metadata: Mapping[str, Sequence[Any]] | None,
    sample_count: int,
) -> dict[str, np.ndarray]:
    if analysis_metadata is None:
        return {}
    if not isinstance(analysis_metadata, Mapping):
        raise TypeError("analysis_metadata must be a mapping of column name to values")
    columns: dict[str, np.ndarray] = {}
    for raw_name, values in analysis_metadata.items():
        name = str(raw_name)
        if not name or name in columns:
            raise ValueError("analysis metadata column names must be non-empty and unique")
        raw = np.asarray(values, dtype=object)
        if raw.shape != (sample_count,):
            raise ValueError(f"analysis metadata {name!r} must have length {sample_count}")
        columns[name] = np.asarray([_category(value) for value in raw], dtype=object)
    return columns


def _selected_fields(columns: Mapping[str, np.ndarray], subgroup_fields: Sequence[str] | None) -> tuple[str, ...]:
    fields = tuple(columns) if subgroup_fields is None else tuple(map(str, subgroup_fields))
    if len(fields) != len(set(fields)):
        raise ValueError("subgroup_fields must be unique")
    missing = set(fields) - set(columns)
    if missing:
        raise ValueError(f"unknown subgroup fields: {sorted(missing)}")
    return fields


def _category_sort_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, json.dumps(value, sort_keys=True, ensure_ascii=False)


def _subgroup_slices(
    columns: Mapping[str, np.ndarray],
    fields: Sequence[str],
    sample_count: int,
    *,
    include_all: bool,
) -> list[tuple[str, Any, np.ndarray]]:
    result: list[tuple[str, Any, np.ndarray]] = []
    if include_all:
        result.append(("__all__", "__all__", np.arange(sample_count, dtype=np.int64)))
    for field in fields:
        values = columns[field]
        categories = sorted(set(values.tolist()), key=_category_sort_key)
        for category in categories:
            result.append((field, category, np.flatnonzero(values == category)))
    return result


def _wilson_interval(successes: int, support: int, confidence_level: float) -> tuple[float, float]:
    if support <= 0:
        return 0.0, 0.0
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rate = successes / support
    denominator = 1.0 + z * z / support
    center = (rate + z * z / (2.0 * support)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / support + z * z / (4.0 * support * support)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _bootstrap_intervals(
    reference: np.ndarray,
    challenger: np.ndarray,
    clusters: np.ndarray | None,
    *,
    iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], int]:
    if not len(reference):
        return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 0
    if clusters is None:
        inverse = np.arange(len(reference), dtype=np.int64)
        unit_count = len(reference)
    else:
        _, inverse = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
        unit_count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=unit_count).astype(np.float64)
    reference_sums = np.bincount(inverse, weights=reference.astype(np.float64), minlength=unit_count)
    challenger_sums = np.bincount(inverse, weights=challenger.astype(np.float64), minlength=unit_count)
    reference_values = np.empty(iterations, dtype=np.float64)
    challenger_values = np.empty(iterations, dtype=np.float64)
    chunk_size = 1000
    for start in range(0, iterations, chunk_size):
        size = min(chunk_size, iterations - start)
        draw = rng.integers(0, unit_count, size=(size, unit_count))
        denominators = counts[draw].sum(axis=1)
        reference_values[start : start + size] = reference_sums[draw].sum(axis=1) / denominators
        challenger_values[start : start + size] = challenger_sums[draw].sum(axis=1) / denominators
    tail = (1.0 - confidence_level) / 2.0

    def interval(values: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))

    return interval(reference_values), interval(challenger_values), interval(challenger_values - reference_values), unit_count


def subgroup_metrics(
    *,
    analysis_metadata: Mapping[str, Sequence[Any]],
    reference_correct: Sequence[bool] | np.ndarray,
    challenger_correct: Sequence[bool] | np.ndarray,
    subgroup_fields: Sequence[str] | None = None,
    cluster_ids: Sequence[Any] | np.ndarray | None = None,
    ci_method: str = "wilson",
    confidence_level: float = 0.95,
    bootstrap_iterations: int = 10_000,
    seed: int = 20260801,
    min_support: int = 1,
) -> list[dict[str, Any]]:
    """Compute evaluation-only rates and deltas for explicit metadata groups.

    ``wilson`` reports Wilson intervals for each rate and a conservative
    difference interval obtained from their endpoints.  ``bootstrap`` performs
    paired resampling; when ``cluster_ids`` is supplied, whole clusters are
    resampled and all member observations are retained.
    """
    reference = _binary_mask(reference_correct, name="reference_correct")
    challenger = _binary_mask(challenger_correct, name="challenger_correct")
    if reference.shape != challenger.shape:
        raise ValueError("correctness masks must have equal length")
    if ci_method not in {"wilson", "bootstrap"}:
        raise ValueError("ci_method must be 'wilson' or 'bootstrap'")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if int(min_support) < 1:
        raise ValueError("min_support must be positive")
    if ci_method == "bootstrap" and int(bootstrap_iterations) < 1:
        raise ValueError("bootstrap_iterations must be positive")
    columns = _metadata_columns(analysis_metadata, len(reference))
    fields = _selected_fields(columns, subgroup_fields)
    clusters = None if cluster_ids is None else np.asarray(cluster_ids, dtype=object)
    if clusters is not None and clusters.shape != reference.shape:
        raise ValueError("cluster_ids must have the same length as correctness masks")
    root_rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for field, value, indices in _subgroup_slices(columns, fields, len(reference), include_all=False):
        support = len(indices)
        if support < int(min_support):
            continue
        ref = reference[indices]
        chal = challenger[indices]
        reference_count = int(ref.sum())
        challenger_count = int(chal.sum())
        reference_rate = reference_count / support
        challenger_rate = challenger_count / support
        if ci_method == "wilson":
            reference_ci = _wilson_interval(reference_count, support, float(confidence_level))
            challenger_ci = _wilson_interval(challenger_count, support, float(confidence_level))
            delta_ci = (
                max(-1.0, challenger_ci[0] - reference_ci[1]),
                min(1.0, challenger_ci[1] - reference_ci[0]),
            )
            resampling_units = None
            iterations = None
        else:
            subgroup_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
            reference_ci, challenger_ci, delta_ci, resampling_units = _bootstrap_intervals(
                ref,
                chal,
                None if clusters is None else clusters[indices],
                iterations=int(bootstrap_iterations),
                confidence_level=float(confidence_level),
                rng=np.random.default_rng(subgroup_seed),
            )
            iterations = int(bootstrap_iterations)
        rows.append(
            {
                "subgroup_field": field,
                "subgroup_value": value,
                "support": support,
                "support_fraction": support / max(len(reference), 1),
                "reference_correct": reference_count,
                "challenger_correct": challenger_count,
                "reference_rate": reference_rate,
                "challenger_rate": challenger_rate,
                "delta": challenger_rate - reference_rate,
                "recovered": int(((~ref) & chal).sum()),
                "harmful": int((ref & (~chal)).sum()),
                "ci_method": ci_method,
                "confidence_level": float(confidence_level),
                "reference_ci_lower": reference_ci[0],
                "reference_ci_upper": reference_ci[1],
                "challenger_ci_lower": challenger_ci[0],
                "challenger_ci_upper": challenger_ci[1],
                "delta_ci_lower": delta_ci[0],
                "delta_ci_upper": delta_ci[1],
                "bootstrap_iterations": iterations,
                "resampling_unit_count": resampling_units,
            }
        )
    return rows


def recovered_harmful_distribution(
    *,
    reference_correct: Sequence[bool] | np.ndarray,
    challenger_correct: Sequence[bool] | np.ndarray,
    analysis_metadata: Mapping[str, Sequence[Any]] | None = None,
    subgroup_fields: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a long-form recovered/harmful/stable outcome distribution."""
    reference = _binary_mask(reference_correct, name="reference_correct")
    challenger = _binary_mask(challenger_correct, name="challenger_correct")
    if reference.shape != challenger.shape:
        raise ValueError("correctness masks must have equal length")
    columns = _metadata_columns(analysis_metadata, len(reference))
    fields = _selected_fields(columns, subgroup_fields)
    outcomes = {
        "recovered": (~reference) & challenger,
        "harmful": reference & (~challenger),
        "stable_correct": reference & challenger,
        "stable_incorrect": (~reference) & (~challenger),
    }
    rows: list[dict[str, Any]] = []
    for field, value, indices in _subgroup_slices(columns, fields, len(reference), include_all=True):
        support = len(indices)
        changed = int((outcomes["recovered"][indices] | outcomes["harmful"][indices]).sum())
        for outcome in OUTCOME_ORDER:
            count = int(outcomes[outcome][indices].sum())
            rows.append(
                {
                    "subgroup_field": field,
                    "subgroup_value": value,
                    "outcome": outcome,
                    "support": support,
                    "count": count,
                    "rate": None if support == 0 else count / support,
                    "outcome_changing_support": changed,
                    "outcome_changing_share": (
                        count / changed if outcome in {"recovered", "harmful"} and changed else None
                    ),
                }
            )
    return rows


def _readonly_array(value: np.ndarray) -> np.ndarray:
    base = np.array(value, copy=True)
    if not np.issubdtype(base.dtype, np.number) and base.dtype != np.bool_:
        raise ValueError("prediction feature groups must contain numeric or boolean arrays")
    if np.issubdtype(base.dtype, np.number) and not np.isfinite(base).all():
        raise ValueError("prediction feature groups must be finite")
    base.setflags(write=False)
    view = base.view()
    view.setflags(write=False)
    return view


def _predict(
    callback: Callable[[Mapping[str, np.ndarray], np.ndarray], np.ndarray],
    features: Mapping[str, np.ndarray],
    candidate_mask: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    payload = MappingProxyType({name: _readonly_array(value) for name, value in features.items()})
    mask = _readonly_array(candidate_mask)
    prediction = np.asarray(callback(payload, mask))
    if prediction.ndim < 1 or prediction.shape[0] != sample_count:
        raise ValueError("prediction callback output must have sample axis N")
    if prediction.size == 0:
        raise ValueError("prediction callback output must not be empty")
    if not np.issubdtype(prediction.dtype, np.number) or not np.isfinite(prediction).all():
        raise ValueError("prediction callback output must be a finite numeric array")
    return np.asarray(prediction, dtype=np.float64)


def _metric(
    callback: Callable[[np.ndarray], float],
    predictions: np.ndarray,
) -> float:
    readonly = _readonly_array(predictions)
    raw = np.asarray(callback(readonly))
    if raw.shape != ():
        raise ValueError("metric callback must return one scalar")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("metric callback must return a finite scalar")
    return value


def _group_seed(seed: int, group_name: str, semantics: str) -> int:
    payload = f"{int(seed)}:{group_name}:{semantics}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _nonidentity_permutation(rng: np.random.Generator, size: int) -> np.ndarray:
    order = rng.permutation(size)
    if size > 1 and np.array_equal(order, np.arange(size)):
        order = np.roll(order, 1)
    return order


def _selected_indices(predictions: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray | None:
    if predictions.ndim != 2 or predictions.shape != candidate_mask.shape:
        return None
    valid_rows = candidate_mask.any(axis=1)
    selected = np.full(len(predictions), -1, dtype=np.int64)
    selected[valid_rows] = np.argmax(np.where(candidate_mask[valid_rows], predictions[valid_rows], -np.inf), axis=1)
    return selected


def feature_group_permutation_sensitivity(
    *,
    feature_groups: Mapping[str, np.ndarray],
    group_semantics: Mapping[str, str],
    prediction_callback: Callable[[Mapping[str, np.ndarray], np.ndarray], np.ndarray],
    metric_callback: Callable[[np.ndarray], float],
    candidate_mask: np.ndarray | Sequence[Sequence[bool]] | None = None,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Measure deterministic feature-group permutation sensitivity.

    ``sample`` semantics move a complete group value between samples.
    ``candidate`` semantics independently reassign complete candidate rows only
    among valid candidates within each sample; padding rows stay fixed.  The
    prediction callback receives no correctness mask, analysis metadata, or
    metric callback.
    """
    if not isinstance(feature_groups, Mapping) or not feature_groups:
        raise ValueError("feature_groups must be a non-empty mapping")
    names = tuple(map(str, feature_groups))
    if len(names) != len(set(names)):
        raise ValueError("feature group names must be unique")
    for name in names:
        token = find_forbidden_token(name)
        if token is not None:
            raise ValueError(f"forbidden inference feature group {name!r} contains {token!r}")
    normalized_semantics = {str(name): str(value) for name, value in group_semantics.items()}
    if len(normalized_semantics) != len(group_semantics) or set(normalized_semantics) != set(names):
        raise ValueError("group_semantics must exactly cover feature_groups")
    arrays = {str(name): np.asarray(value) for name, value in feature_groups.items()}
    sample_counts = {value.shape[0] for value in arrays.values() if value.ndim >= 1}
    if any(value.ndim < 1 for value in arrays.values()) or len(sample_counts) != 1:
        raise ValueError("every feature group must share a non-empty sample axis")
    sample_count = sample_counts.pop()
    if sample_count < 1:
        raise ValueError("permutation sensitivity requires at least one sample")
    semantics = {name: normalized_semantics[name] for name in names}
    if set(semantics.values()) - {"sample", "candidate"}:
        raise ValueError("group semantics must be 'sample' or 'candidate'")
    candidate_widths = {
        arrays[name].shape[1]
        for name in names
        if semantics[name] == "candidate" and arrays[name].ndim >= 2
    }
    if any(semantics[name] == "candidate" and arrays[name].ndim < 2 for name in names):
        raise ValueError("candidate-semantic groups must have shape (N,K,...)")
    if candidate_mask is None:
        if len(candidate_widths) > 1:
            raise ValueError("candidate-semantic groups disagree on K")
        candidate_count = candidate_widths.pop() if candidate_widths else 1
        mask = np.ones((sample_count, candidate_count), dtype=bool)
    else:
        mask = _candidate_mask(candidate_mask, sample_count)
        candidate_count = mask.shape[1]
        if candidate_widths and candidate_widths != {candidate_count}:
            raise ValueError("candidate-semantic feature groups and candidate_mask disagree on K")

    baseline_predictions = _predict(prediction_callback, arrays, mask, sample_count)
    baseline_metric = _metric(metric_callback, baseline_predictions)
    baseline_selected = _selected_indices(baseline_predictions, mask)
    rows: list[dict[str, Any]] = []
    for name in sorted(names):
        semantic = semantics[name]
        rng = np.random.default_rng(_group_seed(int(seed), name, semantic))
        permuted = {key: np.array(value, copy=True) for key, value in arrays.items()}
        permutations: list[list[int]]
        if semantic == "sample":
            order = _nonidentity_permutation(rng, sample_count)
            permuted[name] = np.asarray(arrays[name])[order].copy()
            permutations = [order.astype(int).tolist()]
        else:
            permutations = []
            for sample_index in range(sample_count):
                valid_indices = np.flatnonzero(mask[sample_index])
                source = valid_indices[_nonidentity_permutation(rng, len(valid_indices))]
                permuted[name][sample_index, valid_indices] = arrays[name][sample_index, source]
                permutations.append(source.astype(int).tolist())
        predictions = _predict(prediction_callback, permuted, mask, sample_count)
        if predictions.shape != baseline_predictions.shape:
            raise ValueError("prediction callback output shape changed after permutation")
        permuted_metric = _metric(metric_callback, predictions)
        selected = _selected_indices(predictions, mask)
        valid_selection_rows = mask.any(axis=1)
        selection_change = (
            None
            if baseline_selected is None or selected is None or not valid_selection_rows.any()
            else float(np.mean(selected[valid_selection_rows] != baseline_selected[valid_selection_rows]))
        )
        difference = np.abs(predictions - baseline_predictions)
        rows.append(
            {
                "feature_group": name,
                "permutation_semantics": semantic,
                "baseline_metric": baseline_metric,
                "permuted_metric": permuted_metric,
                "delta": permuted_metric - baseline_metric,
                "performance_drop": baseline_metric - permuted_metric,
                "mean_absolute_prediction_change": float(difference.mean()),
                "max_absolute_prediction_change": float(difference.max()),
                "selection_change_rate": selection_change,
                "permutation": permutations,
            }
        )
    return {
        "kind": "evaluation_only_feature_group_permutation_sensitivity",
        "seed": int(seed),
        "sample_count": sample_count,
        "candidate_count": candidate_count,
        "baseline_metric": baseline_metric,
        "groups": rows,
        "analysis_metadata_passed_to_prediction": False,
        "correctness_passed_to_prediction": False,
    }


def _metric_names(
    reference: Mapping[str, Any],
    others: Sequence[Mapping[str, Any]],
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested is None:
        common = set(reference)
        for value in others:
            common &= set(value)
        names = tuple(sorted(name for name in common if _is_finite_number(reference[name]) and all(_is_finite_number(value[name]) for value in others)))
    else:
        names = tuple(map(str, requested))
    if not names or len(names) != len(set(names)):
        raise ValueError("metric_names must select at least one unique numeric metric")
    for name in names:
        if name not in reference or not _is_finite_number(reference[name]):
            raise ValueError(f"reference metric {name!r} is missing or non-finite")
        if any(name not in value or not _is_finite_number(value[name]) for value in others):
            raise ValueError(f"comparison metric {name!r} is missing or non-finite")
    return names


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def leave_one_group_out_table(
    *,
    full_metrics: Mapping[str, Any],
    leave_one_out_results: Mapping[str, Mapping[str, Any]],
    metric_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Arrange full and leave-one-feature-group-out results as a wide table."""
    if not leave_one_out_results:
        raise ValueError("leave_one_out_results must not be empty")
    normalized_results = {str(name): value for name, value in leave_one_out_results.items()}
    if len(normalized_results) != len(leave_one_out_results):
        raise ValueError("leave-one-out group names collide after string normalization")
    omissions = sorted(normalized_results)
    results = [normalized_results[name] for name in omissions]
    metrics = _metric_names(full_metrics, results, metric_names)
    rows = [{"configuration": "all_groups", "omitted_group": None}]
    rows[0].update({name: float(full_metrics[name]) for name in metrics})
    rows[0].update({f"delta_vs_full_{name}": 0.0 for name in metrics})
    for omitted, result in zip(omissions, results, strict=True):
        row: dict[str, Any] = {"configuration": f"without_{omitted}", "omitted_group": omitted}
        for name in metrics:
            row[name] = float(result[name])
            row[f"delta_vs_full_{name}"] = float(result[name]) - float(full_metrics[name])
        rows.append(row)
    return rows


def native_enhancement_comparison_table(
    *,
    variant_results: Mapping[str, Mapping[str, Any]],
    metric_names: Sequence[str] | None = None,
    required_variants: Sequence[str] = NATIVE_COMPARISON_VARIANTS,
) -> list[dict[str, Any]]:
    """Build the native-vs-RGBD/latent/text/gate/uncertainty comparison."""
    required = tuple(map(str, required_variants))
    if not required or required[0] != "native" or len(required) != len(set(required)):
        raise ValueError("required_variants must be unique and begin with 'native'")
    normalized_results = {str(name): value for name, value in variant_results.items()}
    if len(normalized_results) != len(variant_results):
        raise ValueError("variant names collide after string normalization")
    missing = set(required) - set(normalized_results)
    if missing:
        raise ValueError(f"missing comparison variants: {sorted(missing)}")
    native = normalized_results["native"]
    comparisons = [normalized_results[name] for name in required[1:]]
    metrics = _metric_names(native, comparisons, metric_names)
    rows = []
    for variant in required:
        result = normalized_results[variant]
        row: dict[str, Any] = {
            "variant": variant,
            "reference": "native",
            "is_native": variant == "native",
        }
        for name in metrics:
            row[name] = float(result[name])
            row[f"delta_vs_native_{name}"] = float(result[name]) - float(native[name])
        rows.append(row)
    return rows


def _machine_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_machine_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _machine_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _machine_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_machine_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("machine-readable output cannot contain NaN or infinity")
        return value
    raise TypeError(f"unsupported machine-readable value: {type(value).__name__}")


def write_analysis_json(
    path: str | Path,
    payload: Any,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a JSON-safe analysis payload."""
    return atomic_write_json(path, _machine_value(payload), overwrite=overwrite)


def _csv_cell(value: Any) -> str | int | float:
    value = _machine_value(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return value


def write_analysis_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write heterogeneous analysis rows as a stable CSV table."""
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    field_names: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise TypeError("CSV rows must be mappings")
        row = {str(key): _csv_cell(value) for key, value in raw_row.items()}
        for name in row:
            if name not in field_names:
                field_names.append(name)
        normalized_rows.append(row)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{hashlib.sha256(str(output).encode()).hexdigest()[:8]}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_names, extrasaction="raise")
            if field_names:
                writer.writeheader()
                writer.writerows(normalized_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = (
    "MISSING_CATEGORY",
    "NATIVE_COMPARISON_VARIANTS",
    "OUTCOME_ORDER",
    "feature_group_permutation_sensitivity",
    "leave_one_group_out_table",
    "native_enhancement_comparison_table",
    "recovered_harmful_distribution",
    "subgroup_metrics",
    "write_analysis_csv",
    "write_analysis_json",
)
