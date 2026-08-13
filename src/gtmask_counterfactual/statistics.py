"""Paired descriptive counterfactual inference with cluster-aware intervals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260813
INFERENCE_LABEL = "descriptive counterfactual inference"


def _as_binary(values: Sequence[Any], *, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    numeric = np.asarray(result, dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0, 1]).all():
        raise ValueError(f"{name} must contain finite binary values")
    return numeric.astype(bool)


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _two_sided_binomial_half_p(successes: int, trials: int) -> tuple[float, float]:
    """Return exact doubled lower-tail probability and its natural logarithm."""

    if trials == 0:
        return 1.0, 0.0
    tail = min(int(successes), int(trials) - int(successes))
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        - trials * math.log(2.0)
        for index in range(tail + 1)
    ]
    log_p = min(0.0, math.log(2.0) + _logsumexp(log_terms))
    smallest = float(np.nextafter(0.0, 1.0))
    return max(smallest, min(1.0, math.exp(log_p))), log_p


def format_pvalue(value: float, *, log_value: float | None = None) -> str:
    """Never render an underflowed exact p-value as ``p=0.000000``."""

    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or numeric > 1:
        raise ValueError("p-value must be finite and lie in [0,1]")
    if log_value is not None and log_value < math.log(float(np.nextafter(0.0, 1.0))):
        exponent = log_value / math.log(10.0)
        return f"p < 1e{math.floor(exponent) + 1}"
    if numeric < 1e-6:
        return f"p={numeric:.3e}"
    return f"p={numeric:.6f}"


def exact_mcnemar(
    reference_correct: Sequence[Any], counterfactual_correct: Sequence[Any]
) -> dict[str, Any]:
    """Exact two-sided McNemar support via the discordant-cell binomial test."""

    reference = _as_binary(reference_correct, name="reference outcomes")
    counterfactual = _as_binary(
        counterfactual_correct, name="counterfactual outcomes"
    )
    if reference.shape != counterfactual.shape:
        raise ValueError("paired outcomes must have identical lengths")
    recovered = int((~reference & counterfactual).sum())
    harmful = int((reference & ~counterfactual).sum())
    discordant = recovered + harmful
    pvalue, log_pvalue = _two_sided_binomial_half_p(recovered, discordant)
    return {
        "N": len(reference),
        "both_negative": int((~reference & ~counterfactual).sum()),
        "b_reference_only": harmful,
        "c_counterfactual_only": recovered,
        "both_positive": int((reference & counterfactual).sum()),
        "discordant": discordant,
        "net_recovered": recovered - harmful,
        "reference_numerator": int(reference.sum()),
        "counterfactual_numerator": int(counterfactual.sum()),
        "reference_rate": float(reference.mean()),
        "counterfactual_rate": float(counterfactual.mean()),
        "delta": float(counterfactual.mean() - reference.mean()),
        "delta_percentage_points": float(
            100.0 * (counterfactual.mean() - reference.mean())
        ),
        "raw_p": pvalue,
        "raw_log_p": log_pvalue,
        "raw_p_display": format_pvalue(pvalue, log_value=log_pvalue),
        "method": "exact two-sided binomial McNemar",
        "inference_scope": INFERENCE_LABEL,
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down FWER adjustment, stable under tied p-values."""

    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Holm correction requires a non-empty p-value vector")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Holm p-values must be finite and lie in [0,1]")
    order = np.argsort(values, kind="mergesort")
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, float((len(values) - rank) * values[index]))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def align_paired_frames(
    reference: pd.DataFrame,
    counterfactual: pd.DataFrame,
    *,
    id_columns: Sequence[str] = ("sample_id",),
    outcome_column: str = "correct",
) -> pd.DataFrame:
    """Fail closed on duplicate/missing identities before a paired comparison."""

    keys = list(map(str, id_columns))
    required = set(keys) | {outcome_column}
    for name, frame in (("reference", reference), ("counterfactual", counterfactual)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} frame misses columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} frame has duplicate paired identities")
    left = reference[keys + [outcome_column]].copy()
    right = counterfactual[keys + [outcome_column]].copy()
    aligned = left.merge(
        right,
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
        suffixes=("_reference", "_counterfactual"),
        sort=True,
    )
    if not aligned["_merge"].eq("both").all():
        missing = aligned.loc[aligned["_merge"].ne("both"), keys + ["_merge"]]
        raise ValueError(f"paired identity sets differ: {missing.head().to_dict('records')}")
    return aligned.drop(columns="_merge")


def cluster_bootstrap_delta(
    reference_correct: Sequence[Any],
    counterfactual_correct: Sequence[Any],
    clusters: Sequence[Any],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
    return_distribution: bool = False,
) -> dict[str, Any]:
    """Paired percentile interval obtained by resampling intact clusters."""

    reference = _as_binary(reference_correct, name="reference outcomes")
    counterfactual = _as_binary(
        counterfactual_correct, name="counterfactual outcomes"
    )
    if reference.shape != counterfactual.shape:
        raise ValueError("paired outcomes must have identical lengths")
    cluster_values = np.asarray(clusters, dtype=object)
    if cluster_values.ndim != 1 or len(cluster_values) != len(reference):
        raise ValueError("cluster IDs must be one-dimensional and pair-aligned")
    if any(
        item is None
        or (isinstance(item, (float, np.floating)) and math.isnan(float(item)))
        or not str(item)
        for item in cluster_values.tolist()
    ):
        raise ValueError("cluster IDs must be non-empty and non-null")
    if int(iterations) <= 0 or not 0 < float(confidence) < 1:
        raise ValueError("bootstrap iterations/confidence are invalid")
    unique, inverse = np.unique(cluster_values.astype(str), return_inverse=True)
    difference = counterfactual.astype(float) - reference.astype(float)
    sums = np.bincount(inverse, weights=difference, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    random = np.random.default_rng(int(seed))
    distribution = np.empty(int(iterations), dtype=np.float64)
    for start in range(0, int(iterations), 1000):
        stop = min(start + 1000, int(iterations))
        draws = random.integers(
            0, len(unique), size=(stop - start, len(unique)), endpoint=False
        )
        distribution[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(
            axis=1
        )
    alpha = (1.0 - float(confidence)) / 2.0
    ci = [
        float(np.quantile(distribution, alpha)),
        float(np.quantile(distribution, 1.0 - alpha)),
    ]
    result: dict[str, Any] = {
        "point_estimate": float(difference.mean()),
        "delta_percentage_points": float(100.0 * difference.mean()),
        "ci": ci,
        "ci_percentage_points": [100.0 * ci[0], 100.0 * ci[1]],
        "confidence": float(confidence),
        "iterations": int(iterations),
        "seed": int(seed),
        "N": len(reference),
        "cluster_count": len(unique),
        "resampling_unit": "cluster",
        "inference_scope": INFERENCE_LABEL,
    }
    if return_distribution:
        result["distribution"] = distribution
    return result


def paired_metric_statistics(
    frame: pd.DataFrame,
    *,
    reference_column: str,
    counterfactual_column: str,
    scene_column: str = "scene_id",
    frame_column: str = "frame_id",
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute the preregistered sample/scene/frame paired summaries."""

    required = {
        "sample_id",
        reference_column,
        counterfactual_column,
        scene_column,
        frame_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"statistical input misses columns: {missing}")
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("statistical input must be unique by sample_id")
    reference = _as_binary(frame[reference_column], name=reference_column)
    counterfactual = _as_binary(
        frame[counterfactual_column], name=counterfactual_column
    )
    exact = exact_mcnemar(reference, counterfactual)
    return {
        **exact,
        "scene_cluster_bootstrap": cluster_bootstrap_delta(
            reference,
            counterfactual,
            frame[scene_column],
            iterations=iterations,
            seed=seed,
        ),
        "frame_cluster_bootstrap_sensitivity": cluster_bootstrap_delta(
            reference,
            counterfactual,
            frame[frame_column],
            iterations=iterations,
            seed=seed,
        ),
        "dependence_disclosure": (
            "Sample-level exact McNemar is supportive because repeated queries may "
            "share scenes/frames; the scene-cluster percentile interval is primary "
            "uncertainty and frame clustering is a sensitivity analysis."
        ),
    }


def apply_holm_family(
    comparisons: Sequence[Mapping[str, Any]], *, pvalue_key: str = "raw_p"
) -> list[dict[str, Any]]:
    """Apply Holm once across the declared route-by-metric comparison family."""

    if not comparisons:
        raise ValueError("Holm family must contain at least one comparison")
    rows = [dict(item) for item in comparisons]
    adjusted = holm_adjust([float(row[pvalue_key]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = float(value)
        row["holm_adjusted_p_display"] = format_pvalue(float(value))
        row["multiplicity_method"] = "Holm FWER correction"
        row["inference_scope"] = INFERENCE_LABEL
    return rows
