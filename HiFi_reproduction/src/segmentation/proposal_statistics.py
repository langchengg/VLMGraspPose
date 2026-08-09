"""Grouped uncertainty and paired tests for visual-grounding mask metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid Wilson count")
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z / denominator * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (float(max(0.0, centre - radius)), float(min(1.0, centre + radius)))


def clustered_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    statistic: Callable[[pd.DataFrame], float],
    replicates: int = 10000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    groups = {str(key): value for key, value in frame.groupby(cluster_column, sort=False)}
    keys = np.asarray(list(groups), dtype=object)
    if not len(keys):
        raise ValueError("cluster bootstrap has no groups")
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        replicate = pd.concat([groups[str(key)] for key in sampled], ignore_index=True)
        values[index] = float(statistic(replicate))
    if not np.isfinite(values).all():
        raise RuntimeError("cluster bootstrap produced non-finite values")
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper), values


def clustered_bootstrap_values(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    threshold: float | None = None,
    baseline_values: np.ndarray | None = None,
    replicates: int = 10000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Fast query-weighted cluster bootstrap for a mean, P@X, or paired delta."""

    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters).astype(str)
    if values.ndim != 1 or clusters.shape != values.shape or not len(values):
        raise ValueError("values and clusters must be aligned non-empty vectors")
    transformed = (values > float(threshold)).astype(np.float64) if threshold is not None else values
    if baseline_values is not None:
        baseline = np.asarray(baseline_values, dtype=np.float64)
        if baseline.shape != values.shape:
            raise ValueError("paired baseline values must be aligned")
        baseline = (
            (baseline > float(threshold)).astype(np.float64)
            if threshold is not None
            else baseline
        )
        transformed = transformed - baseline
    keys, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=transformed, minlength=len(keys))
    counts = np.bincount(inverse, minlength=len(keys)).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, len(keys), size=(int(replicates), len(keys)))
    estimates = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper), estimates


def paired_transitions(
    baseline_iou: np.ndarray,
    method_iou: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    baseline = np.asarray(baseline_iou, dtype=np.float64) > float(threshold)
    method = np.asarray(method_iou, dtype=np.float64) > float(threshold)
    if baseline.shape != method.shape:
        raise ValueError("paired transitions require aligned arrays")
    recovered = int(np.count_nonzero(~baseline & method))
    harmed = int(np.count_nonzero(baseline & ~method))
    changed = recovered + harmed
    return {
        "baseline_success": int(np.count_nonzero(baseline)),
        "method_success": int(np.count_nonzero(method)),
        "success_success": int(np.count_nonzero(baseline & method)),
        "success_failure": harmed,
        "failure_success": recovered,
        "failure_failure": int(np.count_nonzero(~baseline & ~method)),
        "recovered": recovered,
        "harmed": harmed,
        "net": recovered - harmed,
        "outcome_changing_precision": float(recovered / changed) if changed else 1.0,
        "mcnemar_exact_p": float(
            binomtest(recovered, n=changed, p=0.5, alternative="two-sided").pvalue
        )
        if changed
        else 1.0,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return [float(value) for value in adjusted]


__all__ = [
    "clustered_bootstrap",
    "clustered_bootstrap_values",
    "holm_adjust",
    "paired_transitions",
    "wilson_interval",
]
