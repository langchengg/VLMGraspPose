from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(reference_correct: np.ndarray, challenger_correct: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference_correct, dtype=bool)
    challenger = np.asarray(challenger_correct, dtype=bool)
    recovered = int(((~reference) & challenger).sum())
    harmful = int((reference & (~challenger)).sum())
    discordant = recovered + harmful
    p_value = 1.0 if discordant == 0 else float(binomtest(min(recovered, harmful), discordant, 0.5, alternative="two-sided").pvalue)
    return {"recovered": recovered, "harmful": harmful, "discordant": discordant, "effect": float(challenger.mean() - reference.mean()), "raw_p": p_value}


def cluster_bootstrap_difference(
    reference_correct: np.ndarray,
    challenger_correct: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = 20260801,
) -> dict[str, Any]:
    reference = np.asarray(reference_correct, dtype=np.float64)
    challenger = np.asarray(challenger_correct, dtype=np.float64)
    group_values, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    if len(reference) != len(challenger) or len(reference) != len(inverse):
        raise ValueError("cluster bootstrap inputs differ in length")
    delta = challenger - reference
    sums = np.bincount(inverse, weights=delta, minlength=len(group_values))
    counts = np.bincount(inverse, minlength=len(group_values))
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(iterations), dtype=np.float64)
    chunk = 1000
    for start in range(0, int(iterations), chunk):
        size = min(chunk, int(iterations) - start)
        draw = rng.integers(0, len(group_values), size=(size, len(group_values)))
        values[start:start + size] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return {
        "point_estimate": float(delta.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "iterations": int(iterations),
        "seed": int(seed),
        "sample_count": len(reference),
        "group_count": len(group_values),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(((name, float(value)) for name, value in p_values.items()), key=lambda value: value[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def paired_cluster_statistics(
    reference_correct: np.ndarray,
    challenger_correct: np.ndarray,
    *,
    frame_ids: np.ndarray,
    scene_ids: np.ndarray,
    iterations: int = 10_000,
    seed: int = 20260801,
) -> dict[str, Any]:
    return {
        "mcnemar": mcnemar_exact(reference_correct, challenger_correct),
        "frame_bootstrap": cluster_bootstrap_difference(reference_correct, challenger_correct, frame_ids, iterations=iterations, seed=seed),
        "scene_bootstrap": cluster_bootstrap_difference(reference_correct, challenger_correct, scene_ids, iterations=iterations, seed=seed + 1),
    }
