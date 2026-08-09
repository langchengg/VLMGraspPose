from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.stats import binomtest


def exact_mcnemar_pvalue(baseline: Iterable[bool], method: Iterable[bool]) -> float:
    left = np.asarray(list(baseline), dtype=bool)
    right = np.asarray(list(method), dtype=bool)
    if left.shape != right.shape:
        raise ValueError("paired outcomes must have equal shape")
    recovered = int((~left & right).sum())
    harmful = int((left & ~right).sum())
    discordant = recovered + harmful
    return 1.0 if discordant == 0 else float(binomtest(min(recovered, harmful), discordant, 0.5).pvalue)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def clustered_bootstrap_delta(
    *,
    baseline: Iterable[bool],
    method: Iterable[bool],
    groups: Iterable[str],
    draws: int = 10_000,
    seed: int = 47,
) -> dict[str, float | int]:
    baseline_values = np.asarray(list(baseline), dtype=np.float64)
    method_values = np.asarray(list(method), dtype=np.float64)
    group_values = np.asarray(list(groups), dtype=object)
    if baseline_values.shape != method_values.shape or baseline_values.shape != group_values.shape:
        raise ValueError("outcomes and cluster IDs must be paired")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_values):
        by_group[str(group)].append(index)
    group_keys = sorted(by_group)
    if not group_keys:
        raise ValueError("no clusters")
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        sampled = rng.choice(group_keys, size=len(group_keys), replace=True)
        indices = np.concatenate([np.asarray(by_group[group], dtype=np.int64) for group in sampled])
        deltas[draw] = 100.0 * float((method_values[indices] - baseline_values[indices]).mean())
    return {
        "draws": int(draws),
        "seed": int(seed),
        "sample_count": int(len(baseline_values)),
        "cluster_count": int(len(group_keys)),
        "delta_pp": 100.0 * float((method_values - baseline_values).mean()),
        "ci_low_pp": float(np.quantile(deltas, 0.025)),
        "ci_high_pp": float(np.quantile(deltas, 0.975)),
    }
