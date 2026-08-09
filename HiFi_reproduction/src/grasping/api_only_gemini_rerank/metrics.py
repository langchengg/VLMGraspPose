"""Full-denominator offline 2D grasp-rectangle consistency statistics."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest


def outcome_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    n = len(values)
    baseline_correct = np.asarray([bool(row["baseline_correct"]) for row in values], dtype=bool)
    final_correct = np.asarray([bool(row["final_correct"]) for row in values], dtype=bool)
    switched = np.asarray([str(row.get("baseline_candidate_id")) != str(row.get("final_candidate_id")) for row in values], dtype=bool)
    recovered = (~baseline_correct) & final_correct
    harmful = baseline_correct & (~final_correct)
    recoverable = np.asarray([bool(row.get("recoverable_error", False)) for row in values], dtype=bool)
    r, h = int(recovered.sum()), int(harmful.sum())
    return {
        "N_total": n,
        "baseline_correct": int(baseline_correct.sum()), "final_correct": int(final_correct.sum()),
        "baseline_j_at_1": float(baseline_correct.mean()) if n else None,
        "final_j_at_1": float(final_correct.mean()) if n else None,
        "recovered": r, "harmful": h, "net": r - h,
        "delta_j_at_1": float((r-h)/n) if n else None,
        "neutral_correct_to_correct": int((baseline_correct & final_correct).sum()),
        "neutral_wrong_to_wrong": int(((~baseline_correct) & (~final_correct)).sum()),
        "switch_count": int(switched.sum()), "switch_rate": float(switched.mean()) if n else None,
        "recoverable_errors": int(recoverable.sum()),
        "recovery_recall": float(r/recoverable.sum()) if recoverable.sum() else None,
        "harm_rate": float(h/baseline_correct.sum()) if baseline_correct.sum() else None,
        "outcome_changing_precision": float(r/(r+h)) if r+h else None,
    }


def mcnemar_exact(recovered: int, harmful: int) -> dict[str, Any]:
    discordant = int(recovered) + int(harmful)
    p = 1.0 if discordant == 0 else float(binomtest(min(int(recovered), int(harmful)), discordant, 0.5, alternative="two-sided").pvalue)
    return {"recovered": int(recovered), "harmful": int(harmful), "discordant": discordant, "two_sided_exact_p": p}


def scene_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]], *, draws: int = 10_000, seed: int = 20260805,
) -> dict[str, Any]:
    if not rows:
        return {"draws": draws, "seed": seed, "delta_mean": None, "ci95": [None, None]}
    scenes: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        scenes.setdefault(str(row["scene_id"]), []).append(row)
    keys = sorted(scenes)
    deltas = np.empty(draws, dtype=float)
    rng = np.random.default_rng(seed)
    for index in range(draws):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        r = h = n = 0
        for scene in sampled:
            group = scenes[str(scene)]
            n += len(group)
            r += sum(not bool(row["baseline_correct"]) and bool(row["final_correct"]) for row in group)
            h += sum(bool(row["baseline_correct"]) and not bool(row["final_correct"]) for row in group)
        deltas[index] = (r-h)/n
    return {
        "draws": int(draws), "seed": int(seed), "scene_count": len(keys),
        "delta_mean": float(deltas.mean()),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
    }


def go_no_go(metrics: Mapping[str, Any], bootstrap: Mapping[str, Any], *, stability_pass: bool,
             invariants_pass: bool, leakage_pass: bool) -> str:
    ci_low = bootstrap.get("ci95", [None])[0]
    if metrics.get("net", 0) <= 0 or metrics.get("recovered", 0) <= metrics.get("harmful", 0):
        return "NO_GO"
    if ci_low is None or float(ci_low) < 0:
        return "INCONCLUSIVE"
    precision = metrics.get("outcome_changing_precision")
    if not (
        float(metrics.get("final_j_at_1", 0)) > float(metrics.get("baseline_j_at_1", 0))
        and float(metrics.get("harm_rate", 1)) <= 0.01
        and precision is not None and float(precision) >= 0.67
        and stability_pass and invariants_pass and leakage_pass
    ):
        return "NO_GO"
    return "GO"
