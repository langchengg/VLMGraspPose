from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import binomtest

from .datasets import JoinedSample


def _safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _ndcg(labels: list[bool]) -> float:
    gains = np.asarray(labels, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(len(gains)) + 2.0)
    dcg = float(np.sum(gains * discounts))
    ideal = float(np.sum(np.sort(gains)[::-1] * discounts))
    return dcg / ideal if ideal else 0.0


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> tuple[float, list[dict[str, Any]]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows, value = [], 0.0
    for index in range(int(bins)):
        if index + 1 == bins:
            keep = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            keep = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        count = int(keep.sum())
        confidence = float(probabilities[keep].mean()) if count else None
        accuracy = float(labels[keep].mean()) if count else None
        if count:
            value += count / len(probabilities) * abs(confidence - accuracy)
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": confidence,
                "empirical_accuracy": accuracy,
            }
        )
    return float(value), rows


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: list[str],
    *,
    iterations: int = 10_000,
    seed: int = 7301,
) -> list[float]:
    grouped = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[str(cluster)].append(index)
    keys = sorted(grouped)
    cluster_sums = np.asarray(
        [np.asarray(values[grouped[key]], dtype=np.float64).sum() for key in keys]
    )
    cluster_counts = np.asarray(
        [len(grouped[key]) for key in keys], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(iterations), dtype=np.float64)
    # Resampling aggregate cluster sums/counts is exactly equivalent to
    # materializing every selected expression, but avoids billions of Python
    # list operations for the 10k-iteration full-cohort report.
    for start in range(0, int(iterations), 1024):
        stop = min(int(iterations), start + 1024)
        sampled = rng.integers(
            0,
            len(keys),
            size=(stop - start, len(keys)),
        )
        draws[start:stop] = (
            cluster_sums[sampled].sum(axis=1)
            / cluster_counts[sampled].sum(axis=1)
        )
    return [
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    ]


def evaluate_rankings(
    samples: list[JoinedSample],
    rankings: dict[str, list[str]],
    *,
    candidate_probabilities: dict[str, np.ndarray] | None = None,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 7301,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    all_probabilities, all_candidate_labels = [], []
    for sample in samples:
        candidates = sample.feature["candidates"]
        original_order = [str(item["candidate_id"]) for item in candidates]
        order = list(map(str, rankings[sample.sample_id]))
        if len(order) != len(original_order) or set(order) != set(original_order):
            raise AssertionError("ranker changed the frozen candidate set")
        label_by_id = {
            str(item["candidate_id"]): bool(item["candidate_correct"])
            for item in sample.label["candidate_labels"]
        }
        original = label_by_id[original_order[0]]
        selected = label_by_id[order[0]]
        oracle_before = any(label_by_id[item] for item in original_order)
        oracle_after = any(label_by_id[item] for item in order)
        if oracle_before != oracle_after:
            raise AssertionError("Oracle@5 changed")
        ranked_labels = [label_by_id[item] for item in order]
        first_positive = next(
            (index + 1 for index, value in enumerate(ranked_labels) if value),
            None,
        )
        probabilities = None
        if candidate_probabilities is not None:
            probabilities = np.asarray(
                candidate_probabilities[sample.sample_id], dtype=np.float64
            )
            if probabilities.shape != (len(candidates),):
                raise ValueError("candidate probabilities must have K entries")
            probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
            all_probabilities.extend(probabilities.tolist())
            all_candidate_labels.extend(
                [float(label_by_id[item]) for item in original_order]
            )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "frame_id": sample.frame_id,
                "sequence_id": sample.sequence_id,
                "original_candidate_id": original_order[0],
                "selected_candidate_id": order[0],
                "switched": order[0] != original_order[0],
                "original_correct": original,
                "selected_correct": selected,
                "oracle_at_5": oracle_after,
                "recovered": (not original) and selected,
                "harmful": original and (not selected),
                "neutral_switch": order[0] != original_order[0]
                and original == selected,
                "first_positive_rank": first_positive,
                "reciprocal_rank": 0.0
                if first_positive is None
                else 1.0 / first_positive,
                "ndcg_at_5": _ndcg(ranked_labels),
            }
        )
    n = len(rows)
    original_count = sum(row["original_correct"] for row in rows)
    selected_count = sum(row["selected_correct"] for row in rows)
    oracle_count = sum(row["oracle_at_5"] for row in rows)
    recovered = sum(row["recovered"] for row in rows)
    harmful = sum(row["harmful"] for row in rows)
    if selected_count - original_count != recovered - harmful:
        raise AssertionError("delta does not equal recovered minus harmful")
    discordant = recovered + harmful
    deltas = np.asarray(
        [
            int(row["selected_correct"]) - int(row["original_correct"])
            for row in rows
        ],
        dtype=np.float64,
    )
    outcome_changing_precision = _safe_div(recovered, discordant)
    summary = {
        "sample_count": n,
        "candidate_count": n * 5,
        "legacy_or_corrected_j1": selected_count / n,
        "q_only_j1": original_count / n,
        "delta_j1_percentage_points": 100.0 * (selected_count - original_count) / n,
        "oracle_at_5": oracle_count / n,
        "oracle_success_count": oracle_count,
        "recovered": recovered,
        "harmful": harmful,
        "net_recovered": recovered - harmful,
        "neutral_switch": sum(row["neutral_switch"] for row in rows),
        "switch_coverage": sum(row["switched"] for row in rows) / n,
        "outcome_changing_switch_precision": outcome_changing_precision,
        "mrr_at_5": float(np.mean([row["reciprocal_rank"] for row in rows])),
        "ndcg_at_5": float(np.mean([row["ndcg_at_5"] for row in rows])),
        "mcnemar_exact_two_sided_pvalue": (
            float(binomtest(recovered, discordant, p=0.5).pvalue)
            if discordant
            else 1.0
        ),
        "frame_cluster_bootstrap_delta_95ci": [
            100.0 * value
            for value in cluster_bootstrap_ci(
                deltas,
                [row["frame_id"] for row in rows],
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
        ],
        "sequence_cluster_bootstrap_delta_95ci": [
            100.0 * value
            for value in cluster_bootstrap_ci(
                deltas,
                [row["sequence_id"] for row in rows],
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + 1,
            )
        ],
    }
    if candidate_probabilities is not None:
        probabilities = np.asarray(all_probabilities, dtype=np.float64)
        labels = np.asarray(all_candidate_labels, dtype=np.float64)
        ece, reliability = expected_calibration_error(probabilities, labels)
        summary.update(
            {
                "candidate_brier": float(np.mean((probabilities - labels) ** 2)),
                "candidate_nll": float(
                    -np.mean(
                        labels * np.log(probabilities)
                        + (1.0 - labels) * np.log(1.0 - probabilities)
                    )
                ),
                "candidate_ece": ece,
                "reliability": reliability,
            }
        )
    return summary, rows


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda name: (pvalues[name], name))
    count = len(ordered)
    adjusted = {}
    running = 0.0
    for index, name in enumerate(ordered):
        value = min(1.0, (count - index) * float(pvalues[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def risk_coverage_curve(
    rows: list[dict[str, Any]], confidence: dict[str, float]
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(confidence[row["sample_id"]]),
            row["sample_id"],
        ),
    )
    result = []
    for count in range(max(1, len(ordered) // 100), len(ordered) + 1, max(1, len(ordered) // 100)):
        kept = ordered[:count]
        errors = sum(not row["selected_correct"] for row in kept)
        result.append(
            {
                "coverage": count / len(ordered),
                "risk": errors / count,
                "count": count,
            }
        )
    if not result or result[-1]["count"] != len(ordered):
        errors = sum(not row["selected_correct"] for row in ordered)
        result.append({"coverage": 1.0, "risk": errors / len(ordered), "count": len(ordered)})
    return result
