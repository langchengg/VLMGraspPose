"""Pure statistical helpers for paired candidate re-ranking evaluation.

The functions in this module do not read or write artifacts.  They accept
array-like values and return ordinary dictionaries/arrays so that a caller can
decide how (or whether) results are persisted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, overload

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar


_K = TypeVar("_K")


def _binary_vector(values: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
    """Return a one-dimensional boolean vector, rejecting non-binary input."""

    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype.kind in "biuf":
        numeric = raw.astype(np.float64, copy=False)
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
            raise ValueError(f"{name} must contain only finite binary values")
        return numeric.astype(bool)
    normalized: list[bool] = []
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)):
            normalized.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            normalized.append(bool(value))
        else:
            raise ValueError(f"{name} must contain only binary values")
    return np.asarray(normalized, dtype=bool)


def _paired_binary(
    reference_correct: Sequence[Any] | np.ndarray,
    challenger_correct: Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = _binary_vector(reference_correct, "reference_correct")
    challenger = _binary_vector(challenger_correct, "challenger_correct")
    if reference.shape != challenger.shape:
        raise ValueError("paired outcome vectors differ in length")
    if reference.size == 0:
        raise ValueError("paired outcome vectors are empty")
    return reference, challenger


def _conditional_odds_interval(
    recovered: int,
    harmful: int,
    *,
    confidence: float = 0.95,
) -> list[float]:
    """Clopper-Pearson interval transformed to matched conditional odds."""

    discordant = recovered + harmful
    if discordant == 0:
        return [0.0, math.inf]
    interval = binomtest(recovered, discordant, p=0.5).proportion_ci(
        confidence_level=float(confidence), method="exact"
    )

    def odds(probability: float) -> float:
        if probability <= 0.0:
            return 0.0
        if probability >= 1.0:
            return math.inf
        return float(probability / (1.0 - probability))

    return [odds(float(interval.low)), odds(float(interval.high))]


def mcnemar_exact(
    reference_correct: Sequence[Any] | np.ndarray,
    challenger_correct: Sequence[Any] | np.ndarray,
    *,
    confidence: float = 0.95,
    cross_check: bool = True,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compute the exact paired McNemar test and matched odds ratio.

    SciPy's exact two-sided binomial test is the primary implementation.
    When ``cross_check`` is true, statsmodels independently evaluates the
    corresponding 2x2 table and disagreement raises ``AssertionError``.
    The matched odds ratio is challenger-only successes divided by
    reference-only successes.  It is one when there are no discordant pairs
    and infinity when only challenger-only successes are observed.
    """

    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    reference, challenger = _paired_binary(reference_correct, challenger_correct)
    both_wrong = int(((~reference) & (~challenger)).sum())
    recovered = int(((~reference) & challenger).sum())
    harmful = int((reference & (~challenger)).sum())
    both_correct = int((reference & challenger).sum())
    discordant = recovered + harmful
    scipy_p = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                recovered,
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    table = np.asarray(
        [[both_wrong, recovered], [harmful, both_correct]], dtype=np.int64
    )
    statsmodels_result = sm_mcnemar(table, exact=True, correction=False)
    statsmodels_p = float(statsmodels_result.pvalue)
    if cross_check and not math.isclose(
        scipy_p, statsmodels_p, rel_tol=0.0, abs_tol=float(atol)
    ):
        raise AssertionError(
            "SciPy and statsmodels exact McNemar p-values disagree: "
            f"{scipy_p} != {statsmodels_p}"
        )
    if discordant == 0:
        odds_ratio = 1.0
    elif harmful == 0:
        odds_ratio = math.inf
    else:
        odds_ratio = float(recovered / harmful)
    return {
        "sample_count": int(reference.size),
        "table": table.tolist(),
        "both_wrong": both_wrong,
        "recovered": recovered,
        "harmful": harmful,
        "both_correct": both_correct,
        "discordant": discordant,
        "net_recovered": recovered - harmful,
        "effect": float(challenger.mean() - reference.mean()),
        "pvalue": scipy_p,
        "raw_p": scipy_p,
        "scipy_exact_pvalue": scipy_p,
        "statsmodels_exact_pvalue": statsmodels_p,
        "cross_check_passed": math.isclose(
            scipy_p, statsmodels_p, rel_tol=0.0, abs_tol=float(atol)
        ),
        "matched_odds_ratio": odds_ratio,
        "matched_odds_ratio_ci": _conditional_odds_interval(
            recovered, harmful, confidence=float(confidence)
        ),
        "confidence": float(confidence),
    }


def cluster_bootstrap_difference(
    reference_correct: Sequence[Any] | np.ndarray,
    challenger_correct: Sequence[Any] | np.ndarray,
    clusters: Sequence[Any] | np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = 20260801,
    confidence: float = 0.95,
    return_distribution: bool = False,
) -> dict[str, Any]:
    """Percentile CI for a paired mean difference, resampling whole clusters.

    Clusters are sampled with replacement and every member of a sampled
    cluster is retained.  Consequently unequal cluster sizes retain their
    natural sample-level weighting within every bootstrap replicate.
    """

    reference, challenger = _paired_binary(reference_correct, challenger_correct)
    raw_clusters = np.asarray(clusters, dtype=object)
    if raw_clusters.ndim != 1 or raw_clusters.shape[0] != reference.shape[0]:
        raise ValueError("clusters must be one-dimensional and match outcomes")
    if any(
        value is None
        or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))
        or str(value) == ""
        for value in raw_clusters.tolist()
    ):
        raise ValueError("clusters contain null or empty values")
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    cluster_values, inverse = np.unique(raw_clusters.astype(str), return_inverse=True)
    if cluster_values.size == 0:
        raise ValueError("at least one cluster is required")
    difference = challenger.astype(np.float64) - reference.astype(np.float64)
    cluster_sums = np.bincount(
        inverse, weights=difference, minlength=cluster_values.size
    )
    cluster_counts = np.bincount(inverse, minlength=cluster_values.size)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(iterations, dtype=np.float64)
    chunk_size = min(1_000, iterations)
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        draws = rng.integers(
            0,
            cluster_values.size,
            size=(stop - start, cluster_values.size),
        )
        bootstrap[start:stop] = cluster_sums[draws].sum(axis=1) / cluster_counts[
            draws
        ].sum(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    result: dict[str, Any] = {
        "point_estimate": float(difference.mean()),
        "ci": [
            float(np.quantile(bootstrap, alpha)),
            float(np.quantile(bootstrap, 1.0 - alpha)),
        ],
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "confidence": float(confidence),
        "iterations": iterations,
        "seed": int(seed),
        "sample_count": int(reference.size),
        "group_count": int(cluster_values.size),
        "cluster_count": int(cluster_values.size),
        "resampling_unit": "cluster",
    }
    if return_distribution:
        result["distribution"] = bootstrap
    return result


def paired_cluster_statistics(
    reference_correct: Sequence[Any] | np.ndarray,
    challenger_correct: Sequence[Any] | np.ndarray,
    *,
    scene_ids: Sequence[Any] | np.ndarray,
    frame_ids: Sequence[Any] | np.ndarray,
    iterations: int = 10_000,
    seed: int = 20260801,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return exact McNemar plus scene- and frame-cluster bootstrap CIs."""

    return {
        "mcnemar": mcnemar_exact(reference_correct, challenger_correct),
        "scene_bootstrap": cluster_bootstrap_difference(
            reference_correct,
            challenger_correct,
            scene_ids,
            iterations=iterations,
            seed=seed,
            confidence=confidence,
        ),
        "frame_bootstrap": cluster_bootstrap_difference(
            reference_correct,
            challenger_correct,
            frame_ids,
            iterations=iterations,
            seed=seed + 1,
            confidence=confidence,
        ),
    }


@overload
def holm_adjust(p_values: Mapping[_K, float]) -> dict[_K, float]: ...


@overload
def holm_adjust(p_values: Sequence[float] | np.ndarray) -> np.ndarray: ...


def holm_adjust(
    p_values: Mapping[_K, float] | Sequence[float] | np.ndarray,
) -> dict[_K, float] | np.ndarray:
    """Holm-Bonferroni adjustment with monotonic step-down correction."""

    is_mapping = isinstance(p_values, Mapping)
    if is_mapping:
        keys = list(p_values)
        values = np.asarray([p_values[key] for key in keys], dtype=np.float64)
    else:
        values = np.asarray(p_values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("p_values must be one-dimensional")
        keys = list(range(values.size))
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        candidate = min(1.0, float((values.size - position) * values[index]))
        running = max(running, candidate)
        adjusted_sorted[position] = running
    adjusted = np.empty(values.size, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    if is_mapping:
        return {key: float(adjusted[index]) for index, key in enumerate(keys)}
    return adjusted


def expected_calibration_error(
    labels: Sequence[Any] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, Any]:
    """Equal-width binary ECE and reliability-bin details."""

    truth = _binary_vector(labels, "labels").astype(np.float64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 1 or probability.shape != truth.shape:
        raise ValueError("probabilities must be one-dimensional and match labels")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    bins = int(bins)
    if bins < 1:
        raise ValueError("bins must be positive")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    reliability: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        members = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        count = int(members.sum())
        if count == 0:
            reliability.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                }
            )
            continue
        confidence_value = float(probability[members].mean())
        accuracy = float(truth[members].mean())
        ece += (count / truth.size) * abs(confidence_value - accuracy)
        reliability.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "confidence": confidence_value,
                "accuracy": accuracy,
            }
        )
    return {"ece": float(ece), "bins": reliability, "bin_count": bins}


def candidate_classification_metrics(
    labels: Sequence[Any] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    probabilities: Sequence[float] | np.ndarray | None = None,
    bins: int = 15,
) -> dict[str, Any]:
    """Candidate-level ROC-AUC, PR-AUC/AP, Brier score and ECE.

    ROC-AUC is undefined and reported as ``None`` when only one class is
    present.  PR-AUC/AP is zero when there are no positive candidates.
    Brier/ECE require calibrated probabilities and are ``None`` when they are
    not supplied.
    """

    truth = _binary_vector(labels, "labels")
    score = np.asarray(scores, dtype=np.float64)
    if score.ndim != 1 or score.shape != truth.shape:
        raise ValueError("scores must be one-dimensional and match labels")
    if not np.isfinite(score).all():
        raise ValueError("scores must be finite")
    positive_count = int(truth.sum())
    negative_count = int(truth.size - positive_count)
    roc_auc = (
        None
        if positive_count == 0 or negative_count == 0
        else float(roc_auc_score(truth, score))
    )
    pr_auc = (
        0.0
        if positive_count == 0
        else float(average_precision_score(truth, score))
    )
    result: dict[str, Any] = {
        "candidate_count": int(truth.size),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "average_precision": pr_auc,
        "brier": None,
        "brier_score": None,
        "ece": None,
        "reliability": None,
    }
    if probabilities is not None:
        probability = np.asarray(probabilities, dtype=np.float64)
        if probability.ndim != 1 or probability.shape != truth.shape:
            raise ValueError("probabilities must be one-dimensional and match labels")
        if not np.isfinite(probability).all() or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError("probabilities must be finite and lie in [0, 1]")
        brier = float(np.mean(np.square(probability - truth.astype(np.float64))))
        calibration = expected_calibration_error(truth, probability, bins=bins)
        result.update(
            {
                "brier": brier,
                "brier_score": brier,
                "ece": calibration["ece"],
                "reliability": calibration["bins"],
                "ece_bin_count": calibration["bin_count"],
            }
        )
    return result


__all__ = [
    "candidate_classification_metrics",
    "cluster_bootstrap_difference",
    "expected_calibration_error",
    "holm_adjust",
    "mcnemar_exact",
    "paired_cluster_statistics",
]
