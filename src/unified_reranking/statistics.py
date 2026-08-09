"""Predeclared paired inference with cluster-aware primary uncertainty."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260808


def _paired_binary(
    reference_correct: Sequence[Any], challenger_correct: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_correct)
    challenger = np.asarray(challenger_correct)
    if reference.ndim != 1 or reference.shape != challenger.shape or not len(reference):
        raise ValueError("paired outcomes must be non-empty equal one-dimensional vectors")
    for name, values in (("reference", reference), ("challenger", challenger)):
        numeric = np.asarray(values, dtype=float)
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0, 1]).all():
            raise ValueError(f"{name} outcomes must be finite binary values")
    return reference.astype(bool), challenger.astype(bool)


def mcnemar_exact(
    reference_correct: Sequence[Any], challenger_correct: Sequence[Any]
) -> dict[str, Any]:
    """Exact two-sided paired McNemar support with explicit discordant cells."""

    reference, challenger = _paired_binary(reference_correct, challenger_correct)
    recovered = int((~reference & challenger).sum())
    harmful = int((reference & ~challenger).sum())
    discordant = recovered + harmful
    pvalue = (
        1.0
        if discordant == 0
        else float(binomtest(recovered, discordant, p=0.5).pvalue)
    )
    return {
        "sample_count": len(reference),
        "both_wrong": int((~reference & ~challenger).sum()),
        "recovered": recovered,
        "harmful": harmful,
        "both_correct": int((reference & challenger).sum()),
        "discordant": discordant,
        "net_recovered": recovered - harmful,
        "effect": float(challenger.mean() - reference.mean()),
        "pvalue": pvalue,
        "raw_p": pvalue,
        "method": "exact two-sided binomial McNemar",
    }


def cluster_bootstrap_difference(
    reference_correct: Sequence[Any],
    challenger_correct: Sequence[Any],
    clusters: Sequence[Any],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
    return_distribution: bool = False,
) -> dict[str, Any]:
    """Paired percentile CI formed by resampling whole clusters."""

    reference, challenger = _paired_binary(reference_correct, challenger_correct)
    raw_clusters = np.asarray(clusters, dtype=object)
    if raw_clusters.ndim != 1 or len(raw_clusters) != len(reference):
        raise ValueError("clusters must be one-dimensional and match outcomes")
    if any(
        value is None
        or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))
        or not str(value)
        for value in raw_clusters.tolist()
    ):
        raise ValueError("clusters contain null or empty values")
    if int(iterations) <= 0 or not 0 < float(confidence) < 1:
        raise ValueError("iterations and confidence are invalid")
    cluster_values, inverse = np.unique(raw_clusters.astype(str), return_inverse=True)
    difference = challenger.astype(float) - reference.astype(float)
    cluster_sums = np.bincount(inverse, weights=difference, minlength=len(cluster_values))
    cluster_counts = np.bincount(inverse, minlength=len(cluster_values))
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(iterations), dtype=float)
    for start in range(0, int(iterations), 1_000):
        stop = min(start + 1_000, int(iterations))
        draws = rng.integers(0, len(cluster_values), size=(stop - start, len(cluster_values)))
        bootstrap[start:stop] = cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)
    alpha = (1 - float(confidence)) / 2
    result: dict[str, Any] = {
        "point_estimate": float(difference.mean()),
        "ci": [float(np.quantile(bootstrap, alpha)), float(np.quantile(bootstrap, 1 - alpha))],
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "confidence": float(confidence),
        "iterations": int(iterations),
        "seed": int(seed),
        "sample_count": len(reference),
        "cluster_count": len(cluster_values),
        "resampling_unit": "cluster",
    }
    if return_distribution:
        result["distribution"] = bootstrap
    return result


def paired_system_statistics(
    reference_correct: Sequence[Any],
    challenger_correct: Sequence[Any],
    *,
    scene_ids: Sequence[Any],
    frame_ids: Sequence[Any],
) -> dict[str, Any]:
    """Return scene-bootstrap primary and conventional McNemar support.

    Repeated language queries within one visual scene are dependent.  Exact
    sample-level McNemar is therefore retained as a conventional/supportive
    test, while the scene-clustered bootstrap is the primary uncertainty check.
    """

    reference = np.asarray(reference_correct, dtype=bool)
    challenger = np.asarray(challenger_correct, dtype=bool)
    if reference.shape != challenger.shape or reference.ndim != 1:
        raise ValueError("paired outcomes must be equal one-dimensional vectors")
    return {
        "primary_uncertainty": "scene_clustered_bootstrap",
        "scene_bootstrap": cluster_bootstrap_difference(
            reference,
            challenger,
            scene_ids,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED,
        ),
        "frame_bootstrap_sensitivity": cluster_bootstrap_difference(
            reference,
            challenger,
            frame_ids,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED,
        ),
        "mcnemar_conventional_supportive": mcnemar_exact(reference, challenger),
        "dependence_disclosure": (
            "Sample-level McNemar treats paired rows as independent; repeated language "
            "queries share visual scenes/frames, so it is supportive rather than the "
            "primary uncertainty analysis."
        ),
    }


def three_system_paired_tests(outcomes: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    """Cochran Q followed by three exact McNemar tests with Holm correction."""

    names = tuple(outcomes)
    if len(names) != 3:
        raise ValueError("exactly three named systems are required")
    matrix = np.column_stack([np.asarray(outcomes[name], dtype=int) for name in names])
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isin(matrix, [0, 1]).all():
        raise ValueError("system outcomes must be non-empty binary vectors of equal length")
    # statsmodels divides by zero when every paired system is identical.  This
    # is the exact null/degenerate case, so report Q=0 and p=1 explicitly.
    identical_systems = bool(np.all(matrix == matrix[:, [0]]))
    omnibus = None if identical_systems else cochrans_q(matrix)
    pairs: list[dict[str, Any]] = []
    for first_index in range(3):
        for second_index in range(first_index + 1, 3):
            first, second = names[first_index], names[second_index]
            test = mcnemar_exact(matrix[:, first_index], matrix[:, second_index])
            pairs.append({"first": first, "second": second, **test})
    adjusted = multipletests([row["pvalue"] for row in pairs], method="holm")[1]
    for row, pvalue in zip(pairs, adjusted):
        row["holm_adjusted_pvalue"] = float(pvalue)
    return {
        "cochran_q": {
            "statistic": 0.0 if omnibus is None else float(omnibus.statistic),
            "pvalue": 1.0 if omnibus is None else float(omnibus.pvalue),
            "df": int(len(names) - 1),
            "degenerate_identical_systems": identical_systems,
            "method_note": "conventional/supportive because sample rows are scene-clustered",
        },
        "pairwise_mcnemar_holm": pairs,
    }
