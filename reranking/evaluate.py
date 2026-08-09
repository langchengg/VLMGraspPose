"""Strict, ID-joined evaluation for frozen-pool candidate re-ranking.

All public functions are pure with respect to the filesystem.  Labels are
always taken from the reference candidate pool and joined by query/candidate
ID; prediction row order is never treated as identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from reranking.statistics import (
    candidate_classification_metrics,
    holm_adjust,
    paired_cluster_statistics,
)


class EvaluationError(ValueError):
    """Raised when a frozen-pool evaluation invariant is violated."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise EvaluationError(f"{name} missing columns: {missing}")


def _normalise_ids(
    frame: pd.DataFrame,
    *,
    query_col: str,
    candidate_id_col: str,
    name: str,
) -> pd.DataFrame:
    _require_columns(frame, (query_col, candidate_id_col), name)
    result = frame.copy()
    for column in (query_col, candidate_id_col):
        if result[column].isna().any():
            raise EvaluationError(f"{name} {column} contains null values")
        result[column] = result[column].astype(str)
        if bool(result[column].eq("").any()):
            raise EvaluationError(f"{name} {column} contains empty values")
    keys = [query_col, candidate_id_col]
    if result.duplicated(keys).any():
        duplicate = result.loc[result.duplicated(keys, keep=False), keys].head(3)
        raise EvaluationError(
            f"{name} has duplicate query/candidate IDs: "
            f"{duplicate.to_dict(orient='records')}"
        )
    return result


def _binary_labels(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise EvaluationError(f"{name} contains null values")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise EvaluationError(f"{name} must contain only binary values")
    return numeric.astype(bool)


def rank_candidates(
    candidates: pd.DataFrame,
    *,
    query_col: str = "query_id",
    candidate_id_col: str = "candidate_id",
    score_col: str = "score",
    rank_col: str = "rank",
) -> pd.DataFrame:
    """Rank each query by score descending and candidate ID ascending.

    The candidate ID is the required, deterministic tie breaker.  Returned
    rows are in canonical query/rank order and ranks start at one.
    """

    frame = _normalise_ids(
        candidates,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        name="candidate table",
    )
    _require_columns(frame, (score_col,), "candidate table")
    scores = pd.to_numeric(frame[score_col], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(np.float64)).all():
        raise EvaluationError("scores must be finite numeric values")
    frame[score_col] = scores.astype(np.float64)
    frame = frame.sort_values(
        [query_col, score_col, candidate_id_col],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame[rank_col] = (
        frame.groupby(query_col, sort=False).cumcount().add(1).astype(np.int64)
    )
    return frame


def validate_rank_permutations(
    rankings: pd.DataFrame,
    *,
    query_col: str = "query_id",
    candidate_id_col: str = "candidate_id",
    rank_col: str = "rank",
    score_col: str | None = "score",
) -> bool:
    """Validate per-query 1..N ranks and, when supplied, strict score order."""

    frame = _normalise_ids(
        rankings,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        name="ranking table",
    )
    _require_columns(frame, (rank_col,), "ranking table")
    ranks = pd.to_numeric(frame[rank_col], errors="coerce")
    if ranks.isna().any() or not np.isfinite(ranks.to_numpy(np.float64)).all():
        raise EvaluationError("ranks must be finite positive integers")
    if not np.allclose(ranks, np.round(ranks)) or bool((ranks < 1).any()):
        raise EvaluationError("ranks must be finite positive integers")
    frame[rank_col] = ranks.astype(np.int64)
    for query_id, group in frame.groupby(query_col, sort=False):
        observed = np.sort(group[rank_col].to_numpy(np.int64))
        expected = np.arange(1, len(group) + 1, dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise EvaluationError(f"{query_id} ranks are not a 1..N permutation")
    if score_col is not None:
        _require_columns(frame, (score_col,), "ranking table")
        expected = rank_candidates(
            frame.drop(columns=[rank_col]),
            query_col=query_col,
            candidate_id_col=candidate_id_col,
            score_col=score_col,
            rank_col=rank_col,
        )
        expected_by_key = expected.set_index([query_col, candidate_id_col])[rank_col]
        observed_by_key = frame.set_index([query_col, candidate_id_col])[rank_col]
        expected_by_key = expected_by_key.sort_index()
        observed_by_key = observed_by_key.sort_index()
        if not expected_by_key.equals(observed_by_key):
            raise EvaluationError(
                "stored ranks disagree with score-descending/candidate-ID-ascending order"
            )
    return True


def assert_same_candidate_pool(
    reference: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    query_col: str = "query_id",
    candidate_id_col: str = "candidate_id",
) -> bool:
    """Require exact equality of unique query/candidate key sets."""

    left = _normalise_ids(
        reference,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        name="reference candidate pool",
    )
    right = _normalise_ids(
        observed,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        name="observed candidate pool",
    )
    keys = [query_col, candidate_id_col]
    expected = set(map(tuple, left[keys].to_numpy()))
    actual = set(map(tuple, right[keys].to_numpy()))
    if expected != actual:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise EvaluationError(
            f"candidate pool changed: missing={missing}, extra={extra}"
        )
    return True


def join_predictions_with_labels(
    candidate_pool: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    query_col: str = "query_id",
    candidate_id_col: str = "candidate_id",
    label_col: str = "label",
    identity_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Perform a one-to-one ID join using labels only from ``candidate_pool``."""

    reference = _normalise_ids(
        candidate_pool,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        name="reference candidate pool",
    )
    predicted = _normalise_ids(
        predictions,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        name="prediction table",
    )
    _require_columns(reference, (label_col,), "reference candidate pool")
    reference[label_col] = _binary_labels(reference[label_col], label_col)
    assert_same_candidate_pool(
        reference,
        predicted,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
    )
    keys = [query_col, candidate_id_col]
    if identity_columns is None:
        identity_columns = (
            ("candidate_identity_sha256",)
            if "candidate_identity_sha256" in reference.columns
            else ()
        )
    identity_columns = tuple(identity_columns)
    _require_columns(reference, identity_columns, "reference candidate pool")
    comparison_columns = [label_col, *identity_columns]
    source = reference[keys + comparison_columns].copy()
    if label_col in predicted.columns:
        predicted_labels = predicted[keys + [label_col]].merge(
            source[keys + [label_col]],
            on=keys,
            how="left",
            validate="one_to_one",
            suffixes=("_prediction", "_reference"),
        )
        left = _binary_labels(
            predicted_labels[f"{label_col}_prediction"],
            f"prediction {label_col}",
        )
        right = _binary_labels(
            predicted_labels[f"{label_col}_reference"],
            f"reference {label_col}",
        )
        if not left.equals(right):
            raise EvaluationError("prediction labels conflict with reference labels")
        predicted = predicted.drop(columns=[label_col])
    for identity_col in identity_columns:
        if identity_col not in predicted.columns:
            continue
        comparison = predicted[keys + [identity_col]].merge(
            source[keys + [identity_col]],
            on=keys,
            how="left",
            validate="one_to_one",
            suffixes=("_prediction", "_reference"),
        )
        left = comparison[f"{identity_col}_prediction"].astype(str)
        right = comparison[f"{identity_col}_reference"].astype(str)
        if not left.equals(right):
            raise EvaluationError(f"candidate identity changed: {identity_col}")
        predicted = predicted.drop(columns=[identity_col])
    return predicted.merge(source, on=keys, how="left", validate="one_to_one")


def _query_universe(
    candidate_pool: pd.DataFrame,
    query_universe: pd.DataFrame | Sequence[Any] | None,
    *,
    query_col: str,
    scene_col: str,
    frame_col: str,
) -> pd.DataFrame:
    observed_columns = [query_col]
    for column in (scene_col, frame_col):
        if column in candidate_pool.columns and column not in observed_columns:
            observed_columns.append(column)
    observed = candidate_pool[observed_columns].copy()
    for column in (scene_col, frame_col):
        if column in observed.columns:
            if observed.groupby(query_col, sort=False)[column].nunique().gt(1).any():
                raise EvaluationError(f"a query maps to multiple {column} values")
    observed = observed.drop_duplicates(query_col)
    if query_universe is None:
        universe = observed
    elif isinstance(query_universe, pd.DataFrame):
        _require_columns(query_universe, (query_col,), "query universe")
        universe = query_universe.copy()
    else:
        universe = pd.DataFrame({query_col: list(query_universe)})
    if universe[query_col].isna().any():
        raise EvaluationError("query universe contains null IDs")
    universe[query_col] = universe[query_col].astype(str)
    if universe[query_col].eq("").any() or universe.duplicated(query_col).any():
        raise EvaluationError("query universe IDs must be unique and non-empty")
    missing = sorted(set(observed[query_col]) - set(universe[query_col]))
    if missing:
        raise EvaluationError(f"query universe omits observed queries: {missing[:5]}")
    for column in (scene_col, frame_col):
        observed_map = (
            observed.set_index(query_col)[column]
            if column in observed.columns
            else pd.Series(dtype=object)
        )
        if column not in universe.columns:
            universe[column] = universe[query_col].map(observed_map)
        elif not observed_map.empty:
            overlap = universe[query_col].isin(observed_map.index)
            expected = universe.loc[overlap, query_col].map(observed_map).astype(str)
            actual = universe.loc[overlap, column].astype(str)
            if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
                raise EvaluationError(f"query universe {column} conflicts with candidates")
        universe[column] = universe[column].where(
            universe[column].notna(), universe[query_col]
        )
        universe[column] = universe[column].astype(str)
        if universe[column].eq("").any():
            raise EvaluationError(f"query universe {column} contains empty values")
    return universe.sort_values(query_col, kind="mergesort").reset_index(drop=True)


def _average_precision(labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    cumulative = np.cumsum(labels, dtype=np.float64)
    precision = cumulative / np.arange(1, labels.size + 1, dtype=np.float64)
    return float(precision[labels].sum() / positives)


def _ndcg(labels: np.ndarray, cutoff: int | None = None) -> float:
    if labels.size == 0 or not labels.any():
        return 0.0
    size = labels.size if cutoff is None else min(labels.size, int(cutoff))
    discounts = 1.0 / np.log2(np.arange(2, size + 2, dtype=np.float64))
    dcg = float(np.dot(labels[:size].astype(np.float64), discounts))
    ideal = np.sort(labels.astype(np.float64))[::-1][:size]
    idcg = float(np.dot(ideal, discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def evaluate_rankings(
    candidate_pool: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    *,
    query_universe: pd.DataFrame | Sequence[Any] | None = None,
    query_col: str = "query_id",
    candidate_id_col: str = "candidate_id",
    label_col: str = "label",
    score_col: str = "score",
    probability_col: str | None = None,
    rank_col: str = "rank",
    scene_col: str = "scene_id",
    frame_col: str = "frame_id",
    identity_columns: Sequence[str] | None = None,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Evaluate a complete frozen-pool ranking, counting all universe queries.

    Empty queries and queries with no positive candidate receive zero for all
    ranking metrics.  Both cases are explicitly counted in the returned
    aggregate metrics and marked in ``per_query``.
    """

    reference = _normalise_ids(
        candidate_pool,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        name="reference candidate pool",
    )
    _require_columns(reference, (label_col,), "reference candidate pool")
    reference[label_col] = _binary_labels(reference[label_col], label_col)
    if predictions is None:
        _require_columns(reference, (score_col,), "reference candidate pool")
        ranked = rank_candidates(
            reference,
            query_col=query_col,
            candidate_id_col=candidate_id_col,
            score_col=score_col,
            rank_col=rank_col,
        )
    else:
        predicted = _normalise_ids(
            predictions,
            query_col=query_col,
            candidate_id_col=candidate_id_col,
            name="prediction table",
        )
        _require_columns(predicted, (score_col,), "prediction table")
        if rank_col in predicted.columns:
            validate_rank_permutations(
                predicted,
                query_col=query_col,
                candidate_id_col=candidate_id_col,
                rank_col=rank_col,
                score_col=score_col,
            )
        else:
            predicted = rank_candidates(
                predicted,
                query_col=query_col,
                candidate_id_col=candidate_id_col,
                score_col=score_col,
                rank_col=rank_col,
            )
        ranked = join_predictions_with_labels(
            reference,
            predicted,
            query_col=query_col,
            candidate_id_col=candidate_id_col,
            label_col=label_col,
            identity_columns=identity_columns,
        )
        ranked = ranked.sort_values(
            [query_col, rank_col, candidate_id_col], kind="mergesort"
        ).reset_index(drop=True)
    validate_rank_permutations(
        ranked,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        rank_col=rank_col,
        score_col=score_col,
    )
    universe = _query_universe(
        reference,
        query_universe,
        query_col=query_col,
        scene_col=scene_col,
        frame_col=frame_col,
    )
    grouped = {key: value for key, value in ranked.groupby(query_col, sort=False)}
    rows: list[dict[str, Any]] = []
    for query in universe.itertuples(index=False):
        query_id = str(getattr(query, query_col))
        group = grouped.get(query_id)
        if group is None:
            group = ranked.iloc[0:0]
        else:
            group = group.sort_values(
                [rank_col, candidate_id_col], kind="mergesort"
            )
        labels = group[label_col].to_numpy(bool)
        empty = labels.size == 0
        no_positive = not bool(labels.any())
        positive_ranks = np.flatnonzero(labels) + 1
        first_positive_rank = (
            None if positive_ranks.size == 0 else int(positive_ranks[0])
        )
        j1 = int(labels.size > 0 and bool(labels[0]))
        j5 = int(bool(labels[:5].any()))
        oracle = int(bool(labels.any()))
        rows.append(
            {
                query_col: query_id,
                scene_col: str(getattr(query, scene_col)),
                frame_col: str(getattr(query, frame_col)),
                "candidate_count": int(labels.size),
                "positive_candidate_count": int(labels.sum()),
                "empty_query": empty,
                "no_positive_query": no_positive,
                "nonempty_no_positive_query": bool((not empty) and no_positive),
                "top_candidate_id": (
                    None if empty else str(group.iloc[0][candidate_id_col])
                ),
                "j_at_1": j1,
                "j_at_5": j5,
                "oracle": oracle,
                "mrr": 0.0 if first_positive_rank is None else 1.0 / first_positive_rank,
                "map": _average_precision(labels),
                "ndcg": _ndcg(labels),
                "ndcg_at_5": _ndcg(labels, cutoff=5),
                "first_positive_rank": first_positive_rank,
                "headroom": oracle - j1,
            }
        )
    per_query = pd.DataFrame(rows)
    metric_columns = ("j_at_1", "j_at_5", "oracle", "mrr", "map", "ndcg", "ndcg_at_5")
    query_count = len(per_query)
    aggregate = {
        column: (
            0.0 if query_count == 0 else float(per_query[column].mean())
        )
        for column in metric_columns
    }
    aggregate.update(
        {
            "query_count": query_count,
            "empty_query_count": (
                0 if query_count == 0 else int(per_query["empty_query"].sum())
            ),
            "no_positive_query_count": (
                0 if query_count == 0 else int(per_query["no_positive_query"].sum())
            ),
            "nonempty_no_positive_query_count": (
                0
                if query_count == 0
                else int(per_query["nonempty_no_positive_query"].sum())
            ),
            "positive_query_count": (
                0 if query_count == 0 else int(per_query["oracle"].sum())
            ),
            "j_at_1_count": (
                0 if query_count == 0 else int(per_query["j_at_1"].sum())
            ),
            "j_at_5_count": (
                0 if query_count == 0 else int(per_query["j_at_5"].sum())
            ),
            "oracle_count": (
                0 if query_count == 0 else int(per_query["oracle"].sum())
            ),
            "headroom_count": (
                0 if query_count == 0 else int(per_query["headroom"].sum())
            ),
        }
    )
    aggregate["headroom"] = aggregate["oracle"] - aggregate["j_at_1"]
    if probability_col is not None:
        _require_columns(ranked, (probability_col,), "ranked candidate table")
        probabilities = ranked[probability_col].to_numpy(np.float64)
    else:
        probabilities = None
    # A calibrated candidate probability is the intended discrimination score
    # when available; otherwise the method's raw ranking score is used.
    classification_scores = (
        probabilities
        if probabilities is not None
        else ranked[score_col].to_numpy(np.float64)
    )
    candidate_metrics = candidate_classification_metrics(
        ranked[label_col].to_numpy(bool),
        classification_scores,
        probabilities=probabilities,
        bins=ece_bins,
    )
    return {
        **aggregate,
        "metrics": aggregate,
        "candidate_metrics": candidate_metrics,
        "per_query": per_query,
        "ranked_candidates": ranked,
    }


def validate_oracle_invariance(
    reference_per_query: pd.DataFrame,
    challenger_per_query: pd.DataFrame,
    *,
    query_col: str = "query_id",
    oracle_col: str = "oracle",
) -> bool:
    """Require the same query cohort and identical oracle outcome per query."""

    _require_columns(reference_per_query, (query_col, oracle_col), "reference outcomes")
    _require_columns(challenger_per_query, (query_col, oracle_col), "challenger outcomes")
    if reference_per_query.duplicated(query_col).any() or challenger_per_query.duplicated(query_col).any():
        raise EvaluationError("per-query outcomes contain duplicate query IDs")
    left = reference_per_query.set_index(query_col)[oracle_col].sort_index()
    right = challenger_per_query.set_index(query_col)[oracle_col].sort_index()
    if not left.index.equals(right.index):
        raise EvaluationError("oracle cohorts differ")
    if not left.astype(int).equals(right.astype(int)):
        raise EvaluationError("reranking changed the frozen-pool oracle")
    return True


def compare_rankings(
    candidate_pool: pd.DataFrame,
    reference_predictions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
    *,
    query_universe: pd.DataFrame | Sequence[Any] | None = None,
    query_col: str = "query_id",
    candidate_id_col: str = "candidate_id",
    label_col: str = "label",
    score_col: str = "score",
    probability_col: str | None = None,
    rank_col: str = "rank",
    scene_col: str = "scene_id",
    frame_col: str = "frame_id",
    identity_columns: Sequence[str] | None = None,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260801,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Evaluate and statistically compare two rankings of one frozen pool."""

    common = dict(
        query_universe=query_universe,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        label_col=label_col,
        score_col=score_col,
        probability_col=probability_col,
        rank_col=rank_col,
        scene_col=scene_col,
        frame_col=frame_col,
        identity_columns=identity_columns,
    )
    reference = evaluate_rankings(candidate_pool, reference_predictions, **common)
    challenger = evaluate_rankings(candidate_pool, challenger_predictions, **common)
    reference_query = reference["per_query"].sort_values(query_col).reset_index(drop=True)
    challenger_query = challenger["per_query"].sort_values(query_col).reset_index(drop=True)
    validate_oracle_invariance(
        reference_query, challenger_query, query_col=query_col
    )
    if len(reference_query) == 0:
        raise EvaluationError("cannot compare an empty query universe")
    reference_correct = reference_query["j_at_1"].to_numpy(bool)
    challenger_correct = challenger_query["j_at_1"].to_numpy(bool)
    reference_top = reference_query["top_candidate_id"].fillna("<EMPTY>")
    challenger_top = challenger_query["top_candidate_id"].fillna("<EMPTY>")
    switched = reference_top.to_numpy(str) != challenger_top.to_numpy(str)
    recovered = (~reference_correct) & challenger_correct
    harmful = reference_correct & (~challenger_correct)
    neutral_switch = switched & (reference_correct == challenger_correct)
    switch_count = int(switched.sum())
    recovered_count = int(recovered.sum())
    harmful_count = int(harmful.sum())
    outcome_changing = recovered_count + harmful_count
    reference_headroom = int(
        (reference_query["oracle"].to_numpy(int) - reference_correct.astype(int)).sum()
    )
    switch_metrics = {
        "query_count": int(len(reference_query)),
        "switch_count": switch_count,
        "switch_rate": float(switched.mean()),
        "switch_coverage": float(switched.mean()),
        "recovered": recovered_count,
        "harmful": harmful_count,
        "net_recovered": recovered_count - harmful_count,
        "neutral_switch": int(neutral_switch.sum()),
        "switch_precision": (
            0.0 if switch_count == 0 else recovered_count / switch_count
        ),
        "outcome_changing_precision": (
            0.0 if outcome_changing == 0 else recovered_count / outcome_changing
        ),
        "reference_headroom_count": reference_headroom,
        "headroom_recovered": (
            0.0
            if reference_headroom == 0
            else (recovered_count - harmful_count) / reference_headroom
        ),
    }
    statistical = paired_cluster_statistics(
        reference_correct,
        challenger_correct,
        scene_ids=reference_query[scene_col].to_numpy(),
        frame_ids=reference_query[frame_col].to_numpy(),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    raw_p = float(statistical["mcnemar"]["pvalue"])
    statistical["holm_adjusted_pvalue"] = float(holm_adjust([raw_p])[0])
    return {
        "reference": reference,
        "challenger": challenger,
        "switch_metrics": switch_metrics,
        "statistics": statistical,
        "oracle_invariant": True,
    }


# Concise aliases for callers that prefer singular or query-centric names.
evaluate = evaluate_rankings
rank_queries = rank_candidates
validate_candidate_pool = assert_same_candidate_pool


__all__ = [
    "EvaluationError",
    "assert_same_candidate_pool",
    "compare_rankings",
    "evaluate",
    "evaluate_rankings",
    "join_predictions_with_labels",
    "rank_candidates",
    "rank_queries",
    "validate_candidate_pool",
    "validate_oracle_invariance",
    "validate_rank_permutations",
]
