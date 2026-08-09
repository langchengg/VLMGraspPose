"""Minimal independent evaluator for locked candidate-level predictions.

This intentionally does not import :mod:`reranking.evaluate`.  It exists as a
second implementation for the final sanity audit, using only strict ID-set
joins and a stable ``score desc, candidate_id asc`` ordering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class IndependentEvaluationError(ValueError):
    """Raised when labels, predictions, or the query universe do not align."""


def independent_j_at_1(
    reference: pd.DataFrame,
    predictions: pd.DataFrame,
    query_universe: pd.DataFrame,
) -> dict[str, Any]:
    required_reference = {"query_id", "candidate_id", "label"}
    required_prediction = {"query_id", "candidate_id", "score"}
    if not required_reference.issubset(reference.columns):
        raise IndependentEvaluationError("reference contract is incomplete")
    if not required_prediction.issubset(predictions.columns):
        raise IndependentEvaluationError("prediction contract is incomplete")
    if "query_id" not in query_universe.columns:
        raise IndependentEvaluationError("query universe lacks query_id")
    left = reference[["query_id", "candidate_id", "label"]].copy()
    right = predictions[["query_id", "candidate_id", "score"]].copy()
    for frame in (left, right):
        frame["query_id"] = frame["query_id"].astype(str)
        frame["candidate_id"] = frame["candidate_id"].astype(str)
    if left.duplicated(["query_id", "candidate_id"]).any():
        raise IndependentEvaluationError("duplicate reference candidate key")
    if right.duplicated(["query_id", "candidate_id"]).any():
        raise IndependentEvaluationError("duplicate prediction candidate key")
    left_keys = set(map(tuple, left[["query_id", "candidate_id"]].to_numpy()))
    right_keys = set(map(tuple, right[["query_id", "candidate_id"]].to_numpy()))
    if left_keys != right_keys:
        raise IndependentEvaluationError("reference/prediction candidate key sets differ")
    scores = pd.to_numeric(right["score"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(scores).all():
        raise IndependentEvaluationError("prediction scores are non-finite")
    labels = pd.to_numeric(left["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise IndependentEvaluationError("reference labels are not binary")
    joined = left.merge(
        right,
        on=["query_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    ).sort_values(
        ["query_id", "score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    top = joined.drop_duplicates("query_id", keep="first").set_index("query_id")
    universe = query_universe["query_id"].astype(str)
    if universe.duplicated().any():
        raise IndependentEvaluationError("duplicate query-universe ID")
    top_labels = top["label"].reindex(universe, fill_value=0).astype(np.int8)
    oracle = (
        joined.groupby("query_id", sort=False)["label"]
        .max()
        .reindex(universe, fill_value=0)
        .astype(np.int8)
    )
    count = int(top_labels.sum())
    query_count = int(len(universe))
    return {
        "query_count": query_count,
        "candidate_count": int(len(joined)),
        "j_at_1_count": count,
        "j_at_1": 0.0 if query_count == 0 else count / query_count,
        "oracle_count": int(oracle.sum()),
        "oracle": 0.0 if query_count == 0 else int(oracle.sum()) / query_count,
        "empty_query_count": int(query_count - joined["query_id"].nunique()),
        "implementation": "independent_id_join_score_desc_candidate_id_asc",
    }


__all__ = ["IndependentEvaluationError", "independent_j_at_1"]
