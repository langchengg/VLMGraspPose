"""Deterministic CPU-only LightGBM LambdaRank adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _finite_matrix(features: Any) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    return matrix


def _binary_integer_labels(labels: Any, *, length: int) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or len(raw) != length:
        raise ValueError("labels must match feature rows")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(numeric).all() or not set(np.unique(numeric)).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("labels must be binary 0/1")
    return numeric.astype(np.int32)


def contiguous_group_order(
    query_ids: Sequence[Any], *, length: int
) -> tuple[np.ndarray, list[int]]:
    """Return a stable row permutation and contiguous LightGBM group sizes."""

    query = np.asarray(list(query_ids), dtype=object)
    if query.ndim != 1 or len(query) != length:
        raise ValueError("query_ids must match feature rows")
    if any(value is None or str(value) == "" for value in query):
        raise ValueError("query_ids must be non-empty")
    string_query = np.asarray([str(value) for value in query], dtype=object)
    ordered_ids = list(dict.fromkeys(string_query.tolist()))
    parts = [np.flatnonzero(string_query == query_id) for query_id in ordered_ids]
    order = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    groups = [int(len(part)) for part in parts]
    if int(sum(groups)) != length or len(np.unique(order)) != length:
        raise AssertionError("contiguous query grouping does not cover rows exactly once")
    return order.astype(np.int64, copy=False), groups


class LightGBMLambdaRank:
    """Sklearn-like LambdaRank wrapper with audited grouping and locked controls."""

    def __init__(self, *, seed: int, **parameters: Any) -> None:
        self.seed = int(seed)
        defaults: dict[str, Any] = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 20,
        }
        defaults.update(parameters)
        # These experiment-contract values cannot be weakened by a trial config.
        defaults.update(
            {
                "objective": "lambdarank",
                "metric": "ndcg",
                "label_gain": [0, 1],
                "deterministic": True,
                "force_col_wise": True,
                "device_type": "cpu",
                "random_state": self.seed,
                "feature_fraction_seed": self.seed,
                "bagging_seed": self.seed,
                "data_random_seed": self.seed,
                "n_jobs": 1,
                "verbosity": -1,
            }
        )
        self.parameters = defaults
        self.model: Any | None = None
        self.training_order_: np.ndarray | None = None
        self.group_sizes_: list[int] | None = None
        self.n_features_in_: int | None = None

    def fit(
        self,
        features: Any,
        labels: Any,
        query_ids: Sequence[Any],
        *,
        eval_set: tuple[Any, Any, Sequence[Any]] | None = None,
    ) -> "LightGBMLambdaRank":
        matrix = _finite_matrix(features)
        integer_labels = _binary_integer_labels(labels, length=len(matrix))
        order, groups = contiguous_group_order(query_ids, length=len(matrix))
        fit_kwargs: dict[str, Any] = {"group": groups, "eval_at": [1]}
        if eval_set is not None:
            eval_features, eval_labels, eval_query_ids = eval_set
            eval_matrix = _finite_matrix(eval_features)
            if eval_matrix.shape[1] != matrix.shape[1]:
                raise ValueError("validation feature width must match training")
            eval_integer_labels = _binary_integer_labels(
                eval_labels, length=len(eval_matrix)
            )
            eval_order, eval_groups = contiguous_group_order(
                eval_query_ids, length=len(eval_matrix)
            )
            fit_kwargs.update(
                {
                    "eval_set": [(eval_matrix[eval_order], eval_integer_labels[eval_order])],
                    "eval_group": [eval_groups],
                    "eval_at": [1],
                }
            )
        # Delay the optional dependency import until the complete local input
        # contract has been checked, so callers receive deterministic data
        # errors even in environments without LightGBM installed.
        import lightgbm as lgb

        self.model = lgb.LGBMRanker(**self.parameters)
        self.model.fit(matrix[order], integer_labels[order], **fit_kwargs)
        self.training_order_ = order
        self.group_sizes_ = groups
        self.n_features_in_ = int(matrix.shape[1])
        return self

    def predict(self, features: Any) -> np.ndarray:
        if self.model is None or self.n_features_in_ is None:
            raise RuntimeError("model is not fitted")
        matrix = _finite_matrix(features)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("feature width does not match fitted model")
        scores = np.asarray(self.model.predict(matrix), dtype=np.float64)
        if scores.shape != (len(matrix),) or not np.isfinite(scores).all():
            raise RuntimeError("LightGBM returned invalid scores")
        return scores

    def artifact(self) -> dict[str, Any]:
        import lightgbm as lgb

        return {
            "kind": "lightgbm_lambdarank",
            "lightgbm_version": str(lgb.__version__),
            "parameters": dict(self.parameters),
            "eval_at": [1],
            "seed": self.seed,
            "group_sizes": self.group_sizes_,
            "integer_binary_labels": True,
            "single_threaded_cpu": True,
        }


__all__ = ["LightGBMLambdaRank", "contiguous_group_order"]
