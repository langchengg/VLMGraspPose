"""Graded, deterministic CPU LambdaMART adapter for frozen 6-DoF pools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FORMAL_SEEDS = (20260815, 20260816, 20260817)
LABEL_GAIN = (0, 1, 3, 7, 15, 31, 63)


def _finite_matrix(features: Any) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("ranker features must be finite after train-fitted imputation")
    return matrix


def graded_relevance(labels: Any, *, length: int) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or len(raw) != int(length):
        raise ValueError("relevance labels must match feature rows")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("relevance labels must be finite integers")
    if bool(((numeric < 0) | (numeric > 6)).any()):
        raise ValueError("graded relevance must be in the closed integer range 0..6")
    return numeric.astype(np.int32)


def validate_group_sizes(group_sizes: Sequence[Any], *, length: int) -> list[int]:
    """Validate an already contiguous LightGBM group-boundary contract."""

    raw = list(group_sizes)
    if not raw:
        raise ValueError("group_sizes must be non-empty")
    groups: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("group sizes must be positive integers")
        try:
            numeric = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("group sizes must be positive integers") from error
        if numeric <= 0 or float(value) != numeric:
            raise ValueError("group sizes must be positive integers")
        groups.append(numeric)
    if sum(groups) != int(length):
        raise ValueError(
            f"group sizes sum to {sum(groups)}, but there are {int(length)} rows"
        )
    return groups


def contiguous_group_sizes(query_ids: Sequence[Any], *, length: int) -> list[int]:
    """Return run sizes, rejecting a query that reappears after a boundary."""

    query = np.asarray(list(query_ids), dtype=object)
    if query.ndim != 1 or len(query) != int(length) or not len(query):
        raise ValueError("query_ids must match feature rows")
    values = [str(value) for value in query]
    if any(value is None for value in query.tolist()) or any(not value for value in values):
        raise ValueError("query_ids must be non-null, non-empty values")
    seen: set[str] = set()
    previous: str | None = None
    groups: list[int] = []
    for value in values:
        if value != previous:
            if value in seen:
                raise ValueError(
                    f"query {value!r} is non-contiguous; sort complete groups before fit"
                )
            seen.add(value)
            groups.append(1)
            previous = value
        else:
            groups[-1] += 1
    return validate_group_sizes(groups, length=length)


def _candidate_ids(values: Sequence[Any], *, length: int) -> tuple[str, ...]:
    identifiers = tuple(str(value) for value in values)
    if len(identifiers) != int(length) or any(not value for value in identifiers):
        raise ValueError("candidate_ids must contain one non-empty ID per feature row")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate_ids must be unique in a prediction batch")
    return identifiers


@dataclass(frozen=True)
class ValidationData:
    features: Any
    relevance: Any
    group_sizes: Sequence[int]


class GradedLightGBMLambdaRank:
    """LightGBM LambdaRank with locked 0..6 relevance and group semantics.

    Input rows must already be group-contiguous.  The adapter never shuffles or
    silently reorders candidates, and :meth:`predict_table` returns exactly one
    score for every caller-supplied candidate ID in the same order.
    """

    def __init__(
        self,
        *,
        seed: int,
        early_stopping_rounds: int = 25,
        **parameters: Any,
    ) -> None:
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.early_stopping_rounds = int(early_stopping_rounds)
        if self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive")
        defaults: dict[str, Any] = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 20,
            "feature_fraction": 1.0,
        }
        defaults.update(parameters)
        # Experiment-critical values are locked after user parameters so a
        # trial cannot silently switch to GPU, binary gains, or nondeterminism.
        defaults.update(
            {
                "objective": "lambdarank",
                "metric": "ndcg",
                "label_gain": list(LABEL_GAIN),
                "deterministic": True,
                "force_col_wise": True,
                "device_type": "cpu",
                "random_state": self.seed,
                "feature_fraction_seed": self.seed,
                "bagging_seed": self.seed,
                "data_random_seed": self.seed,
                "drop_seed": self.seed,
                "n_jobs": 1,
                "verbosity": -1,
            }
        )
        self.parameters = defaults
        self.model: Any | None = None
        self.group_sizes_: list[int] | None = None
        self.validation_group_sizes_: list[int] | None = None
        self.n_features_in_: int | None = None
        self.best_iteration_: int | None = None

    def fit_grouped(
        self,
        features: Any,
        relevance: Any,
        group_sizes: Sequence[Any],
        *,
        validation: ValidationData | None = None,
    ) -> "GradedLightGBMLambdaRank":
        matrix = _finite_matrix(features)
        labels = graded_relevance(relevance, length=len(matrix))
        groups = validate_group_sizes(group_sizes, length=len(matrix))

        fit_kwargs: dict[str, Any] = {"group": groups, "eval_at": [1, 5, 10]}
        callbacks: list[Any] = []
        validation_groups: list[int] | None = None

        # Validate all local contracts before importing the optional dependency.
        if validation is not None:
            validation_matrix = _finite_matrix(validation.features)
            if validation_matrix.shape[1] != matrix.shape[1]:
                raise ValueError("validation feature width must match training")
            validation_labels = graded_relevance(
                validation.relevance, length=len(validation_matrix)
            )
            validation_groups = validate_group_sizes(
                validation.group_sizes, length=len(validation_matrix)
            )
            fit_kwargs.update(
                {
                    "eval_X": validation_matrix,
                    "eval_y": validation_labels,
                    "eval_group": [validation_groups],
                }
            )

        import lightgbm as lgb

        if validation is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                )
            )
            fit_kwargs["callbacks"] = callbacks
        self.model = lgb.LGBMRanker(**self.parameters)
        self.model.fit(matrix, labels, **fit_kwargs)
        self.group_sizes_ = groups
        self.validation_group_sizes_ = validation_groups
        self.n_features_in_ = int(matrix.shape[1])
        best = getattr(self.model, "best_iteration_", None)
        self.best_iteration_ = None if best is None else int(best)
        return self

    def fit(
        self,
        features: Any,
        relevance: Any,
        query_ids: Sequence[Any],
        *,
        eval_set: tuple[Any, Any, Sequence[Any]] | None = None,
    ) -> "GradedLightGBMLambdaRank":
        """Fit using explicit contiguous query IDs.

        ``eval_set`` mirrors the repository's older adapter but also enforces
        contiguity instead of reordering rows behind the caller's back.
        """

        matrix = _finite_matrix(features)
        groups = contiguous_group_sizes(query_ids, length=len(matrix))
        validation: ValidationData | None = None
        if eval_set is not None:
            eval_features, eval_relevance, eval_query_ids = eval_set
            eval_matrix = _finite_matrix(eval_features)
            validation = ValidationData(
                features=eval_matrix,
                relevance=eval_relevance,
                group_sizes=contiguous_group_sizes(
                    eval_query_ids, length=len(eval_matrix)
                ),
            )
        return self.fit_grouped(
            matrix, relevance, groups, validation=validation
        )

    def predict(
        self,
        features: Any,
        *,
        candidate_ids: Sequence[Any] | None = None,
    ) -> np.ndarray:
        if self.model is None or self.n_features_in_ is None:
            raise RuntimeError("ranker is not fitted")
        matrix = _finite_matrix(features)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("feature width does not match fitted model")
        if candidate_ids is not None:
            _candidate_ids(candidate_ids, length=len(matrix))
        scores = np.asarray(self.model.predict(matrix), dtype=np.float64)
        if scores.shape != (len(matrix),) or not np.isfinite(scores).all():
            raise RuntimeError("LightGBM returned invalid prediction scores")
        return scores

    def predict_table(
        self,
        features: Any,
        candidate_ids: Sequence[Any],
        *,
        score_column: str = "rerank_score",
    ) -> pd.DataFrame:
        matrix = _finite_matrix(features)
        identifiers = _candidate_ids(candidate_ids, length=len(matrix))
        scores = self.predict(matrix, candidate_ids=identifiers)
        result = pd.DataFrame(
            {"candidate_id": list(identifiers), str(score_column): scores}
        )
        if tuple(result["candidate_id"].astype(str)) != identifiers or len(result) != len(matrix):
            raise AssertionError("prediction changed frozen candidate membership or order")
        return result

    def artifact(self) -> dict[str, Any]:
        import lightgbm as lgb

        return {
            "kind": "lightgbm_graded_lambdarank_6d",
            "lightgbm_version": str(lgb.__version__),
            "parameters": dict(self.parameters),
            "label_contract": "integer relevance 0..6",
            "label_gain": list(LABEL_GAIN),
            "eval_at": [1, 5, 10],
            "seed": self.seed,
            "formal_seed_compatible": self.seed in FORMAL_SEEDS,
            "group_sizes": self.group_sizes_,
            "validation_group_sizes": self.validation_group_sizes_,
            "early_stopping_rounds": self.early_stopping_rounds,
            "best_iteration": self.best_iteration_,
            "single_threaded_cpu": True,
            "row_reordering": False,
        }


def rank_candidate_table(
    model: GradedLightGBMLambdaRank,
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    candidate_id_column: str = "candidate_id",
    score_column: str = "rerank_score",
) -> pd.DataFrame:
    """Attach scores without filtering, duplicating, or reordering candidates."""

    if candidate_id_column not in frame:
        raise ValueError(f"candidate table lacks {candidate_id_column!r}")
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"candidate table lacks feature columns: {missing}")
    before = tuple(frame[candidate_id_column].astype(str))
    prediction = model.predict_table(
        frame[list(feature_columns)].to_numpy(dtype=float),
        before,
        score_column=score_column,
    )
    result = frame.copy()
    result[score_column] = prediction[score_column].to_numpy(dtype=float)
    after = tuple(result[candidate_id_column].astype(str))
    if before != after or len(result) != len(frame):
        raise AssertionError("ranking adapter changed candidate membership")
    return result


def fit_ranker(
    features: Any,
    relevance: Any,
    group_sizes: Sequence[Any],
    validation_data: ValidationData | None,
    config: dict[str, Any],
) -> GradedLightGBMLambdaRank:
    """Functional experiment entry point around the audited graded adapter."""

    values = dict(config)
    if "seed" not in values:
        raise ValueError("ranker config requires an explicit seed")
    seed = int(values.pop("seed"))
    early_stopping_rounds = int(values.pop("early_stopping_rounds", 25))
    model = GradedLightGBMLambdaRank(
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        **values,
    )
    return model.fit_grouped(
        features,
        relevance,
        group_sizes,
        validation=validation_data,
    )


def predict_scores(
    model: GradedLightGBMLambdaRank,
    features: Any,
    group_sizes: Sequence[Any],
    *,
    candidate_ids: Sequence[Any] | None = None,
) -> np.ndarray:
    """Score a complete grouped pool while validating row/group membership."""

    matrix = _finite_matrix(features)
    validate_group_sizes(group_sizes, length=len(matrix))
    return model.predict(matrix, candidate_ids=candidate_ids)


__all__ = [
    "FORMAL_SEEDS",
    "GradedLightGBMLambdaRank",
    "LABEL_GAIN",
    "ValidationData",
    "contiguous_group_sizes",
    "graded_relevance",
    "fit_ranker",
    "predict_scores",
    "rank_candidate_table",
    "validate_group_sizes",
]
