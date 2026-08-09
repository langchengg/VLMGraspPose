"""Leakage-safe scalar baselines for candidate-grasp reranking.

The estimators in this module deliberately accept labels, query identifiers,
and baseline scores as separate arrays.  Ground-truth columns are therefore
never selected implicitly from a candidate table.  All fit-time state
(scalers and score calibrators) is learned exclusively from the rows passed to
``fit``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FORBIDDEN_COLUMN_TOKENS = frozenset(
    {
        "gt",
        "ground_truth",
        "label",
        "correct",
        "success",
        "positive",
        "iou_to_gt",
        "iou_with_gt",
        "angle_error_to_gt",
        "target_gt_mask",
    }
)

_FORBIDDEN_PHRASES = (
    "ground_truth",
    "groundtruth",
    "iou_to_gt",
    "iou_with_gt",
    "angle_error_to_gt",
    "target_gt_mask",
)

_FORBIDDEN_TOKEN_PREFIXES = ("label", "correct", "success")

HIST_GRADIENT_BOOSTING_FALLBACK_REASON = (
    "LightGBM and XGBoost ranking backends are unavailable in the current "
    "environment; sklearn HistGradientBoostingClassifier is used as a "
    "pointwise fallback. It is not LambdaMART."
)

_LOGISTIC_SOLVER_BY_PENALTY = {
    "l1": "liblinear",
    "l2": "lbfgs",
}


def _column_tokens(column: str) -> tuple[str, ...]:
    """Normalize snake/kebab/camel names for fail-closed leakage scanning."""

    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column))
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", expanded).strip("_").lower()
    return tuple(part for part in normalized.split("_") if part)


def scan_forbidden_columns(
    columns: Sequence[str], *, extra_tokens: Sequence[str] = ()
) -> tuple[str, ...]:
    """Return feature names that can encode labels or ground truth.

    The scanner is intentionally stricter than an exact-name deny list.  It
    catches tokenized variants such as ``is_correct``, ``successProbability``
    and ``candidate_gt_iou`` while avoiding accidental substring matches such
    as ``height``.
    """

    token_set = set(FORBIDDEN_COLUMN_TOKENS)
    token_set.update(str(token).strip().lower() for token in extra_tokens)
    rejected: list[str] = []
    for raw_column in columns:
        column = str(raw_column)
        lowered = re.sub(r"[^a-zA-Z0-9]+", "_", column).strip("_").lower()
        tokens = set(_column_tokens(column))
        compact = lowered.replace("_", "")
        phrase_hit = any(
            phrase in lowered or phrase.replace("_", "") in compact
            for phrase in _FORBIDDEN_PHRASES
        )
        token_hit = bool(tokens & token_set) or any(
            token.startswith(prefix)
            for token in tokens
            for prefix in _FORBIDDEN_TOKEN_PREFIXES
        )
        if phrase_hit or token_hit:
            rejected.append(column)
    return tuple(sorted(set(rejected)))


def assert_no_forbidden_columns(
    columns: Sequence[str], *, extra_tokens: Sequence[str] = ()
) -> None:
    """Fail closed when an inference feature can contain GT information."""

    rejected = scan_forbidden_columns(columns, extra_tokens=extra_tokens)
    if rejected:
        raise ValueError(
            "forbidden ground-truth/label feature columns: "
            f"{list(rejected)}"
        )


def _finite_vector(values: Sequence[Any], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _binary_labels(values: Sequence[Any]) -> np.ndarray:
    labels = _finite_vector(values, name="labels")
    if not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError("labels must be binary (0/1)")
    return labels


def _query_array(query_ids: Sequence[Any], *, length: int) -> np.ndarray:
    query_array = np.asarray(list(query_ids), dtype=object)
    if query_array.ndim != 1 or len(query_array) != length:
        raise ValueError("query_ids length must match candidate rows")
    if any(value is None or str(value) == "" for value in query_array):
        raise ValueError("query_ids must be non-empty")
    return query_array


def _query_slices(query_ids: np.ndarray) -> list[tuple[str, np.ndarray]]:
    string_ids = np.asarray([str(value) for value in query_ids], dtype=object)
    return [
        (query_id, np.flatnonzero(string_ids == query_id))
        for query_id in dict.fromkeys(string_ids.tolist())
    ]


def _query_balanced_weights(
    labels: np.ndarray, query_ids: np.ndarray, *, balance_classes: bool
) -> np.ndarray:
    """Give every query equal total weight, optionally balancing labels inside it."""

    if len(labels) == 0:
        raise ValueError("at least one candidate is required")
    class_multiplier = np.ones(len(labels), dtype=np.float64)
    if balance_classes and len(np.unique(labels)) == 2:
        positive = float(np.sum(labels == 1.0))
        negative = float(np.sum(labels == 0.0))
        class_multiplier[labels == 1.0] = len(labels) / (2.0 * positive)
        class_multiplier[labels == 0.0] = len(labels) / (2.0 * negative)
    weights = np.zeros(len(labels), dtype=np.float64)
    slices = _query_slices(query_ids)
    for _, indices in slices:
        local = class_multiplier[indices]
        weights[indices] = local / local.sum()
    # Mean one is friendlier to sklearn's regularization conventions while
    # preserving equal total mass for every query.
    weights *= len(labels) / weights.sum()
    return weights


def _feature_frame(features: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(features, pd.DataFrame):
        names = tuple(map(str, features.columns))
        assert_no_forbidden_columns(names)
        try:
            matrix = features.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("all tabular features must be numeric") from exc
    else:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix")
        names = tuple(f"x{index}" for index in range(matrix.shape[1]))
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features contain non-finite values")
    return matrix, names


def _check_predict_features(
    features: pd.DataFrame | np.ndarray, expected: tuple[str, ...]
) -> np.ndarray:
    matrix, names = _feature_frame(features)
    if names != expected:
        raise ValueError(
            "feature schema mismatch: "
            f"expected {list(expected)}, received {list(names)}"
        )
    return matrix


def _normalize_importance(
    names: tuple[str, ...], values: Sequence[float]
) -> dict[str, float]:
    importance = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if len(importance) != len(names) or not np.all(np.isfinite(importance)):
        raise ValueError("invalid feature importance vector")
    denominator = float(importance.sum())
    if denominator > 0.0:
        importance = importance / denominator
    return {name: float(value) for name, value in zip(names, importance)}


def derive_q_features(
    frame: pd.DataFrame,
    *,
    query_col: str = "query_id",
    q_col: str = "q_raw",
    rank_col: str | None = None,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Add finite raw-score, rank, margin, entropy and prominence features.

    Ties use average ranks.  Neighbour margins use stable input order as the
    secondary key, so repeated runs are deterministic without manufacturing a
    numerical score difference between tied candidates.
    """

    if not 0.0 < float(eps) < 0.5:
        raise ValueError("eps must be between zero and 0.5")
    score_inputs = [q_col]
    if rank_col is not None:
        score_inputs.append(rank_col)
    assert_no_forbidden_columns(score_inputs)
    missing = [name for name in (query_col, q_col) if name not in frame.columns]
    if rank_col is not None and rank_col not in frame.columns:
        missing.append(rank_col)
    if missing:
        raise ValueError(f"missing q-feature input columns: {sorted(set(missing))}")
    if len(frame) == 0:
        raise ValueError("cannot derive q features from an empty table")

    output = frame.copy()
    q = pd.to_numeric(output[q_col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{q_col} contains non-finite values")
    query_ids = _query_array(output[query_col].tolist(), length=len(output))
    q_clipped = np.clip(q, eps, 1.0 - eps)

    derived: dict[str, np.ndarray] = {
        "q_clipped": q_clipped,
        "q_log": np.log(q_clipped),
        "q_logit": np.log(q_clipped) - np.log1p(-q_clipped),
        "candidate_count": np.zeros(len(q), dtype=np.float64),
        "rank_percentile": np.zeros(len(q), dtype=np.float64),
        "delta_q_top1": np.zeros(len(q), dtype=np.float64),
        "ratio_q_top1": np.zeros(len(q), dtype=np.float64),
        "delta_q_previous": np.zeros(len(q), dtype=np.float64),
        "delta_q_next": np.zeros(len(q), dtype=np.float64),
        "q_zscore_within_query": np.zeros(len(q), dtype=np.float64),
        "q_percentile_within_query": np.zeros(len(q), dtype=np.float64),
        "top1_margin": np.zeros(len(q), dtype=np.float64),
        "top2_margin": np.zeros(len(q), dtype=np.float64),
        "score_entropy": np.zeros(len(q), dtype=np.float64),
        "score_concentration": np.zeros(len(q), dtype=np.float64),
        "score_prominence": np.zeros(len(q), dtype=np.float64),
    }
    original_rank = np.zeros(len(q), dtype=np.float64)

    for _, indices in _query_slices(query_ids):
        local_q = q[indices]
        count = len(indices)
        stable_order = np.argsort(-local_q, kind="stable")
        sorted_q = local_q[stable_order]
        top1 = float(sorted_q[0])
        top2 = float(sorted_q[1]) if count >= 2 else top1
        top3 = float(sorted_q[2]) if count >= 3 else top2

        # Pandas average rank gives every tied score the same rank independent
        # of row order.  Higher q receives a higher percentile.
        average_rank = (
            pd.Series(local_q).rank(method="average", ascending=False).to_numpy()
        )
        percentile = (
            np.ones(count, dtype=np.float64)
            if count == 1
            else 1.0 - (average_rank - 1.0) / (count - 1.0)
        )
        mean = float(local_q.mean())
        std = float(local_q.std())
        probabilities = q_clipped[indices] / q_clipped[indices].sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        normalized_entropy = entropy / np.log(count) if count > 1 else 0.0

        derived["candidate_count"][indices] = count
        derived["rank_percentile"][indices] = percentile
        derived["q_percentile_within_query"][indices] = percentile
        derived["delta_q_top1"][indices] = local_q - top1
        derived["ratio_q_top1"][indices] = q_clipped[indices] / max(
            float(np.clip(top1, eps, 1.0 - eps)), eps
        )
        derived["q_zscore_within_query"][indices] = (
            (local_q - mean) / std if std > 1e-12 else 0.0
        )
        derived["top1_margin"][indices] = top1 - top2
        derived["top2_margin"][indices] = top2 - top3
        derived["score_entropy"][indices] = entropy
        derived["score_concentration"][indices] = 1.0 - normalized_entropy

        previous = np.zeros(count, dtype=np.float64)
        following = np.zeros(count, dtype=np.float64)
        prominence = np.zeros(count, dtype=np.float64)
        for sorted_position, local_position in enumerate(stable_order):
            if sorted_position > 0:
                previous[local_position] = (
                    local_q[local_position] - sorted_q[sorted_position - 1]
                )
            if sorted_position + 1 < count:
                following[local_position] = (
                    local_q[local_position] - sorted_q[sorted_position + 1]
                )
            if count > 1:
                other_max = float(np.max(np.delete(local_q, local_position)))
                prominence[local_position] = local_q[local_position] - other_max
        derived["delta_q_previous"][indices] = previous
        derived["delta_q_next"][indices] = following
        derived["score_prominence"][indices] = prominence
        original_rank[indices] = average_rank

    if rank_col is not None:
        original_rank = pd.to_numeric(
            output[rank_col], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(original_rank)) or np.any(original_rank < 1.0):
            raise ValueError(f"{rank_col} must contain finite ranks >= 1")
    derived["original_rank"] = original_rank

    for name, values in derived.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"derived feature {name} is not finite")
        output[name] = values
    return output


class TrainOnlyScoreCalibrator:
    """One-dimensional Platt or isotonic calibration with fit provenance."""

    def __init__(self, method: str = "platt", *, random_state: int = 0) -> None:
        normalized = str(method).lower().replace("-", "_")
        if normalized not in {"platt", "isotonic"}:
            raise ValueError("calibration method must be 'platt' or 'isotonic'")
        self.method = normalized
        self.random_state = int(random_state)

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[Any],
        *,
        sample_weight: Sequence[float] | None = None,
        fit_sample_ids: Sequence[Any] | None = None,
    ) -> "TrainOnlyScoreCalibrator":
        x = _finite_vector(scores, name="calibration scores")
        y = _binary_labels(labels)
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("calibration scores and labels must have equal length")
        weights = (
            np.ones(len(y), dtype=np.float64)
            if sample_weight is None
            else _finite_vector(sample_weight, name="calibration weights")
        )
        if len(weights) != len(y) or np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError("calibration weights must be non-negative and non-zero")
        if fit_sample_ids is not None and len(fit_sample_ids) != len(y):
            raise ValueError("fit_sample_ids length must match calibration rows")

        self.n_fit_samples_ = len(y)
        self.fit_sample_ids_ = (
            tuple(map(str, fit_sample_ids)) if fit_sample_ids is not None else ()
        )
        classes = np.unique(y)
        if len(classes) == 1:
            self.constant_probability_ = float(np.clip(classes[0], 1e-6, 1.0 - 1e-6))
            self.model_ = None
            self.fit_mode_ = "constant_single_class"
        elif self.method == "platt":
            self.model_ = LogisticRegression(
                solver="lbfgs", max_iter=1000, random_state=self.random_state
            )
            self.model_.fit(x.reshape(-1, 1), y.astype(np.int64), sample_weight=weights)
            self.constant_probability_ = None
            self.fit_mode_ = "platt_logistic_regression"
        else:
            self.model_ = IsotonicRegression(out_of_bounds="clip")
            self.model_.fit(x, y, sample_weight=weights)
            self.constant_probability_ = None
            self.fit_mode_ = "isotonic_regression"
        return self

    def predict_proba(self, scores: Sequence[float]) -> np.ndarray:
        if not hasattr(self, "fit_mode_"):
            raise RuntimeError("calibrator has not been fitted")
        x = _finite_vector(scores, name="calibration scores")
        if self.model_ is None:
            calibrated = np.full(len(x), self.constant_probability_, dtype=np.float64)
        elif self.method == "platt":
            calibrated = self.model_.predict_proba(x.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.model_.predict(x)
        calibrated = np.clip(np.asarray(calibrated, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        if not np.all(np.isfinite(calibrated)):
            raise RuntimeError("calibrator produced non-finite probabilities")
        return calibrated

    transform = predict_proba

    @property
    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "fit_mode_"):
            raise RuntimeError("calibrator has not been fitted")
        return {
            "method": self.method,
            "fit_mode": self.fit_mode_,
            "fit_scope": "rows explicitly supplied to fit only",
            "n_fit_samples": self.n_fit_samples_,
            "fit_sample_ids": list(self.fit_sample_ids_),
        }


# Short alias used by experiment configuration code.
ScoreCalibrator = TrainOnlyScoreCalibrator


class PointwiseRanker:
    """Query-balanced sklearn pointwise baselines with a common interface."""

    _VALID_ESTIMATORS = {"logistic", "random_forest", "hist_gradient_boosting"}

    def __init__(
        self,
        estimator: str = "logistic",
        *,
        random_state: int = 0,
        c: float = 1.0,
        n_estimators: int = 200,
        max_iter: int = 200,
        penalty: str = "l2",
    ) -> None:
        normalized = str(estimator).lower().replace("-", "_")
        aliases = {
            "logistic_regression": "logistic",
            "rf": "random_forest",
            "histgb": "hist_gradient_boosting",
            "hist_gradient_boosting_classifier": "hist_gradient_boosting",
        }
        self.estimator_kind = aliases.get(normalized, normalized)
        if self.estimator_kind not in self._VALID_ESTIMATORS:
            raise ValueError(f"unknown pointwise estimator: {estimator}")
        self.random_state = int(random_state)
        self.c = float(c)
        self.n_estimators = int(n_estimators)
        self.max_iter = int(max_iter)
        self.penalty = str(penalty).strip().lower()
        if self.penalty not in _LOGISTIC_SOLVER_BY_PENALTY:
            raise ValueError("penalty must be 'l1' or 'l2'")
        if self.estimator_kind != "logistic" and self.penalty != "l2":
            raise ValueError("penalty is only configurable for the logistic estimator")
        if self.estimator_kind == "logistic" and (
            not np.isfinite(self.c) or self.c <= 0.0
        ):
            raise ValueError("c must be finite and strictly positive")

    def _logistic_regularization_kwargs(self) -> tuple[dict[str, Any], str]:
        """Return a warning-free sklearn regularization configuration.

        sklearn 1.8 deprecated ``penalty`` in favour of the equivalent
        ``l1_ratio`` values.  Older supported releases still require the
        explicit ``penalty`` spelling, so detect the installed API rather than
        relying on a version-string comparison.
        """

        if getattr(LogisticRegression(), "penalty", "deprecated") == "deprecated":
            return {"l1_ratio": 1.0 if self.penalty == "l1" else 0.0}, "l1_ratio"
        return {"penalty": self.penalty}, "penalty"

    def _build_estimator(self) -> Any:
        if self.estimator_kind == "logistic":
            regularization, self.sklearn_regularization_parameter_ = (
                self._logistic_regularization_kwargs()
            )
            return LogisticRegression(
                C=self.c,
                solver=_LOGISTIC_SOLVER_BY_PENALTY[self.penalty],
                max_iter=self.max_iter,
                random_state=self.random_state,
                **regularization,
            )
        if self.estimator_kind == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=1,
            )
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=self.max_iter,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            random_state=self.random_state,
        )

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        labels: Sequence[Any],
        *,
        query_ids: Sequence[Any],
        baseline_scores: Sequence[float] | None = None,
        sample_ids: Sequence[Any] | None = None,
    ) -> "PointwiseRanker":
        del baseline_scores
        matrix, names = _feature_frame(features)
        y = _binary_labels(labels)
        if len(matrix) != len(y):
            raise ValueError("features and labels must have equal length")
        if len(np.unique(y)) != 2:
            raise ValueError("pointwise ranker requires both label classes")
        queries = _query_array(query_ids, length=len(y))
        if sample_ids is not None and len(sample_ids) != len(y):
            raise ValueError("sample_ids length must match candidate rows")
        weights = _query_balanced_weights(y, queries, balance_classes=True)

        self.feature_names_ = names
        self.scaler_ = StandardScaler() if self.estimator_kind == "logistic" else None
        design = self.scaler_.fit_transform(matrix) if self.scaler_ is not None else matrix
        self.estimator_ = self._build_estimator()
        self.estimator_.fit(design, y.astype(np.int64), sample_weight=weights)
        self.n_fit_samples_ = len(y)
        self.fit_sample_ids_ = tuple(map(str, sample_ids)) if sample_ids is not None else ()
        self._importance_ = self._compute_importance(design, y, weights)
        self._metadata_ = {
            "model_kind": f"pointwise_{self.estimator_kind}",
            "objective": "query-balanced pointwise binary classification",
            "fit_scope": "rows explicitly supplied to fit only",
            "n_fit_samples": len(y),
            "fit_sample_ids": list(self.fit_sample_ids_),
            "query_weighting": "equal total weight per query",
            "is_learning_to_rank": False,
            "lambda_mart": False,
            "ranking_backend_unavailable_reason": (
                HIST_GRADIENT_BOOSTING_FALLBACK_REASON
                if self.estimator_kind == "hist_gradient_boosting"
                else None
            ),
            "feature_importance_kind": (
                "absolute_standardized_coefficient"
                if self.estimator_kind == "logistic"
                else "mean_decrease_impurity"
                if self.estimator_kind == "random_forest"
                else "deterministic_training_permutation_brier"
            ),
        }
        if self.estimator_kind == "logistic":
            solver = _LOGISTIC_SOLVER_BY_PENALTY[self.penalty]
            self._metadata_.update(
                {
                    "penalty": self.penalty,
                    "solver": solver,
                    "C": self.c,
                    "regularization": {
                        "penalty": self.penalty,
                        "inverse_strength_C": self.c,
                        "solver": solver,
                        "sklearn_parameter": self.sklearn_regularization_parameter_,
                    },
                }
            )
        return self

    def _probability(self, design: np.ndarray) -> np.ndarray:
        probabilities = self.estimator_.predict_proba(design)[:, 1]
        return np.asarray(probabilities, dtype=np.float64)

    def _compute_importance(
        self, design: np.ndarray, labels: np.ndarray, weights: np.ndarray
    ) -> dict[str, float]:
        if self.estimator_kind == "logistic":
            values = self.estimator_.coef_.reshape(-1)
        elif self.estimator_kind == "random_forest":
            values = self.estimator_.feature_importances_
        else:
            baseline = self._probability(design)
            base_loss = float(np.average((baseline - labels) ** 2, weights=weights))
            rng = np.random.default_rng(self.random_state)
            values = np.zeros(design.shape[1], dtype=np.float64)
            for column in range(design.shape[1]):
                permuted = design.copy()
                order = rng.permutation(len(design))
                permuted[:, column] = permuted[order, column]
                permuted_loss = float(
                    np.average((self._probability(permuted) - labels) ** 2, weights=weights)
                )
                values[column] = max(0.0, permuted_loss - base_loss)
        return _normalize_importance(self.feature_names_, values)

    def predict_scores(
        self,
        features: pd.DataFrame | np.ndarray,
        *,
        query_ids: Sequence[Any] | None = None,
        baseline_scores: Sequence[float] | None = None,
    ) -> np.ndarray:
        del baseline_scores
        if not hasattr(self, "estimator_"):
            raise RuntimeError("ranker has not been fitted")
        matrix = _check_predict_features(features, self.feature_names_)
        if query_ids is not None:
            _query_array(query_ids, length=len(matrix))
        design = self.scaler_.transform(matrix) if self.scaler_ is not None else matrix
        if self.estimator_kind == "logistic":
            scores = self.estimator_.decision_function(design)
        else:
            scores = self._probability(design)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("pointwise ranker produced non-finite scores")
        return scores

    def feature_importance(self) -> dict[str, float]:
        if not hasattr(self, "_importance_"):
            raise RuntimeError("ranker has not been fitted")
        return dict(self._importance_)

    @property
    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "_metadata_"):
            raise RuntimeError("ranker has not been fitted")
        return dict(self._metadata_)


class LogisticRegressionRanker(PointwiseRanker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("logistic", **kwargs)


class RandomForestRanker(PointwiseRanker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("random_forest", **kwargs)


class HistGradientBoostingRanker(PointwiseRanker):
    """Pointwise sklearn fallback; deliberately not presented as LambdaMART."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("hist_gradient_boosting", **kwargs)


def _load_xgboost_module() -> Any:
    """Load the optional ranking backend without making module import mandatory."""

    try:
        import xgboost  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "XGBoostRanker requires the optional 'xgboost' package with XGBRanker; "
            "no pointwise fallback will be substituted"
        ) from exc
    if not hasattr(xgboost, "XGBRanker"):
        raise RuntimeError(
            "the installed 'xgboost' package does not provide XGBRanker; "
            "no pointwise fallback will be substituted"
        )
    return xgboost


class XGBoostRanker:
    """Optional XGBRanker backend trained with explicit query identifiers.

    ``rank:ndcg`` is the NDCG-scaled LambdaMART configuration.  The accepted
    ``rank:pairwise`` alternative is the unscaled pairwise logistic variant
    and is recorded separately instead of being mislabeled as NDCG-scaled.
    """

    _VALID_OBJECTIVES = {"rank:ndcg", "rank:pairwise"}

    def __init__(
        self,
        *,
        objective: str = "rank:ndcg",
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 0,
        tree_method: str = "hist",
    ) -> None:
        self.objective = str(objective).strip().lower()
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.random_state = int(random_state)
        self.tree_method = str(tree_method).strip()

        if self.objective not in self._VALID_OBJECTIVES:
            raise ValueError("objective must be 'rank:ndcg' or 'rank:pairwise'")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be strictly positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be strictly positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and strictly positive")
        for name, value in (
            ("subsample", self.subsample),
            ("colsample_bytree", self.colsample_bytree),
        ):
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name, value in (
            ("reg_alpha", self.reg_alpha),
            ("reg_lambda", self.reg_lambda),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.tree_method:
            raise ValueError("tree_method must be non-empty")

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        labels: Sequence[Any],
        *,
        query_ids: Sequence[Any],
        baseline_scores: Sequence[float] | None = None,
        sample_ids: Sequence[Any] | None = None,
    ) -> "XGBoostRanker":
        del baseline_scores
        matrix, names = _feature_frame(features)
        y = _binary_labels(labels)
        if len(matrix) != len(y):
            raise ValueError("features and labels must have equal length")
        queries = _query_array(query_ids, length=len(y))
        if sample_ids is not None and len(sample_ids) != len(y):
            raise ValueError("sample_ids length must match candidate rows")

        string_queries = np.asarray([str(value) for value in queries], dtype=object)
        query_values = tuple(dict.fromkeys(string_queries.tolist()))
        query_codes = {query_id: index for index, query_id in enumerate(query_values)}
        qid = np.asarray(
            [query_codes[query_id] for query_id in string_queries], dtype=np.int64
        )
        training_order = np.argsort(qid, kind="stable")
        sorted_qid = qid[training_order]
        group_sizes = np.bincount(sorted_qid, minlength=len(query_values)).astype(
            np.int64
        )
        varied_queries = [
            query_id
            for query_id, indices in _query_slices(queries)
            if len(np.unique(y[indices])) > 1
        ]
        if not varied_queries:
            raise ValueError(
                "XGBoostRanker requires at least one query with distinct relevance labels"
            )

        xgboost = _load_xgboost_module()
        self.feature_names_ = names
        self.estimator_ = xgboost.XGBRanker(
            objective=self.objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            tree_method=self.tree_method,
            importance_type="gain",
            n_jobs=1,
            verbosity=0,
        )
        self.estimator_.fit(matrix[training_order], y[training_order], qid=sorted_qid)

        raw_importance = getattr(self.estimator_, "feature_importances_", None)
        if raw_importance is None:
            raise RuntimeError("XGBRanker did not expose feature_importances_")
        self._importance_ = _normalize_importance(names, raw_importance)
        self.n_fit_samples_ = len(y)
        self.fit_sample_ids_ = (
            tuple(map(str, sample_ids)) if sample_ids is not None else ()
        )
        self._metadata_ = {
            "model_kind": "xgboost_ranker",
            "backend": "xgboost.XGBRanker",
            "backend_version": str(getattr(xgboost, "__version__", "unknown")),
            "objective": self.objective,
            "is_learning_to_rank": True,
            "is_pointwise": False,
            "lambda_mart": True,
            "lambda_mart_kind": (
                "ndcg_scaled_lambda_rank"
                if self.objective == "rank:ndcg"
                else "unscaled_pairwise_logistic"
            ),
            "fit_scope": "rows explicitly supplied to fit only",
            "n_fit_samples": len(y),
            "fit_sample_ids": list(self.fit_sample_ids_),
            "feature_modality": "scalar_tabular_only",
            "query_group_parameter": "qid",
            "query_ordering": "stable sort by first-seen encoded query id",
            "n_queries": len(query_values),
            "query_group_sizes": group_sizes.tolist(),
            "n_queries_with_distinct_labels": len(varied_queries),
            "feature_importance_kind": "xgboost_gain",
            "n_jobs": 1,
            "random_state": self.random_state,
            "single_threaded_execution": True,
            "fallback_backend": None,
        }
        return self

    def predict_scores(
        self,
        features: pd.DataFrame | np.ndarray,
        *,
        query_ids: Sequence[Any] | None = None,
        baseline_scores: Sequence[float] | None = None,
    ) -> np.ndarray:
        del baseline_scores
        if not hasattr(self, "estimator_"):
            raise RuntimeError("ranker has not been fitted")
        matrix = _check_predict_features(features, self.feature_names_)
        if query_ids is not None:
            _query_array(query_ids, length=len(matrix))
        scores = np.asarray(self.estimator_.predict(matrix), dtype=np.float64).reshape(
            -1
        )
        if len(scores) != len(matrix) or not np.all(np.isfinite(scores)):
            raise RuntimeError("XGBRanker produced invalid scores")
        return scores

    def feature_importance(self) -> dict[str, float]:
        if not hasattr(self, "_importance_"):
            raise RuntimeError("ranker has not been fitted")
        return dict(self._importance_)

    @property
    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "_metadata_"):
            raise RuntimeError("ranker has not been fitted")
        return dict(self._metadata_)


# Descriptive alias for experiment configuration and reports.
XGBoostLambdaMARTRanker = XGBoostRanker


class LinearResidualBCE:
    """Linear residual model optimized with weighted binary cross entropy."""

    def __init__(
        self,
        *,
        calibration: str = "platt",
        l2: float = 1e-3,
        max_iter: int = 300,
        random_state: int = 0,
    ) -> None:
        self.calibration = str(calibration)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        if self.l2 < 0.0:
            raise ValueError("l2 must be non-negative")

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        labels: Sequence[Any],
        *,
        query_ids: Sequence[Any],
        baseline_scores: Sequence[float] | None = None,
        sample_ids: Sequence[Any] | None = None,
    ) -> "LinearResidualBCE":
        if baseline_scores is None:
            raise ValueError("LinearResidualBCE requires baseline_scores")
        matrix, names = _feature_frame(features)
        y = _binary_labels(labels)
        if len(matrix) != len(y):
            raise ValueError("features and labels must have equal length")
        queries = _query_array(query_ids, length=len(y))
        baseline = _finite_vector(baseline_scores, name="baseline_scores")
        if len(baseline) != len(y):
            raise ValueError("baseline_scores length must match candidate rows")
        weights = _query_balanced_weights(y, queries, balance_classes=True)
        fit_ids = sample_ids if sample_ids is not None else query_ids
        if len(fit_ids) != len(y):
            raise ValueError("sample_ids length must match candidate rows")

        self.feature_names_ = names
        self.scaler_ = StandardScaler().fit(matrix)
        design = self.scaler_.transform(matrix)
        self.calibrator_ = TrainOnlyScoreCalibrator(
            self.calibration, random_state=self.random_state
        ).fit(
            baseline,
            y,
            sample_weight=weights,
            fit_sample_ids=fit_ids,
        )
        probability = self.calibrator_.predict_proba(baseline)
        offset = np.log(probability) - np.log1p(-probability)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            coefficient = parameters[:-1]
            intercept = parameters[-1]
            logits = offset + design @ coefficient + intercept
            error = expit(logits) - y
            weighted_error = weights * error / weights.sum()
            loss = float(
                np.sum(weights * (np.logaddexp(0.0, logits) - y * logits))
                / weights.sum()
                + 0.5 * self.l2 * np.dot(coefficient, coefficient)
            )
            gradient = np.concatenate(
                [
                    design.T @ weighted_error + self.l2 * coefficient,
                    [weighted_error.sum()],
                ]
            )
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1] + 1, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter},
        )
        if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
            raise RuntimeError(f"residual BCE optimization failed: {result.message}")
        self.coefficient_ = result.x[:-1]
        self.intercept_ = float(result.x[-1])
        self._importance_ = _normalize_importance(names, self.coefficient_)
        self._metadata_ = {
            "model_kind": "linear_residual_bce",
            "objective": "query-balanced weighted binary cross entropy",
            "residual_formula": "logit(calibrated_baseline) + w^T standardized_x + b",
            "query_weighting": "equal total weight per query",
            "calibration": self.calibrator_.metadata,
            "optimizer": "scipy L-BFGS-B",
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "feature_importance_kind": "absolute_standardized_coefficient",
            "is_learning_to_rank": False,
        }
        return self

    def predict_scores(
        self,
        features: pd.DataFrame | np.ndarray,
        *,
        query_ids: Sequence[Any] | None = None,
        baseline_scores: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not hasattr(self, "coefficient_"):
            raise RuntimeError("ranker has not been fitted")
        if baseline_scores is None:
            raise ValueError("LinearResidualBCE requires baseline_scores")
        matrix = _check_predict_features(features, self.feature_names_)
        if query_ids is not None:
            _query_array(query_ids, length=len(matrix))
        baseline = _finite_vector(baseline_scores, name="baseline_scores")
        if len(baseline) != len(matrix):
            raise ValueError("baseline_scores length must match candidate rows")
        probability = self.calibrator_.predict_proba(baseline)
        offset = np.log(probability) - np.log1p(-probability)
        scores = (
            offset
            + self.scaler_.transform(matrix) @ self.coefficient_
            + self.intercept_
        )
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("residual BCE ranker produced non-finite scores")
        return np.asarray(scores, dtype=np.float64)

    def feature_importance(self) -> dict[str, float]:
        if not hasattr(self, "_importance_"):
            raise RuntimeError("ranker has not been fitted")
        return dict(self._importance_)

    @property
    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "_metadata_"):
            raise RuntimeError("ranker has not been fitted")
        return dict(self._metadata_)


@dataclass(frozen=True)
class PairwiseBatch:
    """Same-query positive-minus-negative differences for RankNet."""

    feature_differences: np.ndarray
    baseline_differences: np.ndarray
    weights: np.ndarray
    query_ids: tuple[str, ...]
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    pairs_per_query: Mapping[str, int]
    skipped_no_positive: tuple[str, ...]
    skipped_no_negative: tuple[str, ...]

    @property
    def n_pairs(self) -> int:
        return int(len(self.weights))


def build_pairwise_batch(
    features: pd.DataFrame | np.ndarray,
    labels: Sequence[Any],
    *,
    query_ids: Sequence[Any],
    baseline_scores: Sequence[float] | None = None,
    max_pairs_per_query: int | None = None,
    random_state: int = 0,
) -> PairwiseBatch:
    """Construct RankNet pairs without crossing query boundaries."""

    matrix, _ = _feature_frame(features)
    y = _binary_labels(labels)
    if len(matrix) != len(y):
        raise ValueError("features and labels must have equal length")
    queries = _query_array(query_ids, length=len(y))
    baseline = (
        np.zeros(len(y), dtype=np.float64)
        if baseline_scores is None
        else _finite_vector(baseline_scores, name="baseline_scores")
    )
    if len(baseline) != len(y):
        raise ValueError("baseline_scores length must match candidate rows")
    if max_pairs_per_query is not None and max_pairs_per_query <= 0:
        raise ValueError("max_pairs_per_query must be positive")

    rng = np.random.default_rng(int(random_state))
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    pair_queries: list[str] = []
    pairs_per_query: dict[str, int] = {}
    skipped_no_positive: list[str] = []
    skipped_no_negative: list[str] = []

    for query_id, indices in _query_slices(queries):
        positives = indices[y[indices] == 1.0]
        negatives = indices[y[indices] == 0.0]
        if len(positives) == 0:
            skipped_no_positive.append(query_id)
            continue
        if len(negatives) == 0:
            skipped_no_negative.append(query_id)
            continue
        local_pairs = np.asarray(
            [(positive, negative) for positive in positives for negative in negatives],
            dtype=np.int64,
        )
        if max_pairs_per_query is not None and len(local_pairs) > max_pairs_per_query:
            chosen = np.sort(
                rng.choice(len(local_pairs), size=max_pairs_per_query, replace=False)
            )
            local_pairs = local_pairs[chosen]
        positive_indices.extend(local_pairs[:, 0].tolist())
        negative_indices.extend(local_pairs[:, 1].tolist())
        pair_queries.extend([query_id] * len(local_pairs))
        pairs_per_query[query_id] = len(local_pairs)

    positive_array = np.asarray(positive_indices, dtype=np.int64)
    negative_array = np.asarray(negative_indices, dtype=np.int64)
    if len(positive_array):
        differences = matrix[positive_array] - matrix[negative_array]
        baseline_differences = baseline[positive_array] - baseline[negative_array]
        weights = np.asarray(
            [1.0 / pairs_per_query[query_id] for query_id in pair_queries],
            dtype=np.float64,
        )
    else:
        differences = np.empty((0, matrix.shape[1]), dtype=np.float64)
        baseline_differences = np.empty(0, dtype=np.float64)
        weights = np.empty(0, dtype=np.float64)
    return PairwiseBatch(
        feature_differences=differences,
        baseline_differences=baseline_differences,
        weights=weights,
        query_ids=tuple(pair_queries),
        positive_indices=positive_array,
        negative_indices=negative_array,
        pairs_per_query=pairs_per_query,
        skipped_no_positive=tuple(skipped_no_positive),
        skipped_no_negative=tuple(skipped_no_negative),
    )


class LinearPairwiseRankNet:
    """Linear RankNet scorer with same-query pairs and equal query weight."""

    def __init__(
        self,
        *,
        calibration: str = "platt",
        l2: float = 1e-3,
        max_iter: int = 300,
        max_pairs_per_query: int | None = None,
        random_state: int = 0,
    ) -> None:
        self.calibration = str(calibration)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.max_pairs_per_query = max_pairs_per_query
        self.random_state = int(random_state)
        if self.l2 < 0.0:
            raise ValueError("l2 must be non-negative")

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        labels: Sequence[Any],
        *,
        query_ids: Sequence[Any],
        baseline_scores: Sequence[float] | None = None,
        sample_ids: Sequence[Any] | None = None,
    ) -> "LinearPairwiseRankNet":
        matrix, names = _feature_frame(features)
        y = _binary_labels(labels)
        if len(matrix) != len(y):
            raise ValueError("features and labels must have equal length")
        queries = _query_array(query_ids, length=len(y))
        self.feature_names_ = names
        self.scaler_ = StandardScaler().fit(matrix)
        design = self.scaler_.transform(matrix)

        calibrated_logit: np.ndarray | None = None
        self.uses_baseline_ = baseline_scores is not None
        if baseline_scores is not None:
            baseline = _finite_vector(baseline_scores, name="baseline_scores")
            if len(baseline) != len(y):
                raise ValueError("baseline_scores length must match candidate rows")
            fit_ids = sample_ids if sample_ids is not None else query_ids
            if len(fit_ids) != len(y):
                raise ValueError("sample_ids length must match candidate rows")
            point_weights = _query_balanced_weights(y, queries, balance_classes=True)
            self.calibrator_ = TrainOnlyScoreCalibrator(
                self.calibration, random_state=self.random_state
            ).fit(
                baseline,
                y,
                sample_weight=point_weights,
                fit_sample_ids=fit_ids,
            )
            probability = self.calibrator_.predict_proba(baseline)
            calibrated_logit = np.log(probability) - np.log1p(-probability)

        batch = build_pairwise_batch(
            design,
            y,
            query_ids=queries,
            baseline_scores=calibrated_logit,
            max_pairs_per_query=self.max_pairs_per_query,
            random_state=self.random_state,
        )
        if batch.n_pairs == 0:
            raise ValueError("RankNet requires at least one same-query positive/negative pair")

        def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
            margin = batch.baseline_differences + batch.feature_differences @ coefficient
            weighted_factor = batch.weights / batch.weights.sum()
            loss = float(
                np.sum(weighted_factor * np.logaddexp(0.0, -margin))
                + 0.5 * self.l2 * np.dot(coefficient, coefficient)
            )
            gradient = (
                batch.feature_differences.T
                @ (-weighted_factor * expit(-margin))
                + self.l2 * coefficient
            )
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter},
        )
        if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
            raise RuntimeError(f"RankNet optimization failed: {result.message}")
        self.coefficient_ = result.x
        self.pairwise_batch_ = batch
        self._importance_ = _normalize_importance(names, result.x)
        self._metadata_ = {
            "model_kind": "linear_pairwise_ranknet",
            "objective": "softplus(-(score_positive - score_negative))",
            "pair_scope": "same query only",
            "query_weighting": "equal total pair weight per eligible query",
            "n_pairs": batch.n_pairs,
            "pairs_per_query": dict(batch.pairs_per_query),
            "skipped_no_positive": list(batch.skipped_no_positive),
            "skipped_no_negative": list(batch.skipped_no_negative),
            "uses_calibrated_baseline_residual": self.uses_baseline_,
            "calibration": self.calibrator_.metadata if self.uses_baseline_ else None,
            "optimizer": "scipy L-BFGS-B",
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "feature_importance_kind": "absolute_standardized_coefficient",
            "is_learning_to_rank": True,
        }
        return self

    def predict_scores(
        self,
        features: pd.DataFrame | np.ndarray,
        *,
        query_ids: Sequence[Any] | None = None,
        baseline_scores: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not hasattr(self, "coefficient_"):
            raise RuntimeError("ranker has not been fitted")
        matrix = _check_predict_features(features, self.feature_names_)
        if query_ids is not None:
            _query_array(query_ids, length=len(matrix))
        scores = self.scaler_.transform(matrix) @ self.coefficient_
        if self.uses_baseline_:
            if baseline_scores is None:
                raise ValueError("this fitted RankNet model requires baseline_scores")
            baseline = _finite_vector(baseline_scores, name="baseline_scores")
            if len(baseline) != len(matrix):
                raise ValueError("baseline_scores length must match candidate rows")
            probability = self.calibrator_.predict_proba(baseline)
            scores = scores + np.log(probability) - np.log1p(-probability)
        elif baseline_scores is not None:
            raise ValueError("model was fitted without baseline_scores")
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("RankNet produced non-finite scores")
        return np.asarray(scores, dtype=np.float64)

    def feature_importance(self) -> dict[str, float]:
        if not hasattr(self, "_importance_"):
            raise RuntimeError("ranker has not been fitted")
        return dict(self._importance_)

    @property
    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "_metadata_"):
            raise RuntimeError("ranker has not been fitted")
        return dict(self._metadata_)


def stable_rank_order(
    scores: Sequence[float], candidate_ids: Sequence[Any]
) -> np.ndarray:
    """Sort by descending score and lexicographic candidate id on exact ties."""

    values = _finite_vector(scores, name="scores")
    identifiers = np.asarray([str(value) for value in candidate_ids], dtype=object)
    if len(identifiers) != len(values):
        raise ValueError("candidate_ids length must match scores")
    return np.lexsort((identifiers, -values)).astype(np.int64)


def make_tabular_ranker(kind: str, **kwargs: Any) -> Any:
    """Factory used by experiment matrices while preserving a common API."""

    normalized = str(kind).lower().replace("-", "_")
    factories = {
        "logistic": LogisticRegressionRanker,
        "logistic_regression": LogisticRegressionRanker,
        "random_forest": RandomForestRanker,
        "hist_gradient_boosting": HistGradientBoostingRanker,
        "histgb_fallback": HistGradientBoostingRanker,
        "xgboost_ranker": XGBoostRanker,
        "xgb_ranker": XGBoostRanker,
        "xgboost_lambdamart": XGBoostRanker,
        "lambdamart": XGBoostRanker,
        "linear_residual_bce": LinearResidualBCE,
        "linear_pairwise_ranknet": LinearPairwiseRankNet,
    }
    if normalized not in factories:
        raise ValueError(f"unknown tabular ranker kind: {kind}")
    return factories[normalized](**kwargs)


__all__ = [
    "FORBIDDEN_COLUMN_TOKENS",
    "HIST_GRADIENT_BOOSTING_FALLBACK_REASON",
    "HistGradientBoostingRanker",
    "LinearPairwiseRankNet",
    "LinearResidualBCE",
    "LogisticRegressionRanker",
    "PairwiseBatch",
    "PointwiseRanker",
    "RandomForestRanker",
    "ScoreCalibrator",
    "TrainOnlyScoreCalibrator",
    "XGBoostLambdaMARTRanker",
    "XGBoostRanker",
    "assert_no_forbidden_columns",
    "build_pairwise_batch",
    "derive_q_features",
    "make_tabular_ranker",
    "scan_forbidden_columns",
    "stable_rank_order",
]
