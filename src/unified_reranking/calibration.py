"""Monotone route-specific calibration with persisted grouped OOF predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression


PROBABILITY_EPSILON = 1e-4


def _vectors(scores: Any, labels: Any | None = None, weights: Any | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("calibration scores must be a non-empty finite vector")
    y = None if labels is None else np.asarray(labels, dtype=np.float64).reshape(-1)
    w = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if y is not None and (len(y) != len(x) or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all()):
        raise ValueError("calibration labels must be finite binary values")
    if w is not None and (len(w) != len(x) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0):
        raise ValueError("calibration weights must be finite, non-negative, and non-zero")
    return x, y, w


@dataclass
class MonotonePlattCalibrator:
    intercept_: float | None = None
    slope_: float | None = None
    constant_: float | None = None

    def fit(self, scores: Any, labels: Any, sample_weight: Any | None = None) -> "MonotonePlattCalibrator":
        x, y, weights = _vectors(scores, labels, sample_weight)
        assert y is not None
        w = np.ones(len(x), dtype=np.float64) if weights is None else weights
        w = w / w.sum()
        if np.unique(y).size == 1:
            self.constant_ = float(np.clip(y[0], PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))
            self.intercept_, self.slope_ = None, None
            return self

        prevalence = float(np.clip(np.average(y, weights=w), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))

        def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = float(parameter[0])
            slope = float(np.exp(np.clip(parameter[1], -30, 30)))
            logits = intercept + slope * x
            probability = expit(logits)
            loss = float(np.sum(w * (np.logaddexp(0.0, logits) - y * logits)))
            error = w * (probability - y)
            gradient = np.asarray([error.sum(), slope * np.dot(error, x)], dtype=np.float64)
            return loss, gradient

        initial = np.asarray([math.log(prevalence / (1 - prevalence)), 0.0])
        result = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 1000})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(f"monotone Platt fit failed: {result.message}")
        self.intercept_ = float(result.x[0])
        self.slope_ = float(np.exp(np.clip(result.x[1], -30, 30)))
        self.constant_ = None
        if self.slope_ <= 0:
            raise RuntimeError("monotone Platt slope is not positive")
        return self

    def predict(self, scores: Any) -> np.ndarray:
        x, _, _ = _vectors(scores)
        if self.constant_ is not None:
            probability = np.full(len(x), self.constant_, dtype=np.float64)
        elif self.intercept_ is not None and self.slope_ is not None:
            probability = expit(self.intercept_ + self.slope_ * x)
        else:
            raise RuntimeError("calibrator has not been fitted")
        return np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)

    def serialize(self) -> dict[str, Any]:
        return {"kind": "monotone_platt", "intercept": self.intercept_, "slope": self.slope_, "constant": self.constant_}


@dataclass
class MonotoneIsotonicCalibrator:
    model_: IsotonicRegression | None = None
    constant_: float | None = None

    def fit(self, scores: Any, labels: Any, sample_weight: Any | None = None) -> "MonotoneIsotonicCalibrator":
        x, y, weights = _vectors(scores, labels, sample_weight)
        assert y is not None
        if np.unique(y).size == 1:
            self.constant_ = float(np.clip(y[0], PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))
            self.model_ = None
        else:
            self.model_ = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=PROBABILITY_EPSILON, y_max=1 - PROBABILITY_EPSILON)
            self.model_.fit(x, y, sample_weight=weights)
            self.constant_ = None
        return self

    def predict(self, scores: Any) -> np.ndarray:
        x, _, _ = _vectors(scores)
        if self.constant_ is not None:
            probability = np.full(len(x), self.constant_, dtype=np.float64)
        elif self.model_ is not None:
            probability = self.model_.predict(x)
        else:
            raise RuntimeError("calibrator has not been fitted")
        return np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": "isotonic",
            "constant": self.constant_,
            "x_thresholds": None if self.model_ is None else self.model_.X_thresholds_.tolist(),
            "y_thresholds": None if self.model_ is None else self.model_.y_thresholds_.tolist(),
        }


def make_calibrator(method: str) -> MonotonePlattCalibrator | MonotoneIsotonicCalibrator:
    if method == "platt":
        return MonotonePlattCalibrator()
    if method == "isotonic":
        return MonotoneIsotonicCalibrator()
    raise ValueError(f"unknown calibration method: {method}")


def calibrator_from_serialized(value: dict[str, Any]) -> MonotonePlattCalibrator | MonotoneIsotonicCalibrator:
    kind = value.get("kind")
    if kind == "monotone_platt":
        return MonotonePlattCalibrator(
            intercept_=value.get("intercept"), slope_=value.get("slope"), constant_=value.get("constant")
        )
    if kind == "isotonic":
        model = MonotoneIsotonicCalibrator(constant_=value.get("constant"))
        if value.get("x_thresholds") is not None:
            isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=PROBABILITY_EPSILON, y_max=1 - PROBABILITY_EPSILON)
            isotonic.X_thresholds_ = np.asarray(value["x_thresholds"], dtype=np.float64)
            isotonic.y_thresholds_ = np.asarray(value["y_thresholds"], dtype=np.float64)
            isotonic.X_min_ = float(isotonic.X_thresholds_.min())
            isotonic.X_max_ = float(isotonic.X_thresholds_.max())
            isotonic.f_ = lambda x: np.interp(x, isotonic.X_thresholds_, isotonic.y_thresholds_)
            model.model_ = isotonic
        return model
    raise ValueError(f"unknown serialized calibrator kind: {kind}")


def query_balanced_weights(sample_ids: Any) -> np.ndarray:
    ids = pd.Series(np.asarray(sample_ids).astype(str))
    counts = ids.map(ids.value_counts()).to_numpy(np.float64)
    weights = 1.0 / counts
    return weights / weights.sum()


def calibration_metrics(labels: Any, probabilities: Any, *, bins: int = 15) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("invalid calibration metric inputs")
    brier = float(np.mean((p - y) ** 2))
    nll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p))))
    edges = np.linspace(0.0, 1.0, bins + 1)
    indexes = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = indexes == index
        if selected.any():
            ece += float(selected.mean()) * abs(float(p[selected].mean()) - float(y[selected].mean()))
    return {"brier": brier, "nll": nll, "ece_15": ece}


def assert_order_invariant(frame: pd.DataFrame, probability_column: str) -> None:
    for sample_id, group in frame.groupby("sample_id", sort=False):
        native = group.sort_values(["native_rank", "candidate_id"], kind="mergesort")["candidate_id"].astype(str).tolist()
        calibrated = group.sort_values([probability_column, "native_rank", "candidate_id"], ascending=[False, True, True], kind="mergesort")["candidate_id"].astype(str).tolist()
        if native != calibrated:
            raise ValueError(f"calibration changed native order: {sample_id}")


def grouped_oof_calibration(
    train: pd.DataFrame,
    folds: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    required = {"sample_id", "candidate_id", "native_rank", "native_score", "candidate_success"}
    for name, frame in (("train", train), ("validation", validation)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} calibration table missing: {missing}")
    assignment = folds[["sample_id", "fold"]]
    work = train.merge(assignment, on="sample_id", validate="many_to_one")
    if len(work) != len(train) or work["fold"].isna().any():
        raise ValueError("fold assignment does not cover all calibration candidates")
    oof = work[["sample_id", "candidate_id", "native_rank", "native_score", "candidate_success", "fold"]].copy()
    validation_predictions = validation[["sample_id", "candidate_id", "native_rank", "native_score", "candidate_success"]].copy()
    serialized: dict[str, Any] = {"folds": {}, "full_train": {}}
    validation_metrics: dict[str, Any] = {}
    for method in ("platt", "isotonic"):
        column = f"calibrated_probability_{method}"
        oof[column] = np.nan
        for fold in sorted(work["fold"].unique()):
            fit = work.loc[work["fold"] != fold]
            held = work.loc[work["fold"] == fold]
            model = make_calibrator(method).fit(
                fit["native_score"], fit["candidate_success"], query_balanced_weights(fit["sample_id"])
            )
            oof.loc[oof["fold"] == fold, column] = model.predict(held["native_score"])
            serialized["folds"].setdefault(str(fold), {})[method] = model.serialize()
        if oof[column].isna().any():
            raise RuntimeError(f"OOF calibration did not predict every row: {method}")
        full = make_calibrator(method).fit(
            work["native_score"], work["candidate_success"], query_balanced_weights(work["sample_id"])
        )
        validation_predictions[column] = full.predict(validation["native_score"])
        serialized["full_train"][method] = full.serialize()
        assert_order_invariant(oof, column)
        assert_order_invariant(validation_predictions, column)
        validation_metrics[method] = calibration_metrics(validation_predictions["candidate_success"], validation_predictions[column])
    metric_names = ("brier", "nll", "ece_15")
    ranks = {
        method: sum(
            sorted((validation_metrics[name][metric], name) for name in validation_metrics).index((validation_metrics[method][metric], method))
            for metric in metric_names
        )
        for method in validation_metrics
    }
    selected = min(validation_metrics, key=lambda name: (ranks[name], validation_metrics[name]["brier"], name))
    for frame in (oof, validation_predictions):
        frame["calibrated_native_probability"] = frame[f"calibrated_probability_{selected}"]
        p = np.clip(frame["calibrated_native_probability"].to_numpy(float), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        frame["base_logit"] = np.log(p) - np.log1p(-p)
        assert_order_invariant(frame, "calibrated_native_probability")
    metadata = {
        "selection_rule": "lowest sum of Validation ranks over Brier/NLL/ECE; tie by Brier then method name",
        "selected_method": selected,
        "validation_metrics": validation_metrics,
        "metric_rank_sums": ranks,
        "probability_clip": [PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON],
        "native_rank_secondary_tie_break": True,
    }
    return oof, validation_predictions, serialized, metadata
