"""Backend-specific OOF score calibration for G1 and C1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int | bool], *, bins: int = 15,
    sample_weight: Sequence[float] | None = None,
) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape or not len(p):
        raise ValueError("probabilities and labels must be matching non-empty vectors")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    weight = (
        np.ones(len(p), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if weight.shape != p.shape or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("invalid calibration sample weights")
    weight = weight / weight.sum()
    result = 0.0
    for index in range(int(bins)):
        selected = (p >= edges[index]) & (
            p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1]
        )
        if selected.any():
            local_weight = weight[selected]
            mass = float(local_weight.sum())
            local_weight = local_weight / mass
            result += mass * abs(
                float(np.sum(local_weight * p[selected]))
                - float(np.sum(local_weight * y[selected]))
            )
    return float(result)


def calibration_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int | bool],
    *,
    sample_ids: Sequence[Any] | None = None,
) -> dict[str, float]:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(labels, dtype=int)
    weight = None
    if sample_ids is not None:
        ids = pd.Series(list(map(str, sample_ids)))
        if len(ids) != len(p):
            raise ValueError("sample_ids length must match calibration rows")
        weight = (1.0 / ids.map(ids.value_counts()).to_numpy(dtype=float))
        weight = weight / weight.sum()
    return {
        "brier": float(np.average((p - y) ** 2, weights=weight)),
        "ece_15": expected_calibration_error(p, y, bins=15, sample_weight=weight),
        "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=weight)),
    }


def _sample_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("sample_id")["stable_candidate_id"].transform("size")
    return 1.0 / counts.to_numpy(dtype=float)


@dataclass
class ScoreCalibrator:
    kind: str
    model: Any
    constant: float | None = None

    @classmethod
    def fit(cls, kind: str, frame: pd.DataFrame) -> "ScoreCalibrator":
        required = {"sample_id", "original_score", "candidate_correct"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"calibration frame missing columns: {missing}")
        x = frame["original_score"].to_numpy(dtype=float)
        y = frame["candidate_correct"].to_numpy(dtype=int)
        if not np.isfinite(x).all() or not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("invalid calibration score/label values")
        if len(np.unique(y)) < 2:
            return cls(kind="constant", model=None, constant=float(y.mean()))
        weight = _sample_weights(frame)
        if kind == "platt":
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            model.fit(x[:, None], y, sample_weight=weight)
            return cls(kind=kind, model=model)
        if kind == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(x, y, sample_weight=weight)
            return cls(kind=kind, model=model)
        raise ValueError(f"unknown calibrator kind: {kind}")

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        x = np.asarray(scores, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("calibration scores must be finite")
        if self.kind == "constant":
            assert self.constant is not None
            return np.full(len(x), self.constant, dtype=float)
        if self.kind == "platt":
            return self.model.predict_proba(x[:, None])[:, 1].astype(float)
        if self.kind == "isotonic":
            return np.asarray(self.model.predict(x), dtype=float)
        raise RuntimeError("unknown fitted calibrator")

    def artifact(self) -> dict[str, Any]:
        if self.kind == "constant":
            return {"kind": "constant", "constant": float(self.constant)}
        if self.kind == "platt":
            return {
                "kind": "platt",
                "coefficient": float(self.model.coef_[0, 0]),
                "intercept": float(self.model.intercept_[0]),
            }
        return {
            "kind": "isotonic",
            "x_thresholds": np.asarray(self.model.X_thresholds_, dtype=float).tolist(),
            "y_thresholds": np.asarray(self.model.y_thresholds_, dtype=float).tolist(),
            "out_of_bounds": "clip",
        }

    @classmethod
    def from_artifact(cls, payload: Mapping[str, Any]) -> "ScoreCalibrator":
        kind = str(payload["kind"])
        if kind == "constant":
            return cls(kind=kind, model=None, constant=float(payload["constant"]))
        if kind == "platt":
            model = LogisticRegression()
            model.coef_ = np.asarray([[float(payload["coefficient"])]])
            model.intercept_ = np.asarray([float(payload["intercept"])])
            model.classes_ = np.asarray([0, 1])
            model.n_features_in_ = 1
            return cls(kind=kind, model=model)
        if kind == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.X_thresholds_ = np.asarray(payload["x_thresholds"], dtype=float)
            model.y_thresholds_ = np.asarray(payload["y_thresholds"], dtype=float)
            model.X_min_ = float(model.X_thresholds_[0])
            model.X_max_ = float(model.X_thresholds_[-1])
            model.f_ = None
            # sklearn rebuilds its interpolation function on first predict only
            # after fit; make it explicit for a JSON-only artifact.
            from scipy.interpolate import interp1d

            model.f_ = interp1d(
                model.X_thresholds_,
                model.y_thresholds_,
                kind="linear",
                bounds_error=False,
                fill_value=(model.y_thresholds_[0], model.y_thresholds_[-1]),
            )
            return cls(kind=kind, model=model)
        raise ValueError(f"unknown calibrator artifact: {kind}")


def fit_backend_oof_calibration(
    frame: pd.DataFrame,
    *,
    backend: str,
    folds: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any], ScoreCalibrator]:
    """Choose Platt/isotonic using scene-held-out OOF predictions only."""

    selected = frame.loc[frame["backend"].eq(backend)].copy()
    if selected.empty or selected["scene_id"].nunique() < folds:
        raise ValueError(f"{backend}: insufficient scenes for {folds}-fold calibration")
    splitter = GroupKFold(n_splits=int(folds))
    predictions = {kind: np.full(len(selected), np.nan) for kind in ("platt", "isotonic")}
    fold_records: list[dict[str, Any]] = []
    for fold, (train_index, held_index) in enumerate(
        splitter.split(selected, groups=selected["scene_id"].astype(str))
    ):
        train = selected.iloc[train_index]
        held = selected.iloc[held_index]
        overlap = set(train["scene_id"].astype(str)) & set(held["scene_id"].astype(str))
        if overlap:
            raise AssertionError("scene leakage in calibrator OOF")
        record: dict[str, Any] = {
            "fold": int(fold),
            "train_scenes": int(train["scene_id"].nunique()),
            "held_scenes": int(held["scene_id"].nunique()),
        }
        for kind in predictions:
            calibrator = ScoreCalibrator.fit(kind, train)
            values = calibrator.predict(held["original_score"].to_numpy(dtype=float))
            predictions[kind][held_index] = values
            record[kind] = calibration_metrics(
                values,
                held["candidate_correct"],
                sample_ids=held["sample_id"],
            )
        fold_records.append(record)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise AssertionError("OOF calibration did not cover every candidate")
    metrics = {
        kind: calibration_metrics(
            values,
            selected["candidate_correct"],
            sample_ids=selected["sample_id"],
        )
        for kind, values in predictions.items()
    }
    selected_kind = min(
        metrics,
        key=lambda kind: (
            metrics[kind]["brier"],
            metrics[kind]["ece_15"],
            float(np.std([row[kind]["brier"] for row in fold_records])),
            0 if kind == "platt" else 1,
        ),
    )
    output = selected.copy()
    output["source_score_calibrated_oof"] = predictions[selected_kind]
    output["calibration_fold"] = -1
    for fold, (_, held_index) in enumerate(
        splitter.split(selected, groups=selected["scene_id"].astype(str))
    ):
        output.iloc[held_index, output.columns.get_loc("calibration_fold")] = fold
    full = ScoreCalibrator.fit(selected_kind, selected)
    artifact = {
        "backend": backend,
        "fit_scope": "train scene-grouped OOF",
        "folds": int(folds),
        "candidate_rows": int(len(selected)),
        "scene_count": int(selected["scene_id"].nunique()),
        "candidate_calibrators": metrics,
        "fold_metrics": fold_records,
        "selected_kind": selected_kind,
        "selection_rule": ["lower_brier", "lower_ece", "fold_stability", "simplicity"],
        "beta_calibration": "not_run_raw_backend_scores_are_not_probabilities",
        "full_train_calibrator": full.artifact(),
    }
    return output, artifact, full


def apply_backend_calibrators(
    frame: pd.DataFrame, calibrators: Mapping[str, ScoreCalibrator]
) -> pd.DataFrame:
    output = frame.copy()
    output["source_score_calibrated"] = np.nan
    for backend, calibrator in calibrators.items():
        selected = output["backend"].eq(backend)
        output.loc[selected, "source_score_calibrated"] = calibrator.predict(
            output.loc[selected, "original_score"].to_numpy(dtype=float)
        )
    if not np.isfinite(output["source_score_calibrated"].to_numpy(dtype=float)).all():
        raise ValueError("missing backend-specific calibrated scores")
    return output
