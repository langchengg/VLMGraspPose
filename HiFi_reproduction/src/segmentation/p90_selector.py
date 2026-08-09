"""Grouped OOF lightweight selectors for strict IoU > 0.90 candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .selective_sam3_vg.metrics import summarize_ious


FORBIDDEN_FEATURE_TOKENS = (
    "candidate_iou",
    "continuous_iou",
    "y70",
    "y80",
    "y90",
    "ground_truth",
    "answer_instance",
    "target_instance_id",
)
IDENTITY_COLUMNS = {
    "sample_id",
    "candidate_id",
    "split",
    "scene_id",
    "frame_id",
    "rgb_sha256",
    "query",
    "source_variant",
    "best_reference_candidate_id",
    "eligible_final",
}
CATEGORICAL_COLUMNS = (
    "source_family",
    "query_type",
    "target_category",
    "absolute_location_type",
    "relation_type",
)


@dataclass
class FeatureEncoder:
    numeric_columns: list[str]
    categorical_values: dict[str, list[str]]
    medians: dict[str, float]
    feature_names: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FeatureEncoder":
        forbidden = [
            column
            for column in frame.columns
            if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
        ]
        candidate_columns = [
            column
            for column in frame.columns
            if column not in IDENTITY_COLUMNS
            and column not in CATEGORICAL_COLUMNS
            and column not in forbidden
        ]
        numeric = [
            column
            for column in candidate_columns
            if pd.api.types.is_numeric_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        ]
        medians = {}
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            medians[column] = median
        categories = {
            column: sorted(frame[column].fillna("__MISSING__").astype(str).unique())
            for column in CATEGORICAL_COLUMNS
            if column in frame
        }
        names = list(numeric)
        for column, values in categories.items():
            names.extend(f"{column}={value}" for value in values)
        return cls(numeric, categories, medians, names)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.numeric_columns:
            # Per-sample feature tables intentionally use sparse schemas for
            # query-specific relation features.  Deployment must therefore
            # apply the same missing-value policy used for NaNs seen at fit
            # time instead of requiring every training column to be present.
            numeric_frame = frame.reindex(columns=self.numeric_columns)
            numeric = np.column_stack(
                [
                    pd.to_numeric(numeric_frame[column], errors="coerce")
                    .fillna(self.medians[column])
                    .to_numpy(dtype=np.float32)
                    for column in self.numeric_columns
                ]
            )
            numeric[~np.isfinite(numeric)] = 0.0
            blocks.append(numeric)
        for column, values in self.categorical_values.items():
            observed = (
                frame[column].fillna("__MISSING__").astype(str).to_numpy()
                if column in frame
                else np.full(len(frame), "__MISSING__", dtype=object)
            )
            blocks.append(
                np.column_stack([observed == value for value in values]).astype(np.float32)
            )
        if not blocks:
            raise ValueError("selector feature encoder has no usable columns")
        return np.concatenate(blocks, axis=1)


def inverse_candidate_weights(frame: pd.DataFrame, label: np.ndarray) -> np.ndarray:
    counts = frame.groupby("sample_id")["candidate_id"].transform("count").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    positives = max(int(np.count_nonzero(label)), 1)
    negatives = max(int(len(label) - positives), 1)
    class_weight = np.where(label > 0, len(label) / (2.0 * positives), len(label) / (2.0 * negatives))
    weights *= class_weight
    weights *= len(weights) / max(float(weights.sum()), 1e-12)
    return weights.astype(np.float64)


def p90_labels(iou: np.ndarray) -> np.ndarray:
    return (np.asarray(iou, dtype=np.float64) > 0.90).astype(np.int8)


def _models(seed: int) -> tuple[Any, Any, Any]:
    return (
        LogisticRegression(C=0.2, max_iter=500, random_state=seed, solver="lbfgs"),
        HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ),
        HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            loss="squared_error",
            random_state=seed,
        ),
    )


def grouped_oof_predictions(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = frame[frame["eligible_final"]].copy().reset_index(drop=True)
    if data["frame_id"].nunique() < folds:
        raise ValueError("insufficient frame groups for requested GroupKFold")
    y_iou = data["candidate_iou"].to_numpy(dtype=np.float64)
    y90 = p90_labels(y_iou)
    groups = data["frame_id"].astype(str).to_numpy()
    oof_m0 = np.full(len(data), np.nan, dtype=np.float64)
    oof_m1 = np.full(len(data), np.nan, dtype=np.float64)
    oof_m2 = np.full(len(data), np.nan, dtype=np.float64)
    fold_index = np.full(len(data), -1, dtype=np.int16)
    fold_audit: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=int(folds))
    fold_splits = list(splitter.split(data, y90, groups=groups))
    for fold, (train, validation) in enumerate(fold_splits):
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("GroupKFold frame leakage")
        fold_encoder = FeatureEncoder.fit(data.iloc[train])
        x_train = fold_encoder.transform(data.iloc[train])
        x_validation = fold_encoder.transform(data.iloc[validation])
        m0, m1, m2 = _models(seed + fold)
        weights = inverse_candidate_weights(data.iloc[train], y90[train])
        m0.fit(x_train, y90[train], sample_weight=weights)
        m1.fit(x_train, y90[train], sample_weight=weights)
        regression_weights = 1.0 / data.iloc[train].groupby("sample_id")["candidate_id"].transform("count").to_numpy(float)
        regression_weights *= len(regression_weights) / regression_weights.sum()
        m2.fit(x_train, y_iou[train], sample_weight=regression_weights)
        oof_m0[validation] = m0.predict_proba(x_validation)[:, 1]
        oof_m1[validation] = m1.predict_proba(x_validation)[:, 1]
        oof_m2[validation] = np.clip(m2.predict(x_validation), 0.0, 1.0)
        fold_index[validation] = fold
        fold_audit.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_groups": int(len(set(groups[train]))),
                "validation_groups": int(len(set(groups[validation]))),
                "group_overlap": 0,
            }
        )
    if not np.isfinite(oof_m0).all() or not np.isfinite(oof_m1).all() or not np.isfinite(oof_m2).all():
        raise RuntimeError("OOF predictions are incomplete")
    # Cross-fit calibration so every reported calibrated probability is produced
    # without fitting the calibrator on that row. Separate all-OOF calibrators are
    # retained only for deployment after model selection.
    calibrated_m0 = np.full(len(data), np.nan, dtype=np.float64)
    calibrated_m1 = np.full(len(data), np.nan, dtype=np.float64)
    for fold, (_, validation) in enumerate(fold_splits):
        calibration_train = fold_index != fold
        fold_calibrator_m0 = IsotonicRegression(out_of_bounds="clip").fit(
            oof_m0[calibration_train], y90[calibration_train]
        )
        fold_calibrator_m1 = IsotonicRegression(out_of_bounds="clip").fit(
            oof_m1[calibration_train], y90[calibration_train]
        )
        calibrated_m0[validation] = fold_calibrator_m0.predict(oof_m0[validation])
        calibrated_m1[validation] = fold_calibrator_m1.predict(oof_m1[validation])
    calibrator_m0 = IsotonicRegression(out_of_bounds="clip").fit(oof_m0, y90)
    calibrator_m1 = IsotonicRegression(out_of_bounds="clip").fit(oof_m1, y90)
    output_columns = [
        "sample_id",
        "candidate_id",
        "frame_id",
        "scene_id",
        "source_family",
        "candidate_iou",
    ]
    if "split" in data.columns:
        output_columns.insert(2, "split")
    output = data[output_columns].copy()
    output["y90"] = y90.astype(bool)
    output["fold"] = fold_index
    output["m0_p90_raw"] = oof_m0
    output["m0_p90_calibrated"] = calibrated_m0
    output["m1_p90_raw"] = oof_m1
    output["m1_p90_calibrated"] = calibrated_m1
    output["m2_predicted_iou"] = oof_m2

    encoder = FeatureEncoder.fit(data)
    x = encoder.transform(data)
    full_m0, full_m1, full_m2 = _models(seed)
    weights = inverse_candidate_weights(data, y90)
    full_m0.fit(x, y90, sample_weight=weights)
    full_m1.fit(x, y90, sample_weight=weights)
    regression_weights = 1.0 / data.groupby("sample_id")["candidate_id"].transform("count").to_numpy(float)
    regression_weights *= len(regression_weights) / regression_weights.sum()
    full_m2.fit(x, y_iou, sample_weight=regression_weights)
    artifacts = {
        "encoder": encoder,
        "m0_classifier": full_m0,
        "m1_classifier": full_m1,
        "m2_regressor": full_m2,
        "m0_calibrator": calibrator_m0,
        "m1_calibrator": calibrator_m1,
        "fold_audit": fold_audit,
    }
    return output, artifacts


def _choose(frame: pd.DataFrame, primary: str, secondary: str | None = None) -> pd.DataFrame:
    sort_columns = ["sample_id", primary]
    ascending = [True, False]
    if secondary:
        sort_columns.append(secondary)
        ascending.append(False)
    sort_columns.append("candidate_id")
    ascending.append(True)
    return (
        frame.sort_values(sort_columns, ascending=ascending, kind="stable")
        .drop_duplicates("sample_id", keep="first")
        .sort_values("sample_id")
    )


def selector_metrics(oof: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    methods = {
        "M0_logistic": _choose(oof, "m0_p90_calibrated"),
        "M1_hgb_classifier": _choose(oof, "m1_p90_calibrated"),
        "M2_hgb_regressor": _choose(oof, "m2_predicted_iou"),
        "M3_two_head_ensemble": _choose(oof, "m1_p90_calibrated", "m2_predicted_iou"),
    }
    metrics = {
        name: summarize_ious(selected["candidate_iou"].to_numpy())
        for name, selected in methods.items()
    }
    decisions = pd.concat(
        [selected.assign(method=name) for name, selected in methods.items()],
        ignore_index=True,
    )
    return metrics, decisions


def predict_candidates(frame: pd.DataFrame, model_payload: dict[str, Any]) -> pd.DataFrame:
    """Score an inference-only candidate table with a frozen selector payload."""

    data = frame[frame["eligible_final"]].copy().reset_index(drop=True)
    x = model_payload["encoder"].transform(data)
    raw = model_payload["classifier"].predict_proba(x)[:, 1]
    data["predicted_p90_raw"] = raw
    data["predicted_p90_calibrated"] = model_payload["calibrator"].predict(raw)
    data["predicted_iou"] = np.clip(model_payload["regressor"].predict(x), 0.0, 1.0)
    selected_name = str(model_payload["selected_name"])
    if selected_name.startswith("HIFI_FALLBACK"):
        data["selection_score"] = np.where(
            data["source_family"].isin({"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"}),
            1.0,
            -1.0,
        )
    elif selected_name == "M2_hgb_regressor":
        data["selection_score"] = data["predicted_iou"]
    else:
        data["selection_score"] = data["predicted_p90_calibrated"]
    return data


def select_scored_candidates(
    scored: pd.DataFrame, selected_name: str
) -> pd.DataFrame:
    """Select one inference candidate per sample with the OOF model's tie rule."""

    columns = ["sample_id", "selection_score"]
    ascending = [True, False]
    if selected_name == "M3_two_head_ensemble":
        columns.append("predicted_iou")
        ascending.append(False)
    columns.append("candidate_id")
    ascending.append(True)
    return (
        scored.sort_values(columns, ascending=ascending, kind="stable")
        .drop_duplicates("sample_id", keep="first")
        .sort_values("sample_id", kind="stable")
    )


__all__ = [
    "FeatureEncoder",
    "grouped_oof_predictions",
    "inverse_candidate_weights",
    "p90_labels",
    "predict_candidates",
    "select_scored_candidates",
    "selector_metrics",
]
