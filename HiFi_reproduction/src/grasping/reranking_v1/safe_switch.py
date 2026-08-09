"""OOF-trained safe switch with a fail-closed GQ-CNN fallback."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


GATE_FEATURE_ALLOWLIST = (
    "old_new_q_difference",
    "old_new_reranker_score_difference",
    "old_new_mask_support_difference",
    "old_new_geometry_risk_difference",
    "ranker_confidence",
    "score_entropy",
    "candidate_set_q_mean",
    "candidate_set_q_std",
    "candidate_count",
)
FORBIDDEN_GATE_TOKENS = (
    "candidate_positive",
    "beneficial",
    "harmful",
    "neutral",
    "gt_",
    "with_gt",
    "ground_truth",
)


def validate_gate_features(columns: Sequence[str]) -> tuple[str, ...]:
    names = tuple(map(str, columns))
    rejected = [
        name
        for name in names
        if name not in GATE_FEATURE_ALLOWLIST
        or any(token in name.lower() for token in FORBIDDEN_GATE_TOKENS)
    ]
    if rejected:
        raise ValueError(f"unsafe or unknown safe-gate features: {sorted(rejected)}")
    return names


def _entropy(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))


def build_switch_features(
    scored: pd.DataFrame,
    *,
    old_rank_column: str = "original_gqcnn_rank",
    new_score_column: str = "reranker_score",
    mask_column: str = "p_axis_mean",
    geometry_risk_column: str = "geometry_risk",
) -> pd.DataFrame:
    """Build label-free deployable gate features, one row per sample."""

    required = {
        "sample_id",
        "scene_id",
        "candidate_id",
        "q_raw",
        old_rank_column,
        new_score_column,
        mask_column,
        geometry_risk_column,
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"safe-switch input missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for (sample_id, scene_id), group in scored.groupby(
        ["sample_id", "scene_id"], sort=False
    ):
        old = group.sort_values(
            [old_rank_column, "candidate_id"], ascending=[True, True]
        ).iloc[0]
        new_order = group.sort_values(
            [new_score_column, "candidate_id"], ascending=[False, True]
        )
        new = new_order.iloc[0]
        sorted_scores = new_order[new_score_column].to_numpy(dtype=float)
        confidence = float(
            sorted_scores[0] - sorted_scores[1]
            if len(sorted_scores) > 1
            else abs(sorted_scores[0])
        )
        q_values = group["q_raw"].to_numpy(dtype=float)
        rows.append(
            {
                "sample_id": str(sample_id),
                "scene_id": str(scene_id),
                **(
                    {"oof_fold": int(group["oof_fold"].iloc[0])}
                    if "oof_fold" in group.columns
                    else {}
                ),
                "old_candidate_id": str(old["candidate_id"]),
                "new_candidate_id": str(new["candidate_id"]),
                "new_geometry_safe": bool(
                    float(new[geometry_risk_column]) <= 0.0
                ),
                "old_new_q_difference": float(new["q_raw"] - old["q_raw"]),
                "old_new_reranker_score_difference": float(
                    new[new_score_column] - old[new_score_column]
                ),
                "old_new_mask_support_difference": float(
                    new[mask_column] - old[mask_column]
                ),
                "old_new_geometry_risk_difference": float(
                    old[geometry_risk_column] - new[geometry_risk_column]
                ),
                "ranker_confidence": confidence,
                "score_entropy": _entropy(sorted_scores),
                "candidate_set_q_mean": float(np.mean(q_values)),
                "candidate_set_q_std": float(np.std(q_values)),
                "candidate_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_switch_examples(
    scored: pd.DataFrame,
    *,
    old_rank_column: str = "original_gqcnn_rank",
    new_score_column: str = "reranker_score",
    mask_column: str = "p_axis_mean",
    geometry_risk_column: str = "geometry_risk",
    label_column: str = "candidate_positive",
) -> pd.DataFrame:
    """Join GT outcomes after label-free safe-gate features are frozen."""

    if label_column not in scored:
        raise ValueError(f"safe-switch training label missing: {label_column}")
    output = build_switch_features(
        scored,
        old_rank_column=old_rank_column,
        new_score_column=new_score_column,
        mask_column=mask_column,
        geometry_risk_column=geometry_risk_column,
    )
    label_index = scored.set_index(["sample_id", "candidate_id"])[label_column]
    outcomes = []
    for row in output.itertuples(index=False):
        old_correct = bool(label_index.loc[(row.sample_id, row.old_candidate_id)])
        new_correct = bool(label_index.loc[(row.sample_id, row.new_candidate_id)])
        if not old_correct and new_correct:
            outcome = "beneficial"
        elif old_correct and not new_correct:
            outcome = "harmful"
        else:
            outcome = "neutral"
        outcomes.append((outcome, old_correct, new_correct))
    output["switch_outcome"] = [item[0] for item in outcomes]
    output["old_correct"] = [item[1] for item in outcomes]
    output["new_correct"] = [item[2] for item in outcomes]
    return output


@dataclass
class SafeSwitchGate:
    """Binary estimate that a proposed switch is beneficial.

    Neutral and harmful switches are the negative class.  The final harmful-rate
    guarantee is imposed independently by threshold selection on validation.
    """

    feature_columns: tuple[str, ...] = GATE_FEATURE_ALLOWLIST
    c: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        self.feature_columns = validate_gate_features(self.feature_columns)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.model_: LogisticRegression | None = None
        self.constant_confidence_: float | None = None

    def fit(self, oof_examples: pd.DataFrame) -> "SafeSwitchGate":
        required = {"switch_outcome", "scene_id", *self.feature_columns}
        missing = sorted(required - set(oof_examples.columns))
        if missing:
            raise ValueError(f"safe-gate OOF examples missing columns: {missing}")
        if "oof_fold" not in oof_examples.columns:
            raise ValueError("safe gate may only fit explicitly marked OOF examples")
        folds_per_scene = oof_examples.groupby("scene_id")["oof_fold"].nunique()
        if bool((folds_per_scene > 1).any()):
            raise ValueError("safe-gate OOF folds are not grouped by scene")
        x = oof_examples.loc[:, self.feature_columns].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("safe-gate features must be finite")
        y = (oof_examples["switch_outcome"] == "beneficial").to_numpy(dtype=int)
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        if len(np.unique(y)) < 2:
            # No classifier can be identified.  Preserve a deterministic,
            # auditable constant and let the validation harm constraint choose
            # the conservative fallback threshold.
            self.constant_confidence_ = float(np.mean(y))
            self.model_ = None
            return self
        self.model_ = LogisticRegression(
            C=float(self.c),
            solver="liblinear",
            class_weight="balanced",
            random_state=int(self.seed),
            max_iter=1000,
        )
        self.model_.fit((x - self.mean_) / self.scale_, y)
        return self

    def predict_confidence(self, examples: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.mean_ is None or self.scale_ is None:
            if (
                self.constant_confidence_ is not None
                and self.mean_ is not None
                and self.scale_ is not None
            ):
                return np.full(
                    len(examples), self.constant_confidence_, dtype=np.float64
                )
            raise RuntimeError("safe gate has not been fitted")
        x = examples.loc[:, self.feature_columns].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("safe-gate features must be finite")
        return self.model_.predict_proba((x - self.mean_) / self.scale_)[:, 1]

    def artifact(self) -> dict[str, Any]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("safe gate has not been fitted")
        coefficients = (
            self.model_.coef_[0]
            if self.model_ is not None
            else np.zeros(len(self.feature_columns), dtype=np.float64)
        )
        return {
            "method": "oof_safe_switch_gate",
            "feature_columns": list(self.feature_columns),
            "fit_scope": "scene-grouped OOF reranker predictions",
            "C": float(self.c),
            "seed": int(self.seed),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "gate_kind": (
                "logistic"
                if self.model_ is not None
                else "constant_no_identifiable_classifier"
            ),
            "constant_confidence": self.constant_confidence_,
            "intercept": (
                float(self.model_.intercept_[0])
                if self.model_ is not None
                else None
            ),
            "coefficients": {
                feature: float(value)
                for feature, value in zip(self.feature_columns, coefficients)
            },
        }

    @classmethod
    def from_artifact(cls, payload: Mapping[str, Any]) -> "SafeSwitchGate":
        gate = cls(
            feature_columns=tuple(map(str, payload["feature_columns"])),
            c=float(payload["C"]),
            seed=int(payload["seed"]),
        )
        gate.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        gate.scale_ = np.asarray(payload["scale"], dtype=np.float64)
        if (
            gate.mean_.shape != (len(gate.feature_columns),)
            or gate.scale_.shape != gate.mean_.shape
            or not np.all(np.isfinite(gate.mean_))
            or not np.all(np.isfinite(gate.scale_))
            or np.any(gate.scale_ <= 0.0)
        ):
            raise ValueError("invalid serialized safe-gate scaler")
        gate.constant_confidence_ = (
            None
            if payload.get("constant_confidence") is None
            else float(payload["constant_confidence"])
        )
        if payload["gate_kind"] == "logistic":
            gate.model_ = LogisticRegression(
                C=gate.c,
                solver="liblinear",
                class_weight="balanced",
                random_state=gate.seed,
                max_iter=1000,
            )
            gate.model_.coef_ = np.asarray(
                [[float(payload["coefficients"][name]) for name in gate.feature_columns]],
                dtype=np.float64,
            )
            gate.model_.intercept_ = np.asarray(
                [float(payload["intercept"])], dtype=np.float64
            )
            gate.model_.classes_ = np.asarray([0, 1], dtype=np.int64)
            gate.model_.n_features_in_ = len(gate.feature_columns)
        elif payload["gate_kind"] != "constant_no_identifiable_classifier":
            raise ValueError("unknown serialized safe-gate kind")
        return gate

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.artifact(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SafeSwitchGate":
        return cls.from_artifact(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )


def apply_safe_switch(
    examples: pd.DataFrame,
    confidence: Sequence[float],
    *,
    threshold: float,
    force_no_switch: bool = False,
) -> pd.DataFrame:
    """Apply confidence and geometry gates, otherwise retain original Top-1."""

    values = np.asarray(confidence, dtype=np.float64)
    if values.shape != (len(examples),):
        raise ValueError("one gate confidence is required per sample")
    result = examples.copy()
    old_id = result["old_candidate_id"].fillna("").astype(str)
    new_id = result["new_candidate_id"].fillna("").astype(str)
    valid_id = (
        old_id.str.len().gt(0)
        & new_id.str.len().gt(0)
        & ~old_id.str.lower().isin({"nan", "none"})
        & ~new_id.str.lower().isin({"nan", "none"})
    ).to_numpy()
    valid = np.isfinite(values)
    different = (old_id != new_id).to_numpy()
    geometry_safe = result["new_geometry_safe"].fillna(False).astype(bool).to_numpy()
    switch = (
        valid
        & valid_id
        & different
        & geometry_safe
        & (values >= float(threshold))
    )
    if force_no_switch:
        switch[:] = False
    result["gate_confidence"] = np.where(valid, values, 0.0)
    result["switch_applied"] = switch
    result["selected_candidate_id"] = np.where(
        switch,
        result["new_candidate_id"].astype(str),
        result["old_candidate_id"].astype(str),
    )
    result["fallback_reason"] = np.select(
        [
            ~valid,
            np.full(len(result), bool(force_no_switch)),
            ~valid_id,
            ~different,
            ~geometry_safe,
            values < float(threshold),
        ],
        [
            "invalid_gate_confidence",
            "forced_no_switch",
            "invalid_candidate_id",
            "same_candidate",
            "new_geometry_unsafe",
            "below_threshold",
        ],
        default="switch",
    )
    return result


def threshold_sweep(
    validation_examples: pd.DataFrame,
    confidence: Sequence[float],
    *,
    thresholds: Sequence[float] | None = None,
    harmful_rate_limit: float = 0.01,
    all_sample_count: int | None = None,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Choose the maximum-net-gain threshold under a validation harm cap."""

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)
    rows: list[dict[str, Any]] = []
    if all_sample_count is None:
        all_sample_count = len(validation_examples)
    if all_sample_count < len(validation_examples):
        raise ValueError(
            "all_sample_count cannot be smaller than nonempty validation examples"
        )
    total = max(int(all_sample_count), 1)
    nonempty_total = max(len(validation_examples), 1)
    for threshold in thresholds:
        decisions = apply_safe_switch(
            validation_examples, confidence, threshold=float(threshold)
        )
        switched = decisions["switch_applied"].to_numpy(dtype=bool)
        outcomes = validation_examples["switch_outcome"].astype(str).to_numpy()
        recovered = int(np.sum(switched & (outcomes == "beneficial")))
        harmful = int(np.sum(switched & (outcomes == "harmful")))
        coverage = float(np.mean(switched)) if len(switched) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "recovered": recovered,
                "harmful": harmful,
                "net_gain": recovered - harmful,
                "harmful_rate": harmful / total,
                "harmful_rate_all_samples": harmful / total,
                "harmful_rate_nonempty_samples": harmful / nonempty_total,
                "decision_precision": (
                    recovered / int(np.sum(switched))
                    if int(np.sum(switched))
                    else 0.0
                ),
                "coverage": coverage,
                "coverage_all_samples": int(np.sum(switched)) / total,
                "coverage_nonempty_samples": coverage,
                "eligible_under_harm_limit": harmful / total
                <= float(harmful_rate_limit),
                "force_no_switch": False,
            }
        )
    # This explicit sentinel is always eligible and guarantees that a locked
    # safe-switch policy can never violate its predeclared harmed-rate cap.
    rows.append(
        {
            "threshold": 1.0,
            "recovered": 0,
            "harmful": 0,
            "net_gain": 0,
            "harmful_rate": 0.0,
            "harmful_rate_all_samples": 0.0,
            "harmful_rate_nonempty_samples": 0.0,
            "decision_precision": 0.0,
            "coverage": 0.0,
            "coverage_all_samples": 0.0,
            "coverage_nonempty_samples": 0.0,
            "eligible_under_harm_limit": True,
            "force_no_switch": True,
        }
    )
    sweep = pd.DataFrame(rows)
    eligible = sweep.loc[sweep["eligible_under_harm_limit"]]
    selected = eligible.sort_values(
        [
            "net_gain",
            "decision_precision",
            "coverage",
            "force_no_switch",
            "threshold",
        ],
        ascending=[False, False, True, False, False],
    ).iloc[0]
    reason = (
        "fail_closed_no_switch"
        if bool(selected["force_no_switch"])
        else "max_net_gain_under_harm_limit"
    )
    return sweep, {
        **selected.to_dict(),
        "selection_reason": reason,
        "harmful_rate_limit": float(harmful_rate_limit),
        "harmful_rate_denominator": "all_validation_samples",
        "all_sample_count": int(all_sample_count),
        "nonempty_sample_count": int(len(validation_examples)),
        "selection_split": "validation",
    }


def candidate_predictions_from_switch(
    scored: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    method: str = "residual_mlp_safe_switch",
) -> pd.DataFrame:
    """Convert one selected ID per sample into a full identity-preserving rank."""

    required = {
        "sample_id",
        "candidate_id",
        "candidate_identity_sha256",
        "reranker_score",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"safe-switch candidate frame missing columns: {missing}")
    selected = decisions.set_index("sample_id")["selected_candidate_id"].astype(str)
    if selected.index.duplicated().any():
        raise ValueError("safe-switch decisions contain duplicate samples")
    if set(selected.index.astype(str)) != set(scored["sample_id"].astype(str)):
        raise ValueError("safe-switch decision sample universe mismatch")
    output = scored.copy()
    output["reranker_method"] = method
    output["reranker_score"] = output["reranker_score"].to_numpy(
        dtype=np.float64
    )
    for sample_id, index in output.groupby("sample_id", sort=False).groups.items():
        chosen = str(selected.loc[str(sample_id)])
        local_ids = output.loc[index, "candidate_id"].astype(str)
        if chosen not in set(local_ids):
            raise ValueError(f"safe-switch selected unknown candidate: {sample_id}/{chosen}")
        chosen_index = local_ids.index[local_ids == chosen][0]
        output.loc[chosen_index, "reranker_score"] = float(
            output.loc[index, "reranker_score"].max() + 1.0
        )
    ordered = output.sort_values(
        ["sample_id", "reranker_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ordered["reranker_rank"] = (
        ordered.groupby("sample_id", sort=False).cumcount() + 1
    )
    output = output.drop(columns=["reranker_rank"], errors="ignore").join(
        ordered["reranker_rank"]
    )
    return output
