from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    l2: float
    constant_probability: float | None = None

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("feature dimension changed after calibration")
        if self.constant_probability is not None:
            return np.full(matrix.shape[0], self.constant_probability, dtype=np.float64)
        normalized = (matrix - np.asarray(self.mean)) / np.asarray(self.scale)
        return expit(normalized @ np.asarray(self.coefficients) + float(self.intercept))

    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fit_l2_logistic(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str],
    sample_weight: np.ndarray | None = None,
    l2: float = 1.0,
) -> LogisticModel:
    matrix = np.asarray(values, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("X/y shape mismatch")
    if matrix.shape[1] != len(feature_names) or not np.isfinite(matrix).all():
        raise ValueError("non-finite or mismatched calibration features")
    if not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("logistic labels must be binary")
    weights = np.ones(target.size, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != target.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample weights must be positive finite values")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    if np.unique(target).size == 1:
        probability = float((np.sum(weights * target) + 0.5) / (np.sum(weights) + 1.0))
        return LogisticModel(tuple(feature_names), tuple(mean), tuple(scale), tuple(np.zeros(matrix.shape[1])), 0.0, float(l2), probability)
    normalized = (matrix - mean) / scale

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        beta, intercept = parameters[:-1], parameters[-1]
        logits = normalized @ beta + intercept
        loss = np.sum(weights * (np.logaddexp(0.0, logits) - target * logits)) / np.sum(weights)
        loss += 0.5 * float(l2) * float(beta @ beta) / target.size
        residual = weights * (expit(logits) - target) / np.sum(weights)
        gradient = np.r_[normalized.T @ residual + float(l2) * beta / target.size, np.sum(residual)]
        return float(loss), gradient

    result = minimize(
        lambda parameter: objective(parameter), np.zeros(matrix.shape[1] + 1),
        method="L-BFGS-B", jac=True, options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"logistic calibration failed: {result.message}")
    return LogisticModel(
        tuple(feature_names), tuple(float(x) for x in mean), tuple(float(x) for x in scale),
        tuple(float(x) for x in result.x[:-1]), float(result.x[-1]), float(l2), None,
    )


def group_folds(groups: Sequence[str], *, n_splits: int = 5, seed: int = 20260803) -> np.ndarray:
    """Assign entire capture groups to folds with deterministic size balancing."""

    if n_splits < 2:
        raise ValueError("OOF requires at least two folds")
    groups_array = np.asarray(groups, dtype=object)
    unique, counts = np.unique(groups_array, return_counts=True)
    if unique.size < n_splits:
        raise ValueError("not enough independent groups for requested OOF folds")
    rng = np.random.default_rng(seed)
    order = np.arange(unique.size); rng.shuffle(order)
    order = sorted(order, key=lambda index: -int(counts[index]))
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    mapping: dict[str, int] = {}
    for index in order:
        fold = int(np.argmin(fold_sizes))
        mapping[str(unique[index])] = fold
        fold_sizes[fold] += int(counts[index])
    assignments = np.asarray([mapping[str(group)] for group in groups_array], dtype=np.int64)
    for group in unique:
        if np.unique(assignments[groups_array == group]).size != 1:
            raise AssertionError("group leaked across OOF folds")
    return assignments


@dataclass(frozen=True)
class DualRiskCalibration:
    benefit_model: LogisticModel
    harm_model: LogisticModel
    benefit_platt: LogisticModel
    harm_platt: LogisticModel
    n_splits: int
    seed: int
    calibration_partition: str = "calibration"

    def predict(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        benefit_raw = self.benefit_model.predict_proba(values)
        harm_raw = self.harm_model.predict_proba(values)
        benefit_logit = np.log(np.clip(benefit_raw, 1e-9, 1 - 1e-9) / np.clip(1 - benefit_raw, 1e-9, 1))[:, None]
        harm_logit = np.log(np.clip(harm_raw, 1e-9, 1 - 1e-9) / np.clip(1 - harm_raw, 1e-9, 1))[:, None]
        return self.benefit_platt.predict_proba(benefit_logit), self.harm_platt.predict_proba(harm_logit)


def cross_fit_dual_risk(
    values: np.ndarray,
    beneficial: np.ndarray,
    harmful: np.ndarray,
    groups: Sequence[str],
    *,
    feature_names: Sequence[str],
    sample_weight: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 20260803,
    l2: float = 1.0,
    secondary_values: np.ndarray | None = None,
) -> tuple:
    matrix = np.asarray(values, dtype=np.float64)
    y_benefit = np.asarray(beneficial, dtype=np.float64)
    y_harm = np.asarray(harmful, dtype=np.float64)
    weights = np.ones(matrix.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    groups_array = np.asarray(groups, dtype=object)
    secondary = None if secondary_values is None else np.asarray(secondary_values, dtype=np.float64)
    if secondary is not None and secondary.shape != matrix.shape:
        raise ValueError("secondary calibration matrix shape mismatch")
    folds = group_folds(groups_array, n_splits=n_splits, seed=seed)
    raw_benefit = np.zeros(matrix.shape[0], dtype=np.float64)
    raw_harm = np.zeros(matrix.shape[0], dtype=np.float64)
    calibrated_b = np.zeros(matrix.shape[0], dtype=np.float64)
    calibrated_h = np.zeros(matrix.shape[0], dtype=np.float64)
    secondary_calibrated_b = np.zeros(matrix.shape[0], dtype=np.float64)
    secondary_calibrated_h = np.zeros(matrix.shape[0], dtype=np.float64)
    for fold in range(n_splits):
        train = folds != fold; held_out = folds == fold
        benefit_model = fit_l2_logistic(matrix[train], y_benefit[train], feature_names=feature_names, sample_weight=weights[train], l2=l2)
        harm_model = fit_l2_logistic(matrix[train], y_harm[train], feature_names=feature_names, sample_weight=weights[train], l2=l2)
        raw_benefit[held_out] = benefit_model.predict_proba(matrix[held_out])
        raw_harm[held_out] = harm_model.predict_proba(matrix[held_out])
        # Platt calibration is itself nested group-OOF.  Fitting it once on all
        # outer OOF predictions would expose held-out labels to their own
        # threshold-sweep probabilities.
        train_indices = np.flatnonzero(train)
        inner_folds = group_folds(
            groups_array[train], n_splits=n_splits, seed=seed + fold + 1
        )
        inner_b = np.zeros(train_indices.size, dtype=np.float64)
        inner_h = np.zeros(train_indices.size, dtype=np.float64)
        for inner_fold in range(n_splits):
            inner_train_local = inner_folds != inner_fold
            inner_held_local = inner_folds == inner_fold
            inner_train = train_indices[inner_train_local]
            inner_held = train_indices[inner_held_local]
            inner_b_model = fit_l2_logistic(
                matrix[inner_train], y_benefit[inner_train],
                feature_names=feature_names, sample_weight=weights[inner_train], l2=l2,
            )
            inner_h_model = fit_l2_logistic(
                matrix[inner_train], y_harm[inner_train],
                feature_names=feature_names, sample_weight=weights[inner_train], l2=l2,
            )
            inner_b[inner_held_local] = inner_b_model.predict_proba(matrix[inner_held])
            inner_h[inner_held_local] = inner_h_model.predict_proba(matrix[inner_held])
        inner_b_logit = np.log(
            np.clip(inner_b, 1e-9, 1 - 1e-9) / np.clip(1 - inner_b, 1e-9, 1)
        )[:, None]
        inner_h_logit = np.log(
            np.clip(inner_h, 1e-9, 1 - 1e-9) / np.clip(1 - inner_h, 1e-9, 1)
        )[:, None]
        outer_b_platt = fit_l2_logistic(
            inner_b_logit, y_benefit[train], feature_names=("benefit_raw_logit",),
            sample_weight=weights[train], l2=1e-3,
        )
        outer_h_platt = fit_l2_logistic(
            inner_h_logit, y_harm[train], feature_names=("harm_raw_logit",),
            sample_weight=weights[train], l2=1e-3,
        )
        held_b_logit = np.log(
            np.clip(raw_benefit[held_out], 1e-9, 1 - 1e-9)
            / np.clip(1 - raw_benefit[held_out], 1e-9, 1)
        )[:, None]
        held_h_logit = np.log(
            np.clip(raw_harm[held_out], 1e-9, 1 - 1e-9)
            / np.clip(1 - raw_harm[held_out], 1e-9, 1)
        )[:, None]
        calibrated_b[held_out] = outer_b_platt.predict_proba(held_b_logit)
        calibrated_h[held_out] = outer_h_platt.predict_proba(held_h_logit)
        if secondary is not None:
            secondary_raw_b = benefit_model.predict_proba(secondary[held_out])
            secondary_raw_h = harm_model.predict_proba(secondary[held_out])
            secondary_b_logit = np.log(
                np.clip(secondary_raw_b, 1e-9, 1 - 1e-9)
                / np.clip(1 - secondary_raw_b, 1e-9, 1)
            )[:, None]
            secondary_h_logit = np.log(
                np.clip(secondary_raw_h, 1e-9, 1 - 1e-9)
                / np.clip(1 - secondary_raw_h, 1e-9, 1)
            )[:, None]
            secondary_calibrated_b[held_out] = outer_b_platt.predict_proba(
                secondary_b_logit
            )
            secondary_calibrated_h[held_out] = outer_h_platt.predict_proba(
                secondary_h_logit
            )
    logits_b = np.log(np.clip(raw_benefit, 1e-9, 1 - 1e-9) / np.clip(1 - raw_benefit, 1e-9, 1))[:, None]
    logits_h = np.log(np.clip(raw_harm, 1e-9, 1 - 1e-9) / np.clip(1 - raw_harm, 1e-9, 1))[:, None]
    benefit_platt = fit_l2_logistic(logits_b, y_benefit, feature_names=("benefit_raw_logit",), sample_weight=weights, l2=1e-3)
    harm_platt = fit_l2_logistic(logits_h, y_harm, feature_names=("harm_raw_logit",), sample_weight=weights, l2=1e-3)
    final_b = fit_l2_logistic(matrix, y_benefit, feature_names=feature_names, sample_weight=weights, l2=l2)
    final_h = fit_l2_logistic(matrix, y_harm, feature_names=feature_names, sample_weight=weights, l2=l2)
    bundle = DualRiskCalibration(final_b, final_h, benefit_platt, harm_platt, n_splits, seed)
    if secondary is not None:
        return (
            bundle,
            calibrated_b,
            calibrated_h,
            folds,
            secondary_calibrated_b,
            secondary_calibrated_h,
        )
    return bundle, calibrated_b, calibrated_h, folds


def threshold_sweep(
    rows: Sequence[Mapping[str, Any]],
    *,
    tau_grid: Iterable[float],
    eta_grid: Iterable[float],
    query_grid: Iterable[float] = (0.0,),
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Calibration-only corrected sweep with conservative pre-registered utility."""

    results: list[dict[str, Any]] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row.get("sample_id", index)), []).append(row)
    representative = [values[0] for values in grouped.values()]
    for candidates in grouped.values():
        weights = {float(row.get("sample_weight", 1.0)) for row in candidates}
        if len(weights) != 1 or next(iter(weights)) <= 0.0:
            raise ValueError("all pairs for a sample must share one positive sample_weight")
    baseline_correct_count = sum(
        float(row.get("sample_weight", 1.0))
        for row in representative
        if bool(row["baseline_correct"])
    )
    denominator_weight = sum(
        float(row.get("sample_weight", 1.0)) for row in representative
    )
    for query_threshold in query_grid:
        for tau in tau_grid:
            for eta in eta_grid:
                switched_rows = []
                for candidates in grouped.values():
                    eligible_candidates = [
                        row for row in candidates
                        if float(row.get("query_score", 1.0)) >= float(query_threshold)
                        and float(row["p_benefit"]) >= float(tau)
                        and float(row["p_harm"]) <= float(eta)
                        and float(row.get("confirmation_p_benefit", row["p_benefit"])) >= float(tau)
                        and float(row.get("confirmation_p_harm", row["p_harm"])) <= float(eta)
                        and bool(row.get("evidence_reliable", False))
                        and bool(row.get("challenger_hard_valid", False))
                        and bool(row.get("confirmation_stable", False))
                        and not bool(row.get("hard_veto", False))
                    ]
                    if eligible_candidates:
                        switched_rows.append(sorted(
                            eligible_candidates,
                            key=lambda row: (-float(row["p_benefit"]), float(row["p_harm"]), str(row.get("challenger_id", ""))),
                        )[0])
                recovered = sum(
                    float(row.get("sample_weight", 1.0))
                    for row in switched_rows
                    if not bool(row["baseline_correct"]) and bool(row["challenger_correct"])
                )
                harmful = sum(
                    float(row.get("sample_weight", 1.0))
                    for row in switched_rows
                    if bool(row["baseline_correct"]) and not bool(row["challenger_correct"])
                )
                weighted_switches = sum(
                    float(row.get("sample_weight", 1.0)) for row in switched_rows
                )
                baseline_successes = sum(
                    float(row.get("sample_weight", 1.0))
                    for row in representative
                    if bool(row["baseline_correct"])
                )
                final_successes = baseline_successes + recovered - harmful
                precision = recovered / (recovered + harmful) if recovered + harmful else 0.0
                harm_rate = harmful / baseline_correct_count if baseline_correct_count else 0.0
                result = {
                    "query_threshold": float(query_threshold), "tau": float(tau), "eta": float(eta),
                    "recovered": recovered, "harmful": harmful, "net": recovered - harmful,
                    "utility": recovered - 2 * harmful,
                    "switches": weighted_switches,
                    "raw_switches": len(switched_rows),
                    "switch_rate": weighted_switches / denominator_weight if denominator_weight else 0.0,
                    "outcome_changing_precision": precision, "harm_rate": harm_rate,
                    "final_successes": final_successes,
                    "final_j1": final_successes / denominator_weight if denominator_weight else 0.0,
                    "eligible": bool(recovered > harmful and recovered - harmful > 0 and precision >= 0.67 and harm_rate <= 0.01),
                }
                results.append(result)
    eligible = [row for row in results if row["eligible"]]
    selected = sorted(
        eligible,
        key=lambda row: (-row["utility"], row["harmful"], row["switches"], -row["tau"], row["eta"]),
    )[0] if eligible else None
    return results, selected


def fit_query_gate(
    values: np.ndarray,
    recoverable: np.ndarray,
    groups: Sequence[str],
    *,
    feature_names: Sequence[str],
    n_splits: int = 5,
    seed: int = 20260803,
) -> tuple[LogisticModel, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    labels = np.asarray(recoverable, dtype=np.float64)
    folds = group_folds(groups, n_splits=n_splits, seed=seed)
    oof = np.zeros(labels.size, dtype=np.float64)
    for fold in range(n_splits):
        train = folds != fold; held_out = folds == fold
        model = fit_l2_logistic(matrix[train], labels[train], feature_names=feature_names, l2=1.0)
        oof[held_out] = model.predict_proba(matrix[held_out])
    final = fit_l2_logistic(matrix, labels, feature_names=feature_names, l2=1.0)
    return final, oof, folds


def choose_query_threshold(
    scores: np.ndarray,
    recoverable: np.ndarray,
    *,
    minimum_recall: float = 0.90,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(recoverable, dtype=bool)
    candidates = sorted(set(np.r_[0.0, scores, 1.0]))
    feasible = []
    positives = int(labels.sum())
    for threshold in candidates:
        called = scores >= threshold
        recall = float(np.sum(called & labels) / positives) if positives else 1.0
        if recall >= minimum_recall:
            feasible.append({"threshold": float(threshold), "recoverable_recall": recall, "call_rate": float(called.mean())})
    if not feasible:
        return {"threshold": 0.0, "recoverable_recall": 1.0, "call_rate": 1.0}
    return sorted(feasible, key=lambda row: (row["call_rate"], -row["recoverable_recall"], -row["threshold"]))[0]
