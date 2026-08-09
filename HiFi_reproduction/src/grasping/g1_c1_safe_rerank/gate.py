"""Protected Top-1 query and switch gates learned from scene-held-out OOF data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .contracts import selected_ids_are_frozen


PAIR_FEATURES = (
    "baseline_probability",
    "challenger_probability",
    "calibrated_probability_delta",
    "reranker_score_delta",
    "original_score_delta",
    "mask_support_delta",
    "width_compatibility_delta",
    "contact_depth_difference_delta",
    "clearance_delta",
    "collision_risk_delta",
    "feature_reliability_minimum",
    "score_entropy",
    "candidate_count_normalized",
    "challenger_original_rank",
    "candidate_geometry_difference",
    "cross_backend_agreement_delta",
    "seed_agreement_fraction",
)

OUTCOME_CLASSES = (
    "Recovered",
    "Harmful",
    "Correct-to-Correct",
    "Wrong-to-Wrong",
)


def _top_pair(group: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rank_column = "pool_rank" if "pool_rank" in group.columns else "original_rank"
    baseline = group.sort_values(
        [rank_column, "stable_candidate_id"], kind="mergesort"
    ).iloc[0]
    ordered = group.sort_values(
        ["reranker_score", "stable_candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    reranker_top = ordered.iloc[0]
    # The challenger is the ungated ranker's actual Top-1.  If it already
    # agrees with baseline, there is no switch proposal; never manufacture a
    # second-place challenger merely to make a gate decision possible.
    challenger = (
        baseline
        if str(reranker_top["stable_candidate_id"])
        == str(baseline["stable_candidate_id"])
        else reranker_top
    )
    return baseline, challenger


def build_pair_features(scored: pd.DataFrame, *, include_labels: bool) -> pd.DataFrame:
    required = {
        "sample_id",
        "scene_id",
        "stable_candidate_id",
        "original_rank",
        "original_score",
        "pool_score",
        "source_score_calibrated",
        "reranker_score",
        "grasp_axis_mask_support",
        "width_compatibility",
        "absolute_contact_depth_difference_m",
        "sweep_minimum_clearance_proxy",
        "approach_collision_proxy",
        "feature_reliability_score",
        "score_entropy",
        "candidate_count_normalized",
    }
    if include_labels:
        required.add("candidate_correct")
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"pair feature input missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for (sample_id, scene_id), group in scored.groupby(["sample_id", "scene_id"], sort=False):
        baseline, challenger = _top_pair(group)
        same = str(baseline["stable_candidate_id"]) == str(challenger["stable_candidate_id"])
        def finite_delta(name: str) -> float:
            left, right = float(baseline[name]), float(challenger[name])
            return float(right - left) if np.isfinite([left, right]).all() else 0.0

        ordered_scores = np.sort(group["reranker_score"].to_numpy(dtype=float))[::-1]
        ranker_margin = float(ordered_scores[0] - ordered_scores[1]) if len(ordered_scores) > 1 else 0.0
        dx = float(challenger.get("center_x", 0.0)) - float(baseline.get("center_x", 0.0))
        dy = float(challenger.get("center_y", 0.0)) - float(baseline.get("center_y", 0.0))
        da = abs(float(challenger.get("angle_deg", 0.0)) - float(baseline.get("angle_deg", 0.0)))
        da = min(da % 180.0, 180.0 - (da % 180.0)) / 90.0
        dw = abs(float(challenger.get("width_px", 0.0)) - float(baseline.get("width_px", 0.0))) / 640.0
        ranker_probability_column = (
            "ranker_probability" if "ranker_probability" in group.columns else "source_score_calibrated"
        )
        row: dict[str, Any] = {
            "sample_id": str(sample_id),
            "scene_id": str(scene_id),
            "baseline_candidate_id": str(baseline["stable_candidate_id"]),
            "challenger_candidate_id": str(challenger["stable_candidate_id"]),
            "challenger_available": not same,
            "baseline_probability": float(baseline[ranker_probability_column]),
            "challenger_probability": float(challenger[ranker_probability_column]),
            "calibrated_probability_delta": finite_delta(ranker_probability_column),
            "reranker_score_delta": ranker_margin,
            "original_score_delta": finite_delta("pool_score"),
            "mask_support_delta": finite_delta("grasp_axis_mask_support"),
            "width_compatibility_delta": finite_delta("width_compatibility"),
            "contact_depth_difference_delta": -finite_delta("absolute_contact_depth_difference_m"),
            "clearance_delta": finite_delta("sweep_minimum_clearance_proxy"),
            "collision_risk_delta": -finite_delta("approach_collision_proxy"),
            "feature_reliability_minimum": float(
                min(baseline["feature_reliability_score"], challenger["feature_reliability_score"])
            ),
            "score_entropy": float(baseline["score_entropy"]),
            "candidate_count_normalized": float(baseline["candidate_count_normalized"]),
            "challenger_original_rank": float(challenger.get("original_rank", 1.0)),
            "candidate_geometry_difference": float(np.hypot(dx, dy) / 800.0 + da + dw),
            "cross_backend_agreement_delta": (
                finite_delta("cross_backend_agreement_score")
                if "cross_backend_agreement_score" in group.columns
                else 0.0
            ),
            "seed_agreement_fraction": float(challenger.get("seed_agreement_fraction", 1.0)),
        }
        if "oof_fold" in group:
            row["oof_fold"] = int(group["oof_fold"].iloc[0])
        if include_labels:
            old_correct = bool(baseline["candidate_correct"])
            new_correct = bool(challenger["candidate_correct"])
            row.update(
                {
                    "baseline_correct": old_correct,
                    "challenger_correct": new_correct,
                    "beneficial": bool((not old_correct) and new_correct and not same),
                    "harmful": bool(old_correct and (not new_correct) and not same),
                    "outcome": (
                        "Recovered"
                        if (not old_correct) and new_correct and not same
                        else "Harmful"
                        if old_correct and (not new_correct) and not same
                        else "Correct-to-Correct"
                        if old_correct and new_correct
                        else "Wrong-to-Wrong"
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class _BinaryHead:
    model: LogisticRegression | None
    constant: float | None

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, *, seed: int) -> "_BinaryHead":
        if len(np.unique(y)) < 2:
            return cls(model=None, constant=float(y.mean()))
        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=int(seed),
        )
        model.fit(x, y)
        return cls(model=model, constant=None)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(x), float(self.constant), dtype=float)
        return self.model.predict_proba(x)[:, 1]

    def artifact(self) -> dict[str, Any]:
        if self.model is None:
            return {"kind": "constant", "constant": float(self.constant)}
        return {
            "kind": "logistic",
            "coefficient": self.model.coef_[0].tolist(),
            "intercept": float(self.model.intercept_[0]),
        }


@dataclass
class ConservativePairGate:
    feature_columns: tuple[str, ...] = PAIR_FEATURES
    seed: int = 42
    median: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    benefit_head: _BinaryHead | None = None
    harm_head: _BinaryHead | None = None

    def _matrix(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        matrix = frame.loc[:, self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        if fit:
            safe_matrix = matrix.copy()
            safe_matrix[:, np.isnan(safe_matrix).all(axis=0)] = 0.0
            self.median = np.nanmedian(safe_matrix, axis=0)
            self.median = np.where(np.isfinite(self.median), self.median, 0.0)
            filled = np.where(np.isfinite(matrix), matrix, self.median)
            self.mean = filled.mean(axis=0)
            std = filled.std(axis=0)
            self.scale = np.where(std > 1e-12, std, 1.0)
        if self.median is None or self.mean is None or self.scale is None:
            raise RuntimeError("gate preprocessor is not fitted")
        filled = np.where(np.isfinite(matrix), matrix, self.median)
        return (filled - self.mean) / self.scale

    def fit(self, oof_pairs: pd.DataFrame) -> "ConservativePairGate":
        required = {"scene_id", "oof_fold", "beneficial", "harmful", *self.feature_columns}
        missing = sorted(required - set(oof_pairs.columns))
        if missing:
            raise ValueError(f"gate OOF data missing columns: {missing}")
        if (oof_pairs.groupby("scene_id")["oof_fold"].nunique() > 1).any():
            raise ValueError("gate OOF folds leak scenes")
        x = self._matrix(oof_pairs, fit=True)
        self.benefit_head = _BinaryHead.fit(x, oof_pairs["beneficial"].astype(int).to_numpy(), seed=self.seed)
        self.harm_head = _BinaryHead.fit(x, oof_pairs["harmful"].astype(int).to_numpy(), seed=self.seed + 1)
        return self

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.benefit_head is None or self.harm_head is None:
            raise RuntimeError("gate is not fitted")
        x = self._matrix(pairs, fit=False)
        output = pairs.copy()
        output["p_benefit"] = self.benefit_head.predict(x)
        output["p_harm"] = self.harm_head.predict(x)
        return output

    def artifact(self) -> dict[str, Any]:
        if any(value is None for value in (self.median, self.mean, self.scale, self.benefit_head, self.harm_head)):
            raise RuntimeError("gate is not fitted")
        return {
            "kind": "two_head_logistic",
            "fit_scope": "scene-grouped OOF pair outcomes",
            "feature_columns": list(self.feature_columns),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "benefit_head": self.benefit_head.artifact(),
            "harm_head": self.harm_head.artifact(),
            "default_action": "KEEP_BASELINE",
        }


@dataclass
class ExpectedGainGate:
    """Four-outcome OOF gate used by the preregistered expected-gain rule."""

    kind: str = "multinomial_logistic"
    feature_columns: tuple[str, ...] = PAIR_FEATURES
    seed: int = 42
    median: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    model: Any = None

    def _matrix(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        matrix = frame.loc[:, self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        matrix[~np.isfinite(matrix)] = np.nan
        if fit:
            safe_matrix = matrix.copy()
            safe_matrix[:, np.isnan(safe_matrix).all(axis=0)] = 0.0
            self.median = np.nanmedian(safe_matrix, axis=0)
            self.median = np.where(np.isfinite(self.median), self.median, 0.0)
            filled = np.where(np.isfinite(matrix), matrix, self.median)
            self.mean = filled.mean(axis=0)
            std = filled.std(axis=0)
            self.scale = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        if self.median is None or self.mean is None or self.scale is None:
            raise RuntimeError("expected-gain gate preprocessor is not fitted")
        filled = np.where(np.isfinite(matrix), matrix, self.median)
        return (filled - self.mean) / self.scale

    def fit(self, oof_pairs: pd.DataFrame) -> "ExpectedGainGate":
        required = {"scene_id", "oof_fold", "outcome", *self.feature_columns}
        missing = sorted(required - set(oof_pairs.columns))
        if missing:
            raise ValueError(f"expected-gain OOF data missing columns: {missing}")
        if (oof_pairs.groupby("scene_id")["oof_fold"].nunique() > 1).any():
            raise ValueError("expected-gain gate OOF folds leak scenes")
        labels = oof_pairs["outcome"].astype(str)
        unknown = sorted(set(labels) - set(OUTCOME_CLASSES))
        if unknown:
            raise ValueError(f"unknown expected-gain outcomes: {unknown}")
        x = self._matrix(oof_pairs, fit=True)
        if self.kind == "multinomial_logistic":
            self.model = LogisticRegression(
                C=0.1,
                solver="lbfgs",
                max_iter=3000,
                random_state=self.seed,
            )
        elif self.kind == "gradient_boosted":
            self.model = HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1e-3,
                random_state=self.seed,
            )
        else:
            raise ValueError(f"unknown expected-gain gate kind: {self.kind}")
        self.model.fit(x, labels)
        return self

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("expected-gain gate is not fitted")
        probability = self.model.predict_proba(self._matrix(pairs, fit=False))
        output = pairs.copy()
        available = {str(name): index for index, name in enumerate(self.model.classes_)}
        for outcome in OUTCOME_CLASSES:
            output[f"p_{outcome.lower().replace('-', '_')}"] = (
                probability[:, available[outcome]] if outcome in available else 0.0
            )
        probability_columns = [
            f"p_{outcome.lower().replace('-', '_')}" for outcome in OUTCOME_CLASSES
        ]
        total = output[probability_columns].sum(axis=1).to_numpy(dtype=float)
        if not np.allclose(total, 1.0, atol=1e-8):
            raise AssertionError("expected-gain probabilities do not sum to one")
        return output

    def artifact(self) -> dict[str, Any]:
        if self.model is None or any(value is None for value in (self.median, self.mean, self.scale)):
            raise RuntimeError("expected-gain gate is not fitted")
        return {
            "kind": self.kind,
            "fit_scope": "scene-grouped OOF four-outcome pairs",
            "probability_interpretation": "unweighted empirical four-outcome posterior; operating point selected on held-out Validation",
            "feature_columns": list(self.feature_columns),
            "outcome_classes": list(OUTCOME_CLASSES),
            "observed_classes": list(map(str, self.model.classes_)),
            "seed": self.seed,
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "default_action": "KEEP_BASELINE",
        }


def apply_expected_gain_gate(
    pairs: pd.DataFrame,
    *,
    lambda_h: float,
    tau_u: float,
    tau_margin: float,
    tau_reliability: float,
    minimum_seed_agreement: float = 2.0 / 3.0,
) -> pd.DataFrame:
    """Apply a fail-closed expected-gain switch rule to one row per sample."""

    required = {
        "sample_id",
        "baseline_candidate_id",
        "challenger_candidate_id",
        "challenger_available",
        "p_recovered",
        "p_harmful",
        "reranker_score_delta",
        "feature_reliability_minimum",
        "seed_agreement_fraction",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"expected-gain decisions missing columns: {missing}")
    output = pairs.copy()
    output["expected_utility"] = (
        output["p_recovered"].astype(float)
        - float(lambda_h) * output["p_harmful"].astype(float)
    )
    switch = (
        output["challenger_available"].astype(bool)
        & (output["expected_utility"] > float(tau_u))
        & (output["reranker_score_delta"] > float(tau_margin))
        & (output["feature_reliability_minimum"] >= float(tau_reliability))
        & (output["seed_agreement_fraction"] >= float(minimum_seed_agreement))
    )
    output["switch"] = switch
    output["selected_candidate_id"] = np.where(
        switch,
        output["challenger_candidate_id"],
        output["baseline_candidate_id"],
    )
    output["fallback"] = ~switch
    return output


def expected_gain_sweep(
    predicted_pairs: pd.DataFrame,
    *,
    universe: pd.DataFrame,
    lambda_values: Sequence[float],
    tau_values: Sequence[float] = tuple(np.linspace(-0.05, 0.45, 11)),
    margin_values: Sequence[float] = (0.0, 0.05, 0.10),
    reliability_values: Sequence[float] = (0.0, 0.6),
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20260806,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validation sweep with one shared scene-bootstrap design matrix.

    The never-switch point is explicit.  Every operating point uses the same
    10,000 scene resamples, which makes lower-bound comparisons deterministic.
    """

    required = {"sample_id", "scene_id", "baseline_correct", "challenger_correct"}
    missing = sorted(required - set(predicted_pairs.columns))
    if missing:
        raise ValueError(f"expected-gain sweep labels missing columns: {missing}")
    if not {"sample_id", "scene_id"}.issubset(universe.columns):
        raise ValueError("expected-gain sweep universe requires sample_id and scene_id")
    canonical = universe[["sample_id", "scene_id"]].astype(str).copy()
    if canonical["sample_id"].duplicated().any():
        raise ValueError("expected-gain sweep universe contains duplicate samples")
    if predicted_pairs["sample_id"].astype(str).duplicated().any():
        raise ValueError("expected-gain sweep contains duplicate sample pairs")
    if not set(predicted_pairs["sample_id"].astype(str)).issubset(
        set(canonical["sample_id"])
    ):
        raise ValueError("expected-gain sweep pairs fall outside the canonical universe")
    scenes = np.asarray(sorted(canonical["scene_id"].unique()))
    if not len(scenes):
        raise ValueError("expected-gain sweep requires scenes")
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    sample_scene = canonical["scene_id"].map(scene_index).to_numpy(dtype=int)
    scene_sizes = np.bincount(sample_scene, minlength=len(scenes)).astype(float)
    rng = np.random.default_rng(int(bootstrap_seed))
    sampled = rng.integers(0, len(scenes), size=(int(bootstrap_draws), len(scenes)))
    draw_counts = np.zeros((int(bootstrap_draws), len(scenes)), dtype=np.int16)
    row_index = np.repeat(np.arange(int(bootstrap_draws)), len(scenes))
    np.add.at(draw_counts, (row_index, sampled.ravel()), 1)
    denominators = draw_counts @ scene_sizes

    configurations: list[dict[str, float | str]] = [
        {
            "operating_point": "never_switch",
            "lambda_h": float(max(lambda_values)),
            "tau_u": float("inf"),
            "tau_margin": float("inf"),
            "tau_reliability": 1.0,
        }
    ]
    for lambda_h in lambda_values:
        for tau_u in tau_values:
            for tau_margin in margin_values:
                for tau_reliability in reliability_values:
                    configurations.append(
                        {
                            "operating_point": "candidate",
                            "lambda_h": float(lambda_h),
                            "tau_u": float(tau_u),
                            "tau_margin": float(tau_margin),
                            "tau_reliability": float(tau_reliability),
                        }
                    )
    rows: list[dict[str, Any]] = []
    for config in configurations:
        if config["operating_point"] == "never_switch":
            switched = np.zeros(len(predicted_pairs), dtype=bool)
        else:
            decisions = apply_expected_gain_gate(
                predicted_pairs,
                lambda_h=float(config["lambda_h"]),
                tau_u=float(config["tau_u"]),
                tau_margin=float(config["tau_margin"]),
                tau_reliability=float(config["tau_reliability"]),
            )
            switched = decisions["switch"].to_numpy(dtype=bool)
        baseline = predicted_pairs["baseline_correct"].to_numpy(dtype=bool)
        challenger = predicted_pairs["challenger_correct"].to_numpy(dtype=bool)
        recovered_mask = switched & ~baseline & challenger
        harmful_mask = switched & baseline & ~challenger
        pair_delta = recovered_mask.astype(np.int8) - harmful_mask.astype(np.int8)
        delta_by_sample = dict(
            zip(predicted_pairs["sample_id"].astype(str), pair_delta, strict=True)
        )
        sample_delta = (
            canonical["sample_id"].map(delta_by_sample).fillna(0).to_numpy(dtype=np.int8)
        )
        scene_net = np.bincount(sample_scene, weights=sample_delta, minlength=len(scenes))
        distribution = (draw_counts @ scene_net) / np.maximum(denominators, 1.0)
        recovered = int(recovered_mask.sum())
        harmful = int(harmful_mask.sum())
        rows.append(
            {
                **config,
                "switch_count": int(switched.sum()),
                "switch_rate": float(switched.sum() / max(len(canonical), 1)),
                "recovered": recovered,
                "harmful": harmful,
                "net": recovered - harmful,
                "delta_j_at_1": float(sample_delta.mean()),
                "outcome_changing_precision": recovered / max(recovered + harmful, 1),
                "bootstrap_lower": float(np.quantile(distribution, 0.025)),
                "bootstrap_upper": float(np.quantile(distribution, 0.975)),
                "bootstrap_draws": int(bootstrap_draws),
                "scene_count": int(len(scenes)),
                "complete_denominator": int(len(canonical)),
            }
        )
    sweep = pd.DataFrame(rows)
    safe = sweep.sort_values(
        ["bootstrap_lower", "delta_j_at_1", "harmful", "switch_rate"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    max_gain = sweep.sort_values(
        ["delta_j_at_1", "harmful", "switch_rate"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    selection = {
        "safe_lcb": safe.to_dict(),
        "max_net_gain": max_gain.to_dict(),
        "deploy_safe_lcb": bool(float(safe["bootstrap_lower"]) > 0.0),
        "fallback_if_not_positive": "never_switch",
    }
    return sweep, selection


def apply_gate(
    pairs: pd.DataFrame,
    *,
    tau: float,
    eta: float,
    query_threshold: float,
    reliability_minimum: float = 0.6,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "baseline_candidate_id",
        "challenger_candidate_id",
        "challenger_available",
        "baseline_probability",
        "p_benefit",
        "p_harm",
        "feature_reliability_minimum",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"gate decisions missing columns: {missing}")
    output = pairs.copy()
    query = (1.0 - output["baseline_probability"]) >= float(query_threshold)
    switch = (
        output["challenger_available"].astype(bool)
        & query
        & (output["p_benefit"] >= float(tau))
        & (output["p_harm"] <= float(eta))
        & (output["feature_reliability_minimum"] >= float(reliability_minimum))
    )
    output["query_gate_passed"] = query
    output["switch"] = switch
    output["selected_candidate_id"] = np.where(
        switch,
        output["challenger_candidate_id"],
        output["baseline_candidate_id"],
    )
    output["gate_reason"] = np.select(
        [
            ~output["challenger_available"].astype(bool),
            ~query,
            output["feature_reliability_minimum"] < reliability_minimum,
            output["p_benefit"] < tau,
            output["p_harm"] > eta,
            switch,
        ],
        [
            "KEEP_NO_CHALLENGER",
            "KEEP_QUERY_GATE",
            "KEEP_UNRELIABLE",
            "KEEP_LOW_BENEFIT",
            "KEEP_HIGH_HARM",
            "SWITCH_GATE_PASS",
        ],
        default="KEEP_BASELINE",
    )
    return output


def threshold_sweep(
    predicted_pairs: pd.DataFrame,
    *,
    taus: Sequence[float] = tuple(np.arange(0.50, 0.96, 0.05)),
    etas: Sequence[float] = tuple(np.arange(0.00, 0.21, 0.025)),
    query_thresholds: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
    harm_rate_limit: float = 0.01,
    precision_minimum: float = 0.67,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"baseline_correct", "challenger_correct", "p_benefit", "p_harm"}
    missing = sorted(required - set(predicted_pairs.columns))
    if missing:
        raise ValueError(f"threshold sweep requires labels: {missing}")
    rows: list[dict[str, Any]] = []
    baseline_correct_total = int(predicted_pairs["baseline_correct"].sum())
    for query_threshold in query_thresholds:
        for tau in taus:
            for eta in etas:
                decisions = apply_gate(
                    predicted_pairs,
                    tau=float(tau),
                    eta=float(eta),
                    query_threshold=float(query_threshold),
                )
                switched = decisions["switch"].astype(bool)
                recovered = int((switched & ~decisions["baseline_correct"] & decisions["challenger_correct"]).sum())
                harmful = int((switched & decisions["baseline_correct"] & ~decisions["challenger_correct"]).sum())
                net = recovered - harmful
                precision = recovered / max(recovered + harmful, 1)
                harm_rate = harmful / max(baseline_correct_total, 1)
                rows.append(
                    {
                        "tau": float(tau),
                        "eta": float(eta),
                        "query_threshold": float(query_threshold),
                        "switch_count": int(switched.sum()),
                        "recovered": recovered,
                        "harmful": harmful,
                        "net": net,
                        "utility_recovered_minus_2_harmful": recovered - 2 * harmful,
                        "outcome_changing_precision": precision,
                        "harm_rate": harm_rate,
                        "eligible": bool(
                            recovered > harmful
                            and net > 0
                            and precision >= precision_minimum
                            and harm_rate <= harm_rate_limit
                        ),
                    }
                )
    sweep = pd.DataFrame(rows)
    eligible = sweep.loc[sweep["eligible"]]
    if eligible.empty:
        selection = {
            "status": "no-beneficial-switch",
            "tau": 1.0,
            "eta": 0.0,
            "query_threshold": 1.0,
            "fallback": "q_only",
        }
    else:
        chosen = eligible.sort_values(
            [
                "utility_recovered_minus_2_harmful",
                "harmful",
                "switch_count",
                "outcome_changing_precision",
            ],
            ascending=[False, True, True, False],
            kind="mergesort",
        ).iloc[0]
        selection = {
            "status": "selected",
            "tau": float(chosen["tau"]),
            "eta": float(chosen["eta"]),
            "query_threshold": float(chosen["query_threshold"]),
            "fallback": "q_only",
            "selection_metrics": chosen.to_dict(),
        }
    return sweep, selection


def validate_decisions(decisions: pd.DataFrame, candidates: pd.DataFrame) -> None:
    selected_ids_are_frozen(decisions, candidates, column="selected_candidate_id")
