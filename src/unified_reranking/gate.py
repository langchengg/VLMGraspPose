"""Conservative OOF-trained transition gate for order-only reranking.

This module is deliberately data-source agnostic: it accepts arrays supplied by
the development pipeline and never reads candidate tables or Test artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .statistics import cluster_bootstrap_difference


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260808

SAFE_GATE_FEATURE_COLUMNS = (
    "ranker_score_margin",
    "challenger_calibrated_probability",
    "native_calibrated_probability",
    "calibrated_probability_delta",
    "native_score_delta",
    "challenger_overall_reliability",
    "overall_reliability_delta",
    "challenger_perturbation_stability",
    "perturbation_stability_delta",
    "challenger_mask_reliability",
    "mask_reliability_delta",
    "challenger_exists_numeric",
)

_FORBIDDEN_FEATURE_TOKENS = (
    "candidate_success",
    "ground_truth",
    "best_same_gt",
    "matched_gt",
    "jacquard_margin",
    "first_positive",
    "native_correct",
    "challenger_correct",
    "recovered",
    "harmful",
    "sample_id",
    "scene_id",
    "object_category",
)


def _one_dimensional(values: Any, name: str, *, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must be one-dimensional and match rows")
    return array


def _binary(values: Any, name: str, *, length: int) -> np.ndarray:
    array = _one_dimensional(values, name, length=length)
    numeric = np.asarray(array, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain finite binary values")
    return numeric.astype(bool)


def _finite_vector(values: Any, name: str, *, length: int) -> np.ndarray:
    array = np.asarray(
        _one_dimensional(values, name, length=length), dtype=np.float64
    )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _probability(values: Any, name: str, *, length: int) -> np.ndarray:
    array = _finite_vector(values, name, length=length)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def _identifiers(values: Any, name: str, *, length: int) -> np.ndarray:
    array = _one_dimensional(values, name, length=length)
    if any(value is None or str(value) == "" for value in array.tolist()):
        raise ValueError(f"{name} must contain non-empty identifiers")
    return np.asarray([str(value) for value in array], dtype=object)


@dataclass(frozen=True)
class OOFTransitionData:
    """Ranker OOF evidence and transition outcomes used to train a gate."""

    features: Any
    feature_names: tuple[str, ...]
    native_correct: Any
    challenger_correct: Any
    scene_ids: Any
    oof_fold_ids: Any
    prediction_source: Literal["train_oof"] = "train_oof"

    def validated(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.prediction_source != "train_oof":
            raise ValueError("transition models may train only on Train OOF predictions")
        matrix = np.asarray(self.features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("features must be a non-empty two-dimensional matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("features must be finite")
        names = tuple(map(str, self.feature_names))
        if len(names) != matrix.shape[1] or len(set(names)) != len(names):
            raise ValueError("feature_names must uniquely match feature columns")
        forbidden = [
            name
            for name in names
            if any(token in name.lower() for token in _FORBIDDEN_FEATURE_TOKENS)
        ]
        if forbidden:
            raise ValueError(f"gate features contain supervision/identity columns: {forbidden}")
        length = len(matrix)
        native = _binary(self.native_correct, "native_correct", length=length)
        challenger = _binary(
            self.challenger_correct, "challenger_correct", length=length
        )
        scenes = _identifiers(self.scene_ids, "scene_ids", length=length)
        folds = _identifiers(self.oof_fold_ids, "oof_fold_ids", length=length)
        if len(np.unique(scenes)) < 2 or len(np.unique(folds)) < 2:
            raise ValueError("at least two scenes and OOF folds are required")
        for scene in np.unique(scenes):
            if len(np.unique(folds[scenes == scene])) != 1:
                raise ValueError("a scene crosses OOF folds")
        return matrix, native, challenger, scenes, folds


def _fixed_fold_splits(folds: np.ndarray, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in dict.fromkeys(folds.tolist()):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        if len(training) == 0 or len(validation) == 0:
            raise ValueError("OOF fold assignments contain an empty partition")
        if len(np.unique(labels[training])) < 2 or len(np.unique(labels[validation])) < 2:
            raise ValueError(
                "every calibration train/validation fold must contain both transition classes"
            )
        splits.append((training, validation))
    return splits


def _fit_calibrated_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int,
) -> CalibratedClassifierCV:
    if len(np.unique(labels)) != 2:
        raise ValueError("transition target must contain both classes")
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    max_iter=1_000,
                    solver="lbfgs",
                    random_state=int(seed),
                ),
            ),
        ]
    )
    model = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=_fixed_fold_splits(folds, labels),
        ensemble=True,
    )
    return model.fit(features, labels.astype(np.int32))


class ConservativeTransitionModel:
    """Independent calibrated models for recovery and harm probabilities."""

    def __init__(self, *, seed: int = 20260808) -> None:
        self.seed = int(seed)
        self.feature_names_: tuple[str, ...] | None = None
        self.recover_model_: CalibratedClassifierCV | None = None
        self.harm_model_: CalibratedClassifierCV | None = None
        self.fold_count_: int | None = None

    def fit(self, data: OOFTransitionData) -> "ConservativeTransitionModel":
        features, native, challenger, _scenes, folds = data.validated()
        recovered = (~native & challenger).astype(np.int32)
        harmful = (native & ~challenger).astype(np.int32)
        self.recover_model_ = _fit_calibrated_logistic(
            features, recovered, folds, seed=self.seed
        )
        self.harm_model_ = _fit_calibrated_logistic(
            features, harmful, folds, seed=self.seed + 1
        )
        self.feature_names_ = tuple(map(str, data.feature_names))
        self.fold_count_ = int(len(np.unique(folds)))
        return self

    def predict_probabilities(self, features: Any) -> tuple[np.ndarray, np.ndarray]:
        if self.recover_model_ is None or self.harm_model_ is None or self.feature_names_ is None:
            raise RuntimeError("transition model is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names_):
            raise ValueError("features must match fitted feature width")
        if not np.isfinite(matrix).all():
            raise ValueError("features must be finite")
        recover = self.recover_model_.predict_proba(matrix)[:, 1]
        harm = self.harm_model_.predict_proba(matrix)[:, 1]
        return np.asarray(recover, dtype=np.float64), np.asarray(harm, dtype=np.float64)

    def artifact(self) -> dict[str, Any]:
        if self.feature_names_ is None or self.fold_count_ is None:
            raise RuntimeError("transition model is not fitted")
        return {
            "kind": "independent_sigmoid_calibrated_logistic_transition_models",
            "seed": self.seed,
            "prediction_source": "train_oof",
            "feature_names": list(self.feature_names_),
            "calibration_cv": "provided_group_safe_oof_folds",
            "calibration_method": "sigmoid",
            "base_estimator": {
                "kind": "standardized_logistic_regression",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1_000,
            },
            "fold_count": self.fold_count_,
            "targets": ["recovered", "harmful"],
        }


@dataclass(frozen=True)
class GateEvidence:
    score_margin: Any
    challenger_reliability: Any
    perturbation_stability: Any
    seed_challenger_votes: Any
    candidate_id_unchanged: Any
    geometry_hash_unchanged: Any
    challenger_exists: Any

    def validated(
        self, length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        score_margin = _finite_vector(
            self.score_margin, "score_margin", length=length
        )
        reliability = _probability(
            self.challenger_reliability,
            "challenger_reliability",
            length=length,
        )
        stability = _probability(
            self.perturbation_stability,
            "perturbation_stability",
            length=length,
        )
        votes = _finite_vector(
            self.seed_challenger_votes, "seed_challenger_votes", length=length
        )
        if not np.isin(votes, [0.0, 1.0, 2.0, 3.0]).all():
            raise ValueError("seed_challenger_votes must be integer counts from 0 to 3")
        return (
            score_margin,
            reliability,
            stability,
            votes.astype(np.int8),
            _binary(
                self.candidate_id_unchanged,
                "candidate_id_unchanged",
                length=length,
            ),
            _binary(
                self.geometry_hash_unchanged,
                "geometry_hash_unchanged",
                length=length,
            ),
            _binary(self.challenger_exists, "challenger_exists", length=length),
        )


def candidate_immutability_flags(
    expected_candidate_ids: Any,
    selected_candidate_ids: Any,
    expected_geometry_hashes: Any,
    selected_geometry_hashes: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare locked candidate identity and geometry without inspecting labels."""

    expected = np.asarray(expected_candidate_ids)
    if expected.ndim != 1 or len(expected) == 0:
        raise ValueError("expected_candidate_ids must be a non-empty vector")
    length = len(expected)
    expected_ids = _identifiers(
        expected_candidate_ids, "expected_candidate_ids", length=length
    )
    selected_ids = _identifiers(
        selected_candidate_ids, "selected_candidate_ids", length=length
    )
    expected_hashes = _identifiers(
        expected_geometry_hashes, "expected_geometry_hashes", length=length
    )
    selected_hashes = _identifiers(
        selected_geometry_hashes, "selected_geometry_hashes", length=length
    )
    return expected_ids == selected_ids, expected_hashes == selected_hashes


@dataclass(frozen=True)
class GateOperatingPoint:
    lambda_harm: float
    utility_threshold: float
    score_margin_threshold: float
    reliability_threshold: float
    stability_threshold: float
    minimum_seed_votes: int = 2

    def __post_init__(self) -> None:
        numeric = (
            self.lambda_harm,
            self.utility_threshold,
            self.score_margin_threshold,
            self.reliability_threshold,
            self.stability_threshold,
        )
        if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
            raise ValueError("gate thresholds must be finite")
        if self.lambda_harm not in {1.0, 2.0, 4.0}:
            raise ValueError("lambda_harm must be one of 1, 2, or 4")
        if not 0.0 <= self.reliability_threshold <= 1.0:
            raise ValueError("reliability_threshold must lie in [0, 1]")
        if not 0.0 <= self.stability_threshold <= 1.0:
            raise ValueError("stability_threshold must lie in [0, 1]")
        if self.minimum_seed_votes != 2:
            raise ValueError("the controlled gate requires at least 2-of-3 seed consensus")


def gate_switch_mask(
    probability_recover: Any,
    probability_harm: Any,
    evidence: GateEvidence,
    operating_point: GateOperatingPoint,
) -> np.ndarray:
    recover_raw = np.asarray(probability_recover)
    if recover_raw.ndim != 1 or len(recover_raw) == 0:
        raise ValueError("probability_recover must be a non-empty vector")
    length = len(recover_raw)
    recover = _probability(
        probability_recover, "probability_recover", length=length
    )
    harm = _probability(probability_harm, "probability_harm", length=length)
    margin, reliability, stability, votes, candidate_same, geometry_same, exists = (
        evidence.validated(length)
    )
    utility = recover - operating_point.lambda_harm * harm
    return (
        (utility > operating_point.utility_threshold)
        & (margin > operating_point.score_margin_threshold)
        & (reliability > operating_point.reliability_threshold)
        & (stability >= operating_point.stability_threshold)
        & (votes >= operating_point.minimum_seed_votes)
        & candidate_same
        & geometry_same
        & exists
    )


@dataclass(frozen=True)
class OperatingPointTrial:
    operating_point: GateOperatingPoint
    bootstrap_lower_bound: float
    mean_delta: float
    recovered: int
    harmful: int
    switch_count: int
    switch_rate: float


@dataclass(frozen=True)
class GateSelectionResult:
    status: Literal["GO", "NO_GO_NATIVE"]
    selected_operating_point: GateOperatingPoint | None
    trials: tuple[OperatingPointTrial, ...]
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS
    bootstrap_seed: int = BOOTSTRAP_SEED

    def artifact(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_operating_point": (
                None
                if self.selected_operating_point is None
                else asdict(self.selected_operating_point)
            ),
            "trials": [
                {**asdict(trial), "operating_point": asdict(trial.operating_point)}
                for trial in self.trials
            ],
            "selection_order": [
                "scene_bootstrap_95pct_lower_bound",
                "mean_delta",
                "fewer_harmful",
                "lower_switch_rate",
            ],
            "no_go_rule": "native when every lower bound <= 0",
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
        }


def select_gate_operating_point(
    probability_recover: Any,
    probability_harm: Any,
    evidence: GateEvidence,
    native_correct: Any,
    challenger_correct: Any,
    scene_ids: Any,
    operating_points: Sequence[GateOperatingPoint],
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> GateSelectionResult:
    points = tuple(operating_points)
    if not points:
        raise ValueError("at least one validation operating point is required")
    recover_raw = np.asarray(probability_recover)
    if recover_raw.ndim != 1 or len(recover_raw) == 0:
        raise ValueError("probability_recover must be a non-empty vector")
    length = len(recover_raw)
    native = _binary(native_correct, "native_correct", length=length)
    challenger = _binary(
        challenger_correct, "challenger_correct", length=length
    )
    scenes = _identifiers(scene_ids, "scene_ids", length=length)
    trials: list[OperatingPointTrial] = []
    for point in points:
        switches = gate_switch_mask(
            probability_recover, probability_harm, evidence, point
        )
        selected = np.where(switches, challenger, native)
        bootstrap = cluster_bootstrap_difference(
            native,
            selected,
            scenes,
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed),
        )
        trials.append(
            OperatingPointTrial(
                operating_point=point,
                bootstrap_lower_bound=float(bootstrap["ci95"][0]),
                mean_delta=float(selected.mean() - native.mean()),
                recovered=int((~native & selected).sum()),
                harmful=int((native & ~selected).sum()),
                switch_count=int(switches.sum()),
                switch_rate=float(switches.mean()),
            )
        )
    best = max(
        trials,
        key=lambda trial: (
            trial.bootstrap_lower_bound,
            trial.mean_delta,
            -trial.harmful,
            -trial.switch_rate,
        ),
    )
    if best.bootstrap_lower_bound <= 0.0:
        return GateSelectionResult(
            status="NO_GO_NATIVE",
            selected_operating_point=None,
            trials=tuple(trials),
            bootstrap_iterations=int(bootstrap_iterations),
            bootstrap_seed=int(bootstrap_seed),
        )
    return GateSelectionResult(
        status="GO",
        selected_operating_point=best.operating_point,
        trials=tuple(trials),
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )


__all__ = [
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "ConservativeTransitionModel",
    "GateEvidence",
    "GateOperatingPoint",
    "GateSelectionResult",
    "OOFTransitionData",
    "OperatingPointTrial",
    "candidate_immutability_flags",
    "gate_switch_mask",
    "select_gate_operating_point",
]
