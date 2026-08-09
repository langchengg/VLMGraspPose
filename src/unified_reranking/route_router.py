"""CROG-default conservative cross-route expected-gain router."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .gate import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    ConservativeTransitionModel,
    OOFTransitionData,
    _binary,
    _finite_vector,
    _identifiers,
    _probability,
)
from .statistics import cluster_bootstrap_difference


ALTERNATIVE_ROUTES = ("G1", "C1")
DEFAULT_ROUTE_TIE_BREAK = ("G1", "C1")


class CROGDefaultTransitionRouter:
    """Two OOF-trained calibrated transition models relative to CROG."""

    def __init__(self, *, seed: int = 20260808) -> None:
        self.seed = int(seed)
        self.models_: dict[str, ConservativeTransitionModel] = {}

    def fit(
        self,
        g1_data: OOFTransitionData,
        c1_data: OOFTransitionData,
    ) -> "CROGDefaultTransitionRouter":
        validated = {
            "G1": g1_data.validated(),
            "C1": c1_data.validated(),
        }
        g1 = validated["G1"]
        c1 = validated["C1"]
        # Both alternatives must be genuinely paired against the same CROG OOF rows.
        if not np.array_equal(g1[1], c1[1]):
            raise ValueError("G1 and C1 router data have different CROG outcomes")
        if not np.array_equal(g1[3], c1[3]) or not np.array_equal(g1[4], c1[4]):
            raise ValueError("G1 and C1 router data are not scene/fold aligned")
        for route, data in (("G1", g1_data), ("C1", c1_data)):
            names = tuple(name.lower() for name in data.feature_names)
            if not any(
                "candidate_exists" in name or "no_output" in name for name in names
            ):
                raise ValueError(
                    f"{route} router features must include candidate-existence/no-output flags"
                )
            self.models_[route] = ConservativeTransitionModel(
                seed=self.seed + (0 if route == "G1" else 1_000)
            ).fit(data)
        return self

    def predict_probabilities(
        self, features_by_route: Mapping[str, Any]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if set(self.models_) != set(ALTERNATIVE_ROUTES):
            raise RuntimeError("router transition models are not fitted")
        if set(features_by_route) != set(ALTERNATIVE_ROUTES):
            raise ValueError("features_by_route must contain exactly G1 and C1")
        return {
            route: self.models_[route].predict_probabilities(features_by_route[route])
            for route in ALTERNATIVE_ROUTES
        }

    def artifact(self) -> dict[str, Any]:
        if set(self.models_) != set(ALTERNATIVE_ROUTES):
            raise RuntimeError("router transition models are not fitted")
        return {
            "kind": "crog_default_expected_gain_router",
            "default_route": "CROG",
            "alternative_routes": list(ALTERNATIVE_ROUTES),
            "tie_break": list(DEFAULT_ROUTE_TIE_BREAK),
            "prediction_source": "paired_train_oof",
            "seed": self.seed,
            "transition_models": {
                route: self.models_[route].artifact() for route in ALTERNATIVE_ROUTES
            },
        }


@dataclass(frozen=True)
class RouterEvidence:
    route_margin: Any
    reliability: Any
    perturbation_stability: Any
    candidate_exists: Any

    def validated(
        self, length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            _finite_vector(self.route_margin, "route_margin", length=length),
            _probability(self.reliability, "reliability", length=length),
            _probability(
                self.perturbation_stability,
                "perturbation_stability",
                length=length,
            ),
            _binary(self.candidate_exists, "candidate_exists", length=length),
        )


@dataclass(frozen=True)
class RouterOperatingPoint:
    lambda_router: float
    utility_threshold: float
    margin_threshold: float
    reliability_threshold: float
    stability_threshold: float

    def __post_init__(self) -> None:
        numeric = np.asarray(
            [
                self.lambda_router,
                self.utility_threshold,
                self.margin_threshold,
                self.reliability_threshold,
                self.stability_threshold,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("router thresholds must be finite")
        if self.lambda_router not in {1.0, 2.0, 4.0}:
            raise ValueError("lambda_router must be one of 1, 2, or 4")
        if not 0.0 <= self.reliability_threshold <= 1.0:
            raise ValueError("reliability_threshold must lie in [0, 1]")
        if not 0.0 <= self.stability_threshold <= 1.0:
            raise ValueError("stability_threshold must lie in [0, 1]")


def _validate_route_inputs(
    probabilities: Mapping[str, tuple[Any, Any]],
    evidence_by_route: Mapping[str, RouterEvidence],
) -> tuple[
    int,
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    if set(probabilities) != set(ALTERNATIVE_ROUTES):
        raise ValueError("probabilities must contain exactly G1 and C1")
    if set(evidence_by_route) != set(ALTERNATIVE_ROUTES):
        raise ValueError("evidence_by_route must contain exactly G1 and C1")
    first = np.asarray(probabilities["G1"][0])
    if first.ndim != 1 or len(first) == 0:
        raise ValueError("route probabilities must be non-empty vectors")
    length = len(first)
    clean_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    clean_evidence: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for route in ALTERNATIVE_ROUTES:
        recover, harm = probabilities[route]
        clean_probabilities[route] = (
            _probability(recover, f"{route}_probability_recover", length=length),
            _probability(harm, f"{route}_probability_harm", length=length),
        )
        clean_evidence[route] = evidence_by_route[route].validated(length)
    return length, clean_probabilities, clean_evidence


def route_utilities(
    probabilities: Mapping[str, tuple[Any, Any]],
    *,
    lambda_router: float,
) -> dict[str, np.ndarray]:
    if lambda_router not in {1.0, 2.0, 4.0}:
        raise ValueError("lambda_router must be one of 1, 2, or 4")
    if set(probabilities) != set(ALTERNATIVE_ROUTES):
        raise ValueError("probabilities must contain exactly G1 and C1")
    dummy_evidence = {
        route: RouterEvidence(
            route_margin=np.zeros(len(np.asarray(probabilities[route][0]))),
            reliability=np.ones(len(np.asarray(probabilities[route][0]))),
            perturbation_stability=np.ones(len(np.asarray(probabilities[route][0]))),
            candidate_exists=np.ones(len(np.asarray(probabilities[route][0])), dtype=bool),
        )
        for route in ALTERNATIVE_ROUTES
    }
    _, clean, _ = _validate_route_inputs(probabilities, dummy_evidence)
    return {
        route: clean[route][0] - float(lambda_router) * clean[route][1]
        for route in ALTERNATIVE_ROUTES
    }


def route_decisions(
    probabilities: Mapping[str, tuple[Any, Any]],
    evidence_by_route: Mapping[str, RouterEvidence],
    operating_point: RouterOperatingPoint,
    *,
    tie_break: Sequence[str] = DEFAULT_ROUTE_TIE_BREAK,
) -> np.ndarray:
    tie_break = tuple(map(str, tie_break))
    if len(tie_break) != 2 or set(tie_break) != set(ALTERNATIVE_ROUTES):
        raise ValueError("tie_break must contain G1 and C1 exactly once")
    length, clean_probability, clean_evidence = _validate_route_inputs(
        probabilities, evidence_by_route
    )
    utility = {
        route: clean_probability[route][0]
        - operating_point.lambda_router * clean_probability[route][1]
        for route in ALTERNATIVE_ROUTES
    }
    eligible: dict[str, np.ndarray] = {}
    for route in ALTERNATIVE_ROUTES:
        margin, reliability, stability, exists = clean_evidence[route]
        eligible[route] = (
            (utility[route] > operating_point.utility_threshold)
            & (margin > operating_point.margin_threshold)
            & (reliability > operating_point.reliability_threshold)
            & (stability >= operating_point.stability_threshold)
            & exists
        )
    decisions = np.full(length, "CROG", dtype=object)
    # Iterating in tie-break order and replacing only on strict improvement makes
    # exact G1/C1 utility ties deterministic and auditable.
    best_utility = np.full(length, -np.inf, dtype=np.float64)
    for route in tie_break:
        choose = eligible[route] & (utility[route] > best_utility)
        decisions[choose] = route
        best_utility[choose] = utility[route][choose]
    return decisions


@dataclass(frozen=True)
class RouterOperatingPointTrial:
    operating_point: RouterOperatingPoint
    bootstrap_lower_bound: float
    mean_delta: float
    recovered: int
    harmful: int
    switch_count: int
    switch_rate: float
    g1_switches: int
    c1_switches: int


@dataclass(frozen=True)
class RouterSelectionResult:
    status: Literal["GO", "NO_GO_CROG"]
    selected_operating_point: RouterOperatingPoint | None
    trials: tuple[RouterOperatingPointTrial, ...]
    tie_break: tuple[str, str] = DEFAULT_ROUTE_TIE_BREAK
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS
    bootstrap_seed: int = BOOTSTRAP_SEED

    def artifact(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "default_route": "CROG",
            "tie_break": list(self.tie_break),
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
            "no_go_rule": "CROG when every lower bound <= 0",
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
        }


def select_router_operating_point(
    probabilities: Mapping[str, tuple[Any, Any]],
    evidence_by_route: Mapping[str, RouterEvidence],
    route_correct: Mapping[str, Any],
    scene_ids: Any,
    operating_points: Sequence[RouterOperatingPoint],
    *,
    tie_break: Sequence[str] = DEFAULT_ROUTE_TIE_BREAK,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> RouterSelectionResult:
    points = tuple(operating_points)
    if not points:
        raise ValueError("at least one validation operating point is required")
    length, _, _ = _validate_route_inputs(probabilities, evidence_by_route)
    if set(route_correct) != {"CROG", "G1", "C1"}:
        raise ValueError("route_correct must contain exactly CROG, G1, and C1")
    correct = {
        route: _binary(route_correct[route], f"{route}_correct", length=length)
        for route in ("CROG", "G1", "C1")
    }
    scenes = _identifiers(scene_ids, "scene_ids", length=length)
    normalized_tie_break = tuple(map(str, tie_break))
    trials: list[RouterOperatingPointTrial] = []
    indexes = np.arange(length)
    for point in points:
        decisions = route_decisions(
            probabilities,
            evidence_by_route,
            point,
            tie_break=normalized_tie_break,
        )
        selected = np.asarray(
            [correct[str(route)][index] for index, route in zip(indexes, decisions)],
            dtype=bool,
        )
        crog = correct["CROG"]
        bootstrap = cluster_bootstrap_difference(
            crog,
            selected,
            scenes,
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed),
        )
        switched = decisions != "CROG"
        trials.append(
            RouterOperatingPointTrial(
                operating_point=point,
                bootstrap_lower_bound=float(bootstrap["ci95"][0]),
                mean_delta=float(selected.mean() - crog.mean()),
                recovered=int((~crog & selected).sum()),
                harmful=int((crog & ~selected).sum()),
                switch_count=int(switched.sum()),
                switch_rate=float(switched.mean()),
                g1_switches=int((decisions == "G1").sum()),
                c1_switches=int((decisions == "C1").sum()),
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
        return RouterSelectionResult(
            status="NO_GO_CROG",
            selected_operating_point=None,
            trials=tuple(trials),
            tie_break=normalized_tie_break,
            bootstrap_iterations=int(bootstrap_iterations),
            bootstrap_seed=int(bootstrap_seed),
        )
    return RouterSelectionResult(
        status="GO",
        selected_operating_point=best.operating_point,
        trials=tuple(trials),
        tie_break=normalized_tie_break,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )


__all__ = [
    "ALTERNATIVE_ROUTES",
    "CROGDefaultTransitionRouter",
    "DEFAULT_ROUTE_TIE_BREAK",
    "RouterEvidence",
    "RouterOperatingPoint",
    "RouterOperatingPointTrial",
    "RouterSelectionResult",
    "route_decisions",
    "route_utilities",
    "select_router_operating_point",
]
