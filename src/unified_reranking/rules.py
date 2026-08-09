"""Predeclared, interpretable R1 utilities for candidate reranking.

Feature selection is schema-driven and label-free: each evidence family has a
fixed alias priority and orientation.  Fold-standardised values are required
so an ``alpha`` has the same meaning across routes and tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class RuleFeature:
    family: str
    column: str
    direction: int


# Positive direction always means "more useful / safer".  The first column
# available in a frozen feature schema is selected without consulting labels.
RULE_ALIASES: dict[str, tuple[tuple[str, int], ...]] = {
    "soft_target_support": (
        ("p_center", 1),
        ("rectangle_probability_mean", 1),
        ("center_prob", 1),
        ("image_support", 1),
    ),
    "jaw_support": (
        ("jaw_probability_min", 1),
        ("same_component_for_contacts", 1),
        ("width_symmetry", 1),
        ("binary_coverage", 1),
    ),
    "angle_agreement": (
        ("candidate_angle_to_mask_principal_axis", -1),
        ("angle_consistency", 1),
        ("angle_concentration", 1),
    ),
    "depth_contact": (
        ("contact_depth_symmetry", 1),
        ("contact_depth_abs_difference", -1),
        ("valid_depth_ratio_contacts", 1),
    ),
    "collision_proxy": (
        ("finger_sweep_obstacle_max", -1),
        ("background_intrusion_fraction", -1),
        ("approach_context_obstacle_ratio", -1),
    ),
}


def resolve_rule_features(
    columns: Iterable[str],
    *,
    require_all: bool = False,
) -> tuple[RuleFeature, ...]:
    """Resolve one predeclared feature per family from a frozen schema."""

    available = set(map(str, columns))
    resolved: list[RuleFeature] = []
    missing: list[str] = []
    for family, aliases in RULE_ALIASES.items():
        match = next(((column, direction) for column, direction in aliases if column in available), None)
        if match is None:
            missing.append(family)
        else:
            resolved.append(RuleFeature(family, match[0], match[1]))
    if require_all and missing:
        raise ValueError(f"feature schema cannot support R1 families: {missing}")
    if not resolved:
        raise ValueError("feature schema cannot support any R1 rule")
    return tuple(resolved)


def single_rule_scores(
    base_logits: Sequence[float],
    standardised_features: np.ndarray,
    columns: Sequence[str],
    *,
    family: str,
    alpha: float,
) -> np.ndarray:
    """Apply ``base_logit + alpha * oriented_standardised_feature``."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    matrix = np.asarray(standardised_features, dtype=float)
    base = np.asarray(base_logits, dtype=float)
    if matrix.ndim != 2 or base.shape != (len(matrix),):
        raise ValueError("invalid rule score inputs")
    if not np.isfinite(matrix).all() or not np.isfinite(base).all():
        raise ValueError("rule score inputs must be finite")
    feature = next(
        (item for item in resolve_rule_features(columns) if item.family == family),
        None,
    )
    if feature is None:
        raise ValueError(f"feature family unavailable: {family}")
    index = tuple(map(str, columns)).index(feature.column)
    return base + float(alpha) * feature.direction * matrix[:, index]


@dataclass
class SignConstrainedLinearUtility:
    """Non-negative weights over pre-oriented R1 features.

    The residual is bounded with ``tanh`` and candidate BCE is weighted so each
    query contributes equally, matching the neural R2/R3 supervision contract.
    """

    alpha: float = 0.5
    l2: float = 1e-4
    features_: tuple[RuleFeature, ...] | None = None
    weights_: np.ndarray | None = None
    intercept_: float | None = None
    optimiser_: dict[str, object] | None = None

    def fit(
        self,
        standardised_features: np.ndarray,
        labels: Sequence[int],
        query_ids: Sequence[object],
        base_logits: Sequence[float],
        columns: Sequence[str],
    ) -> "SignConstrainedLinearUtility":
        if self.alpha < 0 or self.l2 < 0:
            raise ValueError("alpha and l2 must be non-negative")
        matrix = np.asarray(standardised_features, dtype=float)
        y = np.asarray(labels, dtype=float)
        base = np.asarray(base_logits, dtype=float)
        queries = np.asarray(list(map(str, query_ids)), dtype=object)
        if matrix.ndim != 2 or y.shape != (len(matrix),) or base.shape != y.shape or queries.shape != y.shape:
            raise ValueError("invalid constrained utility inputs")
        if not np.isfinite(matrix).all() or not np.isfinite(base).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("constrained utility inputs must be finite and binary")
        features = resolve_rule_features(columns)
        column_names = tuple(map(str, columns))
        selected = np.column_stack(
            [item.direction * matrix[:, column_names.index(item.column)] for item in features]
        )
        _, inverse, counts = np.unique(queries, return_inverse=True, return_counts=True)
        row_weight = 1.0 / counts[inverse]
        row_weight /= row_weight.sum()

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            weights, intercept = parameters[:-1], parameters[-1]
            raw = selected @ weights + intercept
            bounded = np.tanh(raw)
            logits = base + self.alpha * bounded
            probabilities = expit(logits)
            eps = 1e-12
            loss = -np.sum(row_weight * (y * np.log(probabilities + eps) + (1 - y) * np.log1p(-probabilities + eps)))
            loss += 0.5 * self.l2 * float(weights @ weights)
            d_logits = row_weight * (probabilities - y)
            d_raw = d_logits * self.alpha * (1.0 - bounded * bounded)
            gradient = np.r_[selected.T @ d_raw + self.l2 * weights, d_raw.sum()]
            return float(loss), gradient

        initial = np.zeros(len(features) + 1, dtype=float)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * len(features) + [(None, None)],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"sign-constrained utility fit failed: {result.message}")
        self.features_ = features
        self.weights_ = np.asarray(result.x[:-1], dtype=float)
        self.intercept_ = float(result.x[-1])
        self.optimiser_ = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(result.fun),
        }
        return self

    def predict(
        self,
        standardised_features: np.ndarray,
        base_logits: Sequence[float],
        columns: Sequence[str],
    ) -> np.ndarray:
        if self.features_ is None or self.weights_ is None or self.intercept_ is None:
            raise RuntimeError("utility has not been fitted")
        matrix = np.asarray(standardised_features, dtype=float)
        base = np.asarray(base_logits, dtype=float)
        if matrix.ndim != 2 or base.shape != (len(matrix),) or not np.isfinite(matrix).all() or not np.isfinite(base).all():
            raise ValueError("invalid constrained utility prediction inputs")
        column_names = tuple(map(str, columns))
        missing = [item.column for item in self.features_ if item.column not in column_names]
        if missing:
            raise ValueError(f"prediction schema misses fitted features: {missing}")
        selected = np.column_stack(
            [item.direction * matrix[:, column_names.index(item.column)] for item in self.features_]
        )
        residual = selected @ self.weights_ + self.intercept_
        return base + self.alpha * np.tanh(residual)

    def artifact(self) -> dict[str, object]:
        if self.features_ is None or self.weights_ is None or self.intercept_ is None:
            raise RuntimeError("utility has not been fitted")
        return {
            "alpha": float(self.alpha),
            "l2": float(self.l2),
            "features": [
                {"family": item.family, "column": item.column, "direction": item.direction}
                for item in self.features_
            ],
            "weights": self.weights_.tolist(),
            "intercept": self.intercept_,
            "optimiser": self.optimiser_,
        }


__all__ = [
    "RULE_ALIASES",
    "RuleFeature",
    "SignConstrainedLinearUtility",
    "resolve_rule_features",
    "single_rule_scores",
]
