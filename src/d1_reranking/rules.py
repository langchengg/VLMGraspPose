"""Predeclared interpretable D1 R1 soft-support rules."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


R1_EPSILON = 1e-4
R1_FORMULA = (
    "base_logit + beta_center*log(clip(p_center,1e-4,1)) + "
    "beta_rect*log(clip(rectangle_probability_mean,1e-4,1)) + "
    "beta_jaw*log(clip(jaw_probability_min,1e-4,1))"
)
R1_SUPPORT_COLUMNS = (
    "p_center",
    "rectangle_probability_mean",
    "jaw_probability_min",
)


def r1_score(
    frame: pd.DataFrame,
    trial: Mapping[str, object],
) -> np.ndarray:
    """Compute the exact log-additive R1 score with no learned state."""

    required = {"base_logit", *R1_SUPPORT_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"D1 R1 features miss columns: {missing}")
    base = pd.to_numeric(frame["base_logit"], errors="coerce").to_numpy(float)
    support = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        for column in R1_SUPPORT_COLUMNS
    }
    if not np.isfinite(base).all() or any(
        not np.isfinite(values).all() for values in support.values()
    ):
        raise ValueError("D1 R1 requires finite calibrated/support features")
    if any(((values < 0) | (values > 1)).any() for values in support.values()):
        raise ValueError("D1 R1 support features must lie in [0,1]")
    beta_center = float(trial.get("beta_center", 0.0))
    beta_rect = float(trial.get("beta_rect", 0.0))
    beta_jaw = float(trial.get("beta_jaw", 0.0))
    if min(beta_center, beta_rect, beta_jaw) < 0:
        raise ValueError("D1 R1 beta coefficients must be non-negative")
    score = (
        base
        + beta_center * np.log(np.clip(support["p_center"], R1_EPSILON, 1.0))
        + beta_rect
        * np.log(np.clip(support["rectangle_probability_mean"], R1_EPSILON, 1.0))
        + beta_jaw * np.log(np.clip(support["jaw_probability_min"], R1_EPSILON, 1.0))
    )
    if not np.isfinite(score).all():
        raise RuntimeError("D1 R1 produced non-finite scores")
    return score
