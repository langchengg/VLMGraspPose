"""D1-specific, label-isolated calibration input and audit helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CALIBRATION_METHODS = ("platt", "isotonic")
RELIABILITY_BINS = 15


def join_development_calibration(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join canonical candidates to development labels by exact candidate key."""

    candidate_columns = ("sample_id", "candidate_id", "native_rank", "native_score")
    label_columns = ("sample_id", "candidate_id", "candidate_success")
    for name, frame, columns in (
        ("candidates", candidates, candidate_columns),
        ("labels", labels, label_columns),
    ):
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"D1 calibration {name} miss columns: {missing}")
        if frame[["sample_id", "candidate_id"]].isna().any().any():
            raise ValueError(f"D1 calibration {name} contain null candidate keys")
        if frame.duplicated(["sample_id", "candidate_id"]).any():
            raise ValueError(f"D1 calibration {name} contain duplicate candidate keys")
    work = candidates.loc[:, candidate_columns].merge(
        labels.loc[:, label_columns],
        on=["sample_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(work) != len(candidates) or len(work) != len(labels):
        raise ValueError("D1 calibration candidate/label membership differs")
    success = pd.to_numeric(work["candidate_success"], errors="coerce")
    if success.isna().any() or not success.isin([0, 1]).all():
        raise ValueError("D1 calibration labels must be binary")
    work["candidate_success"] = success.astype("int8")
    return work


def top1_success_numerator(frame: pd.DataFrame, score_column: str | None = None) -> int:
    """Return the exact selected-candidate success numerator."""

    if score_column is None:
        ordered = frame.sort_values(
            ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
        )
    else:
        if score_column not in frame:
            raise ValueError(f"D1 calibration score is absent: {score_column}")
        ordered = frame.sort_values(
            ["sample_id", score_column, "native_rank", "candidate_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
    selected = ordered.groupby("sample_id", sort=False).head(1)
    return int(selected["candidate_success"].sum())


def reliability_rows(
    frame: pd.DataFrame,
    probability_column: str,
    *,
    bins: int = RELIABILITY_BINS,
) -> list[dict[str, Any]]:
    """Build fixed-width reliability rows, including empty bins."""

    probability = pd.to_numeric(frame[probability_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    labels = pd.to_numeric(frame["candidate_success"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if (
        bins <= 0
        or probability.shape != labels.shape
        or not np.isfinite(probability).all()
        or not np.isfinite(labels).all()
        or ((probability < 0) | (probability > 1)).any()
        or not np.isin(labels, [0, 1]).all()
    ):
        raise ValueError("invalid D1 reliability inputs")
    indexes = np.minimum(
        np.digitize(probability, np.linspace(0.0, 1.0, bins + 1)[1:-1]),
        bins - 1,
    )
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        selected = indexes == index
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": int(selected.sum()),
                "mean_probability": (
                    None if not selected.any() else float(probability[selected].mean())
                ),
                "observed_frequency": (
                    None if not selected.any() else float(labels[selected].mean())
                ),
            }
        )
    return rows
