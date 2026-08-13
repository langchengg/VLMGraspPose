"""Deterministic three-seed aggregation and Validation selection for D1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.hashing import canonical_sha256


PREDICTION_KEY_COLUMNS = (
    "sample_id",
    "candidate_id",
    "native_rank",
    "candidate_identity_sha256",
    "candidate_geometry_sha256",
)


def trial_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"seed", "mode", "held_fold"}
    return {
        str(key): value
        for key, value in configuration.items()
        if str(key) not in excluded
    }


def trial_id(configuration: Mapping[str, Any]) -> str:
    return canonical_sha256(trial_configuration(configuration))[:16]


def ensemble_seed_scores(
    seed_predictions: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    """Require exact candidate identity across three seeds and average scores."""

    if tuple(sorted(seed_predictions)) != (42, 123, 2026):
        raise ValueError("D1 ensemble requires exactly seeds 42, 123 and 2026")
    result: pd.DataFrame | None = None
    for seed in (42, 123, 2026):
        frame = seed_predictions[seed]
        required = {*PREDICTION_KEY_COLUMNS, "score"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"D1 seed {seed} predictions miss columns: {missing}")
        if frame.duplicated(["sample_id", "candidate_id"]).any():
            raise ValueError(f"D1 seed {seed} predictions duplicate candidates")
        values = pd.to_numeric(frame["score"], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"D1 seed {seed} predictions contain non-finite scores")
        current = frame.loc[:, [*PREDICTION_KEY_COLUMNS]].copy()
        current[f"score_seed_{seed}"] = values
        result = (
            current
            if result is None
            else result.merge(
                current,
                on=list(PREDICTION_KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
            )
        )
        if result is None or len(result) != len(frame):
            raise ValueError("D1 ensemble seed candidate universes differ")
    assert result is not None
    score_columns = [f"score_seed_{seed}" for seed in (42, 123, 2026)]
    result["ensemble_score"] = result[score_columns].mean(axis=1)
    return result.sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)


def select_validation_winner(
    rows: Sequence[Mapping[str, Any]],
    *,
    final_tie_column: str,
) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("D1 Validation selection has no eligible trials")
    return min(
        rows,
        key=lambda row: (
            -float(row["validation_j_at_1"]),
            -float(row["validation_mrr_at_5"]),
            -float(row["validation_ndcg_at_5"]),
            str(row[final_tie_column]),
        ),
    )
