"""Candidate-aligned dense evidence shared symmetrically across three backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .backend_maps import candidate_backend_map_features
from .consensus import BACKENDS, tri_backend_consensus_features


def tri_backend_dense_features(
    *,
    anchor_route: str,
    anchor_candidates: pd.DataFrame,
    candidate_pools: Mapping[str, pd.DataFrame],
    calibrations: Mapping[str, pd.DataFrame],
    dense_maps: Mapping[str, Mapping[str, np.ndarray] | None],
    transforms: Mapping[str, Any],
) -> pd.DataFrame:
    """Sample all backend maps at every immutable anchor candidate.

    ``transforms`` map original image coordinates into each backend's dense-map
    coordinates and back.  No peak search is performed here; nearest-candidate
    agreement is computed only against the already frozen canonical pools.
    """

    if anchor_route not in BACKENDS:
        raise ValueError(f"unknown anchor route: {anchor_route}")
    if any(set(value) != set(BACKENDS) for value in (candidate_pools, calibrations, dense_maps, transforms)):
        raise ValueError("tri-backend dense inputs must contain exactly crog/g1/c1")
    result = anchor_candidates[["sample_id", "candidate_id"]].copy()
    quality_columns: list[str] = []
    agreement_columns: list[str] = []
    width_columns: list[str] = []
    for backend in BACKENDS:
        sampled = candidate_backend_map_features(
            anchor_candidates,
            dense_maps[backend],
            transform=transforms[backend],
            allow_missing=True,
        )
        rename = {
            column: f"dense_{backend}_{column}"
            for column in sampled.columns
            if column not in {"sample_id", "candidate_id"}
        }
        sampled = sampled.rename(columns=rename)
        result = result.merge(
            sampled,
            on=["sample_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
        quality_columns.append(f"dense_{backend}_backend_quality_at_candidate")
        agreement_columns.append(f"dense_{backend}_cos_2_angle_difference_to_backend")
        width_columns.append(f"dense_{backend}_log_width_ratio_to_backend")

    nearest = tri_backend_consensus_features(
        anchor_route=anchor_route,
        candidates=candidate_pools,
        calibrations=calibrations,
    )
    result = result.merge(
        nearest,
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    quality = result[quality_columns].to_numpy(float)
    agreement = np.clip(result[agreement_columns].to_numpy(float), -1.0, 1.0)
    width_error = np.abs(result[width_columns].to_numpy(float))
    if not np.isfinite(quality).all() or not np.isfinite(agreement).all() or not np.isfinite(width_error).all():
        raise FloatingPointError("tri-backend dense evidence contains non-finite values")
    result["dense_backend_support_mean"] = quality.mean(axis=1)
    result["dense_backend_support_min"] = quality.min(axis=1)
    result["dense_backend_support_variance"] = quality.var(axis=1)
    result["dense_consensus_score"] = (
        quality * np.maximum(agreement, 0.0) * np.exp(-width_error)
    ).mean(axis=1)
    result["dense_disagreement_score"] = (
        quality.std(axis=1)
        + (1.0 - agreement).mean(axis=1)
        + np.minimum(width_error, 3.0).mean(axis=1) / 3.0
    )
    return result
