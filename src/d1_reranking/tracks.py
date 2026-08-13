"""Final evidence-track assembly for the isolated D1 extension."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.feature_tracks import assemble_common_track


JOIN_KEYS = ("sample_id", "candidate_id")


def _candidate_keys(frame: pd.DataFrame, *, name: str) -> pd.MultiIndex:
    missing = sorted(set(JOIN_KEYS).difference(frame.columns))
    if missing:
        raise ValueError(f"D1 {name} misses candidate keys: {missing}")
    work = frame.loc[:, list(JOIN_KEYS)].astype(str)
    if work.isna().any().any() or work.duplicated().any():
        raise ValueError(f"D1 {name} has invalid candidate keys")
    return pd.MultiIndex.from_frame(work)


def _require_exact_candidate_contract(
    candidates: pd.DataFrame,
    raw_common: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    expected = _candidate_keys(candidates, name="candidates")
    expected_set = set(expected.tolist())
    for name, frame in (("raw common", raw_common), ("calibration", calibration)):
        observed = _candidate_keys(frame, name=name)
        if len(observed) != len(expected) or set(observed.tolist()) != expected_set:
            raise ValueError(f"D1 {name} candidate membership differs")
    canonical = candidates.set_index(list(JOIN_KEYS), drop=False).sort_index()
    for name, frame in (("raw common", raw_common), ("calibration", calibration)):
        observed = frame.set_index(list(JOIN_KEYS), drop=False).sort_index()
        for column in (
            "native_rank",
            "native_score",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ):
            if column not in canonical or column not in observed:
                continue
            if column in {"native_rank"}:
                equal = np.array_equal(
                    canonical[column].to_numpy(int), observed[column].to_numpy(int)
                )
            elif column in {"native_score"}:
                equal = np.array_equal(
                    canonical[column].to_numpy(np.float64),
                    observed[column].to_numpy(np.float64),
                )
            else:
                equal = (
                    canonical[column].astype(str).equals(observed[column].astype(str))
                )
            if not equal:
                raise ValueError(f"D1 {name} {column} differs from candidates")


def finalize_matched_common(
    candidates: pd.DataFrame,
    raw_common: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add label-free calibrated q/logit to the exact common feature universe."""

    _require_exact_candidate_contract(candidates, raw_common, calibration)
    track = assemble_common_track(candidates, raw_common, calibration)
    columns = assert_model_feature_columns(track.model_columns)
    required = {"calibrated_native_probability", "base_logit", "native_rank"}
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"D1 T2 final model schema misses required fields: {missing}")
    selected = track.frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if selected.shape[0] != len(candidates):
        raise ValueError("D1 T2 final row count differs from candidates")
    if np.isinf(selected.to_numpy(dtype=np.float64)).any():
        raise ValueError("D1 T2 final model features contain infinity")
    return track.frame, columns


def missingness_rows(frame: pd.DataFrame, model_columns: Sequence[str]) -> pd.DataFrame:
    columns = assert_model_feature_columns(model_columns)
    return pd.DataFrame(
        {
            "column": columns,
            "missing_fraction": [
                float(
                    1.0
                    - np.isfinite(
                        pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                    ).mean()
                )
                for column in columns
            ],
        }
    )
