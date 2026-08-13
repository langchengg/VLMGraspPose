"""Strict fit-partition calibration for D1 ranker cells."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.calibration import (
    PROBABILITY_EPSILON,
    assert_order_invariant,
    calibrator_from_serialized,
    make_calibrator,
    query_balanced_weights,
)
from unified_reranking.hashing import canonical_sha256


NATIVE_SCORE_COLUMN = "native_score_raw"


def fit_partition_calibrator(
    fit_rows: pd.DataFrame,
    *,
    method: str,
    fit_fold_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Fit one selected calibration family without seeing prediction rows."""

    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        NATIVE_SCORE_COLUMN,
        "candidate_success",
    }
    missing = sorted(required.difference(fit_rows.columns))
    if missing:
        raise ValueError(f"D1 fold calibrator fit rows miss columns: {missing}")
    if not fit_fold_ids or fit_rows.empty:
        raise ValueError("D1 fold calibrator requires fit folds and candidates")
    model = make_calibrator(method).fit(
        fit_rows[NATIVE_SCORE_COLUMN],
        fit_rows["candidate_success"],
        query_balanced_weights(fit_rows["sample_id"]),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "fit_fold_ids": list(map(int, fit_fold_ids)),
        "fit_sample_count": int(fit_rows["sample_id"].astype(str).nunique()),
        "fit_candidate_count": len(fit_rows),
        "fit_sample_identity_sha256": canonical_sha256(
            sorted(fit_rows["sample_id"].astype(str).unique())
        ),
        "model": model.serialize(),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def apply_partition_calibrator(
    frame: pd.DataFrame,
    payload: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply a persisted fit-only calibrator to any exact candidate frame."""

    unsigned = dict(payload)
    expected = unsigned.pop("content_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise RuntimeError("D1 fold calibrator payload content hash mismatch")
    required = {"sample_id", "candidate_id", "native_rank", NATIVE_SCORE_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"D1 fold calibration application misses columns: {missing}")
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError("D1 fold calibrator model payload is absent")
    result = frame.copy()
    model = calibrator_from_serialized(model_payload)
    result["calibrated_native_probability"] = model.predict(result[NATIVE_SCORE_COLUMN])
    probability = np.clip(
        result["calibrated_native_probability"].to_numpy(float),
        PROBABILITY_EPSILON,
        1 - PROBABILITY_EPSILON,
    )
    result["base_logit"] = np.log(probability) - np.log1p(-probability)
    assert_order_invariant(result, "calibrated_native_probability")
    return result
