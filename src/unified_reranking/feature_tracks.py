"""Leakage-safe assembly of native and matched-evidence feature tracks.

Candidate identity is retained only as a join key.  The returned model schema is
explicit, numeric, and checked by :mod:`unified_reranking.contracts` before it
can be consumed by a ranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import assert_model_feature_columns


JOIN_COLUMNS = ("sample_id", "candidate_id")
TRACK_ID_COLUMNS = ("sample_id", "candidate_id", "route")

# These old CROG columns are measurements of input depth or explicitly
# depth-derived contact/collision proxies.  They are forbidden in the natural
# CROG deployment interface even though the historical table contains them.
CROG_NATIVE_FORBIDDEN_TOKENS = (
    "depth",
    "clearance",
    "collision",
    "obstacle",
    "contact_depth",
    "normal",
    "surface",
    "crop_",
    "scanline",
    "axis_valid",
    "z_m",
    "z_reference",
    "width_m",
    "safety",
)

CROG_REFERENCE_NON_MODEL = frozenset(
    {
        "sample_id",
        "scene_id",
        "frame_id",
        "expression_id",
        "candidate_id",
        "candidate_identity_sha256",
        "route",
        "pool_type",
        "language_instruction",
        "image_path",
        "depth_path",
        "pcd_path",
    }
)


@dataclass(frozen=True)
class FeatureTrack:
    frame: pd.DataFrame
    model_columns: tuple[str, ...]


def _unique_keys(frame: pd.DataFrame, *, name: str) -> None:
    missing = sorted(set(JOIN_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing join columns: {missing}")
    if frame[list(JOIN_COLUMNS)].isna().any().any():
        raise ValueError(f"{name} contains null candidate keys")
    if frame.duplicated(list(JOIN_COLUMNS)).any():
        raise ValueError(f"{name} contains duplicate candidate keys")


def _assert_exact_membership(reference: pd.DataFrame, observed: pd.DataFrame) -> None:
    _unique_keys(reference, name="candidate reference")
    _unique_keys(observed, name="feature table")
    expected = set(map(tuple, reference[list(JOIN_COLUMNS)].astype(str).to_numpy()))
    actual = set(map(tuple, observed[list(JOIN_COLUMNS)].astype(str).to_numpy()))
    if expected != actual:
        raise ValueError(
            "feature/candidate membership mismatch: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )


def select_numeric_model_columns(
    frame: pd.DataFrame,
    *,
    excluded: Iterable[str] = TRACK_ID_COLUMNS,
) -> tuple[str, ...]:
    excluded_set = set(map(str, excluded))
    columns = tuple(
        sorted(
            str(column)
            for column in frame.columns
            if str(column) not in excluded_set
            and pd.api.types.is_numeric_dtype(frame[column])
        )
    )
    if not columns:
        raise ValueError("feature track has no numeric model columns")
    return assert_model_feature_columns(columns)


def merge_calibration_features(
    features: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Join only label-free calibration outputs to a feature table."""

    _unique_keys(features, name="candidate features")
    _unique_keys(calibration, name="calibration predictions")
    allowed = (
        *JOIN_COLUMNS,
        "calibrated_native_probability",
        "base_logit",
    )
    missing = sorted(set(allowed).difference(calibration.columns))
    if missing:
        raise ValueError(f"calibration table missing columns: {missing}")
    result = features.merge(
        calibration[list(allowed)],
        on=list(JOIN_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if result[["calibrated_native_probability", "base_logit"]].isna().any().any():
        raise ValueError("calibration does not cover every candidate")
    return result


def assemble_common_track(
    candidates: pd.DataFrame,
    common_features: pd.DataFrame,
    calibration: pd.DataFrame,
) -> FeatureTrack:
    """Build T2 matched-common evidence without importing candidate labels."""

    _assert_exact_membership(candidates, common_features)
    frame = merge_calibration_features(common_features, calibration)
    columns = select_numeric_model_columns(frame)
    return FeatureTrack(frame=frame, model_columns=columns)


def select_crog_native_reference_columns(reference: pd.DataFrame) -> tuple[str, ...]:
    """Select inference-time CROG evidence and reject every depth proxy."""

    columns = []
    for column in reference.columns:
        name = str(column)
        lower = name.lower()
        if name in CROG_REFERENCE_NON_MODEL:
            continue
        if not pd.api.types.is_numeric_dtype(reference[column]):
            continue
        if any(token in lower for token in CROG_NATIVE_FORBIDDEN_TOKENS):
            continue
        columns.append(name)
    result = assert_model_feature_columns(sorted(columns))
    if not result:
        raise ValueError("CROG native reference schema is empty")
    return result


def assemble_crog_native_track(
    candidates: pd.DataFrame,
    paired_manifest: pd.DataFrame,
    historical_reference: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    coordinate_atol: float = 1e-6,
) -> FeatureTrack:
    """Map audited historical CROG evidence onto the paired candidate universe.

    The reference table may contain duplicate frame/language payloads.  They are
    accepted only when every selected numeric feature and candidate geometry is
    byte-equivalent after deterministic sorting.
    """

    required_manifest = {"sample_id", "scene_id", "language"}
    missing = sorted(required_manifest.difference(paired_manifest.columns))
    if missing:
        raise ValueError(f"paired manifest missing columns: {missing}")
    required_reference = {
        "frame_id",
        "language_instruction",
        "candidate_id",
        "original_rank",
        "q_raw",
        "x_px",
        "y_px",
        "angle_rad",
        "width_px",
        "height_px",
    }
    missing = sorted(required_reference.difference(historical_reference.columns))
    if missing:
        raise ValueError(f"historical CROG reference missing columns: {missing}")

    feature_columns = select_crog_native_reference_columns(historical_reference)
    reference = historical_reference.copy()
    reference["_language_key"] = reference["language_instruction"].astype(str)
    # The historical CROG table calls the source-relative OCID-VLG frame name
    # ``frame_id``; the paired manifest calls the same value ``scene_id`` and
    # reserves ``frame_id`` for the content hash.
    mapping = paired_manifest[["sample_id", "scene_id", "language"]].copy()
    mapping["frame_id"] = mapping.pop("scene_id").astype(str)
    mapping["_language_key"] = mapping["language"].astype(str)
    joined = mapping.merge(
        reference,
        on=["frame_id", "_language_key"],
        how="left",
        validate="one_to_many",
    )
    if joined["candidate_id"].isna().any():
        missing_samples = joined.loc[joined["candidate_id"].isna(), "sample_id"].nunique()
        raise ValueError(f"historical CROG reference misses {missing_samples} paired samples")

    payload_columns = [
        "candidate_id",
        "original_rank",
        "q_raw",
        "x_px",
        "y_px",
        "angle_rad",
        "width_px",
        "height_px",
        *feature_columns,
    ]
    payload_columns = list(dict.fromkeys(payload_columns))
    # Duplicate source questions are allowed only when their five-candidate
    # numeric payloads are identical.  Drop byte-equivalent duplicates.
    distinct = joined[["sample_id", *payload_columns]].drop_duplicates()
    counts = distinct.groupby(["sample_id", "candidate_id"], sort=False).size()
    if bool((counts != 1).any()):
        raise ValueError("ambiguous non-equivalent CROG reference payload")
    mapped = distinct.drop_duplicates(["sample_id", "candidate_id"]).copy()

    candidate_columns = [
        "sample_id",
        "candidate_id",
        "route",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    ]
    frozen = candidates[candidate_columns].copy()
    merged = frozen.merge(
        mapped,
        on=list(JOIN_COLUMNS),
        how="left",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    if merged["q_raw"].isna().any():
        raise ValueError("mapped CROG features do not cover the frozen candidate pool")
    comparisons = {
        "native_rank": "original_rank",
        "native_score": "q_raw",
        "cx_px": "x_px",
        "cy_px": "y_px",
        "theta_deg": "angle_rad",
        "width_px": "width_px_reference",
        "height_px": "height_px_reference",
    }
    for frozen_name, reference_name in comparisons.items():
        left = pd.to_numeric(merged[frozen_name], errors="coerce").to_numpy(float)
        right = pd.to_numeric(merged[reference_name], errors="coerce").to_numpy(float)
        if reference_name == "angle_rad":
            right = np.degrees(right)
        if not np.allclose(left, right, rtol=0.0, atol=coordinate_atol, equal_nan=False):
            difference = float(np.max(np.abs(left - right)))
            raise ValueError(
                f"CROG frozen geometry mismatch for {frozen_name}: max_abs={difference}"
            )

    output = merged[[*TRACK_ID_COLUMNS, *feature_columns]].copy()
    output = merge_calibration_features(output, calibration)
    model_columns = select_numeric_model_columns(output)
    return FeatureTrack(frame=output, model_columns=model_columns)
