"""Fail-closed candidate, payload, and response identity contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import CANDIDATE_COLUMNS, FORBIDDEN_PAYLOAD_TOKENS
from .io import canonical_json, sha256_json


def geometry_sha256(row: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "backend": str(row["backend"]),
            "sample_id": str(row["sample_id"]),
            "candidate_id": str(row["candidate_id"]),
            "center_x": float(row["center_x"]).hex(),
            "center_y": float(row["center_y"]).hex(),
            "angle_deg": float(row["angle_deg"]).hex(),
            "width_px": float(row["width_px"]).hex(),
            "height_px": float(row["height_px"]).hex(),
        }
    )


def candidate_set_sha256(group: pd.DataFrame) -> str:
    ordered = group.sort_values(["original_rank", "candidate_id"], kind="mergesort")
    return sha256_json(
        ordered[["candidate_id", "candidate_geometry_sha256", "original_score"]]
        .to_dict(orient="records")
    )


def validate_candidate_manifest(frame: pd.DataFrame) -> None:
    missing = sorted(set(CANDIDATE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"candidate manifest missing: {missing}")
    if frame.duplicated(["backend", "sample_id", "candidate_id"]).any():
        raise ValueError("duplicate candidate ID within a backend/sample")
    numeric = frame[[
        "original_rank", "original_score", "center_x", "center_y", "angle_deg",
        "width_px", "height_px",
    ]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("candidate manifest contains non-finite values")
    if (frame["width_px"].astype(float) <= 0).any() or (frame["height_px"].astype(float) <= 0).any():
        raise ValueError("candidate rectangle dimensions must be positive")
    for (backend, sample_id), group in frame.groupby(["backend", "sample_id"], sort=False):
        ranks = group["original_rank"].astype(int).sort_values().tolist()
        if not 1 <= len(ranks) <= 5 or ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"{backend}/{sample_id}: invalid frozen Top-5 ranks")
        ordered = group.sort_values(["original_score", "candidate_id"], ascending=[False, True], kind="mergesort")
        by_rank = group.sort_values(["original_rank", "candidate_id"], kind="mergesort")
        if ordered["candidate_id"].astype(str).tolist() != by_rank["candidate_id"].astype(str).tolist():
            raise ValueError(f"{backend}/{sample_id}: score/rank ordering drift")
        if group.apply(geometry_sha256, axis=1).tolist() != group["candidate_geometry_sha256"].astype(str).tolist():
            raise ValueError(f"{backend}/{sample_id}: geometry hash drift")
        expected_set = candidate_set_sha256(group)
        if not group["candidate_set_sha256"].astype(str).eq(expected_set).all():
            raise ValueError(f"{backend}/{sample_id}: candidate set hash drift")


def forbidden_payload_hits(value: Any) -> list[str]:
    serialized = canonical_json(value).lower()
    return sorted(token for token in FORBIDDEN_PAYLOAD_TOKENS if token in serialized)


def assert_no_gt_payload(value: Any) -> None:
    hits = forbidden_payload_hits(value)
    if hits:
        raise ValueError(f"GT/evaluator data in API payload: {hits}")


def validate_display_mapping(
    mapping: Mapping[str, str], candidate_ids: Sequence[str]
) -> None:
    internal = list(map(str, candidate_ids))
    if set(mapping) != set(internal) or len(mapping) != len(internal):
        raise ValueError("display mapping does not cover the frozen set")
    displays = list(map(str, mapping.values()))
    expected = [chr(ord("A") + index) for index in range(len(internal))]
    if sorted(displays) != expected:
        raise ValueError("display IDs must be a permutation of A..E")


def map_display_selection(display_id: str, mapping: Mapping[str, str]) -> str:
    reverse = {str(display): str(internal) for internal, display in mapping.items()}
    if len(reverse) != len(mapping) or str(display_id) not in reverse:
        raise ValueError("response selected an unknown or ambiguous display ID")
    return reverse[str(display_id)]


def assert_reranking_invariants(before: pd.DataFrame, after: pd.DataFrame) -> None:
    try:
        validate_candidate_manifest(before)
        validate_candidate_manifest(after)
    except ValueError as error:
        raise AssertionError("pure re-ranking manifest validation failed") from error
    keys = ["backend", "sample_id", "candidate_id"]
    columns = [
        *keys, "original_score", "center_x", "center_y", "angle_deg",
        "width_px", "height_px", "candidate_geometry_sha256", "candidate_set_sha256",
    ]
    left = before[columns].sort_values(keys).reset_index(drop=True)
    right = after[columns].sort_values(keys).reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("pure re-ranking changed candidate set, score, or geometry")


def scan_secret_text(text: str, secrets: Iterable[str]) -> None:
    for secret in secrets:
        if secret and secret in text:
            raise ValueError("credential leaked into serialized artifact")


def api_eligible(candidate_count: int) -> bool:
    """Only frozen sets with 2..5 candidates may be sent to a provider."""
    return 2 <= int(candidate_count) <= 5
