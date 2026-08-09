"""Fail-closed contracts for the frozen G1/C1 candidate experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BACKENDS = ("G1", "C1")
GEOMETRY_COLUMNS = (
    "center_x",
    "center_y",
    "angle_deg",
    "width_px",
    "height_px",
)
IDENTITY_COLUMNS = (
    "sample_id",
    "scene_id",
    "backend",
    "source_candidate_id",
    "stable_candidate_id",
    "original_rank",
    "original_score",
    *GEOMETRY_COLUMNS,
)
FORBIDDEN_INFERENCE_TOKENS = (
    "candidate_success",
    "candidate_correct",
    "candidate_positive",
    "best_rectangle_iou",
    "best_angle_difference",
    "best_gt",
    "pairwise_json",
    "gt_mask",
    "gt_grasp",
    "ground_truth",
    "target_object",
    "oracle",
    "recoverable",
    "unrecoverable",
    "baseline_correct",
    "evaluator",
    "j_at_1",
    "j@1",
    "first_valid_rank",
)
GEMINI_PAYLOAD_ALLOWLIST = frozenset(
    {
        "sample_id",
        "backend",
        "language",
        "baseline_candidate_id",
        "challenger_candidate_id",
        "coordinate_convention",
        "candidate_evidence",
        "reliability",
        "image_sha256",
        "renderer_version",
        "perturbation_variant",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_identity_sha256(row: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for column in IDENTITY_COLUMNS:
        if column not in row:
            raise ValueError(f"candidate identity missing {column}")
        value = row[column]
        if column in (*GEOMETRY_COLUMNS, "original_score"):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"candidate identity has non-finite {column}")
            payload[column] = number
        elif column in {"original_rank"}:
            payload[column] = int(value)
        else:
            payload[column] = str(value)
    return canonical_sha256(payload)


def forbidden_columns(columns: Iterable[str]) -> list[str]:
    rejected: list[str] = []
    for column in map(str, columns):
        lowered = column.lower()
        if any(token in lowered for token in FORBIDDEN_INFERENCE_TOKENS):
            rejected.append(column)
    return sorted(set(rejected))


def assert_inference_columns(columns: Iterable[str]) -> None:
    rejected = forbidden_columns(columns)
    if rejected:
        raise ValueError(f"GT/evaluator fields are forbidden at inference: {rejected}")


def assert_gemini_payload(payload: Mapping[str, Any]) -> None:
    unknown = sorted(set(map(str, payload)) - GEMINI_PAYLOAD_ALLOWLIST)
    if unknown:
        raise ValueError(f"Gemini payload contains non-whitelisted fields: {unknown}")
    serialized = canonical_json_bytes(payload).decode("utf-8").lower()
    leaked = [token for token in FORBIDDEN_INFERENCE_TOKENS if token in serialized]
    if leaked:
        raise ValueError(f"Gemini payload contains forbidden tokens: {sorted(leaked)}")


def validate_frozen_candidates(
    frame: pd.DataFrame,
    *,
    require_top5: bool = True,
    allow_union: bool = False,
) -> None:
    """Validate identity, geometry, ordering, and inference leakage invariants."""

    required = set(IDENTITY_COLUMNS) | {"candidate_identity_sha256", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frozen candidates missing columns: {missing}")
    assert_inference_columns(frame.columns)
    if frame.duplicated(["sample_id", "stable_candidate_id"]).any():
        raise ValueError("duplicate stable candidate identity within a sample")
    if frame.empty:
        return
    numeric = frame.loc[:, ["original_rank", "original_score", *GEOMETRY_COLUMNS]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("candidate rank/score/geometry must be finite")
    if not np.allclose(frame["height_px"].to_numpy(dtype=float), 20.0, atol=1e-12):
        raise ValueError("frozen candidate height must remain 20 px")
    angles = frame["angle_deg"].to_numpy(dtype=float)
    if np.any(angles < -90.0000001) or np.any(angles >= 90.0000001):
        raise ValueError("candidate angle is outside the frozen 180-degree convention")
    valid_backends = set(BACKENDS) | ({"UNION"} if allow_union else set())
    if not set(frame["backend"].astype(str)).issubset(valid_backends):
        raise ValueError("unknown backend identity")
    hashes = frame.apply(lambda row: candidate_identity_sha256(row), axis=1)
    if hashes.tolist() != frame["candidate_identity_sha256"].astype(str).tolist():
        raise ValueError("candidate identity SHA-256 drift")
    for sample_id, group in frame.groupby("sample_id", sort=False):
        if allow_union:
            # Union rows retain their immutable source-backend rank and score.
            # A separate pool_rank/pool_score defines the union baseline.
            continue
        ranks = sorted(group["original_rank"].astype(int).tolist())
        if ranks != list(range(1, len(group) + 1)):
            raise ValueError(f"{sample_id}: ranks are not a one-based permutation")
        if require_top5 and len(group) > 5:
            raise ValueError(f"{sample_id}: primary pool exceeds frozen Top-5")
        ordered = group.sort_values(
            ["original_score", "stable_candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        )["stable_candidate_id"].astype(str).tolist()
        ranked = group.sort_values(
            ["original_rank", "stable_candidate_id"],
            kind="mergesort",
        )["stable_candidate_id"].astype(str).tolist()
        if ordered != ranked:
            raise ValueError(f"{sample_id}: original score/rank ordering drift")


def identity_table_sha256(frame: pd.DataFrame) -> str:
    columns = [*IDENTITY_COLUMNS, "candidate_identity_sha256"]
    ordered = frame.loc[:, columns].sort_values(
        ["sample_id", "backend", "original_rank", "stable_candidate_id"],
        kind="mergesort",
    )
    return canonical_sha256(ordered.to_dict(orient="records"))


def assert_candidate_identity(before: pd.DataFrame, after: pd.DataFrame) -> None:
    columns = ["sample_id", "stable_candidate_id", *GEOMETRY_COLUMNS, "original_score"]
    left = before.loc[:, columns].sort_values(columns[:2]).reset_index(drop=True)
    right = after.loc[:, columns].sort_values(columns[:2]).reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("re-ranking changed candidate identity or geometry")


def selected_ids_are_frozen(
    selections: pd.DataFrame, candidates: pd.DataFrame, *, column: str
) -> None:
    frozen = set(
        zip(
            candidates["sample_id"].astype(str),
            candidates["stable_candidate_id"].astype(str),
            strict=True,
        )
    )
    observed = set(
        zip(
            selections["sample_id"].astype(str),
            selections[column].astype(str),
            strict=True,
        )
    )
    if not observed.issubset(frozen):
        raise ValueError("selection contains a non-frozen candidate ID")


def exact_candidate_count(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("sample_id", sort=False)["stable_candidate_id"].size()


def no_duplicate_successful_hashes(hashes: Sequence[str]) -> bool:
    values = [str(value) for value in hashes]
    return len(values) == len(set(values))
