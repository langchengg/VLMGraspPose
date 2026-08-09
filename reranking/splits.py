"""Leakage-audited query-level grouped cross-validation splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class SplitError(ValueError):
    """Raised when a query/group/candidate split invariant is violated."""


def scene_frame_group_hash(scene_id: Any, frame_id: Any) -> str:
    """Create a stable, unambiguous group ID from scene and frame identity."""

    if pd.isna(scene_id) or pd.isna(frame_id):
        raise SplitError("scene_id and frame_id must be non-null")
    payload = json.dumps(
        [str(scene_id), str(frame_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FoldIndices:
    """Candidate-row indices and identities for one train/validation fold."""

    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_query_ids: tuple[str, ...]
    validation_query_ids: tuple[str, ...]
    train_group_hashes: tuple[str, ...]
    validation_group_hashes: tuple[str, ...]
    train_candidate_keys: tuple[str, ...]
    validation_candidate_keys: tuple[str, ...]


@dataclass(frozen=True)
class SplitPlan:
    """Complete query and candidate assignments plus a JSON-safe audit."""

    folds: tuple[FoldIndices, ...]
    query_assignments: pd.DataFrame
    candidate_assignments: pd.DataFrame
    audit: Mapping[str, Any]

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for fold in self.folds:
            yield fold.train_indices.copy(), fold.validation_indices.copy()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SplitError(f"candidate table missing columns: {missing}")


def _normalise_candidate_table(
    candidates: pd.DataFrame,
    *,
    query_col: str,
    candidate_col: str,
    scene_col: str,
    frame_col: str,
    label_col: str,
) -> pd.DataFrame:
    required = (query_col, candidate_col, scene_col, frame_col, label_col)
    _require_columns(candidates, required)
    if len(candidates) == 0:
        raise SplitError("candidate table is empty")
    frame = candidates.loc[:, required].copy().reset_index(drop=True)
    for column in (query_col, candidate_col, scene_col, frame_col):
        if frame[column].isna().any():
            raise SplitError(f"{column} contains null values")
        frame[column] = frame[column].astype(str)
        if bool(frame[column].eq("").any()):
            raise SplitError(f"{column} contains empty values")
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise SplitError(f"{label_col} must be binary (0/1)")
    frame[label_col] = labels.astype(np.int8)
    if frame.duplicated([query_col, candidate_col]).any():
        raise SplitError("duplicate query/candidate identity")
    for column in (scene_col, frame_col):
        counts = frame.groupby(query_col, sort=False)[column].nunique()
        if bool((counts != 1).any()):
            bad = sorted(counts.index[counts != 1].astype(str).tolist())[:5]
            raise SplitError(f"queries span multiple {column} values: {bad}")
    return frame


def _candidate_key(query_id: str, candidate_id: str) -> str:
    payload = json.dumps(
        [str(query_id), str(candidate_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_stratified_group_folds(
    candidates: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 0,
    query_col: str = "query_id",
    candidate_col: str = "candidate_id",
    scene_col: str = "scene_id",
    frame_col: str = "frame_id",
    label_col: str = "label",
) -> SplitPlan:
    """Split one-row-per-query summaries, then expand folds to candidates.

    Stratification uses whether each query has at least one positive candidate.
    Grouping uses the SHA-256 identity of ``(scene_id, frame_id)``.  Candidate
    rows never participate independently in fold allocation.
    """

    if int(n_splits) < 2:
        raise SplitError("n_splits must be at least two")
    frame = _normalise_candidate_table(
        candidates,
        query_col=query_col,
        candidate_col=candidate_col,
        scene_col=scene_col,
        frame_col=frame_col,
        label_col=label_col,
    )
    query_rows: list[dict[str, Any]] = []
    for query_id, group in frame.groupby(query_col, sort=True):
        scene_id = str(group[scene_col].iloc[0])
        frame_id = str(group[frame_col].iloc[0])
        query_rows.append(
            {
                query_col: str(query_id),
                scene_col: scene_id,
                frame_col: frame_id,
                "group_hash": scene_frame_group_hash(scene_id, frame_id),
                "query_has_positive": int(group[label_col].max()),
                "candidate_count": int(len(group)),
            }
        )
    queries = pd.DataFrame(query_rows).sort_values(query_col, kind="mergesort")
    queries = queries.reset_index(drop=True)
    group_count = int(queries["group_hash"].nunique())
    if group_count < n_splits:
        raise SplitError(
            f"{n_splits} folds require at least {n_splits} scene/frame groups; "
            f"received {group_count}"
        )
    if queries["query_has_positive"].nunique() < 2:
        raise SplitError("stratified folds require positive and no-positive queries")

    splitter = StratifiedGroupKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(random_state)
    )
    query_fold = np.full(len(queries), -1, dtype=np.int64)
    raw_splits = list(
        splitter.split(
            np.zeros((len(queries), 1), dtype=np.float64),
            queries["query_has_positive"].to_numpy(dtype=np.int8),
            groups=queries["group_hash"].to_numpy(dtype=object),
        )
    )
    for fold_number, (_, validation_query_indices) in enumerate(raw_splits):
        if np.any(query_fold[validation_query_indices] != -1):
            raise SplitError("a query was assigned to multiple validation folds")
        query_fold[validation_query_indices] = fold_number
    if np.any(query_fold < 0):
        raise SplitError("some queries were not assigned a validation fold")
    queries["fold"] = query_fold

    fold_by_query = queries.set_index(query_col)["fold"].to_dict()
    group_by_query = queries.set_index(query_col)["group_hash"].to_dict()
    candidate_assignments = pd.DataFrame(
        {
            "row_position": np.arange(len(frame), dtype=np.int64),
            query_col: frame[query_col].to_numpy(dtype=object),
            candidate_col: frame[candidate_col].to_numpy(dtype=object),
        }
    )
    candidate_assignments["candidate_key"] = [
        _candidate_key(query_id, candidate_id)
        for query_id, candidate_id in zip(
            candidate_assignments[query_col], candidate_assignments[candidate_col]
        )
    ]
    candidate_assignments["group_hash"] = candidate_assignments[query_col].map(
        group_by_query
    )
    candidate_assignments["fold"] = (
        candidate_assignments[query_col].map(fold_by_query).astype(np.int64)
    )

    folds: list[FoldIndices] = []
    fold_audits: list[dict[str, Any]] = []
    validation_candidate_count = np.zeros(len(frame), dtype=np.int8)
    for fold_number in range(int(n_splits)):
        validation_mask = candidate_assignments["fold"].eq(fold_number).to_numpy()
        train_mask = ~validation_mask
        train_rows = candidate_assignments.loc[train_mask]
        validation_rows = candidate_assignments.loc[validation_mask]
        train_queries = tuple(sorted(train_rows[query_col].unique().tolist()))
        validation_queries = tuple(
            sorted(validation_rows[query_col].unique().tolist())
        )
        train_groups = tuple(sorted(train_rows["group_hash"].unique().tolist()))
        validation_groups = tuple(
            sorted(validation_rows["group_hash"].unique().tolist())
        )
        train_candidates = tuple(sorted(train_rows["candidate_key"].tolist()))
        validation_candidates = tuple(
            sorted(validation_rows["candidate_key"].tolist())
        )
        intersections = {
            "query_ids": sorted(set(train_queries) & set(validation_queries)),
            "group_hashes": sorted(set(train_groups) & set(validation_groups)),
            "candidate_keys": sorted(
                set(train_candidates) & set(validation_candidates)
            ),
        }
        if any(intersections.values()):
            raise SplitError(f"fold {fold_number} leakage: {intersections}")
        train_indices = train_rows["row_position"].to_numpy(dtype=np.int64)
        validation_indices = validation_rows["row_position"].to_numpy(dtype=np.int64)
        validation_candidate_count[validation_indices] += 1
        folds.append(
            FoldIndices(
                fold=fold_number,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_query_ids=train_queries,
                validation_query_ids=validation_queries,
                train_group_hashes=train_groups,
                validation_group_hashes=validation_groups,
                train_candidate_keys=train_candidates,
                validation_candidate_keys=validation_candidates,
            )
        )
        fold_query_rows = queries.loc[queries["fold"].eq(fold_number)]
        fold_audits.append(
            {
                "fold": fold_number,
                "train_query_count": len(train_queries),
                "validation_query_count": len(validation_queries),
                "train_group_count": len(train_groups),
                "validation_group_count": len(validation_groups),
                "train_candidate_count": len(train_candidates),
                "validation_candidate_count": len(validation_candidates),
                "validation_positive_query_count": int(
                    fold_query_rows["query_has_positive"].sum()
                ),
                "validation_no_positive_query_count": int(
                    len(fold_query_rows)
                    - fold_query_rows["query_has_positive"].sum()
                ),
                "intersections": intersections,
                "leakage_free": True,
            }
        )
    if not np.all(validation_candidate_count == 1):
        raise SplitError("every candidate must occur in exactly one validation fold")

    audit: dict[str, Any] = {
        "schema_version": 1,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "split_level": "query",
        "stratification": "query_has_at_least_one_positive_candidate",
        "grouping": "sha256(JSON([scene_id, frame_id]))",
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "query_count": int(len(queries)),
        "group_count": group_count,
        "candidate_count": int(len(frame)),
        "candidate_validation_assignment_min": int(validation_candidate_count.min()),
        "candidate_validation_assignment_max": int(validation_candidate_count.max()),
        "all_folds_leakage_free": True,
        "folds": fold_audits,
    }
    return SplitPlan(
        folds=tuple(folds),
        query_assignments=queries.copy(),
        candidate_assignments=candidate_assignments,
        audit=audit,
    )


# Intent-revealing alias for experiment runners.
build_query_level_folds = build_stratified_group_folds


__all__ = [
    "FoldIndices",
    "SplitError",
    "SplitPlan",
    "build_query_level_folds",
    "build_stratified_group_folds",
    "scene_frame_group_hash",
]
