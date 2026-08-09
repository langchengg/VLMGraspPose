"""Explicit label joins and padded query tensors for unified reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .contracts import assert_model_feature_columns, forbidden_model_columns


KEYS = ("sample_id", "candidate_id")


@dataclass(frozen=True)
class FoldPreprocessor:
    columns: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str]) -> "FoldPreprocessor":
        names = assert_model_feature_columns(columns)
        matrix = frame.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=True)
        matrix[~np.isfinite(matrix)] = np.nan
        with np.errstate(all="ignore"):
            medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(matrix), matrix, medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
        return cls(names, tuple(medians), tuple(means), tuple(scales))

    @classmethod
    def from_artifact(cls, value: dict[str, object]) -> "FoldPreprocessor":
        columns = assert_model_feature_columns(value.get("columns", ()))
        vectors = []
        for name in ("medians", "means", "scales"):
            vector = tuple(float(item) for item in value.get(name, ()))
            if len(vector) != len(columns) or not np.isfinite(vector).all():
                raise ValueError(f"invalid persisted preprocessor {name}")
            vectors.append(vector)
        if any(scale <= 0 for scale in vectors[2]):
            raise ValueError("persisted preprocessor scales must be positive")
        return cls(columns, vectors[0], vectors[1], vectors[2])

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.columns).difference(frame.columns))
        if missing:
            raise ValueError(f"feature schema mismatch: {missing}")
        matrix = frame.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=True)
        matrix[~np.isfinite(matrix)] = np.nan
        matrix = np.where(np.isfinite(matrix), matrix, np.asarray(self.medians))
        matrix = (matrix - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.isfinite(matrix).all():
            raise RuntimeError("fold preprocessing produced non-finite features")
        return matrix.astype(np.float32)

    def artifact(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "medians": list(self.medians),
            "means": list(self.means),
            "scales": list(self.scales),
        }


@dataclass(frozen=True)
class QueryArrays:
    sample_ids: tuple[str, ...]
    candidate_ids: tuple[tuple[str, ...], ...]
    features: torch.Tensor
    labels: torch.Tensor
    jacquard_margins: torch.Tensor
    native_scores: torch.Tensor
    native_ranks: torch.Tensor
    padding_mask: torch.Tensor
    edge_features: torch.Tensor | None = None

    def subset(self, indexes: Sequence[int]) -> "QueryArrays":
        selected = np.asarray(indexes, dtype=int)
        return QueryArrays(
            sample_ids=tuple(self.sample_ids[index] for index in selected),
            candidate_ids=tuple(self.candidate_ids[index] for index in selected),
            features=self.features[selected],
            labels=self.labels[selected],
            jacquard_margins=self.jacquard_margins[selected],
            native_scores=self.native_scores[selected],
            native_ranks=self.native_ranks[selected],
            padding_mask=self.padding_mask[selected],
            edge_features=None if self.edge_features is None else self.edge_features[selected],
        )

    def __len__(self) -> int:
        return len(self.sample_ids)


def join_development_features_and_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Perform the only permitted development-time label join by exact IDs."""

    for name, frame in (("features", features), ("labels", labels)):
        missing = sorted(set(KEYS).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing join columns: {missing}")
        if frame[list(KEYS)].isna().any().any() or frame.duplicated(list(KEYS)).any():
            raise ValueError(f"{name} has invalid or duplicate candidate keys")
    label_columns = (
        *KEYS,
        "candidate_success",
        "jacquard_margin",
    )
    missing = sorted(set(label_columns).difference(labels.columns))
    if missing:
        raise ValueError(f"candidate labels missing columns: {missing}")
    joined = features.merge(
        labels[list(label_columns)],
        on=list(KEYS),
        how="left",
        validate="one_to_one",
    )
    if len(joined) != len(features) or joined[["candidate_success", "jacquard_margin"]].isna().any().any():
        raise ValueError("candidate labels do not exactly cover feature rows")
    if len(joined) != len(labels):
        raise ValueError("feature rows do not exactly cover candidate labels")
    # Some evidence tracks intentionally omit native rank from their persisted
    # feature table because it is candidate identity, not a model input.  The
    # development label table retains that frozen rank.  Restore it solely for
    # query ordering, or verify exact equality when both sides carry it.
    if "native_rank" in labels.columns:
        rank = labels[[*KEYS, "native_rank"]].rename(
            columns={"native_rank": "_label_native_rank"}
        )
        joined = joined.merge(rank, on=list(KEYS), how="left", validate="one_to_one")
        label_rank = pd.to_numeric(joined.pop("_label_native_rank"), errors="coerce")
        if label_rank.isna().any():
            raise ValueError("candidate labels contain invalid native rank")
        if "native_rank" in joined.columns:
            feature_rank = pd.to_numeric(joined["native_rank"], errors="coerce")
            if feature_rank.isna().any() or not feature_rank.equals(label_rank):
                raise ValueError("feature/label native rank mismatch")
        else:
            joined["native_rank"] = label_rank.astype(labels["native_rank"].dtype)
    return joined


def build_query_arrays(
    joined: pd.DataFrame,
    *,
    preprocessor: FoldPreprocessor,
    base_score_column: str = "base_logit",
    max_candidates: int = 5,
) -> QueryArrays:
    required = {
        *KEYS,
        "native_rank",
        "candidate_success",
        "jacquard_margin",
        base_score_column,
    }
    missing = sorted(required.difference(joined.columns))
    if missing:
        raise ValueError(f"joined query table missing columns: {missing}")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    work = joined.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["candidate_id"] = work["candidate_id"].astype(str)
    work = work.sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
    matrix = preprocessor.transform(work)
    work["_matrix_index"] = np.arange(len(work))
    groups = list(work.groupby("sample_id", sort=False))
    if not groups:
        raise ValueError("at least one candidate-bearing query is required")
    feature_count = matrix.shape[1]
    shape = (len(groups), max_candidates)
    features = np.zeros((*shape, feature_count), dtype=np.float32)
    labels = np.zeros(shape, dtype=np.float32)
    margins = np.zeros(shape, dtype=np.float32)
    native_scores = np.zeros(shape, dtype=np.float32)
    native_ranks = np.zeros(shape, dtype=np.int64)
    padding = np.ones(shape, dtype=bool)
    sample_ids: list[str] = []
    candidate_ids: list[tuple[str, ...]] = []
    for query_index, (sample_id, group) in enumerate(groups):
        count = len(group)
        if count > max_candidates:
            raise ValueError(f"query exceeds frozen pool size: {sample_id}/{count}")
        indexes = group["_matrix_index"].to_numpy(int)
        features[query_index, :count] = matrix[indexes]
        labels[query_index, :count] = group["candidate_success"].astype(float)
        margins[query_index, :count] = group["jacquard_margin"].astype(float)
        native_scores[query_index, :count] = pd.to_numeric(group[base_score_column], errors="raise")
        native_ranks[query_index, :count] = group["native_rank"].astype(int)
        padding[query_index, :count] = False
        sample_ids.append(str(sample_id))
        candidate_ids.append(tuple(group["candidate_id"].astype(str)))
    return QueryArrays(
        sample_ids=tuple(sample_ids),
        candidate_ids=tuple(candidate_ids),
        features=torch.from_numpy(features),
        labels=torch.from_numpy(labels),
        jacquard_margins=torch.from_numpy(margins),
        native_scores=torch.from_numpy(native_scores),
        native_ranks=torch.from_numpy(native_ranks),
        padding_mask=torch.from_numpy(padding),
    )


def build_inference_query_arrays(
    features: pd.DataFrame,
    *,
    preprocessor: FoldPreprocessor,
    base_score_column: str = "base_logit",
    max_candidates: int = 5,
) -> QueryArrays:
    """Build label-free query tensors for Validation-locked/Test inference.

    The tensor container also carries training-only label slots, so this
    function supplies zero-valued placeholders internally.  It neither imports
    nor accepts a label table, and all prediction code ignores those slots.
    """

    required = {*KEYS, "native_rank", base_score_column}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"inference feature table missing columns: {missing}")
    # Fail closed if a mixed feature/label table is passed by mistake.
    # Identity/base-ranking fields are required table keys, not model inputs.
    # Apply the denylist to every additional column while permitting those
    # explicit structural fields.
    structural = {*KEYS, "native_rank", base_score_column}
    forbidden = set(forbidden_model_columns(set(features.columns).difference(structural)))
    if forbidden:
        raise ValueError(f"inference feature table contains supervision: {sorted(forbidden)}")
    work = features.copy()
    work["candidate_success"] = 0.0
    work["jacquard_margin"] = 0.0
    return build_query_arrays(
        work,
        preprocessor=preprocessor,
        base_score_column=base_score_column,
        max_candidates=max_candidates,
    )


def with_query_edge_features(
    arrays: QueryArrays,
    relations: pd.DataFrame,
    *,
    preprocessor: FoldPreprocessor,
) -> QueryArrays:
    """Attach exact directed relation tensors in candidate-ID order."""

    required = {"sample_id", "source_candidate_id", "target_candidate_id"}
    missing = sorted(required.difference(relations.columns))
    if missing:
        raise ValueError(f"candidate relations missing columns: {missing}")
    keys = ["sample_id", "source_candidate_id", "target_candidate_id"]
    if relations[keys].isna().any().any() or relations.duplicated(keys).any():
        raise ValueError("candidate relations contain invalid or duplicate directed keys")
    matrix = preprocessor.transform(relations)
    relation_work = relations[keys].astype(str).copy()
    relation_work["_matrix_index"] = np.arange(len(relation_work))
    grouped = {
        sample_id: group for sample_id, group in relation_work.groupby("sample_id", sort=False)
    }
    batch, count = arrays.padding_mask.shape
    edge = np.zeros((batch, count, count, matrix.shape[1]), dtype=np.float32)
    for query_index, (sample_id, candidate_ids) in enumerate(
        zip(arrays.sample_ids, arrays.candidate_ids, strict=True)
    ):
        index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
        group = grouped.get(str(sample_id))
        expected = len(candidate_ids) * max(len(candidate_ids) - 1, 0)
        if group is None:
            if expected:
                raise ValueError(f"relations miss candidate-bearing sample: {sample_id}")
            continue
        observed = 0
        for _, source_id, target_id, matrix_index in group.itertuples(
            index=False, name=None
        ):
            source = index_by_id.get(str(source_id))
            target = index_by_id.get(str(target_id))
            if source is None or target is None or source == target:
                raise ValueError(f"relation identity is outside query or self-directed: {sample_id}")
            edge[query_index, source, target] = matrix[int(matrix_index)]
            observed += 1
        if observed != expected:
            raise ValueError(
                f"relation tensor is incomplete for {sample_id}: {observed} != {expected}"
            )
    return QueryArrays(
        sample_ids=arrays.sample_ids,
        candidate_ids=arrays.candidate_ids,
        features=arrays.features,
        labels=arrays.labels,
        jacquard_margins=arrays.jacquard_margins,
        native_scores=arrays.native_scores,
        native_ranks=arrays.native_ranks,
        padding_mask=arrays.padding_mask,
        edge_features=torch.from_numpy(edge),
    )
