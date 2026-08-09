"""Memory-bounded candidate-table iteration and deterministic training retention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LABEL_COLUMNS = (
    "sample_id",
    "candidate_id",
    "candidate_iou",
    "continuous_iou",
    "y70",
    "y80",
    "y90",
)


@dataclass(frozen=True)
class CandidateDatasetPair:
    features: Path
    labels: Path
    split: str
    feature_columns: tuple[str, ...] | None = None
    excluded_source_families: tuple[str, ...] = ()
    excluded_source_prefixes: tuple[str, ...] = ()
    remove_noncanonical_threshold_variants: bool = False


REQUIRED_FEATURE_COLUMNS = {
    "sample_id",
    "candidate_id",
    "split",
    "frame_id",
    "scene_id",
    "source_family",
    "eligible_final",
}


def iter_joined_candidate_samples(
    pairs: list[CandidateDatasetPair],
) -> Iterator[pd.DataFrame]:
    """Yield one feature/label joined sample per Parquet row group."""

    for pair in pairs:
        feature_file = pq.ParquetFile(pair.features)
        label_file = pq.ParquetFile(pair.labels)
        if feature_file.num_row_groups != label_file.num_row_groups:
            raise ValueError(f"feature/label row-group mismatch for {pair.split}")
        for group in range(feature_file.num_row_groups):
            feature_columns = None
            if pair.feature_columns is not None:
                feature_columns = sorted(
                    REQUIRED_FEATURE_COLUMNS | set(pair.feature_columns)
                )
            features = feature_file.read_row_group(
                group, columns=feature_columns
            ).to_pandas()
            labels = label_file.read_row_group(
                group, columns=list(LABEL_COLUMNS)
            ).to_pandas()
            feature_ids = features["sample_id"].astype(str).unique()
            label_ids = labels["sample_id"].astype(str).unique()
            if len(feature_ids) != 1 or len(label_ids) != 1:
                raise ValueError("candidate row group must contain exactly one sample")
            if str(feature_ids[0]) != str(label_ids[0]):
                raise ValueError(
                    f"feature/label sample order drift: {feature_ids[0]} != {label_ids[0]}"
                )
            data = features.merge(
                labels,
                on=["sample_id", "candidate_id"],
                how="inner",
                validate="one_to_one",
            )
            if len(data) != len(features):
                raise ValueError(f"incomplete feature/label join: {feature_ids[0]}")
            if set(data["split"].astype(str)) != {pair.split}:
                raise ValueError(f"candidate split drift in {feature_ids[0]}")
            source = data["source_family"].astype(str)
            excluded = source.isin(pair.excluded_source_families)
            if pair.excluded_source_prefixes:
                excluded |= source.str.startswith(pair.excluded_source_prefixes)
            if pair.remove_noncanonical_threshold_variants:
                threshold = pd.to_numeric(
                    data.get("mask_threshold", pd.Series(np.nan, index=data.index)),
                    errors="coerce",
                )
                noncanonical = (source == "HIFI_THRESHOLD") & ~np.isclose(
                    threshold.fillna(np.inf).to_numpy(float), 0.50
                )
                excluded |= noncanonical
            if excluded.any():
                data.loc[excluded, "eligible_final"] = False
            yield data


def infer_dataset_split(path: Path) -> str:
    """Read the split identity without materializing a candidate table."""

    parquet = pq.ParquetFile(path)
    if parquet.num_row_groups == 0:
        raise ValueError(f"candidate feature table has no row groups: {path}")
    values = parquet.read_row_group(0, columns=["split"])["split"].to_pylist()
    splits = {str(value) for value in values}
    if len(splits) != 1:
        raise ValueError(f"first candidate row group has ambiguous split: {path}")
    return splits.pop()


def iter_joined_candidate_samples_with_oof(
    pairs: list[CandidateDatasetPair],
    oof_path: Path,
) -> Iterator[pd.DataFrame]:
    """Yield feature/label samples joined to aligned OOF prediction row groups."""

    oof_file = pq.ParquetFile(oof_path)
    expected_groups = sum(pq.ParquetFile(pair.features).num_row_groups for pair in pairs)
    if oof_file.num_row_groups != expected_groups:
        raise ValueError(
            f"OOF/sample row-group mismatch: {oof_file.num_row_groups} != "
            f"{expected_groups}"
        )
    prediction_columns = (
        "sample_id",
        "candidate_id",
        "y90",
        "fold",
        "m0_p90_raw",
        "m0_p90_calibrated",
        "m1_p90_raw",
        "m1_p90_calibrated",
        "m2_predicted_iou",
    )
    for group, sample in enumerate(iter_joined_candidate_samples(pairs)):
        eligible = sample[sample["eligible_final"]].copy()
        predictions = oof_file.read_row_group(
            group, columns=list(prediction_columns)
        ).to_pandas()
        sample_ids = predictions["sample_id"].astype(str).unique()
        if len(sample_ids) != 1 or str(sample_ids[0]) != str(sample.iloc[0]["sample_id"]):
            raise ValueError(f"OOF sample order drift at row group {group}")
        evidence = eligible.merge(
            predictions,
            on=["sample_id", "candidate_id"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_oof"),
        )
        if len(evidence) != len(eligible) or len(evidence) != len(predictions):
            raise ValueError(f"incomplete OOF join: {sample_ids[0]}")
        yield evidence


def deterministic_training_subset(
    sample: pd.DataFrame,
    *,
    maximum_candidates: int = 64,
) -> pd.DataFrame:
    """Retain bounded positives, hard negatives, and GT-free contenders.

    Ground-truth IoU is used only to choose training rows and labels.  OOF
    inference still scores every eligible bank candidate.
    """

    data = sample[sample["eligible_final"]].copy().reset_index(drop=True)
    if maximum_candidates < 16:
        raise ValueError("training candidate cap must be at least 16")
    hifi = data[data["source_family"].isin({"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"})]
    if len(hifi) != 1:
        raise ValueError("training sample requires exactly one HiFi fallback")
    selected: set[int] = {int(hifi.index[0])}

    def add(frame: pd.DataFrame, limit: int, columns: list[str]) -> None:
        if frame.empty or limit <= 0:
            return
        available = [column for column in columns if column in frame.columns]
        if not available:
            available = ["candidate_iou"]
        ascending = [False] * len(available) + [True]
        ordered = frame.sort_values(
            [*available, "candidate_id"],
            ascending=ascending,
            kind="stable",
            na_position="last",
        )
        selected.update(int(value) for value in ordered.index[:limit])

    positive = data[data["candidate_iou"] > 0.90]
    for _, family in positive.groupby("source_family", sort=True):
        add(family, 2, ["candidate_iou"])
    add(positive, 24, ["candidate_iou", "source_consensus_count"])

    hard_negative = data[data["candidate_iou"] <= 0.90]
    add(hard_negative, 12, ["candidate_iou"])
    for score in (
        "sam_score",
        "hifi_candidate_iou",
        "hifi_probability_mass_precision",
        "clip_full_query_similarity",
        "relation_max_consistency",
        "depth_reliability",
    ):
        if score in data.columns:
            add(data, 4, [score])
    for _, family in data.groupby("source_family", sort=True):
        add(family, 1, ["hifi_candidate_iou", "sam_score"])

    mandatory = set(selected)
    if len(selected) > maximum_candidates:
        # Preserve H0, source-balanced strict positives, and the strongest hard
        # contenders using a stable priority, then cap deterministically.
        priority = data.loc[sorted(selected)].copy()
        priority["_hifi"] = priority.index == hifi.index[0]
        priority["_p90"] = priority["candidate_iou"] > 0.90
        if "source_consensus_count" not in priority:
            priority["source_consensus_count"] = 0.0
        priority = priority.sort_values(
            ["_hifi", "_p90", "candidate_iou", "source_consensus_count", "candidate_id"],
            ascending=[False, False, False, False, True],
            kind="stable",
        )
        selected = set(int(value) for value in priority.index[:maximum_candidates])
    elif len(selected) < maximum_candidates:
        remaining = data.loc[~data.index.isin(selected)].sort_values(
            ["candidate_iou", "candidate_id"],
            ascending=[False, True],
            kind="stable",
        )
        selected.update(
            int(value)
            for value in remaining.index[: maximum_candidates - len(selected)]
        )
    if int(hifi.index[0]) not in selected or not selected.issubset(mandatory | set(data.index)):
        raise RuntimeError("training retention lost the mandatory fallback")
    return data.loc[sorted(selected)].copy()


__all__ = [
    "CandidateDatasetPair",
    "deterministic_training_subset",
    "infer_dataset_split",
    "iter_joined_candidate_samples",
    "iter_joined_candidate_samples_with_oof",
]
