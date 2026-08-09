"""Independent, ID-joined evaluation for frozen-pool candidate re-ranking.

The evaluator deliberately ignores prediction-side labels.  Candidate
correctness is joined from the post-hoc label table by ``sample_id`` and
``candidate_id``; ranks are validated as permutations of a protocol-defined
frozen candidate pool before any result is computed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.grasping.reranking_v1.artifact_contract import (
    PROTOCOL_BASELINES,
    identity_payload,
    validate_public_methods,
)
from src.grasping.reranking_v1.method_namespace import (
    FULL_NMS_BASELINE,
    GQCNN_TOP5_BASELINE,
)

SCHEMA_VERSION = 1
BASELINE_METHOD = FULL_NMS_BASELINE
KEY_COLUMNS = ("sample_id", "candidate_id")
REFERENCE_COLUMNS = (
    "sample_id",
    "scene_id",
    "candidate_id",
    "candidate_identity_sha256",
    "original_gqcnn_rank",
    "candidate_positive",
)
ALLOWED_PROTOCOLS = ("full_nms", "gqcnn_top5")


def _baseline_method(protocol: str) -> str:
    return (
        GQCNN_TOP5_BASELINE
        if str(protocol) == "gqcnn_top5"
        else BASELINE_METHOD
    )


class EvaluationError(ValueError):
    """Raised when an input violates the frozen-pool evaluation contract."""


@dataclass(frozen=True)
class EvaluationResult:
    per_method_metrics: pd.DataFrame
    per_sample_outcomes: pd.DataFrame
    per_candidate_predictions: pd.DataFrame
    stat_tests: pd.DataFrame
    bootstrap: pd.DataFrame
    grouped_analysis: pd.DataFrame
    runtime_metrics: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]


def _as_bool(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise EvaluationError(f"{name} contains null values")
    normalized = values.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else (
            bool(int(value))
            if isinstance(value, (int, np.integer, float, np.floating))
            and float(value) in {0.0, 1.0}
            else None
        )
    )
    if normalized.isna().any():
        raise EvaluationError(f"{name} must be binary")
    return normalized.astype(bool)


def _validate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "scene_id",
        "candidate_id",
        "original_gqcnn_rank",
        "candidate_positive",
        "q_raw",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EvaluationError(f"per-candidate table missing columns: {missing}")
    result = frame.copy()
    for column in ("sample_id", "scene_id", "candidate_id"):
        result[column] = result[column].astype(str)
        if bool(result[column].eq("").any()):
            raise EvaluationError(f"{column} contains empty values")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise EvaluationError("per-candidate table has duplicate candidate IDs")
    result["candidate_positive"] = _as_bool(
        result["candidate_positive"], "candidate_positive"
    )
    ranks = pd.to_numeric(result["original_gqcnn_rank"], errors="coerce")
    if ranks.isna().any() or bool((ranks < 1).any()):
        raise EvaluationError("original_gqcnn_rank must contain positive integers")
    if not np.allclose(ranks, np.round(ranks)):
        raise EvaluationError("original_gqcnn_rank must contain integers")
    result["original_gqcnn_rank"] = ranks.astype(np.int64)
    q_values = pd.to_numeric(result["q_raw"], errors="coerce")
    if q_values.isna().any() or not np.all(np.isfinite(q_values.to_numpy(float))):
        raise EvaluationError("q_raw must contain one finite value per candidate")
    result["q_raw"] = q_values.astype(np.float64)
    for sample_id, group in result.groupby("sample_id", sort=False):
        expected = np.arange(1, len(group) + 1)
        observed = np.sort(group["original_gqcnn_rank"].to_numpy(dtype=int))
        if not np.array_equal(expected, observed):
            raise EvaluationError(
                f"{sample_id} original ranks are not a 1..N permutation"
            )
        if group["scene_id"].nunique() != 1:
            raise EvaluationError(f"{sample_id} maps to multiple scenes")
        independently_ranked = group.sort_values(
            ["q_raw", "candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        expected_by_id = {
            str(candidate_id): rank
            for rank, candidate_id in enumerate(
                independently_ranked["candidate_id"], start=1
            )
        }
        observed_by_id = dict(
            zip(
                group["candidate_id"].astype(str),
                group["original_gqcnn_rank"].astype(int),
                strict=True,
            )
        )
        if observed_by_id != expected_by_id:
            raise EvaluationError(
                f"{sample_id} original_gqcnn_rank disagrees with independent "
                "q_raw/candidate_id ordering"
            )
    if "candidate_identity_sha256" not in result.columns:
        result["candidate_identity_sha256"] = ""
    else:
        result["candidate_identity_sha256"] = result[
            "candidate_identity_sha256"
        ].astype(str)
        if bool(result["candidate_identity_sha256"].eq("").any()):
            raise EvaluationError("candidate identity hashes contain empty values")
    return result


def _canonical_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "reranker_method": "method",
        "reranker_rank": "rank",
        "reranker_score": "score",
    }
    result = frame.rename(
        columns={
            old: new
            for old, new in aliases.items()
            if old in frame.columns and new not in frame.columns
        }
    ).copy()
    required = {"sample_id", "candidate_id", "method", "rank"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise EvaluationError(f"prediction table missing columns: {missing}")
    if "protocol" not in result.columns:
        result["protocol"] = "full_nms"
    for column in ("sample_id", "candidate_id", "method", "protocol"):
        result[column] = result[column].astype(str)
        if bool(result[column].eq("").any()):
            raise EvaluationError(f"prediction {column} contains empty values")
    try:
        validate_public_methods(
            sorted(result["method"].unique()),
            context="prediction table methods",
        )
    except ValueError as error:
        raise EvaluationError(str(error)) from error
    unknown_protocols = sorted(
        set(result["protocol"].unique()) - set(ALLOWED_PROTOCOLS)
    )
    if unknown_protocols:
        raise EvaluationError(
            f"unknown protocol values {unknown_protocols}; expected "
            f"{list(ALLOWED_PROTOCOLS)}"
        )
    ranks = pd.to_numeric(result["rank"], errors="coerce")
    if ranks.isna().any() or bool((ranks < 1).any()):
        raise EvaluationError("prediction rank must contain positive integers")
    if not np.allclose(ranks, np.round(ranks)):
        raise EvaluationError("prediction rank must contain integers")
    result["rank"] = ranks.astype(np.int64)
    if "score" not in result.columns:
        result["score"] = np.nan
    else:
        original_score = result["score"]
        score = pd.to_numeric(original_score, errors="coerce")
        invalid = original_score.notna() & score.isna()
        if bool(invalid.any()) or bool(np.isinf(score.dropna()).any()):
            raise EvaluationError(
                "prediction score must be finite numeric or uniformly absent"
            )
        result["score"] = score.astype(np.float64)
    if result.duplicated(["protocol", "method", *KEY_COLUMNS]).any():
        raise EvaluationError("prediction table has duplicate candidate IDs")
    return result


def predictions_from_wide(
    frame: pd.DataFrame,
    *,
    rank_columns: Mapping[str, str] | None = None,
    score_columns: Mapping[str, str] | None = None,
    protocol: str = "full_nms",
) -> pd.DataFrame:
    """Convert method-specific rank/score columns to the long prediction schema."""

    if rank_columns is None:
        rank_columns = {
            column[: -len("_rank")]: column
            for column in frame.columns
            if column.endswith("_rank")
            and column not in {"original_gqcnn_rank", "gqcnn_rank"}
        }
    if not rank_columns:
        raise EvaluationError("no method rank columns were supplied or discovered")
    score_columns = dict(score_columns or {})
    required_ids = {"sample_id", "candidate_id"}
    missing = sorted(required_ids - set(frame.columns))
    if missing:
        raise EvaluationError(f"wide prediction table missing IDs: {missing}")
    parts: list[pd.DataFrame] = []
    for method, rank_column in sorted(rank_columns.items()):
        if rank_column not in frame.columns:
            raise EvaluationError(f"rank column not found: {rank_column}")
        columns = ["sample_id", "candidate_id", rank_column]
        identity = "candidate_identity_sha256"
        if identity in frame.columns:
            columns.append(identity)
        score_column = score_columns.get(method)
        if score_column:
            if score_column not in frame.columns:
                raise EvaluationError(f"score column not found: {score_column}")
            columns.append(score_column)
        part = frame.loc[:, columns].rename(columns={rank_column: "rank"})
        part["method"] = str(method)
        part["protocol"] = str(protocol)
        part["score"] = (
            frame[score_column].to_numpy()
            if score_column
            else np.full(len(frame), np.nan)
        )
        parts.append(part)
    return _canonical_prediction_columns(pd.concat(parts, ignore_index=True))


def _protocol_pool(candidates: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if protocol == "gqcnn_top5":
        return candidates.loc[candidates["original_gqcnn_rank"] <= 5].copy()
    if protocol == "full_nms":
        return candidates.copy()
    raise EvaluationError(
        f"unknown protocol {protocol!r}; expected one of {list(ALLOWED_PROTOCOLS)}"
    )


def _validate_and_join_predictions(
    candidates: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    reference = candidates.set_index(list(KEY_COLUMNS), drop=False)
    output: list[pd.DataFrame] = []
    for (protocol, method), group in predictions.groupby(
        ["protocol", "method"], sort=True
    ):
        pool = _protocol_pool(candidates, str(protocol))
        expected = set(map(tuple, pool.loc[:, KEY_COLUMNS].to_numpy()))
        observed = set(map(tuple, group.loc[:, KEY_COLUMNS].to_numpy()))
        if observed != expected:
            missing = sorted(expected - observed)[:5]
            added = sorted(observed - expected)[:5]
            raise EvaluationError(
                f"{protocol}/{method} candidate pool changed: "
                f"missing={missing}, added={added}"
            )
        for sample_id, sample in group.groupby("sample_id", sort=False):
            ranks = np.sort(sample["rank"].to_numpy(dtype=int))
            if not np.array_equal(ranks, np.arange(1, len(sample) + 1)):
                raise EvaluationError(
                    f"{protocol}/{method}/{sample_id} ranks are not a 1..N permutation"
                )
            finite_score = pd.to_numeric(
                sample["score"], errors="coerce"
            ).notna()
            if bool(finite_score.any()):
                if not bool(finite_score.all()):
                    raise EvaluationError(
                        f"{protocol}/{method}/{sample_id} mixes finite and missing scores"
                    )
                independently_ranked = sample.sort_values(
                    ["score", "candidate_id"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                expected_by_id = {
                    str(candidate_id): rank
                    for rank, candidate_id in enumerate(
                        independently_ranked["candidate_id"], start=1
                    )
                }
                observed_by_id = dict(
                    zip(
                        sample["candidate_id"].astype(str),
                        sample["rank"].astype(int),
                        strict=True,
                    )
                )
                if observed_by_id != expected_by_id:
                    raise EvaluationError(
                        f"{protocol}/{method}/{sample_id} rank disagrees with "
                        "independent score/candidate_id ordering"
                    )
        joined = group.copy()
        keys = pd.MultiIndex.from_frame(joined.loc[:, KEY_COLUMNS])
        selected = reference.loc[keys].reset_index(drop=True)
        if "candidate_identity_sha256" in joined.columns:
            predicted_identity = joined["candidate_identity_sha256"].astype(str)
            expected_identity = selected["candidate_identity_sha256"].astype(str)
            if not predicted_identity.reset_index(drop=True).equals(
                expected_identity.reset_index(drop=True)
            ):
                raise EvaluationError(
                    f"{protocol}/{method} candidate identity changed"
                )
        if "candidate_positive" in joined.columns:
            predicted_label = _as_bool(
                joined["candidate_positive"], "prediction candidate_positive"
            ).reset_index(drop=True)
            expected_label = selected["candidate_positive"].reset_index(drop=True)
            if not predicted_label.equals(expected_label):
                raise EvaluationError(
                    f"{protocol}/{method} candidate_positive conflicts with "
                    "the independent label table"
                )
        # Prediction-side copies of reference fields are discarded.  The only
        # source of truth for labels, identities, scenes and original ranks is
        # the independent per-candidate table.
        joined = joined.drop(
            columns=[
                column
                for column in (
                    "candidate_positive",
                    "scene_id",
                    "original_gqcnn_rank",
                    "candidate_identity_sha256",
                )
                if column in joined.columns
            ]
        )
        for column in (
            "scene_id",
            "candidate_identity_sha256",
            "original_gqcnn_rank",
            "candidate_positive",
            "q_raw",
        ):
            joined[column] = selected[column].to_numpy()
        output.append(joined)
    if not output:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "scene_id",
                "candidate_id",
                "candidate_identity_sha256",
                "original_gqcnn_rank",
                "candidate_positive",
                "protocol",
                "method",
                "rank",
                "score",
            ]
        )
    result = pd.concat(output, ignore_index=True)
    return result.sort_values(
        ["protocol", "method", "sample_id", "rank", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _baseline_predictions(
    candidates: pd.DataFrame, protocols: Iterable[str]
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for protocol in sorted(set(map(str, protocols))):
        pool = _protocol_pool(candidates, protocol).copy()
        pool["method"] = _baseline_method(protocol)
        pool["protocol"] = protocol
        pool["score"] = pool["q_raw"].astype(np.float64)
        ordered = pool.sort_values(
            ["sample_id", "score", "candidate_id"],
            ascending=[True, False, True],
            kind="mergesort",
        ).copy()
        ordered["rank"] = ordered.groupby("sample_id", sort=False).cumcount() + 1
        pool = pool.drop(columns=["rank"], errors="ignore").join(ordered["rank"])
        parts.append(pool)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _sample_universe(
    candidates: pd.DataFrame, sample_universe: pd.DataFrame | None
) -> pd.DataFrame:
    observed = candidates[["sample_id", "scene_id"]].drop_duplicates()
    if sample_universe is None:
        return observed.sort_values("sample_id").reset_index(drop=True)
    required = {"sample_id", "scene_id"}
    missing = sorted(required - set(sample_universe.columns))
    if missing:
        raise EvaluationError(f"sample universe missing columns: {missing}")
    universe = sample_universe.copy()
    universe["sample_id"] = universe["sample_id"].astype(str)
    universe["scene_id"] = universe["scene_id"].astype(str)
    if universe.duplicated("sample_id").any():
        raise EvaluationError("sample universe has duplicate sample IDs")
    merged = observed.merge(
        universe, on="sample_id", how="left", suffixes=("_candidate", "_universe")
    )
    if merged["scene_id_universe"].isna().any():
        raise EvaluationError("sample universe omits candidate-bearing samples")
    mismatch = merged["scene_id_candidate"] != merged["scene_id_universe"]
    if bool(mismatch.any()):
        raise EvaluationError("sample universe scene IDs conflict with candidates")
    return universe.sort_values("sample_id").reset_index(drop=True)


def _build_sample_outcomes(
    predictions: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol in sorted(predictions["protocol"].unique()):
        baseline = predictions.loc[
            (predictions["protocol"] == protocol)
            & (predictions["method"] == _baseline_method(str(protocol)))
        ]
        baseline_by_sample = {
            str(sample_id): group.sort_values(
                ["rank", "candidate_id"], kind="mergesort"
            )
            for sample_id, group in baseline.groupby("sample_id", sort=False)
        }
        methods = sorted(
            predictions.loc[
                predictions["protocol"] == protocol, "method"
            ].unique()
        )
        for method in methods:
            method_frame = predictions.loc[
                (predictions["protocol"] == protocol)
                & (predictions["method"] == method)
            ]
            method_by_sample = {
                str(sample_id): group.sort_values(
                    ["rank", "candidate_id"], kind="mergesort"
                )
                for sample_id, group in method_frame.groupby(
                    "sample_id", sort=False
                )
            }
            for sample in universe.itertuples(index=False):
                sample_id = str(sample.sample_id)
                scene_id = str(sample.scene_id)
                baseline_group = baseline_by_sample.get(sample_id)
                group = method_by_sample.get(sample_id)
                if group is None:
                    group = method_frame.iloc[0:0]
                if baseline_group is None:
                    baseline_group = baseline.iloc[0:0]
                nonempty = len(group) > 0
                ordered_positive = (
                    group.loc[group["candidate_positive"], "rank"].astype(int)
                    if nonempty
                    else pd.Series(dtype=int)
                )
                baseline_positive = (
                    baseline_group.loc[
                        baseline_group["candidate_positive"], "rank"
                    ].astype(int)
                    if len(baseline_group)
                    else pd.Series(dtype=int)
                )
                first_rank = (
                    int(ordered_positive.min())
                    if len(ordered_positive)
                    else None
                )
                baseline_first_rank = (
                    int(baseline_positive.min())
                    if len(baseline_positive)
                    else None
                )
                top = group.iloc[0] if nonempty else None
                baseline_top = (
                    baseline_group.iloc[0] if len(baseline_group) else None
                )
                correct = bool(top["candidate_positive"]) if top is not None else False
                baseline_correct = (
                    bool(baseline_top["candidate_positive"])
                    if baseline_top is not None
                    else False
                )
                if correct and not baseline_correct:
                    outcome = "recovered"
                elif baseline_correct and not correct:
                    outcome = "harmful"
                elif correct:
                    outcome = "unchanged_success"
                else:
                    outcome = "unchanged_failure"
                top_id = str(top["candidate_id"]) if top is not None else None
                baseline_top_id = (
                    str(baseline_top["candidate_id"])
                    if baseline_top is not None
                    else None
                )
                top_original_rank = (
                    int(top["original_gqcnn_rank"]) if top is not None else None
                )
                baseline_original_rank = (
                    int(baseline_top["original_gqcnn_rank"])
                    if baseline_top is not None
                    else None
                )
                rows.append(
                    {
                        "protocol": str(protocol),
                        "method": str(method),
                        "sample_id": sample_id,
                        "scene_id": scene_id,
                        "nonempty": nonempty,
                        "candidate_count": int(len(group)),
                        "top1_candidate_id": top_id,
                        "baseline_top1_candidate_id": baseline_top_id,
                        "top1_correct": correct,
                        "baseline_top1_correct": baseline_correct,
                        "recall_at_5": bool(
                            first_rank is not None and first_rank <= 5
                        ),
                        "recall_at_10": bool(
                            first_rank is not None and first_rank <= 10
                        ),
                        "j_at_any": bool(first_rank is not None),
                        "first_positive_rank": first_rank,
                        "reciprocal_rank": (
                            1.0 / first_rank if first_rank is not None else 0.0
                        ),
                        "baseline_first_positive_rank": baseline_first_rank,
                        "switched": top_id != baseline_top_id,
                        "top1_original_rank": top_original_rank,
                        "baseline_top1_original_rank": baseline_original_rank,
                        "top1_original_rank_movement": (
                            top_original_rank - baseline_original_rank
                            if top_original_rank is not None
                            and baseline_original_rank is not None
                            else None
                        ),
                        "positive_rank_improvement": (
                            baseline_first_rank - first_rank
                            if first_rank is not None
                            and baseline_first_rank is not None
                            else None
                        ),
                        "outcome": outcome,
                        "recovered": outcome == "recovered",
                        "harmful": outcome == "harmful",
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["protocol", "method", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)


def _safe_rate(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _metrics_from_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (protocol, method), group in outcomes.groupby(
        ["protocol", "method"], sort=True
    ):
        nonempty = group.loc[group["nonempty"]]
        positive_ranks = pd.to_numeric(
            group["first_positive_rank"], errors="coerce"
        ).dropna()
        recovered = int(group["recovered"].sum())
        harmful = int(group["harmful"].sum())
        switches = int(group["switched"].sum())
        all_count = len(group)
        nonempty_count = len(nonempty)
        outcome_changes = recovered + harmful
        movement = pd.to_numeric(
            nonempty["top1_original_rank_movement"], errors="coerce"
        ).dropna()
        positive_movement = pd.to_numeric(
            group["positive_rank_improvement"], errors="coerce"
        ).dropna()
        row: dict[str, Any] = {
            "protocol": str(protocol),
            "method": str(method),
            "sample_count_all": int(all_count),
            "sample_count_nonempty": int(nonempty_count),
            "valid_empty_count": int(all_count - nonempty_count),
            "j_at_1_count": int(group["top1_correct"].sum()),
            "j_at_1_all": _safe_rate(group["top1_correct"].sum(), all_count),
            "j_at_1_nonempty": _safe_rate(
                nonempty["top1_correct"].sum(), nonempty_count
            ),
            "recall_at_5_count": int(group["recall_at_5"].sum()),
            "recall_at_5_all": _safe_rate(group["recall_at_5"].sum(), all_count),
            "recall_at_5_nonempty": _safe_rate(
                nonempty["recall_at_5"].sum(), nonempty_count
            ),
            "recall_at_10_count": int(group["recall_at_10"].sum()),
            "recall_at_10_all": _safe_rate(
                group["recall_at_10"].sum(), all_count
            ),
            "recall_at_10_nonempty": _safe_rate(
                nonempty["recall_at_10"].sum(), nonempty_count
            ),
            "j_at_any_count": int(group["j_at_any"].sum()),
            "j_at_any_all": _safe_rate(group["j_at_any"].sum(), all_count),
            "j_at_any_nonempty": _safe_rate(
                nonempty["j_at_any"].sum(), nonempty_count
            ),
            "mrr_all": _safe_rate(group["reciprocal_rank"].sum(), all_count),
            "mrr_nonempty": _safe_rate(
                nonempty["reciprocal_rank"].sum(), nonempty_count
            ),
            "first_positive_rank_count": int(len(positive_ranks)),
            "mean_first_positive_rank": (
                float(positive_ranks.mean()) if len(positive_ranks) else math.nan
            ),
            "median_first_positive_rank": (
                float(positive_ranks.median()) if len(positive_ranks) else math.nan
            ),
            # A first-valid rank exists only for oracle-positive, nonempty
            # samples.  Keep explicit denominator-bearing names for reporting.
            "mean_first_valid_rank_nonempty_oracle_positive": (
                float(positive_ranks.mean()) if len(positive_ranks) else math.nan
            ),
            "median_first_valid_rank_nonempty_oracle_positive": (
                float(positive_ranks.median()) if len(positive_ranks) else math.nan
            ),
            "recovered_count": recovered,
            "harmful_count": harmful,
            "net_count": recovered - harmful,
            "recovered_rate_all": _safe_rate(recovered, all_count),
            "harmful_rate_all": _safe_rate(harmful, all_count),
            "net_rate_all": _safe_rate(recovered - harmful, all_count),
            # Outcome-changing precision asks whether a proposed switch changes
            # binary correctness at all.  Beneficial-switch precision asks how
            # often a proposed switch recovers a failure.  The legacy
            # outcome_precision is retained as precision conditional on a
            # correctness-changing switch.
            "outcome_changing_precision": _safe_rate(
                outcome_changes, switches
            ),
            "beneficial_switch_precision": _safe_rate(recovered, switches),
            "outcome_precision": _safe_rate(recovered, outcome_changes),
            "outcome_change_count": outcome_changes,
            "switch_count": switches,
            "coverage_all": _safe_rate(switches, all_count),
            "coverage_nonempty": _safe_rate(
                nonempty["switched"].sum(), nonempty_count
            ),
            "mean_top1_original_rank_movement": (
                float(movement.mean()) if len(movement) else math.nan
            ),
            "median_top1_original_rank_movement": (
                float(movement.median()) if len(movement) else math.nan
            ),
            "mean_positive_rank_improvement": (
                float(positive_movement.mean())
                if len(positive_movement)
                else math.nan
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["protocol", "method"], kind="mergesort"
    ).reset_index(drop=True)


def _fixed_bin(
    values: pd.Series,
    *,
    bins: Sequence[float],
    labels: Sequence[str],
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=list(bins),
        labels=list(labels),
        include_lowest=True,
        right=True,
    ).astype("string")


def _attach_offline_groups(
    outcomes: pd.DataFrame,
    candidates: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Attach descriptive groups used only after predictions are frozen."""

    metadata = universe.copy()
    counts = (
        candidates.groupby("sample_id", sort=False)
        .size()
        .rename("candidate_count_full_nms")
        .reset_index()
    )
    metadata = metadata.merge(
        counts, on="sample_id", how="left", validate="one_to_one"
    )
    metadata["candidate_count_full_nms"] = (
        metadata["candidate_count_full_nms"].fillna(0).astype(int)
    )
    metadata["candidate_count_group"] = _fixed_bin(
        metadata["candidate_count_full_nms"],
        bins=[-math.inf, 0, 5, 10, 25, 50, math.inf],
        labels=["0", "1-5", "6-10", "11-25", "26-50", "51+"],
    )
    # Candidate count is a transparent, inference-available clutter proxy; it
    # is not presented as a ground-truth scene-clutter measurement.
    metadata["scene_clutter_proxy_group"] = metadata[
        "candidate_count_group"
    ]

    top = candidates.loc[candidates["original_gqcnn_rank"] == 1].copy()
    q_sorted = candidates.sort_values(
        ["sample_id", "q_raw", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    q_sorted["_q_position"] = q_sorted.groupby("sample_id").cumcount() + 1
    q_top = q_sorted.loc[
        q_sorted["_q_position"] <= 2,
        ["sample_id", "_q_position", "q_raw"],
    ].pivot(index="sample_id", columns="_q_position", values="q_raw")
    q_margin = (
        q_top.get(1, pd.Series(index=q_top.index, dtype=float))
        - q_top.get(2, pd.Series(index=q_top.index, dtype=float))
    ).rename("q_top1_top2_margin")
    metadata = metadata.merge(
        q_margin.reset_index(), on="sample_id", how="left", validate="one_to_one"
    )
    metadata["q_margin_group"] = _fixed_bin(
        metadata["q_top1_top2_margin"],
        bins=[-math.inf, 0.01, 0.05, 0.10, math.inf],
        labels=["<=0.01", "(0.01,0.05]", "(0.05,0.10]", ">0.10"],
    )

    for column in ("query_type", "p_axis_mean", "width_m"):
        if column not in candidates.columns:
            continue
        destination = {
            "query_type": "query_type",
            "p_axis_mean": "original_top1_p_axis_mean",
            "width_m": "original_top1_width_m",
        }[column]
        if destination in metadata.columns:
            continue
        if column == "query_type":
            consistency = candidates.groupby("sample_id")[column].nunique(
                dropna=False
            )
            if bool((consistency > 1).any()):
                raise EvaluationError("query_type varies within a sample")
        metadata = metadata.merge(
            top[["sample_id", column]].rename(
                columns={column: destination}
            ),
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    if "original_top1_p_axis_mean" in metadata.columns:
        metadata["hifi_mask_support_group"] = _fixed_bin(
            metadata["original_top1_p_axis_mean"],
            bins=[-math.inf, 0.25, 0.50, 0.75, math.inf],
            labels=["<=0.25", "(0.25,0.50]", "(0.50,0.75]", ">0.75"],
        )
    if "original_top1_width_m" in metadata.columns:
        metadata["width_group"] = _fixed_bin(
            metadata["original_top1_width_m"],
            bins=[-math.inf, 0.03, 0.06, 0.09, math.inf],
            labels=["<=0.03m", "(0.03,0.06]m", "(0.06,0.09]m", ">0.09m"],
        )
    if (
        "mask_area_px" in metadata.columns
        and "predicted_mask_area_fraction" not in metadata.columns
    ):
        # OCID-VLG native evaluation resolution is 640x480.
        metadata["predicted_mask_area_fraction"] = pd.to_numeric(
            metadata["mask_area_px"], errors="coerce"
        ) / float(640 * 480)
    for source, destination in (
        ("target_area_fraction", "target_area_group"),
        ("predicted_mask_area_fraction", "predicted_target_area_group"),
        ("hifi_mask_iou", "hifi_mask_iou_offline_group"),
    ):
        if source in metadata.columns:
            metadata[destination] = _fixed_bin(
                metadata[source],
                bins=[-math.inf, 0.01, 0.05, 0.15, math.inf],
                labels=["<=0.01", "(0.01,0.05]", "(0.05,0.15]", ">0.15"],
            )
    metadata_columns = [
        column
        for column in metadata.columns
        if column == "sample_id"
        or (
            column != "scene_id"
            and column not in outcomes.columns
        )
    ]
    result = outcomes.merge(
        metadata.loc[:, metadata_columns],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    first_rank = pd.to_numeric(
        result["baseline_first_positive_rank"], errors="coerce"
    )
    result["original_first_valid_rank_group"] = np.select(
        [
            first_rank.eq(1),
            first_rank.between(2, 5),
            first_rank.between(6, 10),
            first_rank.gt(10),
        ],
        ["1", "2-5", "6-10", ">10"],
        default="no_positive",
    )
    return result


def grouped_analysis(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Compute post-hoc descriptive results for available requested groups."""

    dimensions = [
        column
        for column in (
            "query_type",
            "candidate_count_group",
            "q_margin_group",
            "hifi_mask_support_group",
            "target_area_group",
            "predicted_target_area_group",
            "width_group",
            "scene_clutter_group",
            "scene_clutter_proxy_group",
            "original_first_valid_rank_group",
            "failure_category",
            "ranking_failure_category",
        )
        if column in outcomes.columns
    ]
    rows: list[dict[str, Any]] = []
    for (protocol, method), method_rows in outcomes.groupby(
        ["protocol", "method"], sort=True
    ):
        for dimension in dimensions:
            selected = method_rows.loc[method_rows[dimension].notna()].copy()
            selected[dimension] = selected[dimension].astype(str)
            for group_value, group in selected.groupby(dimension, sort=True):
                nonempty = group.loc[group["nonempty"]]
                recovered = int(group["recovered"].sum())
                harmful = int(group["harmful"].sum())
                rows.append(
                    {
                        "protocol": str(protocol),
                        "method": str(method),
                        "group_dimension": dimension,
                        "group_value": str(group_value),
                        "sample_count_all": int(len(group)),
                        "sample_count_nonempty": int(len(nonempty)),
                        "j_at_1_all": _safe_rate(
                            group["top1_correct"].sum(), len(group)
                        ),
                        "j_at_1_nonempty": _safe_rate(
                            nonempty["top1_correct"].sum(), len(nonempty)
                        ),
                        "baseline_j_at_1_all": _safe_rate(
                            group["baseline_top1_correct"].sum(), len(group)
                        ),
                        "recovered_count": recovered,
                        "harmful_count": harmful,
                        "net_count": recovered - harmful,
                        "offline_only": True,
                    }
                )
    columns = [
        "protocol",
        "method",
        "group_dimension",
        "group_value",
        "sample_count_all",
        "sample_count_nonempty",
        "j_at_1_all",
        "j_at_1_nonempty",
        "baseline_j_at_1_all",
        "recovered_count",
        "harmful_count",
        "net_count",
        "offline_only",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["protocol", "method", "group_dimension", "group_value"],
        kind="mergesort",
    ).reset_index(drop=True)


def exact_mcnemar(
    baseline_success: Sequence[bool], method_success: Sequence[bool]
) -> dict[str, Any]:
    """Return the paired, two-sided exact McNemar binomial test."""

    baseline = np.asarray(baseline_success, dtype=bool)
    method = np.asarray(method_success, dtype=bool)
    if baseline.shape != method.shape or baseline.ndim != 1:
        raise EvaluationError("McNemar inputs must be paired one-dimensional arrays")
    recovered = int(np.sum(~baseline & method))
    harmful = int(np.sum(baseline & ~method))
    discordant = recovered + harmful
    p_value = (
        float(
            binomtest(
                min(recovered, harmful),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "recovered": recovered,
        "harmful": harmful,
        "discordant": discordant,
        "p_value_exact_two_sided": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise-error adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise EvaluationError("Holm adjustment requires finite p-values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise EvaluationError("p-values must be in [0, 1]")
    count = len(values)
    if count == 0:
        return []
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        np.asarray(
            [(count - index) * value for index, value in enumerate(sorted_values)]
        )
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def _stat_tests(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol, protocol_rows in outcomes.groupby("protocol", sort=True):
        baseline_method = _baseline_method(str(protocol))
        baseline = protocol_rows.loc[
            protocol_rows["method"] == baseline_method,
            ["sample_id", "top1_correct"],
        ].rename(columns={"top1_correct": "baseline_success"})
        protocol_tests: list[dict[str, Any]] = []
        for method, method_rows in protocol_rows.groupby("method", sort=True):
            if method == baseline_method:
                continue
            paired = method_rows[["sample_id", "top1_correct"]].merge(
                baseline, on="sample_id", how="inner", validate="one_to_one"
            )
            if len(paired) != len(baseline):
                raise EvaluationError(
                    f"{protocol}/{method} cannot be paired to every baseline sample"
                )
            test = exact_mcnemar(
                paired["baseline_success"], paired["top1_correct"]
            )
            protocol_tests.append(
                {
                    "protocol": str(protocol),
                    "method": str(method),
                    "baseline_method": baseline_method,
                    "metric": "j_at_1_all",
                    "sample_count": int(len(paired)),
                    "test": "paired_exact_mcnemar_two_sided",
                    **test,
                }
            )
        adjusted = holm_adjust(
            [row["p_value_exact_two_sided"] for row in protocol_tests]
        )
        for row, value in zip(protocol_tests, adjusted, strict=True):
            row["p_value_holm"] = value
            row["holm_family"] = f"{protocol}/j_at_1_vs_{baseline_method}"
            row["holm_hypothesis_count"] = len(protocol_tests)
        rows.extend(protocol_tests)
    columns = [
        "protocol",
        "method",
        "baseline_method",
        "metric",
        "sample_count",
        "recovered",
        "harmful",
        "discordant",
        "test",
        "p_value_exact_two_sided",
        "p_value_holm",
        "holm_family",
        "holm_hypothesis_count",
    ]
    return pd.DataFrame(rows, columns=columns)


def _stable_group_seed(seed: int, protocol: str, method: str) -> int:
    digest = hashlib.sha256(
        f"{int(seed)}\0{protocol}\0{method}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def scene_grouped_bootstrap(
    outcomes: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Percentile CIs from both sequence-scenes and RGB-D frames.

    OCID-VLG ``scene_id`` values have the form ``sequence_path,image.png``.
    Expressions on one image therefore share a frame, while multiple frames
    can share a sequence-scene.  Both requested clustering units are emitted;
    all query rows in a sampled cluster receive the same multiplicity. Delta
    rows remain paired within each sample before resampling.
    """

    if replicates < 0:
        raise EvaluationError("bootstrap replicates must be non-negative")
    columns = [
        "protocol",
        "method",
        "baseline_method",
        "metric",
        "point_estimate",
        "ci_95_lower",
        "ci_95_upper",
        "bootstrap_standard_error",
        "replicates",
        "seed",
        "cluster_key",
        "cluster_count",
        "method_description",
    ]
    if replicates == 0:
        return pd.DataFrame(columns=columns)
    required = {
        "protocol",
        "method",
        "sample_id",
        "scene_id",
        "nonempty",
        "top1_correct",
        "baseline_top1_correct",
        "recall_at_5",
        "recall_at_10",
        "j_at_any",
        "reciprocal_rank",
        "first_positive_rank",
        "recovered",
        "harmful",
        "switched",
    }
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise EvaluationError(f"bootstrap outcomes missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    metric_specs = {
        "j_at_1_all": ("top1_correct", None),
        "j_at_1_nonempty": ("top1_correct", "nonempty"),
        "recall_at_5_all": ("recall_at_5", None),
        "recall_at_5_nonempty": ("recall_at_5", "nonempty"),
        "recall_at_10_all": ("recall_at_10", None),
        "recall_at_10_nonempty": ("recall_at_10", "nonempty"),
        "j_at_any_all": ("j_at_any", None),
        "j_at_any_nonempty": ("j_at_any", "nonempty"),
        "mrr_all": ("reciprocal_rank", None),
        "mrr_nonempty": ("reciprocal_rank", "nonempty"),
        "coverage_all": ("switched", None),
        "coverage_nonempty": ("switched", "nonempty"),
        "recovered_rate_all": ("recovered", None),
        "recovered_rate_nonempty": ("recovered", "nonempty"),
        "harmful_rate_all": ("harmful", None),
        "harmful_rate_nonempty": ("harmful", "nonempty"),
        "mean_first_valid_rank_nonempty_oracle_positive": (
            "first_positive_rank",
            "j_at_any",
        ),
    }
    for (protocol, method), group in outcomes.groupby(
        ["protocol", "method"], sort=True
    ):
        frame_ids = group["scene_id"].astype(str)
        sequence_ids = frame_ids.map(lambda value: value.split(",", 1)[0])
        cluster_specs = (
            (
                "scene_sequence_id",
                sequence_ids,
                "scene_sequence_cluster_percentile_bootstrap",
            ),
            ("frame_id", frame_ids, "frame_cluster_percentile_bootstrap"),
        )
        specs = dict(metric_specs)
        specs["j_at_1_delta_all"] = ("__delta__", None)
        specs["j_at_1_delta_nonempty"] = ("__delta__", "nonempty")
        specs["net_rate_all"] = ("__net__", None)
        specs["net_rate_nonempty"] = ("__net__", "nonempty")
        for cluster_key, group_cluster_ids, description in cluster_specs:
            clusters = np.asarray(sorted(group_cluster_ids.unique()))
            if len(clusters) == 0:
                raise EvaluationError(
                    f"{cluster_key} bootstrap requires at least one cluster"
                )
            cluster_index = {
                cluster: index for index, cluster in enumerate(clusters)
            }
            # The same method seed intentionally produces identical intervals
            # when the two cluster partitions are genuinely identical.
            rng = np.random.default_rng(
                _stable_group_seed(seed, str(protocol), str(method))
            )
            cluster_by_row = pd.Series(
                group_cluster_ids.to_numpy(), index=group.index, dtype="string"
            )
            for metric, (value_column, filter_column) in specs.items():
                selected = (
                    group.loc[group[filter_column].astype(bool)]
                    if filter_column is not None
                    else group
                )
                if value_column == "__delta__":
                    values = (
                        selected["top1_correct"].astype(float)
                        - selected["baseline_top1_correct"].astype(float)
                    )
                elif value_column == "__net__":
                    values = (
                        selected["recovered"].astype(float)
                        - selected["harmful"].astype(float)
                    )
                else:
                    values = pd.to_numeric(
                        selected[value_column], errors="coerce"
                    )
                finite = np.isfinite(values.to_numpy(dtype=float))
                selected = selected.loc[finite]
                values = values.loc[finite].to_numpy(dtype=np.float64)
                sums = np.zeros(len(clusters), dtype=np.float64)
                counts = np.zeros(len(clusters), dtype=np.float64)
                selected_clusters = cluster_by_row.loc[selected.index]
                for cluster, value in zip(
                    selected_clusters.astype(str), values, strict=True
                ):
                    index = cluster_index[cluster]
                    sums[index] += value
                    counts[index] += 1.0
                denominator = counts.sum()
                point = (
                    float(sums.sum() / denominator)
                    if denominator
                    else math.nan
                )
                distribution = np.empty(replicates, dtype=np.float64)
                batch_size = min(1000, replicates)
                cursor = 0
                while cursor < replicates:
                    batch = min(batch_size, replicates - cursor)
                    draws = rng.integers(
                        0,
                        len(clusters),
                        size=(batch, len(clusters)),
                        endpoint=False,
                    )
                    replicate_sums = sums[draws].sum(axis=1)
                    replicate_counts = counts[draws].sum(axis=1)
                    distribution[cursor : cursor + batch] = np.divide(
                        replicate_sums,
                        replicate_counts,
                        out=np.full(batch, np.nan),
                        where=replicate_counts > 0,
                    )
                    cursor += batch
                finite_distribution = distribution[
                    np.isfinite(distribution)
                ]
                if len(finite_distribution):
                    lower, upper = np.quantile(
                        finite_distribution, [0.025, 0.975]
                    )
                    standard_error = float(
                        np.std(finite_distribution, ddof=1)
                        if len(finite_distribution) > 1
                        else 0.0
                    )
                else:
                    lower = upper = standard_error = math.nan
                rows.append(
                    {
                        "protocol": str(protocol),
                        "method": str(method),
                        "baseline_method": _baseline_method(str(protocol)),
                        "metric": metric,
                        "point_estimate": point,
                        "ci_95_lower": float(lower),
                        "ci_95_upper": float(upper),
                        "bootstrap_standard_error": standard_error,
                        "replicates": int(replicates),
                        "seed": int(seed),
                        "cluster_key": cluster_key,
                        "cluster_count": int(len(clusters)),
                        "method_description": description,
                    }
                )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["protocol", "method", "cluster_key", "metric"], kind="mergesort"
    ).reset_index(drop=True)


def _normalize_runtime_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def vlm_runtime_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeat_rows: Sequence[Mapping[str, Any]] | None = None,
    wall_time_seconds: float | None = None,
    memory_peak_mib: float | None = None,
) -> dict[str, Any]:
    """Aggregate auditable local-VLM latency, fallback and token counters."""

    values = [dict(row) for row in rows]
    sample_count = len(values)
    call_rows = [
        row
        for row in values
        if bool(row.get("eligible_for_vlm", True))
        and str(row.get("skip_reason") or "")
        != "valid_empty_no_vlm_call"
        and str(row.get("fallback_reason") or "")
        != "valid_empty_no_vlm_call"
    ]
    latency = np.asarray(
        [
            number
            for row in call_rows
            if (number := _normalize_runtime_value(row.get("latency_seconds")))
            is not None
        ],
        dtype=np.float64,
    )
    durations = [
        number
        for row in call_rows
        if (number := _normalize_runtime_value(row.get("total_duration_ns")))
        is not None
    ]
    prompts = [
        int(number)
        for row in call_rows
        if (number := _normalize_runtime_value(row.get("prompt_eval_count")))
        is not None
    ]
    outputs = [
        int(number)
        for row in call_rows
        if (number := _normalize_runtime_value(row.get("eval_count"))) is not None
    ]
    fallback_count = sum(bool(row.get("fallback", False)) for row in call_rows)
    abstain_count = sum(bool(row.get("abstain", False)) for row in call_rows)
    cache_hit_count = sum(bool(row.get("cache_hit", False)) for row in call_rows)
    switch_count = sum(
        bool(row.get("switch_from_original_top1", False)) for row in call_rows
    )
    parser_error_count = sum(
        row.get("parser_error") is not None
        and str(row.get("parser_error")) != ""
        for row in call_rows
    )
    called_count = len(call_rows)
    valid_json_count = sum(
        (
            row.get("parser_error") is None
            or str(row.get("parser_error")) == ""
        )
        and isinstance(row.get("parsed_model_response"), Mapping)
        for row in call_rows
    )
    invalid_candidate_count = sum(
        "invalid_candidate" in str(row.get("fallback_reason") or "").lower()
        or "invalid_candidate" in str(row.get("parser_error") or "").lower()
        for row in call_rows
    )
    timeout_count = sum(
        "timeout" in str(row.get("fallback_reason") or "").lower()
        or "timeout" in str(row.get("parser_error") or "").lower()
        for row in call_rows
    )
    fallback_reasons: dict[str, int] = {}
    for row in values:
        if not bool(row.get("fallback", False)):
            continue
        reason = str(row.get("fallback_reason") or "unspecified")
        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    latency_total = float(latency.sum()) if len(latency) else 0.0
    effective_wall_time = (
        float(wall_time_seconds)
        if wall_time_seconds is not None
        else latency_total
    )
    if not math.isfinite(effective_wall_time) or effective_wall_time < 0.0:
        raise EvaluationError("VLM wall time must be a finite non-negative value")
    if memory_peak_mib is not None and (
        not math.isfinite(float(memory_peak_mib)) or float(memory_peak_mib) < 0.0
    ):
        raise EvaluationError("VLM memory peak must be a finite non-negative value")

    repeat_by_id = {
        str(row.get("sample_id")): dict(row) for row in (repeat_rows or [])
    }
    primary_by_id = {str(row.get("sample_id")): row for row in values}
    paired_ids = sorted(set(primary_by_id) & set(repeat_by_id))

    def response_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
        ranking = row.get("ranking", [])
        ranking_ids = tuple(
            str(item.get("candidate_id"))
            for item in ranking
            if isinstance(item, Mapping)
        )
        return (
            row.get("selected_candidate_id"),
            ranking_ids,
            bool(row.get("abstain", False)),
            bool(row.get("fallback", False)),
            row.get("fallback_reason"),
        )

    deterministic_agreement_count = sum(
        response_signature(primary_by_id[sample_id])
        == response_signature(repeat_by_id[sample_id])
        for sample_id in paired_ids
    )
    return {
        "sample_count": sample_count,
        "called_sample_count": int(called_count),
        "valid_empty_skipped_count": int(sample_count - called_count),
        "latency_observation_count": int(len(latency)),
        "latency_total_seconds": latency_total,
        "latency_mean_seconds": float(latency.mean()) if len(latency) else math.nan,
        "latency_median_seconds": (
            float(np.median(latency)) if len(latency) else math.nan
        ),
        "latency_p50_seconds": (
            float(np.quantile(latency, 0.50)) if len(latency) else math.nan
        ),
        "latency_p95_seconds": (
            float(np.quantile(latency, 0.95)) if len(latency) else math.nan
        ),
        "fallback_count": int(fallback_count),
        "fallback_rate": _safe_rate(fallback_count, called_count),
        "abstain_count": int(abstain_count),
        "abstain_rate": _safe_rate(abstain_count, called_count),
        "cache_hit_count": int(cache_hit_count),
        "cache_hit_rate": _safe_rate(cache_hit_count, called_count),
        "switch_count": int(switch_count),
        "switch_rate": _safe_rate(switch_count, called_count),
        "parser_error_count": int(parser_error_count),
        "parser_error_rate": _safe_rate(parser_error_count, called_count),
        "valid_json_count": int(valid_json_count),
        "valid_json_rate": _safe_rate(valid_json_count, called_count),
        "invalid_candidate_count": int(invalid_candidate_count),
        "invalid_candidate_rate": _safe_rate(
            invalid_candidate_count, called_count
        ),
        "timeout_count": int(timeout_count),
        "timeout_rate": _safe_rate(timeout_count, called_count),
        "deterministic_repeat_sample_count": int(len(paired_ids)),
        "deterministic_agreement_count": int(
            deterministic_agreement_count
        ),
        "deterministic_agreement_rate": _safe_rate(
            deterministic_agreement_count, len(paired_ids)
        ),
        "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
        "prompt_tokens_total": int(sum(prompts)),
        "output_tokens_total": int(sum(outputs)),
        "prompt_tokens_mean_per_observation": _safe_rate(sum(prompts), len(prompts)),
        "output_tokens_mean_per_observation": _safe_rate(sum(outputs), len(outputs)),
        "output_tokens_per_latency_second": (
            float(sum(outputs) / latency_total) if latency_total > 0.0 else math.nan
        ),
        "ollama_total_duration_seconds": float(sum(durations) / 1e9),
        "total_wall_time_seconds": effective_wall_time,
        "wall_time_source": (
            "provided_summary"
            if wall_time_seconds is not None
            else "sum_latency_fallback"
        ),
        "memory_peak_mib": (
            float(memory_peak_mib) if memory_peak_mib is not None else math.nan
        ),
        "samples_per_hour": (
            float(called_count * 3600.0 / effective_wall_time)
            if effective_wall_time > 0.0
            else math.nan
        ),
    }


def vlm_results_to_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    protocol: str = "gqcnn_top5",
) -> pd.DataFrame:
    """Convert audited VLM JSONL rows to long candidate ranks."""

    output: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for value in rows:
        sample_id = str(value.get("sample_id", ""))
        if not sample_id or sample_id in seen_samples:
            raise EvaluationError("VLM results require unique non-empty sample IDs")
        seen_samples.add(sample_id)
        ranking = value.get("ranking", [])
        if not isinstance(ranking, list):
            raise EvaluationError(f"{sample_id} VLM ranking must be a list")
        ids: list[str] = []
        ranking_size = len(ranking)
        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, Mapping):
                raise EvaluationError(f"{sample_id} VLM ranking item must be an object")
            candidate_id = str(item.get("candidate_id", ""))
            if not candidate_id or candidate_id in ids:
                raise EvaluationError(f"{sample_id} VLM candidate IDs are invalid")
            ids.append(candidate_id)
            output.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "method": str(method),
                    "protocol": str(protocol),
                    "rank": rank,
                    # The list order is the VLM's ranking contract.  Model
                    # scores are allowed to tie, so use a strict rank-derived
                    # evaluator score and preserve the raw value separately.
                    "score": float(ranking_size - rank + 1),
                    "vlm_model_score": _normalize_runtime_value(
                        item.get("score")
                    ),
                }
            )
        selected = value.get("selected_candidate_id")
        if ids and str(selected) != ids[0]:
            raise EvaluationError(
                f"{sample_id} selected candidate is not rank 1 in VLM output"
            )
        if not ids and selected is not None:
            raise EvaluationError(f"{sample_id} has a selection but no ranking")
    return _canonical_prediction_columns(pd.DataFrame(output)) if output else pd.DataFrame(
        columns=["sample_id", "candidate_id", "method", "protocol", "rank", "score"]
    )


def evaluate_predictions(
    per_candidate: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    *,
    sample_universe: pd.DataFrame | None = None,
    bootstrap_replicates: int = 10_000,
    seed: int = 42,
    runtime_metrics: Mapping[str, Mapping[str, Any]] | None = None,
) -> EvaluationResult:
    """Evaluate one or more methods against independently stored labels."""

    candidates = _validate_candidates(per_candidate)
    universe = _sample_universe(candidates, sample_universe)
    if predictions is None or len(predictions) == 0:
        normalized = pd.DataFrame()
        protocols = {"full_nms"}
    else:
        normalized = _canonical_prediction_columns(predictions)
        protocols = set(normalized["protocol"].astype(str))
        supplied_baseline_mask = normalized.apply(
            lambda row: str(row["method"])
            == _baseline_method(str(row["protocol"])),
            axis=1,
        )
        if bool(supplied_baseline_mask.any()):
            supplied_baseline = normalized.loc[
                supplied_baseline_mask
            ]
            validated_baseline = _validate_and_join_predictions(
                candidates, supplied_baseline
            )
            mismatch = (
                validated_baseline["rank"].to_numpy(dtype=int)
                != validated_baseline["original_gqcnn_rank"].to_numpy(dtype=int)
            )
            if bool(np.any(mismatch)):
                raise EvaluationError(
                    "supplied protocol baseline ranks do not equal "
                    "original_gqcnn_rank"
                )
            normalized = normalized.loc[~supplied_baseline_mask].copy()
    baseline = _baseline_predictions(candidates, protocols)
    joined_parts = [_validate_and_join_predictions(candidates, baseline)]
    if len(normalized):
        joined_parts.append(_validate_and_join_predictions(candidates, normalized))
    joined = pd.concat(joined_parts, ignore_index=True).sort_values(
        ["protocol", "method", "sample_id", "rank", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    for protocol in sorted(protocols):
        expected_baseline = PROTOCOL_BASELINES[protocol]
        observed = set(
            joined.loc[
                joined["protocol"].astype(str).eq(protocol), "method"
            ].astype(str)
        )
        if expected_baseline not in observed:
            raise EvaluationError(
                f"{protocol} is missing public baseline {expected_baseline}"
            )
    outcomes = _build_sample_outcomes(joined, universe)
    outcomes = _attach_offline_groups(outcomes, candidates, universe)
    metrics = _metrics_from_outcomes(outcomes)
    tests = _stat_tests(outcomes)
    bootstrap = scene_grouped_bootstrap(
        outcomes, replicates=bootstrap_replicates, seed=seed
    )
    return EvaluationResult(
        per_method_metrics=metrics,
        per_sample_outcomes=outcomes,
        per_candidate_predictions=joined,
        stat_tests=tests,
        bootstrap=bootstrap,
        grouped_analysis=grouped_analysis(outcomes),
        runtime_metrics=dict(runtime_metrics or {}),
        metadata={
            "schema_version": SCHEMA_VERSION,
            **identity_payload(),
            "sample_count_all": int(len(universe)),
            "sample_count_with_candidates": int(candidates["sample_id"].nunique()),
            "candidate_count": int(len(candidates)),
            "scene_count": int(universe["scene_id"].nunique()),
            "frame_count": int(universe["scene_id"].nunique()),
            "sequence_scene_count": int(
                universe["scene_id"]
                .astype(str)
                .map(lambda value: value.split(",", 1)[0])
                .nunique()
            ),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(seed),
            "candidate_pool_modified": False,
            "labels_source": "per_candidate.candidate_positive joined by IDs",
        },
    )


def _numeric_equal(first: Any, second: Any, *, atol: float = 1e-12) -> bool:
    try:
        left = float(first)
        right = float(second)
    except (TypeError, ValueError):
        return first == second
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isclose(left, right, rel_tol=0.0, abs_tol=atol)


def verify_recomputed_metrics(
    per_candidate_predictions: str | Path | pd.DataFrame,
    per_sample_outcomes: str | Path | pd.DataFrame,
    per_method_metrics: str | Path | pd.DataFrame,
    stat_tests: str | Path | pd.DataFrame | None = None,
    bootstrap: str | Path | pd.DataFrame | None = None,
) -> None:
    """Recompute report tables from saved rows and reject any disagreement."""

    def load(value: str | Path | pd.DataFrame) -> pd.DataFrame:
        return (
            value.copy()
            if isinstance(value, pd.DataFrame)
            else pd.read_parquet(Path(value))
        )

    predictions = load(per_candidate_predictions)
    saved_outcomes = load(per_sample_outcomes)
    saved_metrics = load(per_method_metrics)
    candidate_scenes = (
        predictions[["sample_id", "scene_id"]]
        .drop_duplicates()
        .set_index("sample_id")["scene_id"]
        .astype(str)
    )
    outcome_scenes = (
        saved_outcomes[["sample_id", "scene_id"]]
        .drop_duplicates()
        .set_index("sample_id")["scene_id"]
        .astype(str)
    )
    shared = candidate_scenes.index.intersection(outcome_scenes.index)
    if not candidate_scenes.loc[shared].equals(outcome_scenes.loc[shared]):
        raise EvaluationError(
            "saved outcome scene IDs disagree with candidate prediction rows"
        )
    universe = (
        saved_outcomes[["sample_id", "scene_id"]]
        .drop_duplicates()
        .sort_values("sample_id")
    )
    rebuilt_outcomes = _build_sample_outcomes(predictions, universe)
    compare_columns = [
        "protocol",
        "method",
        "sample_id",
        "top1_candidate_id",
        "baseline_top1_candidate_id",
        "top1_correct",
        "recall_at_5",
        "recall_at_10",
        "j_at_any",
        "first_positive_rank",
        "outcome",
    ]
    left = saved_outcomes[compare_columns].sort_values(
        ["protocol", "method", "sample_id"]
    ).reset_index(drop=True)
    right = rebuilt_outcomes[compare_columns].sort_values(
        ["protocol", "method", "sample_id"]
    ).reset_index(drop=True)
    if not left.equals(right):
        raise EvaluationError("saved per-sample outcomes fail independent recomputation")
    rebuilt_metrics = _metrics_from_outcomes(rebuilt_outcomes)
    keys = ["protocol", "method"]
    saved = saved_metrics.set_index(keys).sort_index()
    rebuilt = rebuilt_metrics.set_index(keys).sort_index()
    if not saved.index.equals(rebuilt.index):
        raise EvaluationError("saved method set fails independent recomputation")
    common = sorted(set(saved.columns) & set(rebuilt.columns))
    for index in saved.index:
        for column in common:
            if not _numeric_equal(saved.loc[index, column], rebuilt.loc[index, column]):
                raise EvaluationError(
                    f"saved metric mismatch at {index}/{column}: "
                    f"{saved.loc[index, column]} != {rebuilt.loc[index, column]}"
                )
    if stat_tests is not None:
        saved_tests = load(stat_tests).sort_values(
            ["protocol", "method"], kind="mergesort"
        ).reset_index(drop=True)
        rebuilt_tests = _stat_tests(rebuilt_outcomes).sort_values(
            ["protocol", "method"], kind="mergesort"
        ).reset_index(drop=True)
        if list(saved_tests.columns) != list(rebuilt_tests.columns):
            raise EvaluationError("saved statistical-test schema cannot be recomputed")
        if len(saved_tests) != len(rebuilt_tests):
            raise EvaluationError(
                "saved statistical-test row count cannot be recomputed"
            )
        for row_index in range(len(saved_tests)):
            for column in saved_tests.columns:
                if not _numeric_equal(
                    saved_tests.at[row_index, column],
                    rebuilt_tests.at[row_index, column],
                ):
                    raise EvaluationError(
                        f"saved statistical test mismatch at row "
                        f"{row_index}/{column}"
                    )
    if bootstrap is not None:
        saved_bootstrap = load(bootstrap).sort_values(
            ["protocol", "method", "cluster_key", "metric"], kind="mergesort"
        ).reset_index(drop=True)
        if len(saved_bootstrap):
            replicate_values = set(
                saved_bootstrap["replicates"].astype(int).unique()
            )
            seed_values = set(saved_bootstrap["seed"].astype(int).unique())
            if len(replicate_values) != 1 or len(seed_values) != 1:
                raise EvaluationError(
                    "saved bootstrap has inconsistent replicate or seed values"
                )
            rebuilt_bootstrap = scene_grouped_bootstrap(
                rebuilt_outcomes,
                replicates=next(iter(replicate_values)),
                seed=next(iter(seed_values)),
            )
        else:
            rebuilt_bootstrap = scene_grouped_bootstrap(
                rebuilt_outcomes, replicates=0, seed=0
            )
        rebuilt_bootstrap = rebuilt_bootstrap.sort_values(
            ["protocol", "method", "cluster_key", "metric"], kind="mergesort"
        ).reset_index(drop=True)
        if list(saved_bootstrap.columns) != list(rebuilt_bootstrap.columns):
            raise EvaluationError("saved bootstrap schema cannot be recomputed")
        if len(saved_bootstrap) != len(rebuilt_bootstrap):
            raise EvaluationError("saved bootstrap row count cannot be recomputed")
        for row_index in range(len(saved_bootstrap)):
            for column in saved_bootstrap.columns:
                if not _numeric_equal(
                    saved_bootstrap.at[row_index, column],
                    rebuilt_bootstrap.at[row_index, column],
                ):
                    raise EvaluationError(
                        f"saved bootstrap mismatch at row {row_index}/{column}"
                    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def _json_safe(value: Any) -> Any:
    """Convert NumPy/pandas scalars and non-finite values to strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if pd.isna(value):
        return None
    return value


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _environment_payload() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
    }


def write_evaluation_bundle(
    output_root: str | Path,
    result: EvaluationResult,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the complete report data layer and verify it from saved rows."""

    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("per-method metrics", result.per_method_metrics),
        ("per-sample outcomes", result.per_sample_outcomes),
        ("per-candidate predictions", result.per_candidate_predictions),
    ):
        if "method" not in frame:
            raise EvaluationError(f"{name} omit method")
        try:
            validate_public_methods(
                sorted(frame["method"].astype(str).unique()),
                context=name,
            )
        except ValueError as error:
            raise EvaluationError(str(error)) from error
    result_protocols = set(
        result.per_method_metrics["protocol"].astype(str).unique()
    )
    for protocol in result_protocols:
        if protocol not in PROTOCOL_BASELINES:
            raise EvaluationError(f"unknown evaluation protocol: {protocol}")
        baseline = PROTOCOL_BASELINES[protocol]
        protocol_methods = set(
            result.per_method_metrics.loc[
                result.per_method_metrics["protocol"].astype(str).eq(protocol),
                "method",
            ].astype(str)
        )
        if baseline not in protocol_methods:
            raise EvaluationError(
                f"{protocol} is missing public baseline {baseline}"
            )
    tables = {
        "per_method_metrics": result.per_method_metrics,
        "per_sample_outcomes": result.per_sample_outcomes,
        "per_candidate_predictions": result.per_candidate_predictions,
        "stat_tests": result.stat_tests,
        "bootstrap": result.bootstrap,
        "grouped_analysis": result.grouped_analysis,
    }
    files: dict[str, Any] = {}
    for name, frame in tables.items():
        path = root / f"{name}.parquet"
        _atomic_parquet(path, frame)
        files[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "row_count": int(len(frame)),
        }
    verify_recomputed_metrics(
        root / "per_candidate_predictions.parquet",
        root / "per_sample_outcomes.parquet",
        root / "per_method_metrics.parquet",
        root / "stat_tests.parquet",
        root / "bootstrap.parquet",
    )
    required_exports = {
        "per_method_metrics_csv": root / "per_method_metrics.csv",
        "statistical_tests_json": root / "statistical_tests.json",
        "bootstrap_intervals_json": root / "bootstrap_intervals.json",
        "vlm_runtime_metrics_json": root / "vlm_runtime_metrics.json",
        "grouped_analysis_json": root / "grouped_analysis.json",
        "commands_log": root / "commands.log",
        "environment_json": root / "environment.json",
    }
    _atomic_csv(required_exports["per_method_metrics_csv"], result.per_method_metrics)
    _atomic_json(
        required_exports["statistical_tests_json"],
        result.stat_tests.to_dict(orient="records"),
    )
    _atomic_json(
        required_exports["bootstrap_intervals_json"],
        result.bootstrap.to_dict(orient="records"),
    )
    _atomic_json(
        required_exports["vlm_runtime_metrics_json"], dict(result.runtime_metrics)
    )
    _atomic_json(
        required_exports["grouped_analysis_json"],
        result.grouped_analysis.to_dict(orient="records"),
    )
    command = str(dict(provenance or {}).get("command") or "library_call")
    _atomic_text(required_exports["commands_log"], command + "\n")
    _atomic_json(required_exports["environment_json"], _environment_payload())
    for name, path in required_exports.items():
        files[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            **(
                {
                    "row_count": int(len(result.per_method_metrics))
                    if name == "per_method_metrics_csv"
                    else int(len(result.stat_tests))
                    if name == "statistical_tests_json"
                    else int(len(result.bootstrap))
                    if name == "bootstrap_intervals_json"
                    else int(len(result.grouped_analysis))
                    if name == "grouped_analysis_json"
                    else None
                }
                if name
                in {
                    "per_method_metrics_csv",
                    "statistical_tests_json",
                    "bootstrap_intervals_json",
                    "grouped_analysis_json",
                }
                else {}
            ),
        }
    bundle: dict[str, Any] = {
        **dict(result.metadata),
        "schema_version": SCHEMA_VERSION,
        **identity_payload(),
        "candidate_pool_modified": False,
        "report_recomputation_verified": True,
        "baseline_method": BASELINE_METHOD,
        "baseline_methods_by_protocol": dict(PROTOCOL_BASELINES),
        "metric_definitions": {
            "all": "all samples in the explicit sample universe, including valid-empty",
            "nonempty": "samples with at least one candidate in the evaluated protocol",
            "j_at_1": "rank-1 candidate has candidate_positive=True",
            "recall_at_k": "at least one positive candidate appears at rank <= K",
            "j_at_any": "at least one positive exists anywhere in the protocol pool",
            "mrr": "1 / first positive rank, or 0 when no positive exists",
            "recovered": "baseline J@1 failure changed to method J@1 success",
            "harmful": "baseline J@1 success changed to method J@1 failure",
            "outcome_precision": "recovered / (recovered + harmful)",
            "outcome_changing_precision": "(recovered + harmful) / all switches",
            "beneficial_switch_precision": "recovered / all switches",
            "coverage": "rank-1 candidate differs from the protocol-matched baseline",
            "rank_movement": "selected candidate original rank minus baseline selected rank",
        },
        "statistical_protocol": {
            "mcnemar": "paired exact two-sided binomial test on J@1 discordances",
            "multiple_testing": "Holm step-down within each protocol",
            "bootstrap": (
                "paired percentile bootstrap at both OCID sequence-scene "
                "and RGB-D frame cluster units"
            ),
        },
        "runtime_metrics": dict(result.runtime_metrics),
        "per_method_metrics": result.per_method_metrics.to_dict(orient="records"),
        "stat_tests": result.stat_tests.to_dict(orient="records"),
        "bootstrap_intervals": result.bootstrap.to_dict(orient="records"),
        "grouped_analysis": result.grouped_analysis.to_dict(orient="records"),
        "files": files,
        "provenance": dict(provenance or {}),
    }
    safe_bundle = _json_safe(bundle)
    _atomic_json(root / "results_bundle.json", safe_bundle)
    return safe_bundle


__all__ = [
    "BASELINE_METHOD",
    "EvaluationError",
    "EvaluationResult",
    "evaluate_predictions",
    "exact_mcnemar",
    "grouped_analysis",
    "holm_adjust",
    "predictions_from_wide",
    "scene_grouped_bootstrap",
    "verify_recomputed_metrics",
    "vlm_results_to_predictions",
    "vlm_runtime_metrics",
    "write_evaluation_bundle",
]
