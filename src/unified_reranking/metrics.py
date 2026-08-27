"""Sample-level metrics for immutable order-only candidate pools."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEYS = ("sample_id", "candidate_id")


def _sample_universe(values: Iterable[object]) -> tuple[str, ...]:
    result = tuple(map(str, values))
    if not result or len(result) != len(set(result)) or any(not item for item in result):
        raise ValueError("sample universe must contain unique non-empty identifiers")
    return result


def rank_by_score(frame: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    """Rank by score with stable native-rank and candidate-ID tie breaks."""

    required = {*KEYS, "native_rank", score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"ranking table missing columns: {missing}")
    work = frame.copy()
    if work[list(KEYS)].isna().any().any() or work.duplicated(list(KEYS)).any():
        raise ValueError("ranking table has invalid or duplicate candidate keys")
    score = pd.to_numeric(work[score_column], errors="coerce").to_numpy(float)
    if not np.isfinite(score).all():
        raise ValueError("ranking scores must be finite")
    work[score_column] = score
    work = work.sort_values(
        ["sample_id", score_column, "native_rank", "candidate_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    work["rerank_rank"] = work.groupby("sample_id", sort=False).cumcount() + 1
    return work


def evaluate_order_only(
    sample_ids: Iterable[object],
    candidates_and_labels: pd.DataFrame,
    *,
    score_column: str,
    max_k: int = 5,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Evaluate a candidate permutation while retaining no-output samples.

    The denominator is the explicit sample universe.  Candidate rows alone are
    never allowed to define it.
    """

    universe = _sample_universe(sample_ids)
    required = {*KEYS, "native_rank", "candidate_success", score_column}
    missing = sorted(required.difference(candidates_and_labels.columns))
    if missing:
        raise ValueError(f"evaluation table missing columns: {missing}")
    work = candidates_and_labels.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    extra = set(work["sample_id"]).difference(universe)
    if extra:
        raise ValueError(f"candidate rows contain {len(extra)} samples outside denominator")
    labels = pd.to_numeric(work["candidate_success"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("candidate_success must be binary")
    work["candidate_success"] = labels.astype(bool)
    ranked = rank_by_score(work, score_column=score_column)

    records: list[dict[str, object]] = []
    groups = {sample_id: part for sample_id, part in ranked.groupby("sample_id", sort=False)}
    for sample_id in universe:
        group = groups.get(sample_id)
        row: dict[str, object] = {
            "sample_id": sample_id,
            "candidate_count": 0 if group is None else len(group),
            "selected_candidate_id": None,
            "selected_correct": False,
            "first_positive_rank": None,
            "reciprocal_rank": 0.0,
        }
        for k in range(1, max_k + 1):
            row[f"j_at_{k}"] = False
        if group is not None and len(group):
            ordered = group.sort_values("rerank_rank", kind="mergesort")
            positive = ordered["candidate_success"].to_numpy(bool)
            row["selected_candidate_id"] = str(ordered.iloc[0]["candidate_id"])
            row["selected_correct"] = bool(positive[0])
            indexes = np.flatnonzero(positive)
            if len(indexes):
                first = int(indexes[0] + 1)
                row["first_positive_rank"] = first
                row["reciprocal_rank"] = 1.0 / first
            for k in range(1, max_k + 1):
                row[f"j_at_{k}"] = bool(positive[:k].any())
        records.append(row)
    per_sample = pd.DataFrame(records)
    denominator = len(per_sample)
    metrics: dict[str, float | int | None] = {"sample_count": denominator}
    for k in range(1, max_k + 1):
        numerator = int(per_sample[f"j_at_{k}"].sum())
        metrics[f"j_at_{k}_numerator"] = numerator
        metrics[f"j_at_{k}"] = numerator / denominator
    metrics["oracle_at_5"] = metrics[f"j_at_{max_k}"]
    metrics["mrr_at_5"] = float(per_sample["reciprocal_rank"].mean())

    # Standard binary nDCG accounts for every relevant candidate.  Reducing it
    # to the first positive rank understates DCG when a query has >1 positive.
    ndcg_at_1: list[float] = []
    ndcg_at_k: list[float] = []
    for sample_id in universe:
        group = groups.get(sample_id)
        positive = (
            np.zeros(0, dtype=bool)
            if group is None
            else group.sort_values("rerank_rank", kind="mergesort")[
                "candidate_success"
            ].to_numpy(bool)[:max_k]
        )
        gains = positive.astype(float)
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2, dtype=float))
        dcg = float(np.sum(gains * discounts))
        relevant = int(gains.sum())
        ideal_discounts = 1.0 / np.log2(
            np.arange(2, min(relevant, max_k) + 2, dtype=float)
        )
        idcg = float(ideal_discounts.sum())
        ndcg_at_1.append(float(bool(len(positive) and positive[0])))
        ndcg_at_k.append(0.0 if idcg == 0.0 else dcg / idcg)
    metrics["ndcg_at_1"] = float(np.mean(ndcg_at_1))
    metrics[f"ndcg_at_{max_k}"] = float(np.mean(ndcg_at_k))
    return metrics, per_sample


def select_order_only(
    sample_ids: Iterable[object],
    candidate_scores: pd.DataFrame,
    *,
    score_column: str,
) -> pd.DataFrame:
    """Select one immutable candidate per sample without consulting labels."""

    universe = _sample_universe(sample_ids)
    work = candidate_scores.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    extra = set(work["sample_id"]).difference(universe)
    if extra:
        raise ValueError(f"candidate rows contain {len(extra)} samples outside denominator")
    ranked = rank_by_score(work, score_column=score_column)
    selected = (
        ranked.loc[ranked["rerank_rank"].eq(1), ["sample_id", "candidate_id"]]
        .rename(columns={"candidate_id": "selected_candidate_id"})
        .copy()
    )
    denominator = pd.DataFrame({"sample_id": universe})
    result = denominator.merge(selected, on="sample_id", how="left", validate="one_to_one")
    result["candidate_count"] = (
        result["sample_id"]
        .map(ranked.groupby("sample_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    return result


def compare_selections(
    native: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    oracle_at_5: float,
) -> dict[str, float | int | None]:
    required = {"sample_id", "selected_correct", "selected_candidate_id"}
    for name, frame in (("native", native), ("challenger", challenger)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} decisions missing columns: {missing}")
    paired = native[list(required)].merge(
        challenger[list(required)],
        on="sample_id",
        validate="one_to_one",
        suffixes=("_native", "_challenger"),
    )
    if len(paired) != len(native) or len(paired) != len(challenger):
        raise ValueError("native and challenger sample universes differ")
    before = paired["selected_correct_native"].astype(bool).to_numpy()
    after = paired["selected_correct_challenger"].astype(bool).to_numpy()
    recovered = int((~before & after).sum())
    harmful = int((before & ~after).sum())
    changed = (
        paired["selected_candidate_id_native"].fillna("").astype(str)
        != paired["selected_candidate_id_challenger"].fillna("").astype(str)
    )
    native_j = float(before.mean())
    challenger_j = float(after.mean())
    headroom = float(oracle_at_5) - native_j
    changing = recovered + harmful
    return {
        "sample_count": len(paired),
        "native_j_at_1": native_j,
        "challenger_j_at_1": challenger_j,
        "delta_j_at_1": challenger_j - native_j,
        "recovered": recovered,
        "harmful": harmful,
        "net": recovered - harmful,
        "switch_count": int(changed.sum()),
        "switch_rate": float(changed.mean()),
        "outcome_changing_precision": None if changing == 0 else recovered / changing,
        "headroom_recovery_at_5": None if headroom <= 0 else (challenger_j - native_j) / headroom,
    }


def candidate_classification_metrics(labels: Iterable[object], scores: Iterable[object]) -> dict[str, float | None]:
    y = np.asarray(list(labels), dtype=float)
    s = np.asarray(list(scores), dtype=float)
    if y.shape != s.shape or y.ndim != 1 or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid candidate classification inputs")
    if np.unique(y).size < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
    }
