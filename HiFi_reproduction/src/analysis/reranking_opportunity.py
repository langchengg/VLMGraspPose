"""Same-pool target-consistency reranking-opportunity diagnostics.

All definitions operate on a frozen official VGN candidate pool. They do not
regenerate candidates, modify VGN quality, or imply physical grasp validity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


MODEL_OUTCOMES = frozenset({"ok", "no_target_grasp", "no_official_grasp"})
OPPORTUNITY_CLASSES = (
    "technical_failure",
    "no_official_candidate",
    "generation_limited",
    "filter_recoverable",
    "post_filter_ranking_recoverable",
    "already_correct",
)


class RerankingOpportunityError(ValueError):
    """Raised when an input table violates the opportunity definitions."""


def _bool_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        raise RerankingOpportunityError(f"candidate table lacks column {name!r}")
    return frame[name].fillna(False).astype(bool)


def _ordered(group: pd.DataFrame) -> pd.DataFrame:
    required = {"vgn_quality", "candidate_index_original"}
    missing = sorted(required - set(group.columns))
    if missing:
        raise RerankingOpportunityError(f"candidate table lacks columns: {missing}")
    qualities = group["vgn_quality"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(qualities)):
        raise RerankingOpportunityError("VGN qualities must be finite")
    return group.sort_values(
        ["vgn_quality", "candidate_index_original"],
        ascending=[False, True],
        kind="mergesort",
    )


def _technical_failure(sample: pd.Series) -> bool:
    if "technical_failure" in sample and pd.notna(sample["technical_failure"]):
        if bool(sample["technical_failure"]):
            return True
    status = sample.get("pred_status", sample.get("status", "ok"))
    return str(status) not in MODEL_OUTCOMES


def build_opportunity_table(
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    positive_column: str = "gt_target_positive_primary",
) -> pd.DataFrame:
    """Build all per-sample reranking booleans and one exclusive class."""

    if "sample_id" not in samples:
        raise RerankingOpportunityError("sample table lacks sample_id")
    if samples["sample_id"].astype(str).duplicated().any():
        raise RerankingOpportunityError("sample table contains duplicate sample_id values")
    if not candidates.empty and "sample_id" not in candidates:
        raise RerankingOpportunityError("candidate table lacks sample_id")
    if not candidates.empty:
        _bool_column(candidates, positive_column)
        _bool_column(candidates, "pred_filter_pass")
    known = set(samples["sample_id"].astype(str))
    unknown = set(candidates.get("sample_id", pd.Series(dtype=str)).astype(str)) - known
    if unknown:
        raise RerankingOpportunityError(
            f"candidates refer to unknown samples: {sorted(unknown)[:3]}"
        )
    groups = {
        str(sample_id): group.copy()
        for sample_id, group in candidates.groupby("sample_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    for sample in samples.itertuples(index=False):
        sample_values = pd.Series(sample._asdict())
        sample_id = str(sample_values["sample_id"])
        group = groups.get(sample_id, candidates.iloc[0:0]).copy()
        if len(group) and group["candidate_index_original"].duplicated().any():
            raise RerankingOpportunityError(
                f"duplicate candidate_index_original for sample {sample_id}"
            )
        ordered = _ordered(group) if len(group) else group
        positive = (
            _bool_column(ordered, positive_column)
            if len(ordered)
            else pd.Series(dtype=bool)
        )
        filtered = (
            _bool_column(ordered, "pred_filter_pass")
            if len(ordered)
            else pd.Series(dtype=bool)
        )
        filtered_pool = ordered.loc[filtered] if len(ordered) else ordered
        all_top1 = ordered.iloc[0] if len(ordered) else None
        hard_top1 = filtered_pool.iloc[0] if len(filtered_pool) else None

        if "is_baseline_top1" in group and group["is_baseline_top1"].fillna(False).astype(bool).any():
            marked = group.loc[group["is_baseline_top1"].fillna(False).astype(bool)]
            if len(marked) != 1:
                raise RerankingOpportunityError(
                    f"sample {sample_id} has multiple baseline top-1 markers"
                )
            expected_index = (
                None if hard_top1 is None else int(hard_top1["candidate_index_original"])
            )
            if int(marked.iloc[0]["candidate_index_original"]) != expected_index:
                raise RerankingOpportunityError(
                    f"sample {sample_id} baseline marker does not match quality ordering"
                )

        has_official = bool(len(ordered))
        has_pred_filtered = bool(len(filtered_pool))
        has_gt_positive = bool(positive.any()) if len(ordered) else False
        has_gt_after_filter = bool(
            (_bool_column(filtered_pool, positive_column)).any()
        ) if len(filtered_pool) else False
        all_top1_positive = (
            bool(all_top1[positive_column]) if all_top1 is not None else False
        )
        hard_top1_positive = (
            bool(hard_top1[positive_column]) if hard_top1 is not None else False
        )
        positive_pool = ordered.loc[positive] if len(ordered) else ordered
        first_positive = positive_pool.iloc[0] if len(positive_pool) else None
        first_rank = (
            int(first_positive["rank_vgn_all"])
            if first_positive is not None and "rank_vgn_all" in positive_pool
            else (
                int(ordered.index.get_loc(first_positive.name)) + 1
                if first_positive is not None
                else None
            )
        )
        technical = _technical_failure(sample_values)
        if technical:
            opportunity_class = "technical_failure"
        elif not has_official:
            opportunity_class = "no_official_candidate"
        elif not has_gt_positive:
            opportunity_class = "generation_limited"
        elif not has_gt_after_filter:
            opportunity_class = "filter_recoverable"
        elif not hard_top1_positive:
            opportunity_class = "post_filter_ranking_recoverable"
        else:
            opportunity_class = "already_correct"

        best_gt_quality = (
            float(first_positive["vgn_quality"]) if first_positive is not None else np.nan
        )
        hard_quality = float(hard_top1["vgn_quality"]) if hard_top1 is not None else np.nan
        n_positive = int(positive.sum()) if len(positive) else 0
        n_negative = int(len(ordered) - n_positive)
        rows.append(
            {
                **sample._asdict(),
                "has_official_candidate": has_official,
                "has_multiple_official_candidates": len(ordered) >= 2,
                "has_pred_filtered_candidate": has_pred_filtered,
                "has_gt_positive_anywhere": has_gt_positive,
                "has_gt_positive_after_pred_filter": has_gt_after_filter,
                "vgn_all_top1_is_gt_positive": all_top1_positive,
                "hard_filter_top1_is_gt_positive": hard_top1_positive,
                "baseline_vgn_all_candidate_index": (
                    int(all_top1["candidate_index_original"])
                    if all_top1 is not None
                    else None
                ),
                "baseline_hard_filter_candidate_index": (
                    int(hard_top1["candidate_index_original"])
                    if hard_top1 is not None
                    else None
                ),
                "rank_first_gt_positive": first_rank,
                "rank_best_gt_positive": first_rank,
                "best_gt_positive_quality": best_gt_quality,
                "baseline_top1_quality": hard_quality,
                "quality_gap": (
                    hard_quality - best_gt_quality
                    if np.isfinite(hard_quality) and np.isfinite(best_gt_quality)
                    else np.nan
                ),
                "n_gt_positive_candidates": n_positive,
                "n_gt_negative_candidates": n_negative,
                "n_positive_negative_candidate_pairs": n_positive * n_negative,
                "pre_filter_recoverable": has_gt_positive and not hard_top1_positive,
                "post_filter_recoverable": has_gt_after_filter and not hard_top1_positive,
                "filter_recoverable": has_gt_positive and not has_gt_after_filter,
                # This diagnostic follows the literal same-pool definition.
                # The exclusive taxonomy still gives no-official samples the
                # more informative ``no_official_candidate`` primary class.
                "generation_limited": not has_gt_positive,
                "already_correct": hard_top1_positive,
                "opportunity_class": opportunity_class,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(samples):
        raise RerankingOpportunityError("opportunity analysis changed sample count")
    if not result["opportunity_class"].isin(OPPORTUNITY_CLASSES).all():
        raise RerankingOpportunityError("an opportunity class is outside the preregistration")
    return result


def opportunity_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Count the mutually exclusive opportunity classes over all samples."""

    if "opportunity_class" not in table:
        raise RerankingOpportunityError("table lacks opportunity_class")
    counts = table["opportunity_class"].value_counts().reindex(
        OPPORTUNITY_CLASSES, fill_value=0
    )
    denominator = len(table)
    return pd.DataFrame(
        {
            "opportunity_class": OPPORTUNITY_CLASSES,
            "numerator": counts.to_numpy(dtype=int),
            "denominator": denominator,
            "percentage": (
                100.0 * counts.to_numpy(dtype=float) / denominator
                if denominator
                else np.zeros(len(counts), dtype=float)
            ),
        }
    )


def first_positive_recall(
    table: pd.DataFrame, ks: Sequence[int] = (1, 2, 3, 5, 10)
) -> pd.DataFrame:
    """Report target-consistent candidate Recall@K and Recall@Any over samples."""

    required = {"rank_first_gt_positive", "has_gt_positive_anywhere"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise RerankingOpportunityError(f"opportunity table lacks columns: {missing}")
    denominator = len(table)
    rows: list[dict[str, object]] = []
    ranks = pd.to_numeric(table["rank_first_gt_positive"], errors="coerce")
    for raw_k in ks:
        k = int(raw_k)
        if k <= 0:
            raise RerankingOpportunityError("Recall@K requires positive K")
        numerator = int((ranks <= k).fillna(False).sum())
        rows.append(
            {
                "metric": f"Recall@{k}",
                "k": k,
                "numerator": numerator,
                "denominator": denominator,
                "percentage": 100.0 * numerator / denominator if denominator else np.nan,
            }
        )
    any_count = int(table["has_gt_positive_anywhere"].fillna(False).astype(bool).sum())
    rows.append(
        {
            "metric": "Recall@Any",
            "k": None,
            "numerator": any_count,
            "denominator": denominator,
            "percentage": 100.0 * any_count / denominator if denominator else np.nan,
        }
    )
    return pd.DataFrame(rows)


def opportunity_funnel(table: pd.DataFrame) -> pd.DataFrame:
    """Build the preregistered target-consistency opportunity funnel."""

    manifest = pd.Series(True, index=table.index)
    official = manifest & table["has_official_candidate"].fillna(False).astype(bool)
    any_positive = official & table["has_gt_positive_anywhere"].fillna(False).astype(bool)
    after_filter = any_positive & table["has_gt_positive_after_pred_filter"].fillna(False).astype(bool)
    current = after_filter & table["hard_filter_top1_is_gt_positive"].fillna(False).astype(bool)
    stages = (
        ("manifest_samples", manifest),
        ("official_candidates_exist", official),
        ("any_gt_positive_exists", any_positive),
        ("gt_positive_survives_pred_filter", after_filter),
        ("current_top1_gt_positive", current),
    )
    total = len(table)
    previous = total
    rows = []
    for name, selector in stages:
        count = int(selector.fillna(False).astype(bool).sum())
        rows.append(
            {
                "stage": name,
                "numerator": count,
                "denominator_all": total,
                "percentage_of_all": 100.0 * count / total if total else np.nan,
                "denominator_previous": previous,
                "percentage_of_previous": 100.0 * count / previous if previous else np.nan,
            }
        )
        previous = count
    return pd.DataFrame(rows)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels].sum())
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(labels.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    true_positive = 0
    observed = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        true_positive += int(labels[start:end].sum())
        observed += end - start
        recall = true_positive / positive_count
        precision = true_positive / observed
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(area)


def quality_discrimination_metrics(
    candidates: pd.DataFrame,
    *,
    positive_column: str = "gt_target_positive_primary",
) -> dict[str, float | int | str]:
    """Candidate-level quality separation; VGN quality is not a target score."""

    if "vgn_quality" not in candidates:
        raise RerankingOpportunityError("candidate table lacks vgn_quality")
    labels = _bool_column(candidates, positive_column).to_numpy(dtype=bool)
    scores = candidates["vgn_quality"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise RerankingOpportunityError("VGN qualities must be finite")
    positive_scores = scores[labels]
    negative_scores = scores[~labels]
    return {
        "candidate_count": int(len(scores)),
        "positive_candidate_count": int(labels.sum()),
        "negative_candidate_count": int((~labels).sum()),
        "positive_quality_mean": (
            float(np.mean(positive_scores)) if len(positive_scores) else float("nan")
        ),
        "positive_quality_median": (
            float(np.median(positive_scores)) if len(positive_scores) else float("nan")
        ),
        "negative_quality_mean": (
            float(np.mean(negative_scores)) if len(negative_scores) else float("nan")
        ),
        "negative_quality_median": (
            float(np.median(negative_scores)) if len(negative_scores) else float("nan")
        ),
        "roc_auc": _roc_auc(labels, scores),
        "pr_auc": _average_precision(labels, scores),
        "cluster_unit": "sample_id",
        "score_interpretation": "official VGN quality is not trained for target identity",
    }


def sample_cluster_bootstrap_quality_auc(
    candidates: pd.DataFrame,
    *,
    positive_column: str = "gt_target_positive_primary",
    replicates: int = 10_000,
    seed: int = 42,
    batch_size: int = 128,
) -> dict[str, float | int | str | list[float] | None]:
    """Sample-cluster bootstrap CIs for candidate ROC-AUC and PR-AUC.

    Scores have a fixed ordering, so each bootstrap replicate only changes
    per-sample multiplicities. Batched weighted tie-group reductions avoid
    sorting 21k candidates 10,000 separate times.
    """

    if replicates < 1 or batch_size < 1:
        raise RerankingOpportunityError("bootstrap replicates/batch size must be positive")
    required = {"sample_id", "vgn_quality", positive_column}
    missing = sorted(required - set(candidates))
    if missing:
        raise RerankingOpportunityError(f"candidate table lacks columns: {missing}")
    labels = candidates[positive_column].fillna(False).astype(bool).to_numpy()
    scores = candidates["vgn_quality"].to_numpy(np.float64)
    sample_values, sample_inverse = np.unique(
        candidates["sample_id"].astype(str).to_numpy(), return_inverse=True
    )
    if not labels.any() or labels.all():
        return {
            **quality_discrimination_metrics(candidates, positive_column=positive_column),
            "roc_auc_ci_95": None,
            "pr_auc_ci_95": None,
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "cluster_count": len(sample_values),
            "bootstrap_method": "sample_cluster_percentile",
        }

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_sample = sample_inverse[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    rng = np.random.default_rng(seed)
    roc_values: list[np.ndarray] = []
    pr_values: list[np.ndarray] = []
    probabilities = np.full(len(sample_values), 1.0 / len(sample_values))
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        sample_weights = rng.multinomial(
            len(sample_values), probabilities, size=count
        ).astype(np.float64)
        candidate_weights = sample_weights[:, sorted_sample]
        positive_weight = candidate_weights * sorted_labels
        negative_weight = candidate_weights * (~sorted_labels)
        group_positive = np.add.reduceat(positive_weight, group_starts, axis=1)
        group_negative = np.add.reduceat(negative_weight, group_starts, axis=1)
        total_positive = group_positive.sum(axis=1)
        total_negative = group_negative.sum(axis=1)
        negative_before = np.cumsum(group_negative, axis=1) - group_negative
        roc_numerator = (
            group_positive * (negative_before + 0.5 * group_negative)
        ).sum(axis=1)
        roc_denominator = total_positive * total_negative
        roc = np.divide(
            roc_numerator,
            roc_denominator,
            out=np.full_like(roc_numerator, np.nan),
            where=roc_denominator > 0,
        )
        # AP walks descending score groups. The tie-aware convention matches
        # the point estimator used above.
        pos_desc = group_positive[:, ::-1]
        total_desc = (group_positive + group_negative)[:, ::-1]
        cumulative_positive = np.cumsum(pos_desc, axis=1)
        cumulative_total = np.cumsum(total_desc, axis=1)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        pr_numerator = (pos_desc * precision).sum(axis=1)
        pr = np.divide(
            pr_numerator,
            total_positive,
            out=np.full_like(pr_numerator, np.nan),
            where=total_positive > 0,
        )
        finite = np.isfinite(roc) & np.isfinite(pr)
        roc_values.append(roc[finite])
        pr_values.append(pr[finite])
    roc_bootstrap = np.concatenate(roc_values)
    pr_bootstrap = np.concatenate(pr_values)
    point = quality_discrimination_metrics(candidates, positive_column=positive_column)
    return {
        **point,
        "roc_auc_ci_95": [float(x) for x in np.quantile(roc_bootstrap, [.025, .975])],
        "pr_auc_ci_95": [float(x) for x in np.quantile(pr_bootstrap, [.025, .975])],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "cluster_count": len(sample_values),
        "bootstrap_method": "sample_cluster_percentile",
    }


def recoverable_quality_margins(table: pd.DataFrame) -> pd.DataFrame:
    """Return finite hard-baseline-minus-best-positive margins for recoverable rows."""

    required = {"pre_filter_recoverable", "quality_gap"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise RerankingOpportunityError(f"opportunity table lacks columns: {missing}")
    mask = table["pre_filter_recoverable"].fillna(False).astype(bool)
    result = table.loc[mask].copy()
    return result.loc[pd.to_numeric(result["quality_gap"], errors="coerce").notna()]


compute_reranking_opportunity = build_opportunity_table
recall_at_k = first_positive_recall


__all__ = [
    "MODEL_OUTCOMES",
    "OPPORTUNITY_CLASSES",
    "RerankingOpportunityError",
    "build_opportunity_table",
    "compute_reranking_opportunity",
    "first_positive_recall",
    "opportunity_counts",
    "opportunity_funnel",
    "quality_discrimination_metrics",
    "sample_cluster_bootstrap_quality_auc",
    "recall_at_k",
    "recoverable_quality_margins",
]
