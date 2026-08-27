"""Target-specific 6-DoF ranking metrics from raw evaluator rows.

The functions here intentionally retain an explicit group universe.  Empty
candidate pools therefore remain in every denominator, and a saved aggregate
can be audited by recomputing it from raw candidate-level predictions.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FRICTION_THRESHOLDS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
LABEL_GAIN = np.asarray([0.0, 1.0, 3.0, 7.0, 15.0, 31.0, 63.0])
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260815


def _mu_key(mu: float) -> str:
    return f"{float(mu):.1f}"


def _binary(values: Iterable[Any], *, name: str) -> np.ndarray:
    raw = np.asarray(list(values))
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain binary values") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain finite binary values")
    return numeric.astype(bool)


def _universe(group_universe: pd.DataFrame) -> pd.DataFrame:
    required = {"group_id", "scene_id"}
    missing = sorted(required.difference(group_universe.columns))
    if missing:
        raise ValueError(f"group universe missing columns: {missing}")
    universe = group_universe[["group_id", "scene_id"]].copy()
    if universe.isna().any().any():
        raise ValueError("group universe contains null identifiers")
    universe["group_id"] = universe["group_id"].astype(str)
    universe["scene_id"] = universe["scene_id"].astype(str)
    if (
        not len(universe)
        or universe["group_id"].eq("").any()
        or universe["scene_id"].eq("").any()
        or universe["group_id"].duplicated().any()
    ):
        raise ValueError("group universe must contain unique non-empty group IDs")
    return universe.reset_index(drop=True)


def _target_match(rows: pd.DataFrame) -> np.ndarray:
    if "target_match" in rows:
        return _binary(rows["target_match"], name="target_match")
    required = {"target_object_id", "associated_object_id"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(
            "raw rows require target_match or both target_object_id and "
            f"associated_object_id; missing {missing}"
        )
    target = rows["target_object_id"]
    associated = rows["associated_object_id"]
    return (
        target.notna()
        & associated.notna()
        & target.astype(str).ne("")
        & associated.astype(str).ne("")
        & target.astype(str).eq(associated.astype(str))
    ).to_numpy(bool)


def derive_target_success(rows: pd.DataFrame, *, mu: float) -> np.ndarray:
    """Derive target success from raw association/collision/friction fields."""

    if float(mu) <= 0:
        raise ValueError("friction threshold mu must be positive")
    required = {"collision", "pose_valid", "friction_required"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"raw evaluator rows missing columns: {missing}")
    target = _target_match(rows)
    collision = _binary(rows["collision"], name="collision")
    pose_valid = _binary(rows["pose_valid"], name="pose_valid")
    friction = pd.to_numeric(rows["friction_required"], errors="coerce").to_numpy(float)
    # graspnetAPI uses non-positive/invalid values for failed grasps.  They are
    # never allowed to pass simply because they are numerically <= mu.
    force_closure = np.isfinite(friction) & (friction > 0.0) & (friction <= float(mu))
    return target & ~collision & pose_valid & force_closure


def derive_graded_relevance(rows: pd.DataFrame) -> np.ndarray:
    """Map raw official-evaluator outcomes to the locked 0..6 relevance scale."""

    relevance = np.zeros(len(rows), dtype=np.int32)
    # Assign from loosest to strictest so stronger grasps receive larger labels.
    for label, mu in enumerate(reversed(FRICTION_THRESHOLDS), start=1):
        relevance[derive_target_success(rows, mu=mu)] = label
    # reversed thresholds produce labels 1..6 exactly: 1.2 -> 1, ..., 0.2 -> 6.
    return relevance


def rank_raw_candidates(rows: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    required = {"group_id", "candidate_id", "native_rank", score_column}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"raw prediction rows missing columns: {missing}")
    work = rows.copy()
    if work[["group_id", "candidate_id"]].isna().any().any():
        raise ValueError("candidate keys contain nulls")
    work["group_id"] = work["group_id"].astype(str)
    work["candidate_id"] = work["candidate_id"].astype(str)
    if work.duplicated(["group_id", "candidate_id"]).any():
        raise ValueError("duplicate candidate IDs within a group")
    native_rank = pd.to_numeric(work["native_rank"], errors="coerce").to_numpy(float)
    score = pd.to_numeric(work[score_column], errors="coerce").to_numpy(float)
    if (
        not np.isfinite(native_rank).all()
        or not np.equal(native_rank, np.floor(native_rank)).all()
        or bool((native_rank <= 0).any())
        or not np.isfinite(score).all()
    ):
        raise ValueError("native ranks must be positive integers and scores finite")
    work["native_rank"] = native_rank.astype(np.int64)
    work[score_column] = score
    work = work.sort_values(
        ["group_id", score_column, "native_rank", "candidate_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    work["rank"] = work.groupby("group_id", sort=False).cumcount() + 1
    return work


def _prefix_precision(success: np.ndarray, k: int) -> float:
    # Missing ranks are failures, matching GraspNet's fixed top-K denominator.
    return float(success[: int(k)].sum()) / int(k)


def _ndcg(relevance: np.ndarray, *, k: int) -> float:
    observed = relevance[: int(k)]
    discounts = 1.0 / np.log2(np.arange(2, len(observed) + 2, dtype=float))
    dcg = float(np.sum(LABEL_GAIN[observed] * discounts))
    ideal = np.sort(relevance)[::-1][: int(k)]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2, dtype=float))
    idcg = float(np.sum(LABEL_GAIN[ideal] * ideal_discounts))
    return 0.0 if idcg <= 0.0 else dcg / idcg


def evaluate_target_rankings(
    raw_rows: pd.DataFrame,
    group_universe: pd.DataFrame,
    *,
    score_column: str,
    max_k: int = 50,
    primary_mu: float = 1.2,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Compute all target/ranking/coverage metrics directly from raw rows."""

    if int(max_k) <= 0:
        raise ValueError("max_k must be positive")
    primary_mu = float(primary_mu)
    if primary_mu not in FRICTION_THRESHOLDS:
        raise ValueError(f"primary_mu must be one of {FRICTION_THRESHOLDS}")
    universe = _universe(group_universe)
    ranked = rank_raw_candidates(raw_rows, score_column=score_column)
    extra = sorted(set(ranked["group_id"]).difference(universe["group_id"]))
    if extra:
        raise ValueError(f"raw rows contain groups outside the universe: {extra[:5]}")
    if "scene_id" in ranked:
        expected_scene = universe.set_index("group_id")["scene_id"]
        observed = ranked["group_id"].map(expected_scene)
        mismatch = ranked["scene_id"].astype(str).ne(observed.astype(str))
        if mismatch.any():
            raise ValueError("raw candidate scene IDs disagree with group universe")

    ranked = ranked.copy()
    ranked["derived_relevance"] = derive_graded_relevance(ranked)
    if "relevance" in ranked:
        reported = pd.to_numeric(ranked["relevance"], errors="coerce").to_numpy(float)
        if (
            not np.isfinite(reported).all()
            or not np.array_equal(reported.astype(np.int32), ranked["derived_relevance"])
            or not np.equal(reported, np.floor(reported)).all()
        ):
            raise ValueError("saved relevance disagrees with raw evaluator outcomes")
    for mu in FRICTION_THRESHOLDS:
        ranked[f"success_mu_{_mu_key(mu)}"] = derive_target_success(ranked, mu=mu)
    ranked["target_match_derived"] = _target_match(ranked)

    ranked_groups = {
        group_id: part.sort_values("rank", kind="mergesort")
        for group_id, part in ranked.groupby("group_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    for group in universe.itertuples(index=False):
        part = ranked_groups.get(group.group_id)
        count = 0 if part is None else int(len(part))
        row: dict[str, Any] = {
            "group_id": group.group_id,
            "scene_id": group.scene_id,
            "candidate_count": count,
            "non_empty_pool": bool(count),
            "top_candidate_id": None,
        }
        if part is None:
            relevance = np.zeros(0, dtype=np.int32)
            target_match = np.zeros(0, dtype=bool)
        else:
            relevance = part["derived_relevance"].to_numpy(np.int32)
            target_match = part["target_match_derived"].to_numpy(bool)
            row["top_candidate_id"] = str(part.iloc[0]["candidate_id"])

        for mu in FRICTION_THRESHOLDS:
            key = _mu_key(mu)
            success = (
                np.zeros(0, dtype=bool)
                if part is None
                else part[f"success_mu_{key}"].to_numpy(bool)
            )
            first_indexes = np.flatnonzero(success)
            first_rank = None if not len(first_indexes) else int(first_indexes[0] + 1)
            row[f"top1_success_mu_{key}"] = bool(len(success) and success[0])
            row[f"first_valid_target_rank_mu_{key}"] = first_rank
            row[f"mrr_mu_{key}"] = 0.0 if first_rank is None else 1.0 / first_rank
            row[f"valid_target_count_mu_{key}"] = int(success.sum())
            precisions = [_prefix_precision(success, k) for k in range(1, int(max_k) + 1)]
            row[f"ap_mu_{key}"] = float(np.mean(precisions))
            for k, value in enumerate(precisions, start=1):
                row[f"precision_at_{k}_mu_{key}"] = value

        for k in (1, 5, 10):
            row[f"ndcg_at_{k}"] = _ndcg(relevance, k=k)

        # Coverage is defined over native candidate-generation order and is
        # therefore invariant to the reranking score passed above.
        if part is None:
            native_primary = np.zeros(0, dtype=bool)
        else:
            native = part.sort_values(
                ["native_rank", "candidate_id"], kind="mergesort"
            )
            native_primary = native[
                f"success_mu_{_mu_key(primary_mu)}"
            ].to_numpy(bool)
        for k in (1, 5, 10, 20, 50):
            row[f"oracle_at_{k}_mu_{_mu_key(primary_mu)}"] = bool(
                native_primary[:k].any()
            )
        row["target_candidate_count"] = int(target_match.sum())
        row["candidate_absent"] = not bool(native_primary.any())
        row["wrong_target_only_pool"] = bool(count and not target_match.any())
        records.append(row)

    per_group = pd.DataFrame(records)
    metrics: dict[str, float | int | None] = {
        "group_count": int(len(per_group)),
        "candidate_count": int(len(ranked)),
        "non_empty_pool_rate": float(per_group["non_empty_pool"].mean()),
        "candidate_absence_rate": float(per_group["candidate_absent"].mean()),
        "wrong_target_only_pool_rate": float(
            per_group["wrong_target_only_pool"].mean()
        ),
    }
    for mu in FRICTION_THRESHOLDS:
        key = _mu_key(mu)
        metrics[f"target_p_at_1_mu_{key}"] = float(
            per_group[f"top1_success_mu_{key}"].mean()
        )
        metrics[f"target_graspnet_style_ap_mu_{key}"] = float(
            per_group[f"ap_mu_{key}"].mean()
        )
        for k in range(1, int(max_k) + 1):
            metrics[f"target_precision_at_{k}_mu_{key}"] = float(
                per_group[f"precision_at_{k}_mu_{key}"].mean()
            )
    metrics["target_graspnet_style_ap_mean_mu_0.2_to_1.2"] = float(
        np.mean(
            [
                metrics[f"target_graspnet_style_ap_mu_{_mu_key(mu)}"]
                for mu in FRICTION_THRESHOLDS
            ]
        )
    )
    primary_key = _mu_key(primary_mu)
    for k in (1, 5, 10):
        metrics[f"ndcg_at_{k}"] = float(per_group[f"ndcg_at_{k}"].mean())
    metrics["mrr"] = float(per_group[f"mrr_mu_{primary_key}"].mean())
    first_ranks = pd.to_numeric(
        per_group[f"first_valid_target_rank_mu_{primary_key}"], errors="coerce"
    ).dropna()
    metrics["mean_first_valid_target_rank"] = (
        None if not len(first_ranks) else float(first_ranks.mean())
    )
    metrics["median_first_valid_target_rank"] = (
        None if not len(first_ranks) else float(first_ranks.median())
    )
    for k in (1, 5, 10, 20, 50):
        metrics[f"oracle_at_{k}"] = float(
            per_group[f"oracle_at_{k}_mu_{primary_key}"].mean()
        )
    metrics["mean_number_of_valid_target_candidates"] = float(
        per_group[f"valid_target_count_mu_{primary_key}"].mean()
    )
    return metrics, per_group


def paired_intervention_outcomes(
    reference_per_group: pd.DataFrame,
    challenger_per_group: pd.DataFrame,
    *,
    mu: float = 1.2,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Compare fixed-group Top-1 outcomes without losing empty groups."""

    key = _mu_key(mu)
    outcome = f"top1_success_mu_{key}"
    required = {"group_id", "scene_id", outcome}
    for name, frame in (
        ("reference", reference_per_group),
        ("challenger", challenger_per_group),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} per-group rows missing columns: {missing}")
        if frame["group_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate group IDs")
    paired = reference_per_group[list(required)].merge(
        challenger_per_group[list(required)],
        on="group_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_challenger"),
    )
    if len(paired) != len(reference_per_group) or len(paired) != len(challenger_per_group):
        raise ValueError("reference and challenger group universes differ")
    if paired["scene_id_reference"].astype(str).ne(
        paired["scene_id_challenger"].astype(str)
    ).any():
        raise ValueError("reference and challenger scene assignments differ")
    before = _binary(paired[f"{outcome}_reference"], name="reference outcome")
    after = _binary(paired[f"{outcome}_challenger"], name="challenger outcome")
    paired = paired.rename(columns={"scene_id_reference": "scene_id"}).drop(
        columns=["scene_id_challenger"]
    )
    paired["recovered"] = ~before & after
    paired["harmful"] = before & ~after
    paired["unchanged_success"] = before & after
    paired["unchanged_failure"] = ~before & ~after
    paired["net_recovered"] = paired["recovered"].astype(int) - paired[
        "harmful"
    ].astype(int)
    recovered = int(paired["recovered"].sum())
    harmful = int(paired["harmful"].sum())
    changing = recovered + harmful
    summary: dict[str, float | int | None] = {
        "group_count": int(len(paired)),
        "recovered": recovered,
        "harmful": harmful,
        "unchanged_success": int(paired["unchanged_success"].sum()),
        "unchanged_failure": int(paired["unchanged_failure"].sum()),
        "net_recovered": recovered - harmful,
        "net_recovered_rate": float((after.astype(float) - before.astype(float)).mean()),
        "outcome_changing_precision": (
            None if changing == 0 else float(recovered / changing)
        ),
        "reference_p_at_1": float(before.mean()),
        "challenger_p_at_1": float(after.mean()),
        "delta_p_at_1": float(after.mean() - before.mean()),
    }
    return summary, paired


def mcnemar_exact(
    reference_correct: Sequence[Any], challenger_correct: Sequence[Any]
) -> dict[str, float | int | str]:
    """Exact two-sided McNemar test using the p=0.5 binomial distribution."""

    reference = _binary(reference_correct, name="reference outcomes")
    challenger = _binary(challenger_correct, name="challenger outcomes")
    if reference.shape != challenger.shape or not len(reference):
        raise ValueError("paired outcomes must be non-empty and equal length")
    recovered = int((~reference & challenger).sum())
    harmful = int((reference & ~challenger).sum())
    discordant = recovered + harmful
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(recovered, harmful) + 1)
        ) / (2.0**discordant)
        pvalue = min(1.0, 2.0 * tail)
    return {
        "sample_count": int(len(reference)),
        "both_wrong": int((~reference & ~challenger).sum()),
        "recovered": recovered,
        "harmful": harmful,
        "both_correct": int((reference & challenger).sum()),
        "discordant": discordant,
        "effect_size": float(challenger.mean() - reference.mean()),
        "pvalue": float(pvalue),
        "method": "exact two-sided binomial McNemar",
    }


def scene_cluster_bootstrap(
    paired_groups: pd.DataFrame,
    *,
    delta_columns: Sequence[str],
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Paired percentile intervals by resampling complete scenes."""

    if "scene_id" not in paired_groups:
        raise ValueError("paired groups require scene_id for clustered bootstrap")
    columns = tuple(map(str, delta_columns))
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("delta_columns must be unique and non-empty")
    missing = sorted(set(columns).difference(paired_groups.columns))
    if missing:
        raise ValueError(f"paired groups lack delta columns: {missing}")
    if int(iterations) <= 0 or not 0.0 < float(confidence) < 1.0:
        raise ValueError("iterations and confidence are invalid")
    work = paired_groups[["scene_id", *columns]].copy()
    if not len(work) or work["scene_id"].isna().any():
        raise ValueError("paired groups/scenes must be non-empty")
    work["scene_id"] = work["scene_id"].astype(str)
    values = work[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("bootstrap delta values must be finite")
    scenes, inverse = np.unique(work["scene_id"].to_numpy(str), return_inverse=True)
    counts = np.bincount(inverse, minlength=len(scenes)).astype(float)
    sums = np.vstack(
        [np.bincount(inverse, weights=values[:, index], minlength=len(scenes)) for index in range(len(columns))]
    ).T
    rng = np.random.default_rng(int(seed))
    distributions = np.empty((int(iterations), len(columns)), dtype=float)
    for start in range(0, int(iterations), 1000):
        stop = min(start + 1000, int(iterations))
        draws = rng.integers(0, len(scenes), size=(stop - start, len(scenes)))
        denominator = counts[draws].sum(axis=1)
        distributions[start:stop] = sums[draws].sum(axis=1) / denominator[:, None]
    alpha = (1.0 - float(confidence)) / 2.0
    records = []
    for index, column in enumerate(columns):
        distribution = distributions[:, index]
        records.append(
            {
                "metric": column,
                "point_estimate": float(values[:, index].mean()),
                "ci_low": float(np.quantile(distribution, alpha)),
                "ci_high": float(np.quantile(distribution, 1.0 - alpha)),
                "confidence": float(confidence),
                "iterations": int(iterations),
                "seed": int(seed),
                "scene_count": int(len(scenes)),
                "group_count": int(len(work)),
                "resampling_unit": "scene",
            }
        )
    return pd.DataFrame(records)


def paired_metric_deltas(
    reference_per_group: pd.DataFrame,
    challenger_per_group: pd.DataFrame,
    *,
    mu: float = 1.2,
) -> pd.DataFrame:
    """Build per-group P@1/AP/MRR/net deltas for scene bootstrap."""

    key = _mu_key(mu)
    columns = {
        "group_id",
        "scene_id",
        f"top1_success_mu_{key}",
        f"ap_mu_{key}",
        f"mrr_mu_{key}",
    }
    for name, frame in (("reference", reference_per_group), ("challenger", challenger_per_group)):
        missing = sorted(columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} per-group rows missing columns: {missing}")
    paired = reference_per_group[list(columns)].merge(
        challenger_per_group[list(columns)],
        on="group_id",
        validate="one_to_one",
        suffixes=("_reference", "_challenger"),
    )
    if len(paired) != len(reference_per_group) or len(paired) != len(challenger_per_group):
        raise ValueError("reference and challenger group universes differ")
    if paired["scene_id_reference"].astype(str).ne(paired["scene_id_challenger"].astype(str)).any():
        raise ValueError("reference and challenger scene assignments differ")
    result = pd.DataFrame(
        {
            "group_id": paired["group_id"].astype(str),
            "scene_id": paired["scene_id_reference"].astype(str),
        }
    )
    before = _binary(
        paired[f"top1_success_mu_{key}_reference"], name="reference P@1"
    ).astype(float)
    after = _binary(
        paired[f"top1_success_mu_{key}_challenger"], name="challenger P@1"
    ).astype(float)
    result["delta_p_at_1"] = after - before
    result["delta_ap"] = (
        paired[f"ap_mu_{key}_challenger"].to_numpy(float)
        - paired[f"ap_mu_{key}_reference"].to_numpy(float)
    )
    result["delta_mrr"] = (
        paired[f"mrr_mu_{key}_challenger"].to_numpy(float)
        - paired[f"mrr_mu_{key}_reference"].to_numpy(float)
    )
    # Per-group net recovered is +1 / -1 / 0 and is algebraically the
    # Top-1 delta, but retained under its intervention interpretation.
    result["net_recovered_rate"] = result["delta_p_at_1"]
    return result


def assert_metrics_match_raw_predictions(
    reported_metrics: Mapping[str, Any],
    raw_rows: pd.DataFrame,
    group_universe: pd.DataFrame,
    *,
    score_column: str,
    max_k: int = 50,
    atol: float = 1e-12,
) -> dict[str, float | int | None]:
    """Recompute reported numeric metrics and fail on any missing/mismatched key."""

    recomputed, _ = evaluate_target_rankings(
        raw_rows,
        group_universe,
        score_column=score_column,
        max_k=max_k,
    )
    for key, expected in recomputed.items():
        if key not in reported_metrics:
            raise ValueError(f"reported metrics missing recomputable key {key!r}")
        observed = reported_metrics[key]
        if expected is None:
            if observed is not None:
                raise ValueError(f"reported metric {key!r} should be null")
        elif isinstance(expected, int):
            if int(observed) != expected:
                raise ValueError(
                    f"reported metric {key!r}={observed!r}, recomputed={expected!r}"
                )
        elif not math.isclose(float(observed), float(expected), abs_tol=atol, rel_tol=0.0):
            raise ValueError(
                f"reported metric {key!r}={observed!r}, recomputed={expected!r}"
            )
    return recomputed


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "FRICTION_THRESHOLDS",
    "assert_metrics_match_raw_predictions",
    "derive_graded_relevance",
    "derive_target_success",
    "evaluate_target_rankings",
    "mcnemar_exact",
    "paired_intervention_outcomes",
    "paired_metric_deltas",
    "rank_raw_candidates",
    "scene_cluster_bootstrap",
]
