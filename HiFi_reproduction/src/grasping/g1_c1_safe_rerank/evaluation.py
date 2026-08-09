"""Complete-denominator offline evaluation and scene-clustered statistics."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def evaluate_selected_ids(
    selections: pd.DataFrame,
    labels: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    method: str,
    allow_empty_selected: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selections = selections.copy()
    labels = labels.copy()
    universe = universe.copy()
    required_universe = {"sample_id", "scene_id"}
    if not required_universe.issubset(universe.columns):
        raise ValueError("sample universe requires sample_id and scene_id")
    if universe[["sample_id", "scene_id"]].isna().any().any():
        raise ValueError("sample universe contains null identifiers")
    universe["sample_id"] = universe["sample_id"].astype(str)
    universe["scene_id"] = universe["scene_id"].astype(str)
    if universe["sample_id"].eq("").any() or universe["scene_id"].eq("").any():
        raise ValueError("sample universe contains empty identifiers")
    if universe["sample_id"].duplicated().any():
        raise ValueError("sample universe contains duplicate sample IDs")
    label_keys = ["sample_id", "stable_candidate_id"]
    if not {*label_keys, "candidate_correct"}.issubset(labels.columns):
        raise ValueError("candidate labels lack required columns")
    if labels[label_keys].isna().any().any():
        raise ValueError("candidate labels contain null identifiers")
    labels["sample_id"] = labels["sample_id"].astype(str)
    labels["stable_candidate_id"] = labels["stable_candidate_id"].astype(str)
    if labels[label_keys].eq("").any().any():
        raise ValueError("candidate labels contain empty identifiers")
    correctness_values = pd.to_numeric(labels["candidate_correct"], errors="coerce")
    if correctness_values.isna().any() or not correctness_values.isin([0, 1]).all():
        raise ValueError("candidate_correct must contain only finite binary values")
    labels["candidate_correct"] = correctness_values.astype(bool)
    if labels.duplicated(label_keys).any():
        raise ValueError("candidate labels contain duplicate keys")
    required_selection = {"sample_id", "baseline_candidate_id", "selected_candidate_id"}
    if not required_selection.issubset(selections.columns):
        raise ValueError("selections lack required columns")
    if selections[list(required_selection)].isna().any().any():
        raise ValueError("selections contain null identifiers")
    for column in required_selection:
        selections[column] = selections[column].astype(str)
    if selections["sample_id"].duplicated().any():
        raise ValueError("duplicate sample selection")
    universe_ids = set(universe["sample_id"].astype(str))
    selection_ids = set(selections["sample_id"].astype(str))
    labelled_sample_ids = set(labels["sample_id"].astype(str))
    if not selection_ids.issubset(universe_ids):
        raise ValueError("selection contains samples outside the evaluation universe")
    if selection_ids != labelled_sample_ids:
        raise ValueError("selections must exactly cover every non-empty labelled sample")
    correctness = labels.set_index(["sample_id", "stable_candidate_id"])["candidate_correct"]
    baseline_candidate_keys = set(
        zip(
            selections["sample_id"].astype(str),
            selections["baseline_candidate_id"].astype(str),
        )
    )
    selected_candidate_keys = set(
        zip(
            selections["sample_id"].astype(str),
            selections["selected_candidate_id"].astype(str),
        )
    )
    # An empty baseline ID is a deliberate representation of an empty
    # baseline backend (for example always-G1 when only C1 emitted a candidate).
    # It is incorrect by definition and must never be backfilled with the
    # challenger's ID, which would inflate the reference method.
    if (not allow_empty_selected) and any(
        str(candidate_id) == "" for _, candidate_id in selected_candidate_keys
    ):
        raise ValueError("a labelled sample has an empty selected candidate ID")
    required_candidate_keys = {
        key for key in selected_candidate_keys if str(key[1]) != ""
    } | {
        key for key in baseline_candidate_keys if str(key[1]) != ""
    }
    available_candidate_keys = set(
        zip(labels["sample_id"].astype(str), labels["stable_candidate_id"].astype(str))
    )
    missing_candidate_keys = required_candidate_keys - available_candidate_keys
    if missing_candidate_keys:
        raise ValueError(f"selected candidate labels are missing: {len(missing_candidate_keys)}")
    selected = selections.set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for sample in universe.itertuples(index=False):
        sample_id = str(sample.sample_id)
        if sample_id in selected.index:
            row = selected.loc[sample_id]
            if isinstance(row, pd.DataFrame):
                raise ValueError("duplicate sample selection")
            old_id = str(row["baseline_candidate_id"])
            new_id = str(row["selected_candidate_id"])
            old_correct = (
                bool(correctness.loc[(sample_id, old_id)]) if old_id else False
            )
            new_correct = (
                bool(correctness.loc[(sample_id, new_id)]) if new_id else False
            )
            switch = old_id != new_id
            fallback = bool(row.get("fallback", False))
        else:
            old_id = new_id = ""
            old_correct = new_correct = switch = fallback = False
        recovered = (not old_correct) and new_correct
        harmful = old_correct and (not new_correct)
        rows.append(
            {
                "sample_id": sample_id,
                "scene_id": str(sample.scene_id),
                "method": str(method),
                "baseline_candidate_id": old_id,
                "selected_candidate_id": new_id,
                "baseline_correct": old_correct,
                "final_correct": new_correct,
                "switch": switch,
                "recovered": recovered,
                "harmful": harmful,
                "neutral_both_correct": old_correct and new_correct,
                "neutral_both_wrong": (not old_correct) and (not new_correct),
                "fallback": fallback,
            }
        )
    outcomes = pd.DataFrame(rows)
    total = len(outcomes)
    recovered = int(outcomes["recovered"].sum())
    harmful = int(outcomes["harmful"].sum())
    baseline_correct = int(outcomes["baseline_correct"].sum())
    final_correct = int(outcomes["final_correct"].sum())
    metrics = {
        "method": str(method),
        "total": total,
        "complete_denominator": True,
        "baseline_j_at_1": baseline_correct / max(total, 1),
        "j_at_1": final_correct / max(total, 1),
        "delta_j_at_1": (final_correct - baseline_correct) / max(total, 1),
        "recovered": recovered,
        "harmful": harmful,
        "net": recovered - harmful,
        "switch_count": int(outcomes["switch"].sum()),
        "switch_rate": float(outcomes["switch"].mean()),
        "keep_rate": float(1.0 - outcomes["switch"].mean()),
        "harm_rate": harmful / max(baseline_correct, 1),
        "outcome_changing_precision": recovered / max(recovered + harmful, 1),
        "fallback_rate": float(outcomes["fallback"].mean()),
    }
    return outcomes, metrics


def oracle_metrics(candidates: pd.DataFrame, labels: pd.DataFrame, universe: pd.DataFrame) -> dict[str, Any]:
    keys = ["sample_id", "stable_candidate_id"]
    if candidates.duplicated(keys).any() or labels.duplicated(keys).any():
        raise ValueError("oracle inputs contain duplicate candidate keys")
    merged = candidates.merge(
        labels[["sample_id", "stable_candidate_id", "candidate_correct"]],
        on=["sample_id", "stable_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["candidate_correct"].isna().any():
        raise ValueError("oracle labels do not cover every candidate")
    merged["candidate_correct"] = merged["candidate_correct"].astype(bool)
    total = len(universe)
    result: dict[str, Any] = {"sample_count": total}
    rank_column = (
        "pool_rank"
        if "pool_rank" in merged.columns
        else "union_rank"
        if "union_rank" in merged.columns
        else "original_rank"
    )
    for k in (1, 5, 10):
        correct = (
            merged.loc[merged[rank_column] <= k]
            .groupby("sample_id")["candidate_correct"]
            .any()
        )
        result[f"oracle_at_{k}"] = int(correct.sum()) / max(total, 1)
        result[f"oracle_at_{k}_count"] = int(correct.sum())
    any_correct = merged.groupby("sample_id")["candidate_correct"].any()
    result["oracle_at_all"] = int(any_correct.sum()) / max(total, 1)
    result["oracle_at_all_count"] = int(any_correct.sum())
    top = merged.loc[merged[rank_column].eq(1)]
    top_correct = top.groupby("sample_id")["candidate_correct"].any()
    result["j_at_1"] = int(top_correct.sum()) / max(total, 1)
    result["j_at_1_count"] = int(top_correct.sum())
    result["non_empty_rate"] = candidates["sample_id"].nunique() / max(total, 1)
    if any(
        int(result[key]) > total
        for key in result
        if key.endswith("_count")
    ):
        raise AssertionError("oracle count exceeds evaluation denominator")
    return result


def exact_mcnemar(outcomes: pd.DataFrame) -> dict[str, Any]:
    recovered = int(outcomes["recovered"].sum())
    harmful = int(outcomes["harmful"].sum())
    discordant = recovered + harmful
    p = 1.0 if discordant == 0 else float(
        binomtest(min(recovered, harmful), discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {"recovered": recovered, "harmful": harmful, "discordant": discordant, "p_value": p}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (len(values) - position) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def scene_cluster_bootstrap(
    outcomes: pd.DataFrame,
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    scenes = list(dict.fromkeys(outcomes["scene_id"].astype(str).tolist()))
    if not scenes:
        raise ValueError("scene bootstrap requires non-empty outcomes")
    by_scene = {scene: group for scene, group in outcomes.groupby("scene_id", sort=False)}
    rng = np.random.default_rng(int(seed))
    delta = np.empty(int(draws), dtype=float)
    net = np.empty(int(draws), dtype=float)
    precision = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        parts = [by_scene[str(scene)] for scene in sampled]
        total = sum(len(part) for part in parts)
        recovered = sum(int(part["recovered"].sum()) for part in parts)
        harmful = sum(int(part["harmful"].sum()) for part in parts)
        delta[draw] = (recovered - harmful) / max(total, 1)
        net[draw] = recovered - harmful
        precision[draw] = recovered / max(recovered + harmful, 1)
    def interval(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return {"mean": float(values.mean()), "lower": float(low), "upper": float(high)}
    return {
        "cluster": "scene_id",
        "scene_count": len(scenes),
        "draws": int(draws),
        "seed": int(seed),
        "delta_j_at_1": interval(delta),
        "net": interval(net),
        "outcome_changing_precision": interval(precision),
    }


def stable_bootstrap_seed(base: int, method: str) -> int:
    digest = hashlib.sha256(f"{base}\t{method}".encode()).digest()
    return int.from_bytes(digest[:4], "big")
