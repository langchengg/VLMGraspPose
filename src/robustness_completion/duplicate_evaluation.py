"""Frozen-outcome evaluation after the independent duplicate map is locked."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.statistics import (
    cluster_bootstrap_difference,
    mcnemar_exact,
)

from .common import (
    atomic_frame,
    atomic_json,
    require_run_dir,
    sha256_file,
    verify_preregistration,
)
from .duplicate_map import _verify_locked_outputs, exclusion_tuple_ids


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260815
PARENT_PAIRED = (
    "artifacts/robustness_suite/20260822_085125_robustness_suite/"
    "4d_threshold_sensitivity/paired_outcomes.parquet"
)
FORMAL_TEST_MANIFEST = (
    "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_test.parquet"
)


def _bootstrap_gain_shift(
    full: pd.DataFrame, keep: np.ndarray, challenger: str
) -> dict[str, Any]:
    clusters = full["sequence_id"].astype(str).to_numpy()
    unique, inverse = np.unique(clusters, return_inverse=True)
    native = full["native_correct"].to_numpy(bool).astype(float)
    reranked = full[challenger].to_numpy(bool).astype(float)
    delta = reranked - native
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    distribution = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    full_sums = np.bincount(inverse, weights=delta, minlength=len(unique))
    full_counts = np.bincount(inverse, minlength=len(unique))
    filtered_sums = np.bincount(
        inverse[keep], weights=delta[keep], minlength=len(unique)
    )
    filtered_counts = np.bincount(inverse[keep], minlength=len(unique))
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        stop = min(start + 1000, BOOTSTRAP_REPLICATES)
        draws = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        full_gain = full_sums[draws].sum(axis=1) / full_counts[draws].sum(axis=1)
        filtered_denominator = filtered_counts[draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            filtered_gain = filtered_sums[draws].sum(axis=1) / filtered_denominator
        distribution[start:stop] = filtered_gain - full_gain
    finite = distribution[np.isfinite(distribution)]
    if not finite.size:
        return {
            "point_estimate": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "valid_replicates": 0,
        }
    point = float(delta[keep].mean() - delta.mean()) if keep.any() else math.nan
    return {
        "point_estimate": point,
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "valid_replicates": int(len(finite)),
    }


def _method_statistics(
    frame: pd.DataFrame, challenger: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    native = frame["native_correct"].to_numpy(bool)
    reranked = frame[challenger].to_numpy(bool)
    test = mcnemar_exact(native, reranked)
    bootstrap = cluster_bootstrap_difference(
        native,
        reranked,
        frame["sequence_id"].astype(str),
        iterations=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    return test, bootstrap


def _metric_row(
    route: str,
    retrospective: bool,
    analysis_set: str,
    subset: str,
    frame: pd.DataFrame,
    full: pd.DataFrame,
    keep_in_full: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if frame.empty:
        raise RuntimeError(f"empty filtered set: {route}/{analysis_set}/{subset}")
    native = frame["native_correct"].to_numpy(bool)
    raw = frame["raw_reranked_correct"].to_numpy(bool)
    gated = frame["reranked_correct"].to_numpy(bool)
    oracle = frame["oracle_at_k"].to_numpy(bool)
    raw_test, raw_boot = _method_statistics(frame, "raw_reranked_correct")
    gated_test, gated_boot = _method_statistics(frame, "reranked_correct")
    raw_shift = _bootstrap_gain_shift(full, keep_in_full, "raw_reranked_correct")
    gated_shift = _bootstrap_gain_shift(full, keep_in_full, "reranked_correct")
    n = len(frame)
    native_n, raw_n, gated_n, oracle_n = map(
        int, (native.sum(), raw.sum(), gated.sum(), oracle.sum())
    )
    headroom = oracle_n - native_n
    row = {
        "route": route,
        "retrospective": retrospective,
        "analysis_set": analysis_set,
        "subset": subset,
        "n_tuples": n,
        "n_unique_observations": int(frame["observation_id"].nunique()),
        "n_sequences": int(frame["sequence_id"].nunique()),
        "low_power": bool(n < 1000 or frame["sequence_id"].nunique() < 20),
        "native_num": native_n,
        "native_j1": native_n / n,
        "raw_num": raw_n,
        "raw_j1": raw_n / n,
        "gated_num": gated_n,
        "gated_j1": gated_n / n,
        "oracle_num": oracle_n,
        "oracle_at_k": oracle_n / n,
        "raw_gain_pp": 100.0 * (raw_n - native_n) / n,
        "gated_gain_pp": 100.0 * (gated_n - native_n) / n,
        "raw_recovered": raw_test["recovered"],
        "raw_harmful": raw_test["harmful"],
        "raw_net_recovered": raw_test["net_recovered"],
        "raw_outcome_changing_precision": (
            raw_test["recovered"] / raw_test["discordant"]
            if raw_test["discordant"]
            else math.nan
        ),
        "gated_recovered": gated_test["recovered"],
        "gated_harmful": gated_test["harmful"],
        "gated_net_recovered": gated_test["net_recovered"],
        "gated_outcome_changing_precision": (
            gated_test["recovered"] / gated_test["discordant"]
            if gated_test["discordant"]
            else math.nan
        ),
        "raw_headroom_recovery": (
            (raw_n - native_n) / headroom if headroom > 0 else math.nan
        ),
        "gated_headroom_recovery": (
            (gated_n - native_n) / headroom if headroom > 0 else math.nan
        ),
        "raw_ci_low_pp": 100.0 * raw_boot["ci95"][0],
        "raw_ci_high_pp": 100.0 * raw_boot["ci95"][1],
        "gated_ci_low_pp": 100.0 * gated_boot["ci95"][0],
        "gated_ci_high_pp": 100.0 * gated_boot["ci95"][1],
        "raw_mcnemar_p": raw_test["pvalue"],
        "gated_mcnemar_p": gated_test["pvalue"],
        "raw_gain_shift_pp": 100.0 * raw_shift["point_estimate"],
        "raw_gain_shift_ci_low_pp": 100.0 * raw_shift["ci_low"],
        "raw_gain_shift_ci_high_pp": 100.0 * raw_shift["ci_high"],
        "gated_gain_shift_pp": 100.0 * gated_shift["point_estimate"],
        "gated_gain_shift_ci_low_pp": 100.0 * gated_shift["ci_low"],
        "gated_gain_shift_ci_high_pp": 100.0 * gated_shift["ci_high"],
    }
    bootstrap_rows = []
    for method, boot, shift in (
        ("raw", raw_boot, raw_shift),
        ("gated", gated_boot, gated_shift),
    ):
        bootstrap_rows.extend(
            [
                {
                    "route": route,
                    "analysis_set": analysis_set,
                    "subset": subset,
                    "method": method,
                    "estimand": "paired_gain_pp",
                    "point_estimate_pp": 100.0 * boot["point_estimate"],
                    "ci_low_pp": 100.0 * boot["ci95"][0],
                    "ci_high_pp": 100.0 * boot["ci95"][1],
                    "iterations": BOOTSTRAP_REPLICATES,
                    "seed": BOOTSTRAP_SEED,
                    "clusters": int(boot["cluster_count"]),
                },
                {
                    "route": route,
                    "analysis_set": analysis_set,
                    "subset": subset,
                    "method": method,
                    "estimand": "gain_shift_from_full_pp",
                    "point_estimate_pp": 100.0 * shift["point_estimate"],
                    "ci_low_pp": 100.0 * shift["ci_low"],
                    "ci_high_pp": 100.0 * shift["ci_high"],
                    "iterations": BOOTSTRAP_REPLICATES,
                    "seed": BOOTSTRAP_SEED,
                    "clusters": int(full["sequence_id"].nunique()),
                },
            ]
        )
    significance = {"raw": raw_test, "gated": gated_test}
    return row, bootstrap_rows, significance


def evaluate_duplicate_exclusion(
    repo: Path, run_dir: Path, *, resume: bool = False
) -> dict[str, Any]:
    repo = repo.resolve()
    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    audit = run_dir / "duplicate_audit"
    lock_path = audit / "DUPLICATE_MAP_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("duplicate map must be locked before outcomes are loaded")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _verify_locked_outputs(audit, lock)
    output = run_dir / "duplicate_exclusion"
    completion_path = output / "completion.json"
    if completion_path.exists() and resume:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        for relative, expected in completion["output_sha256"].items():
            if sha256_file(output / relative) != expected:
                raise RuntimeError(f"duplicate evaluation hash mismatch: {relative}")
        return {"status": "COMPLETE", "resumed": True, **completion["counts"]}

    source_path = repo / PARENT_PAIRED
    source = pd.read_parquet(source_path)
    source = source[source["formal_cell"].astype(bool)].copy()
    if len(source) != 4 * 7675:
        raise RuntimeError("parent formal-cell outcome denominator changed")
    denominator = pd.read_parquet(
        repo / FORMAL_TEST_MANIFEST,
        columns=["sample_id", "rgbd_pair_sha256", "scene_id"],
    ).rename(columns={"rgbd_pair_sha256": "observation_id"})
    denominator["sample_id"] = denominator["sample_id"].astype(str)
    denominator["sequence_id_manifest"] = denominator["scene_id"].astype(str).str.rsplit(",", n=1).str[0]
    source["sample_id"] = source["sample_id"].astype(str)
    paired = source.merge(
        denominator[["sample_id", "observation_id", "sequence_id_manifest"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if paired["observation_id"].isna().any():
        raise RuntimeError("formal outcomes cannot be mapped to observations")
    if not (paired["sequence_id"].astype(str) == paired["sequence_id_manifest"].astype(str)).all():
        raise RuntimeError("sequence identity changed between frozen artifacts")
    paired = paired.drop(columns="sequence_id_manifest")
    removed = {
        tier: exclusion_tuple_ids(audit / f"test_observation_exclusion_{tier}.csv")
        for tier in ("exact", "strict", "moderate")
    }
    if not removed["exact"].issubset(removed["strict"]) or not removed["strict"].issubset(removed["moderate"]):
        raise RuntimeError("duplicate exclusion tiers are not nested")
    all_ids = set(denominator["sample_id"])
    if any(not values.issubset(all_ids) for values in removed.values()):
        raise RuntimeError("duplicate exclusion contains non-formal tuple IDs")
    for tier, values in removed.items():
        paired[f"excluded_{tier}"] = paired["sample_id"].isin(values)
    frozen_columns = list(source.columns)
    if not paired[frozen_columns].reset_index(drop=True).equals(source.reset_index(drop=True)):
        raise RuntimeError("joining exclusions changed frozen prediction values")
    atomic_frame(output / "paired_outcomes.parquet", paired)
    strict_manifest = pd.read_csv(audit / "test_observation_exclusion_strict.csv")
    atomic_frame(output / "strict_exclusion_manifest.csv", strict_manifest)

    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    significance: dict[str, Any] = {"routes": {}, "bootstrap_unit": "sequence"}
    subset_to_column = {
        "full": None,
        "exact_excluded": "excluded_exact",
        "strict_excluded": "excluded_strict",
        "moderate_excluded": "excluded_moderate",
    }
    for route, route_frame in paired.groupby("route", sort=True):
        route_frame = route_frame.reset_index(drop=True)
        significance["routes"][route] = {}
        for subset, exclusion_column in subset_to_column.items():
            keep = (
                np.ones(len(route_frame), dtype=bool)
                if exclusion_column is None
                else ~route_frame[exclusion_column].to_numpy(bool)
            )
            selected = route_frame.loc[keep].copy()
            row, boots, tests = _metric_row(
                route,
                bool(route_frame["retrospective"].iloc[0]),
                "route_specific",
                subset,
                selected,
                route_frame,
                keep,
            )
            metric_rows.append(row)
            bootstrap_rows.extend(boots)
            significance["routes"][route][subset] = tests

    primary_routes = paired[paired["route"].isin(["CROG", "G1", "C1"])]
    route_sets = [set(frame["sample_id"]) for _, frame in primary_routes.groupby("route")]
    common_full = set.intersection(*route_sets)
    for route, route_frame in primary_routes.groupby("route", sort=True):
        route_frame = route_frame[route_frame["sample_id"].isin(common_full)].reset_index(drop=True)
        for subset, exclusion_column in subset_to_column.items():
            keep = (
                np.ones(len(route_frame), dtype=bool)
                if exclusion_column is None
                else ~route_frame[exclusion_column].to_numpy(bool)
            )
            row, boots, _tests = _metric_row(
                route,
                False,
                "primary_common_intersection",
                subset,
                route_frame.loc[keep].copy(),
                route_frame,
                keep,
            )
            metric_rows.append(row)
            bootstrap_rows.extend(boots)

    metrics = pd.DataFrame(metric_rows)
    atomic_frame(output / "full_vs_filtered_metrics.csv", metrics)
    atomic_frame(output / "bootstrap_results.csv", pd.DataFrame(bootstrap_rows))
    atomic_json(output / "significance_tests.json", significance)
    observations = pd.read_parquet(audit / "canonical_observations.parquet")
    train_sequences = set(observations.loc[observations["split"] == "train", "sequence_id"].astype(str))
    test_sequences = set(observations.loc[observations["split"] == "test", "sequence_id"].astype(str))
    overlap = sorted(train_sequences & test_sequences)
    sequence_diagnostic = {
        "train_sequence_count": len(train_sequences),
        "test_sequence_count": len(test_sequences),
        "overlapping_test_sequence_count": len(overlap),
        "sequence_disjoint_test_sequence_count": len(test_sequences - train_sequences),
        "sequence_disjoint_evaluation_estimable": bool(test_sequences - train_sequences),
        "overlapping_sequences": overlap,
        "interpretation": (
            "sequence-disjoint evaluation is not estimable"
            if not (test_sequences - train_sequences)
            else "sequence overlap and frame-level near duplication are separate diagnostics"
        ),
    }
    atomic_json(output / "sequence_overlap_diagnostic.json", sequence_diagnostic)
    names = [
        "strict_exclusion_manifest.csv",
        "full_vs_filtered_metrics.csv",
        "paired_outcomes.parquet",
        "bootstrap_results.csv",
        "significance_tests.json",
        "sequence_overlap_diagnostic.json",
    ]
    counts = {
        "metric_rows": int(len(metrics)),
        "strict_removed_tuples": int(len(removed["strict"])),
        "strict_remaining_tuples": int(7675 - len(removed["strict"])),
    }
    completion = {
        "status": "COMPLETE",
        "duplicate_map_lock_sha256": sha256_file(lock_path),
        "source_paired_outcomes_sha256": sha256_file(source_path),
        "counts": counts,
        "output_sha256": {name: sha256_file(output / name) for name in names},
    }
    atomic_json(completion_path, completion)
    return {"status": "COMPLETE", "resumed": False, **counts}
