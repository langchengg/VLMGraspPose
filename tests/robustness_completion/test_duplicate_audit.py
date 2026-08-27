from __future__ import annotations

import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from robustness_completion.common import PREREGISTRATION_SHA256, sha256_file
from robustness_completion.duplicate_evaluation import BOOTSTRAP_SEED
from robustness_completion.duplicate_map import (
    _aligned_views,
    array_sha256,
    build_duplicate_map,
    dhash64,
    exclusion_tuple_ids,
    phash64,
)
from unified_reranking.statistics import cluster_bootstrap_difference


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "artifacts/robustness_completion/20260822_194405_remaining_robustness"
AUDIT = RUN / "duplicate_audit"
EVALUATION = RUN / "duplicate_exclusion"
PARENT = (
    REPO
    / "artifacts/robustness_suite/20260822_085125_robustness_suite"
    / "4d_threshold_sensitivity/paired_outcomes.parquet"
)


def _observations() -> pd.DataFrame:
    return pd.read_parquet(AUDIT / "canonical_observations.parquet")


def _pairs() -> pd.DataFrame:
    return pd.read_parquet(AUDIT / "all_candidate_pairs.parquet")


def _outcomes() -> pd.DataFrame:
    return pd.read_parquet(EVALUATION / "paired_outcomes.parquet")


def test_duplicate_builder_does_not_import_outcomes() -> None:
    source = (REPO / "src/robustness_completion/duplicate_map.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_prefixes = (
        "robustness_completion.duplicate_evaluation",
        "robustness_suite",
        "unified_reranking",
        "graspnet6d.experiment_analysis",
    )
    assert not any(module.startswith(forbidden_prefixes) for module in imports)


def test_observation_manifest_is_query_deduplicated() -> None:
    observations = _observations()
    assert observations["observation_id"].is_unique
    totals = observations.groupby("split")["number_of_queries"].sum().to_dict()
    assert totals == {"test": 7675, "train": 26295, "validation": 3778}
    tuple_ids = [
        value
        for encoded in observations["tuple_ids"]
        for value in json.loads(encoded)
    ]
    assert len(tuple_ids) == len(set(tuple_ids)) == 37748


def test_rgb_hash_is_deterministic() -> None:
    array = np.arange(11 * 13 * 3, dtype=np.uint8).reshape(11, 13, 3)
    assert array_sha256(array) == array_sha256(array.copy())
    changed = array.copy()
    changed[0, 0, 0] ^= 1
    assert array_sha256(array) != array_sha256(changed)


def test_depth_hash_is_deterministic() -> None:
    depth = np.asarray([[1.0, np.nan], [2.0, 0.0]], dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    first = array_sha256(np.nan_to_num(depth), invalid_mask=valid)
    second = array_sha256(np.nan_to_num(depth.copy()), invalid_mask=valid.copy())
    assert first == second


def test_phash_is_deterministic() -> None:
    rng = np.random.default_rng(20260815)
    luminance = rng.uniform(0, 255, size=(256, 256)).astype(np.float32)
    assert phash64(luminance) == phash64(luminance.copy())
    assert dhash64(luminance) == dhash64(luminance.copy())
    assert len(phash64(luminance)) == len(dhash64(luminance)) == 16


def test_phase_alignment_is_bounded() -> None:
    rng = np.random.default_rng(7)
    luminance = rng.uniform(0, 255, (256, 256)).astype(np.float32)
    rgb = np.repeat(luminance[..., None], 3, axis=2).astype(np.uint8)
    depth = np.ones((256, 256), dtype=np.float32)
    valid = np.ones((256, 256), dtype=bool)
    transform = np.asarray([[1, 0, 3], [0, 1, -2]], np.float32)
    shifted_lum = cv2.warpAffine(luminance, transform, (256, 256))
    shifted_rgb = cv2.warpAffine(rgb, transform, (256, 256))
    shifted_depth = cv2.warpAffine(depth, transform, (256, 256))
    shifted_valid = cv2.warpAffine(valid.astype(np.uint8), transform, (256, 256)).astype(bool)
    _, dx, dy, _ = _aligned_views(
        (rgb, luminance, depth, valid),
        (shifted_rgb, shifted_lum, shifted_depth, shifted_valid),
    )
    assert abs(dx) <= 4 and abs(dy) <= 4


def test_exact_pairs_satisfy_exact_definition() -> None:
    pairs = _pairs()
    fingerprints = pd.read_parquet(AUDIT / "fingerprints.parquet").set_index("observation_id")
    for row in pairs[pairs["exact_match"]].itertuples():
        train = fingerprints.loc[row.train_observation_id]
        test = fingerprints.loc[row.test_observation_id]
        decoded = (
            train.decoded_rgb_sha256 == test.decoded_rgb_sha256
            and train.decoded_depth_sha256 == test.decoded_depth_sha256
        )
        canonical = (
            train.canonical_rgb_sha256 == test.canonical_rgb_sha256
            and train.canonical_valid_sha256 == test.canonical_valid_sha256
            and train.canonical_depth_sha256 == test.canonical_depth_sha256
        )
        assert decoded or canonical


def test_strict_pairs_satisfy_all_strict_conditions() -> None:
    strict = _pairs().query("strict_match")
    for row in strict.itertuples():
        if row.rgb_only_match:
            assert row.phash_hamming <= 2
            assert row.dhash_hamming <= 3
            assert row.translation_magnitude_px <= 2
            assert row.luminance_ssim >= 0.995
            assert row.normalised_rgb_mae <= 0.010
        else:
            assert row.phash_hamming <= 4
            assert row.dhash_hamming <= 6
            assert row.translation_magnitude_px <= 4
            assert row.luminance_ssim >= 0.990
            assert row.normalised_rgb_mae <= 0.015
            assert row.valid_depth_overlap >= 0.95
            assert row.median_relative_depth_error <= 0.005


def test_moderate_pairs_satisfy_all_moderate_conditions() -> None:
    moderate = _pairs().query("moderate_match")
    for row in moderate.itertuples():
        if row.rgb_only_match:
            assert row.phash_hamming <= 4
            assert row.dhash_hamming <= 6
            assert row.translation_magnitude_px <= 4
            assert row.luminance_ssim >= 0.990
            assert row.normalised_rgb_mae <= 0.015
        else:
            assert row.phash_hamming <= 8
            assert row.dhash_hamming <= 10
            assert row.translation_magnitude_px <= 8
            assert row.luminance_ssim >= 0.980
            assert row.normalised_rgb_mae <= 0.030
            assert row.valid_depth_overlap >= 0.90
            assert row.median_relative_depth_error <= 0.015


def test_strict_is_subset_of_moderate_or_exact_union() -> None:
    pairs = _pairs()
    assert not (pairs["strict_match"] & ~(pairs["moderate_match"] | pairs["exact_match"])).any()
    exact = exclusion_tuple_ids(AUDIT / "test_observation_exclusion_exact.csv")
    strict = exclusion_tuple_ids(AUDIT / "test_observation_exclusion_strict.csv")
    moderate = exclusion_tuple_ids(AUDIT / "test_observation_exclusion_moderate.csv")
    assert exact <= strict <= moderate


def test_duplicate_map_lock_is_immutable() -> None:
    lock_path = AUDIT / "DUPLICATE_MAP_LOCK.json"
    before = sha256_file(lock_path)
    with pytest.raises(FileExistsError):
        build_duplicate_map(REPO, RUN, resume=False)
    assert sha256_file(lock_path) == before
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == PREREGISTRATION_SHA256


def test_duplicate_exclusion_removes_all_queries_for_observation() -> None:
    observations = _observations().set_index("observation_id")
    outcomes = _outcomes()
    for tier in ("exact", "strict", "moderate"):
        excluded = outcomes[outcomes[f"excluded_{tier}"]]
        for observation_id in excluded["observation_id"].unique():
            expected = set(json.loads(observations.loc[observation_id, "tuple_ids"]))
            actual = set(excluded.loc[excluded["observation_id"] == observation_id, "sample_id"])
            assert actual == expected


def test_no_unflagged_test_observation_removed() -> None:
    outcomes = _outcomes()
    for tier in ("exact", "strict", "moderate"):
        manifest = pd.read_csv(AUDIT / f"test_observation_exclusion_{tier}.csv")
        flagged = set(manifest.get("test_observation_id", pd.Series(dtype=str)).astype(str))
        actual = set(outcomes.loc[outcomes[f"excluded_{tier}"], "observation_id"].astype(str))
        assert actual == flagged


def test_filtered_prediction_values_equal_source_values() -> None:
    child = _outcomes()
    parent = pd.read_parquet(PARENT)
    parent = parent[parent["formal_cell"].astype(bool)].reset_index(drop=True)
    assert child[list(parent.columns)].reset_index(drop=True).equals(parent)


def test_formal_evaluator_cell_unchanged() -> None:
    outcomes = _outcomes()
    assert set(outcomes["iou_threshold"]) == {0.25}
    assert set(outcomes["angle_threshold_deg"]) == {30}
    assert outcomes["formal_cell"].all()
    completion = json.loads((EVALUATION / "completion.json").read_text(encoding="utf-8"))
    assert completion["source_paired_outcomes_sha256"] == sha256_file(PARENT)


def test_sequence_cluster_bootstrap_deterministic() -> None:
    native = np.asarray([0, 1, 0, 1, 0, 1], bool)
    reranked = np.asarray([1, 1, 0, 1, 1, 0], bool)
    clusters = np.asarray(["a", "a", "b", "b", "c", "c"])
    first = cluster_bootstrap_difference(
        native, reranked, clusters, iterations=1000, seed=BOOTSTRAP_SEED
    )
    second = cluster_bootstrap_difference(
        native, reranked, clusters, iterations=1000, seed=BOOTSTRAP_SEED
    )
    assert first == second


def test_gain_shift_matches_raw_outcomes() -> None:
    outcomes = _outcomes()
    metrics = pd.read_csv(EVALUATION / "full_vs_filtered_metrics.csv")
    for route, frame in outcomes.groupby("route"):
        full_gain = (frame["raw_reranked_correct"].mean() - frame["native_correct"].mean()) * 100
        filtered = frame[~frame["excluded_strict"]]
        filtered_gain = (
            filtered["raw_reranked_correct"].mean() - filtered["native_correct"].mean()
        ) * 100
        row = metrics[
            (metrics["route"] == route)
            & (metrics["analysis_set"] == "route_specific")
            & (metrics["subset"] == "strict_excluded")
        ].iloc[0]
        assert row["raw_gain_shift_pp"] == pytest.approx(filtered_gain - full_gain, abs=1e-12)
