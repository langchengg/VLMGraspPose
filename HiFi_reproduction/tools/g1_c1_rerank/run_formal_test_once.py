#!/usr/bin/env python3
"""Produce label-free formal predictions, then evaluate the locked run once."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (str(REPOSITORY_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from reranking.statistics import (  # noqa: E402
    cluster_bootstrap_difference,
    holm_adjust,
    mcnemar_exact,
)
from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json, atomic_parquet  # noqa: E402
from src.grasping.g1_c1_safe_rerank.calibration import calibration_metrics  # noqa: E402
from src.grasping.g1_c1_safe_rerank.contracts import sha256_file  # noqa: E402
from src.grasping.g1_c1_safe_rerank.evaluation import (  # noqa: E402
    evaluate_selected_ids,
    stable_bootstrap_seed,
)
from src.grasping.g1_c1_safe_rerank.gate import (  # noqa: E402
    apply_expected_gain_gate,
    build_pair_features,
)
from src.grasping.g1_c1_safe_rerank.models import (  # noqa: E402
    manual_method_score,
)
from src.grasping.g1_c1_safe_rerank.pools import (  # noqa: E402
    build_deduplicated_union,
    build_raw_union,
    pool_manifest,
)
from tools.g1_c1_rerank.run_cross_backend import _router_frame  # noqa: E402
from tools.g1_c1_rerank.run_crop_cnn import load_crops  # noqa: E402
from tools.g1_c1_rerank.run_local_matrix import (  # noqa: E402
    BACKENDS,
    FEATURE_FAMILIES,
    MANUAL_METHODS,
    METHODS,
    POOLS,
    SEEDS,
    _attach_calibration,
    _load_features,
    _selections,
    _universe,
)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "evaluate"))
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and most other Unix variants report KiB.
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _verify_artifact_list(artifacts: Sequence[dict[str, Any]], *, context: str) -> None:
    if not artifacts:
        raise PermissionError(f"{context} artifact inventory is empty")
    for artifact in artifacts:
        path_value = artifact.get("path", artifact.get("label_path"))
        sha_value = artifact.get("sha256", artifact.get("label_sha256"))
        if path_value is None or sha_value is None:
            raise PermissionError(f"{context} artifact entry lacks path/hash")
        path = Path(str(path_value)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(sha_value):
            raise PermissionError(f"{context} artifact drift: {path}")


def _verify_artifact(artifact: dict[str, Any], *, context: str) -> None:
    _verify_artifact_list([artifact], context=context)


def _claim_json(path: Path, payload: dict[str, Any]) -> str:
    """Atomically create a one-owner transaction and return its initial SHA."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"transaction already claimed: {path}") from error
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def _lock(run: Path, base: Path) -> dict[str, Any]:
    path = run / "08_lock" / "PRIMARY_METHOD_LOCK.json"
    if not path.is_file():
        raise PermissionError("formal primary lock is missing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED":
        raise PermissionError("formal primary lock is not LOCKED")
    if Path(str(lock.get("base_run", ""))).expanduser().resolve() != base.resolve():
        raise PermissionError("formal base run differs from the locked source run")
    for section in (
        "locked_models",
        "candidate_artifacts",
        "feature_artifacts",
        "code_artifacts",
        "validation_selection_artifacts",
        "source_manifest_artifacts",
        "source_label_artifacts",
        "audit_artifacts",
        "ranker_calibrators",
        "lock_support_artifacts",
    ):
        _verify_artifact_list(lock.get(section, []), context=f"formal lock/{section}")
    _verify_artifact(lock["prediction_plan"], context="formal prediction plan")
    _verify_artifact(lock["evaluator"], context="formal evaluator")
    mirror_path = run / "08_lock" / "FORMAL_TEST_LOCK.json"
    if not mirror_path.is_file() or sha256_file(mirror_path) != sha256_file(path):
        raise PermissionError("FORMAL_TEST_LOCK mirror differs from PRIMARY_METHOD_LOCK")
    return lock


def _test_frame(run: Path, backend: str, pool: str) -> pd.DataFrame:
    return _attach_calibration(run, _load_features(run, backend, "test", pool), backend, "test")


def _write_matrix_predictions(run: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for pool in POOLS:
            frame = _test_frame(run, backend, pool)
            for method in METHODS:
                parts: list[pd.DataFrame] = []
                latencies = []
                for seed in SEEDS:
                    model = joblib.load(run / "05_models" / backend / pool / f"{method}_seed{seed}.joblib")
                    started = time.perf_counter()
                    scores = model.predict_scores(frame)
                    latencies.append(time.perf_counter() - started)
                    part = frame[["sample_id", "stable_candidate_id", "candidate_identity_sha256"]].copy()
                    part[f"score_seed{seed}"] = scores
                    parts.append(part)
                ensemble = parts[0]
                for part in parts[1:]:
                    ensemble = ensemble.merge(
                        part.drop(columns=["candidate_identity_sha256"]),
                        on=["sample_id", "stable_candidate_id"],
                        validate="one_to_one",
                    )
                score_columns = [f"score_seed{seed}" for seed in SEEDS]
                ensemble["reranker_score"] = ensemble[score_columns].mean(axis=1)
                votes = np.zeros(len(ensemble), dtype=float)
                for column in score_columns:
                    selected = ensemble.sort_values(["sample_id", column, "stable_candidate_id"], ascending=[True, False, True], kind="mergesort").groupby("sample_id", sort=False).head(1).index
                    votes[selected] += 1.0
                ensemble["seed_agreement_fraction"] = votes / len(SEEDS)
                destination = run / "09_formal_test" / "candidate_scores" / backend / pool / f"{method}_ensemble.parquet"
                atomic_parquet(destination, ensemble)
                records.append(
                    {
                        "track": "intra_backend",
                        "backend": backend.upper(),
                        "pool": pool,
                        "method": method,
                        "candidate_rows": len(ensemble),
                        "nonempty_samples": int(ensemble["sample_id"].nunique()),
                        "latency_seconds_total_three_seeds": float(sum(latencies)),
                        "latency_ms_per_nonempty_sample": 1000.0 * sum(latencies) / max(ensemble["sample_id"].nunique(), 1),
                        "peak_process_rss_mb": _peak_rss_mb(),
                        "artifact_path": str(destination),
                        "artifact_sha256": sha256_file(destination),
                    }
                )
            crop = load_crops(run, backend, "test", frame)
            parts = []
            latencies = []
            for seed in SEEDS:
                model = joblib.load(
                    run
                    / "05_models"
                    / backend
                    / pool
                    / f"r13_crop_cnn_seed{seed}.joblib"
                )
                started = time.perf_counter()
                values = model.predict_scores(frame, crop)
                latencies.append(time.perf_counter() - started)
                part = frame[
                    ["sample_id", "stable_candidate_id", "candidate_identity_sha256"]
                ].copy()
                part[f"score_seed{seed}"] = values
                parts.append(part)
            ensemble = parts[0]
            for part in parts[1:]:
                ensemble = ensemble.merge(
                    part.drop(columns=["candidate_identity_sha256"]),
                    on=["sample_id", "stable_candidate_id"],
                    validate="one_to_one",
                )
            ensemble["reranker_score"] = ensemble[
                [f"score_seed{seed}" for seed in SEEDS]
            ].mean(axis=1)
            crop_score_columns = [f"score_seed{seed}" for seed in SEEDS]
            votes = np.zeros(len(ensemble), dtype=float)
            for column in crop_score_columns:
                selected_index = (
                    ensemble.sort_values(
                        ["sample_id", column, "stable_candidate_id"],
                        ascending=[True, False, True],
                        kind="mergesort",
                    )
                    .groupby("sample_id", sort=False)
                    .head(1)
                    .index
                )
                votes[selected_index] += 1.0
            ensemble["seed_agreement_fraction"] = votes / len(SEEDS)
            destination = (
                run
                / "09_formal_test"
                / "candidate_scores"
                / backend
                / pool
                / "r13_crop_cnn_ensemble.parquet"
            )
            atomic_parquet(destination, ensemble)
            records.append(
                {
                    "track": "intra_backend",
                    "backend": backend.upper(),
                    "pool": pool,
                    "method": "r13_crop_cnn",
                    "candidate_rows": len(ensemble),
                    "nonempty_samples": int(ensemble["sample_id"].nunique()),
                    "latency_seconds_total_three_seeds": float(sum(latencies)),
                    "latency_ms_per_nonempty_sample": 1000.0
                    * sum(latencies)
                    / max(ensemble["sample_id"].nunique(), 1),
                    "peak_process_rss_mb": _peak_rss_mb(),
                    "artifact_path": str(destination),
                    "artifact_sha256": sha256_file(destination),
                }
            )
    return {"records": records}


def _write_manual_predictions(run: Path, lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay Train-selected R1 utilities label-free on formal candidates."""

    records: list[dict[str, Any]] = []
    plan = json.loads(Path(str(lock["prediction_plan"]["path"])).read_text(encoding="utf-8"))
    for backend in BACKENDS:
        for pool in POOLS:
            frame = _test_frame(run, backend, pool)
            base_probability = np.clip(
                frame["source_score_calibrated"].to_numpy(dtype=float), 1e-4, 1 - 1e-4
            )
            base_logit = np.log(base_probability / (1.0 - base_probability))
            for method, groups in MANUAL_METHODS.items():
                alpha = float(plan["manual_alphas"][backend][pool][method])
                started = time.perf_counter()
                evidence = manual_method_score(
                    frame, groups, FEATURE_FAMILIES["F2"]
                )
                scored = frame[
                    ["sample_id", "stable_candidate_id", "candidate_identity_sha256"]
                ].copy()
                scored["reranker_score"] = base_logit + alpha * evidence
                latency = time.perf_counter() - started
                destination = (
                    run
                    / "09_formal_test"
                    / "candidate_scores"
                    / backend
                    / pool
                    / f"{method}_fixed.parquet"
                )
                atomic_parquet(destination, scored)
                records.append(
                    {
                        "track": "intra_backend_manual",
                        "backend": backend.upper(),
                        "pool": pool,
                        "method": method,
                        "alpha": alpha,
                        "candidate_rows": int(len(scored)),
                        "nonempty_samples": int(scored["sample_id"].nunique()),
                        "latency_seconds": float(latency),
                        "latency_ms_per_nonempty_sample": 1000.0
                        * latency
                        / max(scored["sample_id"].nunique(), 1),
                        "artifact_path": str(destination),
                        "artifact_sha256": sha256_file(destination),
                    }
                )
    return records


def _write_primary_gates(run: Path, lock: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for backend in BACKENDS:
        selected = lock["selected_methods"]["backend"][backend.upper()]
        method = str(selected["primary_ungated_method"])
        pool = str(selected["primary_pool"])
        gate_kind = str(selected["primary_gate_kind"])
        frame = _test_frame(run, backend, pool)
        scores = pd.read_parquet(run / "09_formal_test" / "candidate_scores" / backend / pool / f"{method}_ensemble.parquet")
        scored = frame.merge(
            scores[["sample_id", "stable_candidate_id", "reranker_score", "seed_agreement_fraction"]],
            on=["sample_id", "stable_candidate_id"],
            validate="one_to_one",
        )
        with (run / "04_calibration" / "ranker" / backend / pool / method / "calibrator.pkl").open("rb") as stream:
            calibrator = pickle.load(stream)
        scored["ranker_probability"] = np.clip(calibrator.predict(scored["reranker_score"]), 1e-4, 1 - 1e-4)
        pairs = build_pair_features(scored, include_labels=False)
        gate = joblib.load(run / "07_validation" / "gates" / backend / pool / method / gate_kind / "gate.joblib")
        predicted = gate.predict(pairs)
        operating = selected["primary_gate_operating_point"]
        if bool(selected["gate_deployed"]):
            decisions = apply_expected_gain_gate(
                predicted,
                lambda_h=float(operating["lambda_h"]),
                tau_u=float(operating["tau_u"]),
                tau_margin=float(operating["tau_margin"]),
                tau_reliability=float(operating["tau_reliability"]),
            )
        else:
            decisions = predicted.copy()
            decisions["switch"] = False
            decisions["selected_candidate_id"] = decisions["baseline_candidate_id"]
            decisions["fallback"] = True
        destination = run / "09_formal_test" / "primary" / backend / "gated_selections.parquet"
        atomic_parquet(destination, decisions)
        records.append({"backend": backend.upper(), "method": method, "pool": pool, "gate_kind": gate_kind, "gate_deployed": bool(selected["gate_deployed"]), "selections_path": str(destination), "sha256": sha256_file(destination)})
    return records


def _label_free_union(run: Path, backend_frames: dict[str, pd.DataFrame], pool: str, *, deduplicate: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_only = {}
    for backend in BACKENDS:
        candidate_only[backend] = pd.read_parquet(run / "data" / f"frozen_{backend}_test_candidates.parquet")
        if pool == "top5":
            candidate_only[backend] = candidate_only[backend].loc[candidate_only[backend]["original_rank"].le(5)].copy()
        candidate_only[backend] = candidate_only[backend].merge(
            backend_frames[backend][["sample_id", "stable_candidate_id", "source_score_calibrated"]],
            on=["sample_id", "stable_candidate_id"], validate="one_to_one"
        )
    raw = build_raw_union(candidate_only["g1"], candidate_only["c1"])
    union = build_deduplicated_union(raw) if deduplicate else raw.copy()
    if not deduplicate:
        union = union.sort_values(["sample_id", "source_score_calibrated", "stable_candidate_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
        union["pool_rank"] = union.groupby("sample_id", sort=False).cumcount() + 1
        union["pool_score"] = union["source_score_calibrated"]
    keep = set(zip(union["sample_id"].astype(str), union["stable_candidate_id"].astype(str)))
    features = pd.concat([backend_frames["g1"].assign(backend_g1=1.0), backend_frames["c1"].assign(backend_g1=0.0)], ignore_index=True)
    keys = list(zip(features["sample_id"].astype(str), features["stable_candidate_id"].astype(str)))
    features = features.loc[[key in keep for key in keys]].copy()
    features = features.drop(columns=["pool_rank", "pool_score"], errors="ignore").merge(
        union[["sample_id", "stable_candidate_id", "pool_rank", "pool_score"]],
        on=["sample_id", "stable_candidate_id"], validate="one_to_one"
    )
    return features.reset_index(drop=True), union


def _write_cross_predictions(run: Path, base: Path, lock: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    allnms = {backend: _test_frame(run, backend, "allnms") for backend in BACKENDS}
    router_frame = _router_frame(allnms["g1"], allnms["c1"], _universe(base, "test"), include_labels=False)
    router_lock = lock["selected_methods"]["cross_backend"]["router"]
    router_kind = str(router_lock["kind"])
    router = joblib.load(run / "07_validation" / "cross_backend" / "router" / router_kind / "model.joblib")
    predicted = router.predict(router_frame)
    operating = router_lock["safe_lcb"]
    utility = predicted["p_c1_only"] - float(operating["lambda_h"]) * predicted["p_g1_only"]
    switch = (utility > float(operating["tau"])) & predicted["c1_nonempty"].astype(bool) if bool(router_lock["deploy"]) else np.zeros(len(predicted), dtype=bool)
    predicted["switch"] = switch
    predicted["selected_candidate_id"] = np.where(switch, predicted["c1_candidate_id"], predicted["g1_candidate_id"])
    router_path = run / "09_formal_test" / "cross_backend" / "router_selections.parquet"
    atomic_parquet(router_path, predicted)
    records.append({"track": "router", "method": router_kind, "artifact_path": str(router_path), "artifact_sha256": sha256_file(router_path)})
    for pool in POOLS:
        backend_frames = {backend: _test_frame(run, backend, pool) for backend in BACKENDS}
        for deduplicate, union_name in ((False, "union_concat"), (True, "union_nms")):
            frame, candidate_pool = _label_free_union(run, backend_frames, pool, deduplicate=deduplicate)
            root = run / "09_formal_test" / "cross_backend" / union_name / pool
            atomic_parquet(root / "candidate_pool.parquet", candidate_pool)
            atomic_json(root / "pool_manifest.json", pool_manifest(candidate_pool))
            for method in ("union_mlp", "union_deepsets", "union_gnn"):
                parts = []
                for seed in SEEDS:
                    model = joblib.load(run / "05_models" / "cross_backend" / union_name / pool / method / f"seed{seed}.joblib")
                    part = frame[["sample_id", "stable_candidate_id", "candidate_identity_sha256"]].copy()
                    part[f"score_seed{seed}"] = model.predict_scores(frame)
                    parts.append(part)
                ensemble = parts[0]
                for part in parts[1:]:
                    ensemble = ensemble.merge(part.drop(columns=["candidate_identity_sha256"]), on=["sample_id", "stable_candidate_id"], validate="one_to_one")
                ensemble["reranker_score"] = ensemble[[f"score_seed{seed}" for seed in SEEDS]].mean(axis=1)
                path = root / method / "scores_ensemble.parquet"
                atomic_parquet(path, ensemble)
                records.append({"track": union_name, "pool": pool, "method": method, "artifact_path": str(path), "artifact_sha256": sha256_file(path)})
    return records


def _write_pooled_predictions(run: Path) -> list[dict[str, Any]]:
    records = []
    for pool in POOLS:
        frames = []
        for backend in BACKENDS:
            frame = _test_frame(run, backend, pool)
            frame["backend_g1"] = float(backend == "g1")
            frame["sample_id_original"] = frame["sample_id"].astype(str)
            frame["sample_id"] = backend + ":" + frame["sample_id"].astype(str)
            frame["scene_id"] = backend + ":" + frame["scene_id"].astype(str)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        parts = []
        for seed in SEEDS:
            model = joblib.load(run / "05_models" / "pooled" / pool / f"seed{seed}.joblib")
            part = combined[["sample_id", "sample_id_original", "stable_candidate_id", "backend", "candidate_identity_sha256"]].copy()
            part[f"score_seed{seed}"] = model.predict_scores(combined)
            parts.append(part)
        ensemble = parts[0]
        for part in parts[1:]:
            ensemble = ensemble.merge(part.drop(columns=["sample_id_original", "backend", "candidate_identity_sha256"]), on=["sample_id", "stable_candidate_id"], validate="one_to_one")
        ensemble["reranker_score"] = ensemble[[f"score_seed{seed}" for seed in SEEDS]].mean(axis=1)
        path = run / "09_formal_test" / "pooled" / pool / "scores_ensemble.parquet"
        atomic_parquet(path, ensemble)
        records.append({"track": "r14_pooled", "pool": pool, "artifact_path": str(path), "artifact_sha256": sha256_file(path)})
    return records


def predict(run: Path, base: Path) -> None:
    lock = _lock(run, base)
    marker = run / "09_formal_test" / "PREDICTIONS_COMPLETE.json"
    if (run / "09_formal_test" / "TEST_ACCESS_TRANSACTION.json").exists():
        raise PermissionError("formal predictions cannot run after Test-label access")
    if marker.exists():
        raise FileExistsError(f"formal predictions already completed: {marker}")
    matrix = _write_matrix_predictions(run)
    manual = _write_manual_predictions(run, lock)
    gate_records = _write_primary_gates(run, lock)
    pooled = _write_pooled_predictions(run)
    cross = _write_cross_predictions(run, base, lock)
    prediction_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in sorted((run / "09_formal_test").rglob("*"))
        if path.is_file() and path != marker
    ]
    atomic_json(marker, {
        "status": "COMPLETE",
        "test_labels_accessed": False,
        "primary_lock_sha256": sha256_file(run / "08_lock" / "PRIMARY_METHOD_LOCK.json"),
        "base_run": str(base.resolve()),
        "test_universe_sha256": sha256_file(base / "manifests" / "test_samples.parquet"),
        "prediction_artifacts": prediction_artifacts,
        "matrix": matrix,
        "manual": manual,
        "primary_gates": gate_records,
        "pooled": pooled,
        "cross_backend": cross,
    })


def _joined_test(run: Path, backend: str, pool: str) -> pd.DataFrame:
    features = _test_frame(run, backend, pool)
    labels = pd.read_parquet(run / "09_formal_test" / f"{backend}_candidate_labels.parquet")
    output = features.merge(labels[["sample_id", "stable_candidate_id", "candidate_correct"]], on=["sample_id", "stable_candidate_id"], validate="one_to_one")
    if len(output) != len(features):
        raise AssertionError("formal feature/label coverage mismatch")
    return output


def _rank_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    universe: pd.DataFrame,
    method: str,
    *,
    baseline_ids: pd.DataFrame | None = None,
    probabilities: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored = frame.copy()
    scored["reranker_score"] = np.asarray(scores, dtype=float)
    ordered = scored.sort_values(["sample_id", "reranker_score", "stable_candidate_id"], ascending=[True, False, True], kind="mergesort").copy()
    ordered["reranker_rank"] = ordered.groupby("sample_id", sort=False).cumcount() + 1
    selected = ordered.groupby("sample_id", sort=False).head(1)
    if baseline_ids is None:
        baseline = scored.sort_values(["sample_id", "original_rank", "stable_candidate_id"], kind="mergesort").groupby("sample_id", sort=False).head(1)
        baseline_ids = baseline[["sample_id", "stable_candidate_id"]].rename(columns={"stable_candidate_id": "baseline_candidate_id"})
    selections = baseline_ids.merge(selected[["sample_id", "stable_candidate_id"]].rename(columns={"stable_candidate_id": "selected_candidate_id"}), on="sample_id", how="outer", validate="one_to_one")
    selections["baseline_candidate_id"] = selections["baseline_candidate_id"].fillna("")
    selections["selected_candidate_id"] = selections["selected_candidate_id"].fillna(selections["baseline_candidate_id"])
    selections["fallback"] = False
    outcomes, metrics = evaluate_selected_ids(selections, frame[["sample_id", "stable_candidate_id", "candidate_correct"]], universe, method=method)
    total = len(universe)
    rerank_top5 = ordered.loc[ordered["reranker_rank"].le(5)].groupby("sample_id")["candidate_correct"].any()
    oracle_rank_column = "pool_rank" if "pool_rank" in frame.columns else "original_rank"
    original_top5 = frame.loc[frame[oracle_rank_column].le(5)].groupby("sample_id")["candidate_correct"].any()
    oracle_all = frame.groupby("sample_id")["candidate_correct"].any()
    first_positive = ordered.loc[ordered["candidate_correct"].astype(bool)].groupby("sample_id")["reranker_rank"].min()
    reciprocal = first_positive.map(lambda rank: 1.0 / float(rank)).sum() / max(total, 1)
    ndcg = 0.0
    for _, group in ordered.groupby("sample_id", sort=False):
        relevance = group.head(5)["candidate_correct"].astype(float).to_numpy()
        discount = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
        dcg = float(np.dot(relevance, discount))
        positives = int(group["candidate_correct"].sum())
        ideal = float(discount[: min(positives, 5)].sum())
        ndcg += 0.0 if ideal == 0 else dcg / ideal
    candidate_label = frame["candidate_correct"].astype(int).to_numpy()
    candidate_score = np.asarray(scores, dtype=float)
    indexed = frame.set_index(["sample_id", "stable_candidate_id"])
    nonempty_outcomes = outcomes.loc[outcomes["selected_candidate_id"].astype(str).ne("")]
    comparable_outcomes = nonempty_outcomes.loc[
        nonempty_outcomes["baseline_candidate_id"].astype(str).ne("")
    ]
    baseline_keys = list(zip(comparable_outcomes["sample_id"].astype(str), comparable_outcomes["baseline_candidate_id"].astype(str)))
    selected_keys = list(zip(comparable_outcomes["sample_id"].astype(str), comparable_outcomes["selected_candidate_id"].astype(str)))
    baseline_rows = indexed.loc[baseline_keys]
    final_rows = indexed.loc[selected_keys]
    original_score_change = final_rows["original_score"].to_numpy(dtype=float) - baseline_rows["original_score"].to_numpy(dtype=float)
    probability_change = final_rows["source_score_calibrated"].to_numpy(dtype=float) - baseline_rows["source_score_calibrated"].to_numpy(dtype=float)
    all_selected_keys = list(zip(nonempty_outcomes["sample_id"].astype(str), nonempty_outcomes["selected_candidate_id"].astype(str)))
    selected_rank = indexed.loc[all_selected_keys]["original_rank"].to_numpy(dtype=float)
    calibration = {"brier": math.nan, "log_loss": math.nan, "ece_15": math.nan}
    if probabilities is not None:
        probability_values = np.clip(np.asarray(probabilities, dtype=float), 1e-4, 1.0 - 1e-4)
        if probability_values.shape != candidate_label.shape:
            raise ValueError("post-hoc probability vector is not candidate-aligned")
        calibration = calibration_metrics(
            probability_values,
            candidate_label,
            sample_ids=frame["sample_id"],
        )
    metrics.update(
        {
            "j_at_5": float(rerank_top5.sum() / max(total, 1)),
            "oracle_at_5": float(original_top5.sum() / max(total, 1)),
            "oracle_at_all": float(oracle_all.sum() / max(total, 1)),
            "mrr": float(reciprocal),
            "ndcg_at_1": float(metrics["j_at_1"]),
            "ndcg_at_5": float(ndcg / max(total, 1)),
            "candidate_pr_auc": float(average_precision_score(candidate_label, candidate_score)) if len(np.unique(candidate_label)) > 1 else math.nan,
            "candidate_roc_auc": float(roc_auc_score(candidate_label, candidate_score)) if len(np.unique(candidate_label)) > 1 else math.nan,
            "candidate_brier": float(calibration["brier"]),
            "candidate_nll": float(calibration["log_loss"]),
            "candidate_ece_15": float(calibration["ece_15"]),
            "posthoc_calibrated": probabilities is not None,
            "selected_original_score_change_mean": float(original_score_change.mean()),
            "selected_source_probability_change_mean": float(probability_change.mean()),
            "selected_original_rank_mean": float(selected_rank.mean()),
            "selected_original_rank_median": float(np.median(selected_rank)),
            "selected_original_rank_p90": float(np.quantile(selected_rank, 0.90)),
            "selected_rank_nonempty_denominator": int(len(selected_rank)),
            "selected_change_comparable_denominator": int(len(comparable_outcomes)),
            "headroom_recovery_at_5": (
                float(metrics["delta_j_at_1"] / (float(original_top5.sum() / max(total, 1)) - float(metrics["baseline_j_at_1"])))
                if float(original_top5.sum() / max(total, 1)) > float(metrics["baseline_j_at_1"])
                else math.nan
            ),
        }
    )
    return outcomes, metrics


def evaluate(run: Path, base: Path) -> None:
    lock = _lock(run, base)
    lock_path = run / "08_lock" / "PRIMARY_METHOD_LOCK.json"
    prediction_marker = run / "09_formal_test" / "PREDICTIONS_COMPLETE.json"
    if not prediction_marker.is_file():
        raise PermissionError("formal predictions are incomplete")
    prediction_state = json.loads(prediction_marker.read_text(encoding="utf-8"))
    if (
        prediction_state.get("status") != "COMPLETE"
        or prediction_state.get("primary_lock_sha256") != sha256_file(lock_path)
        or Path(str(prediction_state.get("base_run", ""))).resolve() != base.resolve()
        or prediction_state.get("test_universe_sha256")
        != sha256_file(base / "manifests" / "test_samples.parquet")
    ):
        raise PermissionError("formal predictions are not bound to the current lock")
    _verify_artifact_list(
        prediction_state.get("prediction_artifacts", []),
        context="formal prediction",
    )
    transaction = run / "09_formal_test" / "TEST_ACCESS_TRANSACTION.json"
    transaction_state = (
        json.loads(transaction.read_text(encoding="utf-8"))
        if transaction.is_file()
        else {}
    )
    if transaction_state.get("status") != "CONSUMED":
        raise PermissionError("formal labels were not opened through the one-time transaction")
    if (
        transaction_state.get("primary_lock_sha256") != sha256_file(lock_path)
        or transaction_state.get("predictions_complete_sha256")
        != sha256_file(prediction_marker)
        or Path(str(transaction_state.get("base_run", ""))).resolve()
        != base.resolve()
        or transaction_state.get("test_universe_sha256")
        != sha256_file(base / "manifests" / "test_samples.parquet")
    ):
        raise PermissionError("formal label transaction is not bound to lock/predictions")
    _verify_artifact_list(
        transaction_state.get("label_artifacts", []),
        context="formal label",
    )
    marker = run / "09_formal_test" / "EVALUATION_COMPLETE.json"
    if marker.exists():
        raise RuntimeError("formal evaluation already completed")
    evaluation_transaction = run / "09_formal_test" / "EVALUATION_TRANSACTION.json"
    evaluation_claim_sha = _claim_json(
        evaluation_transaction,
        {
            "status": "OPENING",
            "primary_lock_sha256": sha256_file(lock_path),
            "predictions_complete_sha256": sha256_file(prediction_marker),
            "test_access_transaction_sha256": sha256_file(transaction),
            "base_run": str(base.resolve()),
            "test_universe_sha256": sha256_file(base / "manifests" / "test_samples.parquet"),
        },
    )
    universe = _universe(base, "test")
    result_rows: list[dict[str, Any]] = []
    outcome_index: dict[str, pd.DataFrame] = {}
    audit_inventory = json.loads((run / "AUDIT_INVENTORY.json").read_text(encoding="utf-8"))
    if audit_inventory.get("baseline_regression_match") is not True:
        raise AssertionError("preformal source baseline regression audit is not PASS")
    baseline_regression: dict[str, Any] = {"status": "PASS", "backend": {}}
    for backend in BACKENDS:
        for pool in POOLS:
            frame = _joined_test(run, backend, pool)
            baseline_outcomes, baseline_metric = _rank_metrics(
                frame,
                frame["original_score"].to_numpy(dtype=float),
                universe,
                f"{backend}_{pool}_baseline",
                probabilities=frame["source_score_calibrated"].to_numpy(dtype=float),
            )
            baseline_metric["j_at_5"] = float(
                frame.loc[frame["original_rank"].le(5)]
                .groupby("sample_id")["candidate_correct"]
                .any()
                .sum()
                / len(universe)
            )
            baseline_metric.update({"track": "intra_backend", "backend": backend.upper(), "pool": pool, "method": "r0_baseline", "seed": "fixed"})
            result_rows.append(baseline_metric)
            atomic_parquet(run / "09_formal_test" / "outcomes" / backend / pool / "r0_baseline.parquet", baseline_outcomes)
            if pool == "allnms":
                expected = audit_inventory["source_baselines"][backend.upper()]
                observed = {
                    "candidate_rows": int(len(frame)),
                    "non_empty_rate": float(frame["sample_id"].nunique() / len(universe)),
                    "j_at_1": float(baseline_metric["j_at_1"]),
                    "j_at_5": float(baseline_metric["j_at_5"]),
                }
                exact_counts = observed["candidate_rows"] == int(expected["candidate_rows"])
                close_metrics = all(
                    math.isclose(float(observed[key]), float(expected[key]), rel_tol=0.0, abs_tol=1e-12)
                    for key in ("non_empty_rate", "j_at_1", "j_at_5")
                )
                if not exact_counts or not close_metrics:
                    raise AssertionError(
                        f"{backend.upper()} formal baseline regression mismatch: "
                        f"observed={observed}, expected={expected}"
                    )
                baseline_regression["backend"][backend.upper()] = {
                    "status": "PASS",
                    "observed": observed,
                    "expected": expected,
                }
            for method in MANUAL_METHODS:
                scored = pd.read_parquet(
                    run
                    / "09_formal_test"
                    / "candidate_scores"
                    / backend
                    / pool
                    / f"{method}_fixed.parquet"
                )
                aligned = frame[["sample_id", "stable_candidate_id"]].merge(
                    scored[["sample_id", "stable_candidate_id", "reranker_score"]],
                    on=["sample_id", "stable_candidate_id"],
                    validate="one_to_one",
                )["reranker_score"].to_numpy(dtype=float)
                outcomes, metric = _rank_metrics(frame, aligned, universe, method)
                metric.update(
                    {
                        "track": "intra_backend_manual",
                        "backend": backend.upper(),
                        "pool": pool,
                        "method": method,
                        "seed": "fixed",
                    }
                )
                result_rows.append(metric)
                atomic_parquet(
                    run
                    / "09_formal_test"
                    / "outcomes"
                    / backend
                    / pool
                    / f"{method}_fixed.parquet",
                    outcomes,
                )
                outcome_index[f"{backend}/{pool}/{method}"] = outcomes
            for method in [*METHODS, "r13_crop_cnn"]:
                scored = pd.read_parquet(run / "09_formal_test" / "candidate_scores" / backend / pool / f"{method}_ensemble.parquet")
                calibrator_path = run / "04_calibration" / "ranker" / backend / pool / method / "calibrator.pkl"
                calibrator = None
                locked_calibrators = {
                    Path(str(item["path"])).expanduser().resolve()
                    for item in lock["ranker_calibrators"]
                }
                if calibrator_path.resolve() in locked_calibrators:
                    with calibrator_path.open("rb") as stream:
                        calibrator = pickle.load(stream)
                elif calibrator_path.is_file():
                    raise PermissionError(f"formal evaluator refused unlocked calibrator: {calibrator_path}")
                for seed_name, score_column in [
                    *[(str(seed), f"score_seed{seed}") for seed in SEEDS],
                    ("ensemble", "reranker_score"),
                ]:
                    aligned = frame[["sample_id", "stable_candidate_id"]].merge(
                        scored[["sample_id", "stable_candidate_id", score_column]],
                        on=["sample_id", "stable_candidate_id"],
                        validate="one_to_one",
                    )[score_column].to_numpy(dtype=float)
                    probability = None if calibrator is None else calibrator.predict(aligned)
                    outcomes, metric = _rank_metrics(
                        frame,
                        aligned,
                        universe,
                        f"{method}_{seed_name}",
                        probabilities=probability,
                    )
                    metric.update({"track": "intra_backend", "backend": backend.upper(), "pool": pool, "method": method, "seed": seed_name})
                    result_rows.append(metric)
                    atomic_parquet(run / "09_formal_test" / "outcomes" / backend / pool / f"{method}_{seed_name}.parquet", outcomes)
                    if seed_name == "ensemble":
                        outcome_index[f"{backend}/{pool}/{method}"] = outcomes
        selected = lock["selected_methods"]["backend"][backend.upper()]
        pool = str(selected["primary_pool"])
        frame = _joined_test(run, backend, pool)
        decisions = pd.read_parquet(run / "09_formal_test" / "primary" / backend / "gated_selections.parquet")
        selections = decisions[["sample_id", "baseline_candidate_id", "selected_candidate_id", "fallback"]]
        primary_scores = pd.read_parquet(
            run / "09_formal_test" / "candidate_scores" / backend / pool / f"{selected['primary_ungated_method']}_ensemble.parquet"
        )
        gated_score_frame = frame[["sample_id", "stable_candidate_id"]].merge(
            primary_scores[["sample_id", "stable_candidate_id", "reranker_score"]],
            on=["sample_id", "stable_candidate_id"], validate="one_to_one"
        ).merge(
            selections[["sample_id", "selected_candidate_id"]], on="sample_id", validate="many_to_one"
        )
        maximum = gated_score_frame.groupby("sample_id")["reranker_score"].transform("max")
        gated_scores = gated_score_frame["reranker_score"].to_numpy(dtype=float).copy()
        chosen = gated_score_frame["stable_candidate_id"].astype(str).eq(gated_score_frame["selected_candidate_id"].astype(str)).to_numpy()
        gated_scores[chosen] = maximum.to_numpy(dtype=float)[chosen] + 1.0
        with (run / "04_calibration" / "ranker" / backend / pool / str(selected["primary_ungated_method"]) / "calibrator.pkl").open("rb") as stream:
            primary_calibrator = pickle.load(stream)
        gated_outcomes, gated_metric = _rank_metrics(
            frame,
            gated_scores,
            universe,
            "primary_gated",
            probabilities=primary_calibrator.predict(gated_score_frame["reranker_score"].to_numpy(dtype=float)),
        )
        locked_selected = selections.set_index("sample_id")["selected_candidate_id"].astype(str)
        observed_selected = gated_outcomes.set_index("sample_id")["selected_candidate_id"].astype(str)
        if observed_selected.loc[locked_selected.index].tolist() != locked_selected.tolist():
            raise AssertionError("gated full-ranking reconstruction changed locked Top-1 selections")
        empty_outcomes = observed_selected.loc[~observed_selected.index.isin(locked_selected.index)]
        if not empty_outcomes.eq("").all():
            raise AssertionError("empty-candidate samples acquired a gated selection")
        gated_metric.update({"track": "intra_backend", "backend": backend.upper(), "pool": pool, "method": "primary_gated", "base_method": selected["primary_ungated_method"], "seed": "ensemble"})
        result_rows.append(gated_metric)
        atomic_parquet(run / "09_formal_test" / "outcomes" / backend / pool / "primary_gated.parquet", gated_outcomes)
        outcome_index[f"{backend}/{pool}/primary_gated"] = gated_outcomes

    # R14 pooled backend-conditioned model remains an intra-backend order-only
    # comparison even though its training set is shared.
    for pool in POOLS:
        scores = pd.read_parquet(run / "09_formal_test" / "pooled" / pool / "scores_ensemble.parquet")
        for backend in BACKENDS:
            frame = _joined_test(run, backend, pool)
            local = scores.loc[scores["backend"].astype(str).str.lower().eq(backend)].copy()
            local["sample_id"] = local["sample_id_original"].astype(str)
            for seed_name, score_column in [
                *[(str(seed), f"score_seed{seed}") for seed in SEEDS],
                ("ensemble", "reranker_score"),
            ]:
                aligned = frame[["sample_id", "stable_candidate_id"]].merge(
                    local[["sample_id", "stable_candidate_id", score_column]],
                    on=["sample_id", "stable_candidate_id"], validate="one_to_one"
                )[score_column].to_numpy(dtype=float)
                outcomes, metric = _rank_metrics(
                    frame, aligned, universe, f"r14_pooled_backend_conditioned_{seed_name}"
                )
                metric.update({"track": "pooled_backend_conditioned", "backend": backend.upper(), "pool": pool, "method": "r14_pooled_backend_conditioned", "seed": seed_name})
                result_rows.append(metric)
                atomic_parquet(
                    run / "09_formal_test" / "outcomes" / "pooled" / pool / f"{backend}_{seed_name}.parquet",
                    outcomes,
                )
                if seed_name == "ensemble":
                    outcome_index[f"{backend}/{pool}/r14_pooled_backend_conditioned"] = outcomes

    cross_rows: list[dict[str, Any]] = []
    # Router baseline is always G1 Top-1, as preregistered.
    router = pd.read_parquet(run / "09_formal_test" / "cross_backend" / "router_selections.parquet")
    g1_all = _joined_test(run, "g1", "allnms")
    c1_all = _joined_test(run, "c1", "allnms")
    union_labels_all = pd.concat(
        [
            g1_all[["sample_id", "stable_candidate_id", "candidate_correct"]],
            c1_all[["sample_id", "stable_candidate_id", "candidate_correct"]],
        ],
        ignore_index=True,
    )
    router_selections = router[["sample_id", "g1_candidate_id", "selected_candidate_id", "switch"]].rename(columns={"g1_candidate_id": "baseline_candidate_id"})
    # Empty G1 is an incorrect always-G1 baseline, represented explicitly by
    # the empty ID; never backfill it with the selected C1 candidate.
    router_selections["baseline_candidate_id"] = router_selections["baseline_candidate_id"].fillna("")
    router_selections["selected_candidate_id"] = router_selections["selected_candidate_id"].fillna(router_selections["baseline_candidate_id"])
    router_selections = router_selections.loc[
        router["g1_nonempty"].astype(bool) | router["c1_nonempty"].astype(bool)
    ].copy()
    router_selections["fallback"] = ~router_selections["switch"].astype(bool)
    router_outcomes, router_metric = evaluate_selected_ids(
        router_selections,
        union_labels_all,
        universe,
        method="backend_top1_router",
        allow_empty_selected=True,
    )
    router_metric.update({"track": "cross_backend_router", "backend": "G1+C1", "pool": "top1_pair", "method": "backend_top1_router"})
    cross_rows.append(router_metric)
    atomic_parquet(run / "09_formal_test" / "outcomes" / "cross_backend" / "router.parquet", router_outcomes)
    outcome_index["g1+c1/top1_pair/backend_top1_router"] = router_outcomes

    for pool in POOLS:
        joined_backend = {backend: _joined_test(run, backend, pool) for backend in BACKENDS}
        for union_name in ("union_concat", "union_nms"):
            candidate_pool = pd.read_parquet(run / "09_formal_test" / "cross_backend" / union_name / pool / "candidate_pool.parquet")
            keep = set(zip(candidate_pool["sample_id"].astype(str), candidate_pool["stable_candidate_id"].astype(str)))
            frame = pd.concat([joined_backend["g1"].assign(backend_g1=1.0), joined_backend["c1"].assign(backend_g1=0.0)], ignore_index=True)
            keys = list(zip(frame["sample_id"].astype(str), frame["stable_candidate_id"].astype(str)))
            frame = frame.loc[[key in keep for key in keys]].copy().drop(columns=["pool_rank", "pool_score"], errors="ignore")
            frame = frame.merge(candidate_pool[["sample_id", "stable_candidate_id", "pool_rank", "pool_score"]], on=["sample_id", "stable_candidate_id"], validate="one_to_one")
            baseline_ids = joined_backend["g1"].loc[joined_backend["g1"]["original_rank"].eq(1), ["sample_id", "stable_candidate_id"]].rename(columns={"stable_candidate_id": "baseline_candidate_id"})
            for method in ("union_mlp", "union_deepsets", "union_gnn"):
                score = pd.read_parquet(run / "09_formal_test" / "cross_backend" / union_name / pool / method / "scores_ensemble.parquet")
                aligned = frame[["sample_id", "stable_candidate_id"]].merge(score[["sample_id", "stable_candidate_id", "reranker_score"]], on=["sample_id", "stable_candidate_id"], validate="one_to_one")["reranker_score"].to_numpy(dtype=float)
                outcomes, metric = _rank_metrics(frame, aligned, universe, method, baseline_ids=baseline_ids)
                metric.update({"track": union_name, "backend": "G1+C1", "pool": pool, "method": method})
                cross_rows.append(metric)
                atomic_parquet(run / "09_formal_test" / "outcomes" / "cross_backend" / union_name / pool / f"{method}.parquet", outcomes)
                outcome_index[f"g1+c1/{union_name}_{pool}/{method}"] = outcomes
    result_rows.extend(cross_rows)

    expected_outcome_keys = {
        f"{backend}/{pool}/{method}"
        for backend in BACKENDS
        for pool in POOLS
        for method in [
            *MANUAL_METHODS,
            *METHODS,
            "r13_crop_cnn",
            "r14_pooled_backend_conditioned",
        ]
    }
    expected_outcome_keys.update(
        f"{backend}/{lock['selected_methods']['backend'][backend.upper()]['primary_pool']}/primary_gated"
        for backend in BACKENDS
    )
    expected_outcome_keys.add("g1+c1/top1_pair/backend_top1_router")
    expected_outcome_keys.update(
        f"g1+c1/{union_name}_{pool}/{method}"
        for union_name in ("union_concat", "union_nms")
        for pool in POOLS
        for method in ("union_mlp", "union_deepsets", "union_gnn")
    )
    if set(outcome_index) != expected_outcome_keys:
        missing = sorted(expected_outcome_keys - set(outcome_index))
        extra = sorted(set(outcome_index) - expected_outcome_keys)
        raise AssertionError(f"formal outcome inventory mismatch; missing={missing}, extra={extra}")
    plan = json.loads(Path(str(lock["prediction_plan"]["path"])).read_text(encoding="utf-8"))
    expected_family_sizes = plan.get("statistical_families", {})
    if int(expected_family_sizes.get("formal_intra_backend_all_locked_methods", -1)) != sum(
        not key.startswith("g1+c1/") for key in expected_outcome_keys
    ) or int(expected_family_sizes.get("formal_cross_backend_locked_methods", -1)) != sum(
        key.startswith("g1+c1/") for key in expected_outcome_keys
    ):
        raise AssertionError("formal prediction plan has stale statistical-family sizes")

    result = pd.DataFrame(result_rows)
    pool_integrity: dict[str, Any] = {"status": "PASS", "backend": {}}
    for backend in BACKENDS:
        local = result.loc[result["backend"].astype(str).eq(backend.upper())]
        baseline_by_pool = (
            local.loc[local["method"].astype(str).eq("r0_baseline")]
            .set_index("pool")
        )
        top5_expected = float(baseline_by_pool.loc["top5", "j_at_5"])
        top5_values = local.loc[local["pool"].astype(str).eq("top5"), "j_at_5"].astype(float)
        allnms_oracle = float(baseline_by_pool.loc["allnms", "oracle_at_all"])
        allnms_values = local.loc[
            local["pool"].astype(str).eq("allnms"), "oracle_at_all"
        ].astype(float)
        if not np.allclose(top5_values, top5_expected, rtol=0.0, atol=1e-15):
            raise AssertionError(f"{backend.upper()} Top5 J@5 invariance failed")
        if not np.allclose(allnms_values, allnms_oracle, rtol=0.0, atol=1e-15):
            raise AssertionError(f"{backend.upper()} AllNMS oracle invariance failed")
        pool_integrity["backend"][backend.upper()] = {
            "top5_j_at_5_invariant": True,
            "top5_j_at_5": top5_expected,
            "allnms_oracle_all_invariant": True,
            "allnms_oracle_all": allnms_oracle,
        }
    result.to_csv(run / "tables" / "formal_test_all_methods.csv", index=False)
    seed_values = {str(seed) for seed in SEEDS}
    seed_rows = result.loc[result["seed"].astype(str).isin(seed_values)].copy()
    summary_metrics = [
        "j_at_1",
        "delta_j_at_1",
        "recovered",
        "harmful",
        "net",
        "switch_rate",
        "mrr",
        "ndcg_at_5",
        "candidate_pr_auc",
        "candidate_roc_auc",
    ]
    seed_summary = seed_rows.groupby(
        ["track", "backend", "pool", "method"], as_index=False
    )[summary_metrics].agg(["mean", "std"])
    seed_summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in seed_summary.columns
    ]
    seed_summary.to_csv(run / "tables" / "formal_test_seed_summary.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(run / "tables" / "cross_backend.csv", index=False)
    stats_rows: list[dict[str, Any]] = []
    pvalues: dict[str, float] = {}
    for key, outcomes in sorted(outcome_index.items()):
        backend, pool, method = key.split("/", 2)
        mcnemar = mcnemar_exact(outcomes["baseline_correct"], outcomes["final_correct"])
        bootstrap = cluster_bootstrap_difference(
            outcomes["baseline_correct"],
            outcomes["final_correct"],
            outcomes["scene_id"],
            iterations=10_000,
            seed=stable_bootstrap_seed(20260806, f"formal/{key}"),
        )
        pvalues[key] = float(mcnemar["pvalue"])
        stats_rows.append(
            {
                "family": (
                    "formal_cross_backend_locked_methods"
                    if backend == "g1+c1"
                    else "formal_intra_backend_all_locked_methods"
                ),
                "backend": backend.upper(),
                "pool": pool,
                "method": method,
                **mcnemar,
                "bootstrap_lower": bootstrap["ci"][0],
                "bootstrap_upper": bootstrap["ci"][1],
                "bootstrap_iterations": int(bootstrap["iterations"]),
                "bootstrap_seed": int(bootstrap["seed"]),
                "bootstrap_sample_count": int(bootstrap["sample_count"]),
                "bootstrap_cluster_count": int(bootstrap["cluster_count"]),
                "bootstrap_resampling_unit": str(bootstrap["resampling_unit"]),
                "bootstrap_point_estimate": float(bootstrap["point_estimate"]),
                "comparison_key": key,
            }
        )
    # Core intra-backend and Track-C cross-backend comparisons are two
    # preregistered hypothesis families; Holm is applied within each family.
    for family in sorted({str(row["family"]) for row in stats_rows}):
        local = {
            str(row["comparison_key"]): float(row["pvalue"])
            for row in stats_rows
            if str(row["family"]) == family
        }
        adjusted = holm_adjust(local)
        for row in stats_rows:
            if str(row["family"]) == family:
                row["holm_adjusted_p"] = adjusted[str(row["comparison_key"])]

    primary_rows = []
    for backend in BACKENDS:
        selected = lock["selected_methods"]["backend"][backend.upper()]
        pool, method = str(selected["primary_pool"]), str(selected["primary_ungated_method"])
        for name in (method, "primary_gated"):
            key = f"{backend}/{pool}/{name}"
            statistical = next(row for row in stats_rows if row["backend"] == backend.upper() and row["pool"] == pool and row["method"] == name)
            row = result.loc[
                (result["backend"].eq(backend.upper()))
                & (result["pool"].eq(pool))
                & (result["method"].eq(name))
                & (result["seed"].astype(str).eq("ensemble") if name != "primary_gated" else True)
            ].iloc[0].to_dict()
            row.update({"ci_lower": statistical["bootstrap_lower"], "ci_upper": statistical["bootstrap_upper"], "mcnemar_p": statistical["pvalue"], "mcnemar_recovered": statistical["recovered"], "mcnemar_harmful": statistical["harmful"], "holm_adjusted_p": statistical["holm_adjusted_p"]})
            primary_rows.append(row)
    for row in primary_rows:
        row["decision"] = "GO" if row["delta_j_at_1"] > 0 and row["ci_lower"] > 0 and row["holm_adjusted_p"] < .05 else "CAUTION" if row["delta_j_at_1"] > 0 else "NO-GO"
    pd.DataFrame(primary_rows).to_csv(run / "tables" / "formal_test_primary.csv", index=False)
    statistics_frame = pd.DataFrame(stats_rows)
    statistics_frame.to_csv(run / "tables" / "statistical_tests.csv", index=False)
    statistics_root = run / "10_statistics"
    statistics_root.mkdir(parents=True, exist_ok=True)
    statistics_frame.to_csv(statistics_root / "statistical_tests.csv", index=False)
    pd.DataFrame(primary_rows).to_csv(
        statistics_root / "primary_deployment_tests.csv", index=False
    )
    evaluation_files = [
        *sorted((run / "09_formal_test" / "outcomes").rglob("*.parquet")),
        *sorted((run / "tables").glob("formal_test*.csv")),
        run / "tables" / "cross_backend.csv",
        run / "tables" / "statistical_tests.csv",
        run / "10_statistics" / "statistical_tests.csv",
        run / "10_statistics" / "primary_deployment_tests.csv",
    ]
    evaluation_artifacts = [
        {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
        for path in sorted(set(evaluation_files))
        if path.is_file()
    ]
    required_evaluation_files = {
        (run / "tables" / "formal_test_all_methods.csv").resolve(),
        (run / "tables" / "formal_test_primary.csv").resolve(),
        (run / "tables" / "formal_test_seed_summary.csv").resolve(),
        (run / "tables" / "cross_backend.csv").resolve(),
        (run / "tables" / "statistical_tests.csv").resolve(),
        (run / "10_statistics" / "statistical_tests.csv").resolve(),
        (run / "10_statistics" / "primary_deployment_tests.csv").resolve(),
    }
    inventoried = {Path(str(item["path"])).resolve() for item in evaluation_artifacts}
    if not evaluation_artifacts or not required_evaluation_files.issubset(inventoried):
        raise RuntimeError("formal evaluation output inventory is incomplete")
    atomic_json(marker, {
        "status": "COMPLETE",
        "formal_run_count": 1,
        "primary_rows": primary_rows,
        "baseline_regression": baseline_regression,
        "pool_integrity": pool_integrity,
        "primary_lock_sha256": sha256_file(lock_path),
        "predictions_complete_sha256": sha256_file(prediction_marker),
        "test_access_transaction_sha256": sha256_file(transaction),
        "evaluation_claim_sha256": evaluation_claim_sha,
        "evaluation_artifacts": evaluation_artifacts,
        "base_run": str(base.resolve()),
        "test_universe_sha256": sha256_file(base / "manifests" / "test_samples.parquet"),
    })
    atomic_json(
        evaluation_transaction,
        {
            "status": "CONSUMED",
            "primary_lock_sha256": sha256_file(lock_path),
            "predictions_complete_sha256": sha256_file(prediction_marker),
            "test_access_transaction_sha256": sha256_file(transaction),
            "evaluation_complete_sha256": sha256_file(marker),
            "base_run": str(base.resolve()),
            "test_universe_sha256": sha256_file(base / "manifests" / "test_samples.parquet"),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    base = args.base_run.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    if args.stage == "predict":
        predict(run, base)
    else:
        evaluate(run, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
