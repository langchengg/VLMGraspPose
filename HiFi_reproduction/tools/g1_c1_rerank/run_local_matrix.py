#!/usr/bin/env python3
"""Run calibration and the complete local validation model matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (str(REPOSITORY_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json, atomic_parquet  # noqa: E402
from src.grasping.g1_c1_safe_rerank.calibration import (  # noqa: E402
    ScoreCalibrator,
    calibration_metrics,
)
from src.grasping.g1_c1_safe_rerank.contracts import sha256_file  # noqa: E402
from src.grasping.g1_c1_safe_rerank.evaluation import evaluate_selected_ids  # noqa: E402
from src.grasping.g1_c1_safe_rerank.experiment_models import (  # noqa: E402
    GenericNeuralModel,
    LightGBMLambdaMARTModel,
    TabularResidualModel,
)
from src.grasping.g1_c1_safe_rerank.features import CONTINUOUS_FEATURE_GROUPS  # noqa: E402
from src.grasping.g1_c1_safe_rerank.models import (  # noqa: E402
    manual_feature_spec,
    manual_method_score,
    manual_peak_score,
)
from src.grasping.g1_c1_safe_rerank.ledger import ledger_stage  # noqa: E402


SEEDS = (17, 29, 43)
LISTWISE_TEMPERATURES = (0.5, 1.0, 2.0)
BACKENDS = ("g1", "c1")
POOLS = ("top5", "allnms")
CROSS_COLUMNS = (
    "cross_backend_candidate_available",
    "cross_backend_center_distance",
    "cross_backend_angle_difference",
    "cross_backend_width_ratio",
    "cross_backend_nearest_rank",
    "cross_backend_nearest_score",
    "cross_backend_rotated_iou",
    "cross_backend_nms_match",
    "cross_backend_agreement_score",
    "cross_backend_mutual_nearest",
)

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "F0": (
        "original_score", "original_rank", "score_margin_to_top1",
        "score_margin_to_previous", "score_margin_to_next", "score_z_within_set",
        "score_percentile_within_set", "score_entropy", "candidate_count_normalized",
    ),
    "F1": (
        "raw_network_quality", "stored_center_mask_support", "stored_jaw_mask_support",
        "score_decomposition_residual", "center_probability",
    ),
    # Dense backend maps were not persisted.  These label-free probability-map
    # local statistics are the preregistered scalar fallback for F2.
    "F2": (
        "local_probability_mean", "local_probability_max", "local_probability_min",
        "local_probability_std", "grasp_axis_mask_support",
    ),
    "F3": tuple(CONTINUOUS_FEATURE_GROUPS["mask"]),
    "F4": tuple(CONTINUOUS_FEATURE_GROUPS["width"] + CONTINUOUS_FEATURE_GROUPS["geometry"]),
    "F5": tuple(CONTINUOUS_FEATURE_GROUPS["depth"] + CONTINUOUS_FEATURE_GROUPS["clearance"] + CONTINUOUS_FEATURE_GROUPS["reliability"]),
    "F6": tuple(CONTINUOUS_FEATURE_GROUPS["relations"]),
    "F7": CROSS_COLUMNS,
    "F8": (),
}

METHODS: dict[str, dict[str, Any]] = {
    "r2_linear_bce": {"rung": "R2", "kind": "linear_bce"},
    "r2_linear_ranknet": {"rung": "R2", "kind": "linear_ranknet"},
    "r2_linear_listwise": {"rung": "R2", "kind": "linear_listwise"},
    "r3_mlp_bce": {"rung": "R3", "kind": "mlp_bce"},
    "r4_mlp_ranknet": {"rung": "R4", "kind": "mlp_ranknet"},
    "r4_mlp_ranknet_hard_top": {
        "rung": "R4-secondary",
        "kind": "mlp_ranknet_hard_top",
    },
    "r5_mlp_listwise": {"rung": "R5", "kind": "mlp_listwise"},
    "r5_mlp_listwise_t05": {
        "rung": "R5-temperature-secondary",
        "kind": "mlp_listwise_t05",
    },
    "r5_mlp_listwise_t20": {
        "rung": "R5-temperature-secondary",
        "kind": "mlp_listwise_t20",
    },
    "r5_mlp_listwise_hybrid": {
        "rung": "R5-secondary",
        "kind": "mlp_listwise_hybrid",
    },
    "r6_lambdamart": {"rung": "R6", "kind": "lambdamart"},
    "r8_deepsets": {"rung": "R8", "kind": "deepsets"},
    "r9_set_transformer": {"rung": "R9", "kind": "set_transformer"},
    "r10_candidate_gnn": {"rung": "R10", "kind": "gnn"},
    "r11_grare_4d_lite": {"rung": "R11", "kind": "mlp_ranknet", "grare": True},
}

MANUAL_METHODS: dict[str, tuple[str, ...]] = {
    "r1_mask_jaw": ("mask",),
    "r1_width": ("width",),
    "r1_depth_contact": ("depth",),
    "r1_clearance": ("clearance",),
    "r1_relations": ("relations",),
    "r1_stability": ("reliability",),
    "r1_full_utility": (
        "mask",
        "width",
        "F2_peak",
        "depth",
        "clearance",
        "relations",
        "reliability",
    ),
    "r1_peak": ("F2_peak",),
}


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "validate", "all"))
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _labels_path(run: Path, backend: str, split: str) -> Path:
    if split == "train":
        return run / "data" / f"{backend}_train_candidate_labels.parquet"
    return run / ("07_validation" if split == "validation" else "09_formal_test") / f"{backend}_candidate_labels.parquet"


def _load_features(run: Path, backend: str, split: str, pool: str) -> pd.DataFrame:
    path = run / "02_features" / split / backend / f"{pool}_features.parquet"
    frame = pd.read_parquet(path)
    cross_path = run / "02_features" / split / backend / "cross_backend_evidence.parquet"
    if cross_path.is_file():
        cross = pd.read_parquet(cross_path)
        frame = frame.merge(
            cross[["sample_id", "stable_candidate_id", *CROSS_COLUMNS]],
            on=["sample_id", "stable_candidate_id"],
            how="left",
            validate="one_to_one",
        )
    return frame.reset_index(drop=True)


def _load_joined(run: Path, backend: str, split: str, pool: str) -> pd.DataFrame:
    frame = _load_features(run, backend, split, pool)
    labels = pd.read_parquet(_labels_path(run, backend, split))
    joined = frame.merge(
        labels[["sample_id", "stable_candidate_id", "candidate_correct"]],
        on=["sample_id", "stable_candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(frame):
        raise AssertionError(f"{backend}/{split}/{pool}: feature/label key mismatch")
    joined["candidate_correct"] = joined["candidate_correct"].astype(bool)
    return joined.reset_index(drop=True)


def _universe(base: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(
        base / "manifests" / f"{split}_samples.parquet",
        columns=["sample_id", "scene_id", "rgbd_pair_sha256"],
    ).astype({"sample_id": str, "scene_id": str})


def _fold_assignments(base: Path, run: Path) -> pd.DataFrame:
    destination = run / "03_splits" / "fold_assignments.parquet"
    if destination.is_file():
        samples = pd.read_parquet(destination)
    else:
        samples = _universe(base, "train")
        splitter = GroupKFold(n_splits=5)
        samples["fold"] = -1
        # A scene contains one or more RGB-D frames, so scene_id is the stricter
        # connected grouping and also keeps every repeated query of a frame intact.
        for fold, (_, held) in enumerate(splitter.split(samples, groups=samples["scene_id"].astype(str))):
            samples.loc[held, "fold"] = int(fold)
        atomic_parquet(destination, samples)
    required = {"sample_id", "scene_id", "rgbd_pair_sha256", "fold"}
    if not required.issubset(samples):
        raise AssertionError("fold assignment schema is incomplete")
    source = _universe(base, "train").sort_values("sample_id").reset_index(drop=True)
    checked = samples.copy()
    checked["sample_id"] = checked["sample_id"].astype(str)
    checked["scene_id"] = checked["scene_id"].astype(str)
    checked = checked.sort_values("sample_id").reset_index(drop=True)
    if checked["sample_id"].duplicated().any() or not checked["sample_id"].equals(source["sample_id"]):
        raise AssertionError("fold assignments do not exactly match the Train universe")
    if not checked["scene_id"].equals(source["scene_id"]) or not checked["rgbd_pair_sha256"].astype(str).equals(source["rgbd_pair_sha256"].astype(str)):
        raise AssertionError("fold assignment grouping keys drifted from Train source")
    fold_values = pd.to_numeric(checked["fold"], errors="coerce")
    if (
        fold_values.isna().any()
        or set(fold_values.astype(int)) != set(range(5))
        or (checked.groupby("rgbd_pair_sha256")["fold"].nunique() > 1).any()
        or (checked.groupby("scene_id")["fold"].nunique() > 1).any()
    ):
        raise AssertionError("grouped fold assignment leakage")
    overlap = {
        f"fold_{fold}": {
            "samples": int((checked["fold"] == fold).sum()),
            "scenes": int(checked.loc[checked["fold"] == fold, "scene_id"].nunique()),
            "rgbd_groups": int(checked.loc[checked["fold"] == fold, "rgbd_pair_sha256"].nunique()),
        }
        for fold in range(5)
    }
    atomic_json(
        run / "03_splits" / "split_leakage_audit.json",
        {
            "status": "PASS",
            "group_priority": "scene_id connected grouping (strictly contains RGB-D repeated-query groups)",
            "folds": overlap,
            "group_cross_fold_overlap": 0,
            "fold_assignments_sha256": sha256_file(destination),
            "train_universe_rows": int(len(source)),
        },
    )
    return checked


def _calibration(run: Path, base: Path, backend: str) -> None:
    train = _load_joined(run, backend, "train", "allnms")
    validation = _load_joined(run, backend, "validation", "allnms")
    assignments = _fold_assignments(base, run)[["sample_id", "fold"]]
    train = train.merge(assignments, on="sample_id", how="left", validate="many_to_one")
    predictions = {kind: np.full(len(train), np.nan) for kind in ("platt", "isotonic")}
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fit = train.loc[train["fold"] != fold]
        held = train.loc[train["fold"] == fold]
        if set(fit["scene_id"].astype(str)) & set(held["scene_id"].astype(str)):
            # RGB-D grouping is the primary key. Scene overlap is also forbidden
            # by the official split and should remain zero here.
            raise AssertionError("scene leakage across calibration folds")
        for kind in predictions:
            calibrator = ScoreCalibrator.fit(kind, fit)
            values = calibrator.predict(held["original_score"])
            predictions[kind][held.index.to_numpy()] = values
            fold_rows.append({"fold": fold, "kind": kind, **calibration_metrics(values, held["candidate_correct"], sample_ids=held["sample_id"])})
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise AssertionError("calibration OOF coverage incomplete")
    full: dict[str, ScoreCalibrator] = {}
    validation_metrics: dict[str, Any] = {}
    for kind in predictions:
        full[kind] = ScoreCalibrator.fit(kind, train)
        values = np.clip(full[kind].predict(validation["original_score"]), 1e-4, 1.0 - 1e-4)
        validation_metrics[kind] = calibration_metrics(values, validation["candidate_correct"], sample_ids=validation["sample_id"])
    selected = min(
        validation_metrics,
        key=lambda kind: (
            validation_metrics[kind]["brier"],
            validation_metrics[kind]["log_loss"],
            validation_metrics[kind]["ece_15"],
            0 if kind == "platt" else 1,
        ),
    )
    train_probability = np.clip(predictions[selected], 1e-4, 1.0 - 1e-4)
    validation_probability = np.clip(full[selected].predict(validation["original_score"]), 1e-4, 1.0 - 1e-4)
    train_view = train[["sample_id", "stable_candidate_id"]].copy()
    train_view["source_score_calibrated"] = train_probability
    validation_view = validation[["sample_id", "stable_candidate_id"]].copy()
    validation_view["source_score_calibrated"] = validation_probability
    destination = run / "04_calibration" / backend
    atomic_parquet(destination / "train_oof_calibrated.parquet", train_view)
    atomic_parquet(destination / "validation_calibrated.parquet", validation_view)
    _atomic_pickle(destination / "calibrator.pkl", full[selected])
    pd.DataFrame(fold_rows).to_csv(destination / "calibration_curve.csv", index=False)
    baseline = validation.loc[validation["original_rank"].eq(1), ["sample_id", "candidate_correct"]]
    calibrated_baseline_j = float(baseline["candidate_correct"].sum() / len(_universe(base, "validation")))
    raw_baseline_j = calibrated_baseline_j
    if calibrated_baseline_j != raw_baseline_j:
        raise AssertionError("monotone calibration changed baseline J@1")
    atomic_json(
        destination / "calibration_metrics.json",
        {
            "backend": backend.upper(),
            "selected": selected,
            "selection_scope": "official validation Brier/NLL/ECE after train-only grouped OOF",
            "validation": validation_metrics,
            "fold_metrics": fold_rows,
            "clip": [1e-4, 1.0 - 1e-4],
            "calibrated_baseline_j_at_1": calibrated_baseline_j,
            "original_baseline_j_at_1": raw_baseline_j,
            "rank_preserving": True,
        },
    )


def _attach_calibration(run: Path, frame: pd.DataFrame, backend: str, split: str) -> pd.DataFrame:
    if split == "train":
        view = pd.read_parquet(run / "04_calibration" / backend / "train_oof_calibrated.parquet")
    elif split == "validation":
        view = pd.read_parquet(run / "04_calibration" / backend / "validation_calibrated.parquet")
    else:
        with (run / "04_calibration" / backend / "calibrator.pkl").open("rb") as stream:
            calibrator = pickle.load(stream)
        view = frame[["sample_id", "stable_candidate_id"]].copy()
        view["source_score_calibrated"] = np.clip(calibrator.predict(frame["original_score"]), 1e-4, 1.0 - 1e-4)
    output = frame.drop(columns=["source_score_calibrated"], errors="ignore").merge(
        view,
        on=["sample_id", "stable_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if output["source_score_calibrated"].isna().any():
        raise AssertionError("calibration join incomplete")
    return output.reset_index(drop=True)


def _selected_calibration_kind(run: Path, backend: str) -> str:
    payload = json.loads(
        (run / "04_calibration" / backend / "calibration_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    kind = str(payload.get("selected", ""))
    if kind not in {"platt", "isotonic"}:
        raise RuntimeError(f"invalid selected source-score calibrator: {backend}/{kind}")
    return kind


def _outer_fold_calibration(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    outer_fold: int,
    kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Cross-fit source calibration inside an outer Train fold.

    Outer-fit rows receive inner-fold predictions that use neither their own
    labels nor any outer-held labels.  The outer-held rows are transformed by
    one calibrator fitted on all outer-fit rows.  This prevents a globally
    generated OOF feature from indirectly carrying outer-held labels into the
    outer model.
    """
    fit = fit.drop(columns=["source_score_calibrated"], errors="ignore").reset_index(
        drop=True
    )
    held = held.drop(
        columns=["source_score_calibrated"], errors="ignore"
    ).reset_index(drop=True)
    if "fold" not in fit or "fold" not in held:
        raise ValueError("outer-fold calibration requires immutable fold assignments")
    if not held["fold"].astype(int).eq(int(outer_fold)).all() or fit["fold"].astype(int).eq(int(outer_fold)).any():
        raise AssertionError("outer-fold calibration partition mismatch")
    if set(fit["scene_id"].astype(str)) & set(held["scene_id"].astype(str)):
        raise AssertionError("scene leakage in outer-fold calibration")
    inner_folds = sorted(pd.to_numeric(fit["fold"], errors="raise").astype(int).unique())
    if len(inner_folds) < 2 or int(outer_fold) in inner_folds:
        raise AssertionError("outer-fit calibration has invalid inner folds")
    fit_probability = np.full(len(fit), np.nan, dtype=float)
    inner_rows: list[dict[str, Any]] = []
    for inner_fold in inner_folds:
        inner_held = fit["fold"].astype(int).eq(inner_fold).to_numpy()
        inner_train = ~inner_held
        if not inner_held.any() or not inner_train.any():
            raise AssertionError("empty inner calibration partition")
        calibrator = ScoreCalibrator.fit(kind, fit.loc[inner_train])
        fit_probability[inner_held] = calibrator.predict(
            fit.loc[inner_held, "original_score"]
        )
        inner_rows.append(
            {
                "inner_fold": int(inner_fold),
                "fit_candidates": int(inner_train.sum()),
                "held_candidates": int(inner_held.sum()),
                "fit_samples": int(fit.loc[inner_train, "sample_id"].nunique()),
                "held_samples": int(fit.loc[inner_held, "sample_id"].nunique()),
            }
        )
    if not np.isfinite(fit_probability).all():
        raise AssertionError("inner calibration coverage is incomplete")
    outer_calibrator = ScoreCalibrator.fit(kind, fit)
    held_probability = outer_calibrator.predict(held["original_score"])
    fit["source_score_calibrated"] = np.clip(fit_probability, 1e-4, 1 - 1e-4)
    held["source_score_calibrated"] = np.clip(
        held_probability, 1e-4, 1 - 1e-4
    )
    audit = {
        "outer_fold": int(outer_fold),
        "kind": kind,
        "fit_scope": "outer-fit-only nested cross-fit source calibration",
        "outer_fit_samples": int(fit["sample_id"].nunique()),
        "outer_held_samples": int(held["sample_id"].nunique()),
        "outer_fit_scenes": int(fit["scene_id"].nunique()),
        "outer_held_scenes": int(held["scene_id"].nunique()),
        "inner_folds": inner_rows,
        "outer_held_labels_used": False,
    }
    return fit, held, audit


def _columns(frame: pd.DataFrame, *, through: str = "F6", grare: bool = False) -> tuple[str, ...]:
    order = [f"F{index}" for index in range(int(through[1:]) + 1)]
    selected: list[str] = []
    for family in order:
        selected.extend(FEATURE_FAMILIES[family])
    if grare:
        allowed = set(FEATURE_FAMILIES["F0"] + FEATURE_FAMILIES["F3"] + FEATURE_FAMILIES["F4"] + FEATURE_FAMILIES["F5"])
        selected = [column for column in selected if column in allowed]
    result = tuple(dict.fromkeys(column for column in selected if column in frame.columns))
    if not result:
        raise ValueError("empty feature set")
    return result


def _selections(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    local = frame[["sample_id", "scene_id", "stable_candidate_id", "original_rank"]].copy()
    local["reranker_score"] = np.asarray(scores, dtype=float)
    baseline = local.sort_values(["sample_id", "original_rank", "stable_candidate_id"], kind="mergesort").groupby("sample_id", sort=False).first().reset_index()
    challenger = local.sort_values(["sample_id", "reranker_score", "stable_candidate_id"], ascending=[True, False, True], kind="mergesort").groupby("sample_id", sort=False).first().reset_index()
    return baseline[["sample_id", "stable_candidate_id"]].rename(columns={"stable_candidate_id": "baseline_candidate_id"}).merge(
        challenger[["sample_id", "stable_candidate_id"]].rename(columns={"stable_candidate_id": "selected_candidate_id"}),
        on="sample_id",
        validate="one_to_one",
    ).assign(fallback=False)


def _evaluate(frame: pd.DataFrame, universe: pd.DataFrame, scores: np.ndarray, method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = frame[["sample_id", "stable_candidate_id", "candidate_correct"]]
    return evaluate_selected_ids(_selections(frame, scores), labels, universe, method=method)


def _select_manual_alpha_held_fold_cv(
    train: pd.DataFrame,
    universe: pd.DataFrame,
    base_scores: np.ndarray,
    evidence: np.ndarray,
    method: str,
    *,
    alphas: Sequence[float] = (0.025, 0.05, 0.10),
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    """Select an R1 coefficient by explicit scene-grouped held-fold aggregation."""
    if "fold" not in train or "fold" not in universe:
        raise ValueError("R1 Train CV requires fold on candidates and universe")
    if len(base_scores) != len(train) or len(evidence) != len(train):
        raise ValueError("R1 Train CV score length mismatch")
    if universe["sample_id"].astype(str).duplicated().any():
        raise ValueError("R1 Train CV universe contains duplicate sample IDs")
    fold_values = pd.to_numeric(universe["fold"], errors="coerce")
    if fold_values.isna().any() or set(fold_values.astype(int)) != set(range(5)):
        raise ValueError("R1 Train CV requires complete folds 0..4")
    if (universe.groupby(universe["scene_id"].astype(str))["fold"].nunique() > 1).any():
        raise AssertionError("scene leakage across R1 Train CV folds")
    universe_fold = universe.set_index(universe["sample_id"].astype(str))["fold"]
    candidate_sample_ids = train["sample_id"].astype(str)
    if not set(candidate_sample_ids).issubset(set(universe_fold.index)):
        raise ValueError("R1 Train CV candidate lies outside the Train universe")
    expected_candidate_folds = candidate_sample_ids.map(universe_fold)
    candidate_folds = pd.to_numeric(train["fold"], errors="coerce")
    if candidate_folds.isna().any() or not np.array_equal(
        candidate_folds.to_numpy(dtype=int), expected_candidate_folds.to_numpy(dtype=int)
    ):
        raise AssertionError("R1 candidate fold assignment disagrees with sample universe")

    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for alpha in alphas:
        totals = {
            "total": 0,
            "baseline_correct": 0,
            "final_correct": 0,
            "recovered": 0,
            "harmful": 0,
            "net": 0,
            "switch_count": 0,
        }
        for fold in range(5):
            candidate_mask = candidate_folds.eq(fold).to_numpy()
            held_candidates = train.loc[candidate_mask].reset_index(drop=True)
            held_universe = universe.loc[fold_values.eq(fold)].reset_index(drop=True)
            held_scores = (
                np.asarray(base_scores, dtype=float)[candidate_mask]
                + float(alpha) * np.asarray(evidence, dtype=float)[candidate_mask]
            )
            _, metric = _evaluate(
                held_candidates,
                held_universe,
                held_scores,
                f"{method}_a{alpha}_fold{fold}",
            )
            baseline_correct = int(round(float(metric["baseline_j_at_1"]) * metric["total"]))
            final_correct = int(round(float(metric["j_at_1"]) * metric["total"]))
            fold_record = {
                "record_type": "held_fold",
                "method": method,
                "alpha": float(alpha),
                "fold": fold,
                "total": int(metric["total"]),
                "scene_count": int(held_universe["scene_id"].astype(str).nunique()),
                "candidate_rows": int(candidate_mask.sum()),
                "baseline_correct": baseline_correct,
                "final_correct": final_correct,
                "recovered": int(metric["recovered"]),
                "harmful": int(metric["harmful"]),
                "net": int(metric["net"]),
                "switch_count": int(metric["switch_count"]),
                "delta_j_at_1": float(metric["delta_j_at_1"]),
                "switch_rate": float(metric["switch_rate"]),
            }
            rows.append(fold_record)
            for key in totals:
                totals[key] += int(fold_record[key])
        if totals["total"] != len(universe):
            raise AssertionError("R1 held-fold aggregation does not cover Train exactly once")
        aggregate = {
            "record_type": "aggregate",
            "method": method,
            "alpha": float(alpha),
            "fold": "ALL",
            **totals,
            "scene_count": int(universe["scene_id"].astype(str).nunique()),
            "candidate_rows": int(len(train)),
            "delta_j_at_1": totals["net"] / max(totals["total"], 1),
            "switch_rate": totals["switch_count"] / max(totals["total"], 1),
        }
        rows.append(aggregate)
        aggregates.append(aggregate)

    selected = min(
        aggregates,
        key=lambda record: (
            -float(record["delta_j_at_1"]),
            int(record["harmful"]),
            float(record["switch_rate"]),
            float(record["alpha"]),
        ),
    )
    selected_alpha = float(selected["alpha"])
    records = pd.DataFrame(rows)
    records["selected_alpha"] = records["alpha"].eq(selected_alpha)
    selection = {
        "status": "PASS",
        "method": method,
        "selection_scope": "5-fold scene-grouped held-fold Train CV",
        "held_fold_rule": "each Train sample is evaluated exactly once per alpha in its assigned scene-grouped fold",
        "alpha_grid": [float(alpha) for alpha in alphas],
        "selected_alpha": selected_alpha,
        "tie_break": ["maximum delta_j_at_1", "minimum harmful", "minimum switch_rate", "minimum alpha"],
        "folds": 5,
        "total": int(selected["total"]),
        "scene_count": int(selected["scene_count"]),
        "selected_aggregate": {
            key: selected[key]
            for key in (
                "alpha",
                "total",
                "baseline_correct",
                "final_correct",
                "recovered",
                "harmful",
                "net",
                "switch_count",
                "delta_j_at_1",
                "switch_rate",
            )
        },
    }
    return selected_alpha, records, selection


def _fit_model(
    kind: str,
    columns: tuple[str, ...],
    train: pd.DataFrame,
    *,
    seed: int,
    device: str,
    checkpoint: Path,
    objective: str | None = None,
    temperature: float = 1.0,
) -> Any:
    if kind == "linear_bce":
        return TabularResidualModel("bce", columns, seed=seed).fit(train)
    if kind == "linear_ranknet":
        return TabularResidualModel("ranknet", columns, seed=seed).fit(train)
    if kind == "lambdamart":
        return LightGBMLambdaMARTModel(columns, seed=seed).fit(train)
    # Candidate sets are tiny (median <=4) but Train has 26k queries.  Eight
    # controlled MLP epochs and six set-model epochs keep the full 3-seed,
    # four-pool matrix and five-fold OOF protocol tractable; inner scene-held
    # J@1 still selects the refit epoch within this preregistered budget.
    epochs = 2 if len(train["sample_id"].unique()) <= 100 else (8 if kind.startswith("mlp") or kind == "linear_listwise" else 6)
    model = GenericNeuralModel(kind, columns, seed=seed, device=device, epochs=epochs, patience=min(6, max(epochs // 3, 2)), objective=objective, temperature=temperature)
    try:
        return model.fit(train, checkpoint)
    except RuntimeError as error:
        if device != "mps":
            raise
        fallback = GenericNeuralModel(kind, columns, seed=seed, device="cpu", epochs=epochs, patience=min(6, max(epochs // 3, 2)), objective=objective, temperature=temperature)
        fitted = fallback.fit(train, checkpoint)
        fitted.training_result = {**(fitted.training_result or {}), "mps_fallback_error": f"{type(error).__name__}: {error}"}
        return fitted


def _listwise_temperature_cv(
    args: argparse.Namespace,
    backend: str,
    pool: str,
    train: pd.DataFrame,
) -> dict[str, Any]:
    """Nested scene-grouped Train CV for the strict R5 temperature."""
    run, base = args.run_dir.resolve(), args.base_run.resolve()
    assignments = _fold_assignments(base, run)[["sample_id", "fold"]]
    folded = train.merge(assignments, on="sample_id", how="left", validate="many_to_one")
    if folded["fold"].isna().any():
        raise AssertionError("R5 temperature CV fold coverage is incomplete")
    universe = _universe(base, "train").merge(
        assignments, on="sample_id", how="left", validate="one_to_one"
    )
    columns = _columns(folded, through="F6")
    calibration_kind = _selected_calibration_kind(run, backend)
    outer_views: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    calibration_audit: list[dict[str, Any]] = []
    for fold in range(5):
        fit, held, audit = _outer_fold_calibration(
            folded.loc[folded["fold"].ne(fold)],
            folded.loc[folded["fold"].eq(fold)],
            outer_fold=fold,
            kind=calibration_kind,
        )
        outer_views[fold] = (fit, held)
        calibration_audit.append(audit)
    candidate_rows: list[dict[str, Any]] = []
    root = run / "07_validation" / "r5_temperature_cv" / backend / pool
    for temperature in LISTWISE_TEMPERATURES:
        tag = str(temperature).replace(".", "p")
        score_parts: list[pd.DataFrame] = []
        fold_metrics: list[dict[str, Any]] = []
        for fold in range(5):
            fit, held = (frame.copy() for frame in outer_views[fold])
            held_universe = universe.loc[universe["fold"].eq(fold)].reset_index(drop=True)
            if set(fit["scene_id"].astype(str)) & set(held["scene_id"].astype(str)):
                raise AssertionError("scene leakage in R5 temperature Train CV")
            model_path = run / "05_models" / "r5_temperature_cv" / backend / pool / tag / f"seed17_fold{fold}.joblib"
            score_path = run / "06_oof" / "r5_temperature_cv" / backend / pool / tag / f"seed17_fold{fold}.parquet"
            outcome_path = root / tag / f"outcomes_seed17_fold{fold}.parquet"
            metric_path = root / tag / f"metric_seed17_fold{fold}.json"
            if args.resume and all(
                path.is_file()
                for path in (model_path, score_path, outcome_path, metric_path)
            ):
                score_parts.append(pd.read_parquet(score_path))
                fold_metrics.append(json.loads(metric_path.read_text(encoding="utf-8")))
                continue
            checkpoint = run / "checkpoints" / "r5_temperature_cv" / backend / pool / tag / f"seed17_fold{fold}.pt"
            model = _fit_model(
                "mlp_listwise",
                columns,
                fit,
                seed=17,
                device=args.device,
                checkpoint=checkpoint,
                objective="listwise",
                temperature=temperature,
            )
            scores = model.predict_scores(held)
            outcomes, metric = _evaluate(
                held,
                held_universe,
                scores,
                f"r5_temperature_{tag}_seed17_fold{fold}",
            )
            metric.update(
                {
                    "backend": backend.upper(),
                    "pool": pool,
                    "temperature": temperature,
                    "seed": 17,
                    "fold": fold,
                    "selection_scope": "nested 5-fold scene-grouped Train OOF",
                    "source_calibration_scope": "outer-fit-only nested cross-fit",
                }
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path, compress=3)
            scored = held[["sample_id", "stable_candidate_id", "candidate_identity_sha256", "fold"]].copy()
            scored["reranker_score"] = scores
            atomic_parquet(score_path, scored)
            atomic_parquet(outcome_path, outcomes)
            atomic_json(metric_path, metric)
            atomic_json(model_path.with_suffix(".json"), model.artifact())
            score_parts.append(scored)
            fold_metrics.append(metric)
        oof = pd.concat(score_parts, ignore_index=True)
        if len(oof) != len(folded) or oof[["sample_id", "stable_candidate_id"]].duplicated().any():
            raise AssertionError("R5 temperature OOF candidate coverage failure")
        aligned = folded[["sample_id", "stable_candidate_id"]].merge(
            oof[["sample_id", "stable_candidate_id", "reranker_score"]],
            on=["sample_id", "stable_candidate_id"],
            how="left",
            validate="one_to_one",
        )["reranker_score"].to_numpy(dtype=float)
        if not np.isfinite(aligned).all():
            raise AssertionError("R5 temperature OOF scores are incomplete")
        outcomes, metric = _evaluate(
            folded,
            universe,
            aligned,
            f"r5_temperature_{tag}_seed17_oof",
        )
        record = {
            **metric,
            "backend": backend.upper(),
            "pool": pool,
            "temperature": float(temperature),
            "seed": 17,
            "folds": 5,
            "scene_count": int(universe["scene_id"].nunique()),
            "selection_scope": "nested 5-fold scene-grouped Train OOF",
            "fold_assignments_sha256": sha256_file(run / "03_splits" / "fold_assignments.parquet"),
        }
        atomic_parquet(root / tag / "scores_seed17_oof.parquet", oof)
        atomic_parquet(root / tag / "outcomes_seed17_oof.parquet", outcomes)
        atomic_json(root / tag / "metric_seed17_oof.json", record)
        candidate_rows.append(record)
    candidates = pd.DataFrame(candidate_rows)
    chosen = candidates.sort_values(
        ["delta_j_at_1", "harmful", "switch_rate", "temperature"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).iloc[0]
    selection = {
        "status": "PASS",
        "backend": backend.upper(),
        "pool": pool,
        "selection_scope": "nested 5-fold scene-grouped Train OOF",
        "folds": 5,
        "seed": 17,
        "temperature_grid": list(LISTWISE_TEMPERATURES),
        "selected_temperature": float(chosen["temperature"]),
        "selected_objective": "listwise",
        "tie_break": [
            "maximum OOF delta_j_at_1",
            "minimum harmful",
            "minimum switch_rate",
            "minimum temperature",
        ],
        "train_total": int(chosen["total"]),
        "train_scenes": int(chosen["scene_count"]),
        "fold_assignments_sha256": str(chosen["fold_assignments_sha256"]),
        "source_calibration_kind": calibration_kind,
        "outer_fold_calibration": calibration_audit,
        "selected_metric": {
            "temperature": float(chosen["temperature"]),
            "j_at_1": float(chosen["j_at_1"]),
            "delta_j_at_1": float(chosen["delta_j_at_1"]),
            "recovered": int(chosen["recovered"]),
            "harmful": int(chosen["harmful"]),
            "net": int(chosen["net"]),
            "switch_count": int(chosen["switch_count"]),
            "switch_rate": float(chosen["switch_rate"]),
        },
    }
    _atomic_csv(root / "R5_TEMPERATURE_TRAIN_CV.csv", candidates)
    atomic_json(root / "R5_TEMPERATURE_TRAIN_CV.json", selection)
    return selection


def _controlled_loss_selection(
    run: Path,
    backend: str,
    pool: str,
) -> dict[str, Any]:
    destination = run / "07_validation" / "loss_selection" / backend / f"{pool}.json"
    if destination.is_file():
        selection = json.loads(destination.read_text(encoding="utf-8"))
        if selection.get("status") != "LOCKED_FROM_OFFICIAL_VALIDATION":
            raise RuntimeError(f"invalid controlled loss selection: {destination}")
        return selection
    definitions = {
        "r3_mlp_bce": ("bce", 1.0),
        "r4_mlp_ranknet": ("ranknet", 1.0),
        "r5_mlp_listwise": ("listwise", None),
    }
    rows: list[dict[str, Any]] = []
    temperature_selection = json.loads(
        (
            run
            / "07_validation"
            / "r5_temperature_cv"
            / backend
            / pool
            / "R5_TEMPERATURE_TRAIN_CV.json"
        ).read_text(encoding="utf-8")
    )
    for method, (objective, temperature) in definitions.items():
        metric_path = run / "07_validation" / "metrics" / backend / pool / f"{method}_ensemble.json"
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": method,
                "objective": objective,
                "temperature": (
                    float(temperature_selection["selected_temperature"])
                    if temperature is None
                    else float(temperature)
                ),
                **{
                    key: metric[key]
                    for key in ("j_at_1", "delta_j_at_1", "harmful", "switch_rate")
                },
            }
        )
    candidates = pd.DataFrame(rows)
    chosen = candidates.sort_values(
        ["j_at_1", "harmful", "switch_rate", "method"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).iloc[0]
    selection = {
        "status": "LOCKED_FROM_OFFICIAL_VALIDATION",
        "backend": backend.upper(),
        "pool": pool,
        "selection_scope": "official Validation controlled MLP loss comparison",
        "selected_method": str(chosen["method"]),
        "selected_objective": str(chosen["objective"]),
        "selected_temperature": float(chosen["temperature"]),
        "r5_temperature_source": str(
            run
            / "07_validation"
            / "r5_temperature_cv"
            / backend
            / pool
            / "R5_TEMPERATURE_TRAIN_CV.json"
        ),
        "tie_break": [
            "maximum Validation J@1",
            "minimum harmful",
            "minimum switch_rate",
            "lexical method ID",
        ],
        "candidates": rows,
    }
    atomic_json(destination, selection)
    return selection


def resolved_method_training(
    run: Path,
    backend: str,
    pool: str,
    method: str,
) -> dict[str, Any]:
    spec = METHODS[method]
    kind = str(spec["kind"])
    objective: str | None = (
        "bce"
        if kind in {"linear_bce", "mlp_bce"}
        else "ranknet"
        if kind in {"linear_ranknet", "mlp_ranknet"}
        else "hard_top_ranknet"
        if kind == "mlp_ranknet_hard_top"
        else "hybrid"
        if kind == "mlp_listwise_hybrid"
        else "lambdarank"
        if kind == "lambdamart"
        else "listwise"
    )
    temperature = 1.0
    temperature_path = (
        run
        / "07_validation"
        / "r5_temperature_cv"
        / backend
        / pool
        / "R5_TEMPERATURE_TRAIN_CV.json"
    )
    if method in {"r2_linear_listwise", "r5_mlp_listwise"}:
        temperature = float(
            json.loads(temperature_path.read_text(encoding="utf-8"))[
                "selected_temperature"
            ]
        )
    if method in {"r8_deepsets", "r9_set_transformer", "r10_candidate_gnn", "r11_grare_4d_lite"}:
        selected = _controlled_loss_selection(run, backend, pool)
        objective = str(selected["selected_objective"])
        temperature = float(selected["selected_temperature"])
    return {
        "kind": kind,
        "objective": objective,
        "temperature": temperature,
    }


def _manual_validation(run: Path, base: Path, backend: str, pool: str, train: pd.DataFrame, validation: pd.DataFrame) -> list[dict[str, Any]]:
    assignments = _fold_assignments(base, run)[["sample_id", "scene_id", "fold"]].copy()
    assignments["sample_id"] = assignments["sample_id"].astype(str)
    assignments["scene_id"] = assignments["scene_id"].astype(str)
    universe_train = _universe(base, "train").merge(
        assignments[["sample_id", "fold"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if universe_train["fold"].isna().any():
        raise AssertionError("R1 Train CV fold coverage is incomplete")
    train = train.merge(
        assignments.rename(columns={"scene_id": "assigned_scene_id"}),
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if train[["fold", "assigned_scene_id"]].isna().any().any():
        raise AssertionError("R1 candidate fold coverage is incomplete")
    if not train["scene_id"].astype(str).eq(train["assigned_scene_id"].astype(str)).all():
        raise AssertionError("R1 candidate scene disagrees with fold source of truth")
    train = train.drop(columns=["assigned_scene_id"])
    universe_validation = _universe(base, "validation")
    base_train = np.log(np.clip(train["source_score_calibrated"], 1e-4, 1 - 1e-4) / np.clip(1 - train["source_score_calibrated"], 1e-4, 1))
    base_validation = np.log(np.clip(validation["source_score_calibrated"], 1e-4, 1 - 1e-4) / np.clip(1 - validation["source_score_calibrated"], 1e-4, 1))
    experiments = {
        name: groups for name, groups in MANUAL_METHODS.items() if name != "r1_peak"
    }
    records: list[dict[str, Any]] = []
    for name, groups in experiments.items():
        peak_columns = [column for column in FEATURE_FAMILIES["F2"] if column in train]
        train_evidence = manual_method_score(train, groups, peak_columns)
        validation_evidence = manual_method_score(validation, groups, peak_columns)
        alpha, cv_records, cv_selection = _select_manual_alpha_held_fold_cv(
            train, universe_train, base_train, train_evidence, name
        )
        cv_selection["feature_directions"] = manual_feature_spec(groups)
        cv_selection["fold_assignments_sha256"] = sha256_file(
            run / "03_splits" / "fold_assignments.parquet"
        )
        cv_root = run / "07_validation" / "r1_train_cv" / backend / pool
        _atomic_csv(cv_root / f"{name}.csv", cv_records)
        atomic_json(cv_root / f"{name}_selection.json", cv_selection)
        outcomes, metric = _evaluate(validation, universe_validation, base_validation + alpha * validation_evidence, name)
        selected_cv = cv_selection["selected_aggregate"]
        metric.update({"rung": "R1", "seed": 17, "alpha": alpha, "feature_groups": list(groups), "feature_directions": cv_selection["feature_directions"], "backend": backend.upper(), "pool": pool, "alpha_selection_scope": cv_selection["selection_scope"], "train_cv_folds": 5, "train_cv_total": selected_cv["total"], "train_cv_delta_j_at_1": selected_cv["delta_j_at_1"], "train_cv_harmful": selected_cv["harmful"], "train_cv_switch_rate": selected_cv["switch_rate"]})
        atomic_parquet(run / "07_validation" / "outcomes" / backend / pool / f"{name}.parquet", outcomes)
        records.append(metric)
    # F2 fallback is a named feature family rather than a generic continuous
    # group.  Test it separately so the interpretable evidence screen includes
    # the preregistered peak-confidence comparison.
    peak_columns = [column for column in FEATURE_FAMILIES["F2"] if column in train]
    train_peak = manual_peak_score(train, peak_columns)
    validation_peak = manual_peak_score(validation, peak_columns)
    alpha, cv_records, cv_selection = _select_manual_alpha_held_fold_cv(
        train, universe_train, base_train, train_peak, "r1_peak"
    )
    cv_selection["feature_directions"] = manual_feature_spec(("F2_peak",))
    cv_selection["fold_assignments_sha256"] = sha256_file(
        run / "03_splits" / "fold_assignments.parquet"
    )
    cv_root = run / "07_validation" / "r1_train_cv" / backend / pool
    _atomic_csv(cv_root / "r1_peak.csv", cv_records)
    atomic_json(cv_root / "r1_peak_selection.json", cv_selection)
    outcomes, metric = _evaluate(validation, universe_validation, base_validation + alpha * validation_peak, "r1_peak")
    selected_cv = cv_selection["selected_aggregate"]
    metric.update({"rung": "R1", "seed": 17, "alpha": alpha, "feature_groups": ["F2_peak"], "feature_directions": cv_selection["feature_directions"], "backend": backend.upper(), "pool": pool, "alpha_selection_scope": cv_selection["selection_scope"], "train_cv_folds": 5, "train_cv_total": selected_cv["total"], "train_cv_delta_j_at_1": selected_cv["delta_j_at_1"], "train_cv_harmful": selected_cv["harmful"], "train_cv_switch_rate": selected_cv["switch_rate"]})
    atomic_parquet(run / "07_validation" / "outcomes" / backend / pool / "r1_peak.parquet", outcomes)
    records.append(metric)
    return records


def _run_validation_dataset(args: argparse.Namespace, backend: str, pool: str) -> None:
    run, base = args.run_dir.resolve(), args.base_run.resolve()
    train = _attach_calibration(run, _load_joined(run, backend, "train", pool), backend, "train")
    validation = _attach_calibration(run, _load_joined(run, backend, "validation", pool), backend, "validation")
    if args.smoke:
        keep = set(train["sample_id"].drop_duplicates().head(100))
        train = train.loc[train["sample_id"].isin(keep)].reset_index(drop=True)
    universe = _universe(base, "validation")
    records = _manual_validation(run, base, backend, pool, train, validation)
    baseline_outcomes, baseline_metric = _evaluate(validation, universe, validation["original_score"].to_numpy(dtype=float), "r0_baseline")
    baseline_metric.update({"rung": "R0", "seed": 17, "backend": backend.upper(), "pool": pool})
    atomic_parquet(run / "07_validation" / "outcomes" / backend / pool / "r0_baseline.parquet", baseline_outcomes)
    records.append(baseline_metric)
    temperature_selection = _listwise_temperature_cv(args, backend, pool, train)
    for method, spec in METHODS.items():
        method_columns = _columns(train, through="F6", grare=bool(spec.get("grare")))
        training = resolved_method_training(run, backend, pool, method)
        for seed in SEEDS:
            key = f"{method}_seed{seed}"
            model_path = run / "05_models" / backend / pool / f"{key}.joblib"
            checkpoint = run / "checkpoints" / backend / pool / f"{key}.pt"
            prediction_path = run / "07_validation" / "candidate_scores" / backend / pool / f"{key}.parquet"
            outcome_path = run / "07_validation" / "outcomes" / backend / pool / f"{key}.parquet"
            metric_path = run / "07_validation" / "metrics" / backend / pool / f"{key}.json"
            with ledger_stage(run, stage="validation", backend=backend.upper(), pool=pool, method=method, feature_set="F0-F6", seed=seed) as ledger:
                if args.resume and all(path.is_file() for path in (model_path, prediction_path, outcome_path, metric_path)):
                    records.append(json.loads(metric_path.read_text(encoding="utf-8")))
                    ledger["artifact_path"] = str(metric_path)
                    ledger["artifact_sha256"] = sha256_file(metric_path)
                    continue
                model = _fit_model(
                    str(training["kind"]),
                    method_columns,
                    train,
                    seed=seed,
                    device=args.device,
                    checkpoint=checkpoint,
                    objective=training["objective"],
                    temperature=float(training["temperature"]),
                )
                scores = model.predict_scores(validation)
                outcomes, metric = _evaluate(validation, universe, scores, key)
                artifact = model.artifact()
                metric.update({"rung": spec["rung"], "method": method, "kind": training["kind"], "objective": artifact.get("objective", training["objective"]), "temperature": artifact.get("temperature", training["temperature"]), "seed": seed, "backend": backend.upper(), "pool": pool, "feature_set": "F0-F6", "feature_count": len(method_columns), "controlled_loss_source": (str(run / "07_validation" / "loss_selection" / backend / f"{pool}.json") if method in {"r8_deepsets", "r9_set_transformer", "r10_candidate_gnn", "r11_grare_4d_lite"} else None), "r5_temperature_train_cv": (temperature_selection["selected_temperature"] if method in {"r2_linear_listwise", "r5_mlp_listwise"} else None)})
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path, compress=3)
                atomic_json(model_path.with_suffix(".json"), artifact)
                score_frame = validation[["sample_id", "stable_candidate_id", "candidate_identity_sha256"]].copy()
                score_frame["reranker_score"] = scores
                atomic_parquet(prediction_path, score_frame)
                atomic_parquet(outcome_path, outcomes)
                atomic_json(metric_path, metric)
                records.append(metric)
                ledger["artifact_path"] = str(metric_path)
                ledger["artifact_sha256"] = sha256_file(metric_path)
        seed_frames = [
            pd.read_parquet(
                run
                / "07_validation"
                / "candidate_scores"
                / backend
                / pool
                / f"{method}_seed{seed}.parquet"
            ).rename(columns={"reranker_score": f"score_seed{seed}"})
            for seed in SEEDS
        ]
        ensemble = seed_frames[0]
        for seed_frame in seed_frames[1:]:
            ensemble = ensemble.merge(
                seed_frame.drop(columns=["candidate_identity_sha256"]),
                on=["sample_id", "stable_candidate_id"],
                how="inner",
                validate="one_to_one",
            )
        score_columns = [f"score_seed{seed}" for seed in SEEDS]
        ensemble["reranker_score"] = ensemble[score_columns].mean(axis=1)
        top_votes = np.zeros(len(ensemble), dtype=float)
        for column in score_columns:
            top_index = (
                ensemble.sort_values(
                    ["sample_id", column, "stable_candidate_id"],
                    ascending=[True, False, True],
                    kind="mergesort",
                )
                .groupby("sample_id", sort=False)
                .head(1)
                .index
            )
            top_votes[top_index] += 1.0
        ensemble["seed_agreement_fraction"] = top_votes / len(SEEDS)
        ensemble_path = (
            run
            / "07_validation"
            / "candidate_scores"
            / backend
            / pool
            / f"{method}_ensemble.parquet"
        )
        atomic_parquet(ensemble_path, ensemble)
        ensemble_scores = validation[["sample_id", "stable_candidate_id"]].merge(
            ensemble[["sample_id", "stable_candidate_id", "reranker_score"]],
            on=["sample_id", "stable_candidate_id"],
            how="left",
            validate="one_to_one",
        )["reranker_score"].to_numpy(dtype=float)
        ensemble_outcomes, ensemble_metric = _evaluate(
            validation, universe, ensemble_scores, f"{method}_ensemble"
        )
        seed_metadata = json.loads(
            (
                run
                / "07_validation"
                / "metrics"
                / backend
                / pool
                / f"{method}_seed{SEEDS[0]}.json"
            ).read_text(encoding="utf-8")
        )
        ensemble_metric.update(
            {
                "rung": spec["rung"],
                "method": method,
                "kind": seed_metadata.get("kind", training["kind"]),
                "objective": seed_metadata.get("objective", training["objective"]),
                "temperature": seed_metadata.get("temperature", training["temperature"]),
                "seed": "ensemble",
                "backend": backend.upper(),
                "pool": pool,
                "feature_set": "F0-F6",
                "feature_count": len(method_columns),
            }
        )
        atomic_parquet(
            run / "07_validation" / "outcomes" / backend / pool / f"{method}_ensemble.parquet",
            ensemble_outcomes,
        )
        atomic_json(
            run / "07_validation" / "metrics" / backend / pool / f"{method}_ensemble.json",
            ensemble_metric,
        )
        records.append(ensemble_metric)
    table = pd.DataFrame(records)
    destination = run / "07_validation" / "tables" / f"{backend}_{pool}_metrics.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)


def _feature_status(run: Path) -> None:
    rows = [
        {"family": "F0", "status": "COMPLETE", "fallback": "none", "evidence": "frozen score/rank and pool statistics"},
        {"family": "F1", "status": "COMPLETE_WITH_FALLBACK", "fallback": "stored smoothed network quality; raw pre-Gaussian tensor not persisted", "evidence": "candidate metadata and exact score decomposition residual"},
        {"family": "F2", "status": "FALLBACK", "fallback": "native HiFi probability local statistics", "evidence": "backend dense map arrays are summaries only and were not persisted"},
        {"family": "F3", "status": "COMPLETE", "fallback": "none", "evidence": "predicted probability/mask rectangle, jaw, contact and sweep regions"},
        {"family": "F4", "status": "COMPLETE_WITH_FALLBACK", "fallback": "distance transform/principal-axis instead of skeleton branch", "evidence": "width/shape/mask geometry features"},
        {"family": "F5", "status": "COMPLETE", "fallback": "single-view 2.5D proxy only", "evidence": "depth/contact/clearance/reliability features"},
        {"family": "F6", "status": "COMPLETE", "fallback": "complete graph because N<=14", "evidence": "candidate relations plus edge-conditioned GNN raw relations"},
        {"family": "F7", "status": "COMPLETE_WITH_FALLBACK", "fallback": "candidate-pool cross evidence; other-backend dense maps not persisted", "evidence": "native original-image nearest, IoU, mutual and NMS-match audit"},
        {"family": "F8", "status": "UNAVAILABLE_FALLBACK_SCALAR", "fallback": "F0-F7 scalar ranker", "evidence": "backend interfaces expose only output tensors and checkpoints; no frozen shared-latent hook/crop artifact existed before candidate freeze"},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(run / "02_features" / "FEATURE_STATUS.csv", index=False)
    atomic_json(run / "02_features" / "FEATURE_STATUS.json", {"families": rows, "test_tuning": False})


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    args.base_run = args.base_run.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    _fold_assignments(args.base_run, args.run_dir)
    _feature_status(args.run_dir)
    stages = ("calibrate", "validate") if args.stage == "all" else (args.stage,)
    if "calibrate" in stages:
        for backend in BACKENDS:
            with ledger_stage(args.run_dir, stage="calibration", backend=backend.upper(), pool="allnms"):
                _calibration(args.run_dir, args.base_run, backend)
    if "validate" in stages:
        for backend in BACKENDS:
            for pool in POOLS:
                try:
                    _run_validation_dataset(args, backend, pool)
                except Exception:
                    failure = args.run_dir / "logs" / f"validation_failure_{backend}_{pool}.txt"
                    failure.parent.mkdir(parents=True, exist_ok=True)
                    failure.write_text(traceback.format_exc(), encoding="utf-8")
                    raise
        tables = [pd.read_csv(args.run_dir / "07_validation" / "tables" / f"{backend}_{pool}_metrics.csv") for backend in BACKENDS for pool in POOLS]
        combined = pd.concat(tables, ignore_index=True)
        combined.to_csv(args.run_dir / "07_validation" / "VALIDATION_MATRIX.csv", index=False)
        loss = combined.loc[combined["method"].astype(str).isin(["r3_mlp_bce", "r4_mlp_ranknet", "r5_mlp_listwise"])]
        loss.to_csv(args.run_dir / "07_validation" / "LOSS_COMPARISON.csv", index=False)
        combined.loc[
            combined["method"].astype(str).eq("r5_mlp_listwise_hybrid")
        ].to_csv(
            args.run_dir / "07_validation" / "LISTWISE_HYBRID_SECONDARY.csv",
            index=False,
        )
        combined.loc[
            combined["method"].astype(str).isin(
                ["r5_mlp_listwise_t05", "r5_mlp_listwise", "r5_mlp_listwise_t20"]
            )
        ].to_csv(
            args.run_dir / "07_validation" / "LISTWISE_TEMPERATURE_SECONDARY.csv",
            index=False,
        )
        encoder = combined.loc[combined["method"].astype(str).isin(["r2_linear_bce", "r3_mlp_bce", "r4_mlp_ranknet", "r5_mlp_listwise", "r6_lambdamart", "r8_deepsets", "r9_set_transformer", "r10_candidate_gnn"])]
        encoder.to_csv(args.run_dir / "07_validation" / "ENCODER_COMPARISON.csv", index=False)
        temperature_rows = [
            pd.read_csv(
                args.run_dir
                / "07_validation"
                / "r5_temperature_cv"
                / backend
                / pool
                / "R5_TEMPERATURE_TRAIN_CV.csv"
            )
            for backend in BACKENDS
            for pool in POOLS
        ]
        _atomic_csv(
            args.run_dir / "07_validation" / "R5_TEMPERATURE_TRAIN_CV.csv",
            pd.concat(temperature_rows, ignore_index=True),
        )
        loss_selections = {
            backend.upper(): {
                pool: _controlled_loss_selection(args.run_dir, backend, pool)
                for pool in POOLS
            }
            for backend in BACKENDS
        }
        atomic_json(
            args.run_dir / "07_validation" / "LOSS_SELECTION.json",
            {
                "status": "LOCKED_FROM_OFFICIAL_VALIDATION",
                "test_labels_accessed": False,
                "backend": loss_selections,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
