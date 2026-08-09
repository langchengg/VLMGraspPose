#!/usr/bin/env python3
"""Select validation methods, run controlled ablations, OOF gates, and R14."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd


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
from src.grasping.g1_c1_safe_rerank.crop_model import CropResidualModel  # noqa: E402
from src.grasping.g1_c1_safe_rerank.evaluation import (  # noqa: E402
    evaluate_selected_ids,
    scene_cluster_bootstrap,
    stable_bootstrap_seed,
)
from src.grasping.g1_c1_safe_rerank.gate import (  # noqa: E402
    ExpectedGainGate,
    build_pair_features,
    expected_gain_sweep,
)
from src.grasping.g1_c1_safe_rerank.ledger import ledger_stage  # noqa: E402
from tools.g1_c1_rerank.run_local_matrix import (  # noqa: E402
    BACKENDS,
    FEATURE_FAMILIES,
    METHODS,
    POOLS,
    SEEDS,
    _attach_calibration,
    _columns,
    _evaluate,
    _fit_model,
    _load_joined,
    _outer_fold_calibration,
    _selected_calibration_kind,
    _universe,
    resolved_method_training,
)
from tools.g1_c1_rerank.run_crop_cnn import load_crops  # noqa: E402


CORE = {
    "r2_linear_bce",
    "r2_linear_ranknet",
    "r2_linear_listwise",
    "r3_mlp_bce",
    "r4_mlp_ranknet",
    "r5_mlp_listwise",
    "r6_lambdamart",
}
MLP = {"r3_mlp_bce", "r4_mlp_ranknet", "r5_mlp_listwise"}
SET = {"r8_deepsets", "r9_set_transformer", "r10_candidate_gnn"}


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("select", "ablate", "oof-gate", "pooled", "all")
    )
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _validation_table(run: Path) -> pd.DataFrame:
    path = run / "07_validation" / "VALIDATION_MATRIX.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _ensemble_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["seed"].astype(str).eq("ensemble")].copy()


def _method_bootstrap(run: Path, row: pd.Series) -> dict[str, Any]:
    if str(row["method"]) == "r13_crop_cnn":
        outcome_path = (
            run
            / "07_validation"
            / "crop_cnn"
            / str(row["backend"]).lower()
            / str(row["pool"])
            / "outcomes_ensemble.parquet"
        )
    else:
        outcome_path = (
            run
            / "07_validation"
            / "outcomes"
            / str(row["backend"]).lower()
            / str(row["pool"])
            / f"{row['method']}_ensemble.parquet"
        )
    outcome = pd.read_parquet(outcome_path)
    return scene_cluster_bootstrap(
        outcome,
        draws=10_000,
        seed=stable_bootstrap_seed(20260806, f"{row['backend']}/{row['pool']}/{row['method']}"),
    )


def run_selection(run: Path) -> dict[str, Any]:
    frame = _ensemble_rows(_validation_table(run))
    crop_path = run / "07_validation" / "R13_CROP_CNN_RESULTS.csv"
    if crop_path.is_file():
        frame = pd.concat(
            [frame, _ensemble_rows(pd.read_csv(crop_path))], ignore_index=True
        )
    learned = frame.loc[
        frame["method"].astype(str).isin(set(METHODS) | {"r13_crop_cnn"})
    ].copy()
    rows: list[dict[str, Any]] = []
    for row in learned.to_dict(orient="records"):
        record = dict(row)
        bootstrap = _method_bootstrap(run, pd.Series(row))
        record["bootstrap_lower"] = bootstrap["delta_j_at_1"]["lower"]
        record["bootstrap_upper"] = bootstrap["delta_j_at_1"]["upper"]
        record["bootstrap_draws"] = bootstrap["draws"]
        rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(run / "07_validation" / "VALIDATION_ENSEMBLE_BOOTSTRAP.csv", index=False)
    selections: dict[str, Any] = {
        "selection_scope": "official validation only",
        "test_labels_accessed": False,
        "backend": {},
    }
    size_rank = {
        "r2_linear_bce": 0,
        "r2_linear_ranknet": 0,
        "r2_linear_listwise": 1,
        "r6_lambdamart": 2,
        "r3_mlp_bce": 3,
        "r4_mlp_ranknet": 3,
        "r5_mlp_listwise": 3,
        "r8_deepsets": 4,
        "r9_set_transformer": 6,
        "r10_candidate_gnn": 5,
        "r11_grare_4d_lite": 3,
    }
    table["model_size_rank"] = table["method"].map(size_rank).fillna(99)
    encoder_selection: dict[str, Any] = {
        "status": "LOCKED_FROM_OFFICIAL_VALIDATION",
        "selection_scope": "backend/pool controlled encoder comparison after loss lock",
        "test_labels_accessed": False,
        "backend": {},
    }
    controlled_encoder_rows: list[pd.DataFrame] = []
    linear_for_objective = {
        "bce": "r2_linear_bce",
        "ranknet": "r2_linear_ranknet",
        "listwise": "r2_linear_listwise",
    }
    for backend in BACKENDS:
        encoder_selection["backend"][backend.upper()] = {}
        for pool in POOLS:
            loss_path = run / "07_validation" / "loss_selection" / backend / f"{pool}.json"
            loss = json.loads(loss_path.read_text(encoding="utf-8"))
            objective = str(loss["selected_objective"])
            controlled_methods = {
                linear_for_objective[objective],
                str(loss["selected_method"]),
                "r8_deepsets",
                "r9_set_transformer",
                "r10_candidate_gnn",
            }
            local = table.loc[
                table["backend"].astype(str).str.lower().eq(backend)
                & table["pool"].astype(str).eq(pool)
                & table["method"].astype(str).isin(controlled_methods)
            ].copy()
            if set(local["method"].astype(str)) != controlled_methods:
                raise RuntimeError(f"{backend}/{pool}: controlled encoder rows incomplete")
            if not local["objective"].astype(str).eq(objective).all():
                raise RuntimeError(f"{backend}/{pool}: encoder objective drift")
            local["comparison_scope"] = "fixed_locked_loss"
            local["locked_loss_source"] = str(loss_path)
            controlled_encoder_rows.append(local)
            chosen_encoder = local.sort_values(
                ["j_at_1", "bootstrap_lower", "harmful", "model_size_rank", "switch_rate", "method"],
                ascending=[False, False, True, True, True, True],
                kind="mergesort",
            ).iloc[0]
            encoder_selection["backend"][backend.upper()][pool] = {
                "selected_method": str(chosen_encoder["method"]),
                "selected_objective": objective,
                "selected_temperature": float(loss["selected_temperature"]),
                "validation_j_at_1": float(chosen_encoder["j_at_1"]),
                "validation_delta_j_at_1": float(chosen_encoder["delta_j_at_1"]),
                "loss_selection_path": str(loss_path),
            }
            intrinsic = table.loc[
                table["backend"].astype(str).str.lower().eq(backend)
                & table["pool"].astype(str).eq(pool)
                & table["method"].astype(str).eq("r6_lambdamart")
            ].copy()
            intrinsic["comparison_scope"] = "intrinsic_lambdarank_reference"
            intrinsic["locked_loss_source"] = str(loss_path)
            controlled_encoder_rows.append(intrinsic)
    encoder_table = pd.concat(controlled_encoder_rows, ignore_index=True)
    encoder_table.to_csv(run / "07_validation" / "ENCODER_COMPARISON.csv", index=False)
    atomic_json(run / "07_validation" / "ENCODER_SELECTION.json", encoder_selection)
    for backend in BACKENDS:
        local = table.loc[table["backend"].astype(str).str.lower().eq(backend)]
        eligible_methods: set[str] = {"r6_lambdamart", "r11_grare_4d_lite", "r13_crop_cnn"}
        for pool in POOLS:
            eligible_methods.add(
                str(encoder_selection["backend"][backend.upper()][pool]["selected_method"])
            )
        core = local.loc[local["method"].isin(eligible_methods)].sort_values(
            ["j_at_1", "bootstrap_lower", "harmful", "model_size_rank", "switch_rate", "method"],
            ascending=[False, False, True, True, True, True],
            kind="mergesort",
        )
        if core.empty:
            raise RuntimeError(f"{backend}: no eligible core validation method")
        chosen = core.iloc[0]
        advanced = local.loc[local["method"].isin(SET | {"r11_grare_4d_lite", "r13_crop_cnn"})].sort_values(
            ["j_at_1", "bootstrap_lower", "harmful", "model_size_rank", "method"],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        ).iloc[0]
        chosen_loss = json.loads(
            (
                run
                / "07_validation"
                / "loss_selection"
                / backend
                / f"{chosen['pool']}.json"
            ).read_text(encoding="utf-8")
        )
        selections["backend"][backend.upper()] = {
            "primary_ungated_method": str(chosen["method"]),
            "primary_pool": str(chosen["pool"]),
            "primary_feature_set": "F0-F6",
            "primary_loss": str(chosen["objective"]),
            "primary_temperature": float(chosen["temperature"]),
            "validation_j_at_1": float(chosen["j_at_1"]),
            "validation_delta_j_at_1": float(chosen["delta_j_at_1"]),
            "validation_bootstrap_lower": float(chosen["bootstrap_lower"]),
            "best_controlled_mlp_loss": str(chosen_loss["selected_method"]),
            "controlled_encoder_method": str(
                encoder_selection["backend"][backend.upper()][str(chosen["pool"])][
                    "selected_method"
                ]
            ),
            "best_advanced_method": str(advanced["method"]),
        }
    atomic_json(run / "07_validation" / "VALIDATION_SELECTION.json", selections)
    return selections


def _fit_predict_experiment(
    *,
    args: argparse.Namespace,
    backend: str,
    pool: str,
    name: str,
    kind: str,
    objective: str | None,
    temperature: float,
    columns: tuple[str, ...],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    universe: pd.DataFrame,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        root = args.run_dir / "07_validation" / "ablations" / backend / pool / name
        model_path = root / f"model_seed{seed}.joblib"
        metric_path = root / f"metric_seed{seed}.json"
        score_path = root / f"scores_seed{seed}.parquet"
        outcome_path = root / f"outcomes_seed{seed}.parquet"
        if args.resume and all(path.is_file() for path in (model_path, metric_path, score_path, outcome_path)):
            result.append(json.loads(metric_path.read_text(encoding="utf-8")))
            score_frames.append(pd.read_parquet(score_path).rename(columns={"reranker_score": f"score_seed{seed}"}))
            continue
        checkpoint = args.run_dir / "checkpoints" / "ablations" / backend / pool / f"{name}_seed{seed}.pt"
        model = _fit_model(
            kind,
            columns,
            train,
            seed=seed,
            device=args.device,
            checkpoint=checkpoint,
            objective=objective,
            temperature=temperature,
        )
        scores = model.predict_scores(validation)
        outcomes, metric = _evaluate(validation, universe, scores, f"{name}_seed{seed}")
        metric.update({"backend": backend.upper(), "pool": pool, "method": name, "seed": seed, "feature_count": len(columns), "encoder_kind": kind, "objective": objective, "temperature": temperature})
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path, compress=3)
        score = validation[["sample_id", "stable_candidate_id", "candidate_identity_sha256"]].copy()
        score["reranker_score"] = scores
        atomic_parquet(score_path, score)
        atomic_parquet(outcome_path, outcomes)
        atomic_json(metric_path, metric)
        result.append(metric)
        score_frames.append(score.rename(columns={"reranker_score": f"score_seed{seed}"}))
    ensemble = score_frames[0]
    for frame in score_frames[1:]:
        ensemble = ensemble.merge(
            frame.drop(columns=["candidate_identity_sha256"]),
            on=["sample_id", "stable_candidate_id"],
            validate="one_to_one",
        )
    ensemble_scores = ensemble[[f"score_seed{seed}" for seed in SEEDS]].mean(axis=1)
    aligned = validation[["sample_id", "stable_candidate_id"]].merge(
        ensemble.assign(reranker_score=ensemble_scores)[["sample_id", "stable_candidate_id", "reranker_score"]],
        on=["sample_id", "stable_candidate_id"],
        validate="one_to_one",
    )["reranker_score"].to_numpy(dtype=float)
    outcomes, metric = _evaluate(validation, universe, aligned, f"{name}_ensemble")
    metric.update({"backend": backend.upper(), "pool": pool, "method": name, "seed": "ensemble", "feature_count": len(columns), "encoder_kind": kind, "objective": objective, "temperature": temperature})
    root = args.run_dir / "07_validation" / "ablations" / backend / pool / name
    atomic_parquet(root / "scores_ensemble.parquet", ensemble.assign(reranker_score=ensemble_scores))
    atomic_parquet(root / "outcomes_ensemble.parquet", outcomes)
    atomic_json(root / "metric_ensemble.json", metric)
    result.append(metric)
    return result


def run_ablations(args: argparse.Namespace, selections: dict[str, Any]) -> None:
    cumulative_rows: list[dict[str, Any]] = []
    leave_rows: list[dict[str, Any]] = []
    encoder_selection = json.loads(
        (args.run_dir / "07_validation" / "ENCODER_SELECTION.json").read_text(
            encoding="utf-8"
        )
    )
    for backend in BACKENDS:
        for pool in POOLS:
            method = str(
                encoder_selection["backend"][backend.upper()][pool]["selected_method"]
            )
            training = resolved_method_training(
                args.run_dir, backend, pool, method
            )
            kind = str(training["kind"])
            train = _attach_calibration(args.run_dir, _load_joined(args.run_dir, backend, "train", pool), backend, "train")
            validation = _attach_calibration(args.run_dir, _load_joined(args.run_dir, backend, "validation", pool), backend, "validation")
            universe = _universe(args.base_run, "validation")
            for index in range(8):
                family = f"F{index}"
                columns = _columns(train, through=family)
                records = _fit_predict_experiment(
                    args=args,
                    backend=backend,
                    pool=pool,
                    name=f"cumulative_{family}",
                    kind=kind,
                    objective=training["objective"],
                    temperature=float(training["temperature"]),
                    columns=columns,
                    train=train,
                    validation=validation,
                    universe=universe,
                )
                for record in records:
                    record["feature_set"] = f"F0-{family}"
                cumulative_rows.extend(records)
            all_columns = list(_columns(train, through="F7"))
            for removed in ("F2", "F3", "F4", "F5", "F6"):
                excluded = set(FEATURE_FAMILIES[removed])
                columns = tuple(column for column in all_columns if column not in excluded)
                records = _fit_predict_experiment(
                    args=args,
                    backend=backend,
                    pool=pool,
                    name=f"leave_out_{removed}",
                    kind=kind,
                    objective=training["objective"],
                    temperature=float(training["temperature"]),
                    columns=columns,
                    train=train,
                    validation=validation,
                    universe=universe,
                )
                for record in records:
                    record["feature_set"] = f"F0-F7_minus_{removed}"
                leave_rows.extend(records)
    pd.DataFrame(cumulative_rows).to_csv(args.run_dir / "07_validation" / "FEATURE_ABLATION_CUMULATIVE.csv", index=False)
    pd.DataFrame(leave_rows).to_csv(args.run_dir / "07_validation" / "FEATURE_ABLATION_LEAVE_ONE_OUT.csv", index=False)


def _candidate_scores(run: Path, backend: str, pool: str, method: str, seed: int | str) -> pd.DataFrame:
    if method == "r13_crop_cnn":
        suffix = "ensemble" if seed == "ensemble" else f"seed{seed}"
        return pd.read_parquet(
            run
            / "07_validation"
            / "crop_cnn"
            / backend
            / pool
            / f"scores_{suffix}.parquet"
        )
    return pd.read_parquet(
        run / "07_validation" / "candidate_scores" / backend / pool / f"{method}_{'ensemble' if seed == 'ensemble' else f'seed{seed}'}.parquet"
    )


def _ranker_calibration(
    oof: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[ScoreCalibrator, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_column = "oof_fold" if "oof_fold" in oof else "fold"
    if fold_column not in oof:
        raise ValueError("ranker calibration requires outer OOF fold identifiers")
    folds = sorted(
        pd.to_numeric(oof[fold_column], errors="raise").astype(int).unique()
    )
    if folds != list(range(5)):
        raise ValueError("ranker calibration requires the immutable five Train folds")
    if "ranker_probability_outer_pure" not in oof:
        raise ValueError("gate OOF lacks outer-pure ranker probabilities")
    oof_probability = pd.to_numeric(
        oof["ranker_probability_outer_pure"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(oof_probability).all():
        raise AssertionError("outer-pure ranker probabilities are incomplete")
    # Platt is fixed for this second-level stacking feature.  For every outer
    # held fold its probability was already produced from outer-model scores
    # on outer-fit rows only; see `_outer_pure_ranker_probability`.  The full
    # OOF fit below is used only for Validation/formal inference.
    fit_frame = oof[
        ["sample_id", "stable_candidate_id", "candidate_correct", "reranker_score"]
    ].rename(columns={"reranker_score": "original_score"})
    calibrator = ScoreCalibrator.fit("platt", fit_frame)
    validation_probability = np.clip(
        calibrator.predict(validation["reranker_score"]), 1e-4, 1 - 1e-4
    )
    metrics = {
        "platt": calibration_metrics(
            validation_probability,
            validation["candidate_correct"],
            sample_ids=validation["sample_id"],
        )
    }
    oof_output = oof.copy()
    validation_output = validation.copy()
    oof_output["ranker_probability"] = oof_probability
    validation_output["ranker_probability"] = validation_probability
    return calibrator, oof_output, validation_output, {"selected": "platt", "validation": metrics, "fit_scope": "outer-model outer-fit Platt probabilities for gate OOF; full OOF Platt applied only to Validation/formal", "folds": folds, "outer_held_labels_used": False}


def _outer_pure_ranker_probability(
    train: pd.DataFrame,
    held_ensemble: pd.DataFrame,
    outer_fit_by_seed: Sequence[pd.DataFrame],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Calibrate each outer-held score without second-order OOF leakage."""
    merged = outer_fit_by_seed[0]
    for frame in outer_fit_by_seed[1:]:
        merged = merged.merge(
            frame.drop(columns=["scene_id", "candidate_identity_sha256"]),
            on=["sample_id", "stable_candidate_id", "outer_fold"],
            validate="one_to_one",
        )
    seed_columns = [f"score_seed{seed}" for seed in SEEDS]
    merged["reranker_score"] = merged[seed_columns].mean(axis=1)
    if len(merged) != 4 * len(train) or merged.duplicated(
        ["sample_id", "stable_candidate_id", "outer_fold"]
    ).any():
        raise AssertionError("outer-fit score inventory is incomplete or duplicate")
    coverage = merged[["sample_id", "stable_candidate_id", "outer_fold"]].merge(
        train[["sample_id", "stable_candidate_id", "fold"]],
        on=["sample_id", "stable_candidate_id"],
        validate="many_to_one",
    )
    per_candidate = coverage.groupby(
        ["sample_id", "stable_candidate_id"], sort=False
    )["outer_fold"].agg(["size", "nunique"])
    if (
        len(per_candidate) != len(train)
        or not per_candidate["size"].eq(4).all()
        or not per_candidate["nunique"].eq(4).all()
        or coverage["outer_fold"].astype(int).eq(coverage["fold"].astype(int)).any()
    ):
        raise AssertionError(
            "each outer-fit candidate must appear in exactly the four folds other than its own"
        )
    probability = np.full(len(held_ensemble), np.nan, dtype=float)
    audits: list[dict[str, Any]] = []
    label_view = train[
        ["sample_id", "scene_id", "stable_candidate_id", "candidate_correct", "fold"]
    ]
    for fold in range(5):
        local = merged.loc[merged["outer_fold"].astype(int).eq(fold)].merge(
            label_view.drop(columns="fold"),
            on=["sample_id", "scene_id", "stable_candidate_id"],
            validate="one_to_one",
        )
        if local.empty:
            raise AssertionError("outer-fit ranker calibration partition mismatch")
        fit_frame = local[
            ["sample_id", "stable_candidate_id", "candidate_correct", "reranker_score"]
        ].rename(columns={"reranker_score": "original_score"})
        calibrator = ScoreCalibrator.fit("platt", fit_frame)
        held_mask = held_ensemble["fold"].astype(int).eq(fold).to_numpy()
        probability[held_mask] = calibrator.predict(
            held_ensemble.loc[held_mask, "reranker_score"]
        )
        held_scenes = set(
            train.loc[train["fold"].astype(int).eq(fold), "scene_id"].astype(str)
        )
        if set(local["scene_id"].astype(str)) & held_scenes:
            raise AssertionError("outer-held scene entered ranker calibrator")
        audits.append(
            {
                "outer_fold": fold,
                "calibrator": "platt",
                "fit_samples": int(local["sample_id"].nunique()),
                "held_samples": int(held_ensemble.loc[held_mask, "sample_id"].nunique()),
                "outer_held_labels_used": False,
                "outer_model_fit_scores_used": True,
            }
        )
    if not np.isfinite(probability).all():
        raise AssertionError("outer-pure ranker probability coverage is incomplete")
    return np.clip(probability, 1e-4, 1 - 1e-4), audits


def _finalists(table: pd.DataFrame, backend: str, pool: str) -> list[str]:
    local = _ensemble_rows(table)
    local = local.loc[
        local["backend"].astype(str).str.lower().eq(backend)
        & local["pool"].astype(str).eq(pool)
    ]
    groups = (
        CORE - MLP,
        MLP,
        SET,
        {"r11_grare_4d_lite", "r13_crop_cnn"},
    )
    finalists = []
    for methods in groups:
        selected = local.loc[local["method"].isin(methods)].sort_values(
            ["j_at_1", "harmful", "switch_rate", "method"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        if not selected.empty:
            finalists.append(str(selected.iloc[0]["method"]))
    if len(finalists) < 4:
        raise RuntimeError(f"{backend}/{pool}: fewer than four gate finalists")
    return finalists


def _oof_method(args: argparse.Namespace, backend: str, pool: str, method: str) -> pd.DataFrame:
    if method == "r13_crop_cnn":
        return _oof_crop_method(args, backend, pool)
    train = _load_joined(args.run_dir, backend, "train", pool)
    assignments = pd.read_parquet(args.run_dir / "03_splits" / "fold_assignments.parquet")[["sample_id", "fold"]]
    train = train.merge(assignments, on="sample_id", how="left", validate="many_to_one")
    columns = _columns(train, through="F6", grare=bool(METHODS[method].get("grare")))
    training = resolved_method_training(args.run_dir, backend, pool, method)
    calibration_kind = _selected_calibration_kind(args.run_dir, backend)
    outer_views: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    calibration_audit: list[dict[str, Any]] = []
    for fold in range(5):
        fit, held, audit = _outer_fold_calibration(
            train.loc[train["fold"].ne(fold)],
            train.loc[train["fold"].eq(fold)],
            outer_fold=fold,
            kind=calibration_kind,
        )
        outer_views[fold] = (fit, held)
        calibration_audit.append(audit)
    atomic_json(
        args.run_dir / "06_oof" / backend / pool / method / "OUTER_FOLD_CALIBRATION.json",
        {"status": "PASS", "kind": calibration_kind, "folds": calibration_audit},
    )
    per_seed: list[pd.DataFrame] = []
    per_seed_outer_fit: list[pd.DataFrame] = []
    for seed in SEEDS:
        destination = args.run_dir / "06_oof" / backend / pool / method / f"seed{seed}.parquet"
        outer_fit_destination = args.run_dir / "06_oof" / backend / pool / method / f"outer_fit_seed{seed}.parquet"
        if args.resume and destination.is_file() and outer_fit_destination.is_file():
            per_seed.append(pd.read_parquet(destination).rename(columns={"reranker_score": f"score_seed{seed}"}))
            per_seed_outer_fit.append(
                pd.read_parquet(outer_fit_destination).rename(
                    columns={"reranker_score": f"score_seed{seed}"}
                )
            )
            continue
        parts: list[pd.DataFrame] = []
        fit_parts: list[pd.DataFrame] = []
        for fold in range(5):
            fit, held = (frame.copy() for frame in outer_views[fold])
            if set(fit["scene_id"].astype(str)) & set(held["scene_id"].astype(str)):
                raise AssertionError("scene leakage in ranker OOF")
            checkpoint = args.run_dir / "checkpoints" / "oof" / backend / pool / method / f"seed{seed}_fold{fold}.pt"
            model = _fit_model(
                str(training["kind"]),
                columns,
                fit,
                seed=seed,
                device=args.device,
                checkpoint=checkpoint,
                objective=training["objective"],
                temperature=float(training["temperature"]),
            )
            score = held[["sample_id", "scene_id", "stable_candidate_id", "candidate_identity_sha256", "fold"]].copy()
            score["reranker_score"] = model.predict_scores(held)
            parts.append(score)
            fit_score = fit[["sample_id", "scene_id", "stable_candidate_id", "candidate_identity_sha256"]].copy()
            fit_score["outer_fold"] = fold
            fit_score["reranker_score"] = model.predict_scores(fit)
            fit_parts.append(fit_score)
            model_path = args.run_dir / "05_models" / "oof" / backend / pool / method / f"seed{seed}_fold{fold}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path, compress=3)
        output = pd.concat(parts, ignore_index=True)
        if len(output) != len(train) or output[["sample_id", "stable_candidate_id"]].duplicated().any():
            raise AssertionError("ranker OOF coverage/identity failure")
        atomic_parquet(destination, output)
        outer_fit_output = pd.concat(fit_parts, ignore_index=True)
        if outer_fit_output.duplicated(["sample_id", "stable_candidate_id", "outer_fold"]).any():
            raise AssertionError("duplicate outer-fit ranker scores")
        atomic_parquet(outer_fit_destination, outer_fit_output)
        per_seed.append(output.rename(columns={"reranker_score": f"score_seed{seed}"}))
        per_seed_outer_fit.append(
            outer_fit_output.rename(columns={"reranker_score": f"score_seed{seed}"})
        )
    ensemble = per_seed[0]
    for frame in per_seed[1:]:
        ensemble = ensemble.merge(
            frame.drop(columns=["scene_id", "candidate_identity_sha256", "fold"]),
            on=["sample_id", "stable_candidate_id"],
            validate="one_to_one",
        )
    score_columns = [f"score_seed{seed}" for seed in SEEDS]
    ensemble["reranker_score"] = ensemble[score_columns].mean(axis=1)
    votes = np.zeros(len(ensemble), dtype=float)
    for column in score_columns:
        index = ensemble.sort_values(["sample_id", column, "stable_candidate_id"], ascending=[True, False, True], kind="mergesort").groupby("sample_id", sort=False).head(1).index
        votes[index] += 1.0
    ensemble["seed_agreement_fraction"] = votes / len(SEEDS)
    ensemble["ranker_probability_outer_pure"], probability_audit = (
        _outer_pure_ranker_probability(train, ensemble, per_seed_outer_fit)
    )
    atomic_json(
        args.run_dir / "06_oof" / backend / pool / method / "OUTER_PURE_RANKER_PROBABILITY.json",
        {"status": "PASS", "folds": probability_audit},
    )
    atomic_parquet(args.run_dir / "06_oof" / backend / pool / method / "ensemble.parquet", ensemble)
    return ensemble


def _oof_crop_method(
    args: argparse.Namespace,
    backend: str,
    pool: str,
) -> pd.DataFrame:
    method = "r13_crop_cnn"
    train = _load_joined(args.run_dir, backend, "train", pool)
    crops = load_crops(args.run_dir, backend, "train", train)
    assignments = pd.read_parquet(
        args.run_dir / "03_splits" / "fold_assignments.parquet"
    )[["sample_id", "fold"]]
    train = train.merge(assignments, on="sample_id", how="left", validate="many_to_one")
    if train["fold"].isna().any():
        raise AssertionError("crop OOF fold coverage is incomplete")
    columns = _columns(train, through="F6")
    loss = json.loads(
        (
            args.run_dir
            / "07_validation"
            / "loss_selection"
            / backend
            / f"{pool}.json"
        ).read_text(encoding="utf-8")
    )
    objective = str(loss["selected_objective"])
    temperature = float(loss["selected_temperature"])
    calibration_kind = _selected_calibration_kind(args.run_dir, backend)
    outer_views: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    calibration_audit: list[dict[str, Any]] = []
    for fold in range(5):
        fit, held, audit = _outer_fold_calibration(
            train.loc[train["fold"].ne(fold)],
            train.loc[train["fold"].eq(fold)],
            outer_fold=fold,
            kind=calibration_kind,
        )
        outer_views[fold] = (fit, held)
        calibration_audit.append(audit)
    atomic_json(
        args.run_dir / "06_oof" / backend / pool / method / "OUTER_FOLD_CALIBRATION.json",
        {"status": "PASS", "kind": calibration_kind, "folds": calibration_audit},
    )
    per_seed: list[pd.DataFrame] = []
    per_seed_outer_fit: list[pd.DataFrame] = []
    for seed in SEEDS:
        destination = args.run_dir / "06_oof" / backend / pool / method / f"seed{seed}.parquet"
        outer_fit_destination = args.run_dir / "06_oof" / backend / pool / method / f"outer_fit_seed{seed}.parquet"
        if args.resume and destination.is_file() and outer_fit_destination.is_file():
            per_seed.append(
                pd.read_parquet(destination).rename(
                    columns={"reranker_score": f"score_seed{seed}"}
                )
            )
            per_seed_outer_fit.append(
                pd.read_parquet(outer_fit_destination).rename(
                    columns={"reranker_score": f"score_seed{seed}"}
                )
            )
            continue
        parts: list[pd.DataFrame] = []
        fit_parts: list[pd.DataFrame] = []
        for fold in range(5):
            fit_mask = train["fold"].ne(fold).to_numpy()
            held_mask = train["fold"].eq(fold).to_numpy()
            fit, held = (frame.copy() for frame in outer_views[fold])
            if set(fit["scene_id"].astype(str)) & set(held["scene_id"].astype(str)):
                raise AssertionError("scene leakage in crop ranker OOF")
            checkpoint = args.run_dir / "checkpoints" / "oof" / backend / pool / method / f"seed{seed}_fold{fold}.pt"
            model = CropResidualModel(
                columns,
                seed=seed,
                device=args.device,
                epochs=8,
                objective=objective,
                temperature=temperature,
            )
            try:
                model.fit(fit, crops[fit_mask], checkpoint)
            except RuntimeError:
                if args.device != "mps":
                    raise
                model = CropResidualModel(
                    columns,
                    seed=seed,
                    device="cpu",
                    epochs=8,
                    objective=objective,
                    temperature=temperature,
                ).fit(fit, crops[fit_mask], checkpoint)
            score = held[["sample_id", "scene_id", "stable_candidate_id", "candidate_identity_sha256", "fold"]].copy()
            score["reranker_score"] = model.predict_scores(held, crops[held_mask])
            parts.append(score)
            fit_score = fit[["sample_id", "scene_id", "stable_candidate_id", "candidate_identity_sha256"]].copy()
            fit_score["outer_fold"] = fold
            fit_score["reranker_score"] = model.predict_scores(fit, crops[fit_mask])
            fit_parts.append(fit_score)
            model_path = args.run_dir / "05_models" / "oof" / backend / pool / method / f"seed{seed}_fold{fold}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path, compress=3)
        output = pd.concat(parts, ignore_index=True)
        if len(output) != len(train) or output[["sample_id", "stable_candidate_id"]].duplicated().any():
            raise AssertionError("crop ranker OOF coverage/identity failure")
        atomic_parquet(destination, output)
        outer_fit_output = pd.concat(fit_parts, ignore_index=True)
        if outer_fit_output.duplicated(["sample_id", "stable_candidate_id", "outer_fold"]).any():
            raise AssertionError("duplicate outer-fit crop ranker scores")
        atomic_parquet(outer_fit_destination, outer_fit_output)
        per_seed.append(
            output.rename(columns={"reranker_score": f"score_seed{seed}"})
        )
        per_seed_outer_fit.append(
            outer_fit_output.rename(columns={"reranker_score": f"score_seed{seed}"})
        )
    ensemble = per_seed[0]
    for frame in per_seed[1:]:
        ensemble = ensemble.merge(
            frame.drop(columns=["scene_id", "candidate_identity_sha256", "fold"]),
            on=["sample_id", "stable_candidate_id"],
            validate="one_to_one",
        )
    score_columns = [f"score_seed{seed}" for seed in SEEDS]
    ensemble["reranker_score"] = ensemble[score_columns].mean(axis=1)
    votes = np.zeros(len(ensemble), dtype=float)
    for column in score_columns:
        selected = (
            ensemble.sort_values(
                ["sample_id", column, "stable_candidate_id"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            .groupby("sample_id", sort=False)
            .head(1)
            .index
        )
        votes[selected] += 1.0
    ensemble["seed_agreement_fraction"] = votes / len(SEEDS)
    ensemble["ranker_probability_outer_pure"], probability_audit = (
        _outer_pure_ranker_probability(train, ensemble, per_seed_outer_fit)
    )
    atomic_json(
        args.run_dir / "06_oof" / backend / pool / method / "OUTER_PURE_RANKER_PROBABILITY.json",
        {"status": "PASS", "folds": probability_audit},
    )
    atomic_parquet(
        args.run_dir / "06_oof" / backend / pool / method / "ensemble.parquet",
        ensemble,
    )
    return ensemble


def run_oof_gates(args: argparse.Namespace, selections: dict[str, Any]) -> None:
    validation_table = _validation_table(args.run_dir)
    crop_results = args.run_dir / "07_validation" / "R13_CROP_CNN_RESULTS.csv"
    if crop_results.is_file():
        validation_table = pd.concat(
            [validation_table, pd.read_csv(crop_results)], ignore_index=True
        )
    gate_summary: dict[str, Any] = {"fit_scope": "Train scene-grouped ranker OOF", "backend": {}}
    comparison_rows: list[dict[str, Any]] = []
    for backend in BACKENDS:
        pool = str(selections["backend"][backend.upper()]["primary_pool"])
        finalists = _finalists(validation_table, backend, pool)
        train = _attach_calibration(args.run_dir, _load_joined(args.run_dir, backend, "train", pool), backend, "train")
        validation = _attach_calibration(args.run_dir, _load_joined(args.run_dir, backend, "validation", pool), backend, "validation")
        backend_records: dict[str, Any] = {"pool": pool, "finalists": finalists, "methods": {}}
        for method in finalists:
            with ledger_stage(args.run_dir, stage="oof_gate", backend=backend.upper(), pool=pool, method=method, feature_set="F0-F6"):
                oof_scores = _oof_method(args, backend, pool, method)
                oof_scored = train.merge(
                    oof_scores[["sample_id", "stable_candidate_id", "reranker_score", "seed_agreement_fraction", "ranker_probability_outer_pure", "fold"]].rename(columns={"fold": "oof_fold"}),
                    on=["sample_id", "stable_candidate_id"],
                    validate="one_to_one",
                )
                validation_scores = _candidate_scores(args.run_dir, backend, pool, method, "ensemble")
                validation_scored = validation.merge(
                    validation_scores[["sample_id", "stable_candidate_id", "reranker_score", "seed_agreement_fraction"]],
                    on=["sample_id", "stable_candidate_id"],
                    validate="one_to_one",
                )
                calibrator, oof_scored, validation_scored, calibration_record = _ranker_calibration(oof_scored, validation_scored)
                calibration_root = args.run_dir / "04_calibration" / "ranker" / backend / pool / method
                _atomic_pickle(calibration_root / "calibrator.pkl", calibrator)
                atomic_json(calibration_root / "metrics.json", calibration_record)
                oof_pairs = build_pair_features(oof_scored, include_labels=True)
                validation_pairs = build_pair_features(validation_scored, include_labels=True)
                atomic_parquet(args.run_dir / "06_oof" / backend / pool / method / "gate_pairs.parquet", oof_pairs)
                best: tuple[tuple[float, float, float, str], dict[str, Any]] | None = None
                method_record: dict[str, Any] = {"ranker_calibration": calibration_record, "gates": {}}
                for gate_kind in ("multinomial_logistic", "gradient_boosted"):
                    gate = ExpectedGainGate(kind=gate_kind, seed=17).fit(oof_pairs)
                    predicted = gate.predict(validation_pairs)
                    sweep, gate_selection = expected_gain_sweep(
                        predicted,
                        universe=_universe(args.base_run, "validation"),
                        lambda_values=(1.0, 2.0, 3.0, 5.0) if backend == "g1" else (1.0, 1.5, 2.0, 3.0),
                        bootstrap_draws=10_000,
                        bootstrap_seed=stable_bootstrap_seed(20260806, f"gate/{backend}/{method}/{gate_kind}"),
                    )
                    root = args.run_dir / "07_validation" / "gates" / backend / pool / method / gate_kind
                    atomic_parquet(root / "predicted_pairs.parquet", predicted)
                    root.mkdir(parents=True, exist_ok=True)
                    sweep.to_csv(root / "risk_coverage_sweep.csv", index=False)
                    joblib.dump(gate, root / "gate.joblib", compress=3)
                    atomic_json(root / "gate.json", gate.artifact())
                    atomic_json(root / "selection.json", gate_selection)
                    safe = gate_selection["safe_lcb"]
                    curve = sweep.copy()
                    curve["switch_risk"] = curve["harmful"] / curve["switch_count"].clip(lower=1)
                    curve = (
                        curve.groupby("switch_rate", as_index=False)["switch_risk"]
                        .min()
                        .sort_values("switch_rate", kind="mergesort")
                    )
                    maximum_coverage = float(curve["switch_rate"].max())
                    risk_coverage_auc = (
                        float(np.trapezoid(curve["switch_risk"], curve["switch_rate"]) / maximum_coverage)
                        if maximum_coverage > 0.0 and len(curve) > 1
                        else 0.0
                    )
                    comparison_rows.append({
                        "backend": backend.upper(),
                        "pool": pool,
                        "method": method,
                        "gate_kind": gate_kind,
                        **safe,
                        "risk_coverage_auc": risk_coverage_auc,
                        "risk_definition": "harmful_switches/all_switches; minimum-risk frontier by switch-rate",
                        "deploy_safe_lcb": gate_selection["deploy_safe_lcb"],
                    })
                    method_record["gates"][gate_kind] = gate_selection
                    key = (float(safe["bootstrap_lower"]), float(safe["delta_j_at_1"]), -float(safe["harmful"]), gate_kind)
                    if best is None or key > best[0]:
                        best = (key, {"gate_kind": gate_kind, **gate_selection})
                assert best is not None
                method_record["selected_gate"] = best[1]
                backend_records["methods"][method] = method_record
        primary_method = str(selections["backend"][backend.upper()]["primary_ungated_method"])
        backend_records["primary_method"] = primary_method
        backend_records["primary_gate"] = backend_records["methods"].get(primary_method, {}).get("selected_gate")
        gate_summary["backend"][backend.upper()] = backend_records
    pd.DataFrame(comparison_rows).to_csv(args.run_dir / "07_validation" / "GATE_COMPARISON.csv", index=False)
    atomic_json(args.run_dir / "07_validation" / "GATE_SELECTION.json", gate_summary)


def run_pooled(args: argparse.Namespace) -> None:
    table = _ensemble_rows(_validation_table(args.run_dir))
    records: list[dict[str, Any]] = []
    for pool in POOLS:
        loss_rows = table.loc[table["pool"].astype(str).eq(pool) & table["method"].isin(MLP)]
        best_method = (
            loss_rows.groupby("method", as_index=False)["j_at_1"].mean().sort_values(["j_at_1", "method"], ascending=[False, True]).iloc[0]["method"]
        )
        kind = str(METHODS[str(best_method)]["kind"])
        objective = (
            "bce"
            if str(best_method) == "r3_mlp_bce"
            else "ranknet"
            if str(best_method) == "r4_mlp_ranknet"
            else "listwise"
        )
        temperature = 1.0
        if objective == "listwise":
            temperature_cv = pd.read_csv(
                args.run_dir / "07_validation" / "R5_TEMPERATURE_TRAIN_CV.csv"
            )
            local_temperature = temperature_cv.loc[
                temperature_cv["pool"].astype(str).eq(pool)
            ].copy()
            pooled_temperature = (
                local_temperature.groupby("temperature", as_index=False)
                .agg(net=("net", "sum"), harmful=("harmful", "sum"), switch_count=("switch_count", "sum"), total=("total", "sum"))
            )
            pooled_temperature["delta_j_at_1"] = pooled_temperature["net"] / pooled_temperature["total"]
            pooled_temperature["switch_rate"] = pooled_temperature["switch_count"] / pooled_temperature["total"]
            temperature = float(
                pooled_temperature.sort_values(
                    ["delta_j_at_1", "harmful", "switch_rate", "temperature"],
                    ascending=[False, True, True, True],
                    kind="mergesort",
                ).iloc[0]["temperature"]
            )
        train_parts, validation_parts = [], []
        for backend in BACKENDS:
            for split, parts in (("train", train_parts), ("validation", validation_parts)):
                frame = _attach_calibration(args.run_dir, _load_joined(args.run_dir, backend, split, pool), backend, split)
                frame["backend_g1"] = float(backend == "g1")
                frame["sample_id_original"] = frame["sample_id"].astype(str)
                frame["sample_id"] = backend + ":" + frame["sample_id"].astype(str)
                frame["scene_id"] = backend + ":" + frame["scene_id"].astype(str)
                parts.append(frame)
        train = pd.concat(train_parts, ignore_index=True)
        validation = pd.concat(validation_parts, ignore_index=True)
        columns = (*_columns(train, through="F6"), "backend_g1")
        score_frames: list[pd.DataFrame] = []
        for seed in SEEDS:
            checkpoint = args.run_dir / "checkpoints" / "pooled" / pool / f"seed{seed}.pt"
            model = _fit_model(
                kind,
                columns,
                train,
                seed=seed,
                device=args.device,
                checkpoint=checkpoint,
                objective=objective,
                temperature=temperature,
            )
            scores = model.predict_scores(validation)
            scored = validation[["sample_id", "sample_id_original", "stable_candidate_id", "backend"]].copy()
            scored["reranker_score"] = scores
            atomic_parquet(args.run_dir / "07_validation" / "pooled" / pool / f"scores_seed{seed}.parquet", scored)
            model_path = args.run_dir / "05_models" / "pooled" / pool / f"seed{seed}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path, compress=3)
            score_frames.append(scored.rename(columns={"reranker_score": f"score_seed{seed}"}))
        ensemble = score_frames[0]
        for scored in score_frames[1:]:
            ensemble = ensemble.merge(scored.drop(columns=["sample_id_original", "backend"]), on=["sample_id", "stable_candidate_id"], validate="one_to_one")
        ensemble["reranker_score"] = ensemble[[f"score_seed{seed}" for seed in SEEDS]].mean(axis=1)
        atomic_parquet(args.run_dir / "07_validation" / "pooled" / pool / "scores_ensemble.parquet", ensemble)
        for backend in BACKENDS:
            local = validation.loc[validation["backend"].astype(str).str.lower().eq(backend)].copy()
            score = local[["sample_id", "stable_candidate_id"]].merge(
                ensemble[["sample_id", "stable_candidate_id", "reranker_score"]],
                on=["sample_id", "stable_candidate_id"], validate="one_to_one"
            )["reranker_score"].to_numpy(dtype=float)
            local["sample_id"] = local["sample_id_original"]
            outcomes, metric = _evaluate(local, _universe(args.base_run, "validation"), score, "r14_pooled_backend_conditioned")
            metric.update({"backend": backend.upper(), "pool": pool, "method": "r14_pooled_backend_conditioned", "base_method": best_method, "objective": objective, "temperature": temperature, "seed": "ensemble"})
            atomic_parquet(args.run_dir / "07_validation" / "pooled" / pool / f"outcomes_{backend}.parquet", outcomes)
            records.append(metric)
    pd.DataFrame(records).to_csv(args.run_dir / "07_validation" / "R14_POOLED_RESULTS.csv", index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    args.base_run = args.base_run.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    stages = ("select", "ablate", "pooled", "oof-gate") if args.stage == "all" else (args.stage,)
    selections = (
        run_selection(args.run_dir)
        if "select" in stages
        else json.loads((args.run_dir / "07_validation" / "VALIDATION_SELECTION.json").read_text(encoding="utf-8"))
    )
    if "ablate" in stages:
        run_ablations(args, selections)
    if "pooled" in stages:
        run_pooled(args)
    if "oof-gate" in stages:
        run_oof_gates(args, selections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
