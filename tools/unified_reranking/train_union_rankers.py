"""Plan or execute fixed CPU LambdaMART/DeepSets Top-15 union experiments."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from tools.unified_reranking.prepare_union_features import TRACK, run_split
from unified_reranking.datasets import (
    FoldPreprocessor,
    build_query_arrays,
    join_development_features_and_labels,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.metrics import compare_selections, evaluate_order_only
from unified_reranking.models import DeepSetsResidualScorer, LightGBMLambdaRank
from unified_reranking.training import (
    FORMAL_SEEDS,
    NeuralTrainingConfig,
    fit_neural_ranker,
    predict_neural_ranker,
    set_deterministic_cpu,
)


ENCODERS = ("lambdamart", "deepsets")
ENCODER_TIE_ORDER = ("lambdamart", "deepsets")
FOLDS = tuple(range(5))


@dataclass(frozen=True)
class UnionBudget:
    """Predeclared fixed budget; no Validation-dependent hyperparameter search."""

    lambdamart_num_leaves: int = 31
    lambdamart_learning_rate: float = 0.05
    lambdamart_n_estimators: int = 200
    deepsets_loss: str = "listwise"
    deepsets_learning_rate: float = 3e-4
    deepsets_weight_decay: float = 1e-4
    deepsets_alpha: float = 0.5
    deepsets_epochs: int = 100
    deepsets_patience: int = 10
    deepsets_batch_size: int = 512

    def validate(self) -> None:
        numeric = (
            self.lambdamart_num_leaves,
            self.lambdamart_learning_rate,
            self.lambdamart_n_estimators,
            self.deepsets_learning_rate,
            self.deepsets_weight_decay,
            self.deepsets_alpha,
            self.deepsets_epochs,
            self.deepsets_patience,
            self.deepsets_batch_size,
        )
        if not np.isfinite(np.asarray(numeric, dtype=float)).all() or any(float(value) < 0 for value in numeric):
            raise ValueError("union budget must be finite and non-negative")
        if self.deepsets_loss != "listwise":
            raise ValueError("the fixed union DeepSets objective is multi-positive listwise")
        if min(
            self.lambdamart_num_leaves,
            self.lambdamart_n_estimators,
            self.deepsets_epochs,
            self.deepsets_patience,
            self.deepsets_batch_size,
        ) <= 0:
            raise ValueError("union iteration/size budgets must be positive")


FORMAL_UNION_BUDGET = UnionBudget()


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def formal_plan(budget: UnionBudget = FORMAL_UNION_BUDGET) -> tuple[dict[str, Any], ...]:
    budget.validate()
    return tuple(
        {
            "encoder": encoder,
            "seed": seed,
            "mode": mode,
            "held_fold": fold,
            "early_stop_fold": (fold + 1) % 5 if fold is not None else 0,
            "budget": asdict(budget),
        }
        for encoder in ENCODERS
        for seed in FORMAL_SEEDS
        for mode, folds in (("oof", FOLDS), ("validation", (None,)))
        for fold in folds
    )


def _load_development(run_dir: Path, split: str) -> tuple[pd.DataFrame, tuple[str, ...], Path, Path]:
    root = run_dir / "03_features" / "tracks" / TRACK / f"union_{split}"
    manifest_path = root / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("configuration", {}).get("primary_union_deduplication") != "NONE":
        raise RuntimeError(f"union feature contract is not locked: {manifest_path}")
    feature_path = Path(manifest["artifacts"]["features"]["path"])
    label_path = Path(manifest["artifacts"]["labels"]["path"])
    if sha256_file(feature_path) != manifest["artifacts"]["features"]["sha256"]:
        raise RuntimeError("union feature hash mismatch")
    if sha256_file(label_path) != manifest["artifacts"]["labels"]["sha256"]:
        raise RuntimeError("union label hash mismatch")
    joined = join_development_features_and_labels(
        pd.read_parquet(feature_path), pd.read_parquet(label_path)
    )
    return joined, tuple(map(str, manifest["model_feature_columns"])), feature_path, label_path


def _flat(arrays: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    valid = ~arrays.padding_mask.numpy()
    features = arrays.features.numpy()[valid]
    labels = arrays.labels.numpy()[valid].astype(np.int32)
    query_ids = [
        sample_id
        for sample_id, candidate_ids in zip(arrays.sample_ids, arrays.candidate_ids, strict=True)
        for _ in candidate_ids
    ]
    return features, labels, query_ids


def _flat_prediction_rows(arrays: Any, scores: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cursor = 0
    for sample_id, candidate_ids in zip(arrays.sample_ids, arrays.candidate_ids, strict=True):
        for candidate_id in candidate_ids:
            rows.append(
                {"sample_id": sample_id, "candidate_id": candidate_id, "score": float(scores[cursor])}
            )
            cursor += 1
    if cursor != len(scores):
        raise RuntimeError("union score vector does not match candidate tensors")
    return pd.DataFrame(rows)


def train_union_cell(
    run_dir: Path,
    *,
    encoder: str,
    seed: int,
    mode: str,
    held_fold: int | None,
    budget: UnionBudget = FORMAL_UNION_BUDGET,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Train one deterministic grouped cell from the predeclared union plan."""

    budget.validate()
    if encoder not in ENCODERS or seed not in FORMAL_SEEDS:
        raise ValueError("union cell encoder/seed is outside the formal plan")
    if (mode == "oof") != (held_fold is not None):
        raise ValueError("OOF requires held_fold; Validation forbids it")
    if held_fold is not None and held_fold not in FOLDS:
        raise ValueError("held_fold must be 0..4")
    run_dir = run_dir.resolve()
    train, columns, train_path, train_label_path = _load_development(run_dir, "train")
    fold_path = run_dir / "04_splits" / "fold_assignments.parquet"
    folds = pd.read_parquet(fold_path, columns=["sample_id", "fold"])
    train = train.merge(folds, on="sample_id", validate="many_to_one")
    early_fold = (int(held_fold) + 1) % 5 if held_fold is not None else 0
    fit = train.loc[(train["fold"] != early_fold) & ((train["fold"] != held_fold) if held_fold is not None else True)].copy()
    early = train.loc[train["fold"] == early_fold].copy()
    if mode == "oof":
        predict = train.loc[train["fold"] == held_fold].copy()
        denominator = folds.loc[folds["fold"].eq(held_fold), "sample_id"].astype(str).tolist()
        predict_path = train_path
        predict_label_path = train_label_path
    else:
        predict, validation_columns, predict_path, predict_label_path = _load_development(run_dir, "validation")
        if validation_columns != columns:
            raise RuntimeError("union Train/Validation feature schemas differ")
        denominator = pd.read_parquet(
            run_dir / "01_manifests" / "paired_validation.parquet",
            columns=["sample_id"],
        )["sample_id"].astype(str).tolist()
    preprocessor = FoldPreprocessor.fit(fit, columns)
    fit_arrays = build_query_arrays(fit, preprocessor=preprocessor, max_candidates=15)
    early_arrays = build_query_arrays(early, preprocessor=preprocessor, max_candidates=15)
    predict_arrays = build_query_arrays(predict, preprocessor=preprocessor, max_candidates=15)
    configuration = {
        "pool": "primary_union_top15_no_dedup",
        "track": TRACK,
        "encoder": encoder,
        "loss": "lambdarank" if encoder == "lambdamart" else budget.deepsets_loss,
        "seed": seed,
        "mode": mode,
        "held_fold": held_fold,
        "early_stop_fold": early_fold,
        "deterministic_cpu": True,
        "route_candidate_identity_preserved": True,
        "budget": asdict(budget),
        "sources": {
            "train_features": _record(train_path),
            "train_labels": _record(train_label_path),
            "prediction_features": _record(predict_path),
            "prediction_labels": _record(predict_label_path),
            "folds": _record(fold_path),
            "implementation_tool": _record(Path(__file__)),
        },
    }
    cell_id = canonical_sha256(configuration)[:16]
    root = (
        output_root
        or run_dir / ("06_oof" if mode == "oof" else "07_validation") / "union_cells"
    ) / cell_id
    marker = root / "manifest.json"
    if marker.exists():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("status") != "COMPLETE" or value.get("configuration") != configuration:
            raise RuntimeError("immutable union cell exists with a different configuration")
        for record in value["artifacts"].values():
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError("resumable union cell artifact hash mismatch")
        return value

    if encoder == "lambdamart":
        fit_x, fit_y, fit_q = _flat(fit_arrays)
        early_x, early_y, early_q = _flat(early_arrays)
        predict_x, _, _ = _flat(predict_arrays)
        model = LightGBMLambdaRank(
            seed=seed,
            num_leaves=budget.lambdamart_num_leaves,
            learning_rate=budget.lambdamart_learning_rate,
            n_estimators=budget.lambdamart_n_estimators,
        ).fit(fit_x, fit_y, fit_q, eval_set=(early_x, early_y, early_q))
        predictions = _flat_prediction_rows(predict_arrays, model.predict(predict_x))
        model_path = root / "model.pkl"
        _atomic_pickle(model_path, model)
        training: dict[str, Any] = model.artifact()
    else:
        set_deterministic_cpu(seed)
        model = DeepSetsResidualScorer(len(columns), alpha=budget.deepsets_alpha)
        training_config = NeuralTrainingConfig(
            loss=budget.deepsets_loss,
            learning_rate=budget.deepsets_learning_rate,
            weight_decay=budget.deepsets_weight_decay,
            alpha=budget.deepsets_alpha,
            epochs=budget.deepsets_epochs,
            patience=budget.deepsets_patience,
            batch_size=budget.deepsets_batch_size,
            seed=seed,
        )
        trained = fit_neural_ranker(model, fit_arrays, early_arrays, config=training_config)
        model.load_state_dict(trained.state_dict)
        predictions = predict_neural_ranker(model, predict_arrays)
        model_path = root / "model.pt"
        _atomic_torch(
            model_path,
            {
                "state_dict": trained.state_dict,
                "encoder": "deepsets",
                "input_dim": len(columns),
                "alpha": budget.deepsets_alpha,
            },
        )
        training = {
            "best_epoch": trained.best_epoch,
            "best_validation_loss": trained.best_validation_loss,
            "epochs_ran": trained.epochs_ran,
            "history": list(trained.history),
        }
    evaluation = predict[["sample_id", "candidate_id", "native_rank", "candidate_success"]].merge(
        predictions, on=["sample_id", "candidate_id"], validate="one_to_one"
    )
    metrics, decisions = evaluate_order_only(
        denominator, evaluation, score_column="score", max_k=15
    )
    metrics["oracle_at_15"] = metrics.pop("oracle_at_5")
    metrics["mrr_at_15"] = metrics.pop("mrr_at_5")
    prediction_path = root / "predictions.parquet"
    decision_path = root / "per_sample_decisions.parquet"
    _atomic_parquet(prediction_path, predictions)
    _atomic_parquet(decision_path, decisions)
    artifacts = {
        "model": _record(model_path),
        "predictions": _record(prediction_path),
        "decisions": _record(decision_path),
    }
    result = {
        "status": "COMPLETE",
        "cell_id": cell_id,
        "configuration": configuration,
        "feature_columns": list(columns),
        "feature_schema_sha256": canonical_sha256(columns),
        "preprocessor": preprocessor.artifact(),
        "training": training,
        "metrics": metrics,
        "artifacts": artifacts,
        "test_access": "NONE",
    }
    atomic_json(marker, result)
    return result


def _ensemble(
    run_dir: Path,
    cells: Iterable[dict[str, Any]],
    *,
    encoder: str,
    split: str,
) -> dict[str, Any]:
    selected = [cell for cell in cells if cell["configuration"]["encoder"] == encoder and cell["configuration"]["mode"] == ("oof" if split == "train" else "validation")]
    expected = len(FORMAL_SEEDS) * (5 if split == "train" else 1)
    if len(selected) != expected:
        raise RuntimeError(f"union ensemble misses cells: {encoder}/{split}")
    features, _, feature_path, label_path = _load_development(run_dir, split)
    predictions = features[["sample_id", "candidate_id", "native_rank", "candidate_success", "base_logit", "source_route", "source_candidate_id", "candidate_geometry_sha256"]].copy()
    sources: list[dict[str, str]] = []
    for seed in FORMAL_SEEDS:
        seed_cells = [cell for cell in selected if cell["configuration"]["seed"] == seed]
        pieces = [pd.read_parquet(cell["artifacts"]["predictions"]["path"]) for cell in seed_cells]
        seed_frame = pd.concat(pieces, ignore_index=True)
        if seed_frame[["sample_id", "candidate_id"]].duplicated().any():
            raise RuntimeError("union seed predictions overlap across OOF folds")
        predictions = predictions.merge(
            seed_frame.rename(columns={"score": f"score_seed_{seed}"}),
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
        )
        sources.extend(
            {
                "path": str(Path(cell["artifacts"]["predictions"]["path"]).parent / "manifest.json"),
                "sha256": sha256_file(Path(cell["artifacts"]["predictions"]["path"]).parent / "manifest.json"),
            }
            for cell in seed_cells
        )
    score_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
    predictions["ensemble_score"] = predictions[score_columns].mean(axis=1)
    denominator_path = run_dir / "01_manifests" / f"paired_{split}.parquet"
    denominator = pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"].astype(str).tolist()
    metrics, decisions = evaluate_order_only(
        denominator,
        predictions,
        score_column="ensemble_score",
        max_k=15,
    )
    metrics["oracle_at_15"] = metrics.pop("oracle_at_5")
    metrics["mrr_at_15"] = metrics.pop("mrr_at_5")
    baseline_metrics, baseline = evaluate_order_only(
        denominator,
        predictions,
        score_column="base_logit",
        max_k=15,
    )
    baseline_oracle_at_15 = float(baseline_metrics["oracle_at_5"])
    comparison = compare_selections(
        baseline,
        decisions,
        oracle_at_5=baseline_oracle_at_15,
    )
    baseline_metrics["oracle_at_15"] = baseline_metrics.pop("oracle_at_5")
    baseline_metrics["mrr_at_15"] = baseline_metrics.pop("mrr_at_5")
    comparison["headroom_recovery_at_15"] = comparison.pop("headroom_recovery_at_5")
    identity = {
        "encoder": encoder,
        "split": split,
        "seeds": list(FORMAL_SEEDS),
        "pool": "primary_union_top15_no_dedup",
    }
    ensemble_id = canonical_sha256(identity)[:16]
    root = run_dir / ("06_oof" if split == "train" else "07_validation") / "union_ensembles" / ensemble_id
    prediction_path = root / "per_candidate_scores.parquet"
    decision_path = root / "per_sample_decisions.parquet"
    _atomic_parquet(prediction_path, predictions)
    _atomic_parquet(decision_path, decisions)
    result = {
        "status": "COMPLETE",
        "identity": identity,
        "metrics": metrics,
        "calibrated_union_baseline_metrics": baseline_metrics,
        "comparison_to_calibrated_union_baseline": comparison,
        "sources": {
            "cells": sources,
            "features": _record(feature_path),
            "labels": _record(label_path),
            "denominator": _record(denominator_path),
        },
        "artifacts": {
            "predictions": _record(prediction_path),
            "decisions": _record(decision_path),
        },
    }
    atomic_json(root / "manifest.json", result)
    return result


def select_union_ranker(run_dir: Path, cells: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select only between the two equal-budget, predeclared encoders."""

    cells = tuple(cells)
    ensembles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for encoder in ENCODERS:
        oof = _ensemble(run_dir, cells, encoder=encoder, split="train")
        validation = _ensemble(run_dir, cells, encoder=encoder, split="validation")
        ensembles[encoder] = {"oof": oof, "validation": validation}
        comparison = validation["comparison_to_calibrated_union_baseline"]
        rows.append(
            {
                "encoder": encoder,
                "validation_j_at_1": validation["metrics"]["j_at_1"],
                "harmful": comparison["harmful"],
                "switch_rate": comparison["switch_rate"],
            }
        )
    order = {name: index for index, name in enumerate(ENCODER_TIE_ORDER)}
    best = max(
        rows,
        key=lambda row: (
            row["validation_j_at_1"],
            -row["harmful"],
            -row["switch_rate"],
            -order[row["encoder"]],
        ),
    )
    encoder = str(best["encoder"])
    trial_ensembles = {
        name: {
            split: _record(
                Path(ensemble[split]["artifacts"]["decisions"]["path"]).parent
                / "manifest.json"
            )
            for split in ("validation", "oof")
        }
        for name, ensemble in ensembles.items()
    }
    selection = {
        "status": "VALIDATION_LOCKED",
        "selected_encoder": encoder,
        "selection_order": [
            "validation_j_at_1",
            "fewer_harmful_vs_calibrated_union_baseline",
            "lower_switch_rate",
            "fixed_encoder_tie_order_lambdamart_then_deepsets",
        ],
        "fixed_budget": asdict(FORMAL_UNION_BUDGET),
        "trials": rows,
        "trial_ensembles": trial_ensembles,
        "selected_validation_manifest": trial_ensembles[encoder]["validation"],
        "selected_oof_manifest": trial_ensembles[encoder]["oof"],
        "selected_validation_ensemble": ensembles[encoder]["validation"],
        "selected_oof_ensemble": ensembles[encoder]["oof"],
        "formal_plan": _record(
            run_dir / "configs" / "union_ranker_frozen_plan.json"
        ),
        "selection_tool": _record(Path(__file__)),
        "test_access": "NONE",
    }
    output = run_dir / "08_lock" / "union_ranker"
    atomic_json(output / "selected_union_ranker.json", selection)
    return selection


def run_orchestrator(
    run_dir: Path,
    *,
    execute: bool = False,
    budget: UnionBudget = FORMAL_UNION_BUDGET,
    allow_active_legacy_worker: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    budget.validate()
    headroom_path = run_dir / "07_validation" / "union_headroom" / "manifest.json"
    headroom = json.loads(headroom_path.read_text(encoding="utf-8"))
    if headroom.get("decision") != "UNION_HEADROOM_AVAILABLE":
        raise RuntimeError("complex union models are not eligible without Validation headroom")
    if budget != FORMAL_UNION_BUDGET:
        # Non-formal budgets are available only to focused tests through train_union_cell.
        raise ValueError("orchestrator accepts only the predeclared formal union budget")
    plan = formal_plan(budget)
    frozen_plan_path = run_dir / "configs" / "union_ranker_frozen_plan.json"
    frozen_plan = {
        "status": "FROZEN",
        "decision": "UNION_HEADROOM_AVAILABLE",
        "cell_count": len(plan),
        "encoders": list(ENCODERS),
        "seeds": list(FORMAL_SEEDS),
        "folds": list(FOLDS),
        "fixed_budget": asdict(budget),
        "cells": list(plan),
        "sources": {
            "headroom": _record(headroom_path),
            "implementation_tool": _record(Path(__file__)),
        },
        "test_access": "NONE",
    }
    frozen_plan["content_sha256"] = canonical_sha256(frozen_plan)
    if frozen_plan_path.is_file():
        existing_frozen = json.loads(frozen_plan_path.read_text(encoding="utf-8"))
        if existing_frozen != frozen_plan:
            raise RuntimeError("immutable union ranker plan differs from current contract")
    else:
        atomic_json(frozen_plan_path, frozen_plan)
    plan_payload = {
        "status": "PLANNED",
        "decision": "UNION_HEADROOM_AVAILABLE",
        "cell_count": len(plan),
        "encoders": list(ENCODERS),
        "seeds": list(FORMAL_SEEDS),
        "folds": list(FOLDS),
        "fixed_budget": asdict(budget),
        "cells": list(plan),
        "sources": {
            "headroom": _record(headroom_path),
            "implementation_tool": _record(Path(__file__)),
            "frozen_plan": _record(frozen_plan_path),
        },
        "execution_authorized": bool(execute),
        "test_access": "NONE",
    }
    plan_path = run_dir / "configs" / "union_ranker_plan.json"
    atomic_json(plan_path, plan_payload)
    if not execute:
        return plan_payload
    active_legacy = subprocess.run(
        ("pgrep", "-f", "reranking.run_experiment_matrix"),
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        active_legacy.returncode == 0
        and active_legacy.stdout.strip()
        and not allow_active_legacy_worker
    ):
        raise RuntimeError(
            "active legacy reranking worker detected; refusing concurrent union training"
        )
    plan_payload["status"] = "EXECUTING"
    atomic_json(plan_path, plan_payload)
    run_split(run_dir, "train")
    run_split(run_dir, "validation")
    cells = [
        train_union_cell(
            run_dir,
            encoder=str(cell["encoder"]),
            seed=int(cell["seed"]),
            mode=str(cell["mode"]),
            held_fold=cell["held_fold"],
            budget=budget,
        )
        for cell in plan
    ]
    selection = select_union_ranker(run_dir, cells)
    plan_payload["status"] = "COMPLETE"
    plan_payload["selection"] = {
        "path": str((run_dir / "08_lock" / "union_ranker" / "selected_union_ranker.json").resolve()),
        "sha256": sha256_file(run_dir / "08_lock" / "union_ranker" / "selected_union_ranker.json"),
        "selected_encoder": selection["selected_encoder"],
    }
    atomic_json(plan_path, plan_payload)
    return plan_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="run all 36 fixed cells; default writes plan only")
    parser.add_argument("--allow-active-legacy-worker", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P9",
        substage="union_ranker_execute" if args.execute else "union_ranker_plan",
        route="cross_route",
        evidence_track=TRACK,
        pool="primary_union_top15_no_dedup",
        method="fixed_lambdamart_and_deepsets",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run_orchestrator(
            run_dir,
            execute=args.execute,
            allow_active_legacy_worker=args.allow_active_legacy_worker,
        )
        plan_path = run_dir / "configs" / "union_ranker_plan.json"
        state["artifact_path"] = str(plan_path.resolve())
        state["artifact_sha256"] = sha256_file(plan_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENCODERS",
    "FORMAL_UNION_BUDGET",
    "UnionBudget",
    "formal_plan",
    "run_orchestrator",
    "select_union_ranker",
    "train_union_cell",
]
