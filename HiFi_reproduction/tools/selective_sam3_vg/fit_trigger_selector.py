#!/usr/bin/env python3
"""Fit grouped validation-only trigger and conservative mask selector."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.selective_sam3_vg.io import sha256_file  # noqa: E402
from segmentation.selective_sam3_vg.models import (  # noqa: E402
    deterministic_candidate_score,
    passes_conservative_gate,
)


SEED = 42
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
GATE = {
    "minimum_positive_point_inclusion": 1.0,
    "minimum_probability_mass_recall": 0.75,
    "minimum_area_ratio": 0.40,
    "maximum_area_ratio": 2.00,
    "maximum_low_probability_expansion": 0.75,
    "maximum_fragmentation_penalty": 0.50,
    "minimum_prompt_box_support": 0.95,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--validation-output-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=ROOT / "artifacts/selective_sam3_vg",
    )
    return parser.parse_args()


def _preprocessor(frame: pd.DataFrame, *, scaled: bool) -> ColumnTransformer:
    categorical = ["query_type"]
    numeric = [column for column in frame.columns if column not in categorical]
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scaled:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        sparse_threshold=0.0,
    )


def _trigger_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    rows = []
    for threshold in np.linspace(0.02, 0.98, 193):
        prediction = scores >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(labels, prediction, zero_division=1)),
                "recall": float(recall_score(labels, prediction, zero_division=0)),
                "triggered_fraction": float(np.mean(prediction)),
                "trigger_count": int(np.count_nonzero(prediction)),
            }
        )
    eligible = [
        row for row in rows
        if row["precision"] >= 0.80 and row["triggered_fraction"] >= 0.03
    ]
    if eligible:
        chosen = sorted(
            eligible,
            key=lambda row: (-row["recall"], -row["precision"], row["triggered_fraction"], -row["threshold"]),
        )[0]
        policy = "max_recall_subject_to_precision_ge_0.80_and_trigger_fraction_ge_0.03"
    else:
        for row in rows:
            precision, recall = row["precision"], row["recall"]
            row["f0_5"] = (1.25 * precision * recall / (0.25 * precision + recall)) if precision + recall else 0.0
        chosen = sorted(rows, key=lambda row: (-row["f0_5"], -row["precision"], -row["threshold"]))[0]
        policy = "fallback_max_f0_5"
    return {**chosen, "policy": policy, "grid": rows}


def _summarize(values: np.ndarray) -> dict[str, Any]:
    result = {"mean_iou": float(np.mean(values))}
    for threshold in THRESHOLDS:
        result[f"p_at_{int(threshold*100)}"] = float(np.mean(values > threshold))
    return result


def _selector_result(
    table: pd.DataFrame,
    sample_scores: dict[str, float],
    trigger_threshold: float,
    candidate_scores: np.ndarray,
    *,
    selector_type: str,
    margin: float,
) -> dict[str, Any]:
    working = table.copy()
    working["selector_score_oof"] = candidate_scores
    final_values: list[float] = []
    baseline_values: list[float] = []
    accepted = improved = harmful = 0
    for sample_id, group in working.groupby("sample_id", sort=False):
        group = group.sort_values("candidate_id")
        coarse = group[group["candidate_id"] == "coarse_0"].iloc[0]
        baseline = float(coarse["candidate_iou"])
        baseline_values.append(baseline)
        if sample_scores[sample_id] < trigger_threshold:
            final_values.append(baseline)
            continue
        eligible: list[tuple[float, str, pd.Series]] = []
        for _, candidate in group[group["candidate_id"] != "coarse_0"].iterrows():
            features = {
                key.removeprefix("feature_"): candidate[key]
                for key in candidate.index if key.startswith("feature_")
            }
            valid, _ = passes_conservative_gate(features, GATE)
            if valid:
                eligible.append((float(candidate["selector_score_oof"]), str(candidate["candidate_id"]), candidate))
        if not eligible:
            final_values.append(baseline)
            continue
        _, _, best = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
        gain = float(best["selector_score_oof"] - coarse["selector_score_oof"])
        if gain <= margin:
            final_values.append(baseline)
            continue
        value = float(best["candidate_iou"])
        final_values.append(value)
        accepted += 1
        improved += int(value > baseline)
        harmful += int(value < baseline)
    baseline_array = np.asarray(baseline_values)
    final_array = np.asarray(final_values)
    baseline_metrics = _summarize(baseline_array)
    final_metrics = _summarize(final_array)
    return {
        "selector_type": selector_type,
        "margin": float(margin),
        "sample_count": len(final_array),
        "accepted_count": accepted,
        "accepted_improvement_count": improved,
        "accepted_harmful_count": harmful,
        "baseline": baseline_metrics,
        "hybrid": final_metrics,
        "delta_mean_iou": final_metrics["mean_iou"] - baseline_metrics["mean_iou"],
        **{f"delta_p_at_{int(t*100)}": final_metrics[f"p_at_{int(t*100)}"] - baseline_metrics[f"p_at_{int(t*100)}"] for t in THRESHOLDS},
    }


def _write_artifact(directory: Path, files: dict[str, Any], binary: dict[str, Any] | None = None) -> None:
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite locked artifact directory: {directory}")
    directory.mkdir(parents=True)
    for name, value in files.items():
        path = directory / name
        if name.endswith((".yaml", ".yml")):
            path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
        else:
            path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for name, value in (binary or {}).items():
        with (directory / name).open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir()) if path.name != "hashes.json"
    }
    (directory / "hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _arguments()
    candidates = pd.read_parquet(args.candidate_evaluation)
    coarse_rows = candidates[candidates["candidate_id"] == "coarse_0"].copy()
    if len(coarse_rows) != 3778 or coarse_rows["scene_id"].nunique() != 165:
        raise RuntimeError("training requires the complete authoritative validation split")
    best = candidates[candidates["candidate_id"] != "coarse_0"].groupby("sample_id")["candidate_iou"].max()
    coarse_iou = coarse_rows.set_index("sample_id")["candidate_iou"]
    best = best.reindex(coarse_iou.index)
    crossing = np.zeros(len(coarse_iou), dtype=bool)
    for threshold in THRESHOLDS:
        crossing |= (coarse_iou.to_numpy() <= threshold) & (best.to_numpy() > threshold)
    labels = ((coarse_iou.to_numpy() >= 0.60) & (((best - coarse_iou).to_numpy() >= 0.02) | crossing)).astype(int)

    feature_rows = []
    for row in coarse_rows.itertuples(index=False):
        path = args.validation_output_root / row.sample_id / row.prompt_family / "pre_sam_features.json"
        features = json.loads(path.read_text(encoding="utf-8"))
        feature_rows.append({"sample_id": row.sample_id, "scene_id": row.scene_id, **features})
    pre = pd.DataFrame(feature_rows).set_index("sample_id").reindex(coarse_iou.index)
    raw_feature_names = [column for column in pre.columns if column != "scene_id"]
    X = pre[raw_feature_names]
    groups = pre["scene_id"].astype(str).to_numpy()
    folds = GroupKFold(n_splits=5)
    trigger_candidates = {
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor(X, scaled=True)),
                ("classifier", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=SEED)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", _preprocessor(X, scaled=False)),
                ("classifier", HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0, class_weight="balanced", random_state=SEED)),
            ]
        ),
    }
    trigger_results = []
    trigger_oof: dict[str, np.ndarray] = {}
    for name, estimator in trigger_candidates.items():
        scores = np.zeros(len(X), dtype=np.float64)
        for train_index, validation_index in folds.split(X, labels, groups):
            fitted = clone(estimator).fit(X.iloc[train_index], labels[train_index])
            scores[validation_index] = fitted.predict_proba(X.iloc[validation_index])[:, 1]
        calibration = _trigger_threshold(scores, labels)
        trigger_oof[name] = scores
        trigger_results.append(
            {
                "model": name,
                "pr_auc_oof": float(average_precision_score(labels, scores)),
                "calibration": {key: value for key, value in calibration.items() if key != "grid"},
            }
        )
    trigger_results.sort(key=lambda row: (-row["pr_auc_oof"], row["model"]))
    selected_trigger = trigger_results[0]
    trigger_name = selected_trigger["model"]
    trigger_scores = trigger_oof[trigger_name]
    trigger_threshold = float(selected_trigger["calibration"]["threshold"])
    final_trigger = clone(trigger_candidates[trigger_name]).fit(X, labels)
    trigger_artifact = {
        "artifact_type": "selective_sam3_pre_trigger",
        "model_name": trigger_name,
        "estimator": final_trigger,
        "raw_feature_names": raw_feature_names,
        "threshold": trigger_threshold,
        "seed": SEED,
    }

    feature_columns = [
        column for column in candidates.columns
        if column.startswith("feature_")
        and column not in {"feature_candidate_box_xyxy", "feature_bbox_xyxy"}
        and pd.api.types.is_numeric_dtype(candidates[column])
    ]
    candidate_X = candidates[feature_columns].astype(float).rename(
        columns={column: column.removeprefix("feature_") for column in feature_columns}
    )
    candidate_groups = candidates["scene_id"].astype(str).to_numpy()
    target = candidates["candidate_iou"].astype(float).to_numpy()
    sample_weights = 1.0 / candidates.groupby("sample_id")["sample_id"].transform("count").to_numpy()
    regressor = HistGradientBoostingRegressor(max_iter=250, l2_regularization=1.0, random_state=SEED)
    regression_oof = np.zeros(len(candidates), dtype=np.float64)
    for train_index, validation_index in folds.split(candidate_X, target, candidate_groups):
        fitted = clone(regressor).fit(candidate_X.iloc[train_index], target[train_index], sample_weight=sample_weights[train_index])
        regression_oof[validation_index] = fitted.predict(candidate_X.iloc[validation_index])
    deterministic_scores = np.asarray(
        [
            deterministic_candidate_score(
                {key.removeprefix("feature_"): row[key] for key in feature_columns}
            )
            for _, row in candidates.iterrows()
        ],
        dtype=np.float64,
    )
    sample_trigger_scores = dict(zip(coarse_iou.index, trigger_scores, strict=True))
    selector_comparisons = []
    for margin in np.linspace(0.0, 0.20, 21):
        selector_comparisons.append(
            _selector_result(candidates, sample_trigger_scores, trigger_threshold, deterministic_scores, selector_type="deterministic_rule", margin=float(margin))
        )
    for margin in np.linspace(0.0, 0.05, 21):
        selector_comparisons.append(
            _selector_result(candidates, sample_trigger_scores, trigger_threshold, regression_oof, selector_type="hist_gradient_boosting_regressor", margin=float(margin))
        )
    eligible = [
        item for item in selector_comparisons
        if item["delta_p_at_50"] >= -0.001 and item["delta_p_at_60"] >= -0.001
    ]
    if not eligible:
        raise RuntimeError("no selector configuration satisfies the P@50/P@60 safety constraints")
    eligible.sort(
        key=lambda item: (
            -item["hybrid"]["p_at_90"], -item["hybrid"]["p_at_80"],
            -item["hybrid"]["p_at_70"], -item["hybrid"]["mean_iou"],
            item["accepted_harmful_count"], item["selector_type"], item["margin"],
        )
    )
    selected_selector = eligible[0]
    selector_type = selected_selector["selector_type"]
    final_regressor = None
    if selector_type == "hist_gradient_boosting_regressor":
        final_regressor = clone(regressor).fit(candidate_X, target, sample_weight=sample_weights)
    selector_artifact = {
        "artifact_type": "selective_sam3_candidate_selector",
        "selector_type": selector_type,
        "estimator": final_regressor,
        "feature_names": list(candidate_X.columns),
        "acceptance_margin": float(selected_selector["margin"]),
        "conservative_gate": GATE,
        "seed": SEED,
    }

    artifact_root = args.artifact_root
    trigger_directory = artifact_root / "locked_trigger"
    selector_directory = artifact_root / "locked_selector"
    _write_artifact(
        trigger_directory,
        {
            "feature_schema.json": {"raw_feature_names": raw_feature_names, "forbidden_features": ["gt_iou", "delta_iou", "threshold_outcome"]},
            "calibration.json": {"selected": selected_trigger, "all_models": trigger_results},
            "training_groups.json": {"group_key": "scene_id", "unique_groups": sorted(set(groups)), "group_count": len(set(groups)), "folds": 5},
            "config.yaml": {"seed": SEED, "target": "boundary_recoverable", "threshold": trigger_threshold},
        },
        {"model.pkl": trigger_artifact},
    )
    selector_files = {
        "feature_schema.json": {"feature_names": selector_artifact["feature_names"], "forbidden_features": ["gt_iou", "delta_iou", "threshold_outcome"]},
        "acceptance_margin.json": {"acceptance_margin": selector_artifact["acceptance_margin"], "conservative_gate": GATE},
        "validation_results.json": {"selected": selected_selector, "all_configurations": selector_comparisons},
        "config.yaml": {"seed": SEED, "selector_type": selector_type},
    }
    if selector_type == "deterministic_rule":
        selector_files["rules.json"] = {"score": "deterministic_candidate_score_v1", "conservative_gate": GATE, "acceptance_margin": selector_artifact["acceptance_margin"]}
    _write_artifact(selector_directory, selector_files, {"model.pkl": selector_artifact})
    diagnostics = artifact_root / "validation_training"
    diagnostics.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": coarse_iou.index,
            "scene_id": groups,
            "boundary_recoverable_label": labels,
            "trigger_score_oof": trigger_scores,
            "triggered": trigger_scores >= trigger_threshold,
        }
    ).to_parquet(diagnostics / "trigger_oof_predictions.parquet", index=False)
    candidates.assign(selector_regression_oof=regression_oof, selector_deterministic_score=deterministic_scores).to_parquet(
        diagnostics / "selector_oof_predictions.parquet", index=False
    )
    result = {
        "status": "COMPLETED",
        "seed": SEED,
        "validation_samples": len(coarse_iou),
        "validation_scenes": len(set(groups)),
        "positive_trigger_labels": int(np.count_nonzero(labels)),
        "trigger": selected_trigger,
        "selector": selected_selector,
    }
    (diagnostics / "training_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
