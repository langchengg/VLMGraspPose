#!/usr/bin/env python3
"""Train and select the grouped-OOF Stage-1 strict-P@90 selector."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.calibration import calibration_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.selector_dataset import (  # noqa: E402
    CandidateDatasetPair,
    infer_dataset_split,
    iter_joined_candidate_samples,
)
from src.segmentation.selector_out_of_core import (  # noqa: E402
    grouped_oof_predictions_out_of_core,
)
from src.segmentation.proposal_evaluation import simple_baseline_decisions  # noqa: E402
from src.segmentation.proposal_statistics import paired_transitions  # noqa: E402
from src.segmentation.selective_sam3_vg.io import sha256_file  # noqa: E402
from src.segmentation.selective_sam3_vg.metrics import summarize_ious  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        action="append",
        default=None,
        help="Repeat for each development split; defaults to train plus validation.",
    )
    parser.add_argument(
        "--candidate-labels",
        type=Path,
        action="append",
        default=None,
        help="Repeat in the same order as --features.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/sam3_proposal_bank_p90_v1/selector_training.yaml",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1/stage1_selector",
    )
    parser.add_argument("--selection-split", default="val")
    return parser.parse_args()


def _pickle(path: Path, value) -> None:
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _summarize_decisions(decisions: pd.DataFrame) -> dict[str, dict]:
    return {
        str(method): {
            **summarize_ious(part["candidate_iou"].to_numpy(float)),
            **(
                {"non_deployable_gt_oracle": method == "B12_gt_best_oracle"}
                if str(method).startswith("B")
                else {}
            ),
        }
        for method, part in decisions.groupby("method", sort=True)
    }


def main() -> int:
    args = parse_args()
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    default_features = [
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/features/candidate_features_train.parquet",
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/features/candidate_features_validation.parquet",
    ]
    default_labels = [
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/oracle_stage1_train/candidate_iou.parquet",
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/oracle_stage1/candidate_iou.parquet",
    ]
    feature_paths = args.features or default_features
    label_paths = args.candidate_labels or default_labels
    if len(feature_paths) != len(label_paths):
        raise ValueError("--features and --candidate-labels counts must match")
    pairs = [
        CandidateDatasetPair(feature_path, label_path, infer_dataset_split(feature_path))
        for feature_path, label_path in zip(feature_paths, label_paths, strict=True)
    ]
    if len({pair.split for pair in pairs}) != len(pairs):
        raise ValueError("development candidate tables repeat a split")
    if args.selection_split not in {pair.split for pair in pairs}:
        raise ValueError(f"selection split is absent: {args.selection_split}")

    root = args.artifact_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_decisions, artifacts = grouped_oof_predictions_out_of_core(
        pairs,
        output_path=root / "oof_predictions.parquet",
        folds=int(configuration["folds"]),
        seed=int(configuration["seed"]),
        maximum_training_candidates_per_sample=int(
            configuration["training_candidate_retention"][
                "maximum_candidates_per_sample"
            ]
        ),
    )

    baseline_parts: list[pd.DataFrame] = []
    for sample in iter_joined_candidate_samples(pairs):
        if str(sample.iloc[0]["split"]) != args.selection_split:
            continue
        decisions, _ = simple_baseline_decisions(sample)
        baseline_parts.append(decisions)
    if not baseline_parts:
        raise ValueError(f"selection split is empty: {args.selection_split}")
    baseline_decisions = pd.concat(baseline_parts, ignore_index=True)
    baseline_metrics = _summarize_decisions(baseline_decisions)
    selection_model_decisions = model_decisions[
        model_decisions["split"] == args.selection_split
    ].copy()
    model_metrics = _summarize_decisions(selection_model_decisions)
    baseline_p90 = baseline_decisions[
        baseline_decisions["method"] == "B0_hifi_original"
    ][["sample_id", "candidate_iou"]].rename(
        columns={"candidate_iou": "baseline_iou"}
    )
    model_p90_transitions = {}
    for method, selected in selection_model_decisions.groupby("method", sort=True):
        paired = baseline_p90.merge(
            selected[["sample_id", "candidate_iou"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(baseline_p90):
            raise ValueError(f"incomplete Stage-1 transition pairing: {method}")
        model_p90_transitions[str(method)] = paired_transitions(
            paired["baseline_iou"].to_numpy(float),
            paired["candidate_iou"].to_numpy(float),
            0.90,
        )
    baseline = baseline_metrics["B0_hifi_original"]
    margin = float(configuration["noninferiority_margin_absolute"])
    eligible_models = [
        name
        for name, metrics in model_metrics.items()
        if metrics["p_at_50"] >= baseline["p_at_50"] - margin
        and metrics["p_at_60"] >= baseline["p_at_60"] - margin
    ]
    if not eligible_models:
        selected_name = "HIFI_FALLBACK_NO_NONINFERIOR_MODEL"
    else:
        selected_name = max(
            eligible_models,
            key=lambda name: (
                model_metrics[name]["p_at_90"],
                model_metrics[name]["p_at_80"],
                model_metrics[name]["mean_iou"],
                -model_p90_transitions[name]["harmed"],
                name,
            ),
        )
    model_decisions.to_parquet(root / "oof_selected_candidates.parquet", index=False)
    baseline_decisions.to_parquet(root / "baseline_decisions.parquet", index=False)
    selected_classifier = (
        artifacts["m0_classifier"]
        if selected_name == "M0_logistic"
        else artifacts["m1_classifier"]
    )
    selected_calibrator = (
        artifacts["m0_calibrator"]
        if selected_name == "M0_logistic"
        else artifacts["m1_calibrator"]
    )
    model_payload = {
        "selected_name": selected_name,
        "encoder": artifacts["encoder"],
        "classifier": selected_classifier,
        "regressor": artifacts["m2_regressor"],
        "calibrator": selected_calibrator,
        "strict_y90": "candidate_iou > 0.90",
        "default_fallback": "HIFI_ORIGINAL",
    }
    _pickle(root / "model.pkl", model_payload)
    _pickle(root / "classifier.pkl", selected_classifier)
    _pickle(root / "regressor.pkl", artifacts["m2_regressor"])
    _pickle(root / "calibrator.pkl", selected_calibrator)
    feature_schema = {
        "feature_names": artifacts["encoder"].feature_names,
        "numeric_columns": artifacts["encoder"].numeric_columns,
        "categorical_values": artifacts["encoder"].categorical_values,
        "forbidden_gt_features": True,
    }
    (root / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    training_groups = {
        "group_unit": "exact RGB frame",
        "folds": artifacts["fold_audit"],
        "groups": sorted(artifacts["sample_metadata"]["frame_id"].astype(str).unique()),
    }
    (root / "training_groups.json").write_text(
        json.dumps(training_groups, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    classifier = artifacts["m1_classifier"]
    importance = pd.DataFrame(
        {
            "feature": artifacts["encoder"].feature_names,
            "logistic_abs_coefficient": np.abs(
                artifacts["m0_classifier"].coef_.reshape(-1)
            ),
        }
    ).sort_values("logistic_abs_coefficient", ascending=False)
    # HistGradientBoosting has no native impurity importance; do not fabricate one.
    importance["hgb_native_importance_available"] = hasattr(
        classifier, "feature_importances_"
    )
    importance.to_csv(root / "feature_importance.csv", index=False)
    selection_oof = pd.read_parquet(
        root / "oof_predictions.parquet",
        columns=[
            "split",
            "y90",
            "m0_p90_calibrated",
            "m1_p90_calibrated",
        ],
        filters=[("split", "==", args.selection_split)],
    )
    probabilities = (
        selection_oof["m0_p90_calibrated"].to_numpy(float)
        if selected_name == "M0_logistic"
        else selection_oof["m1_p90_calibrated"].to_numpy(float)
    )
    observed, predicted = calibration_curve(
        selection_oof["y90"].to_numpy(bool),
        probabilities,
        n_bins=10,
        strategy="quantile",
    )
    pd.DataFrame(
        {"mean_predicted_probability": predicted, "observed_fraction": observed}
    ).to_csv(root / "calibration_curve.csv", index=False)
    metadata = artifacts["sample_metadata"]
    selection_metadata = metadata[metadata["split"] == args.selection_split]
    oof_rows = int(pq.ParquetFile(root / "oof_predictions.parquet").metadata.num_rows)
    metrics_payload = {
        "selected_model": selected_name,
        "selection_objective": (
            "lexicographic P@90, P@50/P@60 noninferiority, P@80, mIoU, "
            "minimum harmful P@90 switches; runtime/candidate count are shared"
        ),
        "baseline_metrics": baseline_metrics,
        "model_oof_metrics": model_metrics,
        "model_p90_transitions_vs_hifi": model_p90_transitions,
        "fold_audit": artifacts["fold_audit"],
        "training_subset_audit": artifacts["training_subset_audit"],
        "development_splits": sorted(metadata["split"].astype(str).unique()),
        "selection_split": args.selection_split,
        "oof_eligible_rows": oof_rows,
        "samples": int(metadata["sample_id"].nunique()),
        "groups": int(metadata["frame_id"].nunique()),
        "selection_oof_eligible_rows": len(selection_oof),
        "selection_samples": int(selection_metadata["sample_id"].nunique()),
        "selection_groups": int(selection_metadata["frame_id"].nunique()),
    }
    (root / "validation_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "configuration.yaml").write_text(
        yaml.safe_dump(configuration, sort_keys=True), encoding="utf-8"
    )
    hashes = {
        name: sha256_file(root / name)
        for name in (
            "model.pkl",
            "classifier.pkl",
            "regressor.pkl",
            "calibrator.pkl",
            "feature_schema.json",
            "training_groups.json",
            "oof_predictions.parquet",
            "oof_selected_candidates.parquet",
            "baseline_decisions.parquet",
            "validation_metrics.json",
            "calibration_curve.csv",
            "feature_importance.csv",
            "configuration.yaml",
        )
    }
    (root / "hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics_payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
