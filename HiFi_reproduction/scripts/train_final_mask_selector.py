#!/usr/bin/env python3
"""Train the grouped-OOF Stage-2 selector and validation-locked HiFi gate."""

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

from src.segmentation.conservative_mask_gate import (  # noqa: E402
    proposed_alternatives,
    tune_gate,
)
from src.segmentation.proposal_evaluation import deterministic_rule_score  # noqa: E402
from src.segmentation.proposal_types import load_candidate_masks_npz  # noqa: E402
from src.segmentation.selector_dataset import (  # noqa: E402
    CandidateDatasetPair,
    infer_dataset_split,
    iter_joined_candidate_samples_with_oof,
)
from src.segmentation.selector_out_of_core import (  # noqa: E402
    grouped_oof_predictions_out_of_core,
)
from src.segmentation.selective_sam3_vg.evaluation import (  # noqa: E402
    load_frozen_ground_truth_manifest,
    load_ground_truth_mask,
)
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    resize_binary_mask,
    sha256_file,
)
from src.segmentation.selective_sam3_vg.metrics import boundary_fscores  # noqa: E402


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
        "--selector-config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/sam3_proposal_bank_p90_v1/selector_training.yaml",
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/sam3_proposal_bank_p90_v1/conservative_gate.yaml",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1/final_selector",
    )
    parser.add_argument(
        "--stage2-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/stage2",
    )
    parser.add_argument("--selection-split", default="val")
    return parser.parse_args()


def _pickle(path: Path, value) -> None:
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _importance(artifacts: dict) -> pd.DataFrame:
    coefficients = np.abs(artifacts["m0_classifier"].coef_.reshape(-1))
    return pd.DataFrame(
        {
            "feature": artifacts["encoder"].feature_names,
            "logistic_abs_coefficient": coefficients,
            "hgb_native_importance_available": False,
        }
    ).sort_values("logistic_abs_coefficient", ascending=False)


def _attach_validation_boundary_fscores(
    evidence: pd.DataFrame,
    *,
    stage2_root: Path,
) -> pd.DataFrame:
    """Attach GT-only validation labels for the final gate objective."""

    gt_rows = load_frozen_ground_truth_manifest(
        PROJECT_ROOT / "artifacts/data_audit/frozen_manifests/ocidvlg_unique_val.json",
        hifics_root=PROJECT_ROOT / "hifics",
        expected_count=3778,
    )
    gt_by_prefix = {f"q{int(row['question_index']):07d}_": row for row in gt_rows}
    values: dict[tuple[str, str], float] = {}
    for sample_id, sample in evidence.groupby("sample_id", sort=False):
        prefix = str(sample_id).split("_", 1)[0] + "_"
        target = load_ground_truth_mask(gt_by_prefix[prefix])
        masks = load_candidate_masks_npz(
            stage2_root / str(sample_id) / "candidate_masks.npz"
        )
        candidate_ids = list(dict.fromkeys(sample["candidate_id"].astype(str)))
        aligned = [
            resize_binary_mask(masks[candidate_id], target.shape)
            for candidate_id in candidate_ids
        ]
        scores = boundary_fscores(aligned, target, tolerance_px=2)
        values.update(
            {
                (str(sample_id), candidate_id): float(score)
                for candidate_id, score in zip(candidate_ids, scores, strict=True)
            }
        )
    result = evidence.copy()
    result["validation_boundary_fscore"] = [
        values[(str(sample_id), str(candidate_id))]
        for sample_id, candidate_id in zip(
            result["sample_id"], result["candidate_id"], strict=True
        )
    ]
    return result


def main() -> int:
    args = parse_args()
    selector_config = yaml.safe_load(args.selector_config.read_text(encoding="utf-8"))
    gate_config = yaml.safe_load(args.gate_config.read_text(encoding="utf-8"))
    default_features = [
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/features_stage2/candidate_features_train.parquet",
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/features_stage2/candidate_features_validation.parquet",
    ]
    default_labels = [
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/oracle_stage2_train/candidate_iou.parquet",
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/oracle_stage2/candidate_iou.parquet",
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
        raise ValueError("Stage-2 development candidate tables repeat a split")
    if args.selection_split not in {pair.split for pair in pairs}:
        raise ValueError(f"selection split is absent: {args.selection_split}")

    root = args.artifact_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_decisions, artifacts = grouped_oof_predictions_out_of_core(
        pairs,
        output_path=root / "oof_predictions.parquet",
        folds=int(selector_config["folds"]),
        seed=int(selector_config["seed"]),
        maximum_training_candidates_per_sample=int(
            selector_config["training_candidate_retention"][
                "maximum_candidates_per_sample"
            ]
        ),
    )

    methods = ("F0_deterministic", "F1_hgb_classifier", "F2_classifier_regressor")
    reduced_evidence: list[pd.DataFrame] = []
    proposed_parts: dict[str, list[pd.DataFrame]] = {method: [] for method in methods}
    for evidence in iter_joined_candidate_samples_with_oof(
        pairs, root / "oof_predictions.parquet"
    ):
        if str(evidence.iloc[0]["split"]) != args.selection_split:
            continue
        evidence["deterministic_rule_score"] = deterministic_rule_score(evidence)
        fallback = evidence[
            evidence["source_family"].isin(
                {"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"}
            )
        ]
        if len(fallback) != 1:
            raise ValueError(
                f"Stage-2 sample requires one fallback: {evidence.iloc[0]['sample_id']}"
            )
        sample_rows = [fallback]
        for method in methods:
            proposed = proposed_alternatives(evidence, method)
            proposed_parts[method].append(proposed)
            sample_rows.append(proposed)
        reduced_evidence.append(
            pd.concat(sample_rows, ignore_index=True)
            .drop_duplicates(["sample_id", "candidate_id"], keep="first")
        )
    if not reduced_evidence:
        raise ValueError(f"selection split is empty: {args.selection_split}")
    selection_evidence = pd.concat(reduced_evidence, ignore_index=True)
    if args.selection_split != "val":
        raise ValueError("formal gate Boundary F-score selection is validation-only")
    selection_evidence = _attach_validation_boundary_fscores(
        selection_evidence,
        stage2_root=args.stage2_root.expanduser().resolve(),
    )
    proposed_by_method = {
        method: pd.concat(parts, ignore_index=True)
        for method, parts in proposed_parts.items()
    }
    grid = {name: values for name, values in gate_config["validation_grid"].items()}
    results: dict[str, dict] = {}
    decisions = []
    thresholds_by_method = {}
    for method in methods:
        proposed = proposed_by_method[method]
        thresholds, chosen, metrics = tune_gate(
            selection_evidence,
            proposed,
            grid=grid,
            noninferiority_margin=float(gate_config["noninferiority_margin_absolute"]),
        )
        chosen["method"] = method
        decisions.append(chosen)
        thresholds_by_method[method] = thresholds.to_dict()
        results[method] = metrics
    eligible = [
        method
        for method, result in results.items()
        if not result.get("selected_fallback_only", False)
    ]
    if eligible:
        selected_method = max(
            eligible,
            key=lambda method: (
                results[method]["metrics"]["p_at_90"],
                -results[method]["transitions_p90"]["harmed"],
                results[method]["metrics"]["mean_iou"],
                results[method]["mean_validation_boundary_fscore"],
                results[method]["metrics"]["p_at_80"],
                -results[method]["accepted"],
                method,
            ),
        )
    else:
        selected_method = "HIFI_FALLBACK_NO_NONINFERIOR_FINAL_GATE"
    model_decisions.to_parquet(root / "oof_selected_candidates.parquet", index=False)
    combined_decisions = pd.concat(decisions, ignore_index=True)
    combined_decisions.to_parquet(root / "oof_gate_decisions.parquet", index=False)
    selected_thresholds = (
        thresholds_by_method[selected_method]
        if selected_method in thresholds_by_method
        else {**thresholds_by_method["F1_hgb_classifier"], "minimum_p90_margin": 2.0}
    )
    gate_payload = {
        "selected_method": selected_method,
        "thresholds": selected_thresholds,
        "default_output": "STAGE2_HIFI_FALLBACK",
        "strict_p90": "candidate_iou > 0.90",
        "NO_TEST_GT_USED_FOR_SELECTION": True,
    }
    (root / "gate.json").write_text(
        json.dumps(gate_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _pickle(root / "classifier.pkl", artifacts["m1_classifier"])
    _pickle(root / "regressor.pkl", artifacts["m2_regressor"])
    _pickle(root / "calibrator.pkl", artifacts["m1_calibrator"])
    _pickle(
        root / "model.pkl",
        {
            "selected_method": selected_method,
            "encoder": artifacts["encoder"],
            "m0_classifier": artifacts["m0_classifier"],
            "m0_calibrator": artifacts["m0_calibrator"],
            "classifier": artifacts["m1_classifier"],
            "regressor": artifacts["m2_regressor"],
            "calibrator": artifacts["m1_calibrator"],
            "gate": gate_payload,
        },
    )
    schema = {
        "feature_names": artifacts["encoder"].feature_names,
        "numeric_columns": artifacts["encoder"].numeric_columns,
        "categorical_values": artifacts["encoder"].categorical_values,
        "forbidden_gt_features": True,
    }
    (root / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    importance = _importance(artifacts)
    importance.to_csv(root / "feature_importance.csv", index=False)
    metadata = artifacts["sample_metadata"]
    training_groups = {
        "group_unit": "exact RGB frame",
        "folds": artifacts["fold_audit"],
        "groups": sorted(metadata["frame_id"].astype(str).unique()),
    }
    (root / "training_groups.json").write_text(
        json.dumps(training_groups, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selection_oof = pd.read_parquet(
        root / "oof_predictions.parquet",
        columns=["split", "y90", "m1_p90_calibrated"],
        filters=[("split", "==", args.selection_split)],
    )
    observed, predicted = calibration_curve(
        selection_oof["y90"].to_numpy(bool),
        selection_oof["m1_p90_calibrated"].to_numpy(float),
        n_bins=10,
        strategy="quantile",
    )
    pd.DataFrame(
        {"mean_predicted_probability": predicted, "observed_fraction": observed}
    ).to_csv(root / "calibration_curve.csv", index=False)
    selection_metadata = metadata[metadata["split"] == args.selection_split]
    oof_rows = int(pq.ParquetFile(root / "oof_predictions.parquet").metadata.num_rows)
    selection_eligible_rows = int(
        artifacts["training_subset_audit"]["eligible_candidates_by_split"][
            args.selection_split
        ]
    )
    metrics_payload = {
        "selected_method": selected_method,
        "gate_selection_objective": (
            "lexicographic P@90, P@50/P@60 noninferiority, protect baseline "
            "P@90 successes, mIoU, validation Boundary F-score, P@80, "
            "minimum accepted fraction"
        ),
        "method_results": results,
        "thresholds_by_method": thresholds_by_method,
        "samples": int(metadata["sample_id"].nunique()),
        "oof_eligible_rows": oof_rows,
        "groups": int(metadata["frame_id"].nunique()),
        "fold_audit": artifacts["fold_audit"],
        "training_subset_audit": artifacts["training_subset_audit"],
        "development_splits": sorted(metadata["split"].astype(str).unique()),
        "selection_split": args.selection_split,
        "selection_samples": int(selection_metadata["sample_id"].nunique()),
        "selection_oof_eligible_rows": selection_eligible_rows,
        "selection_gate_evidence_rows": len(selection_evidence),
        "selection_groups": int(selection_metadata["frame_id"].nunique()),
    }
    (root / "validation_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    combined_config = {"selector": selector_config, "gate": gate_config}
    (root / "configuration.yaml").write_text(
        yaml.safe_dump(combined_config, sort_keys=True), encoding="utf-8"
    )
    hash_names = (
        "model.pkl",
        "classifier.pkl",
        "regressor.pkl",
        "calibrator.pkl",
        "gate.json",
        "feature_schema.json",
        "training_groups.json",
        "oof_predictions.parquet",
        "oof_selected_candidates.parquet",
        "oof_gate_decisions.parquet",
        "validation_metrics.json",
        "calibration_curve.csv",
        "feature_importance.csv",
        "configuration.yaml",
    )
    hashes = {name: sha256_file(root / name) for name in hash_names}
    (root / "hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics_payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
