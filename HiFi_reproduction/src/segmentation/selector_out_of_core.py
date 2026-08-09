"""Out-of-core grouped OOF training that scores every eligible candidate."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from .p90_selector import (
    FeatureEncoder,
    _models,
    inverse_candidate_weights,
    p90_labels,
    selector_metrics,
)
from .selector_dataset import (
    CandidateDatasetPair,
    deterministic_training_subset,
    iter_joined_candidate_samples,
)


def _fit_models(frame: pd.DataFrame, seed: int) -> tuple[Any, Any, Any, FeatureEncoder]:
    encoder = FeatureEncoder.fit(frame)
    x = encoder.transform(frame)
    y_iou = frame["candidate_iou"].to_numpy(dtype=np.float64)
    y90 = p90_labels(y_iou)
    if len(np.unique(y90)) != 2:
        raise ValueError("selector training subset must contain both y90 classes")
    m0, m1, m2 = _models(seed)
    weights = inverse_candidate_weights(frame, y90)
    m0.fit(x, y90, sample_weight=weights)
    m1.fit(x, y90, sample_weight=weights)
    regression_weights = (
        1.0
        / frame.groupby("sample_id")["candidate_id"]
        .transform("count")
        .to_numpy(float)
    )
    regression_weights *= len(regression_weights) / regression_weights.sum()
    m2.fit(x, y_iou, sample_weight=regression_weights)
    return m0, m1, m2, encoder


def _raw_path(root: Path, sample_id: str) -> Path:
    digest = sample_id.split("_", 1)[-1]
    return root / digest[:2] / f"{sample_id}.parquet"


def grouped_oof_predictions_out_of_core(
    pairs: list[CandidateDatasetPair],
    *,
    output_path: Path,
    folds: int = 5,
    seed: int = 42,
    maximum_training_candidates_per_sample: int = 64,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit bounded row subsets while producing OOF scores for the full bank."""

    subsets: list[pd.DataFrame] = []
    sample_rows: list[dict[str, Any]] = []
    eligible_by_split: dict[str, int] = {}
    retained_by_split: dict[str, int] = {}
    positive_by_split: dict[str, int] = {}
    retained_positive_by_split: dict[str, int] = {}
    for sample in iter_joined_candidate_samples(pairs):
        eligible = sample[sample["eligible_final"]].copy()
        retained = deterministic_training_subset(
            sample, maximum_candidates=maximum_training_candidates_per_sample
        )
        split = str(sample.iloc[0]["split"])
        sample_id = str(sample.iloc[0]["sample_id"])
        frame_id = str(sample.iloc[0]["frame_id"])
        scene_id = str(sample.iloc[0]["scene_id"])
        subsets.append(retained)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "frame_id": frame_id,
                "scene_id": scene_id,
                "eligible_candidates": len(eligible),
                "retained_training_candidates": len(retained),
            }
        )
        eligible_by_split[split] = eligible_by_split.get(split, 0) + len(eligible)
        retained_by_split[split] = retained_by_split.get(split, 0) + len(retained)
        positive_by_split[split] = positive_by_split.get(split, 0) + int(
            (eligible["candidate_iou"] > 0.90).sum()
        )
        retained_positive_by_split[split] = retained_positive_by_split.get(
            split, 0
        ) + int((retained["candidate_iou"] > 0.90).sum())
    metadata = pd.DataFrame(sample_rows)
    if metadata["frame_id"].nunique() < folds:
        raise ValueError("insufficient exact-frame groups for GroupKFold")
    if metadata["sample_id"].duplicated().any():
        raise ValueError("duplicate development sample identity")
    subset = pd.concat(subsets, ignore_index=True)
    del subsets
    fold_by_sample: dict[str, int] = {}
    splitter = GroupKFold(n_splits=int(folds))
    group_values = metadata["frame_id"].astype(str).to_numpy()
    fold_audit: list[dict[str, Any]] = []
    fold_splits = list(splitter.split(metadata, groups=group_values))
    for fold, (train_samples, validation_samples) in enumerate(fold_splits):
        train_groups = set(group_values[train_samples])
        validation_groups = set(group_values[validation_samples])
        if train_groups & validation_groups:
            raise RuntimeError("GroupKFold exact-frame leakage")
        for sample_id in metadata.iloc[validation_samples]["sample_id"].astype(str):
            fold_by_sample[sample_id] = fold
        fold_audit.append(
            {
                "fold": fold,
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "train_retained_rows": int(
                    metadata.iloc[train_samples][
                        "retained_training_candidates"
                    ].sum()
                ),
                "validation_retained_rows": int(
                    metadata.iloc[validation_samples][
                        "retained_training_candidates"
                    ].sum()
                ),
                "validation_oof_rows": int(
                    metadata.iloc[validation_samples]["eligible_candidates"].sum()
                ),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
            }
        )
    subset["fold"] = subset["sample_id"].astype(str).map(fold_by_sample).astype(int)

    fold_models: list[tuple[Any, Any, Any, FeatureEncoder]] = []
    for fold in range(int(folds)):
        training = subset[subset["fold"] != fold].drop(columns="fold")
        fold_models.append(_fit_models(training, seed + fold))

    work_root = output_path.parent / f".{output_path.stem}.work.{os.getpid()}"
    if work_root.exists():
        raise FileExistsError(f"selector work root already exists: {work_root}")
    raw_root = work_root / "raw"
    raw_root.mkdir(parents=True)
    calibration_raw0: list[np.ndarray] = []
    calibration_raw1: list[np.ndarray] = []
    calibration_y: list[np.ndarray] = []
    calibration_fold: list[np.ndarray] = []
    raw_paths: list[Path] = []
    for sample in iter_joined_candidate_samples(pairs):
        data = sample[sample["eligible_final"]].copy().reset_index(drop=True)
        sample_id = str(data.iloc[0]["sample_id"])
        fold = int(fold_by_sample[sample_id])
        m0, m1, m2, encoder = fold_models[fold]
        x = encoder.transform(data)
        raw0 = m0.predict_proba(x)[:, 1]
        raw1 = m1.predict_proba(x)[:, 1]
        predicted_iou = np.clip(m2.predict(x), 0.0, 1.0)
        raw = data[
            [
                "sample_id",
                "candidate_id",
                "split",
                "frame_id",
                "scene_id",
                "source_family",
                "candidate_iou",
            ]
        ].copy()
        raw["y90"] = data["candidate_iou"].to_numpy(float) > 0.90
        raw["fold"] = fold
        raw["m0_p90_raw"] = raw0
        raw["m1_p90_raw"] = raw1
        raw["m2_predicted_iou"] = predicted_iou
        path = _raw_path(raw_root, sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(path, index=False)
        raw_paths.append(path)

        retained_ids = set(
            deterministic_training_subset(
                sample,
                maximum_candidates=maximum_training_candidates_per_sample,
            )["candidate_id"].astype(str)
        )
        calibration_mask = data["candidate_id"].astype(str).isin(retained_ids).to_numpy()
        calibration_raw0.append(raw0[calibration_mask])
        calibration_raw1.append(raw1[calibration_mask])
        calibration_y.append(raw.loc[calibration_mask, "y90"].to_numpy(bool))
        calibration_fold.append(
            np.full(int(np.count_nonzero(calibration_mask)), fold, dtype=np.int16)
        )

    raw0_values = np.concatenate(calibration_raw0)
    raw1_values = np.concatenate(calibration_raw1)
    y_values = np.concatenate(calibration_y).astype(np.int8)
    fold_values = np.concatenate(calibration_fold)
    del calibration_raw0, calibration_raw1, calibration_y, calibration_fold
    fold_calibrators: list[tuple[IsotonicRegression, IsotonicRegression]] = []
    for fold in range(int(folds)):
        train = fold_values != fold
        fold_calibrators.append(
            (
                IsotonicRegression(out_of_bounds="clip").fit(
                    raw0_values[train], y_values[train]
                ),
                IsotonicRegression(out_of_bounds="clip").fit(
                    raw1_values[train], y_values[train]
                ),
            )
        )
    calibrator_m0 = IsotonicRegression(out_of_bounds="clip").fit(
        raw0_values, y_values
    )
    calibrator_m1 = IsotonicRegression(out_of_bounds="clip").fit(
        raw1_values, y_values
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    decisions: list[pd.DataFrame] = []
    try:
        for path in raw_paths:
            frame = pd.read_parquet(path)
            fold = int(frame.iloc[0]["fold"])
            calibrator0, calibrator1 = fold_calibrators[fold]
            frame["m0_p90_calibrated"] = calibrator0.predict(
                frame["m0_p90_raw"].to_numpy(float)
            )
            frame["m1_p90_calibrated"] = calibrator1.predict(
                frame["m1_p90_raw"].to_numpy(float)
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_output,
                    table.schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            writer.write_table(table)
            _, sample_decisions = selector_metrics(frame)
            decisions.append(sample_decisions)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("OOF writer did not receive candidate rows")
    temporary_output.replace(output_path)
    shutil.rmtree(work_root)

    full_m0, full_m1, full_m2, full_encoder = _fit_models(
        subset.drop(columns="fold"), seed
    )
    training_audit = {
        "maximum_training_candidates_per_sample": maximum_training_candidates_per_sample,
        "eligible_candidates_by_split": eligible_by_split,
        "retained_training_candidates_by_split": retained_by_split,
        "strict_positive_candidates_by_split": positive_by_split,
        "retained_strict_positive_candidates_by_split": retained_positive_by_split,
        "oof_scored_all_eligible_candidates": True,
        "retention_uses_gt_on_development_only": True,
    }
    artifacts = {
        "encoder": full_encoder,
        "m0_classifier": full_m0,
        "m1_classifier": full_m1,
        "m2_regressor": full_m2,
        "m0_calibrator": calibrator_m0,
        "m1_calibrator": calibrator_m1,
        "fold_audit": fold_audit,
        "training_subset_audit": training_audit,
        "sample_metadata": metadata,
    }
    return pd.concat(decisions, ignore_index=True), artifacts


__all__ = ["grouped_oof_predictions_out_of_core"]
