from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .labels import label_lookup, validate_inference_label_join
from .schema import (
    SCALAR_FEATURES,
    assert_inference_record_has_no_evaluation_fields,
    assert_model_feature_names,
    read_jsonl,
    stable_sample_id,
)


RELATIONAL_FEATURES = (
    "q_margin_from_top1",
    "q_entropy",
    "original_rank_fraction",
    "width_log_ratio_to_top1",
    "axial_angle_difference_to_top1",
    "candidate_rectangle_iou_with_top1",
)


def feature_value(candidate: dict[str, Any], name: str) -> tuple[float, float, float]:
    feature = candidate.get("features", {}).get(name, {})
    raw = feature.get("value")
    reliability = float(feature.get("reliability", 0.0) or 0.0)
    missing = raw is None or not np.isfinite(raw) or reliability <= 0.0
    value = 0.0 if missing else float(raw)
    return value, float(np.clip(reliability, 0.0, 1.0)), float(missing)


def scalar_candidate_vector(
    candidate: dict[str, Any],
    *,
    fields: tuple[str, ...] = SCALAR_FEATURES,
) -> np.ndarray:
    assert_model_feature_names(fields)
    values = []
    for field in fields:
        values.extend(feature_value(candidate, field))
    return np.asarray(values, dtype=np.float32)


def axial_difference_fraction(angle_a: float, angle_b: float) -> float:
    difference = abs(((float(angle_a) - float(angle_b) + 90.0) % 180.0) - 90.0)
    return difference / 90.0


def _polygon_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    shape = (480, 640)
    left = np.zeros(shape, dtype=np.uint8)
    right = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(left, [np.asarray(first["polygon"], dtype=np.int32)], 1)
    cv2.fillPoly(right, [np.asarray(second["polygon"], dtype=np.int32)], 1)
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / union if union else 0.0


def list_context(candidates: list[dict[str, Any]]) -> np.ndarray:
    quality = np.asarray([float(item["q_raw"]) for item in candidates], dtype=np.float64)
    clipped = np.clip(quality, 1e-8, None)
    probabilities = clipped / clipped.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities))) / math.log(
        max(2, len(candidates))
    )
    top = candidates[0]
    rows = []
    for index, candidate in enumerate(candidates):
        rows.append(
            [
                float(top["q_raw"]) - float(candidate["q_raw"]),
                entropy,
                index / max(1, len(candidates) - 1),
                math.log(
                    max(float(candidate["width_px"]), 1e-6)
                    / max(float(top["width_px"]), 1e-6)
                ),
                axial_difference_fraction(
                    candidate["angle_deg"], top["angle_deg"]
                ),
                _polygon_iou(candidate, top),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def sample_matrix(
    feature_record: dict[str, Any],
    *,
    fields: tuple[str, ...] = SCALAR_FEATURES,
) -> np.ndarray:
    candidates = feature_record["candidates"]
    scalar = np.stack(
        [scalar_candidate_vector(candidate, fields=fields) for candidate in candidates]
    )
    return np.concatenate((scalar, list_context(candidates)), axis=1)


def gate_pair_matrix(feature_record: dict[str, Any]) -> np.ndarray:
    matrix = sample_matrix(feature_record)
    top = matrix[0]
    pairs = []
    for challenger in matrix[1:]:
        pairs.append(np.concatenate((top, challenger, challenger - top)))
    return np.stack(pairs).astype(np.float32)


def gate_outcome_labels(label_record: dict[str, Any]) -> np.ndarray:
    labels = label_lookup(label_record)
    ordered = [
        bool(labels[item["candidate_id"]])
        for item in label_record["candidate_labels"]
    ]
    top = ordered[0]
    result = []
    for challenger in ordered[1:]:
        if not top and challenger:
            result.append(0)  # Recoverable, R.
        elif top and not challenger:
            result.append(1)  # Harmful, H.
        else:
            result.append(2)  # Neutral, N.
    return np.asarray(result, dtype=np.int64)


@dataclass
class JoinedSample:
    feature: dict[str, Any]
    label: dict[str, Any]

    @property
    def sample_id(self) -> str:
        return str(self.label["sample_id"])

    @property
    def frame_id(self) -> str:
        return str(self.label["frame_id"])

    @property
    def sequence_id(self) -> str:
        return str(self.label["sequence_id"])


@dataclass
class FrozenInferenceSample:
    """A label-free sample used by validation/test inference."""

    feature: dict[str, Any]
    stable_id: str

    @property
    def sample_id(self) -> str:
        return self.stable_id

    @property
    def frame_id(self) -> str:
        return str(self.feature["scene_id"])

    @property
    def sequence_id(self) -> str:
        return str(self.feature["scene_id"]).split(",", 1)[0]


def load_inference_features(
    features_path: str | Path,
    *,
    allowed_sample_ids: set[str] | None = None,
) -> list[FrozenInferenceSample]:
    result = []
    for feature in read_jsonl(features_path):
        assert_inference_record_has_no_evaluation_fields(feature)
        sample_id = stable_sample_id(feature["split"], feature["sample_id"])
        if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
            continue
        result.append(
            FrozenInferenceSample(feature=feature, stable_id=sample_id)
        )
    if len({sample.sample_id for sample in result}) != len(result):
        raise ValueError("duplicate stable sample IDs in inference features")
    return result


def load_joined(
    features_path: str | Path,
    labels_path: str | Path,
    *,
    allowed_sample_ids: set[str] | None = None,
) -> list[JoinedSample]:
    labels_by_source = {
        (str(record["split"]), int(record["source_sample_id"])): record
        for record in read_jsonl(labels_path)
    }
    result = []
    for feature in read_jsonl(features_path):
        assert_inference_record_has_no_evaluation_fields(feature)
        key = (str(feature["split"]), int(feature["sample_id"]))
        label = labels_by_source.get(key)
        if label is None:
            raise ValueError(f"missing labels for {key}")
        validate_inference_label_join(feature, label)
        if allowed_sample_ids is not None and label["sample_id"] not in allowed_sample_ids:
            continue
        result.append(JoinedSample(feature=feature, label=label))
    return result


def fit_standardizer(matrices: list[np.ndarray]) -> dict[str, np.ndarray]:
    if not matrices:
        raise ValueError("cannot fit standardizer without data")
    values = np.concatenate(matrices, axis=0).astype(np.float64)
    median = np.nanmedian(values, axis=0)
    invalid = ~np.isfinite(values)
    if invalid.any():
        values[invalid] = np.take(median, np.where(invalid)[1])
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {
        "median": median.astype(np.float32),
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
    }


def apply_standardizer(
    values: np.ndarray, standardizer: dict[str, np.ndarray]
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    invalid = ~np.isfinite(result)
    if invalid.any():
        result[invalid] = np.take(
            standardizer["median"], np.where(invalid)[1]
        )
    return (result - standardizer["mean"]) / standardizer["scale"]
