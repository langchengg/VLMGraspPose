#!/usr/bin/env python3
"""Compute descriptive, formal-test-only OCID-VLG subgroup results.

The analysis is deliberately downstream of inference and method selection.
GT-derived attributes are used only for offline diagnosis and are never exposed
to a backend.  Bins are fixed in source rather than optimized on test outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TEST_COUNT = 7_675
PREDICTION_COLUMNS = (
    "method_id",
    "sample_id",
    "j_at_1",
    "j_at_5",
    "candidate_pool_oracle",
    "non_empty",
    "nms_candidate_count",
    "empty_reason",
    "first_valid_rank",
    "latency_seconds",
)
SAMPLE_COLUMNS = (
    "sample_id",
    "question_index",
    "language",
    "predicted_mask_path",
    "predicted_probability_path",
    "source_depth_path",
)
LABEL_COLUMNS = (
    "sample_id",
    "prepared_gt_mask_path",
    "official_annotations_path",
    "official_annotations_sha256",
)
SEMANTIC_OPERATORS = {
    "filter_color": "attribute",
    "relate": "relation",
    "locate": "location",
}
KNOWN_OPERATORS = frozenset(SEMANTIC_OPERATORS) | {
    "scene",
    "filter_category",
    "ground",
    "unique",
    "return",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def _write_frame_exclusive(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, compression="zstd", index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


@lru_cache(maxsize=64)
def _depth_valid(path: str) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim == 3:
        value = value[..., 0]
    numeric = np.asarray(value, dtype=np.float64)
    return np.isfinite(numeric) & (numeric > 0.0)


def _binary_mask(path: str, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if shape is not None and image.size != (shape[1], shape[0]):
            image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _probability(path: str, shape: tuple[int, int]) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source) as archive:
            if len(archive.files) != 1:
                raise ValueError(f"probability archive must contain one array: {source}")
            value = np.asarray(archive[archive.files[0]], dtype=np.float32)
    else:
        value = np.asarray(np.load(source), dtype=np.float32)
    value = np.squeeze(value)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid predicted probability: {source}")
    if value.shape != shape:
        value = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    if float(value.min()) < -1e-5 or float(value.max()) > 1.0 + 1e-5:
        raise ValueError(f"predicted probability outside [0,1]: {source}")
    return np.clip(value, 0.0, 1.0)


def _annotation_rows(labels: pd.DataFrame) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for path_text, expected_sha in labels[
        ["official_annotations_path", "official_annotations_sha256"]
    ].drop_duplicates().itertuples(index=False, name=None):
        path = Path(str(path_text)).expanduser().resolve()
        if not path.is_file() or _sha256(path) != str(expected_sha):
            raise ValueError(f"official annotation identity mismatch: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError(f"official annotation payload is invalid: {path}")
        result[str(path)] = rows
    return result


def _query_type(program: Any) -> str:
    """Mirror the repository's frozen ``ocid_vlg_symbolic_v1`` rule."""

    if not isinstance(program, Sequence) or isinstance(program, (str, bytes)):
        return "unknown"
    operators = [
        str(node.get("type"))
        for node in program
        if isinstance(node, Mapping) and isinstance(node.get("type"), str)
    ]
    if len(operators) != len(program) or not operators or not set(operators) <= KNOWN_OPERATORS:
        return "unknown"
    dimensions = sorted({SEMANTIC_OPERATORS[item] for item in operators if item in SEMANTIC_OPERATORS})
    if not dimensions:
        return "name"
    return dimensions[0] if len(dimensions) == 1 else "mixed"


def _common_features(samples: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    _require_columns(samples, SAMPLE_COLUMNS, "test samples")
    _require_columns(labels, LABEL_COLUMNS, "test labels")
    if len(samples) != EXPECTED_TEST_COUNT or len(labels) != EXPECTED_TEST_COUNT:
        raise ValueError("subgroup analysis requires the complete test split")
    if samples.sample_id.duplicated().any() or labels.sample_id.duplicated().any():
        raise ValueError("test sample identities must be unique")
    merged = samples.loc[:, SAMPLE_COLUMNS].merge(
        labels.loc[:, LABEL_COLUMNS], on="sample_id", how="inner", validate="one_to_one"
    )
    if len(merged) != EXPECTED_TEST_COUNT:
        raise ValueError("test sample/label coverage mismatch")
    annotations = _annotation_rows(labels)
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        predicted = _binary_mask(str(row.predicted_mask_path))
        gt_mask = _binary_mask(str(row.prepared_gt_mask_path), shape=predicted.shape)
        probability = _probability(str(row.predicted_probability_path), predicted.shape)
        valid_depth = _depth_valid(str(row.source_depth_path))
        if valid_depth.shape != predicted.shape:
            valid_depth = cv2.resize(
                valid_depth.astype(np.uint8),
                (predicted.shape[1], predicted.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        predicted_area = int(predicted.sum())
        gt_area = int(gt_mask.sum())
        if gt_area <= 0:
            raise ValueError(f"{row.sample_id}: GT target mask is empty")
        union = int(np.logical_or(predicted, gt_mask).sum())
        intersection = int(np.logical_and(predicted, gt_mask).sum())
        gt_rows, gt_columns = np.nonzero(gt_mask)
        annotation_path = str(Path(str(row.official_annotations_path)).resolve())
        question_index = int(row.question_index)
        source_rows = annotations[annotation_path]
        if not 0 <= question_index < len(source_rows):
            raise ValueError(f"{row.sample_id}: question index outside annotations")
        annotation = source_rows[question_index]
        if int(annotation.get("question_index", -1)) != question_index:
            raise ValueError(f"{row.sample_id}: annotation question index drift")
        if str(annotation.get("question")) != str(row.language):
            raise ValueError(f"{row.sample_id}: annotation language drift")
        program = annotation.get("program")
        query_type = _query_type(program)
        scene_outputs = [
            step.get("_output")
            for step in program
            if isinstance(step, Mapping) and step.get("type") == "scene"
        ]
        clutter = scene_outputs[0] if scene_outputs else None
        if not isinstance(clutter, list):
            raise ValueError(f"{row.sample_id}: scene instance inventory unavailable")
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "analysis_scope": "formal_test_only",
                "configuration_selection_performed": False,
                "query_type": str(query_type),
                "target_area_fraction": gt_area / float(gt_mask.size),
                "predicted_mask_iou": 0.0 if union == 0 else intersection / union,
                "predicted_mask_confidence": 0.0
                if predicted_area == 0
                else float(probability[predicted].mean()),
                "predicted_target_depth_valid_fraction": 0.0
                if predicted_area == 0
                else float(np.logical_and(valid_depth, predicted).sum() / predicted_area),
                "target_object_width_px": int(gt_columns.max() - gt_columns.min() + 1),
                "scene_clutter_instance_count": len(clutter),
            }
        )
    return pd.DataFrame(rows)


def _fixed_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target_area_group"] = pd.cut(
        result.target_area_fraction,
        [-np.inf, 0.005, 0.015, 0.04, np.inf],
        labels=("<0.5%", "0.5-1.5%", "1.5-4%", ">=4%"),
        right=False,
    ).astype(str)
    result["predicted_mask_iou_group"] = pd.cut(
        result.predicted_mask_iou,
        [-np.inf, 0.25, 0.5, 0.75, np.inf],
        labels=("<0.25", "0.25-0.5", "0.5-0.75", ">=0.75"),
        right=False,
    ).astype(str)
    result["predicted_mask_confidence_group"] = pd.cut(
        result.predicted_mask_confidence,
        [-np.inf, 0.5, 0.7, 0.85, np.inf],
        labels=("<0.5", "0.5-0.7", "0.7-0.85", ">=0.85"),
        right=False,
    ).astype(str)
    result["candidate_count_group"] = pd.cut(
        result.nms_candidate_count,
        [-np.inf, 1, 6, 21, 51, np.inf],
        labels=("0", "1-5", "6-20", "21-50", ">50"),
        right=False,
    ).astype(str)
    result["depth_validity_group"] = pd.cut(
        result.predicted_target_depth_valid_fraction,
        [-np.inf, 0.5, 0.9, 0.99, np.inf],
        labels=("<0.5", "0.5-0.9", "0.9-0.99", ">=0.99"),
        right=False,
    ).astype(str)
    result["object_width_group"] = pd.cut(
        result.target_object_width_px,
        [-np.inf, 32, 64, 128, np.inf],
        labels=("<32px", "32-63px", "64-127px", ">=128px"),
        right=False,
    ).astype(str)
    result["scene_clutter_group"] = pd.cut(
        result.scene_clutter_instance_count,
        [-np.inf, 4, 8, np.inf],
        labels=("1-3", "4-7", ">=8"),
        right=False,
    ).astype(str)
    result["no_grasp_reason_group"] = np.where(
        result.non_empty.astype(bool),
        "has_grasp",
        result.empty_reason.fillna("unspecified_no_grasp").astype(str),
    )
    rank = pd.to_numeric(result.first_valid_rank, errors="coerce")
    result["first_valid_rank_group"] = np.select(
        [rank.eq(1), rank.between(2, 5, inclusive="both"), rank.gt(5)],
        ["rank1", "rank2-5", "rank>5"],
        default="no_positive",
    )
    return result


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    dimensions = (
        "query_type",
        "target_area_group",
        "predicted_mask_iou_group",
        "predicted_mask_confidence_group",
        "candidate_count_group",
        "depth_validity_group",
        "object_width_group",
        "scene_clutter_group",
        "no_grasp_reason_group",
        "first_valid_rank_group",
    )
    rows: list[dict[str, Any]] = []
    for method_id in sorted(frame.method_id.astype(str).unique()):
        method = frame.loc[frame.method_id.astype(str) == method_id]
        for dimension in dimensions:
            for group, values in method.groupby(dimension, observed=True, sort=True):
                rows.append(
                    {
                        "method_id": method_id,
                        "dimension": dimension,
                        "group": str(group),
                        "sample_count": len(values),
                        "j_at_1": float(values.j_at_1.astype(bool).mean()),
                        "j_at_5": float(values.j_at_5.astype(bool).mean()),
                        "candidate_pool_oracle": float(
                            values.candidate_pool_oracle.astype(bool).mean()
                        ),
                        "non_empty_rate": float(values.non_empty.astype(bool).mean()),
                        "mean_nms_candidates": float(values.nms_candidate_count.mean()),
                        "p50_latency_seconds": float(values.latency_seconds.median()),
                    }
                )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    predictions = pd.read_parquet(predictions_path)
    _require_columns(predictions, PREDICTION_COLUMNS, "formal predictions")
    method_counts = predictions.groupby("method_id").size()
    if method_counts.empty or not method_counts.eq(EXPECTED_TEST_COUNT).all():
        raise ValueError("every formal method must cover the complete test split")
    samples_path = run / "manifests/test_samples.parquet"
    labels_path = run / "manifests/test_labels.parquet"
    common = _common_features(pd.read_parquet(samples_path), pd.read_parquet(labels_path))
    combined = predictions.loc[:, PREDICTION_COLUMNS].merge(
        common, on="sample_id", how="inner", validate="many_to_one"
    )
    if len(combined) != len(predictions):
        raise ValueError("formal prediction/sample feature coverage mismatch")
    selection = json.loads((run / "primary_validation_selection.json").read_text())
    primary = str(selection["primary_method_id"])
    primary_rows = combined.loc[combined.method_id.astype(str) == primary].copy()
    if len(primary_rows) != EXPECTED_TEST_COUNT:
        raise ValueError("locked primary method is absent from formal predictions")
    primary_rows["method_id"] = "locked_primary_4dof_backend"
    combined = pd.concat([combined, primary_rows], ignore_index=True)
    binned = _fixed_bins(combined)
    aggregated = _aggregate(binned)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "subgroup_sample_features.parquet"
    result_path = output / "subgroup_results.csv"
    manifest_path = output / "subgroup_analysis_manifest.json"
    _write_frame_exclusive(binned, feature_path)
    _write_frame_exclusive(aggregated, result_path)
    manifest = {
        "schema_version": 1,
        "analysis_scope": "formal_test_only",
        "configuration_selection_performed": False,
        "gt_derived_features_offline_analysis_only": True,
        "expected_test_count": EXPECTED_TEST_COUNT,
        "method_count_including_primary_alias": int(binned.method_id.nunique()),
        "locked_primary_source_method": primary,
        "fixed_bin_definitions_in_source": str(Path(__file__).resolve()),
        "inputs": {
            "predictions": {"path": str(predictions_path), "sha256": _sha256(predictions_path)},
            "samples": {"path": str(samples_path), "sha256": _sha256(samples_path)},
            "labels": {"path": str(labels_path), "sha256": _sha256(labels_path)},
        },
        "outputs": {
            "features": {"path": str(feature_path), "sha256": _sha256(feature_path)},
            "results": {"path": str(result_path), "sha256": _sha256(result_path)},
        },
    }
    _write_json_exclusive(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
