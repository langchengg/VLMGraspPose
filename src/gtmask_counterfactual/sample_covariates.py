"""Canonical post-lock sample covariates for stratified analysis."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from .execution import append_gt_access_log
from .io import artifact_record, atomic_json, atomic_parquet, canonical_sha256, sha256_file
from .protocol import load_execution_authority, resolve_postlock_access_authority
from .resource import validate_fresh_gate
from .visual_assets import validate_visual_asset_registry


FRAME_RELATIVE_PATH = Path("03_gt_mask_registry/sample_covariates.parquet")
MANIFEST_RELATIVE_PATH = Path("03_gt_mask_registry/SAMPLE_COVARIATES_AUTHORITY.json")


class SampleCovariateError(RuntimeError):
    """A covariate input, pixel registration, or saved result differs."""


def _object(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise SampleCovariateError(f"{label} is absent or unsafe: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SampleCovariateError(f"{label} must contain one JSON object")
    return value


def _verified(record: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise SampleCovariateError(f"{label} is not an artifact record")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if dict(record) != artifact_record(path):
        raise SampleCovariateError(f"{label} artifact differs")
    return path


def _binary(path: str, digest: str, *, label: str) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file() or sha256_file(source) != digest:
        raise SampleCovariateError(f"{label} pixel artifact differs: {source}")
    with Image.open(source) as image:
        value = np.asarray(image)
    if value.ndim == 3:
        value = value[..., 0]
    if value.ndim != 2:
        raise SampleCovariateError(f"{label} must be a 2-D image")
    return value != 0


def _depth(path: str, digest: str) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file() or sha256_file(source) != digest:
        raise SampleCovariateError(f"depth artifact differs: {source}")
    with Image.open(source) as image:
        value = np.asarray(image)
    if value.ndim == 3:
        value = value[..., 0]
    if value.ndim != 2:
        raise SampleCovariateError("depth image must be 2-D")
    return value


def _boundary_complexity(mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area == 0:
        return 0.0
    horizontal = int(np.count_nonzero(mask[:, 1:] != mask[:, :-1]))
    vertical = int(np.count_nonzero(mask[1:, :] != mask[:-1, :]))
    border = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    return float((horizontal + vertical + border) / np.sqrt(float(area)))


def _row_metrics(predicted: np.ndarray, target: np.ndarray, depth: np.ndarray) -> dict[str, Any]:
    if predicted.shape != target.shape or depth.shape != target.shape:
        raise SampleCovariateError("predicted/GT/depth pixel shapes differ")
    union = int(np.logical_or(predicted, target).sum())
    intersection = int(np.logical_and(predicted, target).sum())
    target_area = int(target.sum())
    valid_target = np.logical_and(target, np.isfinite(depth) & (depth > 0))
    _, component_count = ndimage.label(
        predicted, structure=np.ones((3, 3), dtype=np.uint8)
    )
    return {
        "predicted_mask_iou": 1.0 if union == 0 else intersection / union,
        "target_area_fraction": target_area / target.size,
        "mask_component_count": int(component_count),
        "mask_boundary_complexity": _boundary_complexity(predicted),
        "valid_depth_ratio": (
            0.0 if target_area == 0 else int(valid_target.sum()) / target_area
        ),
    }


def write_sample_covariates(
    run_dir: str | Path,
    *,
    protocol_lock: str | Path,
    visual_asset_manifest: Mapping[str, Any],
    resource_gate: Mapping[str, Any],
    expected_count: int = 7_675,
    resume: bool = False,
) -> Path:
    """Compute the only allowed bulk image covariate frame after P3."""

    validate_fresh_gate(resource_gate)
    root = Path(run_dir).expanduser().resolve()
    authority = load_execution_authority(protocol_lock)
    access = resolve_postlock_access_authority(root, authority=authority)
    _, visual = validate_visual_asset_registry(
        root, visual_asset_manifest, expected_count=expected_count
    )
    sample_path = _verified(authority["sample_manifest"], label="sample manifest")
    registry_path = _verified(authority["gt_mask_registry"], label="GT registry")
    sample_columns = set(pd.read_parquet(sample_path).columns)
    selected_sample_columns = ["sample_id", "query_type", "scene_id", "frame_id"]
    if "scene_family" in sample_columns:
        selected_sample_columns.append("scene_family")
    if "frame_family" in sample_columns:
        selected_sample_columns.append("frame_family")
    sample = pd.read_parquet(sample_path, columns=selected_sample_columns)
    registry = pd.read_parquet(
        registry_path,
        columns=[
            "sample_id",
            "original_gt_mask_path",
            "original_gt_mask_sha256",
            "mapping_status",
        ],
    )
    sample["sample_id"] = sample["sample_id"].astype(str)
    registry["sample_id"] = registry["sample_id"].astype(str)
    visual["sample_id"] = visual["sample_id"].astype(str)
    if (
        len(sample) != expected_count
        or len(registry) != expected_count
        or set(sample["sample_id"]) != set(registry["sample_id"])
        or set(sample["sample_id"]) != set(visual["sample_id"])
        or sample["sample_id"].duplicated().any()
        or registry["sample_id"].duplicated().any()
    ):
        raise SampleCovariateError("covariate input universes differ")
    sample_index = sample.set_index("sample_id", drop=False)
    registry_index = registry.set_index("sample_id", drop=False)
    visual_index = visual.set_index("sample_id", drop=False)
    rows: list[dict[str, Any]] = []
    for sample_id in sample["sample_id"]:
        metadata = sample_index.loc[sample_id]
        mapping = registry_index.loc[sample_id]
        assets = visual_index.loc[sample_id]
        if str(mapping["mapping_status"]) != "PASS":
            metrics = {
                "predicted_mask_iou": 0.0,
                "target_area_fraction": 0.0,
                "mask_component_count": 0,
                "mask_boundary_complexity": 0.0,
                "valid_depth_ratio": 0.0,
            }
        else:
            predicted = _binary(
                str(assets["predicted_mask_path"]),
                str(assets["predicted_mask_sha256"]),
                label="predicted mask",
            )
            target = _binary(
                str(mapping["original_gt_mask_path"]),
                str(mapping["original_gt_mask_sha256"]),
                label="GT mask",
            )
            depth = _depth(
                str(assets["source_depth_path"]),
                str(assets["source_depth_sha256"]),
            )
            metrics = _row_metrics(predicted, target, depth)
        rows.append(
            {
                "sample_id": sample_id,
                "query_type": str(metadata["query_type"]),
                **metrics,
                "scene_family": str(
                    metadata.get("scene_family", metadata["scene_id"])
                ),
                "frame_family": str(
                    metadata.get("frame_family", metadata["frame_id"])
                ),
            }
        )
    frame = pd.DataFrame(rows)
    output = root / FRAME_RELATIVE_PATH
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if output.exists() or manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"sample covariates exist; pass --resume: {manifest_path}"
            )
        validate_sample_covariates(
            root, artifact_record(manifest_path), expected_count=expected_count
        )
        return manifest_path
    output = atomic_parquet(frame, output)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "sample_count": expected_count,
        "protocol_lock": dict(authority["protocol_lock"]),
        "execution_authority_mode": access["mode"],
        "execution_claim": dict(access["record"]),
        "sample_manifest": dict(authority["sample_manifest"]),
        "gt_mask_registry": dict(authority["gt_mask_registry"]),
        "visual_assets": dict(visual_asset_manifest),
        "resource_gate": dict(resource_gate),
        "covariates": artifact_record(output),
        "gt_mask_rows_read": int(registry["mapping_status"].eq("PASS").sum()),
        "gt_grasp_rows_read": 0,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(manifest_path, payload)
    append_gt_access_log(
        root,
        {
            "stage": "P3_SAMPLE_COVARIATES",
            "purpose": "locked_stratification_covariates",
            "gt_mask_rows_read": payload["gt_mask_rows_read"],
            "gt_grasp_rows_read": 0,
            "protocol_lock": dict(authority["protocol_lock"]),
        },
    )
    validate_sample_covariates(
        root, artifact_record(manifest_path), expected_count=expected_count
    )
    return manifest_path


def validate_sample_covariates(
    run_dir: str | Path,
    manifest_record: Mapping[str, Any],
    *,
    expected_count: int = 7_675,
) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(run_dir).expanduser().resolve()
    manifest_path = _verified(manifest_record, label="sample covariate authority")
    if manifest_path != root / MANIFEST_RELATIVE_PATH:
        raise PermissionError("sample covariate authority path is noncanonical")
    value = _object(manifest_path, label="sample covariate authority")
    unsigned = dict(value)
    if unsigned.pop("content_sha256", None) != canonical_sha256(unsigned):
        raise SampleCovariateError("sample covariate authority content hash differs")
    authority = load_execution_authority(value["protocol_lock"]["path"])
    access = resolve_postlock_access_authority(root, authority=authority)
    if (
        value.get("status") != "COMPLETE"
        or int(value.get("sample_count", -1)) != expected_count
        or value.get("protocol_lock") != authority["protocol_lock"]
        or value.get("execution_authority_mode") != access["mode"]
        or value.get("execution_claim") != access["record"]
        or value.get("sample_manifest") != authority["sample_manifest"]
        or value.get("gt_mask_registry") != authority["gt_mask_registry"]
        or value.get("gt_grasp_rows_read") != 0
    ):
        raise SampleCovariateError("sample covariate authority binding differs")
    path = _verified(value["covariates"], label="sample covariates")
    if path != root / FRAME_RELATIVE_PATH:
        raise PermissionError("sample covariate frame path is noncanonical")
    frame = pd.read_parquet(path)
    required = {
        "sample_id",
        "query_type",
        "predicted_mask_iou",
        "target_area_fraction",
        "mask_component_count",
        "mask_boundary_complexity",
        "valid_depth_ratio",
        "scene_family",
        "frame_family",
    }
    if set(frame.columns) != required or len(frame) != expected_count:
        raise SampleCovariateError("sample covariate frame schema/count differs")
    return value, frame


__all__ = [
    "FRAME_RELATIVE_PATH",
    "MANIFEST_RELATIVE_PATH",
    "SampleCovariateError",
    "validate_sample_covariates",
    "write_sample_covariates",
]
