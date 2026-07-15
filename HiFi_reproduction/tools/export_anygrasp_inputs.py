#!/usr/bin/env python3
"""Export provenance-verified predicted-mask RGB-D inputs for AnyGrasp.

This tool does not run AnyGrasp. It fails closed unless the completed HiFi
prediction export is a full, provenance-consistent directory/manifest bijection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTED_PROBABILITY = "predicted_probability_original_resolution.npy"
PREDICTED_MASK = "predicted_mask_original_resolution.png"
PREDICTION_METADATA = "sample_metadata.json"
OUTPUT_ARTIFACTS = (
    "color.png",
    "depth.png",
    "target_mask.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
    "metadata.json",
)
PCD_TYPE_MAP = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("U", 1): "u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("I", 1): "i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sample_id(scene_id: str, question_index: int) -> str:
    """Match the production prediction export's filesystem identity contract."""
    question_index = int(question_index)
    identity = f"{scene_id}\t{question_index}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"q{question_index:07d}_{digest}"


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_isolated(source: Path, destination: Path) -> str:
    """Create an isolated copy; never share an inode with immutable sources."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return "copy"


def _parse_pcd(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    header: dict[str, list[str]] = {}
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("pcd_header_missing_data")
            decoded = line.decode("ascii", errors="strict").strip()
            if decoded and not decoded.startswith("#"):
                parts = decoded.split()
                header[parts[0].upper()] = parts[1:]
            if decoded.upper().startswith("DATA "):
                payload = handle.read()
                break
    required = ("FIELDS", "SIZE", "TYPE", "COUNT", "WIDTH", "HEIGHT", "POINTS", "DATA")
    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError(f"pcd_header_missing:{','.join(missing)}")
    if header["DATA"] != ["binary"]:
        raise ValueError(f"pcd_data_not_binary:{' '.join(header['DATA'])}")
    fields = header["FIELDS"]
    sizes = [int(value) for value in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(value) for value in header["COUNT"]]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("pcd_header_field_length_mismatch")
    if any(count != 1 for count in counts):
        raise ValueError("pcd_multicount_fields_unsupported")
    dtype_fields = []
    for name, kind, size in zip(fields, types, sizes):
        dtype_code = PCD_TYPE_MAP.get((kind, size))
        if dtype_code is None:
            raise ValueError(f"pcd_field_type_unsupported:{name}:{kind}{size}")
        dtype_fields.append((name, dtype_code))
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("pcd_xyz_fields_missing")
    width = int(header["WIDTH"][0])
    height = int(header["HEIGHT"][0])
    points = int(header["POINTS"][0])
    if width * height != points:
        raise ValueError("pcd_not_organized")
    dtype = np.dtype(dtype_fields)
    expected_bytes = points * dtype.itemsize
    if len(payload) != expected_bytes:
        raise ValueError(f"pcd_payload_size_mismatch:{len(payload)}:{expected_bytes}")
    records = np.frombuffer(payload, dtype=dtype, count=points).reshape(height, width)
    return records, {
        "width": width,
        "height": height,
        "points": points,
        "fields": fields,
        "data": "binary",
    }


def _derive_scene_geometry(
    pcd_path: Path, depth_mm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    records, pcd_header = _parse_pcd(pcd_path)
    if depth_mm.shape != (pcd_header["height"], pcd_header["width"]):
        raise ValueError(
            "pcd_depth_shape_mismatch:"
            f"{pcd_header['width']}x{pcd_header['height']}:"
            f"{depth_mm.shape[1]}x{depth_mm.shape[0]}"
        )
    x = records["x"].astype(np.float64)
    y = records["y"].astype(np.float64)
    z = records["z"].astype(np.float64)
    pixel_y, pixel_x = np.indices(depth_mm.shape, dtype=np.float64)
    valid_xyz = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0)
    valid_depth = depth_mm > 0
    valid = valid_xyz & valid_depth
    valid_count = int(valid.sum())
    if valid_count < 4:
        raise ValueError(f"pcd_fit_insufficient_points:{valid_count}")
    normalized_x = x[valid] / z[valid]
    normalized_y = y[valid] / z[valid]
    design_x = np.column_stack((normalized_x, np.ones(valid_count)))
    design_y = np.column_stack((normalized_y, np.ones(valid_count)))
    fx, cx = np.linalg.lstsq(design_x, pixel_x[valid], rcond=None)[0]
    fy, cy = np.linalg.lstsq(design_y, pixel_y[valid], rcond=None)[0]
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("pcd_fit_invalid_intrinsics")
    residual_x = fx * normalized_x + cx - pixel_x[valid]
    residual_y = fy * normalized_y + cy - pixel_y[valid]
    residual_2d = np.sqrt(residual_x**2 + residual_y**2)
    depth_delta_mm = np.abs(z[valid] * 1000.0 - depth_mm[valid].astype(np.float64))
    depth_nonzero_count = int(valid_depth.sum())
    pcd_valid_count = int(valid_xyz.sum())
    depth_p95_mm = float(np.percentile(depth_delta_mm, 95))
    depth_scale_verified = (
        depth_nonzero_count == pcd_valid_count == valid_count and depth_p95_mm <= 1.1
    )
    intrinsics = {
        "source": "derived_from_organized_pcd",
        "factory_calibration": False,
        "distortion_coefficients_available": False,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "matrix": [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        "width": pcd_header["width"],
        "height": pcd_header["height"],
        "depth_scale": 1000.0,
        "depth_unit": "millimetres",
        "pcd_coordinate_unit": "metres",
        "fit_valid_points": valid_count,
        "fit_rmse_px": float(np.sqrt(np.mean(residual_2d**2))),
        "fit_p95_px": float(np.percentile(residual_2d, 95)),
        "fit_max_px": float(np.max(residual_2d)),
        "depth_nonzero_points": depth_nonzero_count,
        "pcd_valid_points": pcd_valid_count,
        "depth_pcd_abs_rmse_mm": float(np.sqrt(np.mean(depth_delta_mm**2))),
        "depth_pcd_abs_p95_mm": depth_p95_mm,
        "depth_scale_verified": bool(depth_scale_verified),
        "pcd_header": pcd_header,
    }
    return intrinsics, valid


def derive_intrinsics_from_pcd(pcd_path: Path, depth_mm: np.ndarray) -> dict[str, Any]:
    """Public read-only helper retained for direct geometry audits."""
    return _derive_scene_geometry(pcd_path, depth_mm)[0]


def _resolved_recorded_path(value: Any, run_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (run_dir / path).resolve()


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} file missing:{path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise ValueError(f"{label} hash mismatch:{observed}:{expected}")


def _index_records(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        if field not in row:
            continue
        key = str(row[field])
        if key in result:
            raise ValueError(f"duplicate {field} in {label}:{key}")
        result[key] = row
    return result


def _validate_prediction_export(
    *,
    prediction_manifest_path: Path,
    predictions_dir: Path,
    run_dir: Path,
    run_manifest_path: Path,
    original_manifest_path: Path,
    source_root: Path,
    run_records: list[dict[str, Any]],
    original_records: list[dict[str, Any]],
    expected_count: int,
) -> dict[str, Any]:
    if not prediction_manifest_path.is_file():
        raise FileNotFoundError(
            f"Prediction export manifest not found: {prediction_manifest_path}"
        )
    payload = _read_json(prediction_manifest_path)
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        raise ValueError("prediction export manifest samples are not a list")
    declared = int(payload.get("number_of_samples", -1))
    if declared != expected_count or len(samples) != expected_count:
        raise ValueError(
            "prediction export manifest count mismatch:"
            f"declared={declared}:rows={len(samples)}:expected={expected_count}"
        )
    stable_ids = [str(row.get("stable_sample_id")) for row in samples]
    if len(set(stable_ids)) != expected_count:
        raise ValueError("prediction export manifest has duplicate stable sample IDs")
    if len(run_records) != expected_count or len(original_records) != expected_count:
        raise ValueError(
            "frozen/original manifest count mismatch:"
            f"{len(run_records)}/{len(original_records)}:expected={expected_count}"
        )
    sample_indices = [int(row.get("sample_index", -1)) for row in samples]
    if set(sample_indices) != set(range(expected_count)):
        raise ValueError("prediction export manifest sample_index set is incomplete")
    for row in samples:
        expected_id = stable_sample_id(row.get("scene_id"), row.get("question_index"))
        if str(row.get("stable_sample_id")) != expected_id:
            raise ValueError(
                "stable_sample_id mismatch:"
                f"{row.get('stable_sample_id')}:{expected_id}"
            )

    actual_directories = {path.name for path in predictions_dir.iterdir() if path.is_dir()}
    if actual_directories != set(stable_ids):
        missing = sorted(set(stable_ids) - actual_directories)
        extra = sorted(actual_directories - set(stable_ids))
        raise ValueError(
            f"prediction directory bijection mismatch:missing={missing[:5]}:extra={extra[:5]}"
        )

    run_by_question = _index_records(run_records, "question_index", "frozen manifest")
    original_by_question = _index_records(
        original_records, "question_index", "original manifest"
    )
    if len(run_by_question) != expected_count or len(original_by_question) != expected_count:
        raise ValueError("frozen/original question_index values are not unique")

    checkpoint_path = _resolved_recorded_path(payload.get("checkpoint"), run_dir)
    checkpoint_sha = str(payload.get("checkpoint_sha256"))
    _require_file_hash(checkpoint_path, checkpoint_sha, "checkpoint")
    recorded_frozen = _resolved_recorded_path(payload.get("frozen_test_manifest"), run_dir)
    if recorded_frozen != run_manifest_path:
        raise ValueError(
            f"prediction export frozen manifest path mismatch:{recorded_frozen}:{run_manifest_path}"
        )
    frozen_sha = str(payload.get("frozen_test_manifest_sha256"))
    _require_file_hash(run_manifest_path, frozen_sha, "frozen manifest")
    metrics_path = _resolved_recorded_path(payload.get("metrics_csv"), run_dir)
    metrics_sha = str(payload.get("metrics_csv_sha256"))
    _require_file_hash(metrics_path, metrics_sha, "evaluation metrics CSV")
    recorded_source_root = _resolved_recorded_path(payload.get("source_root"), run_dir)
    if recorded_source_root != source_root:
        raise ValueError(
            f"prediction export source root mismatch:{recorded_source_root}:{source_root}"
        )
    recorded_expression = _resolved_recorded_path(
        payload.get("source_expression_file"), run_dir
    )
    if recorded_expression != original_manifest_path:
        raise ValueError(
            "prediction export source expression path mismatch:"
            f"{recorded_expression}:{original_manifest_path}"
        )
    expression_sha = str(payload.get("source_expression_sha256"))
    _require_file_hash(original_manifest_path, expression_sha, "source expression")

    metadata_by_id = {}
    rows_by_id = {}
    for row in samples:
        sample_id = str(row["stable_sample_id"])
        question_key = str(row["question_index"])
        frozen = run_by_question.get(question_key)
        original = original_by_question.get(question_key)
        if frozen is None or original is None:
            raise ValueError(f"prediction row question missing from source manifests:{sample_id}")
        if str(row.get("scene_id")) != str(frozen.get("scene_id")):
            raise ValueError(f"prediction manifest row scene mismatch:{sample_id}")
        if str(row.get("query")) != str(frozen.get("text")):
            raise ValueError(f"prediction manifest row query mismatch:{sample_id}")
        if str(original.get("image_filename")) != str(frozen.get("scene_id")):
            raise ValueError(f"original/frozen scene mismatch:{sample_id}")
        if str(original.get("question")) != str(frozen.get("text")):
            raise ValueError(f"original/frozen query mismatch:{sample_id}")
        recorded_directory = _resolved_recorded_path(row.get("directory"), run_dir)
        sample_dir = predictions_dir / sample_id
        if recorded_directory != sample_dir:
            raise ValueError(f"prediction manifest directory mismatch:{sample_id}")
        for name in (PREDICTED_PROBABILITY, PREDICTED_MASK, PREDICTION_METADATA):
            if not (sample_dir / name).is_file():
                raise ValueError(f"prediction sample artifact missing:{sample_id}:{name}")
        metadata = _read_json(sample_dir / PREDICTION_METADATA)
        expected_fields = {
            "stable_sample_id": sample_id,
            "sample_index": int(row["sample_index"]),
            "evaluation_sample_id": str(row.get("evaluation_sample_id")),
            "question_index": int(row["question_index"]),
            "scene_id": str(row["scene_id"]),
            "query": str(row["query"]),
            "checkpoint_sha256": checkpoint_sha,
            "frozen_test_manifest_sha256": frozen_sha,
            "source_expression_sha256": expression_sha,
        }
        for field, expected in expected_fields.items():
            observed = metadata.get(field)
            if str(observed) != str(expected):
                label = "checkpoint hash" if field == "checkpoint_sha256" else field
                raise ValueError(
                    f"prediction sample metadata {label} mismatch:"
                    f"{sample_id}:{observed}:{expected}"
                )
        metadata_frozen = _resolved_recorded_path(
            metadata.get("frozen_test_manifest"), run_dir
        )
        if metadata_frozen != run_manifest_path:
            raise ValueError(f"prediction sample metadata frozen path mismatch:{sample_id}")
        metadata_expression = _resolved_recorded_path(
            metadata.get("source_expression_file"), run_dir
        )
        if metadata_expression != original_manifest_path:
            raise ValueError(f"prediction sample metadata source path mismatch:{sample_id}")
        metadata_by_id[sample_id] = metadata
        rows_by_id[sample_id] = row
    return {
        "payload": payload,
        "samples": samples,
        "rows_by_id": rows_by_id,
        "metadata_by_id": metadata_by_id,
        "manifest_path": prediction_manifest_path,
        "manifest_sha256": sha256_file(prediction_manifest_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "frozen_manifest_path": run_manifest_path,
        "frozen_manifest_sha256": frozen_sha,
        "metrics_csv_path": metrics_path,
        "metrics_csv_sha256": metrics_sha,
        "source_expression_path": original_manifest_path,
        "source_expression_sha256": expression_sha,
        "repo_commit": payload.get("repo_commit"),
        "checkpoint_repo_commit": payload.get("checkpoint_repo_commit"),
    }


def _normalise_sample_ids(sample_ids: list[str] | None) -> list[str] | None:
    if sample_ids is None:
        return None
    result = []
    for value in sample_ids:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    if len(result) != len(set(result)):
        raise ValueError("requested sample IDs contain duplicates")
    return result


def _select_prediction_rows(
    prediction_export: dict[str, Any],
    sample_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    ordered = sorted(prediction_export["samples"], key=lambda row: int(row["sample_index"]))
    requested = _normalise_sample_ids(sample_ids)
    if requested is not None:
        missing = [sample_id for sample_id in requested if sample_id not in prediction_export["rows_by_id"]]
        if missing:
            raise ValueError(f"requested sample IDs absent from prediction manifest:{missing}")
        selected = [prediction_export["rows_by_id"][sample_id] for sample_id in requested]
    else:
        selected = ordered
    if limit is not None:
        if limit < 0:
            raise ValueError("limit_must_be_nonnegative")
        selected = selected[:limit]
    return selected


def _source_paths(source_root: Path, original_record: dict[str, Any]) -> tuple[dict[str, Path], str]:
    sequence, image_name = str(original_record["image_filename"]).split(",", 1)
    return {
        "rgb": source_root / sequence / "rgb" / image_name,
        "depth": source_root / sequence / "depth" / image_name,
        "pcd": source_root / sequence / "pcd" / Path(image_name).with_suffix(".pcd").name,
    }, sequence


def _blocked_row(manifest_row: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    return {
        "sample_id": str(manifest_row.get("stable_sample_id")),
        "sample_index": int(manifest_row.get("sample_index", -1)),
        "question_index": manifest_row.get("question_index"),
        "scene_id": manifest_row.get("scene_id"),
        "query": manifest_row.get("query"),
        "ready": False,
        "ready_for_anygrasp": False,
        "blockers": blockers,
        "target_valid_point_count": 0,
        "fit_rmse_px": None,
        "fit_p95_px": None,
        "depth_pcd_abs_p95_mm": None,
    }


def _validate_sample(
    *,
    manifest_row: dict[str, Any],
    run_record: dict[str, Any],
    original_record: dict[str, Any],
    prediction_metadata: dict[str, Any],
    predictions_dir: Path,
    source_root: Path,
    expected_size: tuple[int, int],
    geometry_cache: dict[
        str, tuple[dict[str, Any] | None, np.ndarray | None, str | None]
    ],
    geometry_stats: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    blockers = []
    sample_id = str(manifest_row["stable_sample_id"])
    scene_id = str(manifest_row["scene_id"])
    query = str(manifest_row["query"])
    question_index = int(manifest_row["question_index"])
    if str(run_record.get("scene_id")) != scene_id or str(original_record.get("image_filename")) != scene_id:
        blockers.append("scene_id_mismatch_between_manifests")
    if str(run_record.get("text")) != query or str(original_record.get("question")) != query:
        blockers.append("query_mismatch_between_manifests")
    source_paths, sequence = _source_paths(source_root, original_record)
    sample_prediction_dir = predictions_dir / sample_id
    prediction_paths = {
        "probability": sample_prediction_dir / PREDICTED_PROBABILITY,
        "mask": sample_prediction_dir / PREDICTED_MASK,
        "metadata": sample_prediction_dir / PREDICTION_METADATA,
    }
    for name, path in {**source_paths, **prediction_paths}.items():
        if not path.is_file():
            blockers.append(f"missing_{name}:{path}")

    rgb = depth = mask = probability = None
    try:
        rgb = np.asarray(Image.open(source_paths["rgb"]))
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            blockers.append(f"rgb_not_uint8_rgb:{rgb.dtype}:{rgb.shape}")
    except (OSError, ValueError) as error:
        blockers.append(f"rgb_unreadable:{type(error).__name__}")
    try:
        depth = np.asarray(Image.open(source_paths["depth"]))
        if depth.dtype != np.uint16 or depth.ndim != 2:
            blockers.append(f"depth_not_uint16_2d:{depth.dtype}:{depth.shape}")
    except (OSError, ValueError) as error:
        blockers.append(f"depth_unreadable:{type(error).__name__}")
    try:
        mask = np.asarray(Image.open(prediction_paths["mask"]))
        if mask.dtype != np.uint8 or mask.ndim != 2:
            blockers.append(f"predicted_mask_not_uint8_2d:{mask.dtype}:{mask.shape}")
        else:
            values = set(np.unique(mask).tolist())
            if not values <= {0, 255}:
                blockers.append("predicted_mask_values_invalid")
            if not np.any(mask == 255):
                blockers.append("predicted_mask_empty")
    except (OSError, ValueError) as error:
        blockers.append(f"predicted_mask_unreadable:{type(error).__name__}")
    try:
        probability = np.load(
            prediction_paths["probability"], allow_pickle=False, mmap_mode="r"
        )
        if probability.dtype != np.float32:
            blockers.append(f"probability_dtype_not_float32:{probability.dtype}")
        if probability.ndim != 2:
            blockers.append(f"probability_not_2d:{probability.shape}")
        elif not np.isfinite(probability).all():
            blockers.append("probability_nonfinite")
        elif float(np.min(probability)) < 0.0 or float(np.max(probability)) > 1.0:
            blockers.append("probability_out_of_range")
    except (OSError, ValueError) as error:
        blockers.append(f"probability_unreadable:{type(error).__name__}")

    expected_width, expected_height = expected_size
    actual_size = None
    if rgb is not None and rgb.ndim >= 2:
        actual_size = (int(rgb.shape[1]), int(rgb.shape[0]))
        if actual_size != expected_size:
            blockers.append(
                f"original_size_mismatch:{actual_size[0]}x{actual_size[1]}:"
                f"expected:{expected_width}x{expected_height}"
            )
    shapes = {
        tuple(rgb.shape[:2]) if rgb is not None and rgb.ndim >= 2 else None,
        tuple(depth.shape) if depth is not None and depth.ndim == 2 else None,
        tuple(mask.shape) if mask is not None and mask.ndim == 2 else None,
        tuple(probability.shape) if probability is not None and probability.ndim == 2 else None,
    }
    shapes.discard(None)
    if len(shapes) > 1:
        blockers.append(f"asset_dimension_mismatch:{sorted(shapes)}")

    intrinsics = valid_points = None
    if depth is not None and depth.dtype == np.uint16 and source_paths["pcd"].is_file():
        cache_key = str(source_paths["pcd"].resolve())
        if cache_key in geometry_cache:
            geometry_stats["hits"] += 1
            intrinsics, valid_points, fit_error = geometry_cache[cache_key]
        else:
            geometry_stats["fits"] += 1
            try:
                intrinsics, valid_points = _derive_scene_geometry(source_paths["pcd"], depth)
                fit_error = None
            except (OSError, ValueError, np.linalg.LinAlgError) as error:
                intrinsics = valid_points = None
                fit_error = str(error)
            geometry_cache[cache_key] = (intrinsics, valid_points, fit_error)
        if fit_error:
            blockers.append(f"pcd_intrinsics_fit_failed:{fit_error}")
        elif intrinsics is not None:
            if (intrinsics["width"], intrinsics["height"]) != expected_size:
                blockers.append("pcd_original_size_mismatch")
            if intrinsics["fit_p95_px"] > 2.0:
                blockers.append(f"pcd_fit_p95_exceeds_2px:{intrinsics['fit_p95_px']:.6f}")
            if not intrinsics["depth_scale_verified"]:
                blockers.append(
                    "depth_pcd_scale_unverified:"
                    f"counts={intrinsics['depth_nonzero_points']}/"
                    f"{intrinsics['pcd_valid_points']};"
                    f"p95_mm={intrinsics['depth_pcd_abs_p95_mm']:.6f}"
                )

    target_valid_count = 0
    if mask is not None and mask.ndim == 2 and valid_points is not None and mask.shape == valid_points.shape:
        target_valid_count = int(((mask == 255) & valid_points).sum())
        if target_valid_count == 0:
            blockers.append("target_has_no_valid_depth_pcd_point")

    ready = not blockers
    row = {
        "sample_id": sample_id,
        "sample_index": int(manifest_row["sample_index"]),
        "question_index": question_index,
        "scene_id": scene_id,
        "query": query,
        "ready": ready,
        "ready_for_anygrasp": ready,
        "blockers": blockers,
        "target_valid_point_count": target_valid_count,
        "fit_rmse_px": intrinsics.get("fit_rmse_px") if intrinsics else None,
        "fit_p95_px": intrinsics.get("fit_p95_px") if intrinsics else None,
        "depth_pcd_abs_p95_mm": intrinsics.get("depth_pcd_abs_p95_mm") if intrinsics else None,
    }
    context = {
        "source_paths": source_paths,
        "prediction_paths": prediction_paths,
        "prediction_metadata": prediction_metadata,
        "camera_view": sequence.split("/")[2] if len(sequence.split("/")) >= 3 else None,
        "intrinsics": intrinsics,
    }
    return row, context


def _export_ready_sample(
    *,
    build_root: Path,
    final_output_root: Path,
    row: dict[str, Any],
    context: dict[str, Any],
    provenance: dict[str, Any],
    original_manifest_path: Path,
    exporter_source_path: Path,
    exporter_source_sha256: str,
) -> None:
    sample_dir = build_root / row["sample_id"]
    sample_dir.mkdir(parents=True)
    source_paths = context["source_paths"]
    prediction_paths = context["prediction_paths"]
    materialization = {
        "color.png": _copy_isolated(source_paths["rgb"], sample_dir / "color.png"),
        "depth.png": _copy_isolated(source_paths["depth"], sample_dir / "depth.png"),
        "target_mask.png": _copy_isolated(prediction_paths["mask"], sample_dir / "target_mask.png"),
        "target_probability.npy": _copy_isolated(
            prediction_paths["probability"], sample_dir / "target_probability.npy"
        ),
    }
    _write_text_atomic(sample_dir / "language.txt", row["query"])
    intrinsics = dict(context["intrinsics"])
    intrinsics.update(
        {
            "camera_source": (
                "OCID organized PCD captured by one of two ASUS-PRO Xtion cameras; "
                "exact serial and factory calibration unavailable"
            ),
            "camera_view_from_sequence_path": context["camera_view"],
        }
    )
    _write_json_atomic(sample_dir / "intrinsics.json", intrinsics)
    metadata = {
        "sample_id": row["sample_id"],
        "sample_index": row["sample_index"],
        "question_index": row["question_index"],
        "scene_id": row["scene_id"],
        "query": row["query"],
        "ready": True,
        "ready_for_anygrasp": True,
        "blockers": [],
        "target_valid_point_count": row["target_valid_point_count"],
        "mask_source": "predicted_mask_original_resolution",
        "oracle_artifacts_exported": False,
        "anygrasp_inference_ran": False,
        "intrinsics_source": "derived_from_organized_pcd",
        "factory_calibration_claimed": False,
        "depth_scale": 1000.0,
        "depth_unit": "millimetres",
        "camera_view_from_sequence_path": context["camera_view"],
        "materialization": materialization,
        "checkpoint_path": str(provenance["checkpoint_path"]),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "frozen_test_manifest": str(provenance["frozen_manifest_path"]),
        "frozen_test_manifest_sha256": provenance["frozen_manifest_sha256"],
        "evaluation_metrics_csv": str(provenance["metrics_csv_path"]),
        "evaluation_metrics_csv_sha256": provenance["metrics_csv_sha256"],
        "prediction_export_manifest": str(provenance["manifest_path"]),
        "prediction_export_manifest_sha256": provenance["manifest_sha256"],
        "prediction_export_repo_commit": provenance["repo_commit"],
        "checkpoint_repo_commit": provenance["checkpoint_repo_commit"],
        "prediction_sample_metadata": str(context["prediction_paths"]["metadata"]),
        "prediction_sample_metadata_sha256": sha256_file(context["prediction_paths"]["metadata"]),
        "source_expression_file": str(original_manifest_path),
        "source_expression_sha256": provenance["source_expression_sha256"],
        "anygrasp_exporter_source": str(exporter_source_path),
        "anygrasp_exporter_source_sha256": exporter_source_sha256,
        "source_rgb": str(source_paths["rgb"]),
        "source_rgb_sha256": sha256_file(source_paths["rgb"]),
        "source_depth": str(source_paths["depth"]),
        "source_depth_sha256": sha256_file(source_paths["depth"]),
        "source_pcd": str(source_paths["pcd"]),
        "source_pcd_sha256": sha256_file(source_paths["pcd"]),
        "prediction_mask": str(prediction_paths["mask"]),
        "prediction_mask_sha256": sha256_file(prediction_paths["mask"]),
        "prediction_probability": str(prediction_paths["probability"]),
        "prediction_probability_sha256": sha256_file(prediction_paths["probability"]),
        "output_bundle": str(final_output_root / row["sample_id"]),
        "prediction_export_manifest_provenance_validated": True,
    }
    _write_json_atomic(sample_dir / "metadata.json", metadata)
    checksums = [f"{sha256_file(sample_dir / name)}  {name}" for name in OUTPUT_ARTIFACTS]
    _write_text_atomic(sample_dir / "checksums.sha256", "\n".join(checksums) + "\n")


def _write_manifests(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_atomic(
        output_dir / "manifest.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    fields = (
        "sample_id",
        "sample_index",
        "question_index",
        "scene_id",
        "query",
        "ready",
        "ready_for_anygrasp",
        "blockers",
        "target_valid_point_count",
        "fit_rmse_px",
        "fit_p95_px",
        "depth_pcd_abs_p95_mm",
        "output_dir",
    )
    with (output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row.get(field) for field in fields}
            csv_row["blockers"] = ";".join(row["blockers"])
            writer.writerow(csv_row)


def _summary_markdown(
    status: str, rows: list[dict[str, Any]], run_dir: Path, source_root: Path
) -> str:
    ready = sum(bool(row["ready_for_anygrasp"]) for row in rows)
    lines = [
        "# AnyGrasp predicted-mask input export",
        "",
        f"- Status: **{status}**",
        f"- Run: `{run_dir}`",
        f"- Original OCID-VLG source: `{source_root}`",
        f"- Selected samples: {len(rows)}",
        f"- Ready: {ready}",
        f"- Blocked: {len(rows) - ready}",
        "- Materialization: isolated copies only",
        "- Mask source: predicted original-resolution mask only",
        "- Intrinsics: effective pinhole fit derived from organized PCD, not factory calibration",
        "- AnyGrasp inference ran: false",
        "- Oracle artifacts exported: false",
        "",
    ]
    blocked = [row for row in rows if not row["ready_for_anygrasp"]]
    if blocked:
        lines.extend(("## Blockers", ""))
        lines.extend(
            f"- `{row['sample_id']}`: " + "; ".join(row["blockers"])
            for row in blocked
        )
        lines.append("")
    return "\n".join(lines)


def _publish_atomic(build_root: Path, output_dir: Path, force: bool) -> None:
    if output_dir.exists() and not force:
        raise FileExistsError(f"Output already exists; pass --force to replace: {output_dir}")
    if not output_dir.exists():
        build_root.replace(output_dir)
        return
    backup = output_dir.with_name(f".{output_dir.name}.backup-{os.getpid()}")
    if backup.exists():
        raise FileExistsError(f"Atomic backup path already exists: {backup}")
    output_dir.replace(backup)
    try:
        build_root.replace(output_dir)
    except Exception:
        backup.replace(output_dir)
        raise
    shutil.rmtree(backup)


def export_run(
    *,
    run_dir: Path,
    source_root: Path,
    original_manifest_path: Path | None = None,
    run_manifest_path: Path | None = None,
    predictions_dir: Path | None = None,
    prediction_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    report_path: Path | None = None,
    sample_ids: list[str] | None = None,
    limit: int | None = None,
    expected_count: int = 7675,
    expected_size: tuple[int, int] = (640, 480),
    validation_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    source_root = Path(source_root).resolve()
    original_manifest_path = Path(
        original_manifest_path
        or source_root / "refer" / "unique" / "test_expressions.json"
    ).resolve()
    run_manifest_path = Path(run_manifest_path or run_dir / "ocid_vlg_test.json").resolve()
    predictions_dir = Path(predictions_dir or run_dir / "predictions").resolve()
    prediction_manifest_path = Path(
        prediction_manifest_path or predictions_dir / "export_manifest.json"
    ).resolve()
    output_dir = Path(output_dir or run_dir / "anygrasp_input_predicted_mask").resolve()
    report_path = Path(
        report_path or REPO_ROOT / "reports" / "anygrasp_export_summary.md"
    ).resolve()
    expected_count = int(expected_count)
    expected_size = tuple(int(value) for value in expected_size)
    if expected_count <= 0:
        raise ValueError("expected_count_must_be_positive")
    if len(expected_size) != 2 or any(value <= 0 for value in expected_size):
        raise ValueError("expected_size_must_be_width_height")
    if not original_manifest_path.is_file():
        raise FileNotFoundError(f"Original unique manifest not found: {original_manifest_path}")
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Frozen run manifest not found: {run_manifest_path}")
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    if output_dir.exists() and not validation_only and not force:
        raise FileExistsError(f"Output already exists; pass --force to replace: {output_dir}")

    original_payload = _read_json(original_manifest_path)
    original_records = original_payload.get("data")
    run_records = _read_json(run_manifest_path)
    if not isinstance(original_records, list) or not isinstance(run_records, list):
        raise ValueError("original_or_frozen_manifest_not_list")
    prediction_export = _validate_prediction_export(
        prediction_manifest_path=prediction_manifest_path,
        predictions_dir=predictions_dir,
        run_dir=run_dir,
        run_manifest_path=run_manifest_path,
        original_manifest_path=original_manifest_path,
        source_root=source_root,
        run_records=run_records,
        original_records=original_records,
        expected_count=expected_count,
    )
    selected = _select_prediction_rows(prediction_export, sample_ids, limit)
    run_by_question = _index_records(run_records, "question_index", "frozen manifest")
    original_by_question = _index_records(
        original_records, "question_index", "original manifest"
    )
    geometry_cache: dict[
        str, tuple[dict[str, Any] | None, np.ndarray | None, str | None]
    ] = {}
    geometry_stats = {"fits": 0, "hits": 0}
    rows = []
    contexts = {}
    for manifest_row in selected:
        question_key = str(manifest_row["question_index"])
        row, context = _validate_sample(
            manifest_row=manifest_row,
            run_record=run_by_question[question_key],
            original_record=original_by_question[question_key],
            prediction_metadata=prediction_export["metadata_by_id"][manifest_row["stable_sample_id"]],
            predictions_dir=predictions_dir,
            source_root=source_root,
            expected_size=expected_size,
            geometry_cache=geometry_cache,
            geometry_stats=geometry_stats,
        )
        rows.append(row)
        contexts[row["sample_id"]] = context
    ready_count = sum(bool(row["ready_for_anygrasp"]) for row in rows)
    if not rows or ready_count == 0:
        status = "BLOCKED"
    elif ready_count == len(rows):
        status = "DONE"
    else:
        status = "DONE_WITH_CONCERNS"

    if not validation_only:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        build_root = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
        if build_root.exists():
            raise FileExistsError(f"Atomic build path already exists: {build_root}")
        build_root.mkdir()
        exporter_source_path = Path(__file__).resolve()
        exporter_source_sha256 = sha256_file(exporter_source_path)
        try:
            for row in rows:
                row["output_dir"] = (
                    str(output_dir / row["sample_id"])
                    if row["ready_for_anygrasp"]
                    else None
                )
                if row["ready_for_anygrasp"]:
                    _export_ready_sample(
                        build_root=build_root,
                        final_output_root=output_dir,
                        row=row,
                        context=contexts[row["sample_id"]],
                        provenance=prediction_export,
                        original_manifest_path=original_manifest_path,
                        exporter_source_path=exporter_source_path,
                        exporter_source_sha256=exporter_source_sha256,
                    )
            _write_manifests(build_root, rows)
            _publish_atomic(build_root, output_dir, force)
        except Exception:
            if build_root.exists():
                shutil.rmtree(build_root)
            raise
        _write_text_atomic(
            report_path, _summary_markdown(status, rows, run_dir, source_root)
        )

    return {
        "status": status,
        "selected": len(rows),
        "ready": ready_count,
        "blocked": len(rows) - ready_count,
        "validation_only": validation_only,
        "rows": rows,
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "prediction_export_manifest": str(prediction_manifest_path),
        "prediction_export_manifest_sha256": prediction_export["manifest_sha256"],
        "expected_count": expected_count,
        "expected_size": list(expected_size),
        "anygrasp_inference_ran": False,
        "oracle_artifacts_exported": False,
        "geometry_scenes_fitted": geometry_stats["fits"],
        "geometry_cache_hits": geometry_stats["hits"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT.parent / "crog_reproduction" / "OCID-VLG",
    )
    parser.add_argument("--original-manifest", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int, default=7675)
    parser.add_argument(
        "--expected-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 480),
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_run(
        run_dir=args.run_dir,
        source_root=args.source_root,
        original_manifest_path=args.original_manifest,
        run_manifest_path=args.run_manifest,
        predictions_dir=args.predictions_dir,
        prediction_manifest_path=args.prediction_manifest,
        output_dir=args.output_dir,
        report_path=args.report_path,
        sample_ids=args.sample_ids,
        limit=args.limit,
        expected_count=args.expected_count,
        expected_size=tuple(args.expected_size),
        validation_only=args.validation_only,
        force=args.force,
    )
    console_result = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(console_result, indent=2, sort_keys=True))
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
