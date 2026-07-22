"""Deterministic serialization of planar parallel-jaw grasp candidates."""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .camera_geometry import T_CAMERA_GRASP_FIXED_APPROACH_KEY


CANONICAL_FIELDS = (
    "candidate_id",
    "sample_id",
    "query",
    "representation",
    "approach_constraint",
    "center_u_px",
    "center_v_px",
    "center_depth_m",
    "center_camera_xyz_m",
    "angle_rad",
    "angle_deg",
    "width_m",
    "width_px",
    "endpoint_1_uv",
    "endpoint_2_uv",
    "contact_points_uv",
    "contact_normals",
    "centre_inside_mask",
    "centre_boundary_distance_px",
    "endpoint_1_mask_support",
    "endpoint_2_mask_support",
    "grasp_axis_mask_support",
    "valid_depth_support",
    "sampler_rank",
    "gqcnn_q_value",
    "gqcnn_rank",
    "rejection_reason",
    "camera_frame",
    "seed",
    "sampler_configuration_hash",
    "model_name",
    T_CAMERA_GRASP_FIXED_APPROACH_KEY,
)


NUMERIC_FIELDS = {
    "center_u_px",
    "center_v_px",
    "center_depth_m",
    "angle_rad",
    "angle_deg",
    "width_m",
    "width_px",
    "centre_boundary_distance_px",
    "endpoint_1_mask_support",
    "endpoint_2_mask_support",
    "grasp_axis_mask_support",
    "valid_depth_support",
    "sampler_rank",
    "gqcnn_q_value",
    "gqcnn_rank",
    "seed",
}


def _get(candidate: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(candidate, Mapping) and name in candidate:
            return candidate[name]
        if hasattr(candidate, name):
            return getattr(candidate, name)
    return default


def _float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _vector(value: Any, length: int) -> List[float]:
    if value is None:
        return [float("nan")] * length
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != length:
        return [float("nan")] * length
    return [float(item) for item in array]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def candidate_to_record(candidate: Any, index: int = 0) -> Dict[str, Any]:
    """Convert a candidate mapping/object into a stable, complete record."""

    center = _get(candidate, "center_uv")
    center_values = _vector(center, 2) if center is not None else [float("nan")] * 2
    u = _float(_get(candidate, "center_u_px", "u", default=center_values[0]))
    v = _float(_get(candidate, "center_v_px", "v", default=center_values[1]))

    endpoints = _get(candidate, "endpoints_uv")
    if endpoints is not None and np.asarray(endpoints).size == 4:
        endpoint_values = np.asarray(endpoints, dtype=np.float64).reshape(2, 2)
        endpoint_1 = endpoint_values[0].tolist()
        endpoint_2 = endpoint_values[1].tolist()
    else:
        endpoint_1 = _vector(_get(candidate, "endpoint_1_uv"), 2)
        endpoint_2 = _vector(_get(candidate, "endpoint_2_uv"), 2)

    angle_rad = _float(_get(candidate, "angle_rad", "angle"))
    angle_deg = _float(_get(candidate, "angle_deg"))
    if math.isnan(angle_deg) and not math.isnan(angle_rad):
        angle_deg = float(np.degrees(angle_rad))

    pose = _get(candidate, T_CAMERA_GRASP_FIXED_APPROACH_KEY)
    pose_value = (
        np.asarray(pose, dtype=np.float64).reshape(4, 4).tolist()
        if pose is not None and np.asarray(pose).size == 16
        else np.full((4, 4), np.nan).tolist()
    )

    record: Dict[str, Any] = {
        "candidate_id": str(_get(candidate, "candidate_id", default=f"candidate_{index:04d}")),
        "sample_id": _get(candidate, "sample_id"),
        "query": _get(candidate, "query"),
        "representation": _get(
            candidate, "representation", default="planar_parallel_jaw_4dof"
        ),
        "approach_constraint": _get(
            candidate, "approach_constraint", default="fixed_camera_optical_axis"
        ),
        "center_u_px": u,
        "center_v_px": v,
        "center_depth_m": _float(_get(candidate, "center_depth_m", "depth")),
        "center_camera_xyz_m": _vector(_get(candidate, "center_camera_xyz_m"), 3),
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "width_m": _float(_get(candidate, "width_m", "width")),
        "width_px": _float(_get(candidate, "width_px")),
        "endpoint_1_uv": [float(value) for value in endpoint_1],
        "endpoint_2_uv": [float(value) for value in endpoint_2],
        "contact_points_uv": _json_value(_get(candidate, "contact_points_uv")),
        "contact_normals": _json_value(_get(candidate, "contact_normals")),
        "centre_inside_mask": _get(
            candidate, "centre_inside_mask", "center_inside_mask"
        ),
        "centre_boundary_distance_px": _float(
            _get(candidate, "centre_boundary_distance_px", "center_boundary_distance_px")
        ),
        "endpoint_1_mask_support": _float(_get(candidate, "endpoint_1_mask_support")),
        "endpoint_2_mask_support": _float(_get(candidate, "endpoint_2_mask_support")),
        "grasp_axis_mask_support": _float(
            _get(candidate, "grasp_axis_mask_support", "mask_support")
        ),
        "valid_depth_support": _float(_get(candidate, "valid_depth_support")),
        "sampler_rank": _float(_get(candidate, "sampler_rank")),
        "gqcnn_q_value": _float(_get(candidate, "gqcnn_q_value", "quality")),
        "gqcnn_rank": _float(_get(candidate, "gqcnn_rank")),
        "rejection_reason": _get(candidate, "rejection_reason"),
        "camera_frame": _get(candidate, "camera_frame"),
        "seed": _float(_get(candidate, "seed")),
        "sampler_configuration_hash": _get(candidate, "sampler_configuration_hash"),
        "model_name": _get(candidate, "model_name"),
        T_CAMERA_GRASP_FIXED_APPROACH_KEY: pose_value,
    }

    # Preserve non-canonical research metadata deterministically rather than
    # silently discarding it.
    if isinstance(candidate, Mapping):
        for key in sorted(candidate, key=str):
            if str(key) not in record and str(key) not in {"center_uv", "endpoints_uv"}:
                record[str(key)] = _json_value(candidate[key])
    return record


def candidates_to_records(candidates: Iterable[Any]) -> List[Dict[str, Any]]:
    return [candidate_to_record(candidate, index) for index, candidate in enumerate(candidates)]


def candidates_to_arrays(candidates: Iterable[Any]) -> Dict[str, np.ndarray]:
    """Convert candidates into the numeric arrays required by the NPZ format."""

    records = candidates_to_records(candidates)
    count = len(records)

    def scalar(name: str, dtype: Any = np.float32) -> np.ndarray:
        return np.asarray([record[name] for record in records], dtype=dtype)

    arrays: Dict[str, np.ndarray] = {
        "center_uv": np.asarray(
            [[record["center_u_px"], record["center_v_px"]] for record in records],
            dtype=np.float32,
        ).reshape(count, 2),
        "center_depth_m": scalar("center_depth_m"),
        "center_camera_xyz_m": np.asarray(
            [record["center_camera_xyz_m"] for record in records], dtype=np.float32
        ).reshape(count, 3),
        "angle_rad": scalar("angle_rad"),
        "width_m": scalar("width_m"),
        "width_px": scalar("width_px"),
        "endpoints_uv": np.asarray(
            [
                [record["endpoint_1_uv"], record["endpoint_2_uv"]]
                for record in records
            ],
            dtype=np.float32,
        ).reshape(count, 2, 2),
        "mask_support": scalar("grasp_axis_mask_support"),
        "boundary_distance_px": scalar("centre_boundary_distance_px"),
        "gqcnn_q_value": scalar("gqcnn_q_value"),
        "valid": np.asarray(
            [
                record["rejection_reason"] in (None, "")
                and record["centre_inside_mask"] is not False
                for record in records
            ],
            dtype=np.bool_,
        ),
        T_CAMERA_GRASP_FIXED_APPROACH_KEY: np.asarray(
            [record[T_CAMERA_GRASP_FIXED_APPROACH_KEY] for record in records],
            dtype=np.float64,
        ).reshape(count, 4, 4),
    }
    return arrays


def save_candidates_json(
    path: Path | str,
    candidates: Iterable[Any],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write stable UTF-8 JSON; unavailable numeric fields are JSON ``NaN``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = candidates_to_records(candidates)
    payload: Any = records
    if metadata is not None:
        payload = {"metadata": _json_value(metadata), "candidates": records}
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=True,
        sort_keys=False,
        indent=2,
        separators=(",", ": "),
    ) + "\n"
    destination.write_text(text, encoding="utf-8", newline="\n")
    return destination


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=True, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def save_candidates_csv(path: Path | str, candidates: Iterable[Any]) -> Path:
    """Write stable RFC-4180-style CSV with canonical columns first."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = candidates_to_records(candidates)
    extra_fields = sorted(
        {key for record in records for key in record if key not in CANONICAL_FIELDS}
    )
    fields = list(CANONICAL_FIELDS) + extra_fields
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: _csv_value(record.get(field)) for field in fields})
    return destination


def save_candidates_npz(path: Path | str, candidates: Iterable[Any]) -> Path:
    """Write a byte-for-byte deterministic, NumPy-compatible NPZ archive."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = candidates_to_arrays(candidates)
    # np.savez embeds current ZIP timestamps.  Fixed ZipInfo metadata makes the
    # research artefact reproducible without changing the standard NPZ format.
    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return destination


def save_candidate_bundle(
    candidates: Iterable[Any],
    *,
    json_path: Path | str,
    npz_path: Path | str,
    csv_path: Path | str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Write JSON, NPZ and CSV from one materialized candidate sequence."""

    materialized = list(candidates)
    return {
        "json": save_candidates_json(json_path, materialized, metadata=metadata),
        "npz": save_candidates_npz(npz_path, materialized),
        "csv": save_candidates_csv(csv_path, materialized),
    }

