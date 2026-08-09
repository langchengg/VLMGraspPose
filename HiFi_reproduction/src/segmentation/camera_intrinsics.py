"""Resolve or reproducibly derive inference-time camera intrinsics from organized PCD."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .selective_sam3_vg.io import sha256_file


DERIVATION_SCHEMA_VERSION = 1


def _validate(payload: dict[str, Any], shape: tuple[int, int]) -> dict[str, float]:
    if (int(payload.get("height", -1)), int(payload.get("width", -1))) != shape:
        raise ValueError("camera intrinsics image shape mismatch")
    values = {name: float(payload[name]) for name in ("fx", "fy", "cx", "cy")}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("camera intrinsics contain non-finite values")
    if values["fx"] <= 0.0 or values["fy"] <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    return values


def _explicit_path(row: Any) -> Path | None:
    raw = row.raw
    for name in ("intrinsics_path", "camera_intrinsics_path"):
        value = raw.get(name)
        if value and Path(str(value)).expanduser().resolve().is_file():
            return Path(str(value)).expanduser().resolve()
    metadata = raw.get("frozen_source_metadata_path")
    if metadata:
        candidate = Path(str(metadata)).expanduser().resolve().parent / "intrinsics.json"
        if candidate.is_file():
            return candidate
    candidate = row.native_mask_path.parent / "intrinsics.json"
    return candidate if candidate.is_file() else None


def resolve_camera_intrinsics(
    row: Any,
    *,
    cache_root: Path,
    image_shape: tuple[int, int] = (480, 640),
) -> tuple[dict[str, float], Path, dict[str, Any]]:
    """Return validated values, source/cache path, and full provenance payload."""

    explicit = _explicit_path(row)
    if explicit is not None:
        payload = json.loads(explicit.read_text(encoding="utf-8"))
        return _validate(payload, image_shape), explicit, payload

    pcd_value = row.raw.get("source_pcd_path")
    expected_pcd_sha = str(row.raw.get("source_pcd_sha256", ""))
    if not pcd_value or len(expected_pcd_sha) != 64:
        raise FileNotFoundError(f"no explicit or PCD-derived intrinsics for {row.sample_id}")
    pcd_path = Path(str(pcd_value)).expanduser().resolve()
    if not pcd_path.is_file():
        raise FileNotFoundError(f"organized PCD is missing: {pcd_path}")
    cache_path = (
        cache_root.expanduser().resolve()
        / f"derived_organized_pcd_v{DERIVATION_SCHEMA_VERSION}"
        / expected_pcd_sha[:2]
        / f"{expected_pcd_sha}.json"
    )
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            payload.get("source_pcd_sha256") != expected_pcd_sha
            or int(payload.get("derivation_schema_version", -1))
            != DERIVATION_SCHEMA_VERSION
        ):
            raise RuntimeError(f"camera intrinsics cache identity mismatch: {cache_path}")
        return _validate(payload, image_shape), cache_path, payload

    observed_pcd_sha = sha256_file(pcd_path)
    if observed_pcd_sha != expected_pcd_sha:
        raise RuntimeError(f"organized PCD checksum mismatch: {pcd_path}")
    depth_mm = np.asarray(Image.open(row.depth_path))
    if depth_mm.dtype != np.uint16 or depth_mm.shape != image_shape:
        raise ValueError(f"depth contract mismatch for intrinsics derivation: {row.depth_path}")
    # Reuse the repository's existing audited, GT-free organized-PCD fit.
    from tools.export_anygrasp_inputs import derive_intrinsics_from_pcd

    payload = derive_intrinsics_from_pcd(pcd_path, depth_mm)
    payload.update(
        {
            "derivation_schema_version": DERIVATION_SCHEMA_VERSION,
            "source_pcd_path": str(pcd_path),
            "source_pcd_sha256": observed_pcd_sha,
            "source_depth_path": str(row.depth_path),
            "source_depth_sha256": str(row.raw.get("source_depth_sha256", "")),
            "uses_ground_truth": False,
        }
    )
    if payload.get("depth_scale_verified") is not True:
        raise RuntimeError(f"PCD/depth scale validation failed: {pcd_path}")
    _validate(payload, image_shape)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return _validate(payload, image_shape), cache_path, payload


__all__ = ["DERIVATION_SCHEMA_VERSION", "resolve_camera_intrinsics"]
