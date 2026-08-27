"""Canonical label-free visual assets for qualitative counterfactual boards."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .io import artifact_record, atomic_json, atomic_parquet, canonical_sha256
from .protocol import LOCK_RELATIVE_PATH, verify_protocol_lock


EXPECTED_SAMPLE_COUNT = 7_675
REGISTRY_RELATIVE_PATH = Path("04_predicted_replay/VISUAL_ASSET_REGISTRY.parquet")
MANIFEST_RELATIVE_PATH = Path("04_predicted_replay/VISUAL_ASSET_REGISTRY.json")
_SOURCE_COLUMNS = (
    "sample_id",
    "source_rgb_path",
    "source_rgb_sha256",
    "source_depth_path",
    "source_depth_sha256",
    "predicted_mask_path",
    "predicted_mask_sha256",
    "predicted_probability_path",
    "predicted_probability_sha256",
    "intrinsics_path",
    "intrinsics_sha256",
)


class VisualAssetContractError(RuntimeError):
    """A saved board asset registry differs from the locked source evidence."""


def _object(path: Path, *, name: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise VisualAssetContractError(f"{name} is absent or unsafe: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VisualAssetContractError(f"cannot parse {name}: {source}") from error
    if not isinstance(value, dict):
        raise VisualAssetContractError(f"{name} must contain one JSON object")
    return value


def _verify_record(record: Mapping[str, Any], *, name: str) -> Path:
    if not isinstance(record, Mapping):
        raise VisualAssetContractError(f"{name} is not an artifact record")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    observed = artifact_record(path)
    if any(record.get(key) != observed[key] for key in ("path", "sha256", "bytes")):
        raise VisualAssetContractError(f"{name} artifact differs")
    return path


def _locked_inventory(lock: Mapping[str, Any]) -> set[tuple[str, str, int]]:
    # Import locally because postprocess validates visual assets and therefore
    # imports this module.  Reuse its canonical parser instead of recursively
    # rehashing every transitive artifact referenced by a final source lock.
    from .postprocess import PostprocessContractError, _locked_source_inventory

    try:
        result = _locked_source_inventory(lock)
    except PostprocessContractError as error:
        raise VisualAssetContractError(
            "protocol source inventory differs"
        ) from error
    if not result:
        raise VisualAssetContractError("protocol lock exposes no source inventory")
    return result


def _source_frame(path: Path, *, name: str, expected_count: int) -> pd.DataFrame:
    names = set(pq.read_schema(path).names)
    missing = sorted(set(_SOURCE_COLUMNS).difference(names))
    if missing:
        raise VisualAssetContractError(f"{name} misses visual columns: {missing}")
    frame = pd.read_parquet(path, columns=list(_SOURCE_COLUMNS))
    frame["sample_id"] = frame["sample_id"].astype(str)
    if (
        len(frame) != int(expected_count)
        or frame["sample_id"].duplicated().any()
        or frame["sample_id"].str.strip().eq("").any()
    ):
        raise VisualAssetContractError(f"{name} visual denominator differs")
    for prefix in ("source_rgb", "source_depth", "predicted_mask", "predicted_probability"):
        if frame[f"{prefix}_path"].astype(str).str.strip().eq("").any():
            raise VisualAssetContractError(f"{name} has an empty {prefix} path")
        if not frame[f"{prefix}_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise VisualAssetContractError(f"{name} has an invalid {prefix} hash")
    return frame.sort_values("sample_id").reset_index(drop=True)


def _canonical_frame(
    g1_c1: pd.DataFrame, d1: pd.DataFrame, *, expected_count: int
) -> pd.DataFrame:
    if set(g1_c1["sample_id"]) != set(d1["sample_id"]):
        raise VisualAssetContractError("unified and D1 visual sample universes differ")
    compared = g1_c1.merge(
        d1,
        on="sample_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_unified", "_d1"),
    )
    shared = tuple(column for column in _SOURCE_COLUMNS if column != "sample_id")
    for column in shared:
        left = compared[f"{column}_unified"].fillna("").astype(str)
        right = compared[f"{column}_d1"].fillna("").astype(str)
        if not left.eq(right).all():
            raise VisualAssetContractError(
                f"unified and D1 visual evidence differs: {column}"
            )
    result = pd.DataFrame({"sample_id": compared["sample_id"].astype(str)})
    for column in shared:
        result[column] = compared[f"{column}_unified"]
    result["coordinate_frame"] = "rgb_native"
    result["predicted_probability_status"] = "AVAILABLE"
    if len(result) != int(expected_count):
        raise VisualAssetContractError("canonical visual registry count differs")
    return result.sort_values("sample_id").reset_index(drop=True)


def _verify_source_membership(
    record: Mapping[str, Any], *, inventory: set[tuple[str, str, int]], name: str
) -> Path:
    path = _verify_record(record, name=name)
    identity = (str(path), str(record["sha256"]), int(record["bytes"]))
    if identity not in inventory:
        raise PermissionError(f"{name} is absent from the protocol source inventory")
    return path


def write_visual_asset_registry(
    run_dir: str | Path,
    *,
    protocol_lock: str | Path,
    unified_samples: Mapping[str, Any],
    d1_samples: Mapping[str, Any],
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> Path:
    """Reconcile two independently locked label-free sample manifests."""

    root = Path(run_dir).expanduser().resolve()
    lock_path = Path(protocol_lock).expanduser().resolve()
    if lock_path != root / LOCK_RELATIVE_PATH:
        raise PermissionError("visual registry requires the canonical protocol lock")
    lock = verify_protocol_lock(root)
    inventory = _locked_inventory(lock)
    unified_path = _verify_source_membership(
        unified_samples, inventory=inventory, name="unified visual source"
    )
    d1_path = _verify_source_membership(
        d1_samples, inventory=inventory, name="D1 visual source"
    )
    frame = _canonical_frame(
        _source_frame(unified_path, name="unified visual source", expected_count=expected_count),
        _source_frame(d1_path, name="D1 visual source", expected_count=expected_count),
        expected_count=expected_count,
    )
    registry = atomic_parquet(frame, root / REGISTRY_RELATIVE_PATH)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "scientific_role": "label-free qualitative visual source registry",
        "raw_test_ground_truth_rows_read": 0,
        "sample_count": len(frame),
        "coordinate_frame": "rgb_native",
        "shared_hifics_visual_assets_across_routes": True,
        "unified_samples": dict(unified_samples),
        "d1_samples": dict(d1_samples),
        "registry": artifact_record(registry),
        "protocol_lock": artifact_record(lock_path),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    manifest = atomic_json(root / MANIFEST_RELATIVE_PATH, payload)
    validate_visual_asset_registry(root, artifact_record(manifest), expected_count=expected_count)
    return manifest


def validate_visual_asset_registry(
    run_dir: str | Path,
    manifest_record: Mapping[str, Any],
    *,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Replay source reconciliation and compare the saved registry exactly."""

    root = Path(run_dir).expanduser().resolve()
    manifest_path = _verify_record(manifest_record, name="visual registry manifest")
    if manifest_path != root / MANIFEST_RELATIVE_PATH:
        raise PermissionError("visual registry manifest path is noncanonical")
    value = _object(manifest_path, name="visual registry manifest")
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise VisualAssetContractError("visual registry manifest content hash differs")
    if (
        value.get("status") != "COMPLETE"
        or value.get("raw_test_ground_truth_rows_read") != 0
        or int(value.get("sample_count", -1)) != int(expected_count)
    ):
        raise VisualAssetContractError("visual registry manifest contract differs")
    lock = verify_protocol_lock(root)
    if value.get("protocol_lock") != artifact_record(root / LOCK_RELATIVE_PATH):
        raise VisualAssetContractError("visual registry protocol binding differs")
    inventory = _locked_inventory(lock)
    unified_path = _verify_source_membership(
        value["unified_samples"], inventory=inventory, name="unified visual source"
    )
    d1_path = _verify_source_membership(
        value["d1_samples"], inventory=inventory, name="D1 visual source"
    )
    expected = _canonical_frame(
        _source_frame(unified_path, name="unified visual source", expected_count=expected_count),
        _source_frame(d1_path, name="D1 visual source", expected_count=expected_count),
        expected_count=expected_count,
    )
    registry_path = _verify_record(value["registry"], name="visual asset registry")
    if registry_path != root / REGISTRY_RELATIVE_PATH:
        raise PermissionError("visual asset registry path is noncanonical")
    observed = pd.read_parquet(registry_path)
    pd.testing.assert_frame_equal(
        observed.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=True
    )
    return value, observed


__all__ = [
    "EXPECTED_SAMPLE_COUNT",
    "MANIFEST_RELATIVE_PATH",
    "REGISTRY_RELATIVE_PATH",
    "VisualAssetContractError",
    "validate_visual_asset_registry",
    "write_visual_asset_registry",
]
