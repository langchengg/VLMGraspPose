"""Post-lock materialisation of the protocol-bound Test grasp geometry."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .execution import append_gt_access_log
from .io import artifact_record, atomic_json, atomic_parquet, canonical_sha256
from .protocol import load_execution_authority, resolve_postlock_access_authority


REGISTRY_RELATIVE_PATH = Path("02_sample_manifest/gt_grasp_registry.parquet")
MANIFEST_RELATIVE_PATH = Path("02_sample_manifest/GT_GRASP_AUTHORITY.json")


class GTGraspAuthorityError(RuntimeError):
    """The post-lock grasp geometry differs from its frozen source."""


def _object(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise GTGraspAuthorityError(f"{label} is absent or unsafe: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GTGraspAuthorityError(f"{label} is not a JSON object")
    return value


def _verified_record(record: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise GTGraspAuthorityError(f"{label} is not an artifact record")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if dict(record) != artifact_record(path):
        raise GTGraspAuthorityError(f"{label} artifact differs")
    return path


def _grasp_column(path: Path) -> str:
    names = set(pq.read_schema(path).names)
    for name in ("gt_grasp_rectangles", "gt_grasp_list_json"):
        if name in names:
            return name
    raise GTGraspAuthorityError("frozen GT grasp source has no grasp geometry column")


def _normalize_json(value: Any) -> str:
    def plain(item: Any) -> Any:
        if hasattr(item, "tolist"):
            return plain(item.tolist())
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        if hasattr(item, "item"):
            return item.item()
        return item

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise GTGraspAuthorityError("frozen GT grasp JSON is malformed") from error
    value = plain(value)
    if not isinstance(value, (list, tuple)):
        raise GTGraspAuthorityError("frozen GT grasp set is not a sequence")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def materialize_gt_grasp_authority(
    run_dir: str | Path,
    *,
    protocol_lock: str | Path,
    resume: bool = False,
    expected_count: int = 7_675,
) -> Path:
    """Read only the locked grasp column after P3 and publish a canonical frame."""

    root = Path(run_dir).expanduser().resolve()
    lock_path = Path(protocol_lock).expanduser().resolve()
    authority = load_execution_authority(lock_path)
    access = resolve_postlock_access_authority(root, authority=authority)
    source_record = authority.get("gt_grasp_source")
    source_path = _verified_record(source_record, label="locked GT grasp source")
    sample_path = _verified_record(
        authority["sample_manifest"], label="locked sample manifest"
    )
    column = _grasp_column(source_path)

    # This is the deliberate post-lock Test-GT access boundary.  No other
    # source column is loaded into memory.
    source = pd.read_parquet(source_path, columns=["sample_id", column])
    sample = pd.read_parquet(sample_path, columns=["sample_id"])
    source["sample_id"] = source["sample_id"].astype(str)
    sample["sample_id"] = sample["sample_id"].astype(str)
    if (
        len(source) != expected_count
        or len(sample) != expected_count
        or source["sample_id"].duplicated().any()
        or sample["sample_id"].duplicated().any()
        or set(source["sample_id"]) != set(sample["sample_id"])
    ):
        raise GTGraspAuthorityError("GT grasp/sample denominator differs")
    by_id = source.set_index("sample_id", drop=False)
    output = pd.DataFrame(
        {
            "sample_id": sample["sample_id"].tolist(),
            "gt_grasp_rectangles": [
                _normalize_json(by_id.loc[sample_id, column])
                for sample_id in sample["sample_id"]
            ],
        }
    )
    registry_path = root / REGISTRY_RELATIVE_PATH
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if registry_path.exists() or manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"GT grasp authority exists; pass --resume: {manifest_path}"
            )
        validate_gt_grasp_authority(
            root, artifact_record(manifest_path), expected_count=expected_count
        )
        return manifest_path
    registry = atomic_parquet(output, registry_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "scientific_role": "post-lock frozen Test grasp geometry",
        "sample_count": expected_count,
        "gt_grasp_rows_read": expected_count,
        "source_column": column,
        "protocol_lock": dict(authority["protocol_lock"]),
        "execution_authority_mode": access["mode"],
        "execution_claim": dict(access["record"]),
        "sample_manifest": dict(authority["sample_manifest"]),
        "gt_grasp_source": dict(source_record),
        "registry": artifact_record(registry),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(manifest_path, payload)
    append_gt_access_log(
        root,
        {
            "stage": "P3_GT_GRASP_AUTHORITY",
            "purpose": "post_lock_evaluator_geometry_materialisation",
            "protocol_lock": dict(authority["protocol_lock"]),
            "execution_authority_mode": access["mode"],
            "execution_claim": dict(access["record"]),
            "source": dict(source_record),
            "gt_grasp_rows_read": expected_count,
            "gt_mask_pixels_read": 0,
        },
    )
    validate_gt_grasp_authority(
        root, artifact_record(manifest_path), expected_count=expected_count
    )
    return manifest_path


def validate_gt_grasp_authority(
    run_dir: str | Path,
    manifest_record: Mapping[str, Any],
    *,
    expected_count: int = 7_675,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Replay the post-lock authority and exact source projection."""

    root = Path(run_dir).expanduser().resolve()
    manifest_path = _verified_record(manifest_record, label="GT grasp authority")
    if manifest_path != root / MANIFEST_RELATIVE_PATH:
        raise PermissionError("GT grasp authority path is noncanonical")
    value = _object(manifest_path, label="GT grasp authority")
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise GTGraspAuthorityError("GT grasp authority content hash differs")
    authority = load_execution_authority(value["protocol_lock"]["path"])
    access = resolve_postlock_access_authority(root, authority=authority)
    if (
        value.get("status") != "COMPLETE"
        or int(value.get("sample_count", -1)) != expected_count
        or int(value.get("gt_grasp_rows_read", -1)) != expected_count
        or value.get("protocol_lock") != authority["protocol_lock"]
        or value.get("execution_authority_mode") != access["mode"]
        or value.get("execution_claim") != access["record"]
        or value.get("sample_manifest") != authority["sample_manifest"]
        or value.get("gt_grasp_source") != authority.get("gt_grasp_source")
    ):
        raise GTGraspAuthorityError("GT grasp authority binding differs")
    registry_path = _verified_record(value["registry"], label="GT grasp registry")
    if registry_path != root / REGISTRY_RELATIVE_PATH:
        raise PermissionError("GT grasp registry path is noncanonical")
    observed = pd.read_parquet(registry_path)
    if (
        list(observed.columns) != ["sample_id", "gt_grasp_rectangles"]
        or len(observed) != expected_count
        or observed["sample_id"].astype(str).duplicated().any()
    ):
        raise GTGraspAuthorityError("GT grasp registry schema/count differs")
    return value, observed


__all__ = [
    "GTGraspAuthorityError",
    "MANIFEST_RELATIVE_PATH",
    "REGISTRY_RELATIVE_PATH",
    "materialize_gt_grasp_authority",
    "validate_gt_grasp_authority",
]
