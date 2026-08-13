"""Content-addressed P2-PASS source view for the frozen G1/C1 runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io import (
    artifact_record,
    atomic_copy,
    atomic_json,
    atomic_parquet,
    canonical_sha256,
    sha256_file,
)


EXPECTED_SAMPLE_COUNT = 7_675
LABEL_PROJECTION = (
    "sample_id",
    "prepared_gt_mask_path",
    "prepared_gt_mask_sha256",
)
MANIFEST_NAME = "ADAPTER_MANIFEST.json"


class G1C1AdapterError(RuntimeError):
    """The executable source view differs from the locked P2 partition."""


def _object(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=False)
    if path.expanduser().is_symlink() or not source.is_file():
        raise G1C1AdapterError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise G1C1AdapterError(f"cannot parse {label}: {source}") from error
    if not isinstance(value, dict):
        raise G1C1AdapterError(f"{label} must contain one JSON object")
    return value


def _verify_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(unsigned):
        raise G1C1AdapterError(f"{label} content hash differs")


def _ids(frame: pd.DataFrame, *, label: str) -> list[str]:
    if "sample_id" not in frame or frame["sample_id"].isna().any():
        raise G1C1AdapterError(f"{label} lacks sample IDs")
    values = frame["sample_id"].astype(str).tolist()
    if len(set(values)) != len(values) or any(not value for value in values):
        raise G1C1AdapterError(f"{label} sample identities are duplicated or empty")
    return values


def _copy_exact(source: Path, destination: Path) -> Path:
    if destination.exists():
        if destination.is_symlink() or sha256_file(destination) != sha256_file(source):
            raise G1C1AdapterError(f"existing adapter member differs: {destination}")
        return destination
    return atomic_copy(source, destination)


def build_g1_c1_source_adapter(
    *,
    run_dir: str | Path,
    source_run: str | Path,
    registry_path: str | Path,
    mapping_audit_path: str | Path,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
    resume: bool = False,
) -> Path:
    """Project only the executable P2 partition without opening GT mask pixels.

    The label source is column-projected at read time, so GT grasp rectangles
    are neither loaded nor copied into the executable view.
    """

    root = Path(run_dir).expanduser().resolve()
    source = Path(source_run).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    mapping_path = Path(mapping_audit_path).expanduser().resolve()
    sample_source = source / "manifests/test_samples.parquet"
    label_source = source / "manifests/test_labels.parquet"
    for path, label in (
        (sample_source, "frozen deployment manifest"),
        (label_source, "frozen label manifest"),
        (registry, "P2 GT registry"),
        (mapping_path, "P2 mapping audit"),
    ):
        if path.is_symlink() or not path.is_file():
            raise G1C1AdapterError(f"{label} is missing or unsafe: {path}")
    mapping = _object(mapping_path, label="P2 mapping audit")
    _verify_self_hash(mapping, label="P2 mapping audit")
    if (
        mapping.get("status") != "PASS"
        or int(mapping.get("sample_count", -1)) != expected_count
        or mapping.get("outputs", {}).get("gt_mask_registry")
        != artifact_record(registry)
    ):
        raise G1C1AdapterError("P2 mapping audit/registry binding differs")

    deployment = pd.read_parquet(sample_source)
    # Security boundary: this is the only read of the frozen label table, and
    # the Parquet reader is explicitly limited to the three mask-reference
    # columns required by CompactSampleLoader.
    labels = pd.read_parquet(label_source, columns=list(LABEL_PROJECTION))
    registry_frame = pd.read_parquet(
        registry,
        columns=[
            "sample_id",
            "mapping_status",
            "pixel_qa_status",
            "bulk_gt_pixels_read",
            "prepared_gt_mask_path",
            "prepared_gt_mask_sha256",
        ],
    )
    deployment_ids = _ids(deployment, label="frozen deployment")
    label_ids = _ids(labels, label="frozen label projection")
    registry_ids = _ids(registry_frame, label="P2 registry")
    if (
        len(deployment_ids) != expected_count
        or set(deployment_ids) != set(label_ids)
        or set(deployment_ids) != set(registry_ids)
    ):
        raise G1C1AdapterError("G1/C1 adapter source universes differ")
    label_by_id = labels.set_index("sample_id", drop=False)
    registry_by_id = registry_frame.set_index("sample_id", drop=False)
    evaluable_ids = [
        sample_id
        for sample_id in deployment_ids
        if (
            registry_by_id.loc[sample_id, "mapping_status"] == "PASS"
            and registry_by_id.loc[sample_id, "pixel_qa_status"]
            == "P2_MAPPING_QA_PASS"
            and bool(registry_by_id.loc[sample_id, "bulk_gt_pixels_read"])
        )
    ]
    unresolved_ids = sorted(set(deployment_ids).difference(evaluable_ids))
    for sample_id in evaluable_ids:
        label_row = label_by_id.loc[sample_id]
        registry_row = registry_by_id.loc[sample_id]
        if (
            str(label_row["prepared_gt_mask_path"])
            != str(registry_row["prepared_gt_mask_path"])
            or str(label_row["prepared_gt_mask_sha256"])
            != str(registry_row["prepared_gt_mask_sha256"])
        ):
            raise G1C1AdapterError(
                f"P2/frozen prepared-mask identity differs: {sample_id}"
            )
    partition_identity = {
        "source_samples": artifact_record(sample_source),
        "source_labels": artifact_record(label_source),
        "registry": artifact_record(registry),
        "mapping_audit": artifact_record(mapping_path),
        "ordered_evaluable_ids_sha256": canonical_sha256(evaluable_ids),
        "sorted_unresolved_ids_sha256": canonical_sha256(unresolved_ids),
    }
    partition_sha = canonical_sha256(partition_identity)
    destination = (
        root
        / "05_gtmask_inputs/g1_c1_oracle_source_adapter"
        / partition_sha[:24]
    )
    manifest_path = destination / MANIFEST_NAME
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"G1/C1 adapter exists; pass --resume: {manifest_path}")
        verify_g1_c1_source_adapter(manifest_path, expected_count=expected_count)
        return manifest_path

    executable_set = set(evaluable_ids)
    deployment_projection = deployment.loc[
        deployment["sample_id"].astype(str).isin(executable_set)
    ].copy()
    label_projection = labels.set_index("sample_id", drop=False).loc[
        evaluable_ids
    ].reset_index(drop=True)
    if _ids(deployment_projection, label="adapter deployment") != evaluable_ids or (
        _ids(label_projection, label="adapter labels") != evaluable_ids
    ):
        raise G1C1AdapterError("adapter output order differs from frozen deployment")
    sample_output = atomic_parquet(
        deployment_projection,
        destination / "manifests/test_samples.parquet",
    )
    label_output = atomic_parquet(
        label_projection.loc[:, list(LABEL_PROJECTION)],
        destination / "manifests/test_labels.parquet",
    )
    configs: dict[str, dict[str, Any]] = {}
    for route in ("G1", "C1"):
        source_config = source / f"selected_configs/{route}.json"
        output_config = _copy_exact(
            source_config, destination / f"selected_configs/{route}.json"
        )
        configs[route.lower()] = artifact_record(output_config)
    unresolved_path = atomic_json(
        destination / "unresolved_partition.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "sample_count": len(unresolved_ids),
            "sample_ids": unresolved_ids,
            "sample_ids_sha256": canonical_sha256(unresolved_ids),
        },
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_run": str(source),
        "denominator_sample_count": expected_count,
        "evaluable_sample_count": len(evaluable_ids),
        "unresolved_sample_count": len(unresolved_ids),
        "ordered_evaluable_ids_sha256": canonical_sha256(evaluable_ids),
        "sorted_unresolved_ids_sha256": canonical_sha256(unresolved_ids),
        "partition_identity_sha256": partition_sha,
        "source_artifacts": partition_identity,
        "test_samples": artifact_record(sample_output),
        "test_labels_projection": artifact_record(label_output),
        "selected_configs": configs,
        "unresolved_partition": artifact_record(unresolved_path),
        "label_projection_columns": list(LABEL_PROJECTION),
        "gt_grasp_rows_read": 0,
        "gt_mask_pixels_read": 0,
        "candidate_generation_performed": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    verify_g1_c1_source_adapter(manifest_path, expected_count=expected_count)
    return manifest_path


def verify_g1_c1_source_adapter(
    manifest_path: str | Path, *, expected_count: int = EXPECTED_SAMPLE_COUNT
) -> dict[str, Any]:
    """Replay the adapter's exact membership and source-byte closure."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = _object(path, label="G1/C1 adapter manifest")
    _verify_self_hash(manifest, label="G1/C1 adapter manifest")
    if (
        manifest.get("status") != "COMPLETE"
        or int(manifest.get("denominator_sample_count", -1)) != expected_count
        or manifest.get("label_projection_columns") != list(LABEL_PROJECTION)
        or manifest.get("gt_grasp_rows_read") != 0
        or manifest.get("gt_mask_pixels_read") != 0
        or manifest.get("candidate_generation_performed") is not False
    ):
        raise G1C1AdapterError("G1/C1 adapter declaration differs")
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise G1C1AdapterError("G1/C1 adapter source inventory is malformed")
    if set(source_artifacts) != {
        "source_samples",
        "source_labels",
        "registry",
        "mapping_audit",
        "ordered_evaluable_ids_sha256",
        "sorted_unresolved_ids_sha256",
    }:
        raise G1C1AdapterError("G1/C1 adapter source keys differ")
    records = [
        manifest.get("test_samples"),
        manifest.get("test_labels_projection"),
        manifest.get("unresolved_partition"),
        *(
            manifest.get("selected_configs", {}).get(route)
            for route in ("g1", "c1")
        ),
        *(value for value in source_artifacts.values() if isinstance(value, Mapping)),
    ]
    for record in records:
        if not isinstance(record, Mapping):
            raise G1C1AdapterError("G1/C1 adapter artifact inventory is malformed")
        source = Path(str(record.get("path", ""))).expanduser().resolve()
        if source.is_symlink() or not source.is_file() or (
            sha256_file(source) != record.get("sha256")
        ):
            raise G1C1AdapterError("G1/C1 adapter artifact hash differs")
    samples = pd.read_parquet(manifest["test_samples"]["path"])
    labels = pd.read_parquet(manifest["test_labels_projection"]["path"])
    unresolved = _object(
        Path(manifest["unresolved_partition"]["path"]),
        label="G1/C1 unresolved partition",
    )
    sample_ids = _ids(samples, label="adapter deployment")
    label_ids = _ids(labels, label="adapter labels")
    unresolved_ids = [str(value) for value in unresolved.get("sample_ids", [])]
    if (
        sample_ids != label_ids
        or len(set(unresolved_ids)) != len(unresolved_ids)
        or set(sample_ids).intersection(unresolved_ids)
        or len(sample_ids) + len(unresolved_ids) != expected_count
        or manifest.get("evaluable_sample_count") != len(sample_ids)
        or manifest.get("unresolved_sample_count") != len(unresolved_ids)
        or manifest.get("ordered_evaluable_ids_sha256")
        != canonical_sha256(sample_ids)
        or manifest.get("sorted_unresolved_ids_sha256")
        != canonical_sha256(sorted(unresolved_ids))
        or unresolved.get("sample_ids_sha256") != canonical_sha256(unresolved_ids)
        or list(labels.columns) != list(LABEL_PROJECTION)
    ):
        raise G1C1AdapterError("G1/C1 adapter membership differs")
    expected_partition = {
        **{
            name: dict(source_artifacts[name])
            for name in ("source_samples", "source_labels", "registry", "mapping_audit")
        },
        "ordered_evaluable_ids_sha256": canonical_sha256(sample_ids),
        "sorted_unresolved_ids_sha256": canonical_sha256(sorted(unresolved_ids)),
    }
    if (
        dict(source_artifacts) != expected_partition
        or manifest.get("partition_identity_sha256")
        != canonical_sha256(expected_partition)
        or path.parent.name != canonical_sha256(expected_partition)[:24]
    ):
        raise G1C1AdapterError("G1/C1 adapter partition identity differs")
    return manifest


__all__ = [
    "EXPECTED_SAMPLE_COUNT",
    "G1C1AdapterError",
    "LABEL_PROJECTION",
    "MANIFEST_NAME",
    "build_g1_c1_source_adapter",
    "verify_g1_c1_source_adapter",
]
