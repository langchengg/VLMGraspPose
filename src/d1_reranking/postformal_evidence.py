"""Predeclare label-free inputs needed by D1 postformal analysis.

This producer runs before P13/P14.  It may hash the opaque visual-Test table,
but it must never deserialize any of its rows.  P15 is the first stage allowed
to open that table, after the exactly-once formal transaction is COMPLETE and
the independent recompute is PASS.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file

from .execution import load_content_manifest
from .provenance import load_source_closure
from .postformal_sources import (
    COVARIATES_RELATIVE_PATH,
    FORBIDDEN_COVARIATE_TOKENS,
    RUNTIME_COMPONENTS,
    load_postformal_sources,
)
from .validation_evidence import TABLE_NAMES, load_validation_evidence_tables


EVIDENCE_RELATIVE_PATH = "configs/d1_postformal_evidence.json"
EVIDENCE_TABLE_NAMES = TABLE_NAMES


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"D1 postformal evidence is not a regular file: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _same_record(observed: object, expected: Mapping[str, Any], *, name: str) -> None:
    if not isinstance(observed, Mapping):
        raise RuntimeError(f"{name} record is absent")
    for field in ("path", "sha256", "bytes"):
        if observed.get(field) != expected.get(field):
            raise RuntimeError(f"{name} {field} differs")


def _opaque_visual_record(closure: Mapping[str, Any]) -> dict[str, Any]:
    canonical = closure.get("canonical_inputs")
    test = canonical.get("test") if isinstance(canonical, Mapping) else None
    record = (
        test.get("opaque_visual_ground_truth") if isinstance(test, Mapping) else None
    )
    if not isinstance(record, Mapping):
        raise RuntimeError("D1 source closure misses opaque visual Test ground truth")
    current = _record(str(record.get("path", "")))
    _same_record(record, current, name="D1 opaque visual Test ground truth")
    return current


def _validate_runtime_manifest(path: Path, *, component: str) -> dict[str, Any]:
    value = load_content_manifest(
        path,
        name=f"D1 {component} runtime source",
        statuses=("COMPLETE", "NOT_AVAILABLE"),
    )
    if value.get("candidate_test_labels_read") is not False:
        raise PermissionError(f"D1 {component} runtime source is not label-free")
    if (
        value.get("status") == "NOT_AVAILABLE"
        and not str(value.get("unavailable_reason", "")).strip()
    ):
        raise RuntimeError(f"D1 {component} unavailable telemetry lacks a reason")
    if component == "feature_extraction" and value.get("status") == "COMPLETE":
        telemetry = value.get("telemetry", value)
        if (
            not isinstance(telemetry, Mapping)
            or telemetry.get("measurement_semantics")
            != "feature_extraction_not_artifact_loading"
        ):
            raise RuntimeError(
                "D1 feature telemetry must measure extraction, not artifact loading"
            )
    verify_artifact_records_recursive(
        value,
        name=f"D1 {component} runtime source",
        require_at_least_one=False,
    )
    return _record(path)


def assemble_postformal_evidence(
    run_dir: str | Path,
    *,
    q_saturation_threshold: float,
    mask_quality_threshold: float,
    resume: bool = False,
) -> dict[str, Any]:
    """Write the immutable P15 evidence declaration without opening Test rows."""

    root = Path(run_dir).expanduser().resolve()
    destination = root / EVIDENCE_RELATIVE_PATH
    forbidden = (
        root / "08_lock" / "FORMAL_TEST_LOCK.json",
        root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json",
        root / "FINAL_RUN_LOCK.json",
        root / "COMPLETE",
    )
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise PermissionError(
            f"D1 postformal evidence must be declared before formal lock: {present}"
        )
    if not 0.75 < float(q_saturation_threshold) <= 1.0:
        raise ValueError("D1 q-saturation threshold must be in (0.75, 1]")
    if not 0.0 <= float(mask_quality_threshold) <= 1.0:
        raise ValueError("D1 mask-quality threshold must be in [0, 1]")
    validation_path, validation = load_validation_evidence_tables(root)
    declared_tables = validation.get("artifacts", {}).get("tables")
    if not isinstance(declared_tables, Mapping) or set(declared_tables) != set(
        EVIDENCE_TABLE_NAMES
    ):
        raise RuntimeError("D1 fixed Validation evidence table inventory differs")
    table_records: dict[str, dict[str, Any]] = {}
    for name in EVIDENCE_TABLE_NAMES:
        record = declared_tables[name]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"D1 fixed Validation table record is absent: {name}")
        path = Path(str(record.get("path", ""))).resolve()
        frame = pd.read_csv(path)
        if frame.empty:
            raise RuntimeError(f"D1 postformal evidence table is empty: {name}")
        forbidden_columns = [
            column
            for column in frame.columns
            if "test" in str(column).lower()
            and any(
                token in str(column).lower()
                for token in ("label", "correct", "success")
            )
        ]
        if forbidden_columns:
            raise PermissionError(
                f"D1 preformal evidence table exposes Test outcomes: {name}/{forbidden_columns}"
            )
        table_records[name] = _record(path)
    source_manifest_path, source_manifest = load_postformal_sources(root)
    source_artifacts = source_manifest.get("artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise RuntimeError("D1 postformal source artifact inventory is absent")
    covariates_path = Path(
        str(source_artifacts.get("sample_covariates", {}).get("path", ""))
    ).resolve()
    if covariates_path != (root / COVARIATES_RELATIVE_PATH).resolve():
        raise RuntimeError("D1 postformal sample-covariate path is not fixed")
    covariates = pd.read_parquet(covariates_path)
    required_covariates = {
        "sample_id",
        "target_size",
        "relation_query",
        "clutter",
        "depth_missing",
        "predicted_mask_confidence",
        "native_mask_support",
        "selected_mask_support",
    }
    if required_covariates.difference(covariates.columns):
        raise RuntimeError("D1 sample covariates miss predeclared columns")
    if covariates.empty or covariates["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("D1 sample covariates are empty or duplicate sample IDs")
    forbidden_covariates = [
        str(column)
        for column in covariates.columns
        if any(token in str(column).lower() for token in FORBIDDEN_COVARIATE_TOKENS)
    ]
    if forbidden_covariates:
        raise PermissionError(
            "D1 preformal covariates expose outcome/GT-derived fields: "
            f"{sorted(forbidden_covariates)}"
        )

    closure_path, closure = load_source_closure(root)
    visual_record = _opaque_visual_record(closure)
    declared_runtime = source_artifacts.get("runtime_manifests")
    if not isinstance(declared_runtime, Mapping) or set(declared_runtime) != set(
        RUNTIME_COMPONENTS
    ):
        raise RuntimeError("D1 fixed runtime evidence component inventory differs")
    runtime_records = {}
    for component in RUNTIME_COMPONENTS:
        record = declared_runtime[component]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"D1 fixed runtime record is absent: {component}")
        runtime_records[component] = _validate_runtime_manifest(
            Path(str(record.get("path", ""))).resolve(), component=component
        )
        _same_record(
            record,
            runtime_records[component],
            name=f"D1 fixed runtime {component}",
        )
    contribution_record = source_artifacts.get("ranker_contributions")
    if not isinstance(contribution_record, Mapping):
        raise RuntimeError("D1 fixed ranker-contribution declaration is absent")
    contribution_path = Path(str(contribution_record.get("path", ""))).resolve()
    contribution = load_content_manifest(
        contribution_path,
        name="D1 ranker contribution declaration",
        statuses=("COMPLETE", "NOT_APPLICABLE"),
    )
    if contribution.get("candidate_test_labels_read") is not False:
        raise PermissionError("D1 ranker contributions are not label-free")
    _same_record(
        contribution_record,
        _record(contribution_path),
        name="D1 fixed ranker contribution declaration",
    )
    sources = {
        "source_closure": _record(closure_path),
        "postformal_sources": _record(source_manifest_path),
        "validation_evidence_tables": _record(validation_path),
        "opaque_visual_ground_truth": visual_record,
        "tables": table_records,
        "sample_covariates": _record(covariates_path),
        "runtime_manifests": runtime_records,
        "ranker_contributions": _record(contribution_path),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED",
        "declared_before_prelock": True,
        "declared_before_formal_lock": True,
        "candidate_test_labels_read": False,
        "opaque_visual_ground_truth_opened_as_table": False,
        "postformal_derived_fields": {
            "mask_quality": "binary IoU from locked predicted/GT masks after P14 COMPLETE and P17 PASS",
            "three_route_correct": "frozen best-ranked non-D1 router candidate outcome after P14 COMPLETE",
        },
        "q_saturation_threshold": float(q_saturation_threshold),
        "mask_quality_threshold": float(mask_quality_threshold),
        "tables": table_records,
        "sample_covariates": _record(covariates_path),
        "runtime_manifests": runtime_records,
        "ranker_contributions": _record(contribution_path),
        "opaque_visual_ground_truth": visual_record,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    if destination.exists():
        existing = load_content_manifest(
            destination, name="D1 postformal evidence", statuses=("LOCKED",)
        )
        if not resume:
            raise FileExistsError(
                f"D1 postformal evidence already exists; pass --resume: {destination}"
            )
        if existing != payload:
            raise RuntimeError("D1 postformal evidence differs from current sources")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    verify_artifact_records_recursive(
        sources,
        name="D1 postformal evidence source recheck",
        require_at_least_one=True,
    )
    atomic_json(destination, payload)
    return payload


def validate_postformal_evidence_prelock(
    run_dir: str | Path,
    *,
    source_closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact P13 record tree for a previously produced declaration."""

    root = Path(run_dir).expanduser().resolve()
    path = root / EVIDENCE_RELATIVE_PATH
    value = load_content_manifest(
        path, name="D1 postformal evidence", statuses=("LOCKED",)
    )
    postformal_sources_path, _ = load_postformal_sources(root)
    validation_path, _ = load_validation_evidence_tables(root)
    if (
        value.get("declared_before_prelock") is not True
        or value.get("declared_before_formal_lock") is not True
        or value.get("candidate_test_labels_read") is not False
        or value.get("opaque_visual_ground_truth_opened_as_table") is not False
    ):
        raise RuntimeError("D1 postformal evidence declaration state differs")
    expected_visual = _opaque_visual_record(source_closure)
    _same_record(
        value.get("opaque_visual_ground_truth"),
        expected_visual,
        name="D1 postformal visual authority",
    )
    sources = value.get("sources")
    if not isinstance(sources, Mapping) or value.get(
        "source_signature_sha256"
    ) != canonical_sha256(sources):
        raise RuntimeError("D1 postformal evidence source signature differs")
    _same_record(
        sources.get("postformal_sources"),
        _record(postformal_sources_path),
        name="D1 postformal source manifest",
    )
    _same_record(
        sources.get("validation_evidence_tables"),
        _record(validation_path),
        name="D1 Validation evidence table manifest",
    )
    verify_artifact_records_recursive(
        sources,
        name="D1 postformal evidence prelock sources",
        require_at_least_one=True,
    )
    return {"manifest": _record(path), "sources": dict(sources)}
