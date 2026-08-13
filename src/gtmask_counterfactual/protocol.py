"""Content-addressed protocol lock and exactly-once bulk claim."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import verify_artifact_records_recursive

from .audit import transition_pipeline_status
from .execution_contracts import validate_route_contracts, validate_scientific_bindings
from .contracts import REQUIRED_PROTOCOL_BINDINGS, RunState
from .io import (
    artifact_record,
    atomic_json,
    canonical_sha256,
    exclusive_json,
    sha256_file,
)
from .mapping import EXPECTED_SAMPLE_COUNT


LOCK_RELATIVE_PATH = Path("01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json")
EXECUTION_RELATIVE_PATH = Path("01_protocol_lock/COUNTERFACTUAL_EXECUTION.json")
EXECUTION_COMPLETION_RELATIVE_PATH = Path(
    "01_protocol_lock/COUNTERFACTUAL_EXECUTION_COMPLETE.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"protocol artifact is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"protocol artifact is not a JSON object: {path}")
    return value


def _verify_binding_artifacts(value: Any, *, location: str) -> int:
    """Verify every artifact record while permitting canonical inline values."""

    count = 0
    if isinstance(value, Mapping):
        if value.get("kind") == "inline":
            inline = value.get("value")
            if value.get("sha256") != canonical_sha256(inline):
                raise RuntimeError(f"{location} inline canonical hash differs")
            return 0
        has_path = "path" in value
        has_sha = "sha256" in value
        if has_path != has_sha:
            raise ValueError(f"{location} has an incomplete artifact record")
        if has_path:
            verify_artifact_records_recursive(
                value, name=location, require_at_least_one=True
            )
            return 1
        for key, child in value.items():
            count += _verify_binding_artifacts(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            count += _verify_binding_artifacts(child, location=f"{location}[{index}]")
    return count


def inline_binding(value: Any) -> dict[str, Any]:
    return {"kind": "inline", "value": value, "sha256": canonical_sha256(value)}


def _contains_inline_binding(value: Any) -> bool:
    if isinstance(value, Mapping):
        return value.get("kind") == "inline" or any(
            _contains_inline_binding(child) for child in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_inline_binding(child) for child in value)
    return False


def _artifact_records(value: Any) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if value.get("kind") == "inline":
            return records
        if "path" in value and "sha256" in value:
            records.append(value)
            return records
        for child in value.values():
            records.extend(_artifact_records(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            records.extend(_artifact_records(child))
    return records


def _single_artifact_record(value: Any, *, name: str) -> dict[str, Any]:
    records = _artifact_records(value)
    if len(records) != 1:
        raise ValueError(f"protocol binding must contain exactly one artifact: {name}")
    return dict(records[0])


def _binding_payload(value: Any, *, name: str) -> Any:
    if isinstance(value, Mapping) and value.get("kind") == "inline":
        return value.get("value")
    record = _single_artifact_record(value, name=name)
    payload = _read_object(Path(str(record["path"])).expanduser().resolve())
    if "self_sha256" in payload:
        unsigned = dict(payload)
        recorded = unsigned.pop("self_sha256")
        if recorded != canonical_sha256(unsigned):
            raise RuntimeError(f"{name} self hash differs")
    if "content_sha256" in payload:
        unsigned = dict(payload)
        recorded = unsigned.pop("content_sha256")
        if recorded != canonical_sha256(unsigned):
            raise RuntimeError(f"{name} content hash differs")
    return payload


def _validate_declaration(
    declaration: Mapping[str, Any], *, test_only_allow_synthetic_contract: bool = False
) -> dict[str, Any]:
    if declaration.get("gt_candidate_generation_authorized") is not True:
        raise PermissionError("protocol must authorize GT candidate generation")
    if declaration.get("bulk_execution_max_count") != 1:
        raise ValueError("counterfactual bulk execution maximum must equal one")
    mapping_reads = declaration.get("mapping_qa_gt_mask_rows_read_before_lock")
    if isinstance(mapping_reads, bool) or not isinstance(mapping_reads, int):
        raise ValueError("mapping-QA GT-mask row count must be an integer")
    if mapping_reads <= 0:
        raise ValueError("mapping-QA GT-mask row count must be positive")
    if declaration.get("candidate_generation_gt_mask_rows_read_before_lock") != 0:
        raise PermissionError("candidate generation must not read GT masks before lock")
    routes = declaration.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != {"g1", "c1", "d1"}:
        raise ValueError("protocol routes must be exactly g1, c1, and d1")
    if not all(isinstance(routes[route], Mapping) for route in ("g1", "c1", "d1")):
        raise ValueError("every protocol route contract must be an object")
    if not test_only_allow_synthetic_contract:
        validate_route_contracts(routes)
    d1 = routes["d1"]
    if (
        d1.get("case") != "B"
        or d1.get("mask_affects_raw_sampling") is not True
        or d1.get("raw_candidate_regeneration_required") is not True
        or d1.get("filter_only_primary_allowed") is not False
    ):
        raise ValueError(
            "D1 must lock Case B raw regeneration and forbid filter-only primary"
        )
    return dict(declaration)


def _validate_bindings(
    bindings: Mapping[str, Any],
    *,
    declaration: Mapping[str, Any],
    test_only_allow_synthetic_contract: bool = False,
) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_PROTOCOL_BINDINGS).difference(bindings))
    if missing:
        raise ValueError(f"protocol lock lacks required bindings: {missing}")
    artifact_required = {
        "source_locks",
        "source_code",
        "configs",
        "baseline_replay",
        "sample_manifest",
        "gt_mask_registry",
        "mapping_qa",
        "evaluator",
    }
    total_artifacts = 0
    for name in REQUIRED_PROTOCOL_BINDINGS:
        value = bindings[name]
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"protocol binding is empty or malformed: {name}")
        artifact_count = _verify_binding_artifacts(value, location=f"bindings.{name}")
        if name in artifact_required and artifact_count == 0:
            raise ValueError(f"protocol binding must contain an artifact: {name}")
        if artifact_count == 0 and not _contains_inline_binding(value):
            raise ValueError(f"protocol binding is not content-addressed: {name}")
        total_artifacts += artifact_count
    if total_artifacts == 0:
        raise ValueError("protocol lock must bind at least one real source artifact")
    baseline_replay = _single_artifact_record(
        bindings["baseline_replay"], name="baseline_replay"
    )
    sample_manifest = _single_artifact_record(
        bindings["sample_manifest"], name="sample_manifest"
    )
    gt_mask_registry = _single_artifact_record(
        bindings["gt_mask_registry"], name="gt_mask_registry"
    )
    mapping_qa_record = _single_artifact_record(
        bindings["mapping_qa"], name="mapping_qa"
    )
    mapping_qa = _binding_payload(bindings["mapping_qa"], name="mapping_qa")
    if not isinstance(mapping_qa, Mapping):
        raise ValueError("mapping QA artifact must be a JSON object")
    expected_reads = declaration["mapping_qa_gt_mask_rows_read_before_lock"]
    mapping_outputs = mapping_qa.get("outputs")
    mapping_inputs = mapping_qa.get("inputs")
    manual_qa = mapping_qa.get("manual_asset_qa")
    asset_qa = mapping_qa.get("asset_qa")
    query_counts = asset_qa.get("query_type_counts") if isinstance(asset_qa, Mapping) else None
    quartile_counts = (
        asset_qa.get("target_size_quartile_counts")
        if isinstance(asset_qa, Mapping)
        else None
    )
    if test_only_allow_synthetic_contract:
        if (
            mapping_qa.get("status") != "PASS"
            or mapping_qa.get("stage") != RunState.P2_GT_MAPPING_PASS.value
            or mapping_qa.get("pixel_qa_status") != "P2_MAPPING_QA_PASS"
            or mapping_qa.get("mapping_qa_gt_mask_rows_read") != expected_reads
            or mapping_qa.get("candidate_generation_gt_mask_rows_read") != 0
        ):
            raise ValueError("synthetic P2 mapping pixel-QA artifact contract differs")
    elif (
        mapping_qa.get("status") != "PASS"
        or mapping_qa.get("stage") != RunState.P2_GT_MAPPING_PASS.value
        or mapping_qa.get("pixel_qa_status") != "P2_MAPPING_QA_PASS"
        or mapping_qa.get("sample_count") != EXPECTED_SAMPLE_COUNT
        or mapping_qa.get("partition_total") != EXPECTED_SAMPLE_COUNT
        or not isinstance(mapping_qa.get("counterfactual_evaluable_count"), int)
        or not isinstance(mapping_qa.get("unresolved_count"), int)
        or mapping_qa["counterfactual_evaluable_count"]
        + mapping_qa["unresolved_count"]
        != EXPECTED_SAMPLE_COUNT
        or mapping_qa.get("mapping_qa_gt_mask_rows_read") != expected_reads
        or mapping_qa.get("candidate_generation_gt_mask_rows_read") != 0
        or not isinstance(mapping_outputs, Mapping)
        or mapping_outputs.get("gt_mask_registry") != gt_mask_registry
        or not isinstance(mapping_inputs, Mapping)
        or mapping_inputs.get("counterfactual_manifest") != sample_manifest
        or not isinstance(manual_qa, Mapping)
        or manual_qa.get("status") != "PASS"
        or manual_qa.get("reviewed_case_count", 0) < 150
        or manual_qa.get("reviewed_case_count") != manual_qa.get("expected_case_count")
        or manual_qa.get("reviewed_sample_ids_sha256")
        != manual_qa.get("selected_sample_ids_sha256")
        or not isinstance(manual_qa.get("artifact"), Mapping)
        or not isinstance(asset_qa, Mapping)
        or asset_qa.get("status") != "PASS"
        or asset_qa.get("selected_case_count", 0) < 150
        or asset_qa.get("same_cases_shared_across_routes") is not True
        or not isinstance(query_counts, Mapping)
        or not query_counts
        or any(int(count) < 20 for count in query_counts.values())
        or not isinstance(quartile_counts, Mapping)
        or set(quartile_counts) != {"1", "2", "3", "4"}
        or any(int(count) < 20 for count in quartile_counts.values())
    ):
        raise ValueError("final P2 mapping pixel-QA artifact contract differs")
    if not test_only_allow_synthetic_contract:
        verify_artifact_records_recursive(
            mapping_inputs,
            name="P2 mapping inputs",
            require_at_least_one=True,
        )
        verify_artifact_records_recursive(
            manual_qa["artifact"],
            name="P2 manual mapping QA",
            require_at_least_one=True,
        )
        contact_sheets = mapping_qa.get("contact_sheets")
        contact_artifact = (
            contact_sheets.get("artifact")
            if isinstance(contact_sheets, Mapping)
            else None
        )
        if (
            not isinstance(contact_sheets, Mapping)
            or contact_sheets.get("status") != "RENDERED_PENDING_MANUAL_QA"
            or contact_sheets.get("case_count", 0) < 150
            or not isinstance(contact_artifact, Mapping)
        ):
            raise ValueError("P2 contact-sheet artifact contract differs")
        verify_artifact_records_recursive(
            contact_artifact,
            name="P2 contact-sheet manifest",
            require_at_least_one=True,
        )
        _validate_contact_sheet_manifest(
            contact_artifact,
            expected_count=int(asset_qa["selected_case_count"]),
            expected_sample_ids_sha256=str(asset_qa["selected_sample_ids_sha256"]),
        )
        evidence_record = mapping_outputs.get("per_sample_evidence_manifest")
        if not isinstance(evidence_record, Mapping):
            raise ValueError("P2 per-sample evidence manifest is absent")
        _validate_per_sample_evidence_manifest(evidence_record)
        _validate_p1_audit(mapping_qa=mapping_qa, sample_manifest=sample_manifest)
        _validate_exact_partition(
            sample_manifest=sample_manifest,
            gt_mask_registry=gt_mask_registry,
            mapping_qa=mapping_qa,
        )
    route_contracts = _binding_payload(
        bindings["route_contracts"], name="route_contracts"
    )
    if isinstance(route_contracts, Mapping) and "routes" in route_contracts:
        route_contracts = route_contracts["routes"]
    if route_contracts != declaration["routes"]:
        raise ValueError("route-contract binding differs from protocol declaration")
    if not test_only_allow_synthetic_contract:
        from .predicted_replay import validate_predicted_replay_closure

        validate_predicted_replay_closure(baseline_replay["path"])
        validate_scientific_bindings(bindings)
    return {
        "baseline_replay": baseline_replay,
        "sample_manifest": sample_manifest,
        "gt_mask_registry": gt_mask_registry,
        "mapping_qa": mapping_qa_record,
        "routes": dict(declaration["routes"]),
    }


def _validate_exact_partition(
    *,
    sample_manifest: Mapping[str, Any],
    gt_mask_registry: Mapping[str, Any],
    mapping_qa: Mapping[str, Any],
) -> None:
    """Recompute the locked 7,675-row PASS/unresolved partition from artifacts."""

    import pandas as pd

    manifest = pd.read_parquet(
        Path(str(sample_manifest["path"])), columns=["sample_id"]
    )
    registry = pd.read_parquet(
        Path(str(gt_mask_registry["path"])),
        columns=["sample_id", "mapping_status"],
    )
    manifest_ids = [str(value) for value in manifest["sample_id"].tolist()]
    registry_ids = [str(value) for value in registry["sample_id"].tolist()]
    if (
        len(manifest_ids) != EXPECTED_SAMPLE_COUNT
        or len(set(manifest_ids)) != EXPECTED_SAMPLE_COUNT
        or len(registry_ids) != EXPECTED_SAMPLE_COUNT
        or len(set(registry_ids)) != EXPECTED_SAMPLE_COUNT
        or set(manifest_ids) != set(registry_ids)
    ):
        raise ValueError("P2 manifest/registry denominator identity differs")
    unresolved_ids = sorted(
        str(row.sample_id)
        for row in registry.itertuples(index=False)
        if row.mapping_status != "PASS"
    )
    evaluable = EXPECTED_SAMPLE_COUNT - len(unresolved_ids)
    if (
        mapping_qa.get("counterfactual_evaluable_count") != evaluable
        or mapping_qa.get("unresolved_count") != len(unresolved_ids)
        or mapping_qa.get("unresolved_sample_ids_sha256")
        != canonical_sha256(unresolved_ids)
    ):
        raise ValueError("P2 PASS/unresolved partition differs from registry rows")
    outputs = mapping_qa.get("outputs")
    unresolved_record = (
        outputs.get("mapping_unresolved_samples")
        if isinstance(outputs, Mapping)
        else None
    )
    if not isinstance(unresolved_record, Mapping):
        raise ValueError("P2 unresolved partition artifact is absent")
    verify_artifact_records_recursive(
        unresolved_record, name="P2 unresolved partition", require_at_least_one=True
    )
    unresolved_frame = pd.read_csv(Path(str(unresolved_record["path"])))
    if "sample_id" not in unresolved_frame.columns:
        raise ValueError("P2 unresolved partition lacks sample_id")
    recorded_ids = sorted(str(value) for value in unresolved_frame["sample_id"].tolist())
    if len(recorded_ids) != len(set(recorded_ids)) or recorded_ids != unresolved_ids:
        raise ValueError("P2 unresolved CSV differs from registry partition")


def _validate_p1_audit(
    *, mapping_qa: Mapping[str, Any], sample_manifest: Mapping[str, Any]
) -> None:
    inputs = mapping_qa.get("inputs")
    record = inputs.get("join_audit") if isinstance(inputs, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError("P2 does not bind the P1 join audit")
    verify_artifact_records_recursive(
        record, name="P1 join audit", require_at_least_one=True
    )
    audit = _read_object(Path(str(record["path"])))
    unsigned = dict(audit)
    content_hash = unsigned.pop("content_sha256", None)
    outputs = audit.get("outputs")
    if (
        content_hash != canonical_sha256(unsigned)
        or audit.get("status") != "PASS"
        or audit.get("stage") != RunState.P1_BASELINE_REPLAY_PASS.value
        or audit.get("sample_count") != EXPECTED_SAMPLE_COUNT
        or audit.get("partition_total") != EXPECTED_SAMPLE_COUNT
        or audit.get("counterfactual_evaluable_count", -1)
        + audit.get("unresolved_count", -1)
        != EXPECTED_SAMPLE_COUNT
        or audit.get("row_number_join_used") is not False
        or audit.get("gt_pixels_read") is not False
        or audit.get("gt_grasp_rows_read") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get("counterfactual_manifest_parquet") != sample_manifest
        or not isinstance(audit.get("sources"), Mapping)
        or set(audit["sources"]) != {
            "denominator",
            "g1",
            "c1",
            "d1",
            "unified_final_lock",
            "d1_final_lock",
        }
    ):
        raise ValueError("P1 exact denominator join audit contract differs")
    verify_artifact_records_recursive(
        audit["sources"], name="P1 locked sources", require_at_least_one=True
    )
    verify_artifact_records_recursive(
        outputs, name="P1 manifest outputs", require_at_least_one=True
    )


def _validate_contact_sheet_manifest(
    record: Mapping[str, Any],
    *,
    expected_count: int,
    expected_sample_ids_sha256: str,
) -> None:
    value = _read_object(Path(str(record["path"])))
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    sheets = value.get("sheets")
    if (
        recorded != canonical_sha256(unsigned)
        or value.get("status") != "RENDERED_PENDING_MANUAL_QA"
        or value.get("case_count") != expected_count
        or value.get("selected_sample_ids_sha256") != expected_sample_ids_sha256
        or not isinstance(sheets, Sequence)
        or isinstance(sheets, (str, bytes, bytearray))
        or len(sheets) != value.get("sheet_count")
    ):
        raise ValueError("P2 contact-sheet manifest content differs")
    verify_artifact_records_recursive(
        sheets, name="P2 contact-sheet PNGs", require_at_least_one=True
    )
    sample_ids = [
        str(sample_id)
        for sheet in sheets
        if isinstance(sheet, Mapping)
        for sample_id in sheet.get("sample_ids", [])
    ]
    if (
        len(sample_ids) != expected_count
        or len(sample_ids) != len(set(sample_ids))
        or canonical_sha256(sorted(sample_ids)) != expected_sample_ids_sha256
    ):
        raise ValueError("P2 contact-sheet sample coverage differs")


def _validate_per_sample_evidence_manifest(record: Mapping[str, Any]) -> None:
    verify_artifact_records_recursive(
        record, name="P2 per-sample evidence manifest", require_at_least_one=True
    )
    value = _read_object(Path(str(record["path"])))
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    records = value.get("records")
    if (
        recorded != canonical_sha256(unsigned)
        or value.get("status") != "COMPLETE"
        or value.get("sample_count") != EXPECTED_SAMPLE_COUNT
        or not isinstance(records, Sequence)
        or isinstance(records, (str, bytes, bytearray))
        or len(records) != EXPECTED_SAMPLE_COUNT
    ):
        raise ValueError("P2 per-sample evidence manifest content differs")
    verify_artifact_records_recursive(
        records, name="P2 per-sample evidence shards", require_at_least_one=True
    )


def create_protocol_lock(
    run_dir: str | Path,
    *,
    bindings: Mapping[str, Any],
    declaration: Mapping[str, Any],
    test_only_allow_synthetic_contract: bool = False,
) -> Path:
    root = Path(run_dir).expanduser().resolve()
    destination = root / LOCK_RELATIVE_PATH
    if destination.exists():
        raise FileExistsError("counterfactual protocol lock already exists")
    status = _read_object(root / "pipeline_status.json")
    if status.get("status") != RunState.P2_GT_MAPPING_PASS.value:
        raise PermissionError("protocol lock requires P2_GT_MAPPING_PASS")
    if int(status.get("counterfactual_execution_count", -1)) != 0:
        raise PermissionError("counterfactual execution count is already non-zero")
    if (
        test_only_allow_synthetic_contract
        and "PYTEST_CURRENT_TEST" not in __import__("os").environ
    ):
        raise PermissionError("synthetic route contracts are restricted to pytest")
    if not test_only_allow_synthetic_contract:
        for name, value in (("bindings", bindings), ("declaration", declaration)):
            unsigned = dict(value)
            recorded = unsigned.pop("content_sha256", None)
            if recorded != canonical_sha256(unsigned):
                raise RuntimeError(f"protocol {name} input content hash differs")
    validated_declaration = _validate_declaration(
        declaration,
        test_only_allow_synthetic_contract=test_only_allow_synthetic_contract,
    )
    execution_bindings = _validate_bindings(
        bindings,
        declaration=validated_declaration,
        test_only_allow_synthetic_contract=test_only_allow_synthetic_contract,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED",
        "created_at_utc": _now(),
        "scientific_role": "post-formal oracle stage-replacement diagnostic",
        "counterfactual_execution_count": 0,
        "formal_test_execution_count_increment": 0,
        "mapping_qa_gt_mask_rows_read_before_lock": validated_declaration[
            "mapping_qa_gt_mask_rows_read_before_lock"
        ],
        "candidate_generation_gt_mask_rows_read_before_lock": 0,
        "gt_candidate_generation_authorized": True,
        "gt_mapping_pixel_qa_status": "PASS",
        "sample_manifest": execution_bindings["sample_manifest"],
        "gt_mask_registry": execution_bindings["gt_mask_registry"],
        "mapping_qa": execution_bindings["mapping_qa"],
        "routes": execution_bindings["routes"],
        "bindings": dict(bindings),
        "bindings_sha256": canonical_sha256(bindings),
        "declaration": validated_declaration,
        "test_only_synthetic_contract": test_only_allow_synthetic_contract,
    }
    payload["self_sha256"] = canonical_sha256(payload)
    exclusive_json(destination, payload)
    verify_protocol_lock(root)
    transition_pipeline_status(
        root,
        RunState.P3_PROTOCOL_LOCKED,
        first_incomplete_stage=RunState.P4_G1_COUNTERFACTUAL_COMPLETE.value,
    )
    return destination


def verify_protocol_lock(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    lock = _read_object(root / LOCK_RELATIVE_PATH)
    unsigned = dict(lock)
    recorded = unsigned.pop("self_sha256", None)
    bindings = lock.get("bindings")
    declaration = lock.get("declaration")
    if (
        lock.get("status") != "LOCKED"
        or lock.get("counterfactual_execution_count") != 0
        or recorded != canonical_sha256(unsigned)
        or not isinstance(bindings, Mapping)
        or not isinstance(declaration, Mapping)
        or lock.get("bindings_sha256") != canonical_sha256(bindings)
    ):
        raise RuntimeError("counterfactual protocol lock contract differs")
    try:
        synthetic = lock.get("test_only_synthetic_contract") is True
        if synthetic and "PYTEST_CURRENT_TEST" not in __import__("os").environ:
            raise RuntimeError("synthetic protocol lock cannot be used outside pytest")
        validated_declaration = _validate_declaration(
            declaration, test_only_allow_synthetic_contract=synthetic
        )
        execution_bindings = _validate_bindings(
            bindings,
            declaration=validated_declaration,
            test_only_allow_synthetic_contract=synthetic,
        )
        if (
            lock.get("mapping_qa_gt_mask_rows_read_before_lock")
            != validated_declaration["mapping_qa_gt_mask_rows_read_before_lock"]
            or lock.get("candidate_generation_gt_mask_rows_read_before_lock") != 0
            or lock.get("gt_candidate_generation_authorized") is not True
            or lock.get("gt_mapping_pixel_qa_status") != "PASS"
            or lock.get("sample_manifest") != execution_bindings["sample_manifest"]
            or lock.get("gt_mask_registry") != execution_bindings["gt_mask_registry"]
            or lock.get("mapping_qa") != execution_bindings["mapping_qa"]
            or lock.get("routes") != execution_bindings["routes"]
        ):
            raise RuntimeError("execution-facing protocol fields differ")
    except (PermissionError, ValueError, RuntimeError) as error:
        raise RuntimeError("counterfactual protocol lock bindings differ") from error
    return lock


def load_execution_authority(protocol_lock_path: str | Path) -> dict[str, Any]:
    """Return one normalized, verified contract for every route executor."""

    path = Path(protocol_lock_path).expanduser().resolve()
    if (
        path.name != LOCK_RELATIVE_PATH.name
        or path.parent.name != LOCK_RELATIVE_PATH.parent.name
    ):
        raise ValueError("execution authority must be the canonical protocol lock")
    root = path.parent.parent
    lock = verify_protocol_lock(root)
    return {
        "protocol_lock": artifact_record(path),
        "gt_candidate_generation_authorized": True,
        "bulk_execution_max_count": 1,
        "mapping_qa_gt_mask_rows_read_before_lock": lock[
            "mapping_qa_gt_mask_rows_read_before_lock"
        ],
        "candidate_generation_gt_mask_rows_read_before_lock": 0,
        "sample_manifest": dict(lock["sample_manifest"]),
        "gt_mask_registry": dict(lock["gt_mask_registry"]),
        "mapping_qa": dict(lock["mapping_qa"]),
        "routes": dict(lock["routes"]),
    }


def claim_bulk_execution(run_dir: str | Path, *, resume: bool = False) -> Path:
    """Create the sole GT-mask bulk execution claim; never mutates source runs."""

    root = Path(run_dir).expanduser().resolve()
    lock = verify_protocol_lock(root)
    status = _read_object(root / "pipeline_status.json")
    allowed_resume_states = {
        RunState.P3_PROTOCOL_LOCKED.value,
        RunState.P4_G1_COUNTERFACTUAL_COMPLETE.value,
        RunState.P5_C1_COUNTERFACTUAL_COMPLETE.value,
        RunState.P6_D1_COUNTERFACTUAL_COMPLETE.value,
    }
    if status.get("status") != RunState.P3_PROTOCOL_LOCKED.value and not (
        resume and status.get("status") in allowed_resume_states
    ):
        raise PermissionError("bulk execution requires P3 or an exact resumed route stage")
    observed_count = int(status.get("counterfactual_execution_count", -1))
    if observed_count != 0 and not resume:
        raise PermissionError("counterfactual execution count is already non-zero")
    payload = {
        "schema_version": 1,
        "status": "RUNNING",
        "claimed_at_utc": _now(),
        "execution_count": 1,
        "protocol_lock_file_sha256": sha256_file(root / LOCK_RELATIVE_PATH),
        "protocol_lock_self_sha256": lock["self_sha256"],
    }
    destination = root / EXECUTION_RELATIVE_PATH
    if destination.exists():
        if not resume:
            raise FileExistsError("counterfactual bulk execution was already claimed")
        existing = _read_object(destination)
        if (
            existing.get("status") not in {"RUNNING", "COMPLETE"}
            or existing.get("execution_count") != 1
            or existing.get("protocol_lock_file_sha256")
            != payload["protocol_lock_file_sha256"]
            or existing.get("protocol_lock_self_sha256")
            != payload["protocol_lock_self_sha256"]
        ):
            raise RuntimeError("existing counterfactual execution claim differs")
    else:
        exclusive_json(destination, payload)
    status["counterfactual_execution_count"] = 1
    atomic_json(root / "pipeline_status.json", status)
    manifest_path = root / "manifest.json"
    manifest = _read_object(manifest_path)
    manifest["counterfactual_execution_count"] = 1
    atomic_json(manifest_path, manifest)
    return destination


def complete_bulk_execution(
    run_dir: str | Path, *, route_status_manifest: str | Path
) -> Path:
    """Finish the sole claim after a self-hashed route-status closure exists."""

    root = Path(run_dir).expanduser().resolve()
    claim_path = root / EXECUTION_RELATIVE_PATH
    claim = _read_object(claim_path)
    if claim.get("status") != "RUNNING" or claim.get("execution_count") != 1:
        raise RuntimeError("counterfactual claim is not uniquely RUNNING")
    route_path = Path(route_status_manifest).expanduser().resolve()
    if route_path != root / "08_metrics/ROUTE_STATUS.json":
        raise ValueError("execution completion requires canonical route status")
    route_record = artifact_record(route_path)
    route_status = _read_object(Path(route_record["path"]))
    unsigned = dict(route_status)
    recorded = unsigned.pop("content_sha256", None)
    routes = route_status.get("routes")
    if (
        recorded != canonical_sha256(unsigned)
        or not isinstance(routes, Mapping)
        or set(routes) != {"G1", "C1", "D1"}
        or routes.get("G1") != "COMPLETE"
        or routes.get("C1") != "COMPLETE"
        or routes.get("D1") not in {"COMPLETE", "UNRECOVERABLE_BLOCKER"}
    ):
        raise RuntimeError("route status cannot complete counterfactual execution")
    postprocess = route_status.get("artifacts", {}).get("postprocess_manifest")
    if not isinstance(postprocess, Mapping):
        raise RuntimeError("route status does not bind the postprocess manifest")
    verify_artifact_records_recursive(
        postprocess,
        name="execution completion postprocess manifest",
        require_at_least_one=True,
    )
    if route_status.get("protocol_lock") != artifact_record(
        root / LOCK_RELATIVE_PATH
    ):
        raise RuntimeError("route status protocol authority differs")
    destination = root / EXECUTION_COMPLETION_RELATIVE_PATH
    if destination.exists():
        existing = _read_object(destination)
        unsigned_existing = dict(existing)
        recorded_existing = unsigned_existing.pop("content_sha256", None)
        if (
            recorded_existing != canonical_sha256(unsigned_existing)
            or existing.get("status") != "COMPLETE"
            or existing.get("execution_count") != 1
            or existing.get("execution_claim") != artifact_record(claim_path)
            or existing.get("protocol_lock")
            != artifact_record(root / LOCK_RELATIVE_PATH)
            or existing.get("route_status") != route_record
            or existing.get("postprocess_manifest") != dict(postprocess)
        ):
            raise RuntimeError("existing execution completion differs")
        return destination
    completed = {
        "schema_version": 1,
        "status": "COMPLETE",
        "execution_count": 1,
        "completed_at_utc": _now(),
        "execution_claim": artifact_record(claim_path),
        "protocol_lock": artifact_record(root / LOCK_RELATIVE_PATH),
        "route_status": route_record,
        "postprocess_manifest": dict(postprocess),
    }
    completed["content_sha256"] = canonical_sha256(completed)
    exclusive_json(destination, completed)
    return destination


def protocol_lock_record(run_dir: str | Path) -> dict[str, Any]:
    verify_protocol_lock(run_dir)
    return artifact_record(Path(run_dir).expanduser().resolve() / LOCK_RELATIVE_PATH)
