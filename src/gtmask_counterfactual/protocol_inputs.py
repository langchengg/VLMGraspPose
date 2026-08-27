"""Canonical P2-to-P3 protocol input assembler.

This module deliberately exposes no caller-selectable scientific parameter.
It resolves the fixed source/code/config/semantics from the repository and the
canonical P1/P2 artifacts in the isolated counterfactual run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .contracts import RunState
from .execution_contracts import (
    CANONICAL_EVALUATOR,
    FROZEN_G1_C1_SOURCE,
    canonical_config_inventory,
    canonical_route_contracts,
    canonical_semantic_contracts,
    canonical_source_code_inventory,
)
from .g1_c1_adapter import build_g1_c1_source_adapter
from .io import artifact_record, atomic_json, atomic_text, canonical_sha256
from .pilot import build_c1_pilot_source_adapter
from .protocol import inline_binding


BINDINGS_RELATIVE_PATH = Path("configs/COUNTERFACTUAL_PROTOCOL_BINDINGS.json")
DECLARATION_RELATIVE_PATH = Path("configs/COUNTERFACTUAL_PROTOCOL_DECLARATION.json")
PROTOCOL_MARKDOWN_RELATIVE_PATH = Path("01_protocol_lock/COUNTERFACTUAL_PROTOCOL.md")


def _object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain one JSON object")
    return value


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _publish_exact(path: Path, value: Mapping[str, Any], *, resume: bool) -> Path:
    expected = dict(value)
    if path.exists():
        if not resume:
            raise FileExistsError(f"protocol input exists; pass --resume: {path}")
        observed = _object(path, name=path.name)
        if observed != expected:
            raise RuntimeError(f"existing protocol input differs: {path}")
        return path
    return atomic_json(path, expected)


def _retrospective_reconstruction(
    root: Path, routes: Mapping[str, Mapping[str, Any]], *, resume: bool
) -> tuple[Path, Path]:
    """Publish the honest, pre-lock reconstruction required for Case A."""

    records: dict[str, dict[str, Any]] = {}
    required = {
        "retrospective_source_lock",
        "retrospective_finalization",
        "retrospective_result_hashes",
        "retrospective_canonical_candidates",
        "retrospective_native_inference",
        "retrospective_native_output",
    }
    missing = {
        f"{route}.{name}"
        for route in ("g1", "c1")
        for name in required.difference(routes[route])
    }
    if missing and "PYTEST_CURRENT_TEST" not in os.environ:
        raise RuntimeError(
            f"production retrospective reconstruction lacks bindings: {sorted(missing)}"
        )
    for route in ("g1", "c1"):
        contract = routes[route]
        for name in (
            "retrospective_source_lock",
            "retrospective_finalization",
            "retrospective_result_hashes",
            "retrospective_canonical_candidates",
            "retrospective_native_inference",
        ):
            if name not in contract:
                continue
            record = dict(contract[name])
            path = Path(str(record["path"]))
            record["mtime_utc"] = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            records.setdefault(name, record)
        for name, value in contract.get("retrospective_native_output", {}).items():
            record = dict(value)
            path = Path(str(record["path"]))
            record["mtime_utc"] = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            records[f"{route}_native_{name}"] = record
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "protocol_kind": (
            "synthetic_test_fixture_reconstruction"
            if missing
            else "retrospective_protocol_reconstruction"
        ),
        "created_before_current_protocol_lock": True,
        "historical_source_execution_count": 1,
        "current_run_model_inference_count": 0,
        "formal_test_execution_count_change": 0,
        "artifacts_existing_before_current_audit": records,
        "test_outcomes_exposed_before_current_binding": [
            "G1/C1 GT-mask candidate memberships and native ranks",
            "candidate-level same-GT labels in the finalized canonical table",
            "aggregate native J@1, Oracle@5, and Oracle@All results",
            "the frozen decoder, checkpoint, configuration, and evaluator choices",
        ],
        "verification_only_components": [
            "source provenance and byte-hash closure",
            "raw-to-formally-locked canonical candidate equality",
            "independent frozen-evaluator metric recomputation",
            "source immutability before/after comparison",
        ],
        "retrospective_analyses": [
            "Top-(K) sensitivity",
            "T0-T7 native and R0-R5 post-R7 taxonomy",
            "paired McNemar and clustered-bootstrap statistics",
            "stratified and qualitative case analysis",
        ],
        "scientific_interpretation": (
            "No outcome analysis in this run is presented as prospectively "
            "preregistered. The lock freezes a reproducible re-analysis of "
            "already-existing Test outcomes."
        ),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    output = root / "01_protocol_lock"
    json_path = output / "RETROSPECTIVE_PROTOCOL_RECONSTRUCTION.json"
    md_path = output / "RETROSPECTIVE_PROTOCOL_RECONSTRUCTION.md"
    markdown = "\n".join(
        [
            "# Retrospective Protocol Reconstruction",
            "",
            "This is a Case-A reconstruction, not a backdated preregistration. "
            "The G1/C1 GT-mask outcomes and aggregate Test results existed before "
            "this audit and before the current protocol lock.",
            "",
            "## Pre-existing artifacts",
            "",
            *[
                f"- `{name}`: `{record['path']}`; SHA-256 "
                f"`{record['sha256']}`; mtime `{record['mtime_utc']}`."
                for name, record in sorted(records.items())
            ],
            "",
            "## Decisions and outcomes already exposed",
            "",
            *[
                f"- {value}."
                for value in payload["test_outcomes_exposed_before_current_binding"]
            ],
            "",
            "## Classification of subsequent work",
            "",
            "Byte closure, raw-to-canonical equality, frozen-evaluator recomputation, "
            "and source-immutability checks are verification-only components. "
            "Top-(K), taxonomy, paired statistics, stratification, and case analysis "
            "are retrospective analyses of already-exposed Test outcomes.",
            "",
            "Current-run G1/C1 model inference count is **0**; the historical "
            "source execution count is **1**; the Formal Test execution-count "
            "change is **0**.",
            "",
        ]
    )
    if json_path.exists() and not resume:
        raise FileExistsError(f"retrospective reconstruction exists: {json_path}")
    if md_path.exists() and not resume:
        raise FileExistsError(f"retrospective reconstruction exists: {md_path}")
    atomic_json(json_path, payload)
    atomic_text(md_path, markdown)
    return json_path, md_path


def _protocol_markdown(
    root: Path,
    routes: Mapping[str, Mapping[str, Any]],
    semantics: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    """Publish the human-readable protocol before the machine lock is created."""

    path = root / PROTOCOL_MARKDOWN_RELATIVE_PATH
    lines = [
        "# GT-mask stage-replacement counterfactual protocol",
        "",
        "Status: pre-lock protocol declaration for a retrospective Case-A G1/C1 "
        "re-analysis and a post-core prospective D1 Case-B extension.",
        "",
        "## Intervention",
        "",
        "Only the target-mask support changes from the deployed predicted mask to "
        "the registered ground-truth target-instance mask. Checkpoints, decoder "
        "parameters, candidate budgets, native order, frozen R7 outcomes, and the "
        "offline evaluator remain fixed.",
        "",
        "## Route order and authority",
        "",
        "1. Deterministic C1 200-sample audit pilot.",
        "2. Full C1 retrospective verified import.",
        "3. Full G1 retrospective verified import.",
        "4. Top-(K), taxonomy, paired statistics, independent recompute, figures, "
        "gallery, and core reports.",
        "5. Only after core acceptance, attempt D1 Case B with raw candidate "
        "regeneration; a filter-only diagnostic cannot substitute for the primary.",
        "",
        "G1/C1 model inference in the current run is forbidden because complete "
        "compatible historical outputs already exist. Their outcomes were exposed "
        "before this protocol; all subsequent G1/C1 analyses are retrospective.",
        "",
        "## Frozen route contracts",
        "",
    ]
    for route in ("c1", "g1", "d1"):
        contract = routes[route]
        lines.append(
            f"- {route.upper()}: execution mode `{contract.get('execution_mode', 'prospective')}`; "
            f"candidate budget `{contract.get('candidate_budget', contract.get('num_candidates', 'locked'))}`."
        )
    evaluator = semantics["evaluator_contract"]
    lines.extend(
        [
            "",
            "## Evaluator and inference",
            "",
            f"- Same-GT conjunction; rotated IoU `{evaluator['iou_comparator']}` "
            f"`{evaluator['iou_threshold']}` and periodic angle "
            f"`{evaluator['angle_comparator']}` `{evaluator['angle_threshold_degrees']}` degrees.",
            "- No-output samples remain in the denominator.",
            "- Exact McNemar tests use paired sample outcomes; confidence intervals "
            "use scene-clustered bootstrap with the locked seed and iteration count; "
            "Holm correction covers the preregistered family.",
            "",
            "## Claim boundary",
            "",
            "The GT mask is unavailable at deployment. Results diagnose offline "
            "bottlenecks under the frozen 4-DoF same-GT criterion; they are not "
            "physical grasp success rates or deployable performance gains.",
            "",
        ]
    )
    content = "\n".join(lines)
    if path.exists():
        if not resume or path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing protocol markdown differs: {path}")
        return path
    return atomic_text(path, content)


def assemble_protocol_inputs(
    run_dir: str | Path, *, resume: bool = False
) -> tuple[Path, Path]:
    """Write the unique production bindings/declaration after a PASS P2 audit."""

    root = Path(run_dir).expanduser().resolve()
    pipeline = _object(root / "pipeline_status.json", name="pipeline status")
    if pipeline.get("status") != RunState.P2_GT_MAPPING_PASS.value:
        raise PermissionError("protocol input assembly requires P2_GT_MAPPING_PASS")
    if int(pipeline.get("counterfactual_execution_count", -1)) != 0:
        raise PermissionError("protocol input assembly requires execution count zero")
    source_verification = pipeline.get("source_lock_verification")
    if not isinstance(source_verification, Mapping):
        raise RuntimeError("pipeline lacks source-lock verification authority")
    sample_path = root / "02_sample_manifest/counterfactual_manifest.parquet"
    registry_path = root / "03_gt_mask_registry/gt_mask_registry.parquet"
    mapping_path = root / "03_gt_mask_registry/GT_MASK_MAPPING_AUDIT.json"
    baseline_path = root / "04_predicted_replay/BASELINE_REPLAY_MANIFEST.json"
    mapping = _object(mapping_path, name="GT mask mapping audit")
    unsigned_mapping = dict(mapping)
    mapping_hash = unsigned_mapping.pop("content_sha256", None)
    if (
        mapping_hash != canonical_sha256(unsigned_mapping)
        or mapping.get("status") != "PASS"
        or mapping.get("stage") != RunState.P2_GT_MAPPING_PASS.value
        or mapping.get("sample_count") != 7_675
    ):
        raise RuntimeError("canonical P2 mapping audit is not a full PASS")
    mapping_inputs = mapping.get("inputs")
    join_record = (
        mapping_inputs.get("join_audit") if isinstance(mapping_inputs, Mapping) else None
    )
    if not isinstance(join_record, Mapping):
        raise RuntimeError("canonical P2 audit does not bind the P1 join audit")
    join_audit = _object(Path(str(join_record["path"])), name="P1 join audit")
    join_unsigned = dict(join_audit)
    if join_unsigned.pop("content_sha256", None) != canonical_sha256(join_unsigned):
        raise RuntimeError("P1 join audit content hash differs")
    join_sources = join_audit.get("sources")
    gt_grasp_source = (
        join_sources.get("denominator") if isinstance(join_sources, Mapping) else None
    )
    if not isinstance(gt_grasp_source, Mapping):
        raise RuntimeError("P1 join audit lacks its frozen denominator source")
    # Reconstructing the record also rejects a post-P1 byte replacement.
    if artifact_record(Path(str(gt_grasp_source["path"]))) != dict(gt_grasp_source):
        raise RuntimeError("P1 frozen denominator source differs")

    adapter_manifest = build_g1_c1_source_adapter(
        run_dir=root,
        source_run=FROZEN_G1_C1_SOURCE,
        registry_path=registry_path,
        mapping_audit_path=mapping_path,
        resume=resume,
    )
    pilot_manifest = build_c1_pilot_source_adapter(
        run_dir=root,
        full_adapter_manifest=adapter_manifest,
        sample_manifest_path=sample_path,
        registry_path=registry_path,
        baseline_manifest_path=baseline_path,
        resume=resume,
    )
    routes = canonical_route_contracts(adapter_manifest, pilot_manifest)
    reconstruction_json, reconstruction_md = _retrospective_reconstruction(
        root, routes, resume=resume
    )
    semantics = canonical_semantic_contracts()
    protocol_markdown = _protocol_markdown(
        root, routes, semantics, resume=resume
    )
    bindings = _self_hashed(
        {
            "schema_version": 1,
            "source_locks": {"verification": dict(source_verification)},
            "source_code": canonical_source_code_inventory(),
            "configs": canonical_config_inventory(),
            "baseline_replay": artifact_record(baseline_path),
            "sample_manifest": artifact_record(sample_path),
            "gt_grasp_source": dict(gt_grasp_source),
            "gt_mask_registry": artifact_record(registry_path),
            "mapping_qa": artifact_record(mapping_path),
            "retrospective_protocol_reconstruction": {
                "json": artifact_record(reconstruction_json),
                "markdown": artifact_record(reconstruction_md),
                "protocol_markdown": artifact_record(protocol_markdown),
            },
            "c1_pilot_source_adapter": artifact_record(pilot_manifest),
            "route_contracts": inline_binding(routes),
            "resize_rules": inline_binding(semantics["resize_rules"]),
            "evaluator": {
                "implementation": artifact_record(CANONICAL_EVALUATOR),
                "contract": inline_binding(semantics["evaluator_contract"]),
            },
            "taxonomy": inline_binding(semantics["taxonomy"]),
            "statistics": inline_binding(semantics["statistics"]),
            "case_selection": inline_binding(semantics["case_selection"]),
        }
    )
    declaration = _self_hashed(
        {
            "schema_version": 1,
            "branch": "gt_oracle",
            "execution_mode": "retrospective_verified_import",
            "retrospective_test_outcomes_exposed_before_binding": True,
            "gt_candidate_generation_authorized": False,
            "d1_candidate_generation_authorized": True,
            "bulk_execution_max_count": 1,
            "mapping_qa_gt_mask_rows_read_before_lock": int(
                mapping["mapping_qa_gt_mask_rows_read"]
            ),
            "candidate_generation_gt_mask_rows_read_before_lock": 0,
            "routes": routes,
        }
    )
    bindings_path = _publish_exact(
        root / BINDINGS_RELATIVE_PATH, bindings, resume=resume
    )
    declaration_path = _publish_exact(
        root / DECLARATION_RELATIVE_PATH, declaration, resume=resume
    )
    return bindings_path, declaration_path


__all__ = [
    "BINDINGS_RELATIVE_PATH",
    "DECLARATION_RELATIVE_PATH",
    "PROTOCOL_MARKDOWN_RELATIVE_PATH",
    "assemble_protocol_inputs",
]
