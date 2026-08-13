"""Canonical P2-to-P3 protocol input assembler.

This module deliberately exposes no caller-selectable scientific parameter.
It resolves the fixed source/code/config/semantics from the repository and the
canonical P1/P2 artifacts in the isolated counterfactual run.
"""

from __future__ import annotations

import json
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
from .io import artifact_record, atomic_json, canonical_sha256
from .protocol import inline_binding


BINDINGS_RELATIVE_PATH = Path("configs/COUNTERFACTUAL_PROTOCOL_BINDINGS.json")
DECLARATION_RELATIVE_PATH = Path("configs/COUNTERFACTUAL_PROTOCOL_DECLARATION.json")


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

    adapter_manifest = build_g1_c1_source_adapter(
        run_dir=root,
        source_run=FROZEN_G1_C1_SOURCE,
        registry_path=registry_path,
        mapping_audit_path=mapping_path,
        resume=resume,
    )
    routes = canonical_route_contracts(adapter_manifest)
    semantics = canonical_semantic_contracts()
    bindings = _self_hashed(
        {
            "schema_version": 1,
            "source_locks": {"verification": dict(source_verification)},
            "source_code": canonical_source_code_inventory(),
            "configs": canonical_config_inventory(),
            "baseline_replay": artifact_record(baseline_path),
            "sample_manifest": artifact_record(sample_path),
            "gt_mask_registry": artifact_record(registry_path),
            "mapping_qa": artifact_record(mapping_path),
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
            "gt_candidate_generation_authorized": True,
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
    "assemble_protocol_inputs",
]
