"""Audited compatibility sidecar for the completed raw-feature execution.

The frozen nine-job feature plan bound the resource-gate CLI.  That CLI was
later extended with additional, unrelated P10/P12 resource scopes, while the
feature worker code and every scientific input stayed unchanged.  The old CLI
was untracked and its bytes are not recoverable, so this adapter does not claim
byte equivalence.  Instead it proves that only that authorization-tool record
drifted, reconstructs the exact frozen plan after substituting its frozen
record, and independently replays the already-persisted PASS gate evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import verified_artifact_path
from unified_reranking.hashing import atomic_json, canonical_sha256

from .execution import artifact_record, load_content_manifest
from .feature_plan import (
    FEATURE_PLAN_POINTER_RELATIVE,
    build_feature_extraction_plan,
)
from .resource_gate import evaluate_resource_gate


ADAPTER_RELATIVE = Path("configs/d1_feature_source_adapter.json")
AUDIT_TOOL_RELATIVE = Path("tools/d1_reranking/audit_resources.py")
EXECUTION_POINTER_RELATIVE = Path("configs/d1_feature_execution.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _load_plan(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    pointer_path = root / FEATURE_PLAN_POINTER_RELATIVE
    pointer = load_content_manifest(
        pointer_path,
        name="D1 feature-plan pointer",
        statuses=("PLANNED_POINTER",),
    )
    plan_path = verified_artifact_path(
        _mapping(pointer.get("active_plan"), name="D1 active feature plan"),
        name="D1 active feature plan",
    )
    plan = load_content_manifest(
        plan_path, name="D1 feature plan", statuses=("PLANNED",)
    )
    return pointer_path, plan_path, plan


def _frozen_audit_record(plan: Mapping[str, Any]) -> dict[str, Any]:
    sources = _mapping(plan.get("sources"), name="D1 feature-plan sources")
    code = sources.get("code")
    if not isinstance(code, list):
        raise RuntimeError("D1 feature-plan code inventory differs")
    audit_path = (_repo_root() / AUDIT_TOOL_RELATIVE).resolve()
    matches = [
        dict(record)
        for record in code
        if isinstance(record, Mapping)
        and Path(str(record.get("path", ""))).resolve() == audit_path
    ]
    if len(matches) != 1:
        raise RuntimeError("D1 frozen feature audit-tool record is not unique")
    return matches[0]


def _rebuild_plan(
    root: Path, plan: Mapping[str, Any], frozen_record: Mapping[str, Any]
) -> None:
    sources = _mapping(plan.get("sources"), name="D1 feature-plan sources")
    environment = _mapping(
        sources.get("environment"), name="D1 feature-plan environment"
    )
    executable = verified_artifact_path(
        _mapping(
            environment.get("python_executable"),
            name="D1 feature Python executable",
        ),
        name="D1 feature Python executable",
    )
    code = sources.get("code")
    if not isinstance(code, list):
        raise RuntimeError("D1 feature-plan code inventory differs")
    tool_paths: list[Path] = []
    audit_path = (_repo_root() / AUDIT_TOOL_RELATIVE).resolve()
    for index, raw_record in enumerate(code):
        record = _mapping(raw_record, name=f"D1 feature code record {index}")
        path = Path(str(record.get("path", ""))).resolve()
        if path != audit_path:
            verified_artifact_path(record, name=f"D1 feature code record {index}")
        tool_paths.append(path)
    expected = build_feature_extraction_plan(
        root, python_path=executable, tool_paths=tuple(tool_paths)
    )
    expected_code = expected["sources"]["code"]
    matches = [
        index
        for index, record in enumerate(expected_code)
        if Path(str(record.get("path", ""))).resolve() == audit_path
    ]
    if len(matches) != 1:
        raise RuntimeError("rebuilt D1 feature audit-tool record is not unique")
    expected_code[matches[0]] = dict(frozen_record)
    expected["source_signature_sha256"] = canonical_sha256(expected["sources"])
    unsigned = dict(expected)
    unsigned.pop("content_sha256", None)
    expected["content_sha256"] = canonical_sha256(unsigned)
    if dict(plan) != expected:
        raise RuntimeError(
            "D1 feature plan differs beyond the resource authorization tool record"
        )
    audit_module = "tools.d1_reranking.audit_resources"
    if any(
        audit_module in map(str, job.get("worker_argv", []))
        for job in plan.get("jobs", [])
        if isinstance(job, Mapping)
    ):
        raise RuntimeError("D1 resource audit tool appears in a feature worker command")


def _execution_and_gate(
    root: Path, plan_path: Path, frozen: Mapping[str, Any]
) -> dict[str, Any]:
    pointer_path = root / EXECUTION_POINTER_RELATIVE
    pointer = load_content_manifest(
        pointer_path, name="D1 feature execution pointer", statuses=("COMPLETE",)
    )
    execution_path = verified_artifact_path(
        _mapping(pointer.get("execution"), name="D1 feature execution"),
        name="D1 feature execution",
    )
    execution = load_content_manifest(
        execution_path, name="D1 feature execution", statuses=("ACTIVE",)
    )
    gate_path = verified_artifact_path(
        _mapping(execution.get("resource_gate"), name="D1 feature gate"),
        name="D1 feature gate",
    )
    gate = load_content_manifest(gate_path, name="D1 feature gate", statuses=("PASS",))
    policy_path = verified_artifact_path(
        _mapping(gate.get("policy"), name="D1 feature gate policy"),
        name="D1 feature gate policy",
    )
    policy = load_content_manifest(
        policy_path, name="D1 feature gate policy", statuses=("LOCKED_POLICY",)
    )
    gate_sources = _mapping(gate.get("sources"), name="D1 feature gate sources")
    if (
        execution.get("plan") != artifact_record(plan_path)
        or gate.get("plan") != artifact_record(plan_path)
        or policy.get("plan") != artifact_record(plan_path)
        or gate_sources.get("tool") != dict(frozen)
        or gate.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 feature authorization lineage differs")
    verified_artifact_path(
        _mapping(gate_sources.get("resource_gate"), name="D1 gate primitive"),
        name="D1 gate primitive",
    )
    windows = gate.get("windows")
    if not isinstance(windows, list):
        raise RuntimeError("D1 feature gate windows are absent")
    passed, reasons = evaluate_resource_gate(windows)
    if not passed or reasons or gate.get("failure_reasons") != []:
        raise RuntimeError("D1 feature gate semantic replay failed")
    return {
        "execution_pointer": artifact_record(pointer_path),
        "execution": artifact_record(execution_path),
        "resource_gate": artifact_record(gate_path),
        "resource_policy": artifact_record(policy_path),
        "resource_gate_primitive": dict(gate_sources["resource_gate"]),
    }


def _expected(root: Path) -> dict[str, Any]:
    pointer_path, plan_path, plan = _load_plan(root)
    frozen = _frozen_audit_record(plan)
    _rebuild_plan(root, plan, frozen)
    authorization = _execution_and_gate(root, plan_path, frozen)
    sources = {
        "plan_pointer": artifact_record(pointer_path),
        "plan": artifact_record(plan_path),
        "current_audit_tool": artifact_record(
            (_repo_root() / AUDIT_TOOL_RELATIVE).resolve()
        ),
        "adapter_module": artifact_record(Path(__file__).resolve()),
        "adapter_tool": artifact_record(
            (
                _repo_root() / "tools/d1_reranking/prepare_feature_source_adapter.py"
            ).resolve()
        ),
        **authorization,
    }
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "scientific_values_changed": False,
        "compatibility_scope": "postexecution_resource_authorizer_extension_only",
        "frozen_source_bytes_recoverable": False,
        "frozen_audit_tool": frozen,
        "numerical_worker_commands_reference_audit_tool": False,
        "plan_rebuild_after_frozen_record_substitution": "EXACT",
        "resource_gate_semantic_replay": "PASS",
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def prepare_feature_source_adapter(
    run_dir: str | Path, *, resume: bool
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    value = _expected(root)
    path = root / ADAPTER_RELATIVE
    if path.exists():
        existing = load_content_manifest(
            path, name="D1 feature source adapter", statuses=("COMPLETE",)
        )
        if resume and existing == value:
            return existing
        raise RuntimeError("D1 feature source adapter exists and differs")
    atomic_json(path, value)
    return value


def load_feature_plan_with_source_adapter(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    _pointer_path, plan_path, plan = _load_plan(root)
    adapter_path = root / ADAPTER_RELATIVE
    adapter = load_content_manifest(
        adapter_path, name="D1 feature source adapter", statuses=("COMPLETE",)
    )
    if adapter != _expected(root):
        raise RuntimeError("D1 feature source adapter contract differs")
    return plan_path, plan, artifact_record(adapter_path)


__all__ = [
    "ADAPTER_RELATIVE",
    "load_feature_plan_with_source_adapter",
    "prepare_feature_source_adapter",
]
