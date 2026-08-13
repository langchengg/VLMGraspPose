"""Audited native-rank dtype source adapter for the completed K plan."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import verified_artifact_path
from unified_reranking.hashing import atomic_json, canonical_sha256

from .execution import artifact_record, load_content_manifest
from .k_sensitivity import build_k_sensitivity_plan, validate_k_sensitivity_plan
from .primary_source_adapter import (
    load_primary_plan_with_source_adapter,
    verify_records_with_primary_source_adapter,
)


ADAPTER_RELATIVE = Path("configs/d1_k_source_adapter.json")
PLAN_RELATIVE = Path("configs/d1_k_sensitivity_plan.json")
DATASETS_RELATIVE = Path("src/unified_reranking/datasets.py")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _adapted_plan(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    plan_path = root / PLAN_RELATIVE
    plan = validate_k_sensitivity_plan(
        load_content_manifest(
            plan_path, name="D1 K-sensitivity plan", statuses=("PLANNED",)
        )
    )
    _primary_plan_path, _primary_plan, primary_record, primary_adapter = (
        load_primary_plan_with_source_adapter(root)
    )
    adapted_count = verify_records_with_primary_source_adapter(
        _mapping(plan.get("sources"), name="D1 K-sensitivity sources"),
        adapter=primary_adapter,
        name="D1 K-sensitivity sources",
    )
    if adapted_count != 1:
        raise RuntimeError("D1 K plan must contain one frozen datasets.py record")
    sources = _mapping(plan.get("sources"), name="D1 K-sensitivity sources")
    code = sources.get("code")
    if not isinstance(code, list):
        raise RuntimeError("D1 K code inventory differs")
    datasets_path = (_repo_root() / DATASETS_RELATIVE).resolve()
    tool_paths: list[Path] = []
    frozen_record: dict[str, Any] | None = None
    for index, raw_record in enumerate(code):
        record = _mapping(raw_record, name=f"D1 K code record {index}")
        path = Path(str(record.get("path", ""))).resolve()
        if path == datasets_path:
            if frozen_record is not None:
                raise RuntimeError("D1 K frozen datasets.py record is not unique")
            frozen_record = record
        else:
            verified_artifact_path(record, name=f"D1 K code record {index}")
        tool_paths.append(path)
    if frozen_record is None:
        raise RuntimeError("D1 K frozen datasets.py record is absent")
    expected = build_k_sensitivity_plan(root, tool_paths=tuple(tool_paths))
    expected_code = expected["sources"]["code"]
    matches = [
        index
        for index, record in enumerate(expected_code)
        if Path(str(record.get("path", ""))).resolve() == datasets_path
    ]
    if len(matches) != 1:
        raise RuntimeError("rebuilt D1 K datasets.py record is not unique")
    expected_code[matches[0]] = frozen_record
    expected["source_signature_sha256"] = canonical_sha256(expected["sources"])
    unsigned = dict(expected)
    unsigned.pop("content_sha256", None)
    expected["content_sha256"] = canonical_sha256(unsigned)
    if plan != expected:
        raise RuntimeError(
            "D1 K plan differs beyond the native-rank dtype source patch"
        )
    return plan_path, plan, primary_record


def _expected(root: Path) -> dict[str, Any]:
    plan_path, _plan, primary_record = _adapted_plan(root)
    sources = {
        "k_plan": artifact_record(plan_path),
        "primary_source_adapter": primary_record,
        "adapter_module": artifact_record(Path(__file__).resolve()),
        "adapter_tool": artifact_record(
            (_repo_root() / "tools/d1_reranking/prepare_k_source_adapter.py").resolve()
        ),
    }
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "scientific_values_changed": False,
        "compatibility_scope": "native_rank_numeric_dtype_equality_only",
        "adapted_source_record_count": 1,
        "plan_rebuild_after_frozen_record_substitution": "EXACT",
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def prepare_k_source_adapter(run_dir: str | Path, *, resume: bool) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    value = _expected(root)
    path = root / ADAPTER_RELATIVE
    if path.exists():
        existing = load_content_manifest(
            path, name="D1 K source adapter", statuses=("COMPLETE",)
        )
        if resume and existing == value:
            return existing
        raise RuntimeError("D1 K source adapter exists and differs")
    atomic_json(path, value)
    return value


def load_k_plan_with_source_adapter(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    plan_path, plan, _primary_record = _adapted_plan(root)
    adapter_path = root / ADAPTER_RELATIVE
    adapter = load_content_manifest(
        adapter_path, name="D1 K source adapter", statuses=("COMPLETE",)
    )
    if adapter != _expected(root):
        raise RuntimeError("D1 K source adapter contract differs")
    return plan_path, plan, artifact_record(adapter_path)


__all__ = [
    "ADAPTER_RELATIVE",
    "load_k_plan_with_source_adapter",
    "prepare_k_source_adapter",
]
