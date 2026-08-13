"""Adapter-aware wrapper around the frozen K Test/formal-input replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import k_formal_replay
from .execution import artifact_record
from .k_source_adapter import ADAPTER_RELATIVE, load_k_plan_with_source_adapter
from .primary_source_adapter import (
    load_primary_plan_with_source_adapter,
    verify_records_with_primary_source_adapter,
)


def validate_adapted_k_formal_replay(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    load_k_plan_with_source_adapter(root)
    _primary_path, _primary_plan, _primary_record, primary_adapter = (
        load_primary_plan_with_source_adapter(root)
    )
    original = k_formal_replay.verify_artifact_records_recursive
    original_artifact_record = k_formal_replay.artifact_record
    frozen = primary_adapter["frozen_source"]
    datasets_path = Path(str(frozen["path"])).resolve()

    def adapted_artifact_record(path: str | Path) -> dict[str, Any]:
        observed = Path(path).expanduser().resolve()
        if observed == datasets_path:
            return {"path": str(frozen["path"]), "sha256": str(frozen["sha256"])}
        return original_artifact_record(observed)

    def adapted_verifier(
        value: Any, *, name: str, require_at_least_one: bool = False
    ) -> list[dict[str, str]]:
        def contains_frozen(node: Any) -> bool:
            if isinstance(node, dict):
                return (
                    node.get("path") == frozen["path"]
                    and node.get("sha256") == frozen["sha256"]
                ) or any(contains_frozen(child) for child in node.values())
            if isinstance(node, list):
                return any(contains_frozen(child) for child in node)
            return False

        if contains_frozen(value):
            verify_records_with_primary_source_adapter(
                value, adapter=primary_adapter, name=name
            )
            return [
                {
                    "name": name,
                    "path": str(frozen["path"]),
                    "sha256": str(frozen["sha256"]),
                }
            ]
        return original(value, name=name, require_at_least_one=require_at_least_one)

    k_formal_replay.verify_artifact_records_recursive = adapted_verifier
    k_formal_replay.artifact_record = adapted_artifact_record
    try:
        records, checks = k_formal_replay.validate_k_formal_replay(root)
    finally:
        k_formal_replay.verify_artifact_records_recursive = original
        k_formal_replay.artifact_record = original_artifact_record
    records["source_adapter"] = artifact_record(root / ADAPTER_RELATIVE)
    return records, checks


__all__ = ["validate_adapted_k_formal_replay"]
