"""Adapter-aware wrapper around the frozen K semantic replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .execution import artifact_record
from . import k_replay, k_sensitivity
from .k_source_adapter import ADAPTER_RELATIVE, load_k_plan_with_source_adapter
from .primary_source_adapter import (
    load_primary_plan_with_source_adapter,
    verify_records_with_primary_source_adapter,
)


def validate_adapted_k_sensitivity_replay(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    plan_path, plan, _adapter_record = load_k_plan_with_source_adapter(root)
    _primary_path, _primary_plan, _primary_record, primary_adapter = (
        load_primary_plan_with_source_adapter(root)
    )
    frozen = primary_adapter["frozen_source"]
    datasets_path = Path(str(frozen["path"])).resolve()
    original_replay_loader = k_replay.load_k_sensitivity_plan
    original_plan_loader = k_sensitivity.load_k_sensitivity_plan
    original_artifact_record = k_sensitivity.artifact_record
    original_recursive_verifier = k_sensitivity.verify_artifact_records_recursive

    def adapted_loader(path: str | Path) -> dict[str, Any]:
        observed = Path(path).expanduser().resolve()
        if observed != plan_path:
            raise RuntimeError("D1 adapted K replay plan path differs")
        return plan

    def adapted_artifact_record(path: str | Path) -> dict[str, Any]:
        observed = Path(path).expanduser().resolve()
        if observed == datasets_path:
            return {"path": str(frozen["path"]), "sha256": str(frozen["sha256"])}
        return original_artifact_record(observed)

    def adapted_recursive_verifier(
        value: Any, *, name: str, require_at_least_one: bool = False
    ) -> list[dict[str, str]]:
        def contains_frozen(node: Any) -> bool:
            if isinstance(node, dict):
                if Path(str(node.get("path", ""))).resolve() == datasets_path:
                    return node.get("sha256") == frozen["sha256"]
                return any(contains_frozen(child) for child in node.values())
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
                    "path": str(datasets_path),
                    "sha256": str(frozen["sha256"]),
                }
            ]
        return original_recursive_verifier(
            value, name=name, require_at_least_one=require_at_least_one
        )

    k_replay.load_k_sensitivity_plan = adapted_loader
    k_sensitivity.load_k_sensitivity_plan = adapted_loader
    k_sensitivity.artifact_record = adapted_artifact_record
    k_sensitivity.verify_artifact_records_recursive = adapted_recursive_verifier
    try:
        records, checks = k_replay.validate_k_sensitivity_replay(root)
    finally:
        k_replay.load_k_sensitivity_plan = original_replay_loader
        k_sensitivity.load_k_sensitivity_plan = original_plan_loader
        k_sensitivity.artifact_record = original_artifact_record
        k_sensitivity.verify_artifact_records_recursive = original_recursive_verifier
    records["source_adapter"] = artifact_record(root / ADAPTER_RELATIVE)
    return records, checks


__all__ = ["validate_adapted_k_sensitivity_replay"]
