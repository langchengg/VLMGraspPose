"""Adapter-aware wrapper around the frozen raw-feature semantic replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .execution import artifact_record
from . import feature_replay
from .feature_source_adapter import (
    ADAPTER_RELATIVE,
    load_feature_plan_with_source_adapter,
)


def validate_adapted_feature_execution_replay(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    plan_path, plan, _adapter_record = load_feature_plan_with_source_adapter(root)
    original = feature_replay.load_active_feature_extraction_plan

    def adapted_loader(_run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
        observed = Path(_run_dir).expanduser().resolve()
        if observed != root:
            raise RuntimeError("D1 adapted feature replay run root differs")
        return plan_path, plan

    feature_replay.load_active_feature_extraction_plan = adapted_loader
    try:
        records, checks = feature_replay.validate_feature_execution_replay(root)
    finally:
        feature_replay.load_active_feature_extraction_plan = original
    records["source_adapter"] = artifact_record(root / ADAPTER_RELATIVE)
    return records, checks


__all__ = ["validate_adapted_feature_execution_replay"]
