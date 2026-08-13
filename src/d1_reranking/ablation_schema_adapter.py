"""Audited post-execution adapter for the frozen P10 gate-metric field name.

The completed P10 plan binds selector/replay code that reads ``metrics`` from
the primary gate.  The canonical primary-gate producer persisted the same
payload under ``validation_metrics``.  This module preserves every plan-bound
byte and exposes a narrowly scoped, in-memory compatibility view.  A sidecar
binds the adapter, exact projection, immutable execution, gate, and selection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.hashing import atomic_json, canonical_sha256

from . import ablation_replay, ablation_selection
from .ablation import ABLATION_PLAN_RELATIVE
from .ablation_replay import (
    validate_ablation_execution_results,
    validate_ablation_replay,
    validate_ablation_selection,
)
from .ablation_selection import SELECTION_RELATIVE, write_ablation_selection
from .execution import artifact_record, load_content_manifest


ADAPTER_SCHEMA_VERSION = 1
ADAPTER_RELATIVE = Path(
    "12_feature_ablation/ablation_selection_schema_adapter.json"
)
SCHEMA_PROJECTION = {
    "source_manifest": "07_validation/gate/d1/gate_selection.json",
    "source_field": "validation_metrics",
    "compatibility_field": "metrics",
    "required_metric_fields": ["gated_j_at_1", "sample_count"],
    "mutation_scope": "in_memory_only",
    "scientific_values_changed": False,
}


def project_primary_gate_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact legacy field view or fail on an ambiguous gate."""

    gate = {str(key): child for key, child in value.items()}
    validation_metrics = gate.get("validation_metrics")
    if not isinstance(validation_metrics, Mapping):
        raise RuntimeError("D1 primary gate validation_metrics is missing")
    metrics = {str(key): child for key, child in validation_metrics.items()}
    missing = set(SCHEMA_PROJECTION["required_metric_fields"]).difference(metrics)
    if missing:
        raise RuntimeError(
            f"D1 primary gate validation_metrics misses {sorted(missing)}"
        )
    legacy = gate.get("metrics")
    if legacy is not None and legacy != validation_metrics:
        raise RuntimeError("D1 primary gate metrics aliases disagree")
    gate["metrics"] = dict(metrics)
    return gate


@contextmanager
def projected_primary_gate_loaders(
    primary_gate_path: str | Path,
) -> Iterator[None]:
    """Patch only the two frozen modules and only for the exact bound gate."""

    gate_path = Path(primary_gate_path).expanduser().resolve()
    original_selection_loader = ablation_selection.load_content_manifest
    original_replay_loader = ablation_replay.load_content_manifest

    def wrapper(original: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        def load(path: str | Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
            value = original(path, *args, **kwargs)
            if Path(path).expanduser().resolve() == gate_path:
                return project_primary_gate_metrics(value)
            return value

        return load

    ablation_selection.load_content_manifest = wrapper(original_selection_loader)
    ablation_replay.load_content_manifest = wrapper(original_replay_loader)
    try:
        yield
    finally:
        ablation_selection.load_content_manifest = original_selection_loader
        ablation_replay.load_content_manifest = original_replay_loader


def _adapter_tool() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "tools/d1_reranking/select_ablation_with_schema_adapter.py"
    )


def _adapter_sources(
    root: Path,
    *,
    execution_records: Mapping[str, Any],
) -> dict[str, Any]:
    plan_path = root / ABLATION_PLAN_RELATIVE
    plan = load_content_manifest(
        plan_path, name="D1 P10 ablation plan", statuses=("PLANNED",)
    )
    return {
        "plan": artifact_record(plan_path),
        "execution": dict(execution_records),
        "primary_gate": dict(plan["sources"]["primary_gate"]),
        "selection": artifact_record(root / SELECTION_RELATIVE),
        "adapter_module": artifact_record(Path(__file__).resolve()),
        "adapter_tool": artifact_record(_adapter_tool()),
    }


def _write_or_verify_adapter(
    root: Path,
    *,
    execution_records: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    selection = load_content_manifest(
        root / SELECTION_RELATIVE,
        name="D1 P10 adapted selection",
        statuses=("COMPLETE",),
    )
    sources = _adapter_sources(root, execution_records=execution_records)
    artifacts = {
        "selection": artifact_record(root / SELECTION_RELATIVE),
        "evidence_track_table": dict(
            selection["artifacts"]["evidence_track_table"]
        ),
        "feature_ablation_table": dict(
            selection["artifacts"]["feature_ablation_table"]
        ),
    }
    value: dict[str, Any] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "scientific_values_changed": False,
        "projection": dict(SCHEMA_PROJECTION),
        "source_signature_sha256": canonical_sha256(sources),
        "sources": sources,
        "artifacts": artifacts,
    }
    value["content_sha256"] = canonical_sha256(value)
    path = root / ADAPTER_RELATIVE
    if path.exists():
        existing = load_content_manifest(
            path, name="D1 P10 schema adapter", statuses=("COMPLETE",)
        )
        if resume and existing == value:
            return existing
        raise RuntimeError("D1 P10 schema-adapter sidecar exists and differs")
    atomic_json(path, value)
    return value


def run_adapted_ablation_selection(
    run_dir: str | Path, *, resume: bool
) -> dict[str, Any]:
    """Publish the frozen selection through the explicit schema projection."""

    root = Path(run_dir).expanduser().resolve()
    execution_records, execution_values = validate_ablation_execution_results(root)
    primary_gate_path = execution_values["plan"]["sources"]["primary_gate"]["path"]
    with projected_primary_gate_loaders(primary_gate_path):
        selection = write_ablation_selection(
            root,
            execution_records=execution_records,
            result_values=execution_values["results"],
            resume=resume,
        )
        validate_ablation_selection(
            root,
            execution_records=execution_records,
            execution_values=execution_values,
        )
    _write_or_verify_adapter(
        root, execution_records=execution_records, resume=resume
    )
    return selection


def validate_adapted_ablation_replay(run_dir: str | Path) -> dict[str, Any]:
    """P13-facing replay that proves the adapter and unchanged legacy replay."""

    root = Path(run_dir).expanduser().resolve()
    execution_records, execution_values = validate_ablation_execution_results(root)
    sidecar_path = root / ADAPTER_RELATIVE
    sidecar = load_content_manifest(
        sidecar_path, name="D1 P10 schema adapter", statuses=("COMPLETE",)
    )
    expected_sources = _adapter_sources(root, execution_records=execution_records)
    if (
        sidecar.get("schema_version") != ADAPTER_SCHEMA_VERSION
        or sidecar.get("projection") != SCHEMA_PROJECTION
        or sidecar.get("scientific_values_changed") is not False
        or sidecar.get("candidate_test_labels_read") is not False
        or sidecar.get("test_inputs_referenced") is not False
        or sidecar.get("sources") != expected_sources
        or sidecar.get("source_signature_sha256")
        != canonical_sha256(expected_sources)
    ):
        raise RuntimeError("D1 P10 schema-adapter contract differs")
    verify_artifact_records_recursive(
        {"sources": sidecar["sources"], "artifacts": sidecar["artifacts"]},
        name="D1 P10 schema-adapter closure",
        require_at_least_one=True,
    )
    primary_gate_path = execution_values["plan"]["sources"]["primary_gate"]["path"]
    with projected_primary_gate_loaders(primary_gate_path):
        replay = validate_ablation_replay(root)
    return {
        **replay,
        "schema_version": 2,
        "checks": {
            **replay["checks"],
            "primary_gate_metric_schema_projection_exact": True,
            "schema_projection_scientific_values_changed": False,
        },
        "sources": {
            **replay["sources"],
            "schema_adapter": artifact_record(sidecar_path),
            "schema_adapter_sources": sidecar["sources"],
        },
    }


__all__ = [
    "ADAPTER_RELATIVE",
    "SCHEMA_PROJECTION",
    "project_primary_gate_metrics",
    "projected_primary_gate_loaders",
    "run_adapted_ablation_selection",
    "validate_adapted_ablation_replay",
]
