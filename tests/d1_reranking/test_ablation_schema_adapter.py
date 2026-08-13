from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from d1_reranking import ablation_replay, ablation_selection
from d1_reranking.ablation_schema_adapter import (
    project_primary_gate_metrics,
    projected_primary_gate_loaders,
)
from d1_reranking.execution import load_content_manifest
from unified_reranking.hashing import atomic_json, canonical_sha256


def _manifest(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return path


def test_primary_gate_projection_is_exact_and_non_mutating() -> None:
    source = {
        "status": "COMPLETE",
        "validation_metrics": {"gated_j_at_1": 0.7, "sample_count": 12},
    }
    original = deepcopy(source)
    projected = project_primary_gate_metrics(source)

    assert source == original
    assert projected["metrics"] == source["validation_metrics"]
    assert projected["metrics"] is not source["validation_metrics"]

    disagreeing = {**source, "metrics": {"gated_j_at_1": 0.1, "sample_count": 12}}
    with pytest.raises(RuntimeError, match="aliases disagree"):
        project_primary_gate_metrics(disagreeing)


def test_projection_patches_only_exact_gate_and_restores_loaders(tmp_path: Path) -> None:
    gate_path = _manifest(
        tmp_path / "gate.json",
        {
            "status": "COMPLETE",
            "validation_metrics": {"gated_j_at_1": 0.7, "sample_count": 12},
        },
    )
    other_path = _manifest(
        tmp_path / "other.json",
        {
            "status": "COMPLETE",
            "validation_metrics": {"gated_j_at_1": 0.2, "sample_count": 4},
        },
    )
    original_selection_loader = ablation_selection.load_content_manifest
    original_replay_loader = ablation_replay.load_content_manifest

    with projected_primary_gate_loaders(gate_path):
        assert ablation_selection.load_content_manifest(
            gate_path, name="gate", statuses=("COMPLETE",)
        )["metrics"]["gated_j_at_1"] == 0.7
        assert ablation_replay.load_content_manifest(
            gate_path, name="gate", statuses=("COMPLETE",)
        )["metrics"]["sample_count"] == 12
        assert "metrics" not in ablation_selection.load_content_manifest(
            other_path, name="other", statuses=("COMPLETE",)
        )

    assert ablation_selection.load_content_manifest is original_selection_loader
    assert ablation_replay.load_content_manifest is original_replay_loader
    assert "metrics" not in load_content_manifest(
        gate_path, name="gate", statuses=("COMPLETE",)
    )
