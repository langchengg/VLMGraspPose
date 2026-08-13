from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from d1_reranking.command_dag import (
    PROMPT_ARTIFACT_GAPS,
    audit_catalog,
    command_nodes,
    render_markdown,
    topological_order,
    validate_catalog,
)


ROOT = Path(__file__).resolve().parents[2]


def test_command_dag_is_acyclic_and_dependency_ordered() -> None:
    nodes = validate_catalog()
    ordered = topological_order(nodes)
    positions = {node.node_id: index for index, node in enumerate(ordered)}
    assert len(nodes) == len(positions)
    for node in ordered:
        assert all(
            positions[dependency] < positions[node.node_id]
            for dependency in node.dependencies
        )


def test_every_declared_runnable_tool_exists_and_heavy_gate_is_explicit() -> None:
    audit = audit_catalog(ROOT)
    assert audit["missing_tools"] == []
    nodes = {node.node_id: node for node in command_nodes()}
    assert nodes["candidate_gate"].gate_scope == "candidates"
    assert nodes["feature_gate"].gate_scope == "features"
    assert nodes["primary_gate"].gate_scope == "primary"
    assert nodes["k_gate"].gate_scope == "k_sensitivity"
    assert nodes["p12_gate"].gate_scope == "four_route_validation"
    assert nodes["p12_validation"].gate_scope == "four_route_validation"
    assert nodes["p12_plan"].executable
    assert nodes["p12_policy"].executable
    assert nodes["p12_authorize"].executable
    assert nodes["p12_validation"].executable
    assert nodes["ablation_gate"].gate_scope == "ablation"
    assert nodes["ablation_matrix"].executable


def test_postformal_sources_are_fixed_run_local_inputs() -> None:
    nodes = {node.node_id: node for node in command_nodes()}
    assert nodes["postformal_sources"].command == (
        "{PYTHON} tools/d1_reranking/prepare_postformal_sources.py "
        "--run-dir {RUN_DIR} --resume"
    )
    assert "--sample-covariates" not in str(nodes["postformal_evidence"].command)
    assert "--runtime" not in str(nodes["postformal_evidence"].command)
    assert "--table" not in str(nodes["postformal_evidence"].command)
    assert nodes["validation_evidence_tables"].command == (
        "{PYTHON} tools/d1_reranking/assemble_validation_evidence_tables.py "
        "--run-dir {RUN_DIR} --resume"
    )
    assert nodes["lightweight_audits"].command == (
        "{PYTHON} tools/d1_reranking/assemble_lightweight_audits.py "
        "--run-dir {RUN_DIR} --resume"
    )
    assert PROMPT_ARTIFACT_GAPS == ()


def test_catalog_exposes_reachable_p12_and_downstream_pipeline() -> None:
    audit = audit_catalog(ROOT)
    blocked = {node["node_id"] for node in audit["nodes"] if node["blocked_reason"]}
    assert {
        "p12_plan",
        "p12_policy",
        "p12_gate",
        "p12_authorize",
        "p12_validation",
        "ablation_plan",
        "ablation_policy",
        "ablation_gate",
        "ablation_authority",
        "ablation_matrix",
        "ablation_selection",
        "validation_evidence_tables",
        "lightweight_audits",
        "postformal_evidence",
        "prelock",
        "formal_plan",
        "formal_lock",
        "formal_execute",
        "independent_recompute",
        "postformal",
        "finalize",
    }.isdisjoint(blocked)
    report = render_markdown(audit)
    assert "Catalog status: **READY**" in report
    assert "run_four_route_validation.py" in report
    assert "CANDIDATE_GEOMETRY_VISUAL_CHECK.pdf" in report


def test_cycle_and_missing_dependency_are_rejected() -> None:
    nodes = list(command_nodes())
    nodes[0] = replace(nodes[0], dependencies=(nodes[-1].node_id,))
    with pytest.raises(ValueError, match="dependency cycle"):
        validate_catalog(nodes)
    nodes = list(command_nodes())
    nodes[0] = replace(nodes[0], dependencies=("absent",))
    with pytest.raises(ValueError, match="missing dependencies"):
        validate_catalog(nodes)
