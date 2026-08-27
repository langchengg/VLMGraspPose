from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import gtmask_counterfactual.protocol_inputs as protocol_inputs_module
from gtmask_counterfactual.execution_contracts import (
    canonical_route_contracts,
    canonical_semantic_contracts,
    canonical_source_code_inventory,
    validate_scientific_bindings,
)
from gtmask_counterfactual.io import artifact_record, atomic_json, canonical_sha256
from gtmask_counterfactual.protocol_inputs import assemble_protocol_inputs


@pytest.fixture(autouse=True)
def _synthetic_execution_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = tmp_path / "synthetic-adapter/ADAPTER_MANIFEST.json"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("{}\n", encoding="utf-8")
    pilot = tmp_path / "synthetic-pilot/C1_PILOT_SOURCE_ADAPTER.json"
    pilot.parent.mkdir(parents=True)
    pilot.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        protocol_inputs_module,
        "build_g1_c1_source_adapter",
        lambda **kwargs: adapter,
    )
    monkeypatch.setattr(
        protocol_inputs_module,
        "build_c1_pilot_source_adapter",
        lambda **kwargs: pilot,
    )
    monkeypatch.setattr(
        protocol_inputs_module,
        "canonical_route_contracts",
        lambda adapter_manifest, pilot_manifest: {
            "g1": {"synthetic_adapter": str(adapter_manifest)},
            "c1": {
                "synthetic_adapter": str(adapter_manifest),
                "synthetic_pilot": str(pilot_manifest),
            },
            "d1": {"case": "B"},
        },
    )


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    return path


def _run(tmp_path: Path) -> Path:
    root = tmp_path / "runs/fair_gtmask_counterfactual_g1_c1_d1_synthetic"
    source = _write_json(root / "00_audit/source.json", {"status": "PASS"})
    _write_json(
        root / "04_predicted_replay/BASELINE_REPLAY_MANIFEST.json",
        {"status": "PASS"},
    )
    (root / "02_sample_manifest").mkdir(parents=True)
    (root / "03_gt_mask_registry").mkdir(parents=True)
    (root / "02_sample_manifest/counterfactual_manifest.parquet").write_bytes(b"sample")
    (root / "03_gt_mask_registry/gt_mask_registry.parquet").write_bytes(b"registry")
    join: dict[str, object] = {
        "status": "PASS",
        "sources": {
            "denominator": artifact_record(
                root / "02_sample_manifest/counterfactual_manifest.parquet"
            )
        },
    }
    join["content_sha256"] = canonical_sha256(join)
    join_path = _write_json(root / "02_sample_manifest/JOIN_AUDIT.json", join)
    mapping: dict[str, object] = {
        "status": "PASS",
        "stage": "P2_GT_MAPPING_PASS",
        "sample_count": 7_675,
        "mapping_qa_gt_mask_rows_read": 7_675,
        "inputs": {"join_audit": artifact_record(join_path)},
    }
    mapping["content_sha256"] = canonical_sha256(mapping)
    _write_json(root / "03_gt_mask_registry/GT_MASK_MAPPING_AUDIT.json", mapping)
    _write_json(
        root / "pipeline_status.json",
        {
            "status": "P2_GT_MAPPING_PASS",
            "counterfactual_execution_count": 0,
            "source_lock_verification": artifact_record(source),
        },
    )
    return root


def test_canonical_protocol_input_assembly_and_resume(tmp_path: Path) -> None:
    root = _run(tmp_path)
    bindings_path, declaration_path = assemble_protocol_inputs(root)
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    unsigned_bindings = dict(bindings)
    assert unsigned_bindings.pop("content_sha256") == canonical_sha256(
        unsigned_bindings
    )
    unsigned_declaration = dict(declaration)
    assert unsigned_declaration.pop("content_sha256") == canonical_sha256(
        unsigned_declaration
    )
    validate_scientific_bindings(bindings)
    assert set(declaration["routes"]) == {"g1", "c1", "d1"}
    assert assemble_protocol_inputs(root, resume=True) == (
        bindings_path,
        declaration_path,
    )


def test_canonical_protocol_input_resume_and_semantic_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    root = _run(tmp_path)
    bindings_path, _ = assemble_protocol_inputs(root)
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    bindings["resize_rules"]["value"]["test_tuned_blur"] = True
    bindings["resize_rules"]["sha256"] = canonical_sha256(
        bindings["resize_rules"]["value"]
    )
    bindings["content_sha256"] = canonical_sha256(
        {key: value for key, value in bindings.items() if key != "content_sha256"}
    )
    atomic_json(bindings_path, bindings)
    with pytest.raises(RuntimeError, match="existing protocol input differs"):
        assemble_protocol_inputs(root, resume=True)
    with pytest.raises(ValueError, match="semantic binding differs"):
        validate_scientific_bindings(bindings)


def test_non_mask_config_and_candidate_budget_invariance() -> None:
    controls = canonical_semantic_contracts()["resize_rules"]
    assert controls["allowed_changed_variable"] == (
        "target_mask_or_probability_support_only"
    )
    assert controls["non_mask_configuration_must_remain_frozen"] is True
    assert controls["candidate_budget_parameters_must_equal_route_contract"] is True
    assert controls["checkpoint_and_decoder_must_equal_route_contract"] is True
    routes = canonical_route_contracts()
    assert routes["g1"]["candidate_budget"] == 100
    assert routes["c1"]["candidate_budget"] == 100
    assert routes["g1"]["quality_threshold"] == 0.2
    assert routes["c1"]["quality_threshold"] == 0.2
    assert routes["d1"]["raw_candidate_budget"] == 256
    assert routes["d1"]["nms_candidate_budget"] == 30
    assert routes["d1"]["native_pools"] == [5, 10, "allnms"]


def test_no_gt_feedback_to_training_or_selection_modules() -> None:
    controls = canonical_semantic_contracts()["resize_rules"]
    assert controls["training_allowed"] is False
    assert controls["ranker_retraining_allowed"] is False
    assert controls["gate_or_selector_retuning_allowed"] is False
    assert controls["gt_feedback_to_training_or_selection_allowed"] is False
    forbidden = (
        "unified_reranking.training",
        "d1_reranking.train",
        "torch.optim",
        "lightgbm.train",
    )
    for record in canonical_source_code_inventory().values():
        source = Path(str(record["path"]))
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imported
            for prefix in forbidden
        ), f"GT diagnostic imports forbidden training path: {source}"
