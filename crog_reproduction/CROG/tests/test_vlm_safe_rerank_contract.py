from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from failure_analysis.vlm_safe_rerank.policy import (
    GateThresholds,
    PairGateEvidence,
    full_denominator_metrics,
    protected_select,
)
from failure_analysis.vlm_safe_rerank.schema import PairwiseDecision
from failure_analysis.vlm_safe_rerank.security import assert_no_ground_truth
from failure_analysis.vlm_safe_rerank.runner import (
    assert_audited_feature_source,
    assert_phase_inference_manifest_frozen,
)


REPO = Path(__file__).resolve().parents[1]


def _unsafe_evidence(**changes: object) -> PairGateEvidence:
    values = dict(
        challenger_id="c1", p_benefit=.99, p_harm=0.0, reliability=1.0,
        challenger_hard_valid=True, critic_decision=PairwiseDecision.PREFER_CHALLENGER,
        confirmation_decision=PairwiseDecision.PREFER_CHALLENGER, confirmation_valid=True,
    )
    values.update(changes)
    return PairGateEvidence(**values)


def test_network_failure_keeps_q_only() -> None:
    result = protected_select("c0", [_unsafe_evidence(terminal_failure=True)], GateThresholds(.8, .1), query_gate_passed=True)
    assert result.selected_id == "c0"


def test_schema_failure_keeps_q_only() -> None:
    result = protected_select("c0", [_unsafe_evidence(critic_decision=None)], GateThresholds(.8, .1), query_gate_passed=True)
    assert result.selected_id == "c0"


def test_abstain_keeps_q_only() -> None:
    result = protected_select("c0", [_unsafe_evidence(critic_decision=PairwiseDecision.KEEP_BASELINE)], GateThresholds(.8, .1), query_gate_passed=True)
    assert result.selected_id == "c0"


def test_insufficient_evidence_keeps_q_only() -> None:
    result = protected_select("c0", [_unsafe_evidence(critic_decision=PairwiseDecision.INSUFFICIENT_EVIDENCE)], GateThresholds(.8, .1), query_gate_passed=True)
    assert result.selected_id == "c0"


def test_no_ground_truth_in_prompt() -> None:
    assert_no_ground_truth({"instruction": "compare baseline and challenger", "evidence": {"baseline": {"candidate_id": "c0"}, "challenger": {"candidate_id": "c1"}}})
    with pytest.raises(AssertionError):
        assert_no_ground_truth({"corrected_evaluator": {"candidate_correct": True}})


@pytest.mark.parametrize(
    "payload",
    [
        {"labels": {"candidate_0": True}},
        {"gt": {"candidate_id": "candidate_1"}},
        {"path": "/tmp/evaluation_only/labels.jsonl"},
        {"nested": {"Evaluation": {"result": 1}}},
    ],
)
def test_no_gt_guard_rejects_label_and_path_smuggling(payload: dict) -> None:
    with pytest.raises(AssertionError):
        assert_no_ground_truth(payload)


def test_corrected_evaluator_read_only() -> None:
    path = REPO / "failure_analysis/reranking_outputs/v2_20260727T174412+0100/labels_val/corrected/labels.jsonl"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    assert len(first["candidate_labels"]) == 5
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_full_denominator_includes_fallbacks() -> None:
    metrics = full_denominator_metrics([
        {"baseline": True, "selected": True, "switched": False, "fallback": True},
        {"baseline": False, "selected": False, "switched": False, "fallback": True},
    ], baseline_key="baseline", selected_key="selected")
    assert metrics["total"] == 2 and metrics["final_successes"] == 1


def test_group_split_has_no_scene_overlap() -> None:
    payload = json.loads((REPO / "failure_analysis/reranking_outputs/v2_20260727T174412+0100/split_manifest.json").read_text())
    partitions = {}
    for row in payload["rows"]:
        partition = str(row["development_partition"])
        if partition in {"calibration", "validation", "formal_test"}:
            partitions.setdefault(partition, set()).add(str(row["frame_id"]))
    names = sorted(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            assert not (partitions[left] & partitions[right])


def test_audited_feature_source_detects_any_file_drift(tmp_path: Path) -> None:
    feature = tmp_path / "features.jsonl"
    feature.write_text('{"sample_id":"s"}\n', encoding="utf-8")
    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    audit_path = tmp_path / "audit_inventory.json"
    audit_path.write_text(json.dumps({
        "frozen_sources": {"train": {"path": str(feature.resolve()), "file_sha256": digest}}
    }), encoding="utf-8")
    (tmp_path / "AUDIT_INVENTORY_IDENTITY.json").write_text(json.dumps({
        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()
    }), encoding="utf-8")
    assert_audited_feature_source(tmp_path, feature)
    feature.write_text('{"sample_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity changed"):
        assert_audited_feature_source(tmp_path, feature)


def test_phase_manifest_sidecar_detects_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "inference_manifest.json"
    manifest.write_text('{"rows":[]}\n', encoding="utf-8")
    (tmp_path / "INFERENCE_MANIFEST_IDENTITY.json").write_text(json.dumps({
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
    }), encoding="utf-8")
    assert_phase_inference_manifest_frozen(tmp_path)
    manifest.write_text('{"rows":[{"sample_id":"changed"}]}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity changed"):
        assert_phase_inference_manifest_frozen(tmp_path)
