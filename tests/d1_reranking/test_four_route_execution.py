from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from d1_reranking.execution import CANONICAL_RANK1_RUN_DIR, artifact_record
from d1_reranking.four_route_execution import (
    P12_EXECUTION_POINTER_RELATIVE,
    _artifact_record_binds_path,
    load_four_route_execution_plan,
    validate_complete_execution,
    write_resource_policy,
)
from d1_reranking.four_route_plan import PLAN_RELATIVE_PATH, PRODUCER_JOB_STAGES
from tools.d1_reranking import run_four_route_validation as runner
from unified_reranking.hashing import atomic_json, canonical_sha256


def _write(path: Path, value: str = "fixed") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _manifest(path: Path, value: dict[str, Any]) -> Path:
    value = dict(value)
    value["content_sha256"] = canonical_sha256(value)
    atomic_json(path, value)
    return path


def _enriched_artifact_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = dict(artifact_record(path))
    record["bytes"] = path.stat().st_size
    return record


def _plan(root: Path) -> tuple[Path, dict[str, Any]]:
    inputs = {
        name: artifact_record(_write(root / "development" / f"{name}.bin", name))
        for name in (
            "router_train",
            "router_validation",
            "union_train",
            "union_validation",
            "folds",
            "train_denominator",
            "validation_denominator",
            "train_d1",
            "train_t3",
            "validation_d1",
            "validation_t3",
            "train_crog",
            "train_g1",
            "train_c1",
            "validation_crog",
            "validation_g1",
            "validation_c1",
        )
    }
    spec = _manifest(
        root / "configs/p12_validation_producer_spec.json",
        {
            "schema_version": 1,
            "status": "VALIDATION_PRODUCER_DECLARED",
            "router": {
                "train_oof": inputs["router_train"]["path"],
                "validation": inputs["router_validation"]["path"],
                "feature_columns": {
                    route: [f"{route.lower()}_feature"] for route in ("G1", "C1", "D1")
                },
            },
            "union": {
                "train_top20": inputs["union_train"]["path"],
                "validation_top20": inputs["union_validation"]["path"],
                "folds": inputs["folds"]["path"],
                "train_denominator": inputs["train_denominator"]["path"],
                "validation_denominator": inputs["validation_denominator"]["path"],
                "feature_columns": ["feature_a"],
            },
            "t4": {
                split: {
                    "d1_top5": inputs[f"{split}_d1"]["path"],
                    "t3_manifest": inputs[f"{split}_t3"]["path"],
                    "peer_top5": {
                        route: inputs[f"{split}_{route.lower()}"]["path"]
                        for route in ("CROG", "G1", "C1")
                    },
                }
                for split in ("train", "validation")
            },
            "candidate_test_labels_read": False,
            "selection_used_test_metrics": False,
        },
    )
    configurations = [
        {"stage": stage, "device": "cpu", "threads": 1} for stage in PRODUCER_JOB_STAGES
    ]
    jobs = [
        {"job_id": canonical_sha256(configuration)[:16], "configuration": configuration}
        for configuration in configurations
    ]
    producer: dict[str, Any] = {
        "schema_version": 1,
        "status": "READY",
        "spec": artifact_record(spec),
        "router": {
            "train_oof": inputs["router_train"],
            "validation": inputs["router_validation"],
            "feature_columns": {
                route: [f"{route.lower()}_feature"] for route in ("G1", "C1", "D1")
            },
        },
        "union": {
            "train_top20": inputs["union_train"],
            "validation_top20": inputs["union_validation"],
            "folds": inputs["folds"],
            "train_denominator": inputs["train_denominator"],
            "validation_denominator": inputs["validation_denominator"],
            "feature_columns": ["feature_a"],
        },
        "t4": {
            split: {
                "d1_top5": inputs[f"{split}_d1"],
                "t3_manifest": inputs[f"{split}_t3"],
                "peer_top5": {
                    route: inputs[f"{split}_{route.lower()}"]
                    for route in ("CROG", "G1", "C1")
                },
            }
            for split in ("train", "validation")
        },
        "jobs": jobs,
        "job_count": len(jobs),
        "job_ids_sha256": canonical_sha256([job["job_id"] for job in jobs]),
        "device": "cpu",
        "max_parallel": 1,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    producer["content_sha256"] = canonical_sha256(producer)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "status": "PLANNED",
        "validation_producer_ready": True,
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
        "sources": {"validation_producer": producer},
    }
    plan["source_signature_sha256"] = canonical_sha256(plan["sources"])
    plan_path = _manifest(root / PLAN_RELATIVE_PATH, plan)
    return plan_path, plan


def _producer_manifest(root: Path, *, stage: str, plan_path: Path) -> Path:
    artifact = _write(root / "synthetic_outputs" / f"{stage}.bin", stage)
    path = runner._producer_output_path(root, stage)
    if stage == "top20_union_validation":
        internal = _manifest(
            root / "13_four_route_extension/union/union_plan.json",
            {
                "schema_version": 1,
                "status": "PLANNED",
                "sources": {"p12_plan": _enriched_artifact_record(plan_path)},
                "candidate_test_labels_read": False,
            },
        )
        sources = {"plan": artifact_record(internal)}
        status = "VALIDATION_LOCKED"
    else:
        sources = {"p12_plan": _enriched_artifact_record(plan_path)}
        status = "COMPLETE"
    return _manifest(
        path,
        {
            "schema_version": 1,
            "status": status,
            "sources": sources,
            "artifacts": {"synthetic": artifact_record(artifact)},
            "candidate_test_labels_read": False,
            "selection_used_test_metrics": False,
        },
    )


def _fake_worker(
    root: Path,
    job: Mapping[str, Any],
    claim_path: Path,
    execution_path: Path,
    _lease_path: Path,
) -> Path:
    plan_path, _plan_value = load_four_route_execution_plan(root)
    producer = _producer_manifest(
        root, stage=str(job["configuration"]["stage"]), plan_path=plan_path
    )
    return runner.write_job_result(
        root,
        job=job,
        plan_path=plan_path,
        execution_path=execution_path,
        claim_path=claim_path,
        producer_manifest_path=producer,
        resume=True,
    )


def _authorize_without_live_gate(
    monkeypatch: pytest.MonkeyPatch, root: Path, gate: Path
) -> dict[str, Any]:
    monkeypatch.setattr(
        runner, "validate_gate_for_execution", lambda *_args, **_kwargs: {}
    )
    return runner.authorize(
        root,
        gate_manifest=gate,
        owner="synthetic-p12",
        rank1_run_dir=CANONICAL_RANK1_RUN_DIR,
        collect_snapshot=lambda **_kwargs: {"synthetic": True},
        evaluate_snapshot=lambda *_args, **_kwargs: [],
    )


def test_serial_validation_execution_positive_is_plan_bound_and_has_no_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    plan_path, plan = _plan(root)
    write_resource_policy(root, resume=False)
    gate = _write(root / "00_audit/resource_gates/synthetic.json", "gate")
    _authorize_without_live_gate(monkeypatch, root, gate)
    event = runner.run_serial(
        root,
        python=Path("/usr/bin/python3"),
        rank1_run_dir=CANONICAL_RANK1_RUN_DIR,
        resource_lease_path=root.parent / ".d1_heavy_resource.lock",
        worker_executor=_fake_worker,
        collect_snapshot=lambda **_kwargs: {"synthetic": True},
        evaluate_snapshot=lambda *_args, **_kwargs: [],
    )
    replay = validate_complete_execution(root)

    assert event["status"] == "COMPLETE"
    assert event["completed_jobs"] == 5
    pointer = json.loads((root / P12_EXECUTION_POINTER_RELATIVE).read_text())
    assert replay["event"] == pointer["latest_event"]
    assert plan["sources"]["validation_producer"]["test_inputs_referenced"] is False
    assert all("test" not in stage for stage in PRODUCER_JOB_STAGES)
    for record in event["outputs"].values():
        result = json.loads(Path(record["path"]).read_text())
        assert result["sources"]["plan"] == artifact_record(plan_path)


def test_complete_execution_rejects_wrong_declared_plan_bytes(
    tmp_path: Path,
) -> None:
    artifact = _write(tmp_path / "artifact.bin", "artifact")
    record = _enriched_artifact_record(artifact)
    record["bytes"] += 1

    assert not _artifact_record_binds_path(
        record, expected_path=artifact, name="synthetic artifact"
    )


def test_serial_execution_recovers_only_completed_claimed_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    _plan(root)
    write_resource_policy(root, resume=False)
    first_gate = _write(root / "00_audit/resource_gates/first.json", "first")
    _authorize_without_live_gate(monkeypatch, root, first_gate)
    calls: list[str] = []

    def fail_third(
        root_value: Path,
        job: Mapping[str, Any],
        claim: Path,
        execution: Path,
        lease: Path,
    ) -> Path:
        calls.append(str(job["configuration"]["stage"]))
        if len(calls) == 3:
            raise RuntimeError("synthetic interruption")
        return _fake_worker(root_value, job, claim, execution, lease)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run_serial(
            root,
            python=Path("/usr/bin/python3"),
            rank1_run_dir=CANONICAL_RANK1_RUN_DIR,
            resource_lease_path=root.parent / ".d1_heavy_resource.lock",
            worker_executor=fail_third,
            collect_snapshot=lambda **_kwargs: {},
            evaluate_snapshot=lambda *_args, **_kwargs: [],
        )
    failed_pointer = json.loads((root / P12_EXECUTION_POINTER_RELATIVE).read_text())
    assert failed_pointer["status"] == "FAILED"
    assert failed_pointer["completed_jobs"] == 2

    second_gate = _write(root / "00_audit/resource_gates/second.json", "second")
    recovered = _authorize_without_live_gate(monkeypatch, root, second_gate)
    assert len(recovered["resume_outputs"]) == 2
    event = runner.run_serial(
        root,
        python=Path("/usr/bin/python3"),
        rank1_run_dir=CANONICAL_RANK1_RUN_DIR,
        resource_lease_path=root.parent / ".d1_heavy_resource.lock",
        worker_executor=_fake_worker,
        collect_snapshot=lambda **_kwargs: {},
        evaluate_snapshot=lambda *_args, **_kwargs: [],
    )
    assert event["status"] == "COMPLETE"
    assert event["completed_jobs"] == 5
    validate_complete_execution(root)


def test_execution_rejects_plan_input_drift_and_rehashed_job_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    _plan(root)
    input_path = root / "development/router_train.bin"
    input_path.write_text("drift", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_four_route_execution_plan(root)

    root = tmp_path / "run2"
    _plan(root)
    write_resource_policy(root, resume=False)
    gate = _write(root / "00_audit/resource_gates/synthetic.json", "gate")
    _authorize_without_live_gate(monkeypatch, root, gate)
    runner.run_serial(
        root,
        python=Path("/usr/bin/python3"),
        rank1_run_dir=CANONICAL_RANK1_RUN_DIR,
        resource_lease_path=root.parent / ".d1_heavy_resource.lock",
        worker_executor=_fake_worker,
        collect_snapshot=lambda **_kwargs: {},
        evaluate_snapshot=lambda *_args, **_kwargs: [],
    )
    pointer = json.loads((root / P12_EXECUTION_POINTER_RELATIVE).read_text())
    event_path = Path(pointer["latest_event"]["path"])
    event = json.loads(event_path.read_text())
    result_path = Path(next(iter(event["outputs"].values()))["path"])
    result = json.loads(result_path.read_text())
    result["configuration"]["threads"] = 2
    result.pop("content_sha256")
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(result_path, result)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_complete_execution(root)


def test_execution_plan_rejects_any_test_job(tmp_path: Path) -> None:
    root = tmp_path / "run"
    path, plan = _plan(root)
    producer = plan["sources"]["validation_producer"]
    producer["jobs"][-1]["configuration"]["stage"] = "t4_test"
    producer["jobs"][-1]["job_id"] = canonical_sha256(
        producer["jobs"][-1]["configuration"]
    )[:16]
    producer["job_ids_sha256"] = canonical_sha256(
        [job["job_id"] for job in producer["jobs"]]
    )
    producer["content_sha256"] = canonical_sha256(
        {key: value for key, value in producer.items() if key != "content_sha256"}
    )
    plan["source_signature_sha256"] = canonical_sha256(plan["sources"])
    plan.pop("content_sha256", None)
    plan["content_sha256"] = canonical_sha256(plan)
    atomic_json(path, plan)
    with pytest.raises(RuntimeError, match="contract differs"):
        load_four_route_execution_plan(root)
