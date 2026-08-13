from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import d1_reranking.feature_plan as feature_plan
from d1_reranking.contracts import RunState
from d1_reranking.execution import artifact_record, exclusive_heavy_resource_lease
from d1_reranking.feature_execution import execution_directory
from d1_reranking.run import transition_pipeline_status
from d1_reranking.run import assert_writable_prelock as real_prelock_guard
from tools.d1_reranking import authorize_feature_extraction as authorizer
from tools.d1_reranking import run_feature_extraction_matrix as orchestrator
from unified_reranking.hashing import atomic_json, canonical_sha256


def _manifest(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return path


def test_feature_plan_is_exact_nine_job_self_and_source_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    closure_path = _manifest(
        root / "01_manifests/source_closures/closure.json",
        {"schema_version": 1, "status": "PASS"},
    )
    atomic_json(root / "environment.txt", {"synthetic": True})
    source = root / "source.bin"
    source.write_bytes(b"source")
    record = artifact_record(source)
    closure = {
        "canonical_snapshot": "A",
        "candidate_test_labels_read": False,
    }
    monkeypatch.setattr(
        feature_plan, "load_source_closure", lambda _root: (closure_path, closure)
    )
    monkeypatch.setattr(
        feature_plan,
        "_candidate_sources",
        lambda *_args, **_kwargs: (
            {split: record for split in feature_plan.FEATURE_SPLITS},
            {
                split: {pool: record for pool in feature_plan.FEATURE_POOLS}
                for split in feature_plan.FEATURE_SPLITS
            },
            {split: record for split in feature_plan.FEATURE_SPLITS},
            {
                split: {"source": record, "run_copy": record}
                for split in feature_plan.FEATURE_SPLITS
            },
        ),
    )
    plan_path = root / "configs/d1_feature_extraction_plan.json"
    plan = feature_plan.write_feature_extraction_plan(
        plan_path,
        run_dir=root,
        python_path=Path(sys.executable),
        tool_paths=(source,),
        resume=False,
    )
    assert plan["job_count"] == 9
    assert plan["max_parallel"] == 1
    assert [
        (job["configuration"]["split"], job["configuration"]["pool"])
        for job in plan["jobs"]
    ] == [
        (split, pool)
        for split in ("train", "validation", "test")
        for pool in ("top5", "top10", "allnms")
    ]
    assert feature_plan.load_feature_extraction_plan(plan_path) == plan
    source.write_bytes(b"tampered")
    with pytest.raises((RuntimeError, ValueError), match="SHA-256|differs"):
        feature_plan.load_feature_extraction_plan(plan_path)
    source.unlink()
    with pytest.raises(RuntimeError, match="missing|regular file"):
        feature_plan.load_feature_extraction_plan(plan_path)


def test_feature_lifecycle_is_atomic_idempotent_and_lock_sensitive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    atomic_json(
        root / "pipeline_status.json",
        {
            "schema_version": 1,
            "status": "CANDIDATES_FROZEN",
            "first_incomplete_stage": "P4_FEATURES_AND_DEVELOPMENT_LABELS",
            "formal_test_executed": False,
            "test_candidate_labels_read": False,
        },
    )
    first = transition_pipeline_status(
        root,
        status=RunState.FEATURES_READY,
        first_incomplete_stage="P7_SPLITS_AND_OOF",
        formal_test_executed=False,
        test_candidate_labels_read=False,
        formal_test_execution_count=0,
    )
    assert first["path"] == str((root / "pipeline_status.json").resolve())
    transition_pipeline_status(
        root,
        status=RunState.FEATURES_READY,
        first_incomplete_stage="P7_SPLITS_AND_OOF",
        formal_test_executed=False,
        test_candidate_labels_read=False,
        formal_test_execution_count=0,
    )
    transition_pipeline_status(
        root,
        status=RunState.TRAIN_OOF,
        first_incomplete_stage="P9_VALIDATION_SCREEN",
        formal_test_executed=False,
        test_candidate_labels_read=False,
        formal_test_execution_count=0,
    )
    transition_pipeline_status(
        root,
        status=RunState.VALIDATION_SCREEN,
        first_incomplete_stage="P13_PRELOCK_READINESS",
        formal_test_executed=False,
        test_candidate_labels_read=False,
        formal_test_execution_count=0,
    )
    locked = tmp_path / "locked"
    atomic_json(
        locked / "pipeline_status.json",
        {
            "schema_version": 1,
            "status": "CANDIDATES_FROZEN",
            "first_incomplete_stage": "P4_FEATURES_AND_DEVELOPMENT_LABELS",
            "formal_test_executed": False,
            "test_candidate_labels_read": False,
        },
    )
    atomic_json(locked / "evidence.json", {"complete": True})
    atomic_json(locked / "08_lock/FORMAL_TEST_LOCK.json", {"locked": True})
    with pytest.raises(PermissionError):
        orchestrator.assert_writable_prelock(locked)


def test_feature_global_lease_rejects_competing_sibling_run(tmp_path: Path) -> None:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    with exclusive_heavy_resource_lease(run_a, purpose="feature-a"):
        with pytest.raises(RuntimeError, match="holds the D1 heavy resource lease"):
            with exclusive_heavy_resource_lease(run_b, purpose="feature-b"):
                pass


def _synthetic_jobs(root: Path) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for index in range(9):
        configuration = {
            "split": ("train", "validation", "test")[index // 3],
            "pool": ("top5", "top10", "allnms")[index % 3],
        }
        job_id = canonical_sha256(configuration)[:16]
        jobs.append(
            {
                "job_id": job_id,
                "configuration": configuration,
                "worker_argv": ["-m", "synthetic.worker", "--job-id", job_id],
                "output_manifest": f"outputs/{job_id}.json",
            }
        )
    return jobs


def test_feature_failure_resume_runs_only_remaining_and_completes_nine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    python = Path(sys.executable).resolve()
    jobs = _synthetic_jobs(root)
    plan = {
        "jobs": jobs,
        "sources": {"environment": {"python_executable": artifact_record(python)}},
    }
    executions = [
        {
            "execution_id": "a" * 24,
            "resume_outputs": {},
        },
        {
            "execution_id": "b" * 24,
            "resume_outputs": {},
        },
        {
            "execution_id": "c" * 24,
            "resume_outputs": {},
        },
    ]
    attempt = {"index": 0, "calls": 0}
    failed_outputs: dict[str, dict[str, str]] = {}
    completed: list[dict[str, object]] = []

    monkeypatch.setattr(
        orchestrator,
        "load_active_feature_extraction_plan",
        lambda _root: (_root / "configs/d1_feature_extraction_plan.json", plan),
    )
    monkeypatch.setattr(orchestrator, "_live_recheck", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(orchestrator, "assert_writable_prelock", lambda _root: None)
    monkeypatch.setattr(
        orchestrator,
        "_validate_resume_outputs",
        lambda **kwargs: dict(kwargs["execution"]["resume_outputs"]),
    )
    monkeypatch.setattr(
        orchestrator,
        "_validate_bijection",
        lambda **kwargs: completed.append(dict(kwargs["outputs"])),
    )
    monkeypatch.setattr(orchestrator, "feature_lifecycle_evidence", lambda _root: {})
    monkeypatch.setattr(
        orchestrator, "transition_pipeline_status", lambda *_a, **_k: {}
    )

    def fake_validate(*_args: object, **_kwargs: object):
        execution = executions[attempt["index"]]
        event = {
            "sequence": 0,
            "current_job_id": None,
            "claim": None,
            "commands": [],
            "outputs": execution["resume_outputs"],
            "completed_jobs": len(execution["resume_outputs"]),
        }
        return root / "execution.json", execution, root / "active.json", event

    monkeypatch.setattr(orchestrator, "validate_active_execution", fake_validate)

    def fake_event(
        _root: Path,
        *,
        sequence: int,
        status: str,
        outputs: dict[str, object],
        **_kwargs: object,
    ):
        path = _manifest(
            root / f"events/{attempt['index']}_{sequence}_{status}.json",
            {"schema_version": 1, "status": status},
        )
        if status == "FAILED":
            failed_outputs.update(outputs)
        return path, {"status": status, "outputs": dict(outputs)}

    monkeypatch.setattr(orchestrator, "write_execution_event", fake_event)

    def fake_subprocess(*_args: object, **_kwargs: object) -> SimpleNamespace:
        attempt["calls"] += 1
        fail = attempt["index"] == 0 and attempt["calls"] == 4
        return SimpleNamespace(returncode=1 if fail else 0, stdout="", stderr="boom")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_subprocess)

    def fake_child(*_args: object, job: dict[str, object], **_kwargs: object):
        path = root / str(job["output_manifest"])
        atomic_json(path, {"job_id": job["job_id"]})
        return artifact_record(path)

    monkeypatch.setattr(orchestrator, "_child_result", fake_child)
    with pytest.raises(RuntimeError, match="failed"):
        orchestrator._run_under_lease(
            root, python=python, rank1_run_dir=orchestrator.CANONICAL_RANK1_RUN_DIR
        )
    assert len(failed_outputs) == 3
    assert attempt["calls"] == 4
    executions[1]["resume_outputs"] = dict(failed_outputs)
    attempt["index"] = 1
    attempt["calls"] = 0
    orchestrator._run_under_lease(
        root, python=python, rank1_run_dir=orchestrator.CANONICAL_RANK1_RUN_DIR
    )
    assert attempt["calls"] == 6
    assert len(completed[-1]) == 9

    attempt["index"] = 2
    attempt["calls"] = 0
    guard_calls = {"count": 0}

    def lock_during_recheck(run_dir: Path) -> None:
        guard_calls["count"] += 1
        if guard_calls["count"] == 2:
            atomic_json(run_dir / "08_lock/FORMAL_TEST_LOCK.json", {"locked": True})
        real_prelock_guard(run_dir)

    monkeypatch.setattr(orchestrator, "assert_writable_prelock", lock_during_recheck)
    with pytest.raises(PermissionError, match="refuses to write after lock"):
        orchestrator._run_under_lease(
            root, python=python, rank1_run_dir=orchestrator.CANONICAL_RANK1_RUN_DIR
        )
    assert attempt["calls"] == 0


def test_feature_failed_resume_rejects_same_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    plan_path = _manifest(
        root / "configs/d1_feature_extraction_plan.json",
        {"schema_version": 1, "status": "PLANNED"},
    )
    gate_path = root / "00_audit/resource_gates/old.json"
    atomic_json(gate_path, {"gate": "old"})
    execution_id = "a" * 24
    authority_path = execution_directory(root, execution_id) / "execution.json"
    authority = _manifest(
        authority_path,
        {
            "schema_version": 1,
            "status": "ACTIVE",
            "execution_id": execution_id,
            "resource_gate": artifact_record(gate_path),
        },
    )
    failed_path = _manifest(
        authority.parent / "events/0001_failed.json",
        {
            "schema_version": 1,
            "status": "FAILED",
            "execution_id": execution_id,
            "outputs": {},
        },
    )
    _manifest(
        root / "configs/d1_feature_execution.json",
        {
            "schema_version": 1,
            "status": "FAILED",
            "execution_id": execution_id,
            "execution": artifact_record(authority),
            "latest_event": artifact_record(failed_path),
        },
    )
    monkeypatch.setattr(
        authorizer,
        "load_active_feature_extraction_plan",
        lambda _root: (
            plan_path,
            {"jobs": [], "job_ids_sha256": canonical_sha256([])},
        ),
    )
    assert plan_path.is_file()
    with pytest.raises(RuntimeError, match="newly completed fresh gate"):
        authorizer.run(
            root,
            gate_manifest=gate_path,
            owner="synthetic",
            rank1_run_dir=authorizer.CANONICAL_RANK1_RUN_DIR,
        )
