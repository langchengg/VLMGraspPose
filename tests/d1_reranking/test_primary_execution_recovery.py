from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import d1_reranking.plan as primary_plan_module
import d1_reranking.primary_execution as primary_execution
from d1_reranking.execution import artifact_record
from d1_reranking.plan import load_primary_plan, primary_plan
from unified_reranking.hashing import atomic_json, canonical_sha256


ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(
        name, ROOT / "tools/d1_reranking" / filename
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


authorizer = _load_tool("primary_test_authorizer", "authorize_primary_execution.py")
orchestrator = _load_tool("primary_test_orchestrator", "run_primary_matrix.py")


def _manifest(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return path


def _plain(path: Path, value: str = "synthetic\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _bound_primary_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    root = tmp_path / "run"
    root.mkdir()
    _plain(root / "environment.txt")
    python = _plain(tmp_path / "python")
    closure_path = _manifest(
        root / "00_audit/source_closures/synthetic.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "canonical_snapshot": "A",
            "candidate_test_labels_read": False,
        },
    )
    monkeypatch.setattr(
        primary_plan_module,
        "load_source_closure",
        lambda _root: (
            closure_path.resolve(),
            {
                "status": "PASS",
                "canonical_snapshot": "A",
                "candidate_test_labels_read": False,
            },
        ),
    )
    folds = _plain(root / "04_splits/fold_assignments.parquet", "folds\n")
    _manifest(
        root / "04_splits/split_leakage_audit.json",
        {
            "schema_version": 1,
            "status": "PASS_WITH_SEQUENCE_OVERLAP_LIMITATION",
            "candidate_test_labels_read": False,
            "fold_assignments": artifact_record(folds),
        },
    )
    calibration_artifact = _plain(root / "05_calibration/top5/calibrator.pkl")
    _manifest(
        root / "05_calibration/top5/calibration_manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "selected_method": "platt",
            "candidate_test_labels_read": False,
            "sources": {"code": artifact_record(Path(__file__))},
            "artifacts": {"calibrator": artifact_record(calibration_artifact)},
        },
    )
    for split in ("train", "validation"):
        denominator = _plain(root / f"01_manifests/d1_paired_{split}.parquet")
        assert denominator.is_file()
        candidate_pool = _plain(root / f"02_candidates/{split}/top5.parquet")
        candidate_hashes = _plain(root / f"02_candidates/{split}/hashes.parquet")
        _manifest(
            root / f"02_candidates/{split}/manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "artifacts": {
                    "top5": artifact_record(candidate_pool),
                    "candidate_hashes": artifact_record(candidate_hashes),
                },
            },
        )
        raw_artifact = _plain(
            root / f"03_features/{split}/top5/matched_common_raw/features.parquet"
        )
        _manifest(
            root / f"03_features/{split}/top5/matched_common_raw/manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "artifacts": {"candidate_features": artifact_record(raw_artifact)},
            },
        )
        final_artifact = _plain(
            root / f"03_features/{split}/top5/T2_matched_common/features.parquet"
        )
        _manifest(
            root / f"03_features/{split}/top5/T2_matched_common/manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "artifacts": {"candidate_features": artifact_record(final_artifact)},
            },
        )
        labels = _plain(root / f"03_features/{split}/top5/labels/labels.parquet")
        _manifest(
            root / f"03_features/{split}/top5/labels/manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "artifact": artifact_record(labels),
            },
        )
    plan_path = root / "configs/d1_primary_matrix_plan.json"
    value = primary_plan(
        run_dir=root,
        python_path=python,
        tool_paths=(Path(__file__),),
    )
    atomic_json(plan_path, value)
    return root, plan_path, python


def test_bound_primary_plan_rebuilds_and_rejects_input_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan_path, _python = _bound_primary_fixture(tmp_path, monkeypatch)
    loaded = load_primary_plan(plan_path)
    assert loaded["input_closure_bound"] is True
    assert loaded["job_count"] == 360
    assert set(loaded["sources"]["development"]) == {"train", "validation"}
    assert loaded["sources"]["selected_calibration"]["selected_method"] == "platt"
    assert "test" not in loaded["sources"]["development"]

    raw_path = root / "03_features/train/top5/matched_common_raw/features.parquet"
    raw_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_primary_plan(plan_path)


def test_primary_failure_fresh_gate_resume_runs_only_remaining_360(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    executable = _plain(tmp_path / "python")
    _manifest(
        root / "configs/d1_primary_matrix_plan.json",
        {"schema_version": 1, "status": "PLANNED"},
    )
    jobs = []
    for index in range(360):
        configuration = {"schema_version": 1, "index": index}
        job_id = canonical_sha256(configuration)[:16]
        jobs.append(
            {
                "job_id": job_id,
                "configuration": configuration,
                "worker_argv": [
                    "-m",
                    "tools.d1_reranking.train_primary_cell",
                    "--job-id",
                    job_id,
                    "--resume",
                ],
            }
        )
    plan = {
        "jobs": jobs,
        "job_count": 360,
        "job_ids_sha256": canonical_sha256([job["job_id"] for job in jobs]),
        "sources": {"environment": {"python_executable": artifact_record(executable)}},
    }
    gate_one = _manifest(
        root / "00_audit/resource_gates/gate_one.json",
        {"schema_version": 1, "status": "PASS", "candidate_test_labels_read": False},
    )
    gate_two = _manifest(
        root / "00_audit/resource_gates/gate_two.json",
        {"schema_version": 1, "status": "PASS", "candidate_test_labels_read": False},
    )
    clean = {
        "rank1_workers": [],
        "rank1_claim_paths": [],
        "d1_heavy_workers": [],
        "foreign_heavy_processes": [],
    }
    monkeypatch.setattr(authorizer, "load_primary_plan", lambda _path: plan)
    monkeypatch.setattr(
        authorizer,
        "validate_resource_gate",
        lambda *args, **kwargs: {"gate_id": Path(args[0]).stem},
    )
    monkeypatch.setattr(authorizer, "collect_resource_snapshot", lambda **kwargs: clean)
    monkeypatch.setattr(authorizer, "evaluate_resource_snapshot", lambda *a, **k: [])
    monkeypatch.setattr(primary_execution, "host_contract", lambda: {"synthetic": True})
    first = authorizer.run(
        root,
        gate_manifest=gate_one,
        owner="synthetic",
        rank1_run_dir=primary_execution.CANONICAL_RANK1_RUN_DIR,
    )
    inherited: dict[str, dict[str, str]] = {}
    for job in jobs[:7]:
        result = _manifest(
            root / f"synthetic_outputs/{job['job_id']}.json",
            {"schema_version": 1, "status": "COMPLETE"},
        )
        inherited[str(job["job_id"])] = artifact_record(result)
    primary_execution.write_execution_event(
        root,
        execution=first,
        sequence=1,
        status="FAILED",
        owner_pid=None,
        current_job_id=None,
        claim=None,
        outputs=inherited,
        commands=(),
        failure="synthetic failure",
    )
    monkeypatch.setattr(authorizer, "validate_primary_result", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="newly completed fresh resource gate"):
        authorizer.run(
            root,
            gate_manifest=gate_one,
            owner="synthetic",
            rank1_run_dir=primary_execution.CANONICAL_RANK1_RUN_DIR,
        )
    second = authorizer.run(
        root,
        gate_manifest=gate_two,
        owner="synthetic",
        rank1_run_dir=primary_execution.CANONICAL_RANK1_RUN_DIR,
    )
    assert second["execution_id"] != first["execution_id"]
    assert second["resume_outputs"] == inherited

    pointer = primary_execution.load_content_manifest(
        root / primary_execution.PRIMARY_EXECUTION_POINTER_RELATIVE,
        name="synthetic primary pointer",
        statuses=("ACTIVE",),
    )
    event_path = Path(pointer["latest_event"]["path"])
    initial_event = primary_execution.load_content_manifest(
        event_path, name="synthetic primary event", statuses=("ACTIVE",)
    )
    process_calls: list[str] = []
    events: list[dict[str, object]] = []
    monkeypatch.setattr(orchestrator, "assert_writable_prelock", lambda _root: None)
    monkeypatch.setattr(orchestrator, "load_primary_plan", lambda _path: plan)
    monkeypatch.setattr(
        orchestrator,
        "validate_active_execution",
        lambda *args, **kwargs: (
            primary_execution.execution_directory(root, str(second["execution_id"]))
            / "execution.json",
            second,
            event_path,
            initial_event,
        ),
    )
    monkeypatch.setattr(orchestrator, "_live_recheck", lambda *args, **kwargs: {})
    monkeypatch.setattr(orchestrator, "validate_primary_result", lambda *a, **k: {})
    monkeypatch.setattr(orchestrator, "transition_pipeline_status", lambda *a, **k: {})

    def fake_claim(
        root: Path, *, job: dict[str, object], **kwargs: object
    ) -> tuple[Path, dict[str, object]]:
        path = _plain(root / f"synthetic_claims/{job['job_id']}.json")
        return path, {"status": "CLAIMED"}

    def fake_process(command: list[str], **kwargs: object) -> SimpleNamespace:
        process_calls.append(command[command.index("--job-id") + 1])
        environment = kwargs["env"]
        assert all(environment[name] == "1" for name in orchestrator.THREAD_ENVIRONMENT)
        assert environment["PYTHONHASHSEED"] == "0"
        return SimpleNamespace(returncode=0, stdout="synthetic", stderr="")

    def fake_child(
        stdout: str, *, root: Path, job: dict[str, object], **kwargs: object
    ) -> dict[str, str]:
        path = _manifest(
            root / f"synthetic_outputs/{job['job_id']}.json",
            {"schema_version": 1, "status": "COMPLETE"},
        )
        return artifact_record(path)

    def fake_event(root: Path, **kwargs: object) -> tuple[Path, dict[str, object]]:
        value = dict(kwargs)
        value["completed_jobs"] = len(value["outputs"])
        events.append(value)
        path = _plain(root / f"event-{len(events)}.json")
        return path, value

    monkeypatch.setattr(orchestrator, "create_job_claim", fake_claim)
    monkeypatch.setattr(orchestrator.subprocess, "run", fake_process)
    monkeypatch.setattr(orchestrator, "_child_result", fake_child)
    monkeypatch.setattr(orchestrator, "write_execution_event", fake_event)
    monkeypatch.setattr(
        orchestrator,
        "_validate_bijection",
        lambda **kwargs: (
            None
            if set(kwargs["outputs"]) == {str(job["job_id"]) for job in jobs}
            else (_ for _ in ()).throw(AssertionError("incomplete outputs"))
        ),
    )
    complete = orchestrator._run_under_lease(
        root,
        python=executable,
        rank1_run_dir=primary_execution.CANONICAL_RANK1_RUN_DIR,
    )
    assert process_calls == [str(job["job_id"]) for job in jobs[7:]]
    assert complete["status"] == "COMPLETE"
    assert complete["completed_jobs"] == 360
    assert events[-1]["status"] == "COMPLETE"
