from __future__ import annotations

from datetime import datetime, timedelta, timezone
import multiprocessing
from pathlib import Path
import sys

import pytest

import d1_reranking.execution as execution
from d1_reranking.execution import (
    CANONICAL_RANK1_RUN_DIR,
    CANDIDATE_RESOURCE_SCOPE,
    artifact_record,
    candidate_resource_policy,
    exclusive_heavy_resource_lease,
    load_candidate_resource_policy,
    python_invocation_path,
    resource_policy,
    validate_resource_gate,
)
from d1_reranking.resource_gate import resource_thresholds
from unified_reranking.hashing import atomic_json, canonical_sha256


def _write(path: Path, value: dict[str, object]) -> None:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    atomic_json(path, value)


def _hold_heavy_lease(
    run_dir: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with exclusive_heavy_resource_lease(
        Path(run_dir), purpose="synthetic lease holder"
    ):
        ready.set()
        if not release.wait(timeout=10):
            raise RuntimeError("synthetic lease release timed out")


def _plan(tmp_path: Path) -> Path:
    path = tmp_path / "configs" / "d1_primary_matrix_plan.json"
    _write(
        path,
        {
            "schema_version": 1,
            "status": "PLANNED",
            "job_count": 360,
            "execution_authorized": False,
            "jobs": [],
        },
    )
    return path


def test_python_invocation_path_preserves_virtualenv_launcher_symlink(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "bin" / "python"
    launcher.parent.mkdir()
    launcher.symlink_to(Path(sys.executable).resolve())

    observed = python_invocation_path(launcher)

    assert observed == launcher.absolute()
    assert observed.is_symlink()
    assert observed != launcher.resolve()
    with pytest.raises(FileNotFoundError, match="Python launcher is absent"):
        python_invocation_path(tmp_path / "missing-python")


def _windows() -> list[dict[str, object]]:
    started = datetime.now(timezone.utc) - timedelta(minutes=15)
    return [
        {
            "index": index,
            "monotonic_start_seconds": index * 300.0,
            "monotonic_end_seconds": (index + 1) * 300.0,
            "observations": [
                {
                    "captured_at_utc": (
                        started + timedelta(seconds=index * 300 + sample * 15)
                    ).isoformat(),
                    "monotonic_offset_seconds": index * 300 + sample * 15,
                    "memory_free_percent": 50.0,
                    "swap_used_bytes": 0,
                    "disk_free_bytes": 200 * 1024**3,
                    "normalized_load_5m": 0.1,
                    "rank1_workers": [],
                    "rank1_claim_paths": [],
                    "d1_heavy_workers": [],
                    "foreign_heavy_processes": [],
                }
                for sample in range(21)
            ],
        }
        for index in range(3)
    ]


def test_resource_gate_is_plan_policy_host_and_semantics_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = _plan(tmp_path)
    monkeypatch.setattr(
        "d1_reranking.primary_execution.load_primary_plan",
        lambda _path: {"job_count": 360},
    )
    policy_path = tmp_path / "configs" / "d1_resource_gate_policy.json"
    policy = resource_policy(plan_path=plan_path, source_paths=(Path(__file__),))
    atomic_json(policy_path, policy)
    fake_host = {
        "hostname": "test-host",
        "boot_session": "boot-1",
        "logical_cpu_count": 15,
        "memory_total_bytes": 24 * 1024**3,
    }
    monkeypatch.setattr(execution, "host_contract", lambda: fake_host)
    gate_path = tmp_path / "00_audit" / "resource_gates" / "gate.json"
    gate = {
        "schema_version": 1,
        "status": "PASS",
        "gate_id": "gate",
        "run_dir": str(tmp_path.resolve()),
        "started_at_utc": (
            datetime.now(timezone.utc) - timedelta(minutes=15)
        ).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_elapsed_seconds": 900.0,
        "host": fake_host,
        "rank1_run_dir": str(CANONICAL_RANK1_RUN_DIR),
        "plan": artifact_record(plan_path),
        "policy": artifact_record(policy_path),
        "scope": policy["scope"],
        "thresholds": resource_thresholds(),
        "windows": _windows(),
        "failure_reasons": [],
        "candidate_test_labels_read": False,
        "sources": {"test": artifact_record(Path(__file__))},
    }
    _write(gate_path, gate)
    _write(
        tmp_path / "00_audit" / "RESOURCE_AUDIT.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "latest_gate": artifact_record(gate_path),
            "scope": policy["scope"],
            "policy": artifact_record(policy_path),
            "candidate_test_labels_read": False,
        },
    )
    observed = validate_resource_gate(
        gate_path,
        run_dir=tmp_path,
        plan_path=plan_path,
        require_fresh=True,
    )
    assert observed["gate_id"] == "gate"

    with pytest.raises(RuntimeError, match="finish time is in the future"):
        validate_resource_gate(
            gate_path,
            run_dir=tmp_path,
            plan_path=plan_path,
            require_fresh=True,
            now=datetime.now(timezone.utc) - timedelta(minutes=1),
        )

    gate["windows"][1]["observations"][0]["swap_used_bytes"] = 1
    _write(gate_path, gate)
    _write(
        tmp_path / "00_audit" / "RESOURCE_AUDIT.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "latest_gate": artifact_record(gate_path),
            "scope": policy["scope"],
            "policy": artifact_record(policy_path),
            "candidate_test_labels_read": False,
        },
    )
    with pytest.raises(RuntimeError, match="semantic replay failed"):
        validate_resource_gate(
            gate_path,
            run_dir=tmp_path,
            plan_path=plan_path,
            require_fresh=False,
        )


def test_candidate_resource_policy_has_no_primary_plan_dependency(
    tmp_path: Path,
) -> None:
    closure_path = tmp_path / "00_audit" / "source_reconciliations" / "x.json"
    _write(
        closure_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "canonical_snapshot": "A",
            "candidate_test_labels_read": False,
        },
    )
    policy = candidate_resource_policy(
        source_closure_path=closure_path, source_paths=(Path(__file__),)
    )
    assert policy["scope"] == CANDIDATE_RESOURCE_SCOPE
    assert policy["rank1_run_dir"] == str(CANONICAL_RANK1_RUN_DIR)
    assert policy["prerequisite"] == artifact_record(closure_path)
    assert "plan" not in policy


def test_active_candidate_resource_policy_is_versioned_and_closure_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closure_path = (
        tmp_path / "00_audit" / "source_reconciliations" / "closure" / "manifest.json"
    )
    _write(
        closure_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "canonical_snapshot": "A",
            "candidate_test_labels_read": False,
        },
    )
    monkeypatch.setattr(
        "d1_reranking.provenance.load_source_closure",
        lambda _root: (closure_path.resolve(), {"status": "PASS"}),
    )
    policy = candidate_resource_policy(
        source_closure_path=closure_path, source_paths=(Path(__file__),)
    )
    policy_path = (
        tmp_path
        / "configs"
        / "candidate_resource_policies"
        / f"{str(policy['content_sha256'])[:20]}.json"
    )
    atomic_json(policy_path, policy)
    _write(
        tmp_path / "configs" / "d1_candidate_resource_gate_policy_active.json",
        {
            "schema_version": 1,
            "status": "LOCKED_POLICY_POINTER",
            "active_policy": artifact_record(policy_path),
            "source_closure": artifact_record(closure_path),
            "candidate_test_labels_read": False,
        },
    )
    observed_path, observed = load_candidate_resource_policy(tmp_path)
    assert observed_path == policy_path.resolve()
    assert observed == policy

    replacement = (
        tmp_path / "00_audit" / "source_reconciliations" / "other" / "manifest.json"
    )
    _write(
        replacement,
        {
            "schema_version": 1,
            "status": "PASS",
            "canonical_snapshot": "A",
            "candidate_test_labels_read": False,
        },
    )
    monkeypatch.setattr(
        "d1_reranking.provenance.load_source_closure",
        lambda _root: (replacement.resolve(), {"status": "PASS"}),
    )
    with pytest.raises(RuntimeError, match="source-closure binding differs"):
        load_candidate_resource_policy(tmp_path)


def test_heavy_resource_lease_is_atomic_across_sibling_runs(tmp_path: Path) -> None:
    run_a = tmp_path / "runs" / "run-a"
    run_b = tmp_path / "runs" / "run-b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_heavy_lease,
        args=(str(run_a), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=10)
        with pytest.raises(RuntimeError, match="holds the D1 heavy resource lease"):
            with exclusive_heavy_resource_lease(run_b, purpose="synthetic contender"):
                raise AssertionError("contender unexpectedly acquired the lease")
    finally:
        release.set()
        process.join(timeout=10)
    assert process.exitcode == 0
    with exclusive_heavy_resource_lease(run_b, purpose="post-release acquisition"):
        assert (run_b.parent / ".d1_heavy_resource.lock").is_file()


def test_all_d1_cli_ledgers_are_guarded_before_entry() -> None:
    tools = Path(__file__).resolve().parents[2] / "tools" / "d1_reranking"
    for path in sorted(tools.glob("*.py")):
        if path.name == "bootstrap.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "with ledger_stage(" not in source or "def main()" not in source:
            continue
        main_source = source.split("def main()", maxsplit=1)[1]
        ledger_index = main_source.find("with ledger_stage(")
        if ledger_index < 0:
            continue
        guard_indices = [
            index
            for marker in (
                "assert_writable_prelock(",
                "assert_writable_postformal(",
                "assert_writable_finalization(",
            )
            if (index := main_source.find(marker)) >= 0
        ]
        guard_index = min(guard_indices, default=-1)
        assert 0 <= guard_index < ledger_index, (
            f"{path.name} can enter ledger_stage before its lifecycle write guard"
        )
