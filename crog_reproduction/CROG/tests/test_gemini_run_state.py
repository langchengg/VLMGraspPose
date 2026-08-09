from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from failure_analysis.gemini_crog_evidence_v1.run_state import (
    DO_NOT_PRUNE_FILENAME,
    LOCK_FILENAME,
    PID_FILENAME,
    PROGRESS_FILENAMES,
    RUN_ACTIVE_FILENAME,
    FullRunState,
    RunAlreadyActiveError,
    RunStateError,
    atomic_write_json,
    inspect_active_runner,
    read_json_object,
)


def test_atomic_write_json_is_complete_and_leaves_no_temporary_file(tmp_path: Path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not list(tmp_path.glob(".state.json.tmp-*"))
    assert path.stat().st_mode & 0o777 == 0o600


def test_atomic_write_json_serialization_failure_preserves_previous_file(tmp_path: Path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"value": 1})
    with pytest.raises(ValueError):
        atomic_write_json(path, {"not_finite": float("nan")})
    assert read_json_object(path) == {"value": 1}
    assert not list(tmp_path.glob(".state.json.tmp-*"))


def test_full_run_lifecycle_creates_durable_state_and_retains_prune_guard(tmp_path: Path):
    state = FullRunState(tmp_path, experiment_id="exp", current_phase="preflight", run_id="run-a")
    state.acquire()
    assert (tmp_path / LOCK_FILENAME).exists()
    assert (tmp_path / RUN_ACTIVE_FILENAME).exists()
    assert (tmp_path / DO_NOT_PRUNE_FILENAME).exists()
    owner = read_json_object(tmp_path / PID_FILENAME)
    assert owner is not None
    assert owner["pid"] == os.getpid()
    assert owner["active"] is True
    assert owner["current_phase"] == "preflight"
    for filename in PROGRESS_FILENAMES.values():
        assert (tmp_path / filename).exists()

    state.update_phase("pilot", completed=3, total=100)
    assert read_json_object(tmp_path / PID_FILENAME)["current_phase"] == "pilot"
    state.release(status="interrupted")

    assert not (tmp_path / RUN_ACTIVE_FILENAME).exists()
    assert (tmp_path / DO_NOT_PRUNE_FILENAME).exists()
    assert (tmp_path / LOCK_FILENAME).exists()
    owner = read_json_object(tmp_path / PID_FILENAME)
    assert owner["active"] is False
    assert owner["status"] == "interrupted"
    assert owner["current_phase"] == "pilot"
    runtime = read_json_object(tmp_path / PROGRESS_FILENAMES["runtime"])
    assert runtime["active_worker_pids"] == []


def test_nonblocking_lock_rejects_second_process(tmp_path: Path):
    state = FullRunState(tmp_path, experiment_id="exp", run_id="parent").acquire()
    code = """
import sys
from failure_analysis.gemini_crog_evidence_v1.run_state import FullRunState, RunAlreadyActiveError
try:
    FullRunState(sys.argv[1], experiment_id='exp', run_id='child').acquire()
except RunAlreadyActiveError as exc:
    assert exc.report['active']
    raise SystemExit(23)
raise SystemExit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert result.returncode == 23, result.stderr
    finally:
        state.release()


def test_inspect_active_runner_reports_owner_without_secrets(tmp_path: Path):
    with FullRunState(tmp_path, experiment_id="exp", current_phase="calibration", run_id="run") as state:
        report = state.active_pid_report()
        assert report["active"] is True
        assert report["pid"] == os.getpid()
        assert report["current_phase"] == "calibration"
        assert "process_start_token" not in report


def test_stale_pid_is_recovered_and_preserved_in_history(tmp_path: Path):
    atomic_write_json(
        tmp_path / PID_FILENAME,
        {
            "kind": "full_run_owner",
            "run_id": "old-run",
            "pid": 999_999_999,
            "active": True,
            "started_at_utc": "2026-01-01T00:00:00Z",
            "current_phase": "pilot",
        },
    )
    state = FullRunState(tmp_path, experiment_id="exp", run_id="new-run").acquire()
    try:
        owner = read_json_object(tmp_path / PID_FILENAME)
        assert owner["recovered_stale_owner"] is True
        assert owner["owner_history"][-1]["run_id"] == "old-run"
        assert owner["owner_history"][-1]["pid"] == 999_999_999
    finally:
        state.release()


def test_resume_reuses_prior_run_id_when_not_explicitly_overridden(tmp_path: Path):
    first = FullRunState(tmp_path, experiment_id="exp", run_id="formal-run").acquire()
    first.release(status="interrupted")
    resumed = FullRunState(tmp_path, experiment_id="exp").acquire()
    try:
        assert resumed.run_id == "formal-run"
        assert read_json_object(tmp_path / PID_FILENAME)["run_id"] == "formal-run"
    finally:
        resumed.release()


def test_malformed_pid_metadata_is_not_overwritten(tmp_path: Path):
    path = tmp_path / PID_FILENAME
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(RunStateError):
        FullRunState(tmp_path, experiment_id="exp").acquire()
    assert path.read_text(encoding="utf-8") == "not json\n"


def test_existing_progress_is_resume_safe_and_preserves_custom_data(tmp_path: Path):
    atomic_write_json(
        tmp_path / PROGRESS_FILENAMES["phase"],
        {
            "kind": "phase_status",
            "schema_version": "1.0",
            "current_phase": "pilot",
            "custom_resume_cursor": 73,
            "phases": {"pilot": {"completed": 73, "total": 100}},
        },
    )
    state = FullRunState(tmp_path, experiment_id="exp", run_id="resume").acquire()
    try:
        payload = read_json_object(tmp_path / PROGRESS_FILENAMES["phase"])
        assert payload["custom_resume_cursor"] == 73
        assert payload["phases"]["pilot"]["completed"] == 73
    finally:
        state.release()


def test_failed_acquire_does_not_leave_a_live_owner_record(tmp_path: Path):
    atomic_write_json(
        tmp_path / PROGRESS_FILENAMES["phase"],
        {"kind": "wrong_kind", "user_data": True},
    )
    with pytest.raises(RunStateError, match="unexpected progress schema"):
        FullRunState(tmp_path, experiment_id="exp", run_id="failed").acquire()
    owner = read_json_object(tmp_path / PID_FILENAME)
    assert owner["active"] is False
    assert owner["status"] == "acquire_failed"
    assert not (tmp_path / RUN_ACTIVE_FILENAME).exists()
    assert read_json_object(tmp_path / PROGRESS_FILENAMES["phase"])["user_data"] is True

    # The failed owner did not strand the fcntl lock.
    atomic_write_json(
        tmp_path / PROGRESS_FILENAMES["phase"],
        {"kind": "phase_status", "phases": {}},
    )
    state = FullRunState(tmp_path, experiment_id="exp", run_id="recovered").acquire()
    state.release()


def test_update_progress_keeps_schema_and_active_pid(tmp_path: Path):
    with FullRunState(tmp_path, experiment_id="exp", run_id="run") as state:
        value = state.update_progress(
            "api",
            {"counts": {"planned": 200, "succeeded": 25}, "last_successful_request_timestamp": "now"},
        )
        assert value["kind"] == "api_progress"
        assert value["run_id"] == "run"
        assert value["counts"]["succeeded"] == 25
        report = state.active_pid_report()
        assert report["active_worker_pids"] == [os.getpid()]


def test_release_does_not_delete_foreign_active_marker(tmp_path: Path):
    state = FullRunState(tmp_path, experiment_id="exp", run_id="ours").acquire()
    atomic_write_json(tmp_path / RUN_ACTIVE_FILENAME, {"run_id": "foreign", "user_data": True})
    state.release()
    assert read_json_object(tmp_path / RUN_ACTIVE_FILENAME) == {
        "run_id": "foreign",
        "user_data": True,
    }


def test_progress_updates_require_the_lock(tmp_path: Path):
    state = FullRunState(tmp_path, experiment_id="exp")
    with pytest.raises(RunStateError, match="not acquired"):
        state.update_progress("api", {})


def test_unknown_progress_kind_is_rejected(tmp_path: Path):
    with FullRunState(tmp_path, experiment_id="exp") as state:
        with pytest.raises(ValueError, match="unknown progress kind"):
            state.update_progress("unknown", {})


def test_inspect_reports_inactive_after_clean_release(tmp_path: Path):
    state = FullRunState(tmp_path, experiment_id="exp", run_id="run").acquire()
    state.release(status="complete")
    report = inspect_active_runner(tmp_path)
    assert report["active"] is False
    assert report["pid"] == os.getpid()
    assert report["pid_alive"] is True
