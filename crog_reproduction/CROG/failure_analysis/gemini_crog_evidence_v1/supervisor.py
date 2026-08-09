"""Secret-free single-instance supervisor for the resumable full runner.

The supervisor never imports the provider SDK and never issues requests.  It
only observes the durable runner lock/PID/status and starts the same CLI after
an unexpected local-process interruption.  Registered hard-stop states are
terminal and are never restarted automatically.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .run_state import atomic_write_json, inspect_active_runner, read_json_object, utc_now


HARD_STOP_PREFIX = "blocked_"


def _phase_status(run_root: Path) -> str:
    payload = read_json_object(run_root / "phase_status.json") or {}
    return str(payload.get("status") or "unknown")


def should_restart(*, active: bool, phase_status: str) -> bool:
    if active or phase_status == "complete" or phase_status.startswith(HARD_STOP_PREFIX):
        return False
    return True


def _write_status(path: Path, **fields: Any) -> None:
    existing = read_json_object(path) or {}
    atomic_write_json(
        path,
        {
            **existing,
            "schema_version": "1.0",
            "kind": "full_run_supervisor",
            "pid": os.getpid(),
            "updated_at_utc": utc_now(),
            **fields,
        },
    )


def supervise(
    run_root: str | Path,
    *,
    poll_seconds: float = 30.0,
    delete_committed_boards: bool = True,
    max_consecutive_failures: int = 3,
) -> int:
    root = Path(run_root).resolve()
    lock_path = root / "full_run.supervisor.lock"
    status_path = root / "full_run.supervisor.json"
    log_path = root / "full_run.console.log"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    os.fchmod(descriptor, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 4
        failures = 0
        _write_status(status_path, status="monitoring", started_at_utc=utc_now())
        while True:
            report = inspect_active_runner(root)
            status = _phase_status(root)
            if report["active"]:
                failures = 0
                _write_status(
                    status_path,
                    status="monitoring_active_runner",
                    runner_pid=report.get("pid"),
                    current_phase=report.get("current_phase"),
                    run_id=report.get("run_id"),
                )
                time.sleep(poll_seconds)
                continue
            if status == "complete":
                _write_status(status_path, status="complete", runner_pid=None)
                return 0
            if status.startswith(HARD_STOP_PREFIX):
                _write_status(
                    status_path,
                    status="hard_stop_observed",
                    runner_pid=None,
                    hard_stop_status=status,
                )
                return 2
            command = [
                sys.executable,
                "-m",
                "failure_analysis.gemini_crog_evidence_v1.full_run",
                "--run-root",
                str(root),
            ]
            if delete_committed_boards:
                command.append("--delete-committed-boards")
            _write_status(
                status_path,
                status="restarting_runner",
                runner_pid=None,
                previous_phase_status=status,
                consecutive_failures=failures,
            )
            with log_path.open("ab", buffering=0) as log:
                child = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _write_status(status_path, status="runner_restarted", runner_pid=child.pid)
                return_code = child.wait()
            if return_code == 0 and _phase_status(root) == "complete":
                _write_status(status_path, status="complete", runner_pid=None)
                return 0
            if _phase_status(root).startswith(HARD_STOP_PREFIX):
                continue
            failures += 1
            if failures >= max_consecutive_failures:
                _write_status(
                    status_path,
                    status="restart_limit_reached",
                    runner_pid=None,
                    consecutive_failures=failures,
                )
                return 3
            time.sleep(min(60.0, 2.0**failures))
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--retain-boards", action="store_true")
    args = parser.parse_args(argv)
    return supervise(
        args.run_root,
        poll_seconds=args.poll_seconds,
        delete_committed_boards=not args.retain_boards,
    )


if __name__ == "__main__":
    raise SystemExit(main())
