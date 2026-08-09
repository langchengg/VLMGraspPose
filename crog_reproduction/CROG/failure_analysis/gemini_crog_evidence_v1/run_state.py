from __future__ import annotations

import errno
import fcntl
import json
import os
import subprocess
import tempfile
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


RUN_STATE_SCHEMA_VERSION = "1.0"
RUN_ACTIVE_FILENAME = ".RUN_ACTIVE"
DO_NOT_PRUNE_FILENAME = ".DO_NOT_PRUNE"
PID_FILENAME = "full_run.pid"
LOCK_FILENAME = "full_run.lock"
PROGRESS_FILENAMES = {
    "phase": "phase_status.json",
    "api": "api_progress.json",
    "cost": "cost_progress.json",
    "runtime": "runtime_progress.json",
}


class RunStateError(RuntimeError):
    """Raised when durable run-state metadata is unsafe or inconsistent."""


class RunAlreadyActiveError(RunStateError):
    """Raised when another process owns, or credibly appears to own, the run."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        pid = self.report.get("pid")
        phase = self.report.get("current_phase")
        super().__init__(f"another full runner is active (pid={pid}, phase={phase})")


@dataclass(frozen=True)
class RunStatePaths:
    root: Path
    run_active: Path
    do_not_prune: Path
    pid: Path
    lock: Path
    phase: Path
    api: Path
    cost: Path
    runtime: Path

    @classmethod
    def under(cls, root: str | Path) -> "RunStatePaths":
        resolved = Path(root).resolve()
        return cls(
            root=resolved,
            run_active=resolved / RUN_ACTIVE_FILENAME,
            do_not_prune=resolved / DO_NOT_PRUNE_FILENAME,
            pid=resolved / PID_FILENAME,
            lock=resolved / LOCK_FILENAME,
            phase=resolved / PROGRESS_FILENAMES["phase"],
            api=resolved / PROGRESS_FILENAMES["api"],
            cost=resolved / PROGRESS_FILENAMES["cost"],
            runtime=resolved / PROGRESS_FILENAMES["runtime"],
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    mode: int = 0o600,
) -> None:
    """Durably replace a JSON file without exposing a partially written value."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            # fdopen owns the descriptor from this point, including exceptional
            # exits from JSON serialization.
            descriptor = -1
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def read_json_object(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunStateError(f"cannot safely read run-state JSON: {source}") from exc
    if not isinstance(value, dict):
        raise RunStateError(f"run-state JSON must contain an object: {source}")
    return value


def pid_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        raise
    return True


def process_start_token(pid: int) -> str | None:
    """Return a PID-reuse token without exposing command arguments."""

    proc_stat = Path(f"/proc/{int(pid)}/stat")
    if proc_stat.exists():
        try:
            # Field 22 is the process start time in clock ticks.  The command
            # name may contain spaces, so split only after its final ')'.
            tail = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            return f"proc-start-ticks:{tail[19]}"
        except (OSError, IndexError):
            return None
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(int(pid))],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return None if result.returncode or not value else f"ps-lstart:{value}"


def _owner_process_matches(owner: Mapping[str, Any]) -> bool:
    if not bool(owner.get("active", False)):
        return False
    try:
        pid = int(owner["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if not pid_is_alive(pid):
        return False
    recorded = owner.get("process_start_token")
    if recorded is None:
        # Legacy PID metadata cannot distinguish a live owner from PID reuse;
        # stopping is safer than starting a duplicate paid runner.
        return True
    observed = process_start_token(pid)
    return observed is None or str(recorded) == observed


def _try_lock(descriptor: int) -> bool:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def inspect_active_runner(run_root: str | Path) -> dict[str, Any]:
    """Return a race-tolerant, secret-free active-runner report."""

    paths = RunStatePaths.under(run_root)
    owner = read_json_object(paths.pid) or {}
    lock_held = False
    if paths.lock.exists():
        descriptor = os.open(paths.lock, os.O_RDWR)
        try:
            if _try_lock(descriptor):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            else:
                lock_held = True
        finally:
            os.close(descriptor)
    try:
        pid = int(owner["pid"])
    except (KeyError, TypeError, ValueError):
        pid = None
    pid_alive = bool(pid is not None and pid_is_alive(pid))
    owner_matches = _owner_process_matches(owner)
    active = bool(lock_held or owner_matches)
    return {
        "active": active,
        "lock_held": lock_held,
        "pid": pid,
        "pid_alive": pid_alive,
        "started_at_utc": owner.get("started_at_utc"),
        "current_phase": owner.get("current_phase"),
        "run_id": owner.get("run_id"),
        "active_worker_pids": [pid] if active and pid is not None else [],
        "metadata_inconsistent": bool(lock_held != owner_matches),
    }


def _default_progress_payloads(
    *,
    experiment_id: str,
    run_id: str,
    pid: int,
    phase: str,
    started_at_utc: str,
) -> dict[str, dict[str, Any]]:
    common = {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "updated_at_utc": started_at_utc,
    }
    return {
        "phase": {
            **common,
            "kind": "phase_status",
            "status": "running",
            "current_phase": phase,
            "phases": {},
        },
        "api": {
            **common,
            "kind": "api_progress",
            "current_phase": phase,
            "counts": {
                "planned": 0,
                "in_flight": 0,
                "succeeded": 0,
                "retryable_failed": 0,
                "permanent_failed": 0,
                "abstain": 0,
                "technical_fallback": 0,
                "cache_hits": 0,
                "new_api_attempts": 0,
                "retries": 0,
            },
            "models": {},
            "last_successful_request_timestamp": None,
        },
        "cost": {
            **common,
            "kind": "cost_progress",
            "currency": "USD",
            "max_spend_usd": None,
            "actual_estimated_spend_usd": 0.0,
            "outstanding_request_reserve_usd": 0.0,
            "next_request_upper_bound_usd": 0.0,
            "remaining_budget_usd": None,
            "token_usage": {"input": 0, "output": 0, "thought": 0},
        },
        "runtime": {
            **common,
            "kind": "runtime_progress",
            "status": "running",
            "current_phase": phase,
            "started_at_utc": started_at_utc,
            "wall_time_seconds": 0.0,
            "p50_latency_seconds": None,
            "p95_latency_seconds": None,
            "requests_per_hour": 0.0,
            "eta_seconds": None,
            "active_worker_pids": [int(pid)],
        },
    }


class FullRunState:
    """Durable lifecycle and progress state for one resumable full runner."""

    def __init__(
        self,
        run_root: str | Path,
        *,
        experiment_id: str,
        current_phase: str = "preflight",
        run_id: str | None = None,
        now: Callable[[], str] = utc_now,
    ) -> None:
        self.paths = RunStatePaths.under(run_root)
        self.experiment_id = str(experiment_id)
        self.current_phase = str(current_phase)
        self._run_id_was_explicit = run_id is not None
        self.run_id = str(run_id or uuid.uuid4())
        self._now = now
        self._descriptor: int | None = None
        self._owner: dict[str, Any] | None = None

    @property
    def acquired(self) -> bool:
        return self._descriptor is not None

    @property
    def owner_metadata(self) -> dict[str, Any] | None:
        return None if self._owner is None else deepcopy(self._owner)

    def _require_acquired(self) -> None:
        if not self.acquired:
            raise RunStateError("full-run lock is not acquired")

    def acquire(self) -> "FullRunState":
        if self.acquired:
            return self
        self.paths.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        if not _try_lock(descriptor):
            os.close(descriptor)
            raise RunAlreadyActiveError(inspect_active_runner(self.paths.root))

        self._descriptor = descriptor
        try:
            previous = read_json_object(self.paths.pid)
            if previous and _owner_process_matches(previous):
                # Do not probe the lock through a second descriptor here: on
                # some flock implementations that can alter this process's
                # newly acquired lock.  The live owner metadata is sufficient
                # to take the conservative stop path.
                report = {
                    "active": True,
                    "lock_held": False,
                    "pid": previous.get("pid"),
                    "pid_alive": True,
                    "started_at_utc": previous.get("started_at_utc"),
                    "current_phase": previous.get("current_phase"),
                    "run_id": previous.get("run_id"),
                    "active_worker_pids": [previous.get("pid")],
                    "metadata_inconsistent": True,
                }
                raise RunAlreadyActiveError(report)
            if (
                previous
                and not self._run_id_was_explicit
                and previous.get("experiment_id") == self.experiment_id
                and previous.get("run_id")
            ):
                # A process restart is still the same formal run.  Callers can
                # explicitly supply a new run_id when they truly intend a new
                # experiment instead of a resume.
                self.run_id = str(previous["run_id"])

            started = self._now()
            history: list[dict[str, Any]] = []
            if previous:
                existing_history = previous.get("owner_history", [])
                if isinstance(existing_history, list):
                    history.extend(deepcopy(existing_history))
                prior = deepcopy(previous)
                prior.pop("owner_history", None)
                history.append(prior)
            owner = {
                "schema_version": RUN_STATE_SCHEMA_VERSION,
                "kind": "full_run_owner",
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
                "pid": os.getpid(),
                "process_start_token": process_start_token(os.getpid()),
                "started_at_utc": started,
                "updated_at_utc": started,
                "current_phase": self.current_phase,
                "active": True,
                "recovered_stale_owner": bool(previous),
                "owner_history": history,
            }
            atomic_write_json(self.paths.pid, owner)
            self._owner = owner
            atomic_write_json(self.paths.run_active, owner)
            if not self.paths.do_not_prune.exists():
                atomic_write_json(
                    self.paths.do_not_prune,
                    {
                        "schema_version": RUN_STATE_SCHEMA_VERSION,
                        "kind": "do_not_prune",
                        "experiment_id": self.experiment_id,
                        "created_at_utc": started,
                        "reason": "resumable paid full run artifacts",
                    },
                )
            defaults = _default_progress_payloads(
                experiment_id=self.experiment_id,
                run_id=self.run_id,
                pid=os.getpid(),
                phase=self.current_phase,
                started_at_utc=started,
            )
            for kind, path in (
                ("phase", self.paths.phase),
                ("api", self.paths.api),
                ("cost", self.paths.cost),
                ("runtime", self.paths.runtime),
            ):
                if path.exists():
                    existing = read_json_object(path)
                    if existing is None or existing.get("kind") != defaults[kind]["kind"]:
                        raise RunStateError(f"unexpected progress schema: {path}")
                else:
                    atomic_write_json(path, defaults[kind])
            self._refresh_runtime_owner(status="running")
            return self
        except Exception:
            self._remove_owned_active_marker()
            try:
                pid_payload = self._owned_file(self.paths.pid)
                if pid_payload is not None:
                    failed_at = self._now()
                    pid_payload.update(
                        {
                            "active": False,
                            "status": "acquire_failed",
                            "updated_at_utc": failed_at,
                            "ended_at_utc": failed_at,
                        }
                    )
                    atomic_write_json(self.paths.pid, pid_payload)
            except RunStateError:
                # Preserve unreadable metadata exactly as found.  Releasing the
                # kernel lock remains mandatory even when the audit file broke.
                pass
            if self._descriptor is not None:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
                os.close(self._descriptor)
                self._descriptor = None
            self._owner = None
            raise

    def _owned_file(self, path: Path) -> dict[str, Any] | None:
        value = read_json_object(path)
        if value is None or value.get("run_id") != self.run_id:
            return None
        return value

    def _remove_owned_active_marker(self) -> None:
        if not self.paths.run_active.exists():
            return
        try:
            value = read_json_object(self.paths.run_active)
        except RunStateError:
            return
        if value is not None and value.get("run_id") == self.run_id:
            self.paths.run_active.unlink()
            _fsync_directory(self.paths.root)

    def _refresh_runtime_owner(self, *, status: str) -> None:
        runtime = read_json_object(self.paths.runtime)
        if runtime is None:
            return
        runtime.update(
            {
                "schema_version": RUN_STATE_SCHEMA_VERSION,
                "kind": "runtime_progress",
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
                "updated_at_utc": self._now(),
                "status": status,
                "current_phase": self.current_phase,
                "active_worker_pids": [os.getpid()] if status == "running" else [],
            }
        )
        atomic_write_json(self.paths.runtime, runtime)

    def update_phase(
        self,
        phase: str,
        *,
        status: str = "running",
        completed: int | None = None,
        total: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_acquired()
        self.current_phase = str(phase)
        payload = read_json_object(self.paths.phase) or {}
        phases = payload.setdefault("phases", {})
        entry = dict(phases.get(self.current_phase, {}))
        entry.update({"status": str(status), "updated_at_utc": self._now()})
        if completed is not None:
            entry["completed"] = int(completed)
        if total is not None:
            entry["total"] = int(total)
        if details:
            entry["details"] = {**dict(entry.get("details", {})), **dict(details)}
        phases[self.current_phase] = entry
        payload.update(
            {
                "schema_version": RUN_STATE_SCHEMA_VERSION,
                "kind": "phase_status",
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
                "updated_at_utc": self._now(),
                "status": str(status),
                "current_phase": self.current_phase,
            }
        )
        atomic_write_json(self.paths.phase, payload)
        for path in (self.paths.pid, self.paths.run_active):
            value = self._owned_file(path)
            if value is not None:
                value.update(
                    {
                        "current_phase": self.current_phase,
                        "updated_at_utc": self._now(),
                    }
                )
                atomic_write_json(path, value)
                if path == self.paths.pid:
                    self._owner = value
        self._refresh_runtime_owner(status="running")
        return deepcopy(payload)

    def update_progress(self, kind: str, updates: Mapping[str, Any]) -> dict[str, Any]:
        self._require_acquired()
        if kind not in PROGRESS_FILENAMES:
            raise ValueError(f"unknown progress kind: {kind}")
        path = getattr(self.paths, kind)
        payload = read_json_object(path) or {}
        payload.update(deepcopy(dict(updates)))
        payload.update(
            {
                "schema_version": RUN_STATE_SCHEMA_VERSION,
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
                "updated_at_utc": self._now(),
            }
        )
        atomic_write_json(path, payload)
        return deepcopy(payload)

    def active_pid_report(self) -> dict[str, Any]:
        self._require_acquired()
        return {
            "active": True,
            "pid": os.getpid(),
            "started_at_utc": None if self._owner is None else self._owner.get("started_at_utc"),
            "current_phase": self.current_phase,
            "run_id": self.run_id,
            "active_worker_pids": [os.getpid()],
        }

    def release(self, *, status: str = "released") -> None:
        if not self.acquired:
            return
        descriptor, self._descriptor = self._descriptor, None
        assert descriptor is not None
        try:
            ended = self._now()
            pid_payload = self._owned_file(self.paths.pid)
            if pid_payload is not None:
                pid_payload.update(
                    {
                        "active": False,
                        "status": str(status),
                        "current_phase": self.current_phase,
                        "updated_at_utc": ended,
                        "ended_at_utc": ended,
                    }
                )
                atomic_write_json(self.paths.pid, pid_payload)
                self._owner = pid_payload
            self._refresh_runtime_owner(status=str(status))
            self._remove_owned_active_marker()
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def __enter__(self) -> "FullRunState":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release(status="failed" if exc_type is not None else "released")
