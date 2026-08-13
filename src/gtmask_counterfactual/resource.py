"""Fresh 3x5-minute resource gate and global D1 kernel lease."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import os
from pathlib import Path
import stat
import time
from typing import Any

from unified_reranking.hashing import canonical_sha256

from d1_reranking.resource_gate import (
    RESOURCE_SNAPSHOT_PERIOD_SECONDS,
    RESOURCE_WINDOW_COUNT,
    RESOURCE_WINDOW_DURATION_SECONDS,
    collect_resource_snapshot,
    evaluate_resource_gate,
    evaluate_resource_snapshot,
    host_contract,
    resource_thresholds,
)


GATE_FRESHNESS_SECONDS = 300


@contextmanager
def exclusive_d1_flock(run_dir: Path, *, purpose: str) -> Iterator[Path]:
    """Hold the same repository-global flock used by all existing D1 work."""

    root = run_dir.expanduser().resolve()
    if not purpose.strip():
        raise ValueError("D1 resource lease purpose is required")
    lock_path = root.parent / ".d1_heavy_resource.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"D1 lock is not a regular file: {lock_path}")
        os.set_inheritable(descriptor, False)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"D1 global resource lease is held: {lock_path}"
            ) from error
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def collect_fresh_three_by_five_gate(
    *,
    repo_root: Path,
    rank1_run_dir: Path,
    snapshot_collector: Callable[..., dict[str, Any]] = collect_resource_snapshot,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Collect exactly three continuous five-minute windows.

    Tests inject a synthetic clock and collector; production callers receive the
    actual 15-minute gate.  The caller must hold :func:`exclusive_d1_flock`
    across both this gate and the authorized D1 launch.
    """

    started_at = datetime.now(timezone.utc)
    gate_started = monotonic()
    windows: list[dict[str, Any]] = []
    for index in range(RESOURCE_WINDOW_COUNT):
        window_started = monotonic()
        observations: list[dict[str, Any]] = []
        while True:
            observation = dict(
                snapshot_collector(
                    repo_root=repo_root,
                    rank1_run_dir=rank1_run_dir,
                )
            )
            observation["monotonic_offset_seconds"] = monotonic() - gate_started
            observations.append(observation)
            elapsed = monotonic() - window_started
            if elapsed >= RESOURCE_WINDOW_DURATION_SECONDS:
                break
            sleeper(
                min(
                    RESOURCE_SNAPSHOT_PERIOD_SECONDS,
                    RESOURCE_WINDOW_DURATION_SECONDS - elapsed,
                )
            )
        windows.append(
            {
                "index": index,
                "monotonic_start_seconds": window_started - gate_started,
                "monotonic_end_seconds": monotonic() - gate_started,
                "observations": observations,
            }
        )
    passed, reasons = evaluate_resource_gate(windows)
    finished_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "gate_type": "gtmask_d1_three_continuous_five_minute_windows_v1",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "monotonic_elapsed_seconds": monotonic() - gate_started,
        "host": host_contract(),
        "thresholds": resource_thresholds(),
        "windows": windows,
        "failure_reasons": reasons,
        "candidate_test_labels_read": False,
        "gt_mask_pixels_read": False,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def validate_fresh_gate(
    gate: Mapping[str, Any], *, now: datetime | None = None
) -> None:
    """Reject stale, incomplete, failed, or semantically altered D1 gates."""

    if gate.get("status") != "PASS":
        raise RuntimeError("D1 resource gate did not pass")
    if gate.get("gate_type") != "gtmask_d1_three_continuous_five_minute_windows_v1":
        raise RuntimeError("D1 resource gate type differs")
    if gate.get("thresholds") != resource_thresholds():
        raise RuntimeError("D1 resource gate thresholds differ")
    if gate.get("host") != host_contract():
        raise RuntimeError("D1 resource gate host/boot contract differs")
    unsigned = dict(gate)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("D1 resource gate content hash differs")
    windows = gate.get("windows")
    if not isinstance(windows, list):
        raise RuntimeError("D1 resource gate windows are absent")
    passed, reasons = evaluate_resource_gate(windows)
    if not passed:
        raise RuntimeError(f"D1 resource gate replay failed: {reasons}")
    try:
        finished = datetime.fromisoformat(str(gate["finished_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("D1 resource gate finish time is invalid") from error
    if finished.tzinfo is None:
        raise RuntimeError("D1 resource gate finish time lacks timezone")
    current = datetime.now(timezone.utc) if now is None else now
    if current.tzinfo is None:
        raise RuntimeError("D1 gate validation time lacks timezone")
    if finished > current + timedelta(seconds=5):
        raise RuntimeError("D1 resource gate finish time is in the future")
    if current > finished + timedelta(seconds=GATE_FRESHNESS_SECONDS):
        raise RuntimeError("D1 resource gate is stale")


def validate_live_resources(
    *, repo_root: Path, rank1_run_dir: Path, prefix: str
) -> dict[str, Any]:
    """Fail closed on a fresh instantaneous launch-time resource snapshot."""

    snapshot = collect_resource_snapshot(
        repo_root=repo_root.expanduser().resolve(),
        rank1_run_dir=rank1_run_dir.expanduser().resolve(),
    )
    failures = evaluate_resource_snapshot(snapshot, prefix=prefix)
    if failures:
        raise RuntimeError(f"GT-mask live resource recheck failed: {failures}")
    return snapshot


__all__ = [
    "GATE_FRESHNESS_SECONDS",
    "collect_fresh_three_by_five_gate",
    "exclusive_d1_flock",
    "validate_fresh_gate",
    "validate_live_resources",
]
