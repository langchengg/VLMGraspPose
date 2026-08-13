"""Fail-closed resource gate for heavy D1 selector work on macOS."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESOURCE_WINDOW_COUNT = 3
RESOURCE_WINDOW_DURATION_SECONDS = 300
RESOURCE_SNAPSHOT_PERIOD_SECONDS = 15
RESOURCE_OBSERVATIONS_PER_WINDOW = (
    RESOURCE_WINDOW_DURATION_SECONDS // RESOURCE_SNAPSHOT_PERIOD_SECONDS + 1
)
RESOURCE_TIMING_TOLERANCE_SECONDS = 2.0
RESOURCE_MAX_OBSERVATION_GAP_SECONDS = RESOURCE_SNAPSHOT_PERIOD_SECONDS * 1.5
MIN_MEMORY_FREE_PERCENT = 35.0
MIN_DISK_FREE_BYTES = 100 * 1024**3
MAX_FOREIGN_CPU_PERCENT = 80.0
MAX_FOREIGN_RSS_BYTES = 1024**3

_D1_HEAVY_MARKERS = (
    "tools.d1_reranking.build_candidates",
    "tools.d1_reranking.train_primary_cell",
    "tools.d1_reranking.run_primary_matrix",
    "tools.d1_reranking.run_four_route_validation",
    "tools/d1_reranking/build_candidates.py",
    "tools/d1_reranking/train_primary_cell.py",
    "tools/d1_reranking/run_primary_matrix.py",
    "tools/d1_reranking/run_four_route_validation.py",
)
_RANK1_WORKER_MARKERS = ("train-worker", "train_worker", "train-worker.py")
_APPLE_SYSTEM_SERVICE_PREFIXES = (
    "/System/Library/",
    "/Library/Apple/System/Library/",
    "/usr/libexec/",
)
_ORCHESTRATOR_UI_PREFIXES = ("/Applications/ChatGPT.app/Contents/",)


def _run(command: list[str]) -> str:
    value = subprocess.run(command, check=True, capture_output=True, text=True)
    return value.stdout.strip()


def _bytes(text: str) -> int:
    match = re.fullmatch(r"\s*([0-9.]+)\s*([BKMGTP]?)\s*", text, re.I)
    if match is None:
        raise ValueError(f"cannot parse byte quantity: {text!r}")
    scale = {
        "": 1,
        "B": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }
    return int(float(match.group(1)) * scale[match.group(2).upper()])


def _memory_free_percent() -> float:
    output = _run(["/usr/bin/memory_pressure"])
    match = re.search(r"System-wide memory free percentage:\s*([0-9.]+)%", output)
    if match is None:
        raise RuntimeError("memory_pressure did not report free percentage")
    return float(match.group(1))


def _swap_used_bytes() -> int:
    output = _run(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    match = re.search(r"used\s*=\s*([0-9.]+)([BKMGTP])", output, re.I)
    if match is None:
        raise RuntimeError("sysctl vm.swapusage did not report used swap")
    return _bytes(f"{match.group(1)}{match.group(2)}")


def host_contract() -> dict[str, Any]:
    """Return a boot-session-bound host identity for gate reuse checks."""

    hostname = _run(["/bin/hostname"])
    boot_session = _run(["/usr/sbin/sysctl", "-n", "kern.boottime"])
    memory_total = int(_run(["/usr/sbin/sysctl", "-n", "hw.memsize"]))
    cpu_count = os.cpu_count() or 0
    if not hostname or not boot_session or cpu_count <= 0:
        raise RuntimeError("D1 resource gate cannot resolve the host contract")
    return {
        "hostname": hostname,
        "boot_session": boot_session,
        "logical_cpu_count": cpu_count,
        "memory_total_bytes": memory_total,
    }


def _processes() -> list[dict[str, Any]]:
    output = _run(["/bin/ps", "-axo", "pid=,ppid=,uid=,%cpu=,rss=,command="])
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        pieces = line.strip().split(maxsplit=5)
        if len(pieces) != 6:
            continue
        pid, ppid, uid, cpu, rss_kib, command = pieces
        rows.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "uid": int(uid),
                "cpu_percent": float(cpu),
                "rss_bytes": int(rss_kib) * 1024,
                "command": command,
            }
        )
    return rows


def _ancestor_pids(processes: list[dict[str, Any]], pid: int) -> set[int]:
    parents = {int(row["pid"]): int(row["ppid"]) for row in processes}
    result = {pid}
    cursor = pid
    while cursor in parents and parents[cursor] > 0 and parents[cursor] not in result:
        cursor = parents[cursor]
        result.add(cursor)
    return result


def _foreign_heavy_processes(
    processes: list[dict[str, Any]], *, ignored: set[int], owner_uid: int
) -> list[dict[str, Any]]:
    """Return heavy user processes, excluding unrelated OS control daemons.

    Aggregate pressure from every process remains covered by the frozen load,
    memory and swap thresholds.  The per-process veto is specifically a
    foreign *worker* interlock, so root-owned macOS services such as
    ``oahd-helper`` and ``PerfPowerServices`` are not user workers.  The Codex
    desktop host is the orchestration UI rather than a competing compute
    worker; sustained pressure from it remains visible to the aggregate gates.
    """

    return [
        row
        for row in processes
        if row["pid"] not in ignored
        and int(row["uid"]) == int(owner_uid)
        and not str(row["command"]).startswith(_APPLE_SYSTEM_SERVICE_PREFIXES)
        and not str(row["command"]).startswith(_ORCHESTRATOR_UI_PREFIXES)
        and (
            float(row["cpu_percent"]) >= MAX_FOREIGN_CPU_PERCENT
            or int(row["rss_bytes"]) >= MAX_FOREIGN_RSS_BYTES
        )
    ]


def collect_resource_snapshot(
    *, repo_root: Path, rank1_run_dir: Path, observer_pid: int | None = None
) -> dict[str, Any]:
    """Collect one label-free, read-only resource snapshot."""

    processes = _processes()
    ignored = _ancestor_pids(
        processes, os.getpid() if observer_pid is None else observer_pid
    )
    rank1_root = str(rank1_run_dir.expanduser().resolve())
    rank1_workers = [
        row
        for row in processes
        if row["pid"] not in ignored
        and rank1_root in str(row["command"])
        and any(marker in str(row["command"]) for marker in _RANK1_WORKER_MARKERS)
    ]
    d1_workers = [
        row
        for row in processes
        if row["pid"] not in ignored
        and any(marker in str(row["command"]) for marker in _D1_HEAVY_MARKERS)
    ]
    foreign_heavy = _foreign_heavy_processes(
        processes, ignored=ignored, owner_uid=os.getuid()
    )
    claims_dir = rank1_run_dir / "logs" / "train_workers" / "claims"
    claim_paths = (
        sorted(str(path.resolve()) for path in claims_dir.glob("*") if path.is_file())
        if claims_dir.is_dir()
        else []
    )
    disk_free = shutil.disk_usage(repo_root).free
    load_1m, load_5m, load_15m = os.getloadavg()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "memory_free_percent": _memory_free_percent(),
        "swap_used_bytes": _swap_used_bytes(),
        "disk_free_bytes": disk_free,
        "load_average": [load_1m, load_5m, load_15m],
        "normalized_load_5m": load_5m / max(1, os.cpu_count() or 1),
        "rank1_workers": rank1_workers,
        "rank1_claim_paths": claim_paths,
        "d1_heavy_workers": d1_workers,
        "foreign_heavy_processes": foreign_heavy,
    }


def evaluate_resource_snapshot(sample: dict[str, Any], *, prefix: str) -> list[str]:
    """Evaluate one instantaneous observation under the locked thresholds."""

    reasons: list[str] = []
    if float(sample.get("memory_free_percent", -1.0)) < MIN_MEMORY_FREE_PERCENT:
        reasons.append(f"{prefix} memory free is below {MIN_MEMORY_FREE_PERCENT}%")
    if int(sample.get("disk_free_bytes", -1)) < MIN_DISK_FREE_BYTES:
        reasons.append(f"{prefix} disk free is below {MIN_DISK_FREE_BYTES} bytes")
    normalized_load = float(sample.get("normalized_load_5m", -1.0))
    if normalized_load < 0 or normalized_load > 0.5:
        reasons.append(f"{prefix} normalized five-minute load exceeds 0.5")
    for key in (
        "rank1_workers",
        "rank1_claim_paths",
        "d1_heavy_workers",
        "foreign_heavy_processes",
    ):
        if sample.get(key):
            reasons.append(f"{prefix} {key} is non-empty")
    return reasons


def evaluate_resource_gate(
    windows: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Evaluate three complete, non-overlapping five-minute windows."""

    reasons: list[str] = []
    if len(windows) != RESOURCE_WINDOW_COUNT:
        reasons.append(f"window_count={len(windows)} != {RESOURCE_WINDOW_COUNT}")
    previous_end: float | None = None
    samples: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        start = float(window.get("monotonic_start_seconds", -1.0))
        end = float(window.get("monotonic_end_seconds", -1.0))
        duration = end - start
        if int(window.get("index", -1)) != window_index:
            reasons.append(f"window[{window_index}] index differs")
        if previous_end is not None:
            if start < previous_end:
                reasons.append(f"window[{window_index}] overlaps the previous window")
            elif start - previous_end > RESOURCE_TIMING_TOLERANCE_SECONDS:
                reasons.append(f"window[{window_index}] is not continuous")
        previous_end = end
        if duration < RESOURCE_WINDOW_DURATION_SECONDS:
            reasons.append(f"window[{window_index}] is shorter than five minutes")
        if duration > (
            RESOURCE_WINDOW_DURATION_SECONDS + RESOURCE_MAX_OBSERVATION_GAP_SECONDS
        ):
            reasons.append(f"window[{window_index}] is longer than the allowed cadence")
        observations = window.get("observations")
        if (
            not isinstance(observations, list)
            or len(observations) < RESOURCE_OBSERVATIONS_PER_WINDOW
        ):
            reasons.append(
                f"window[{window_index}] has fewer than "
                f"{RESOURCE_OBSERVATIONS_PER_WINDOW} observations"
            )
            continue
        offsets: list[float] = []
        wall_times: list[datetime] = []
        samples.extend(observations)
        for sample_index, sample in enumerate(observations):
            if not isinstance(sample, dict):
                reasons.append(
                    f"window[{window_index}].observations[{sample_index}] is invalid"
                )
                continue
            try:
                offset = float(sample["monotonic_offset_seconds"])
            except (KeyError, TypeError, ValueError):
                reasons.append(
                    f"window[{window_index}].observations[{sample_index}] "
                    "lacks monotonic timing"
                )
            else:
                offsets.append(offset)
            try:
                captured = datetime.fromisoformat(str(sample["captured_at_utc"]))
                if captured.tzinfo is None:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                reasons.append(
                    f"window[{window_index}].observations[{sample_index}] "
                    "has invalid UTC time"
                )
            else:
                wall_times.append(captured.astimezone(timezone.utc))
            reasons.extend(
                evaluate_resource_snapshot(
                    sample,
                    prefix=f"window[{window_index}].observations[{sample_index}]",
                )
            )
        if len(offsets) == len(observations):
            if (
                offsets[0] < start - RESOURCE_TIMING_TOLERANCE_SECONDS
                or offsets[0] > start + RESOURCE_TIMING_TOLERANCE_SECONDS
            ):
                reasons.append(
                    f"window[{window_index}] first observation is off cadence"
                )
            if offsets[-1] < (
                start
                + RESOURCE_WINDOW_DURATION_SECONDS
                - RESOURCE_TIMING_TOLERANCE_SECONDS
            ):
                reasons.append(f"window[{window_index}] lacks a five-minute endpoint")
            gaps = [after - before for before, after in zip(offsets, offsets[1:])]
            if any(
                gap <= 0 or gap > RESOURCE_MAX_OBSERVATION_GAP_SECONDS for gap in gaps
            ):
                reasons.append(f"window[{window_index}] observation cadence is invalid")
        if len(wall_times) == len(observations) and any(
            after < before for before, after in zip(wall_times, wall_times[1:])
        ):
            reasons.append(f"window[{window_index}] wall-clock observations regress")
    swaps = [int(sample.get("swap_used_bytes", -1)) for sample in samples]
    if any(value < 0 for value in swaps):
        reasons.append("swap usage is missing")
    elif any(after > before for before, after in zip(swaps, swaps[1:])):
        reasons.append("swap usage increased during the gate")
    return not reasons, reasons


def resource_thresholds() -> dict[str, Any]:
    return {
        "window_count": RESOURCE_WINDOW_COUNT,
        "window_duration_seconds": RESOURCE_WINDOW_DURATION_SECONDS,
        "snapshot_period_seconds": RESOURCE_SNAPSHOT_PERIOD_SECONDS,
        "minimum_observations_per_window": RESOURCE_OBSERVATIONS_PER_WINDOW,
        "maximum_observation_gap_seconds": RESOURCE_MAX_OBSERVATION_GAP_SECONDS,
        "minimum_memory_free_percent": MIN_MEMORY_FREE_PERCENT,
        "minimum_disk_free_bytes": MIN_DISK_FREE_BYTES,
        "maximum_foreign_cpu_percent": MAX_FOREIGN_CPU_PERCENT,
        "maximum_foreign_rss_bytes": MAX_FOREIGN_RSS_BYTES,
        "foreign_worker_owner_uid": os.getuid(),
        "excluded_apple_system_service_prefixes": list(_APPLE_SYSTEM_SERVICE_PREFIXES),
        "maximum_normalized_load_5m": 0.5,
        "rank1_workers_required": 0,
        "rank1_claims_required": 0,
        "d1_heavy_workers_required": 0,
        "swap_growth_bytes_allowed": 0,
    }
