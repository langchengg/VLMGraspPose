from __future__ import annotations

from datetime import datetime, timedelta, timezone

from d1_reranking.resource_gate import (
    _foreign_heavy_processes,
    _processes,
    evaluate_resource_gate,
)


def _windows() -> list[dict[str, object]]:
    start = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    return [
        {
            "index": index,
            "monotonic_start_seconds": index * 300.0,
            "monotonic_end_seconds": (index + 1) * 300.0,
            "observations": [
                {
                    "captured_at_utc": (
                        start + timedelta(seconds=index * 300 + sample * 15)
                    ).isoformat(),
                    "monotonic_offset_seconds": index * 300 + sample * 15,
                    "memory_free_percent": 50.0,
                    "swap_used_bytes": 0,
                    "disk_free_bytes": 200 * 1024**3,
                    "normalized_load_5m": 0.2,
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


def test_resource_gate_requires_three_clean_five_minute_samples() -> None:
    passed, reasons = evaluate_resource_gate(_windows())
    assert passed
    assert reasons == []


def test_resource_gate_fails_closed_on_worker_and_swap_growth() -> None:
    windows = _windows()
    windows[1]["observations"][0]["foreign_heavy_processes"] = [{"pid": 99}]
    windows[2]["observations"][0]["swap_used_bytes"] = 4096
    passed, reasons = evaluate_resource_gate(windows)
    assert not passed
    assert any("foreign_heavy_processes" in reason for reason in reasons)
    assert "swap usage increased during the gate" in reasons


def test_resource_gate_rejects_shortened_or_overlapping_windows() -> None:
    windows = _windows()
    windows[1]["monotonic_start_seconds"] = 299.0
    windows[1]["monotonic_end_seconds"] = 300.0
    passed, reasons = evaluate_resource_gate(windows)
    assert not passed
    assert any("overlaps" in reason for reason in reasons)
    assert any("shorter than five minutes" in reason for reason in reasons)


def test_resource_gate_rejects_sparse_or_off_cadence_observations() -> None:
    sparse = _windows()
    sparse[0]["observations"] = sparse[0]["observations"][:2]
    passed, reasons = evaluate_resource_gate(sparse)
    assert not passed
    assert any("fewer than 21 observations" in reason for reason in reasons)

    off_cadence = _windows()
    off_cadence[1]["observations"][10]["monotonic_offset_seconds"] += 60
    passed, reasons = evaluate_resource_gate(off_cadence)
    assert not passed
    assert any("observation cadence is invalid" in reason for reason in reasons)


def test_process_inventory_records_uid(monkeypatch) -> None:
    monkeypatch.setattr(
        "d1_reranking.resource_gate._run",
        lambda _command: (
            "12 1 501 81.5 2048 user worker --flag\n386 1 0 99.0 64 root daemon"
        ),
    )
    assert _processes() == [
        {
            "pid": 12,
            "ppid": 1,
            "uid": 501,
            "cpu_percent": 81.5,
            "rss_bytes": 2 * 1024**2,
            "command": "user worker --flag",
        },
        {
            "pid": 386,
            "ppid": 1,
            "uid": 0,
            "cpu_percent": 99.0,
            "rss_bytes": 64 * 1024,
            "command": "root daemon",
        },
    ]


def test_foreign_worker_veto_excludes_root_daemons_but_not_user_workers() -> None:
    processes = [
        {
            "pid": 10,
            "ppid": 1,
            "uid": 0,
            "cpu_percent": 99.0,
            "rss_bytes": 64 * 1024**2,
            "command": "/usr/libexec/PerfPowerServices",
        },
        {
            "pid": 11,
            "ppid": 1,
            "uid": 501,
            "cpu_percent": 81.0,
            "rss_bytes": 64 * 1024**2,
            "command": "python foreign_worker.py",
        },
        {
            "pid": 12,
            "ppid": 1,
            "uid": 501,
            "cpu_percent": 1.0,
            "rss_bytes": 2 * 1024**3,
            "command": "python memory_worker.py",
        },
        {
            "pid": 13,
            "ppid": 1,
            "uid": 501,
            "cpu_percent": 95.0,
            "rss_bytes": 512 * 1024**2,
            "command": (
                "/Library/Apple/System/Library/CoreServices/XProtect.app/Contents/"
                "MacOS/XProtectRemediatorAdload"
            ),
        },
        {
            "pid": 14,
            "ppid": 1,
            "uid": 501,
            "cpu_percent": 95.0,
            "rss_bytes": 64 * 1024**2,
            "command": (
                "/System/Library/PrivateFrameworks/"
                "IntelligencePlatformCompute.framework/Versions/A/XPCServices/"
                "IntelligencePlatformComputeService.xpc/Contents/MacOS/"
                "IntelligencePlatformComputeService"
            ),
        },
        {
            "pid": 15,
            "ppid": 1,
            "uid": 501,
            "cpu_percent": 95.0,
            "rss_bytes": 768 * 1024**2,
            "command": (
                "/Applications/ChatGPT.app/Contents/Frameworks/"
                "Codex Framework.framework/Helpers/Codex (Renderer) --type=renderer"
            ),
        },
        {
            "pid": 16,
            "ppid": 1,
            "uid": 501,
            "cpu_percent": 95.0,
            "rss_bytes": 64 * 1024**2,
            "command": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        },
    ]
    assert [
        row["pid"]
        for row in _foreign_heavy_processes(processes, ignored=set(), owner_uid=501)
    ] == [11, 12, 16]
