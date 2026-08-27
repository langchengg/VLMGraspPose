from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import gtmask_counterfactual.resource as resource
from unified_reranking.hashing import canonical_sha256


DOCKER_COMMAND = "/Applications/Docker.app/Contents/MacOS/com.docker.backend services"


def _snapshot(*, foreign_command: str = DOCKER_COMMAND) -> dict[str, object]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "memory_free_percent": 75.0,
        "swap_used_bytes": 0,
        "disk_free_bytes": 200 * 1024**3,
        "load_average": [1.0, 1.0, 1.0],
        "normalized_load_5m": 0.1,
        "rank1_workers": [],
        "rank1_claim_paths": [],
        "d1_heavy_workers": [],
        "foreign_heavy_processes": [
            {
                "pid": 100,
                "ppid": 1,
                "uid": 501,
                "cpu_percent": 0.0,
                "rss_bytes": 3 * 1024**3,
                "command": foreign_command,
            }
        ],
    }


def _synthetic_gate(*, resource_scope: str) -> dict[str, object]:
    elapsed = [0.0]

    def monotonic() -> float:
        return elapsed[0]

    def sleeper(seconds: float) -> None:
        elapsed[0] += seconds

    return resource.collect_fresh_three_by_five_gate(
        repo_root=Path("/repo"),
        rank1_run_dir=Path("/rank1"),
        snapshot_collector=lambda **_: _snapshot(),
        monotonic=monotonic,
        sleeper=sleeper,
        resource_scope=resource_scope,
    )


def test_docker_scoring_gate_keeps_daemon_evidence_but_not_worker_veto() -> None:
    gate = _synthetic_gate(resource_scope=resource.DOCKER_SCORING_RESOURCE_SCOPE)

    assert gate["status"] == "PASS"
    resource.validate_fresh_gate(
        gate,
        resource_scope=resource.DOCKER_SCORING_RESOURCE_SCOPE,
    )
    observation = gate["windows"][0]["observations"][0]
    assert observation["foreign_heavy_processes"] == []
    assert observation["required_docker_infrastructure_processes"][0][
        "command"
    ] == DOCKER_COMMAND


def test_standard_gate_does_not_exclude_docker_daemon() -> None:
    gate = _synthetic_gate(resource_scope=resource.STANDARD_RESOURCE_SCOPE)

    assert gate["status"] == "FAIL"
    assert any("foreign_heavy_processes" in reason for reason in gate["failure_reasons"])


def test_docker_scoring_gate_rejects_rebound_overbroad_exception() -> None:
    gate = _synthetic_gate(resource_scope=resource.DOCKER_SCORING_RESOURCE_SCOPE)
    observation = gate["windows"][0]["observations"][0]
    observation["required_docker_infrastructure_processes"][0]["command"] = (
        "/tmp/unrelated-heavy-worker"
    )
    unsigned = dict(gate)
    unsigned.pop("content_sha256")
    gate["content_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(RuntimeError, match="over-broad"):
        resource.validate_fresh_gate(
            gate,
            resource_scope=resource.DOCKER_SCORING_RESOURCE_SCOPE,
        )


def test_live_resource_scope_only_allows_exact_docker_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resource, "collect_resource_snapshot", lambda **_: _snapshot())

    with pytest.raises(RuntimeError, match="foreign_heavy_processes"):
        resource.validate_live_resources(
            repo_root=Path("/repo"),
            rank1_run_dir=Path("/rank1"),
            prefix="standard",
        )
    scoped = resource.validate_live_resources(
        repo_root=Path("/repo"),
        rank1_run_dir=Path("/rank1"),
        prefix="docker",
        resource_scope=resource.DOCKER_SCORING_RESOURCE_SCOPE,
    )
    assert scoped["foreign_heavy_processes"] == []
    assert len(scoped["required_docker_infrastructure_processes"]) == 1
