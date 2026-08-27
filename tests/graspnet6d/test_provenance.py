from __future__ import annotations

import json
from pathlib import Path

import pytest

from graspnet6d import provenance
from graspnet6d.audit import sha256_file


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "repo"
    profile = """
profile: paper-lite
seed: 7
vgn:
  checkpoint: checkpoints/vgn.pth
grounding:
  hifics_checkpoint: checkpoints/hifics.pth
""".lstrip()
    _write(root / "configs/graspnet6d/paper_lite.yaml", profile)
    _write(
        root / "configs/graspnet6d/upstream_versions.lock",
        "sources:\n  vgn:\n    commit: abc123\n",
    )
    for relative in provenance._LOCKED_CONFIG_PATHS:
        _write(root / relative, f"content for {relative}\n")
    _write(root / "checkpoints/vgn.pth", "vgn checkpoint\n")
    _write(root / "checkpoints/hifics.pth", "hifics checkpoint\n")
    state: dict[str, object] = {"git_sha": "deadbeef", "dirty": True, "tree": "tree-a"}

    def fake_git(args: list[str]) -> str:
        if args == ["rev-parse", "HEAD"]:
            return str(state["git_sha"])
        if args == ["status", "--porcelain"]:
            return " M tracked.py" if state["dirty"] else ""
        raise AssertionError(args)

    monkeypatch.setattr(provenance, "repository_root", lambda: root)
    monkeypatch.setattr(provenance, "_git", fake_git)
    monkeypatch.setattr(provenance, "dirty_tree_hash", lambda: str(state["tree"]))
    return root, state


def test_create_and_resume_preserve_exact_immutable_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repository(tmp_path, monkeypatch)
    run_dir = provenance.create_run(
        "run-1", profile="paper-lite", command=["graspnet6d", "prepare"]
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_path = run_dir / "resolved_config.yaml"

    assert manifest["schema_version"] == provenance.RUN_MANIFEST_SCHEMA_VERSION
    assert manifest["immutable_identity"]["resolved_config"]["seed"] == 7
    assert manifest["resolved_config_sha256"] == sha256_file(resolved_path)
    assert manifest["vgn_checkpoint_hash"] == sha256_file(root / "checkpoints/vgn.pth")
    assert manifest["hifics_checkpoint_hash"] == sha256_file(
        root / "checkpoints/hifics.pth"
    )

    resumed = provenance.create_run(
        "run-1", profile="paper-lite", command=["graspnet6d", "evaluate"]
    )
    assert resumed == run_dir
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["commands_executed"] == [
        ["graspnet6d", "prepare"],
        ["graspnet6d", "evaluate"],
    ]
    assert "last_resumed_at_utc" in updated


def test_resume_rejects_changed_resolved_config_before_appending_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repository(tmp_path, monkeypatch)
    run_dir = provenance.create_run("run-1", profile="paper-lite", command=["first"])
    _write(
        root / "configs/graspnet6d/paper_lite.yaml",
        "profile: paper-lite\nseed: 8\nvgn:\n  checkpoint: checkpoints/vgn.pth\n"
        "grounding:\n  hifics_checkpoint: checkpoints/hifics.pth\n",
    )

    with pytest.raises(ValueError, match="immutable identity changed.*resolved_config"):
        provenance.create_run(
            "run-1", profile="paper-lite", command=["must-not-append"]
        )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["commands_executed"] == [["first"]]
    assert "last_resumed_at_utc" not in manifest


def test_resume_reopens_a_blocked_run_without_losing_status_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repository(tmp_path, monkeypatch)
    run_dir = provenance.create_run("run-1", profile="paper-lite", command=["first"])
    manifest_path = run_dir / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "BLOCKED",
            "blocked_stage": "candidates",
            "blocked_reason": "awaiting independent review",
            "ended_at_utc": "2026-08-16T00:00:00+00:00",
            "formal_results_emitted": False,
        }
    )
    provenance.atomic_json(manifest_path, payload)

    provenance.create_run(
        "run-1", profile="paper-lite", command=["graspnet6d", "all", "--resume"]
    )
    resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed["status"] == "IN_PROGRESS"
    assert resumed["formal_results_emitted"] is False
    assert "blocked_stage" not in resumed
    assert "blocked_reason" not in resumed
    assert "ended_at_utc" not in resumed
    assert resumed["status_history"][-1]["status"] == "BLOCKED"
    assert resumed["status_history"][-1]["blocked_stage"] == "candidates"
    assert resumed["status_history"][-1]["blocked_reason"] == (
        "awaiting independent review"
    )


def test_resume_reopens_a_complete_run_before_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _ = _repository(tmp_path, monkeypatch)
    run_dir = provenance.create_run("run-1", profile="paper-lite", command=["first"])
    manifest_path = run_dir / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "COMPLETE",
            "formal_results_emitted": True,
            "ended_at_utc": "2026-08-16T00:00:00+00:00",
            "paper_artifact_manifest_sha256": "a" * 64,
        }
    )
    provenance.atomic_json(manifest_path, payload)

    provenance.create_run(
        "run-1", profile="paper-lite", command=["graspnet6d", "all", "--resume"]
    )

    resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed["status"] == "IN_PROGRESS"
    assert resumed["formal_results_emitted"] is False
    assert "ended_at_utc" not in resumed
    assert resumed["status_history"][-1]["status"] == "COMPLETE"
    assert resumed["status_history"][-1]["ended_at_utc"] == (
        "2026-08-16T00:00:00+00:00"
    )


@pytest.mark.parametrize(
    ("relative", "replacement", "message"),
    [
        ("configs/graspnet6d/ranker.yaml", "changed: true\n", "config_file_hashes"),
        ("checkpoints/vgn.pth", "different checkpoint\n", "checkpoint_hashes"),
        (
            "configs/graspnet6d/upstream_versions.lock",
            "sources:\n  vgn:\n    commit: changed\n",
            "upstream",
        ),
    ],
)
def test_resume_rejects_changed_locked_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    replacement: str,
    message: str,
) -> None:
    root, _ = _repository(tmp_path, monkeypatch)
    provenance.create_run("run-1", profile="paper-lite", command=["first"])
    _write(root / relative, replacement)

    with pytest.raises(ValueError, match=message):
        provenance.create_run("run-1", profile="paper-lite", command=["second"])


def test_resume_rejects_changed_repository_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, state = _repository(tmp_path, monkeypatch)
    provenance.create_run("run-1", profile="paper-lite", command=["first"])
    state["tree"] = "tree-b"

    with pytest.raises(ValueError, match="repository.dirty_working_tree_sha256"):
        provenance.create_run("run-1", profile="paper-lite", command=["second"])


def test_resume_rejects_tampered_resolved_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _ = _repository(tmp_path, monkeypatch)
    run_dir = provenance.create_run("run-1", profile="paper-lite", command=["first"])
    _write(run_dir / "resolved_config.yaml", "profile: paper-lite\nseed: 999\n")

    with pytest.raises(ValueError, match="resolved config snapshot differs"):
        provenance.create_run("run-1", profile="paper-lite", command=["second"])


def test_resume_rejects_legacy_manifest_without_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repository(tmp_path, monkeypatch)
    run_dir = root / "artifacts/graspnet6d/run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"schema_version": 1, "profile": "paper-lite"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="immutable resume cannot be proven"):
        provenance.create_run("run-1", profile="paper-lite", command=["second"])
