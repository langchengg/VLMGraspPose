from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from failure_analysis.reranking_v3.artifacts import write_sha_sidecar
from failure_analysis.reranking_v3.protocol import (
    assert_lockcheck_complete,
    claim_stage_once,
    complete_stage_once,
    verify_locked_manifest,
    write_locked_manifest,
)
from failure_analysis.reranking_v3.schema import (
    artifact_identity,
    canonical_json,
    sha256_bytes,
)


@pytest.fixture(autouse=True)
def fixed_code_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "failure_analysis.reranking_v3.protocol.code_fingerprint", lambda: "code-v1"
    )


def locked_manifest(tmp_path: Path, *, name: str = "frozen.json") -> Path:
    artifact = tmp_path / f"{name}.artifact"
    artifact.write_text("immutable input\n", encoding="utf-8")
    manifest = tmp_path / name
    write_locked_manifest(
        manifest,
        {
            "code_fingerprint": "code-v1",
            "locked_artifacts": [artifact_identity(artifact)],
        },
        kind="frozen_experiment_manifest",
    )
    return manifest


@pytest.mark.parametrize(
    "reserved",
    ["schema_version", "kind", "status", "locked_at", "content_sha256"],
)
def test_write_locked_manifest_rejects_reserved_payload_fields(
    tmp_path: Path, reserved: str
) -> None:
    target = tmp_path / "frozen.json"
    with pytest.raises(ValueError, match="reserved fields"):
        write_locked_manifest(
            target,
            {reserved: "attacker", "code_fingerprint": "code-v1"},
            kind="frozen_experiment_manifest",
        )
    assert not target.exists()
    assert not target.with_suffix(".json.sha256").exists()


def test_verify_locked_manifest_rejects_manifest_and_artifact_tamper(
    tmp_path: Path,
) -> None:
    manifest = locked_manifest(tmp_path)
    assert verify_locked_manifest(
        manifest, expected_kind="frozen_experiment_manifest"
    )["status"] == "locked"

    artifact = Path(json.loads(manifest.read_text())["locked_artifacts"][0]["path"])
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_locked_manifest(manifest)

    artifact.write_text("immutable input\n", encoding="utf-8")
    value = json.loads(manifest.read_text())
    value["status"] = "draft"
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = sha256_bytes(canonical_json(unsigned).encode())
    manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    manifest.with_suffix(".json.sha256").unlink()
    write_sha_sidecar(manifest)
    with pytest.raises(ValueError, match="status"):
        verify_locked_manifest(manifest)


def test_verify_locked_manifest_checks_sidecar_kind_and_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = locked_manifest(tmp_path)
    with pytest.raises(ValueError, match="kind mismatch"):
        verify_locked_manifest(manifest, expected_kind="preliminary_manifest")

    monkeypatch.setattr(
        "failure_analysis.reranking_v3.protocol.code_fingerprint", lambda: "code-v2"
    )
    with pytest.raises(ValueError, match="code changed"):
        verify_locked_manifest(manifest)

    monkeypatch.setattr(
        "failure_analysis.reranking_v3.protocol.code_fingerprint", lambda: "code-v1"
    )
    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_locked_manifest(manifest)


def test_claim_stage_once_is_atomic_under_concurrency(tmp_path: Path) -> None:
    manifest = locked_manifest(tmp_path)
    stage_dir = tmp_path / "stage"
    barrier = Barrier(2)

    def claim() -> str:
        barrier.wait()
        try:
            claim_stage_once(stage_dir, stage="lockcheck", manifest_path=manifest)
            return "claimed"
        except FileExistsError:
            return "already_claimed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sorted(results) == ["already_claimed", "claimed"]
    claim_value = json.loads(
        (stage_dir / "LOCKCHECK_RUN_CLAIM.json").read_text(encoding="utf-8")
    )
    assert len(claim_value["claim_token"]) == 64
    assert claim_value["manifest_file_sha256"]


def test_resume_requires_the_exact_claim_manifest(tmp_path: Path) -> None:
    first = locked_manifest(tmp_path, name="first.json")
    second = locked_manifest(tmp_path, name="second.json")
    stage_dir = tmp_path / "stage"
    claim = claim_stage_once(stage_dir, stage="lockcheck", manifest_path=first)
    assert claim_stage_once(
        stage_dir, stage="lockcheck", manifest_path=first, resume=True
    ) == claim
    with pytest.raises(ValueError, match="manifest_path mismatch"):
        claim_stage_once(
            stage_dir, stage="lockcheck", manifest_path=second, resume=True
        )


def test_complete_stage_once_rejects_empty_or_tampered_results(tmp_path: Path) -> None:
    manifest = locked_manifest(tmp_path)
    stage_dir = tmp_path / "stage"
    claim_stage_once(stage_dir, stage="lockcheck", manifest_path=manifest)
    with pytest.raises(ValueError, match="at least one result artifact"):
        complete_stage_once(stage_dir, stage="lockcheck", result_artifacts=[])
    with pytest.raises(ValueError, match="at least one result artifact"):
        complete_stage_once(
            stage_dir, stage="lockcheck", result_artifacts=(value for value in ())
        )

    result = tmp_path / "result.json"
    result.write_text("result\n", encoding="utf-8")
    identity = artifact_identity(result)
    result.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        complete_stage_once(
            stage_dir, stage="lockcheck", result_artifacts=[identity]
        )
    assert not (stage_dir / "LOCKCHECK_RUN_COMPLETE.json").exists()


def test_complete_stage_once_rejects_claim_tamper(tmp_path: Path) -> None:
    manifest = locked_manifest(tmp_path)
    stage_dir = tmp_path / "stage"
    claim = claim_stage_once(stage_dir, stage="lockcheck", manifest_path=manifest)
    value = json.loads(claim.read_text())
    value["claim_token"] = "0" * 64
    claim.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    result = tmp_path / "result.json"
    result.write_text("result\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        complete_stage_once(
            stage_dir,
            stage="lockcheck",
            result_artifacts=[artifact_identity(result)],
        )


def test_valid_lockcheck_lifecycle_and_post_completion_tamper(tmp_path: Path) -> None:
    manifest = locked_manifest(tmp_path)
    stage_dir = tmp_path / "stage"
    claim_path = claim_stage_once(
        stage_dir, stage="lockcheck", manifest_path=manifest
    )
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    result = tmp_path / "result.json"
    result.write_text("result\n", encoding="utf-8")
    completion_path = complete_stage_once(
        stage_dir,
        stage="lockcheck",
        result_artifacts=[artifact_identity(result)],
    )
    completion = assert_lockcheck_complete(completion_path)
    assert completion["claim_token"] == claim["claim_token"]
    assert completion["manifest_file_sha256"] == claim["manifest_file_sha256"]
    assert completion_path.with_suffix(".json.sha256").is_file()

    with pytest.raises(FileExistsError, match="already complete"):
        complete_stage_once(
            stage_dir,
            stage="lockcheck",
            result_artifacts=[artifact_identity(result)],
        )
    with pytest.raises(FileExistsError, match="already completed"):
        claim_stage_once(
            stage_dir, stage="lockcheck", manifest_path=manifest, resume=True
        )

    result.write_text("changed after completion\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        assert_lockcheck_complete(completion_path)


def test_assert_lockcheck_complete_rejects_completion_tamper(tmp_path: Path) -> None:
    manifest = locked_manifest(tmp_path)
    stage_dir = tmp_path / "stage"
    claim_stage_once(stage_dir, stage="lockcheck", manifest_path=manifest)
    result = tmp_path / "result.json"
    result.write_text("result\n", encoding="utf-8")
    completion = complete_stage_once(
        stage_dir, stage="lockcheck", result_artifacts=[artifact_identity(result)]
    )
    value = json.loads(completion.read_text())
    value["claim_token"] = "0" * 64
    completion.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        assert_lockcheck_complete(completion)
