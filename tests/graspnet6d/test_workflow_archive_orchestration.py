from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from graspnet6d.compact_download import ArchiveVerification
from graspnet6d.workflow import (
    ArchiveState,
    WorkflowBlocked,
    _download_one,
    _profile_contract,
    _split_payload,
    download,
    orchestrate_archive_downloads,
    require_verified_archives,
)


def _verification(root: Path, filename: str) -> ArchiveVerification:
    return ArchiveVerification(
        path=str((root / "downloads/graspnet" / filename).resolve()),
        bytes=100,
        sha256=(filename.encode("utf-8").hex() + "0" * 64)[:64],
        member_count=1,
        compressed_member_bytes=10,
        uncompressed_member_bytes=20,
        file_description="Zip archive data",
        file_type_passed=True,
        unzip_test_passed=True,
        python_zip_test_passed=True,
    )


def _scenes(count: int, *, offset: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        scenes=tuple(SimpleNamespace(scene_id=offset + index) for index in range(count))
    )


def test_train3_profile_contract_is_exact_and_never_requests_train4() -> None:
    contract = _profile_contract("paper-lite-train3")
    assert contract.required_scene_archives == ("train_3.zip",)
    assert contract.required_archives == (
        "train_3.zip",
        "grasp_label.zip",
        "collision_label.zip",
        "models.zip",
    )
    assert contract.split_counts == (18, 5, 7)
    assert contract.minimum_unique_scenes == 30
    assert contract.camera == "kinect"
    assert "train_4.zip" not in contract.requested_archives
    assert "train_2.zip" not in contract.requested_archives


def test_train3_profile_accepts_verified_train3_and_marks_train4_not_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_committed_adoption(monkeypatch)
    calls: list[str] = []

    def fake_download(root: Path, filename: str) -> ArchiveVerification:
        calls.append(filename)
        return _verification(root, filename)

    monkeypatch.setattr("graspnet6d.workflow._download_one", fake_download)
    monkeypatch.setattr(
        "graspnet6d.workflow._inspect_training_archive",
        lambda root, verification: _scenes(30, offset=60),
    )
    result = orchestrate_archive_downloads("paper-lite-train3", root=tmp_path)
    assert calls == [
        "grasp_label.zip",
        "collision_label.zip",
        "models.zip",
        "dex_models.zip",
        "train_3.zip",
    ]
    assert result.readiness_status == "READY_FROM_VERIFIED_LOCAL_ARCHIVES"
    assert (
        result.status_by_filename["train_4.zip"].state
        is ArchiveState.NOT_REQUIRED_FOR_THIS_RUN
    )
    assert not result.status_by_filename["train_4.zip"].attempted
    assert (
        result.status_by_filename["train_2.zip"].state
        is ArchiveState.NOT_REQUIRED_FOR_THIS_RUN
    )
    require_verified_archives(
        result.statuses,
        _profile_contract("paper-lite-train3").required_archives,
        context="train3 subset",
    )


def test_train3_profile_missing_train3_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_committed_adoption(monkeypatch)

    def fake_download(root: Path, filename: str) -> ArchiveVerification:
        if filename == "train_3.zip":
            raise WorkflowBlocked("train3 unavailable")
        return _verification(root, filename)

    monkeypatch.setattr("graspnet6d.workflow._download_one", fake_download)
    result = orchestrate_archive_downloads("paper-lite-train3", root=tmp_path)
    with pytest.raises(WorkflowBlocked, match="train_3.zip=BLOCKED"):
        require_verified_archives(
            result.statuses,
            _profile_contract("paper-lite-train3").required_archives,
            context="train3 subset",
        )


def test_train3_split_allocates_exact_disjoint_18_5_7() -> None:
    scenes = tuple(
        SimpleNamespace(
            scene_id=60 + index,
            object_ids=(index % 8, (index + 3) % 11),
            has_object_id_list=True,
            has_rs_wrt_kn=True,
        )
        for index in range(30)
    )
    inventory = SimpleNamespace(
        archive_path="/verified/train_3.zip",
        scenes=scenes,
    )
    payload = _split_payload(
        [inventory], profile="paper-lite-train3", seed=20260815
    )
    train = set(payload["train"])
    validation = set(payload["validation"])
    test = set(payload["test"])
    assert (len(train), len(validation), len(test)) == (18, 5, 7)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train | validation | test) == 30
    assert payload["unused"] == []


def test_manifested_google_archive_is_reused_without_baidu_relabelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "downloads/graspnet/models.zip"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"verified-google-bytes")
    verification = _verification(tmp_path, "models.zip")
    verification = ArchiveVerification(
        **{**verification.__dict__, "bytes": len(b"verified-google-bytes")}
    )
    manifest = {
        "archives": [
            {
                "filename": "models.zip",
                "source": "google_drive",
                "bytes": verification.bytes,
                "sha256": verification.sha256,
                "zip_test_passed": True,
            }
        ]
    }
    (destination.parent / "download_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr("graspnet6d.workflow.verify_archive", lambda *a, **k: verification)
    monkeypatch.setattr(
        "graspnet6d.workflow.inspect_baidu_manual_archive",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("manifested Google archive must not enter Baidu adoption")
        ),
    )
    assert _download_one(tmp_path, "models.zip") is verification
    saved = json.loads(
        (destination.parent / "download_manifest.json").read_text(encoding="utf-8")
    )
    assert saved["archives"][0]["source"] == "google_drive"


def _disable_committed_adoption(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "graspnet6d.workflow._validate_committed_extraction",
        lambda root, filename: None,
    )


def test_paper_download_attempts_every_independent_archive_before_aggregate_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_committed_adoption(monkeypatch)
    calls: list[str] = []

    def fake_download(
        root: Path, filename: str, *, authenticated_baidu_archive: bool = False
    ) -> ArchiveVerification:
        assert not authenticated_baidu_archive
        calls.append(filename)
        if filename in {"train_4.zip", "grasp_label.zip"}:
            raise WorkflowBlocked(
                f"WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD: {filename}"
            )
        return _verification(root, filename)

    def fake_inventory(root: Path, verification: ArchiveVerification) -> object:
        del root
        filename = Path(verification.path).name
        if filename == "train_3.zip":
            return _scenes(10)
        raise AssertionError(filename)

    monkeypatch.setattr("graspnet6d.workflow._download_one", fake_download)
    monkeypatch.setattr("graspnet6d.workflow._inspect_training_archive", fake_inventory)
    run_dir = tmp_path / "artifacts/graspnet6d/same-run"
    result = orchestrate_archive_downloads("paper-lite", root=tmp_path, run_dir=run_dir)

    assert calls == [
        "grasp_label.zip",
        "collision_label.zip",
        "models.zip",
        "dex_models.zip",
        "train_4.zip",
        "train_3.zip",
    ]
    states = result.status_by_filename
    assert states["train_4.zip"].state is ArchiveState.BLOCKED
    assert states["grasp_label.zip"].state is ArchiveState.BLOCKED
    assert states["train_3.zip"].state is ArchiveState.VERIFIED
    assert states["train_2.zip"].state is ArchiveState.MISSING
    assert result.unique_train_3_train_4_scenes == 10
    assert not result.train_2_condition_met

    global_state = tmp_path / "downloads/graspnet/archive_orchestration.json"
    run_state = run_dir / "archive_download_states.json"
    assert global_state.is_file() and run_state.is_file()
    payload = json.loads(run_state.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["profile"] == "paper-lite"
    assert {row["state"] for row in payload["states"]} >= {
        "BLOCKED",
        "VERIFIED",
    }

    with pytest.raises(WorkflowBlocked) as raised:
        require_verified_archives(
            result.statuses,
            ("train_4.zip", "grasp_label.zip", "models.zip"),
            context="smoke",
        )
    message = str(raised.value)
    assert "train_4.zip=BLOCKED" in message
    assert "grasp_label.zip=BLOCKED" in message
    assert "WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD" in message


def test_smoke_gate_does_not_require_train_3_or_optional_dex_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_committed_adoption(monkeypatch)
    calls: list[str] = []

    def fake_download(
        root: Path, filename: str, *, authenticated_baidu_archive: bool = False
    ) -> ArchiveVerification:
        del authenticated_baidu_archive
        calls.append(filename)
        if filename == "dex_models.zip":
            raise RuntimeError("optional evaluator accelerator unavailable")
        return _verification(root, filename)

    monkeypatch.setattr("graspnet6d.workflow._download_one", fake_download)
    monkeypatch.setattr(
        "graspnet6d.workflow._inspect_training_archive",
        lambda root, verification: _scenes(3),
    )
    result = orchestrate_archive_downloads("smoke", root=tmp_path)
    assert "train_3.zip" not in calls
    assert result.status_by_filename["train_3.zip"].state is ArchiveState.AVAILABLE
    assert result.status_by_filename["train_2.zip"].state is ArchiveState.MISSING
    assert result.status_by_filename["dex_models.zip"].state is ArchiveState.BLOCKED
    require_verified_archives(
        result.statuses,
        (
            "train_4.zip",
            "grasp_label.zip",
            "collision_label.zip",
            "models.zip",
        ),
        context="real smoke",
    )


def test_train_2_is_conditional_on_both_prior_training_archives_being_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_committed_adoption(monkeypatch)
    calls: list[str] = []

    def fake_download(root: Path, filename: str) -> ArchiveVerification:
        calls.append(filename)
        return _verification(root, filename)

    def fake_inventory(root: Path, verification: ArchiveVerification) -> object:
        del root
        filename = Path(verification.path).name
        return {
            "train_4.zip": _scenes(10),
            "train_3.zip": _scenes(10, offset=10),
            "train_2.zip": _scenes(20, offset=100),
        }[filename]

    monkeypatch.setattr("graspnet6d.workflow._download_one", fake_download)
    monkeypatch.setattr("graspnet6d.workflow._inspect_training_archive", fake_inventory)
    result = orchestrate_archive_downloads("paper-lite", root=tmp_path)
    assert calls == [
        "grasp_label.zip",
        "collision_label.zip",
        "models.zip",
        "dex_models.zip",
        "train_4.zip",
        "train_3.zip",
        "train_2.zip",
    ]
    assert result.unique_train_3_train_4_scenes == 20
    assert result.train_2_condition_met
    assert result.status_by_filename["train_2.zip"].state is ArchiveState.VERIFIED


def test_prior_twelve_attempt_exhaustion_never_restarts_google_or_jbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "artifacts/graspnet6d/audit/download_log.jsonl"
    log.parent.mkdir(parents=True)
    rows = [
        {
            "filename": "train_4.zip",
            "source": "google_drive" if attempt % 2 else "jbox",
            "attempt": attempt,
            "status": "RETRYABLE_FAILURE",
        }
        for attempt in range(1, 13)
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    network_calls: list[str] = []
    monkeypatch.setattr(
        "graspnet6d.workflow.download_archive",
        lambda *args, **kwargs: (
            network_calls.append("network")
            or (_ for _ in ()).throw(AssertionError("network retry forbidden"))
        ),
    )
    monkeypatch.setattr(
        "graspnet6d.workflow.inspect_baidu_manual_archive",
        lambda spec, root: SimpleNamespace(
            status="AUTHENTICATION_REQUIRED",
            detail="authenticated local archive absent",
            official_baidu_url=spec.baidu_url,
            expected_path=str(Path(root) / spec.filename),
        ),
    )
    monkeypatch.setattr(
        "graspnet6d.workflow._write_budget",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("manual-wait state needs no new-space budget")
        ),
    )

    with pytest.raises(
        WorkflowBlocked, match="WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD"
    ):
        _download_one(tmp_path, "train_4.zip")
    assert network_calls == []


def test_partial_later_batch_does_not_erase_durable_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "artifacts/graspnet6d/audit/download_log.jsonl"
    log.parent.mkdir(parents=True)
    rows = [
        {
            "filename": "train_4.zip",
            "source": "google_drive" if attempt % 2 else "jbox",
            "attempt": attempt,
            "status": "RETRYABLE_FAILURE",
        }
        for attempt in range(1, 13)
    ] + [
        {
            "filename": "train_4.zip",
            "source": "google_drive" if attempt % 2 else "jbox",
            "attempt": attempt,
            "status": "RETRYABLE_FAILURE",
        }
        for attempt in range(1, 4)
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(
        "graspnet6d.workflow.download_archive",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("durably exhausted archive must not retry")
        ),
    )
    monkeypatch.setattr(
        "graspnet6d.workflow.inspect_baidu_manual_archive",
        lambda spec, root: SimpleNamespace(
            status="AUTHENTICATION_REQUIRED",
            detail="exact-name file absent",
            official_baidu_url=spec.baidu_url,
            expected_path=str(Path(root) / spec.filename),
        ),
    )
    with pytest.raises(
        WorkflowBlocked, match="WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD"
    ):
        _download_one(tmp_path, "train_4.zip")


def test_no_zip_outcome_is_durable_and_only_blocks_after_all_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_committed_adoption(monkeypatch)
    calls: list[str] = []

    def blocked(
        root: Path, filename: str, *, authenticated_baidu_archive: bool = False
    ) -> ArchiveVerification:
        del root, authenticated_baidu_archive
        calls.append(filename)
        raise WorkflowBlocked(f"WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD: {filename}")

    monkeypatch.setattr("graspnet6d.workflow._download_one", blocked)
    run_dir = tmp_path / "run"
    result = orchestrate_archive_downloads("paper-lite", root=tmp_path, run_dir=run_dir)
    assert calls == [
        "grasp_label.zip",
        "collision_label.zip",
        "models.zip",
        "dex_models.zip",
        "train_4.zip",
        "train_3.zip",
    ]
    state_path = run_dir / "archive_download_states.json"
    assert state_path.is_file()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    states = {row["filename"]: row["state"] for row in payload["states"]}
    assert all(
        states[filename] == "BLOCKED"
        for filename in (
            "grasp_label.zip",
            "collision_label.zip",
            "models.zip",
            "dex_models.zip",
            "train_4.zip",
            "train_3.zip",
        )
    )
    assert states["train_2.zip"] == "MISSING"
    with pytest.raises(
        WorkflowBlocked, match="WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD"
    ):
        require_verified_archives(
            result.statuses,
            (
                "train_4.zip",
                "grasp_label.zip",
                "collision_label.zip",
                "models.zip",
            ),
            context="compact extraction and real-data smoke",
        )


def test_exact_name_ready_archive_is_adopted_before_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "downloads/graspnet/models.zip"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"locally authenticated fixture")
    verification = _verification(tmp_path, "models.zip")
    events: list[str] = []
    monkeypatch.setattr(
        "graspnet6d.workflow.inspect_baidu_manual_archive",
        lambda spec, root: SimpleNamespace(status="READY_FOR_ADOPTION"),
    )
    monkeypatch.setattr(
        "graspnet6d.workflow.adopt_baidu_authenticated_archive",
        lambda *args, **kwargs: events.append("adopt") or verification,
    )
    monkeypatch.setattr(
        "graspnet6d.workflow.download_archive",
        lambda *args, **kwargs: (
            events.append("network")
            or (_ for _ in ()).throw(AssertionError("network must not run"))
        ),
    )
    monkeypatch.setattr("graspnet6d.workflow._write_budget", lambda *args, **kwargs: {})
    assert _download_one(tmp_path, "models.zip") is verification
    assert events == ["adopt"]


@pytest.mark.parametrize(
    "manual_status", ["AUTH_RESPONSE_REJECTED", "INVALID_LOCAL_ARCHIVE"]
)
def test_invalid_exact_name_local_file_blocks_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manual_status: str
) -> None:
    destination = tmp_path / "downloads/graspnet/models.zip"
    destination.parent.mkdir(parents=True)
    destination.write_text("<html>authentication required</html>", encoding="utf-8")
    network_calls: list[str] = []
    monkeypatch.setattr(
        "graspnet6d.workflow.inspect_baidu_manual_archive",
        lambda spec, root: SimpleNamespace(
            status=manual_status,
            detail="local exact-name file is not a verified ZIP",
            official_baidu_url=spec.baidu_url,
            expected_path=str(Path(root) / spec.filename),
        ),
    )
    monkeypatch.setattr(
        "graspnet6d.workflow.download_archive",
        lambda *args, **kwargs: network_calls.append("network"),
    )
    with pytest.raises(
        WorkflowBlocked, match="WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD"
    ):
        _download_one(tmp_path, "models.zip")
    assert network_calls == []


def test_current_call_exhaustion_is_translated_to_precise_baidu_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from graspnet6d.compact_download import DownloadExhaustedError

    monkeypatch.setattr(
        "graspnet6d.workflow.download_archive",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            DownloadExhaustedError("twelve attempts exhausted")
        ),
    )
    monkeypatch.setattr(
        "graspnet6d.workflow.inspect_baidu_manual_archive",
        lambda spec, root: SimpleNamespace(
            status="AUTHENTICATION_REQUIRED",
            detail="exact-name file absent",
            official_baidu_url=spec.baidu_url,
            expected_path=str(Path(root) / spec.filename),
        ),
    )
    monkeypatch.setattr("graspnet6d.workflow._write_budget", lambda *args, **kwargs: {})
    with pytest.raises(
        WorkflowBlocked, match="WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD"
    ):
        _download_one(tmp_path, "collision_label.zip")


def test_paper_lite_never_accepts_a_smoke_compact_manifest_as_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "data_external/graspnet/compact_dataset_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "complete": True,
                "profile": "smoke",
                "camera": "kinect",
                "frames_per_scene": 16,
                "scene_count": 35,
                "extractions": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("graspnet6d.workflow.repository_root", lambda: tmp_path)
    with pytest.raises(WorkflowBlocked, match="profile/scene-count contract"):
        download("paper-lite")
