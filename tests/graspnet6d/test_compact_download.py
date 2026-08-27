from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from graspnet6d.compact_download import (
    OFFICIAL_ARCHIVES,
    ArchiveValidationError,
    BaiduManualArchiveError,
    DownloadExhaustedError,
    ExtractionPlanError,
    OfficialArchive,
    adopt_baidu_authenticated_archive,
    build_collision_extraction_plan,
    build_subtree_extraction_plan,
    build_training_extraction_plan,
    deterministic_frame_ids,
    download_archive,
    extract_selective,
    inspect_archive,
    inspect_baidu_manual_archive,
    parse_zipinfo_filelist,
    plan_verified_archive_deletion,
    verify_archive,
    write_inventory_manifest,
)


def _write_training_zip(path: Path) -> None:
    prefix = "train_4/scenes/scene_0100"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{prefix}/object_id_list.txt", "1\n7\n")
        archive.writestr(f"{prefix}/rs_wrt_kn.npy", b"rs")
        for camera in ("kinect", "realsense"):
            archive.writestr(f"{prefix}/{camera}/camK.npy", camera.encode())
            archive.writestr(f"{prefix}/{camera}/camera_poses.npy", b"poses")
            archive.writestr(f"{prefix}/{camera}/cam0_wrt_table.npy", b"table")
            for frame_id in (0, 1, 255):
                for channel, extension in {
                    "rgb": "png",
                    "depth": "png",
                    "label": "png",
                    "meta": "mat",
                    "annotations": "xml",
                }.items():
                    archive.writestr(
                        f"{prefix}/{camera}/{channel}/{frame_id:04d}.{extension}",
                        f"{camera}:{channel}:{frame_id}".encode(),
                    )
            archive.writestr(f"{prefix}/{camera}/rect/0000.npy", b"excluded")


def _write_auxiliary_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "payload/collision_label/scene_0100/collision_labels.npz", b"c100"
        )
        archive.writestr(
            "payload/collision_label/scene_0101/collision_labels.npz", b"c101"
        )
        archive.writestr("payload/models/000/nontextured.ply", b"model")
        archive.writestr("payload/grasp_label/000_labels.npz", b"grasp")


def _tiny_spec(filename: str = "fixture.zip") -> OfficialArchive:
    return OfficialArchive(
        filename=filename,
        google_drive_id="official-id",
        jbox_url="https://jbox.sjtu.edu.cn/l/official",
        published_size_label="fixture",
        minimum_plausible_bytes=1,
        baidu_url="https://pan.baidu.com/s/official-fixture",
    )


def _real_tool_or_download_runner(
    download_handler,
):
    def runner(argv):
        if argv[0] in {"/usr/bin/file", "/usr/bin/unzip"}:
            return subprocess.run(argv, capture_output=True, text=True, check=False)
        return download_handler(argv)

    return runner


def test_official_sources_are_exactly_the_protocol_sources() -> None:
    assert OFFICIAL_ARCHIVES["train_4.zip"].google_drive_id == (
        "1e8Xy7-lFhiXk0ugPOKvHKDiGTparmx00"
    )
    assert OFFICIAL_ARCHIVES["train_3.zip"].jbox_url == (
        "https://jbox.sjtu.edu.cn/l/wJorXZ"
    )
    assert OFFICIAL_ARCHIVES["train_2.zip"].google_drive_id == (
        "1b1Z1goPV0o_wdwXZ8qTlHd2TBRU5-CmH"
    )
    assert OFFICIAL_ARCHIVES["grasp_label.zip"].jbox_url == (
        "https://jbox.sjtu.edu.cn/l/noXqUa"
    )
    assert OFFICIAL_ARCHIVES["collision_label.zip"].google_drive_id == (
        "1p43sntiN9HJZRDFDNpzaEaEYoPY6IWsu"
    )
    assert OFFICIAL_ARCHIVES["models.zip"].jbox_url == (
        "https://jbox.sjtu.edu.cn/l/jFF3no"
    )
    assert OFFICIAL_ARCHIVES["dex_models.zip"].jbox_url is None
    assert OFFICIAL_ARCHIVES["train_4.zip"].baidu_url == (
        "https://pan.baidu.com/s/1A3Tyc7l_u9UwgKqhVJSrNg"
    )
    assert OFFICIAL_ARCHIVES["train_3.zip"].baidu_url == (
        "https://pan.baidu.com/s/1D9Mq6VsIHE7Pra8QplEWkQ"
    )
    assert OFFICIAL_ARCHIVES["train_2.zip"].baidu_url == (
        "https://pan.baidu.com/s/158mOzU6bx4cvexQx5FXn_A"
    )
    assert OFFICIAL_ARCHIVES["grasp_label.zip"].baidu_url == (
        "https://pan.baidu.com/s/18yLPWIwM9uJBih6GMoQRNg"
    )
    assert OFFICIAL_ARCHIVES["collision_label.zip"].baidu_url == (
        "https://pan.baidu.com/s/1cj3Wea0RtgHrLb4iGUyA1g"
    )
    assert OFFICIAL_ARCHIVES["models.zip"].baidu_url == (
        "https://pan.baidu.com/s/1SoaE_7AqfR5R6w8dO79rsg"
    )
    assert OFFICIAL_ARCHIVES["dex_models.zip"].baidu_url == (
        "https://pan.baidu.com/s/1KTPJMAayVQkgx2uwUCNMOQ"
    )


def test_deterministic_frames_match_required_numpy_linspace() -> None:
    assert deterministic_frame_ids() == (
        0,
        17,
        34,
        51,
        68,
        85,
        102,
        119,
        136,
        153,
        170,
        187,
        204,
        221,
        238,
        255,
    )
    assert deterministic_frame_ids(1) == (0,)


def test_baidu_missing_archive_reports_authentication_without_network(
    tmp_path: Path,
) -> None:
    def forbidden(_argv):
        raise AssertionError("Baidu manual-state inspection must not access a network")

    state = inspect_baidu_manual_archive(
        _tiny_spec(),
        tmp_path,
        runner=forbidden,
        allow_unregistered_spec=True,
    )
    assert state.status == "AUTHENTICATION_REQUIRED"
    assert state.expected_path == str((tmp_path / "fixture.zip").resolve())
    assert state.automated_access_attempted is False
    assert list(tmp_path.iterdir()) == [], "read-only inspection must create no files"


def test_baidu_html_auth_response_is_rejected_and_never_manifested(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "fixture.zip"
    archive_path.write_text(
        "<!doctype html><html>login or captcha</html>", encoding="utf-8"
    )

    state = inspect_baidu_manual_archive(
        _tiny_spec(), tmp_path, allow_unregistered_spec=True
    )
    assert state.status == "AUTH_RESPONSE_REJECTED"
    assert "not a ZIP" in state.detail
    original = archive_path.read_bytes()
    with pytest.raises(BaiduManualArchiveError) as raised:
        adopt_baidu_authenticated_archive(
            _tiny_spec(),
            tmp_path,
            authenticated_download_confirmed=True,
            allow_unregistered_spec=True,
        )
    assert raised.value.state.status == "AUTH_RESPONSE_REJECTED"
    assert "captcha" in str(raised.value)
    assert archive_path.read_bytes() == original
    assert not (tmp_path / "download_manifest.json").exists()
    assert not (tmp_path / "archive_checksums.sha256").exists()
    log = json.loads(
        (tmp_path / "download_log.jsonl").read_text(encoding="utf-8").strip()
    )
    assert log["status"] == "AUTH_RESPONSE_REJECTED"
    assert log["automated_access_attempted"] is False
    assert log["command_argv"] == []


def test_baidu_exact_name_valid_zip_is_adopted_without_redownload(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "fixture.zip"
    _write_training_zip(archive_path)
    commands: list[tuple[str, ...]] = []

    def local_verification_only(argv):
        commands.append(tuple(argv))
        assert argv[0] in {"/usr/bin/file", "/usr/bin/unzip"}
        return subprocess.run(argv, capture_output=True, text=True, check=False)

    verification = adopt_baidu_authenticated_archive(
        _tiny_spec(),
        tmp_path,
        authenticated_download_confirmed=True,
        runner=local_verification_only,
        allow_unregistered_spec=True,
    )
    assert verification.zip_test_passed
    assert [command[0] for command in commands] == [
        "/usr/bin/file",
        "/usr/bin/unzip",
    ]
    manifest = json.loads(
        (tmp_path / "download_manifest.json").read_text(encoding="utf-8")
    )
    record = manifest["archives"][0]
    assert record["filename"] == "fixture.zip"
    assert record["source"] == "official_baidu_manual_authenticated"
    assert record["source_file_id"] == "official-fixture"
    assert record["authenticated_download_confirmed"] is True
    assert record["automated_access_attempted"] is False
    assert record["sha256"] in (tmp_path / "archive_checksums.sha256").read_text(
        encoding="utf-8"
    )
    log = json.loads(
        (tmp_path / "download_log.jsonl").read_text(encoding="utf-8").strip()
    )
    assert log["status"] == "COMPLETE"
    assert log["attempt"] == 0
    assert log["command_argv"] == []
    assert archive_path.exists()


def test_baidu_adoption_requires_explicit_provenance_confirmation(
    tmp_path: Path,
) -> None:
    _write_training_zip(tmp_path / "fixture.zip")
    with pytest.raises(ValueError, match="authenticated_download_confirmed"):
        adopt_baidu_authenticated_archive(
            _tiny_spec(),
            tmp_path,
            authenticated_download_confirmed=False,
            allow_unregistered_spec=True,
        )
    assert not (tmp_path / "download_manifest.json").exists()


def test_baidu_adoption_checks_only_the_exact_protocol_filename(
    tmp_path: Path,
) -> None:
    _write_training_zip(tmp_path / "renamed.zip")
    state = inspect_baidu_manual_archive(
        _tiny_spec(), tmp_path, allow_unregistered_spec=True
    )
    assert state.status == "AUTHENTICATION_REQUIRED"
    assert state.exists is False


def test_verify_archive_rejects_html_and_runs_independent_zip_checks(
    tmp_path: Path,
) -> None:
    html = tmp_path / "login.zip"
    html.write_text("<!doctype html><html>login</html>", encoding="utf-8")
    with pytest.raises(ArchiveValidationError, match="HTML/XML"):
        verify_archive(html)

    archive_path = tmp_path / "valid.zip"
    _write_training_zip(archive_path)
    result = verify_archive(archive_path)
    assert result.zip_test_passed
    assert result.member_count > 0
    assert result.uncompressed_member_bytes > 0
    assert len(result.sha256) == 64


def test_inspect_plan_and_extract_only_selected_camera_and_frames(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "train_4.zip"
    _write_training_zip(archive_path)
    filelist = tmp_path / "train_4.zip.filelist.txt"
    inventory = inspect_archive(archive_path, filelist_path=filelist)
    assert [scene.scene_id for scene in inventory.scenes] == [100]
    scene = inventory.scenes[0]
    assert scene.cameras == ("kinect", "realsense")
    assert scene.object_ids == (1, 7)
    assert scene.frame_ids_by_camera["realsense"] == (0, 1, 255)
    assert "train_4/scenes/scene_0100" in filelist.read_text(encoding="utf-8")

    inventory_manifest = write_inventory_manifest(
        tmp_path / "inventory.json", [inventory]
    )
    payload = json.loads(inventory_manifest.read_text(encoding="utf-8"))
    assert payload["unique_scene_count"] == 1
    assert payload["duplicate_scene_ids"] == []

    plan = build_training_extraction_plan(
        inventory, [100], camera="realsense", frame_ids=[0, 255]
    )
    selected = {item.destination_relative for item in plan.items}
    assert len(selected) == 15
    assert "scenes/scene_0100/realsense/rgb/0000.png" in selected
    assert "scenes/scene_0100/realsense/meta/0255.mat" in selected
    assert not any("kinect" in path or "/rect/" in path for path in selected)
    assert not any("0001" in path for path in selected)

    destination = tmp_path / "compact"
    manifest = tmp_path / "compact_manifest.json"
    result = extract_selective(
        plan,
        destination,
        staging_root=tmp_path / "staging",
        manifest_path=manifest,
    )
    assert result.complete
    assert result.extracted_file_count == result.selected_file_count == 15
    assert (
        destination / "scenes/scene_0100/realsense/rgb/0000.png"
    ).read_bytes() == b"realsense:rgb:0"
    assert not (destination / "scenes/scene_0100/kinect").exists()
    assert not (destination / "scenes/scene_0100/realsense/rect").exists()

    resumed = extract_selective(
        plan,
        destination,
        staging_root=tmp_path / "staging",
        manifest_path=manifest,
    )
    assert resumed.extracted_file_count == 0
    assert resumed.resumed_file_count == 15

    verification = verify_archive(archive_path)
    blocked = plan_verified_archive_deletion(
        verification, resumed, downstream_loader_verified=False
    )
    assert not blocked.eligible
    assert "downstream loader" in blocked.reason
    eligible = plan_verified_archive_deletion(
        verification, resumed, downstream_loader_verified=True
    )
    assert eligible.eligible
    assert eligible.command_argv == ("rm", "--", str(archive_path.resolve()))
    assert archive_path.exists(), "planning must never execute the deletion"


def test_auxiliary_plans_select_scenes_or_full_official_subtree(tmp_path: Path) -> None:
    archive_path = tmp_path / "aux.zip"
    _write_auxiliary_zip(archive_path)
    inventory = inspect_archive(archive_path)
    collision = build_collision_extraction_plan(inventory, [101])
    assert [item.destination_relative for item in collision.items] == [
        "collision_label/scene_0101/collision_labels.npz"
    ]
    models = build_subtree_extraction_plan(inventory, "models")
    assert [item.destination_relative for item in models.items] == [
        "models/000/nontextured.ply"
    ]
    with pytest.raises(ExtractionPlanError, match="collision labels absent"):
        build_collision_extraction_plan(inventory, [99])


def test_subtree_plan_can_select_only_locked_object_ids(tmp_path: Path) -> None:
    archive_path = tmp_path / "objects.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for object_id in (0, 1, 7):
            archive.writestr(
                f"payload/models/{object_id:03d}/nontextured.ply", b"ply\n"
            )
            archive.writestr(
                f"payload/models/{object_id:03d}/textured.obj", b"v 0 0 0\n"
            )
    inventory = inspect_archive(archive_path)
    plan = build_subtree_extraction_plan(
        inventory, "models", object_ids=(1, 7)
    )
    paths = [item.destination_relative for item in plan.items]
    assert paths == [
        "models/001/nontextured.ply",
        "models/001/textured.obj",
        "models/007/nontextured.ply",
        "models/007/textured.obj",
    ]
    assert not any("models/000" in path for path in paths)
    with pytest.raises(ExtractionPlanError, match="object IDs"):
        build_subtree_extraction_plan(inventory, "models", object_ids=(2,))


@pytest.mark.parametrize("count", [16, 24, 32])
def test_train3_frame_sampling_is_exact_numpy_floor_contract(count: int) -> None:
    import numpy as np

    expected = tuple(np.linspace(0, 255, num=count, dtype=int).tolist())
    assert deterministic_frame_ids(count) == expected


def test_archive_member_traversal_is_rejected_before_planning(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../scenes/scene_0000/object_id_list.txt", "1")
    with pytest.raises(ArchiveValidationError, match="unsafe ZIP member"):
        inspect_archive(archive_path)
    with pytest.raises(ArchiveValidationError, match="unsafe ZIP member"):
        verify_archive(archive_path)


def test_download_rotates_sources_quarantines_html_and_commits_evidence(
    tmp_path: Path,
) -> None:
    download_root = tmp_path / "downloads"
    valid = tmp_path / "source.zip"
    _write_training_zip(valid)
    calls: list[tuple[str, ...]] = []

    def download_handler(argv):
        calls.append(tuple(argv))
        destination = download_root / "fixture.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            destination.write_text("<html>Drive quota</html>", encoding="utf-8")
        else:
            destination.write_bytes(valid.read_bytes())
        return subprocess.CompletedProcess(argv, 0, "", "")

    waits: list[float] = []
    verification = download_archive(
        _tiny_spec(),
        download_root,
        maximum_attempts=3,
        runner=_real_tool_or_download_runner(download_handler),
        sleeper=waits.append,
        allow_unregistered_spec=True,
    )
    assert verification.zip_test_passed
    assert waits == [30.0]
    assert "gdown" in calls[0]
    assert "https://drive.google.com/uc?id=official-id" in calls[0]
    assert calls[1][0] == "/usr/bin/curl"
    assert "https://jbox.sjtu.edu.cn/l/official" in calls[1]
    assert not any("pan.baidu.com" in argument for call in calls for argument in call)
    assert (download_root / "fixture.zip.invalid_html.attempt_01").is_file()

    manifest = json.loads(
        (download_root / "download_manifest.json").read_text(encoding="utf-8")
    )
    record = manifest["archives"][0]
    assert record["source"] == "jbox"
    assert record["retry_count"] == 1
    assert record["zip_test_passed"] is True
    assert record["bytes"] == valid.stat().st_size
    assert record["sha256"] in (download_root / "archive_checksums.sha256").read_text(
        encoding="utf-8"
    )
    attempts = [
        json.loads(line)
        for line in (download_root / "download_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["status"] for item in attempts] == [
        "RETRYABLE_FAILURE",
        "COMPLETE",
    ]


def test_download_rejects_unregistered_sources_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="protocol-locked official archives"):
        download_archive(_tiny_spec(), tmp_path, maximum_attempts=1)


def test_failed_transfer_keeps_partial_and_next_attempt_uses_resume(
    tmp_path: Path,
) -> None:
    download_root = tmp_path / "downloads"
    valid = tmp_path / "source.zip"
    _write_training_zip(valid)
    calls = 0

    def download_handler(argv):
        nonlocal calls
        calls += 1
        destination = download_root / "fixture.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if calls == 1:
            destination.write_bytes(valid.read_bytes()[:20])
            return subprocess.CompletedProcess(argv, 18, "", "interrupted")
        assert destination.stat().st_size == 20
        destination.write_bytes(valid.read_bytes())
        return subprocess.CompletedProcess(argv, 0, "", "")

    verification = download_archive(
        _tiny_spec(),
        download_root,
        maximum_attempts=2,
        runner=_real_tool_or_download_runner(download_handler),
        sleeper=lambda _seconds: None,
        allow_unregistered_spec=True,
    )
    assert verification.zip_test_passed
    manifest = json.loads(
        (download_root / "download_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["archives"][0]["resume_used"] is True
    attempts = [
        json.loads(line)
        for line in (download_root / "download_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert attempts[0]["bytes_present"] == 20
    assert attempts[1]["resume_used"] is True


def test_download_exhaustion_is_bounded_and_retains_partial(tmp_path: Path) -> None:
    download_root = tmp_path / "downloads"

    def fail(argv):
        destination = download_root / "fixture.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(b"partial download")
        return subprocess.CompletedProcess(argv, 1, "", "quota")

    waits: list[float] = []
    with pytest.raises(DownloadExhaustedError, match="after 3 attempts"):
        download_archive(
            _tiny_spec(),
            download_root,
            maximum_attempts=3,
            runner=_real_tool_or_download_runner(fail),
            sleeper=waits.append,
            allow_unregistered_spec=True,
        )
    assert waits == [30.0, 60.0]
    assert (download_root / "fixture.zip").read_bytes() == b"partial download"
    attempts = (download_root / "download_log.jsonl").read_text(encoding="utf-8")
    assert attempts.count("RETRYABLE_FAILURE") == 3


def test_training_plan_fails_closed_on_incomplete_frame_modalities(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "incomplete.zip"
    prefix = "scene_0001"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{prefix}/object_id_list.txt", "1")
        archive.writestr(f"{prefix}/rs_wrt_kn.npy", b"rs")
        archive.writestr(f"{prefix}/realsense/camK.npy", b"K")
        archive.writestr(f"{prefix}/realsense/camera_poses.npy", b"poses")
        archive.writestr(f"{prefix}/realsense/cam0_wrt_table.npy", b"table")
        archive.writestr(f"{prefix}/realsense/rgb/0000.png", b"rgb")
    inventory = inspect_archive(archive_path)
    with pytest.raises(ExtractionPlanError, match="lacks complete selected frames"):
        build_training_extraction_plan(
            inventory, [1], camera="realsense", frame_ids=[0]
        )


def test_precomputed_zipinfo_filelist_is_parsed_and_cross_checked(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "train_4.zip"
    _write_training_zip(archive_path)
    output = subprocess.run(
        ["/usr/bin/zipinfo", "-1", str(archive_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    filelist = tmp_path / "train_4.zip.filelist.txt"
    filelist.write_text(output, encoding="utf-8")
    members = parse_zipinfo_filelist(filelist)
    inventory = inspect_archive(archive_path, filelist_path=filelist)
    assert members == inventory.members

    filelist.write_text(output + "unexpected/member\n", encoding="utf-8")
    with pytest.raises(ArchiveValidationError, match="does not match"):
        inspect_archive(archive_path, filelist_path=filelist)
