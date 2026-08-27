from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from graspnet6d.staged_disk import (
    ArchiveInspectionError,
    CleanupRecord,
    basic_cleanup_whitelist_rule,
    build_staged_disk_budget,
    inspect_zipinfo_archive,
    make_cleanup_candidate,
    write_cleanup_log,
    write_cleanup_report,
    write_staged_disk_budget,
)


def test_staged_budget_uses_fixed_reserve_and_single_stage_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "graspnet6d.staged_disk.shutil.disk_usage",
        lambda path: shutil._ntuple_diskusage(500_000_000_000, 0, 80_000_000_000),
    )
    budget = build_staged_disk_budget(
        tmp_path,
        phase="download train_4.zip",
        archive_bytes=7_000_000_000,
        selected_extraction_bytes=25_000_000_000,
        temporary_bytes=2_000_000_000,
        estimated_cache_bytes=5_000_000_000,
        minimum_free_reserve_gb=20,
    )
    assert budget.mandatory_reserve_bytes == 20_000_000_000
    assert budget.required_total_bytes == 59_000_000_000
    assert budget.surplus_or_deficit_bytes == 21_000_000_000
    assert budget.allowed
    assert budget.to_record()["assumptions"]["existing_files_are_not_counted_twice"]


def test_staged_budget_reports_signed_deficit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "graspnet6d.staged_disk.shutil.disk_usage",
        lambda path: shutil._ntuple_diskusage(100, 0, 25),
    )
    budget = build_staged_disk_budget(
        tmp_path,
        phase="extract",
        archive_bytes=0,
        selected_extraction_bytes=10,
        temporary_bytes=0,
        estimated_cache_bytes=0,
        minimum_free_reserve_gb=0.000000020,
    )
    assert budget.required_total_bytes == 30
    assert budget.surplus_or_deficit_bytes == -5
    assert budget.deficit_bytes == 5
    assert not budget.allowed


@pytest.mark.skipif(shutil.which("zipinfo") is None, reason="zipinfo unavailable")
def test_zipinfo_inventory_uses_exact_selected_uncompressed_sizes(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "scene.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("scenes/scene_0001/realsense/rgb/0000.png", b"a" * 11)
        handle.writestr("scenes/scene_0001/realsense/depth/0000.png", b"b" * 17)
        handle.writestr("scenes/scene_0001/kinect/rgb/0000.png", b"c" * 23)

    inventory = inspect_zipinfo_archive(
        archive, selector=lambda name: "/realsense/" in name
    )
    assert inventory.uncompressed_total_bytes == 51
    assert inventory.selected_extraction_bytes == 28
    assert inventory.member_count == 3
    assert inventory.selected_member_count == 2
    assert not inventory.unsafe_members
    assert len(inventory.zipinfo_listing_sha256) == 64


@pytest.mark.skipif(shutil.which("zipinfo") is None, reason="zipinfo unavailable")
def test_zipinfo_inventory_fails_closed_for_absent_selection(tmp_path: Path) -> None:
    archive = tmp_path / "scene.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("present.txt", b"ok")
    with pytest.raises(ArchiveInspectionError, match="selected archive members are absent"):
        inspect_zipinfo_archive(archive, selected_members={"missing.txt"})


def test_budget_audit_outputs_every_required_line_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "graspnet6d.staged_disk.shutil.disk_usage",
        lambda path: shutil._ntuple_diskusage(100, 0, 90),
    )
    budget = build_staged_disk_budget(
        tmp_path,
        phase="unit",
        archive_bytes=1,
        selected_extraction_bytes=2,
        temporary_bytes=3,
        estimated_cache_bytes=4,
        minimum_free_reserve_gb=0,
    )
    json_path = tmp_path / "staged_disk_budget.json"
    md_path = tmp_path / "staged_disk_budget.md"
    write_staged_disk_budget(budget, json_path=json_path, markdown_path=md_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))["budget"]
    for field in (
        "current_free_bytes",
        "archive_bytes",
        "selected_extraction_bytes",
        "temporary_bytes",
        "estimated_cache_bytes",
        "mandatory_reserve_bytes",
        "required_total_bytes",
        "surplus_or_deficit_bytes",
    ):
        assert field in payload
    markdown = md_path.read_text(encoding="utf-8")
    assert "Current free" in markdown
    assert "Selected extraction" in markdown
    assert "Mandatory reserve" in markdown
    assert "Surplus or deficit" in markdown


def test_basic_cleanup_whitelist_never_authorizes_personal_or_model_caches(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    home = tmp_path / "home"
    assert (
        basic_cleanup_whitelist_rule(
            repo / "downloads/graspnet/train_4.zip.part",
            repository_root=repo,
            home=home,
        )
        == "project_partial_download"
    )
    assert (
        basic_cleanup_whitelist_rule(
            repo / "data_external/graspnet/staging/chunk",
            repository_root=repo,
            home=home,
        )
        == "project_staging"
    )
    assert (
        basic_cleanup_whitelist_rule(
            home / "Library/Developer/Xcode/DerivedData/build",
            repository_root=repo,
            home=home,
        )
        == "xcode_derived_data_content"
    )
    assert (
        basic_cleanup_whitelist_rule(
            home / "Documents/paper.tex", repository_root=repo, home=home
        )
        is None
    )
    assert (
        basic_cleanup_whitelist_rule(
            home / ".cache/huggingface/model.bin",
            repository_root=repo,
            home=home,
        )
        is None
    )


def test_cleanup_log_and_report_have_mandatory_audit_fields(tmp_path: Path) -> None:
    target = tmp_path / "rebuildable.tmp"
    target.write_bytes(b"1234")
    candidate = make_cleanup_candidate(
        target,
        cleanup_type="partial_download",
        reason="interrupted official archive transfer",
        command=("rm", "-f", "--", str(target)),
        whitelist_rule="project_partial_download",
    )
    record = CleanupRecord.from_candidate(
        candidate,
        result="deleted",
        free_space_before=100,
        free_space_after=104,
        timestamp="2026-08-18T00:00:00+00:00",
    )
    csv_path = tmp_path / "disk_cleanup_log.csv"
    md_path = tmp_path / "disk_cleanup_report.md"
    write_cleanup_log(csv_path, [record])
    write_cleanup_report(
        md_path, [record], initial_free_bytes=100, final_free_bytes=104
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["path"] == str(target.resolve())
    assert row["size_before_bytes"] == "4"
    assert row["reconstructable"] == "True"
    assert "Personal files touched: **No**" in md_path.read_text(encoding="utf-8")
