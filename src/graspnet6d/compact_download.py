"""Resumable official GraspNet downloads and compact archive extraction.

This module deliberately separates *planning* from destructive cleanup.  It
downloads only from the official Google Drive and JBox locations, validates an
archive with both the host tools requested by the experiment protocol and
Python's ZIP reader, and can materialize a camera/frame subset without ever
calling ``extractall``.  A verified archive deletion is returned as an audit
plan; callers remain responsible for recording and executing that plan.
"""

from __future__ import annotations

import binascii
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from .io import atomic_json, atomic_text, sha256_file


OFFICIAL_DATASET_PAGE = "https://graspnet.net/datasets.html"
_GOOGLE_DOWNLOAD = "https://drive.google.com/uc?id={file_id}"
_SCENE_RE = re.compile(r"^scene_(\d{4})$")
_FRAME_RE = re.compile(r"^(\d{4})\.(png|mat|xml)$")
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_HTML_PREFIXES = (b"<!doctype html", b"<html", b"<?xml")


@dataclass(frozen=True)
class OfficialArchive:
    """One archive whose URLs are published by the GraspNet team."""

    filename: str
    google_drive_id: str
    jbox_url: str | None
    published_size_label: str
    minimum_plausible_bytes: int
    expected_bytes: int | None = None
    baidu_url: str | None = None

    @property
    def google_url(self) -> str:
        return _GOOGLE_DOWNLOAD.format(file_id=self.google_drive_id)


# Exact byte counts are retained only where the repository already recorded a
# completed official Drive size.  The website labels train_2/train_3 as 20 GB,
# so those archives use a conservative magnitude check rather than inventing
# an exact checksum-like byte count.
OFFICIAL_ARCHIVES: Mapping[str, OfficialArchive] = {
    "train_4.zip": OfficialArchive(
        "train_4.zip",
        "1e8Xy7-lFhiXk0ugPOKvHKDiGTparmx00",
        "https://jbox.sjtu.edu.cn/l/SHwJVL",
        "6.3 GB",
        5 * 1024**3,
        6_803_985_160,
        "https://pan.baidu.com/s/1A3Tyc7l_u9UwgKqhVJSrNg",
    ),
    "train_3.zip": OfficialArchive(
        "train_3.zip",
        "1oNcmZno2ymsDUWTmfFOxewMBjTXhL95c",
        "https://jbox.sjtu.edu.cn/l/wJorXZ",
        "20 GB",
        10 * 1024**3,
        baidu_url="https://pan.baidu.com/s/1D9Mq6VsIHE7Pra8QplEWkQ",
    ),
    "train_2.zip": OfficialArchive(
        "train_2.zip",
        "1b1Z1goPV0o_wdwXZ8qTlHd2TBRU5-CmH",
        "https://jbox.sjtu.edu.cn/l/G57uyS",
        "20 GB",
        10 * 1024**3,
        baidu_url="https://pan.baidu.com/s/158mOzU6bx4cvexQx5FXn_A",
    ),
    "grasp_label.zip": OfficialArchive(
        "grasp_label.zip",
        "1FCV6j2J2eQpVk_ddJXljJvjRT1KU3sJ6",
        "https://jbox.sjtu.edu.cn/l/noXqUa",
        "1.9 GB",
        1024**3,
        2_059_130_127,
        "https://pan.baidu.com/s/18yLPWIwM9uJBih6GMoQRNg",
    ),
    "collision_label.zip": OfficialArchive(
        "collision_label.zip",
        "1p43sntiN9HJZRDFDNpzaEaEYoPY6IWsu",
        "https://jbox.sjtu.edu.cn/l/DuUptQ",
        "0.4 GB",
        256 * 1024**2,
        441_783_131,
        "https://pan.baidu.com/s/1cj3Wea0RtgHrLb4iGUyA1g",
    ),
    "models.zip": OfficialArchive(
        "models.zip",
        "1Gxwu2C5wRQ0QwjdA8CbMXx-bYf_wwPT5",
        "https://jbox.sjtu.edu.cn/l/jFF3no",
        "4.3 GB",
        3 * 1024**3,
        4_599_338_858,
        "https://pan.baidu.com/s/1SoaE_7AqfR5R6w8dO79rsg",
    ),
    "dex_models.zip": OfficialArchive(
        "dex_models.zip",
        "1RElNqUHNoA9l_muTGNu7yAc3ql_e7pL3",
        None,
        "8.9 GB",
        6 * 1024**3,
        9_518_063_724,
        "https://pan.baidu.com/s/1KTPJMAayVQkgx2uwUCNMOQ",
    ),
}


class ArchiveValidationError(RuntimeError):
    """An archive is absent, implausible, unsafe, HTML, or corrupt."""


class DownloadExhaustedError(RuntimeError):
    """All bounded official-source download attempts failed."""


class ExtractionPlanError(RuntimeError):
    """A requested compact subset is absent or unsafe to materialize."""


class BaiduManualArchiveError(ArchiveValidationError):
    """Baidu requires a user-authenticated transfer before local adoption."""

    def __init__(self, state: "BaiduManualArchiveState") -> None:
        self.state = state
        super().__init__(
            f"{state.status}: {state.detail}; authenticate at "
            f"{state.official_baidu_url} and save the exact filename to "
            f"{state.expected_path}; automated login, captcha, and access-control "
            "bypass are intentionally disabled"
        )


@dataclass(frozen=True)
class ArchiveVerification:
    path: str
    bytes: int
    sha256: str
    member_count: int
    compressed_member_bytes: int
    uncompressed_member_bytes: int
    file_description: str
    file_type_passed: bool
    unzip_test_passed: bool
    python_zip_test_passed: bool

    @property
    def zip_test_passed(self) -> bool:
        return (
            self.file_type_passed
            and self.unzip_test_passed
            and self.python_zip_test_passed
        )

    def to_record(self) -> dict[str, Any]:
        return {**asdict(self), "zip_test_passed": self.zip_test_passed}


@dataclass(frozen=True)
class SceneArchiveRecord:
    scene_id: int
    archive_scene_prefix: str
    dataset_scene_path: str
    cameras: tuple[str, ...]
    frame_ids_by_camera: Mapping[str, tuple[int, ...]]
    object_ids: tuple[int, ...]
    has_object_id_list: bool
    has_rs_wrt_kn: bool

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["frame_ids_by_camera"] = {
            key: list(value) for key, value in self.frame_ids_by_camera.items()
        }
        return record


@dataclass(frozen=True)
class ArchiveInventory:
    archive_path: str
    archive_sha256: str
    archive_bytes: int
    member_count: int
    compressed_member_bytes: int
    uncompressed_member_bytes: int
    members: tuple[str, ...]
    dataset_paths: Mapping[str, str]
    member_sizes: Mapping[str, int]
    member_crc32: Mapping[str, int]
    scenes: tuple[SceneArchiveRecord, ...]
    path_adapter: str

    def to_record(self, *, include_members: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "member_count": self.member_count,
            "compressed_member_bytes": self.compressed_member_bytes,
            "uncompressed_member_bytes": self.uncompressed_member_bytes,
            "scene_ids": [scene.scene_id for scene in self.scenes],
            "scenes": [scene.to_record() for scene in self.scenes],
            "path_adapter": self.path_adapter,
        }
        if include_members:
            record["members"] = list(self.members)
            record["dataset_paths"] = dict(self.dataset_paths)
        return record


@dataclass(frozen=True)
class ExtractionItem:
    source_member: str
    destination_relative: str
    bytes: int
    crc32: int


@dataclass(frozen=True)
class ExtractionPlan:
    archive_path: str
    archive_sha256: str
    kind: Literal["training_scenes", "collision_scenes", "subtree"]
    items: tuple[ExtractionItem, ...]
    selected_scene_ids: tuple[int, ...] = ()
    camera: str | None = None
    frame_ids: tuple[int, ...] = ()
    subtree: str | None = None

    @property
    def selected_bytes(self) -> int:
        return sum(item.bytes for item in self.items)

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "selected_bytes": self.selected_bytes,
            "selected_file_count": len(self.items),
        }


@dataclass(frozen=True)
class ExtractionResult:
    archive_path: str
    archive_sha256: str
    destination_root: str
    selected_file_count: int
    selected_bytes: int
    extracted_file_count: int
    resumed_file_count: int
    verified_file_count: int
    complete: bool
    manifest_path: str | None
    manifest_sha256: str | None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveDeletionPlan:
    eligible: bool
    path: str
    size_before_bytes: int
    reason: str
    command_argv: tuple[str, ...]
    reconstructable: bool
    unmet_conditions: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaiduManualArchiveState:
    """Read-only state for a manually authenticated official Baidu archive."""

    status: Literal[
        "AUTHENTICATION_REQUIRED",
        "AUTH_RESPONSE_REJECTED",
        "INVALID_LOCAL_ARCHIVE",
        "READY_FOR_ADOPTION",
    ]
    filename: str
    official_baidu_url: str
    expected_path: str
    exists: bool
    bytes: int
    detail: str
    automated_access_attempted: bool = False
    verification: ArchiveVerification | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        if self.verification is not None:
            record["verification"] = self.verification.to_record()
        return record


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Sleeper = Callable[[float], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, check=False)


def _safe_member_name(name: str) -> str:
    if "\\" in name or "\x00" in name:
        raise ArchiveValidationError(f"unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveValidationError(f"unsafe ZIP member name: {name!r}")
    return path.as_posix()


def _looks_like_html(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        prefix = stream.read(4096).lstrip().lower()
    return any(prefix.startswith(marker) for marker in _HTML_PREFIXES) or (
        b"<html" in prefix and not prefix.startswith(_ZIP_MAGIC)
    )


def _zip_magic_passes(path: Path) -> bool:
    with path.open("rb") as stream:
        prefix = stream.read(4)
    return prefix in _ZIP_MAGIC


def verify_archive(
    path: str | os.PathLike[str],
    *,
    minimum_plausible_bytes: int = 0,
    expected_bytes: int | None = None,
    runner: CommandRunner = _run_command,
) -> ArchiveVerification:
    """Run file(1), unzip(1), ZIP CRC, size, HTML, and SHA-256 checks."""

    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ArchiveValidationError(f"expected regular archive: {source}")
    observed_bytes = source.stat().st_size
    if observed_bytes < minimum_plausible_bytes:
        raise ArchiveValidationError(
            f"archive is too small: {observed_bytes} < {minimum_plausible_bytes} bytes"
        )
    if expected_bytes is not None and observed_bytes != expected_bytes:
        raise ArchiveValidationError(
            f"archive size mismatch: {observed_bytes} != {expected_bytes} bytes"
        )
    if _looks_like_html(source):
        raise ArchiveValidationError(f"HTML/XML response rejected as archive: {source}")
    if not _zip_magic_passes(source):
        raise ArchiveValidationError(f"ZIP magic is absent: {source}")

    file_result = runner(("/usr/bin/file", "-b", str(source)))
    description = (file_result.stdout or file_result.stderr or "").strip()
    file_passed = file_result.returncode == 0 and "zip archive" in description.lower()
    if not file_passed:
        raise ArchiveValidationError(
            f"file(1) did not identify a Zip archive: {description!r}"
        )

    unzip_result = runner(("/usr/bin/unzip", "-tqq", str(source)))
    if unzip_result.returncode != 0:
        detail = (unzip_result.stderr or unzip_result.stdout or "").strip()
        raise ArchiveValidationError(f"unzip -t failed for {source}: {detail}")

    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            names: set[str] = set()
            for info in infos:
                safe_name = _safe_member_name(info.filename.rstrip("/"))
                if safe_name in names:
                    raise ArchiveValidationError(
                        f"duplicate ZIP member rejected: {safe_name}"
                    )
                names.add(safe_name)
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ArchiveValidationError(
                    f"ZIP CRC test failed at member {bad_member!r}"
                )
            compressed = sum(info.compress_size for info in infos if not info.is_dir())
            uncompressed = sum(info.file_size for info in infos if not info.is_dir())
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ArchiveValidationError(
            f"invalid ZIP archive {source}: {error}"
        ) from error

    return ArchiveVerification(
        path=str(source),
        bytes=observed_bytes,
        sha256=sha256_file(source),
        member_count=len(infos),
        compressed_member_bytes=compressed,
        uncompressed_member_bytes=uncompressed,
        file_description=description,
        file_type_passed=True,
        unzip_test_passed=True,
        python_zip_test_passed=True,
    )


def _require_official_spec(
    spec: OfficialArchive, *, allow_unregistered_spec: bool
) -> None:
    registered = OFFICIAL_ARCHIVES.get(spec.filename)
    if not allow_unregistered_spec and registered != spec:
        raise ValueError(
            "download spec is not one of the protocol-locked official archives"
        )


def inspect_baidu_manual_archive(
    spec: OfficialArchive,
    download_root: str | os.PathLike[str],
    *,
    runner: CommandRunner = _run_command,
    allow_unregistered_spec: bool = False,
) -> BaiduManualArchiveState:
    """Inspect one exact-name local file without contacting Baidu.

    Baidu login, extraction-code, captcha, and download interactions must be
    completed by the user in an authenticated client.  This function performs
    no HTTP request and never treats an authentication page as archive data.
    """

    _require_official_spec(spec, allow_unregistered_spec=allow_unregistered_spec)
    if not spec.baidu_url:
        raise ValueError(f"no official Baidu URL is locked for {spec.filename}")
    root = Path(download_root).expanduser().resolve()
    expected = root / spec.filename
    if not expected.exists():
        return BaiduManualArchiveState(
            status="AUTHENTICATION_REQUIRED",
            filename=spec.filename,
            official_baidu_url=spec.baidu_url,
            expected_path=str(expected),
            exists=False,
            bytes=0,
            detail=(
                "exact-name local archive is absent; an authenticated manual "
                "Baidu download is required"
            ),
        )
    observed_bytes = expected.stat().st_size if expected.is_file() else 0
    if expected.is_file() and _looks_like_html(expected):
        return BaiduManualArchiveState(
            status="AUTH_RESPONSE_REJECTED",
            filename=spec.filename,
            official_baidu_url=spec.baidu_url,
            expected_path=str(expected),
            exists=True,
            bytes=observed_bytes,
            detail=(
                "local file is an HTML/XML authentication response, not a ZIP; "
                "it was not adopted or added to the checksum manifest"
            ),
        )
    try:
        verification = verify_archive(
            expected,
            minimum_plausible_bytes=spec.minimum_plausible_bytes,
            expected_bytes=spec.expected_bytes,
            runner=runner,
        )
    except ArchiveValidationError as error:
        return BaiduManualArchiveState(
            status="INVALID_LOCAL_ARCHIVE",
            filename=spec.filename,
            official_baidu_url=spec.baidu_url,
            expected_path=str(expected),
            exists=True,
            bytes=observed_bytes,
            detail=f"exact-name local file failed archive verification: {error}",
        )
    return BaiduManualArchiveState(
        status="READY_FOR_ADOPTION",
        filename=spec.filename,
        official_baidu_url=spec.baidu_url,
        expected_path=str(expected),
        exists=True,
        bytes=verification.bytes,
        detail=(
            "exact-name local ZIP passed file, unzip, CRC, size, and SHA-256 "
            "verification without network access"
        ),
        verification=verification,
    )


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_download_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "graspnet6d.download_manifest.v1",
            "official_dataset_page": OFFICIAL_DATASET_PAGE,
            "archives": [],
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveValidationError(
            f"invalid download manifest {path}: {error}"
        ) from error
    if not isinstance(value, dict) or not isinstance(value.get("archives"), list):
        raise ArchiveValidationError(f"invalid download manifest schema: {path}")
    return value


def _commit_download_evidence(
    *,
    manifest_path: Path,
    checksum_path: Path,
    record: Mapping[str, Any],
) -> None:
    manifest = _read_download_manifest(manifest_path)
    archives: list[dict[str, Any]] = []
    for item in manifest["archives"]:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise ArchiveValidationError(
                f"invalid archive row in download manifest: {manifest_path}"
            )
        if item["filename"] != record["filename"]:
            archives.append(item)
    archives.append(dict(record))
    archives.sort(key=lambda item: str(item["filename"]))
    manifest["archives"] = archives
    manifest["updated_at_utc"] = _utc_now()
    atomic_json(manifest_path, manifest)
    checksum_lines = [f"{item['sha256']}  {item['filename']}" for item in archives]
    atomic_text(checksum_path, "\n".join(checksum_lines) + "\n")


def adopt_baidu_authenticated_archive(
    spec: OfficialArchive,
    download_root: str | os.PathLike[str],
    *,
    authenticated_download_confirmed: bool,
    manifest_path: str | os.PathLike[str] | None = None,
    checksum_path: str | os.PathLike[str] | None = None,
    log_path: str | os.PathLike[str] | None = None,
    runner: CommandRunner = _run_command,
    allow_unregistered_spec: bool = False,
) -> ArchiveVerification:
    """Adopt an already-downloaded Baidu ZIP without any network operation.

    The explicit confirmation prevents an arbitrary pre-existing archive from
    being mislabeled as a user-authenticated Baidu transfer.  Only the exact
    filename inside ``download_root`` is considered; no personal directory is
    searched and no file is moved, replaced, downloaded, or deleted.
    """

    if not authenticated_download_confirmed:
        raise ValueError(
            "authenticated_download_confirmed must be true before assigning "
            "official_baidu_manual_authenticated provenance"
        )
    root = Path(download_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path
        else root / "download_manifest.json"
    )
    checksums = (
        Path(checksum_path).expanduser().resolve()
        if checksum_path
        else root / "archive_checksums.sha256"
    )
    log = (
        Path(log_path).expanduser().resolve()
        if log_path
        else root / "download_log.jsonl"
    )
    checked_at = _utc_now()
    state = inspect_baidu_manual_archive(
        spec,
        root,
        runner=runner,
        allow_unregistered_spec=allow_unregistered_spec,
    )
    source_id = PurePosixPath(spec.baidu_url.split("?", 1)[0]).name
    audit_record = {
        "filename": spec.filename,
        "source": "official_baidu_manual_authenticated",
        "source_file_id": source_id,
        "source_url": spec.baidu_url,
        "attempt": 0,
        "start_time": checked_at,
        "end_time": _utc_now(),
        "resume_used": False,
        "command_argv": [],
        "returncode": None,
        "status": state.status,
        "error": None if state.status == "READY_FOR_ADOPTION" else state.detail,
        "quarantine_path": None,
        "bytes_present": state.bytes,
        "authenticated_download_confirmed": True,
        "automated_access_attempted": False,
    }
    if state.status != "READY_FOR_ADOPTION" or state.verification is None:
        _append_jsonl(log, audit_record)
        raise BaiduManualArchiveError(state)

    verification = state.verification
    manifest_record = {
        "filename": spec.filename,
        "source": "official_baidu_manual_authenticated",
        "source_file_id": source_id,
        "source_url": spec.baidu_url,
        "start_time": checked_at,
        "end_time": audit_record["end_time"],
        "bytes": verification.bytes,
        "sha256": verification.sha256,
        "zip_test_passed": True,
        "retry_count": 0,
        "resume_used": False,
        "authenticated_download_confirmed": True,
        "automated_access_attempted": False,
    }
    _commit_download_evidence(
        manifest_path=manifest,
        checksum_path=checksums,
        record=manifest_record,
    )
    audit_record["status"] = "COMPLETE"
    audit_record["sha256"] = verification.sha256
    _append_jsonl(log, audit_record)
    return verification


def _download_sources(spec: OfficialArchive) -> tuple[tuple[str, str], ...]:
    sources: list[tuple[str, str]] = [("google_drive", spec.google_url)]
    if spec.jbox_url:
        sources.append(("jbox", spec.jbox_url))
    return tuple(sources)


def _download_command(
    source: str,
    url: str,
    destination: Path,
    *,
    python_executable: str,
    curl_executable: str,
) -> tuple[str, ...]:
    if source == "google_drive":
        return (
            python_executable,
            "-m",
            "gdown",
            "--continue",
            url,
            "-O",
            str(destination),
        )
    if source == "jbox":
        return (
            curl_executable,
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        )
    raise ValueError(f"unsupported official source {source!r}")


def _quarantine_invalid_response(path: Path, *, attempt: int, reason: str) -> Path:
    label = "html" if _looks_like_html(path) else "zip"
    destination = path.with_name(f"{path.name}.invalid_{label}.attempt_{attempt:02d}")
    if destination.exists():
        raise ArchiveValidationError(
            f"refusing to replace prior quarantine evidence: {destination}"
        )
    os.replace(path, destination)
    atomic_text(
        destination.with_suffix(destination.suffix + ".reason.txt"),
        reason.rstrip() + "\n",
    )
    return destination


def download_archive(
    spec: OfficialArchive,
    download_root: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    checksum_path: str | os.PathLike[str] | None = None,
    log_path: str | os.PathLike[str] | None = None,
    python_executable: str = sys.executable,
    curl_executable: str = "/usr/bin/curl",
    maximum_attempts: int = 12,
    initial_wait_seconds: float = 30.0,
    maximum_wait_seconds: float = 600.0,
    runner: CommandRunner = _run_command,
    sleeper: Sleeper = time.sleep,
    allow_unregistered_spec: bool = False,
) -> ArchiveVerification:
    """Download one official archive with resume, retries, and source rotation."""

    _require_official_spec(spec, allow_unregistered_spec=allow_unregistered_spec)
    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be positive")
    if initial_wait_seconds < 0 or maximum_wait_seconds < initial_wait_seconds:
        raise ValueError("invalid retry wait bounds")
    root = Path(download_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / spec.filename
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path
        else root / "download_manifest.json"
    )
    checksums = (
        Path(checksum_path).resolve()
        if checksum_path
        else root / "archive_checksums.sha256"
    )
    log = Path(log_path).resolve() if log_path else root / "download_log.jsonl"
    overall_start = _utc_now()

    if destination.exists():
        try:
            verification = verify_archive(
                destination,
                minimum_plausible_bytes=spec.minimum_plausible_bytes,
                expected_bytes=spec.expected_bytes,
                runner=runner,
            )
        except ArchiveValidationError:
            pass
        else:
            record = {
                "filename": spec.filename,
                "source": "verified_existing",
                "source_file_id": spec.google_drive_id,
                "start_time": overall_start,
                "end_time": _utc_now(),
                "bytes": verification.bytes,
                "sha256": verification.sha256,
                "zip_test_passed": True,
                "retry_count": 0,
                "resume_used": True,
            }
            _commit_download_evidence(
                manifest_path=manifest, checksum_path=checksums, record=record
            )
            return verification

    sources = _download_sources(spec)
    errors: list[str] = []
    any_resume = destination.exists() and destination.stat().st_size > 0
    for attempt in range(1, maximum_attempts + 1):
        source, url = sources[(attempt - 1) % len(sources)]
        resumed_this_attempt = destination.exists() and destination.stat().st_size > 0
        any_resume = any_resume or resumed_this_attempt
        started = _utc_now()
        command = _download_command(
            source,
            url,
            destination,
            python_executable=python_executable,
            curl_executable=curl_executable,
        )
        completed = runner(command)
        error: str | None = None
        quarantine_path: str | None = None
        verification: ArchiveVerification | None = None
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            error = f"download command exited {completed.returncode}: {detail}"
            if destination.exists() and _looks_like_html(destination):
                quarantined = _quarantine_invalid_response(
                    destination, attempt=attempt, reason=error
                )
                quarantine_path = str(quarantined)
        elif not destination.is_file():
            error = "download command reported success without an output file"
        else:
            try:
                verification = verify_archive(
                    destination,
                    minimum_plausible_bytes=spec.minimum_plausible_bytes,
                    expected_bytes=spec.expected_bytes,
                    runner=runner,
                )
            except ArchiveValidationError as validation_error:
                error = str(validation_error)
                quarantined = _quarantine_invalid_response(
                    destination, attempt=attempt, reason=error
                )
                quarantine_path = str(quarantined)

        attempt_record = {
            "filename": spec.filename,
            "source": source,
            "source_file_id": spec.google_drive_id,
            "source_url": url,
            "attempt": attempt,
            "start_time": started,
            "end_time": _utc_now(),
            "resume_used": resumed_this_attempt,
            "command_argv": list(command),
            "returncode": completed.returncode,
            "status": "COMPLETE" if verification else "RETRYABLE_FAILURE",
            "error": error,
            "quarantine_path": quarantine_path,
            "bytes_present": destination.stat().st_size if destination.exists() else 0,
        }
        _append_jsonl(log, attempt_record)
        if verification is not None:
            record = {
                "filename": spec.filename,
                "source": source,
                "source_file_id": spec.google_drive_id,
                "start_time": overall_start,
                "end_time": attempt_record["end_time"],
                "bytes": verification.bytes,
                "sha256": verification.sha256,
                "zip_test_passed": True,
                "retry_count": attempt - 1,
                "resume_used": any_resume,
            }
            _commit_download_evidence(
                manifest_path=manifest, checksum_path=checksums, record=record
            )
            return verification

        errors.append(f"attempt {attempt} ({source}): {error}")
        if attempt < maximum_attempts:
            delay = min(
                initial_wait_seconds * (2 ** (attempt - 1)), maximum_wait_seconds
            )
            sleeper(delay)

    raise DownloadExhaustedError(
        f"failed to download {spec.filename} after {maximum_attempts} attempts; "
        + " | ".join(errors)
    )


def deterministic_frame_ids(
    frames_per_scene: int = 16, *, first: int = 0, last: int = 255
) -> tuple[int, ...]:
    """Match ``np.linspace(first, last, num=N, dtype=int)`` exactly."""

    if frames_per_scene < 1:
        raise ValueError("frames_per_scene must be positive")
    if first < 0 or last < first:
        raise ValueError("invalid frame range")
    values = np.linspace(first, last, num=frames_per_scene, dtype=int)
    return tuple(sorted({int(value) for value in values.tolist()}))


def _dataset_relative_path(member_name: str) -> tuple[str | None, str]:
    """Return an official-compatible relative path plus adapter description."""

    name = _safe_member_name(member_name)
    parts = PurePosixPath(name).parts
    if "scenes" in parts:
        index = parts.index("scenes")
        return PurePosixPath(*parts[index:]).as_posix(), "strip_prefix_before_scenes"
    for root in ("models", "grasp_label", "collision_label", "dex_models"):
        if root in parts:
            index = parts.index(root)
            return PurePosixPath(
                *parts[index:]
            ).as_posix(), f"strip_prefix_before_{root}"
    for index, part in enumerate(parts):
        if _SCENE_RE.fullmatch(part):
            relative = PurePosixPath("scenes", *parts[index:]).as_posix()
            return relative, "insert_scenes_and_strip_archive_prefix"
    return None, "unmapped"


def parse_zipinfo_filelist(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Parse and safety-check output previously produced by ``zipinfo -1``."""

    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ArchiveValidationError(f"expected regular zipinfo file list: {source}")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ArchiveValidationError(
            f"invalid zipinfo file list {source}: {error}"
        ) from error
    members: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        if not raw:
            continue
        member = _safe_member_name(raw.rstrip("/"))
        if member in seen:
            raise ArchiveValidationError(
                f"duplicate member in zipinfo file list: {member}"
            )
        seen.add(member)
        members.append(member)
    if not members:
        raise ArchiveValidationError(f"empty zipinfo file list: {source}")
    return tuple(members)


def inspect_archive(
    archive_path: str | os.PathLike[str],
    *,
    filelist_path: str | os.PathLike[str] | None = None,
) -> ArchiveInventory:
    """Read only the central directory and tiny per-scene object lists."""

    source = Path(archive_path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ArchiveValidationError(f"expected regular archive: {source}")
    members: list[str] = []
    dataset_paths: dict[str, str] = {}
    sizes: dict[str, int] = {}
    crc32: dict[str, int] = {}
    adapters: set[str] = set()
    scene_members: dict[int, dict[str, Any]] = {}
    try:
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                safe_name = _safe_member_name(info.filename.rstrip("/"))
                if safe_name in sizes:
                    raise ArchiveValidationError(
                        f"duplicate ZIP member rejected: {safe_name}"
                    )
                members.append(safe_name)
                sizes[safe_name] = int(info.file_size)
                crc32[safe_name] = int(info.CRC)
                if info.is_dir():
                    continue
                relative, adapter = _dataset_relative_path(safe_name)
                if relative is None:
                    continue
                if relative in dataset_paths:
                    raise ArchiveValidationError(
                        f"multiple members map to dataset path {relative!r}"
                    )
                dataset_paths[relative] = safe_name
                adapters.add(adapter)
                parts = PurePosixPath(relative).parts
                if len(parts) < 2 or parts[0] != "scenes":
                    continue
                match = _SCENE_RE.fullmatch(parts[1])
                if not match:
                    continue
                scene_id = int(match.group(1))
                source_parts = PurePosixPath(safe_name).parts
                source_scene_index = next(
                    index
                    for index, source_part in enumerate(source_parts)
                    if _SCENE_RE.fullmatch(source_part)
                )
                state = scene_members.setdefault(
                    scene_id,
                    {
                        "prefix": PurePosixPath(
                            *source_parts[: source_scene_index + 1]
                        ).as_posix(),
                        "cameras": set(),
                        "frames": {},
                        "object_ids": (),
                        "has_object_id_list": False,
                        "has_rs_wrt_kn": False,
                    },
                )
                remainder = parts[2:]
                if remainder == ("object_id_list.txt",):
                    state["has_object_id_list"] = True
                    raw = archive.read(info).decode("utf-8", errors="strict")
                    try:
                        state["object_ids"] = tuple(
                            int(token) for token in re.findall(r"-?\d+", raw)
                        )
                    except ValueError as error:
                        raise ArchiveValidationError(
                            f"invalid object_id_list in {safe_name}"
                        ) from error
                elif remainder == ("rs_wrt_kn.npy",):
                    state["has_rs_wrt_kn"] = True
                elif len(remainder) >= 2 and remainder[0] in {"kinect", "realsense"}:
                    camera = remainder[0]
                    state["cameras"].add(camera)
                    if len(remainder) == 3 and remainder[1] in {
                        "rgb",
                        "depth",
                        "label",
                        "meta",
                        "annotations",
                    }:
                        frame_match = _FRAME_RE.fullmatch(remainder[2])
                        if frame_match:
                            frame_sets = state["frames"].setdefault(camera, {})
                            frame_sets.setdefault(remainder[1], set()).add(
                                int(frame_match.group(1))
                            )
            compressed = sum(
                info.compress_size for info in archive.infolist() if not info.is_dir()
            )
            uncompressed = sum(
                info.file_size for info in archive.infolist() if not info.is_dir()
            )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise ArchiveValidationError(f"cannot inspect ZIP {source}: {error}") from error

    scenes: list[SceneArchiveRecord] = []
    for scene_id, state in sorted(scene_members.items()):
        frame_ids: dict[str, tuple[int, ...]] = {}
        for camera, channels in state["frames"].items():
            common: set[int] | None = None
            for channel in ("rgb", "depth", "label", "meta", "annotations"):
                values = set(channels.get(channel, set()))
                common = values if common is None else common & values
            frame_ids[camera] = tuple(sorted(common or set()))
        scenes.append(
            SceneArchiveRecord(
                scene_id=scene_id,
                archive_scene_prefix=str(state["prefix"]),
                dataset_scene_path=f"scenes/scene_{scene_id:04d}",
                cameras=tuple(sorted(state["cameras"])),
                frame_ids_by_camera=frame_ids,
                object_ids=tuple(state["object_ids"]),
                has_object_id_list=bool(state["has_object_id_list"]),
                has_rs_wrt_kn=bool(state["has_rs_wrt_kn"]),
            )
        )
    adapter = "+".join(sorted(adapters)) if adapters else "unmapped"
    inventory = ArchiveInventory(
        archive_path=str(source),
        archive_sha256=sha256_file(source),
        archive_bytes=source.stat().st_size,
        member_count=len(members),
        compressed_member_bytes=compressed,
        uncompressed_member_bytes=uncompressed,
        members=tuple(members),
        dataset_paths=dataset_paths,
        member_sizes=sizes,
        member_crc32=crc32,
        scenes=tuple(scenes),
        path_adapter=adapter,
    )
    if filelist_path is not None:
        external_filelist = Path(filelist_path).expanduser().resolve()
        if external_filelist.exists():
            listed_members = parse_zipinfo_filelist(external_filelist)
            if listed_members != inventory.members:
                raise ArchiveValidationError(
                    "zipinfo -1 file list does not match the inspected archive"
                )
        else:
            atomic_text(external_filelist, "\n".join(inventory.members) + "\n")
    return inventory


def write_inventory_manifest(
    path: str | os.PathLike[str],
    inventories: Iterable[ArchiveInventory],
) -> Path:
    values = sorted(inventories, key=lambda value: value.archive_path)
    scene_sources: dict[int, str] = {}
    duplicate_scenes: list[int] = []
    for inventory in values:
        for scene in inventory.scenes:
            if scene.scene_id in scene_sources:
                duplicate_scenes.append(scene.scene_id)
            else:
                scene_sources[scene.scene_id] = inventory.archive_path
    payload = {
        "schema_version": "graspnet6d.archive_inventory.v1",
        "created_at_utc": _utc_now(),
        "official_dataset_page": OFFICIAL_DATASET_PAGE,
        "archive_count": len(values),
        "unique_scene_count": len(scene_sources),
        "scene_sources": {
            str(key): value for key, value in sorted(scene_sources.items())
        },
        "duplicate_scene_ids": sorted(set(duplicate_scenes)),
        "archives": [value.to_record() for value in values],
    }
    return atomic_json(path, payload)


def _item_for_dataset_path(
    inventory: ArchiveInventory, dataset_path: str
) -> ExtractionItem:
    member = inventory.dataset_paths.get(dataset_path)
    if member is None:
        raise ExtractionPlanError(f"required archive member is absent: {dataset_path}")
    return ExtractionItem(
        source_member=member,
        destination_relative=dataset_path,
        bytes=inventory.member_sizes[member],
        crc32=inventory.member_crc32[member],
    )


def build_training_extraction_plan(
    inventory: ArchiveInventory,
    scene_ids: Iterable[int],
    *,
    camera: str,
    frame_ids: Iterable[int] | None = None,
) -> ExtractionPlan:
    """Select official scene metadata plus five modalities for one camera."""

    if camera not in {"kinect", "realsense"}:
        raise ValueError("camera must be 'kinect' or 'realsense'")
    selected_scenes = tuple(sorted({int(value) for value in scene_ids}))
    if not selected_scenes:
        raise ValueError("at least one scene must be selected")
    selected_frames = tuple(
        sorted(
            {
                int(value)
                for value in (
                    deterministic_frame_ids() if frame_ids is None else frame_ids
                )
            }
        )
    )
    available = {scene.scene_id: scene for scene in inventory.scenes}
    items: list[ExtractionItem] = []
    for scene_id in selected_scenes:
        scene = available.get(scene_id)
        if scene is None:
            raise ExtractionPlanError(f"scene_{scene_id:04d} is absent from archive")
        if camera not in scene.cameras:
            raise ExtractionPlanError(
                f"camera {camera!r} is absent from scene_{scene_id:04d}"
            )
        missing_frames = sorted(
            set(selected_frames) - set(scene.frame_ids_by_camera.get(camera, ()))
        )
        if missing_frames:
            raise ExtractionPlanError(
                f"scene_{scene_id:04d}/{camera} lacks complete selected frames: "
                f"{missing_frames}"
            )
        base = f"scenes/scene_{scene_id:04d}"
        for relative in ("object_id_list.txt", "rs_wrt_kn.npy"):
            items.append(_item_for_dataset_path(inventory, f"{base}/{relative}"))
        for relative in ("camK.npy", "camera_poses.npy", "cam0_wrt_table.npy"):
            items.append(
                _item_for_dataset_path(inventory, f"{base}/{camera}/{relative}")
            )
        extensions = {
            "rgb": "png",
            "depth": "png",
            "label": "png",
            "meta": "mat",
            "annotations": "xml",
        }
        for frame_id in selected_frames:
            for channel, extension in extensions.items():
                items.append(
                    _item_for_dataset_path(
                        inventory,
                        f"{base}/{camera}/{channel}/{frame_id:04d}.{extension}",
                    )
                )
    destinations = [item.destination_relative for item in items]
    if len(destinations) != len(set(destinations)):
        raise ExtractionPlanError(
            "selective extraction plan has duplicate destinations"
        )
    return ExtractionPlan(
        archive_path=inventory.archive_path,
        archive_sha256=inventory.archive_sha256,
        kind="training_scenes",
        items=tuple(items),
        selected_scene_ids=selected_scenes,
        camera=camera,
        frame_ids=selected_frames,
    )


def build_collision_extraction_plan(
    inventory: ArchiveInventory, scene_ids: Iterable[int]
) -> ExtractionPlan:
    selected = tuple(sorted({int(value) for value in scene_ids}))
    if not selected:
        raise ValueError("at least one collision-label scene must be selected")
    prefixes = tuple(f"collision_label/scene_{value:04d}/" for value in selected)
    items = [
        ExtractionItem(
            source_member=member,
            destination_relative=relative,
            bytes=inventory.member_sizes[member],
            crc32=inventory.member_crc32[member],
        )
        for relative, member in sorted(inventory.dataset_paths.items())
        if relative.startswith(prefixes)
    ]
    present = {
        int(PurePosixPath(item.destination_relative).parts[1].split("_")[1])
        for item in items
    }
    missing = sorted(set(selected) - present)
    if missing:
        raise ExtractionPlanError(f"collision labels absent for scenes: {missing}")
    return ExtractionPlan(
        archive_path=inventory.archive_path,
        archive_sha256=inventory.archive_sha256,
        kind="collision_scenes",
        items=tuple(items),
        selected_scene_ids=selected,
    )


def build_subtree_extraction_plan(
    inventory: ArchiveInventory,
    subtree: Literal["models", "grasp_label", "dex_models"],
    *,
    object_ids: Iterable[int] | None = None,
) -> ExtractionPlan:
    prefix = subtree + "/"
    selected_objects = (
        None if object_ids is None else {int(value) for value in object_ids}
    )
    if selected_objects is not None and (
        not selected_objects
        or any(value < 0 or value >= 88 for value in selected_objects)
    ):
        raise ValueError("object_ids must be a non-empty subset of [0, 87]")

    def selected(relative: str) -> bool:
        if not relative.startswith(prefix):
            return False
        if selected_objects is None:
            return True
        leaf = PurePosixPath(relative).parts[1]
        match = re.match(r"^(\d{3})(?:_|\.|$)", leaf)
        return match is not None and int(match.group(1)) in selected_objects

    items = [
        ExtractionItem(
            source_member=member,
            destination_relative=relative,
            bytes=inventory.member_sizes[member],
            crc32=inventory.member_crc32[member],
        )
        for relative, member in sorted(inventory.dataset_paths.items())
        if selected(relative)
    ]
    if not items:
        detail = (
            f" for object IDs {sorted(selected_objects)}"
            if selected_objects is not None
            else ""
        )
        raise ExtractionPlanError(f"archive has no {subtree}/ assets{detail}")
    if selected_objects is not None:
        observed: set[int] = set()
        for item in items:
            leaf = PurePosixPath(item.destination_relative).parts[1]
            match = re.match(r"^(\d{3})(?:_|\.|$)", leaf)
            if match is not None:
                observed.add(int(match.group(1)))
        missing = sorted(selected_objects - observed)
        if missing:
            raise ExtractionPlanError(
                f"{subtree} assets absent for object IDs: {missing}"
            )
    return ExtractionPlan(
        archive_path=inventory.archive_path,
        archive_sha256=inventory.archive_sha256,
        kind="subtree",
        items=tuple(items),
        subtree=subtree,
    )


def _crc32_file(path: Path, chunk_size: int = 1024 * 1024) -> int:
    value = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            value = binascii.crc32(block, value)
    return value & 0xFFFFFFFF


def _matches_item(path: Path, item: ExtractionItem) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == item.bytes
        and _crc32_file(path) == item.crc32
    )


def extract_selective(
    plan: ExtractionPlan,
    destination_root: str | os.PathLike[str],
    *,
    staging_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str] | None = None,
    resume: bool = True,
) -> ExtractionResult:
    """Atomically promote only planned regular files; never remove an archive."""

    archive_path = Path(plan.archive_path).resolve()
    if sha256_file(archive_path) != plan.archive_sha256:
        raise ExtractionPlanError("archive changed after the extraction plan was built")
    destination = Path(destination_root).expanduser().resolve()
    staging = Path(staging_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    extracted = 0
    resumed = 0
    verified = 0
    with zipfile.ZipFile(archive_path) as archive:
        infos = {info.filename.rstrip("/"): info for info in archive.infolist()}
        for index, item in enumerate(plan.items):
            relative = PurePosixPath(item.destination_relative)
            if relative.is_absolute() or ".." in relative.parts:
                raise ExtractionPlanError(
                    f"unsafe extraction destination: {item.destination_relative}"
                )
            output = destination.joinpath(*relative.parts)
            try:
                output.resolve().relative_to(destination)
            except ValueError as error:
                raise ExtractionPlanError(
                    f"extraction escaped destination: {output}"
                ) from error
            if output.exists():
                if resume and _matches_item(output, item):
                    resumed += 1
                    verified += 1
                    continue
                raise ExtractionPlanError(
                    f"refusing to replace non-matching existing output: {output}"
                )
            info = infos.get(item.source_member)
            if info is None or info.is_dir():
                raise ExtractionPlanError(
                    f"planned source member disappeared: {item.source_member}"
                )
            if info.file_size != item.bytes or info.CRC != item.crc32:
                raise ExtractionPlanError(
                    f"planned source member changed: {item.source_member}"
                )
            temporary = staging / (
                f"{index:08d}.{os.getpid()}.{Path(item.destination_relative).name}.part"
            )
            if temporary.exists():
                raise ExtractionPlanError(
                    f"staging collision retained for audit: {temporary}"
                )
            temporary.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, temporary.open("xb") as target:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            if not _matches_item(temporary, item):
                raise ExtractionPlanError(
                    f"extracted bytes failed size/CRC validation: {item.source_member}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, output)
            extracted += 1
            verified += int(_matches_item(output, item))

    complete = verified == len(plan.items)
    committed_manifest: str | None = None
    manifest_sha256: str | None = None
    result_base = {
        "schema_version": "graspnet6d.compact_extraction.v1",
        "completed_at_utc": _utc_now(),
        "plan": plan.to_record(),
        "destination_root": str(destination),
        "selected_file_count": len(plan.items),
        "selected_bytes": plan.selected_bytes,
        "extracted_file_count": extracted,
        "resumed_file_count": resumed,
        "verified_file_count": verified,
        "complete": complete,
    }
    if manifest_path is not None:
        committed = atomic_json(manifest_path, result_base)
        committed_manifest = str(committed.resolve())
        manifest_sha256 = sha256_file(committed)
    return ExtractionResult(
        archive_path=str(archive_path),
        archive_sha256=plan.archive_sha256,
        destination_root=str(destination),
        selected_file_count=len(plan.items),
        selected_bytes=plan.selected_bytes,
        extracted_file_count=extracted,
        resumed_file_count=resumed,
        verified_file_count=verified,
        complete=complete,
        manifest_path=committed_manifest,
        manifest_sha256=manifest_sha256,
    )


def plan_verified_archive_deletion(
    verification: ArchiveVerification,
    extraction: ExtractionResult,
    *,
    downstream_loader_verified: bool,
) -> ArchiveDeletionPlan:
    """Return, but never execute, the fully audited archive cleanup decision."""

    unmet: list[str] = []
    archive_path = Path(verification.path).resolve()
    if not verification.zip_test_passed:
        unmet.append("archive ZIP verification did not pass")
    if extraction.archive_sha256 != verification.sha256:
        unmet.append("extraction and archive SHA-256 differ")
    if not extraction.complete:
        unmet.append("selected extraction is incomplete")
    if extraction.verified_file_count != extraction.selected_file_count:
        unmet.append("not every selected file was verified")
    if not extraction.manifest_path or not extraction.manifest_sha256:
        unmet.append("extraction manifest is not durably committed")
    elif not Path(extraction.manifest_path).is_file():
        unmet.append("extraction manifest disappeared")
    else:
        try:
            manifest_matches = (
                sha256_file(extraction.manifest_path) == extraction.manifest_sha256
            )
        except ValueError:
            manifest_matches = False
        if not manifest_matches:
            unmet.append("extraction manifest changed after commit")
    if not downstream_loader_verified:
        unmet.append("downstream loader has not read the compact dataset")
    try:
        archive_matches = (
            archive_path.is_file()
            and not archive_path.is_symlink()
            and sha256_file(archive_path) == verification.sha256
        )
    except ValueError:
        archive_matches = False
    if not archive_matches:
        unmet.append("archive changed or disappeared after verification")
    eligible = not unmet
    reason = (
        "verified archive is reconstructable from the recorded official source; "
        "selected files, manifest, and downstream loader are verified"
        if eligible
        else "archive retention required: " + "; ".join(unmet)
    )
    return ArchiveDeletionPlan(
        eligible=eligible,
        path=str(archive_path),
        size_before_bytes=verification.bytes,
        reason=reason,
        command_argv=("rm", "--", str(archive_path)) if eligible else (),
        reconstructable=eligible,
        unmet_conditions=tuple(unmet),
    )


__all__ = [
    "ArchiveDeletionPlan",
    "ArchiveInventory",
    "ArchiveValidationError",
    "ArchiveVerification",
    "BaiduManualArchiveError",
    "BaiduManualArchiveState",
    "DownloadExhaustedError",
    "ExtractionItem",
    "ExtractionPlan",
    "ExtractionPlanError",
    "ExtractionResult",
    "OFFICIAL_ARCHIVES",
    "OFFICIAL_DATASET_PAGE",
    "OfficialArchive",
    "SceneArchiveRecord",
    "adopt_baidu_authenticated_archive",
    "build_collision_extraction_plan",
    "build_subtree_extraction_plan",
    "build_training_extraction_plan",
    "deterministic_frame_ids",
    "download_archive",
    "extract_selective",
    "inspect_archive",
    "inspect_baidu_manual_archive",
    "parse_zipinfo_filelist",
    "plan_verified_archive_deletion",
    "verify_archive",
    "write_inventory_manifest",
]
