"""Staged storage budgets and auditable cleanup records.

The compact GraspNet route never needs every archive, every extracted camera,
and every generated cache to coexist.  This module therefore budgets only the
bytes that a *single next stage* can add.  Existing bytes are already reflected
in ``shutil.disk_usage(...).free`` and must not be counted a second time.

Archive inspection intentionally runs both ``zipinfo -l`` and ``zipinfo -1``.
The reported totals and names are cross-checked against Python's ZIP central
directory reader before they can be used for a selective-extraction budget.
No extraction or deletion is performed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import zipfile
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


BYTES_PER_DECIMAL_GB = 1_000_000_000
DEFAULT_MINIMUM_FREE_RESERVE_GB = 20.0


class ArchiveInspectionError(RuntimeError):
    """Raised when archive metadata is missing, unsafe, or inconsistent."""


@dataclass(frozen=True)
class ArchiveInventory:
    """Measured ZIP central-directory facts used by a staged budget."""

    archive_path: str
    archive_bytes: int
    compressed_member_bytes: int
    uncompressed_total_bytes: int
    selected_extraction_bytes: int
    member_count: int
    selected_member_count: int
    unsafe_members: tuple[str, ...]
    zipinfo_listing_sha256: str
    zipinfo_names_sha256: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["unsafe_members"] = list(self.unsafe_members)
        return record


@dataclass(frozen=True)
class StagedDiskBudget:
    """Incremental disk budget for one download/extraction/cache stage.

    Each byte term means additional space the next stage may allocate.  For
    example, an archive that is already fully downloaded has
    ``archive_bytes == 0`` during the subsequent extraction stage because its
    current footprint is already reflected in ``current_free_bytes``.
    """

    phase: str
    filesystem_path: str
    filesystem_total_bytes: int
    current_free_bytes: int
    archive_bytes: int
    selected_extraction_bytes: int
    temporary_bytes: int
    estimated_cache_bytes: int
    mandatory_reserve_bytes: int
    required_total_bytes: int
    surplus_or_deficit_bytes: int
    allowed: bool

    @property
    def deficit_bytes(self) -> int:
        return max(0, -self.surplus_or_deficit_bytes)

    @property
    def surplus_bytes(self) -> int:
        return max(0, self.surplus_or_deficit_bytes)

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "deficit_bytes": self.deficit_bytes,
            "surplus_bytes": self.surplus_bytes,
            "assumptions": {
                "incremental_budget": True,
                "one_archive_at_a_time": True,
                "existing_files_are_not_counted_twice": True,
                "minimum_free_reserve_gb": (
                    self.mandatory_reserve_bytes / BYTES_PER_DECIMAL_GB
                ),
            },
        }


@dataclass(frozen=True)
class CleanupCandidate:
    """A measured, explicitly reconstructable cleanup candidate.

    This is an audit record, not an authorization to delete.  The caller must
    still prove that the path belongs to the experiment's cleanup whitelist.
    """

    path: str
    type: str
    size_before_bytes: int
    reason: str
    command: tuple[str, ...]
    reconstructable: bool
    whitelist_rule: str

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "command": shlex.join(self.command),
        }


@dataclass(frozen=True)
class CleanupRecord:
    """Outcome row matching the mandatory disk-cleanup audit schema."""

    timestamp: str
    path: str
    type: str
    size_before_bytes: int
    reason: str
    command: str
    reconstructable: bool
    result: str
    free_space_before: int
    free_space_after: int

    @classmethod
    def from_candidate(
        cls,
        candidate: CleanupCandidate,
        *,
        result: str,
        free_space_before: int,
        free_space_after: int,
        timestamp: str | None = None,
    ) -> CleanupRecord:
        if free_space_before < 0 or free_space_after < 0:
            raise ValueError("free-space observations must be non-negative")
        return cls(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            path=candidate.path,
            type=candidate.type,
            size_before_bytes=candidate.size_before_bytes,
            reason=candidate.reason,
            command=shlex.join(candidate.command),
            reconstructable=candidate.reconstructable,
            result=str(result),
            free_space_before=int(free_space_before),
            free_space_after=int(free_space_after),
        )


CLEANUP_LOG_FIELDS = (
    "timestamp",
    "path",
    "type",
    "size_before_bytes",
    "reason",
    "command",
    "reconstructable",
    "result",
    "free_space_before",
    "free_space_after",
)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _existing_ancestor(path: Path) -> Path:
    target = path.expanduser().resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    if not target.exists():  # pragma: no cover - every supported OS has a root
        raise FileNotFoundError(f"no existing ancestor for {path}")
    return target


def _nonnegative(name: str, value: int) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def build_staged_disk_budget(
    filesystem_path: Path | str,
    *,
    phase: str,
    archive_bytes: int,
    selected_extraction_bytes: int,
    temporary_bytes: int,
    estimated_cache_bytes: int,
    minimum_free_reserve_gb: float = DEFAULT_MINIMUM_FREE_RESERVE_GB,
) -> StagedDiskBudget:
    """Measure free space and budget only the next stage's new allocations."""

    if not str(phase).strip():
        raise ValueError("phase must be non-empty")
    if minimum_free_reserve_gb < 0:
        raise ValueError("minimum_free_reserve_gb must be non-negative")
    archive = _nonnegative("archive_bytes", archive_bytes)
    selected = _nonnegative("selected_extraction_bytes", selected_extraction_bytes)
    temporary = _nonnegative("temporary_bytes", temporary_bytes)
    cache = _nonnegative("estimated_cache_bytes", estimated_cache_bytes)
    reserve = int(round(float(minimum_free_reserve_gb) * BYTES_PER_DECIMAL_GB))
    target = _existing_ancestor(Path(filesystem_path))
    usage = shutil.disk_usage(target)
    required = archive + selected + temporary + cache + reserve
    balance = int(usage.free) - required
    return StagedDiskBudget(
        phase=str(phase),
        filesystem_path=str(target),
        filesystem_total_bytes=int(usage.total),
        current_free_bytes=int(usage.free),
        archive_bytes=archive,
        selected_extraction_bytes=selected,
        temporary_bytes=temporary,
        estimated_cache_bytes=cache,
        mandatory_reserve_bytes=reserve,
        required_total_bytes=required,
        surplus_or_deficit_bytes=balance,
        allowed=balance >= 0,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _zipinfo(
    archive: Path,
    option: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    try:
        completed = runner(
            ["zipinfo", option, str(archive)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise ArchiveInspectionError(
            "zipinfo is required for measured archive inventory"
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ArchiveInspectionError(
            f"zipinfo {option} failed for {archive}: {detail or 'no diagnostic'}"
        )
    return completed.stdout


def _zipinfo_uncompressed_total(listing: str) -> int:
    # Info-ZIP ends the long listing with e.g.:
    # ``2 files, 9 bytes uncompressed, 9 bytes compressed: 0.0%``.
    import re

    matches = re.findall(
        r"(?:files?|entries?),\s*([0-9][0-9,]*)\s+bytes uncompressed",
        listing,
        flags=re.IGNORECASE,
    )
    if not matches:
        raise ArchiveInspectionError(
            "zipinfo -l output has no uncompressed-byte summary"
        )
    return int(matches[-1].replace(",", ""))


def _unsafe_member(name: str) -> bool:
    if not name or "\x00" in name or "\\" in name:
        return True
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts


def inspect_zipinfo_archive(
    archive_path: Path | str,
    *,
    selected_members: Collection[str] | None = None,
    selector: Callable[[str], bool] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ArchiveInventory:
    """Inspect a ZIP and compute exact selected uncompressed bytes.

    ``selected_members`` is fail-closed: every requested name must exist.
    Alternatively, ``selector`` may encode the deterministic scene/frame
    selection.  The two selection mechanisms cannot be combined.
    """

    if selected_members is not None and selector is not None:
        raise ValueError("provide selected_members or selector, not both")
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise ArchiveInspectionError(f"archive does not exist: {archive}")
    if not zipfile.is_zipfile(archive):
        raise ArchiveInspectionError(f"not a ZIP archive: {archive}")

    listing = _zipinfo(archive, "-l", runner=runner)
    names_output = _zipinfo(archive, "-1", runner=runner)
    zipinfo_names = names_output.splitlines()
    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as handle:
            infos = handle.infolist()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ArchiveInspectionError(f"cannot read ZIP directory: {archive}") from error

    central_names = [item.filename for item in infos]
    if zipinfo_names != central_names:
        raise ArchiveInspectionError(
            "zipinfo -1 names disagree with Python ZIP central directory"
        )
    total = sum(int(item.file_size) for item in infos)
    reported_total = _zipinfo_uncompressed_total(listing)
    if total != reported_total:
        raise ArchiveInspectionError(
            "zipinfo -l uncompressed total disagrees with ZIP central directory: "
            f"{reported_total} != {total}"
        )

    requested = None if selected_members is None else set(selected_members)
    if requested is not None:
        missing = requested.difference(central_names)
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ArchiveInspectionError(
                f"{len(missing)} selected archive members are absent: {preview}"
            )
    chosen = [
        item
        for item in infos
        if (
            True
            if requested is None and selector is None
            else item.filename in requested
            if requested is not None
            else bool(selector and selector(item.filename))
        )
    ]
    unsafe = tuple(sorted(name for name in central_names if _unsafe_member(name)))
    return ArchiveInventory(
        archive_path=str(archive),
        archive_bytes=int(archive.stat().st_size),
        compressed_member_bytes=sum(int(item.compress_size) for item in infos),
        uncompressed_total_bytes=total,
        selected_extraction_bytes=sum(int(item.file_size) for item in chosen),
        member_count=len(infos),
        selected_member_count=len(chosen),
        unsafe_members=unsafe,
        zipinfo_listing_sha256=_sha256_text(listing),
        zipinfo_names_sha256=_sha256_text(names_output),
    )


def write_staged_disk_budget(
    budget: StagedDiskBudget,
    *,
    json_path: Path | str,
    markdown_path: Path | str,
    inventory: ArchiveInventory | None = None,
) -> None:
    """Atomically write the required JSON and human-readable budget audit."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "budget": budget.to_record(),
    }
    if inventory is not None:
        payload["archive_inventory"] = inventory.to_record()
    _atomic_text(
        Path(json_path), json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    status = "GO" if budget.allowed else "NO-GO"
    lines = [
        "# Staged disk budget",
        "",
        f"Phase: `{budget.phase}`  ",
        f"Decision: **{status}**",
        "",
        "| Item | Bytes | Decimal GB |",
        "|---|---:|---:|",
    ]
    fields = (
        ("Current free", budget.current_free_bytes),
        ("Archive (additional this stage)", budget.archive_bytes),
        ("Selected extraction", budget.selected_extraction_bytes),
        ("Temporary", budget.temporary_bytes),
        ("Estimated cache", budget.estimated_cache_bytes),
        ("Mandatory reserve", budget.mandatory_reserve_bytes),
        ("Required total", budget.required_total_bytes),
        ("Surplus or deficit", budget.surplus_or_deficit_bytes),
    )
    for label, value in fields:
        lines.append(f"| {label} | {value} | {value / BYTES_PER_DECIMAL_GB:.3f} |")
    lines.extend(
        [
            "",
            "The budget is incremental and assumes one archive at a time. Existing "
            "files are already reflected in current free space and are not counted twice.",
        ]
    )
    if inventory is not None:
        lines.extend(
            [
                "",
                "## Measured archive inventory",
                "",
                f"- Archive: `{inventory.archive_path}`",
                f"- Members: {inventory.member_count}",
                f"- Total uncompressed bytes: {inventory.uncompressed_total_bytes}",
                f"- Selected members: {inventory.selected_member_count}",
                f"- Selected extraction bytes: {inventory.selected_extraction_bytes}",
                f"- Unsafe member names: {len(inventory.unsafe_members)}",
            ]
        )
    _atomic_text(Path(markdown_path), "\n".join(lines) + "\n")


def measure_path_bytes(path: Path | str) -> int:
    """Measure allocated file sizes without following directory symlinks."""

    target = Path(path)
    if not target.exists() and not target.is_symlink():
        return 0
    if target.is_symlink() or target.is_file():
        return int(target.lstat().st_size)
    total = int(target.lstat().st_size)
    for root, directories, filenames in os.walk(target, followlinks=False):
        root_path = Path(root)
        for name in directories:
            child = root_path / name
            if child.is_symlink():
                total += int(child.lstat().st_size)
        for name in filenames:
            child = root_path / name
            total += int(child.lstat().st_size)
    return total


def basic_cleanup_whitelist_rule(
    path: Path | str,
    *,
    repository_root: Path | str,
    home: Path | str,
) -> str | None:
    """Classify only mechanically provable cleanup-whitelist paths.

    Evidence-dependent cases (duplicate/verified archives and run caches) are
    deliberately not authorized by this helper.  They require manifest/hash
    evidence from the caller before a :class:`CleanupCandidate` is created.
    """

    target = Path(path).expanduser().resolve(strict=False)
    repo = Path(repository_root).expanduser().resolve(strict=False)
    home_path = Path(home).expanduser().resolve(strict=False)
    download_root = repo / "downloads" / "graspnet"
    data_root = repo / "data_external" / "graspnet"

    if target.parent == download_root and target.suffix in {
        ".part",
        ".tmp",
        ".crdownload",
    }:
        return "project_partial_download"
    for name in ("staging", "tmp"):
        allowed_root = data_root / name
        if target == allowed_root or allowed_root in target.parents:
            return f"project_{name}"

    exact_cache_roots = {
        home_path / ".cache" / "pip": "pip_cache",
        home_path / "Library" / "Caches" / "pip": "pip_cache",
        home_path / ".cache" / "uv": "uv_cache",
        home_path / "Library" / "Caches" / "Homebrew": "homebrew_cache",
    }
    for allowed_root, rule in exact_cache_roots.items():
        if target == allowed_root or allowed_root in target.parents:
            return rule
    derived_data = home_path / "Library" / "Developer" / "Xcode" / "DerivedData"
    if target != derived_data and derived_data in target.parents:
        return "xcode_derived_data_content"
    return None


def make_cleanup_candidate(
    path: Path | str,
    *,
    cleanup_type: str,
    reason: str,
    command: Sequence[str],
    whitelist_rule: str,
    reconstructable: bool = True,
) -> CleanupCandidate:
    """Measure a candidate before deletion and make its audit intent explicit."""

    if not whitelist_rule.strip():
        raise ValueError("whitelist_rule must cite the rule that permits cleanup")
    if not reason.strip():
        raise ValueError("cleanup reason must be non-empty")
    if not command:
        raise ValueError("cleanup command must be non-empty")
    if not reconstructable:
        raise ValueError("non-reconstructable files are never cleanup candidates")
    target = Path(path).expanduser().resolve(strict=False)
    return CleanupCandidate(
        path=str(target),
        type=str(cleanup_type),
        size_before_bytes=measure_path_bytes(target),
        reason=str(reason),
        command=tuple(str(item) for item in command),
        reconstructable=True,
        whitelist_rule=str(whitelist_rule),
    )


def write_cleanup_log(
    path: Path | str,
    records: Iterable[CleanupRecord],
    *,
    append: bool = False,
) -> None:
    """Atomically write cleanup outcomes using the mandated CSV columns."""

    output = Path(path)
    rows: list[dict[str, Any]] = []
    if append and output.is_file():
        with output.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CLEANUP_LOG_FIELDS:
                raise ValueError("existing cleanup log has an incompatible schema")
            rows.extend(dict(row) for row in reader)
    rows.extend(asdict(record) for record in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANUP_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def write_cleanup_report(
    path: Path | str,
    records: Sequence[CleanupRecord],
    *,
    initial_free_bytes: int,
    final_free_bytes: int,
) -> None:
    """Atomically render the human-readable cleanup audit."""

    initial = _nonnegative("initial_free_bytes", initial_free_bytes)
    final = _nonnegative("final_free_bytes", final_free_bytes)
    lines = [
        "# Disk cleanup report",
        "",
        f"Initial free bytes: {initial}",
        f"Final free bytes: {final}",
        f"Observed free-space change: {final - initial}",
        "Personal files touched: **No**",
        "",
        "| Timestamp | Path | Type | Bytes before | Result |",
        "|---|---|---|---:|---|",
    ]
    for record in records:
        safe_path = record.path.replace("|", "\\|")
        safe_result = record.result.replace("|", "\\|")
        lines.append(
            f"| {record.timestamp} | `{safe_path}` | {record.type} | "
            f"{record.size_before_bytes} | {safe_result} |"
        )
    if not records:
        lines.append("| — | — | — | 0 | No cleanup action was executed |")
    _atomic_text(Path(path), "\n".join(lines) + "\n")


__all__ = [
    "ArchiveInspectionError",
    "ArchiveInventory",
    "BYTES_PER_DECIMAL_GB",
    "CLEANUP_LOG_FIELDS",
    "CleanupCandidate",
    "CleanupRecord",
    "DEFAULT_MINIMUM_FREE_RESERVE_GB",
    "StagedDiskBudget",
    "basic_cleanup_whitelist_rule",
    "build_staged_disk_budget",
    "inspect_zipinfo_archive",
    "make_cleanup_candidate",
    "measure_path_bytes",
    "write_cleanup_log",
    "write_cleanup_report",
    "write_staged_disk_budget",
]
