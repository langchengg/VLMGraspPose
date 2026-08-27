"""Fail-closed orchestration and official archive handling."""

from __future__ import annotations

import binascii
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .audit import MANDATORY_PAPER_LITE_ARCHIVE_BYTES, repository_root, sha256_file
from .compact_download import (
    OFFICIAL_ARCHIVES,
    ArchiveInventory,
    ArchiveVerification,
    DownloadExhaustedError,
    ExtractionItem,
    ExtractionPlan,
    ExtractionResult,
    adopt_baidu_authenticated_archive,
    build_collision_extraction_plan,
    build_subtree_extraction_plan,
    build_training_extraction_plan,
    deterministic_frame_ids,
    download_archive,
    extract_selective,
    inspect_baidu_manual_archive,
    inspect_archive,
    plan_verified_archive_deletion,
    verify_archive,
    write_inventory_manifest,
)
from .splits import deterministic_stratified_scene_split
from .staged_disk import (
    CleanupRecord,
    build_staged_disk_budget,
    write_cleanup_log,
    write_staged_disk_budget,
)
from .provenance import (
    atomic_json,
    atomic_text,
    record_stage,
    resolve_profile_config,
    update_manifest,
)


@dataclass(frozen=True)
class Archive:
    filename: str
    google_drive_id: str
    expected_bytes: int


PAPER_LITE_ARCHIVES = (
    Archive(
        "train_4.zip",
        "1e8Xy7-lFhiXk0ugPOKvHKDiGTparmx00",
        MANDATORY_PAPER_LITE_ARCHIVE_BYTES["train_4.zip"],
    ),
    Archive(
        "grasp_label.zip",
        "1FCV6j2J2eQpVk_ddJXljJvjRT1KU3sJ6",
        MANDATORY_PAPER_LITE_ARCHIVE_BYTES["grasp_label.zip"],
    ),
    Archive(
        "collision_label.zip",
        "1p43sntiN9HJZRDFDNpzaEaEYoPY6IWsu",
        MANDATORY_PAPER_LITE_ARCHIVE_BYTES["collision_label.zip"],
    ),
    Archive(
        "models.zip",
        "1Gxwu2C5wRQ0QwjdA8CbMXx-bYf_wwPT5",
        MANDATORY_PAPER_LITE_ARCHIVE_BYTES["models.zip"],
    ),
)

MINIMUM_FREE_RESERVE_GB = 20.0
FORMAL_CACHE_ESTIMATE_BYTES = 16_000_000_000


class WorkflowBlocked(RuntimeError):
    """Expected external-data or acceptance gate, never a successful stage."""


_ARCHIVE_PLANNING_BYTES = {
    "train_4.zip": 6_803_985_160,
    "train_3.zip": 20_000_000_000,
    "train_2.zip": 20_000_000_000,
    "grasp_label.zip": 2_059_130_127,
    "collision_label.zip": 441_783_131,
    "models.zip": 4_599_338_858,
    "dex_models.zip": 9_518_063_724,
}


class ArchiveState(str, Enum):
    """Observable state of one independently orchestrated official archive."""

    AVAILABLE = "AVAILABLE"
    DOWNLOADED = "DOWNLOADED"
    VERIFIED = "VERIFIED"
    BLOCKED = "BLOCKED"
    MISSING = "MISSING"
    NOT_REQUIRED_FOR_THIS_RUN = "NOT_REQUIRED_FOR_THIS_RUN"


@dataclass(frozen=True)
class ArchiveStatus:
    """Final state and evidence for one archive in one orchestration pass."""

    filename: str
    state: ArchiveState
    requested: bool
    required_for_smoke: bool
    attempted: bool
    adopted_from_verified_extraction: bool
    path: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {**self.__dict__, "state": self.state.value}


@dataclass(frozen=True)
class ArchiveOrchestration:
    """Independent download results plus in-memory verified objects."""

    profile: str
    statuses: tuple[ArchiveStatus, ...]
    verifications: Mapping[str, ArchiveVerification]
    inventories: Mapping[str, ArchiveInventory]
    unique_train_3_train_4_scenes: int
    train_2_condition_met: bool

    @property
    def status_by_filename(self) -> dict[str, ArchiveStatus]:
        return {status.filename: status for status in self.statuses}

    @property
    def readiness_status(self) -> str:
        required = [status for status in self.statuses if status.required_for_smoke]
        return (
            "READY_FROM_VERIFIED_LOCAL_ARCHIVES"
            if required
            and all(status.state is ArchiveState.VERIFIED for status in required)
            else "REQUIRED_ARCHIVES_NOT_READY"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": "graspnet6d_archive_orchestration_v1",
            "profile": self.profile,
            "unique_train_3_train_4_scenes": self.unique_train_3_train_4_scenes,
            "train_2_condition_met": self.train_2_condition_met,
            "readiness_status": self.readiness_status,
            "states": [status.to_record() for status in self.statuses],
        }


_INDEPENDENT_ARCHIVES = (
    "grasp_label.zip",
    "collision_label.zip",
    "models.zip",
    "dex_models.zip",
    "train_4.zip",
    "train_3.zip",
)
_SMOKE_REQUIRED_ARCHIVES = (
    "train_4.zip",
    "grasp_label.zip",
    "collision_label.zip",
    "models.zip",
)
_ALL_ORCHESTRATED_ARCHIVES = (*_INDEPENDENT_ARCHIVES, "train_2.zip")


@dataclass(frozen=True)
class _ProfileContract:
    profile: str
    required_scene_archives: tuple[str, ...]
    requested_archives: tuple[str, ...]
    required_archives: tuple[str, ...]
    minimum_unique_scenes: int
    split_counts: tuple[int, int, int]
    split_filename: str
    camera: str
    frames_per_scene: int
    source_archive: str | None


def _profile_contract(profile: str) -> _ProfileContract:
    config, _ = resolve_profile_config(profile)
    dataset = config.get("dataset", {})
    split = config.get("split", {})
    sampling = config.get("sampling", {})
    if not isinstance(dataset, dict) or not isinstance(split, dict):
        raise WorkflowBlocked(f"invalid dataset/split mapping in profile {profile}")
    default_scene = {
        "smoke": ("train_4.zip",),
        "paper-lite": ("train_4.zip", "train_3.zip"),
        "paper-extended": ("train_4.zip", "train_3.zip"),
    }
    required_scene = tuple(
        str(value)
        for value in dataset.get(
            "required_scene_archives", default_scene.get(profile, ())
        )
    )
    if not required_scene or any(
        value not in {"train_2.zip", "train_3.zip", "train_4.zip"}
        for value in required_scene
    ):
        raise WorkflowBlocked(
            f"profile {profile} has invalid required_scene_archives"
        )
    required_auxiliary = (
        "grasp_label.zip",
        "collision_label.zip",
        "models.zip",
    )
    configured_archives = tuple(str(value) for value in dataset.get("archives", ()))
    missing_auxiliary = [
        value for value in required_auxiliary if value not in configured_archives
    ]
    if missing_auxiliary:
        raise WorkflowBlocked(
            f"profile {profile} omits required evaluator archives: {missing_auxiliary}"
        )
    optional = tuple(str(value) for value in dataset.get("optional_archives", ()))
    # The historical smoke route attempted dex_models opportunistically; keep
    # that behavior while allowing formal profiles to declare it explicitly.
    if profile == "smoke" and "dex_models.zip" not in optional:
        optional = (*optional, "dex_models.zip")
    requested_set = set(required_scene) | set(required_auxiliary) | set(optional)
    unknown = requested_set - set(_ALL_ORCHESTRATED_ARCHIVES)
    if unknown:
        raise WorkflowBlocked(f"profile {profile} names unknown archives: {sorted(unknown)}")
    counts = (
        int(split.get("train_scenes", 1 if profile == "smoke" else 20)),
        int(split.get("validation_scenes", 1 if profile == "smoke" else 5)),
        int(split.get("test_scenes", 1 if profile == "smoke" else 10)),
    )
    if any(value <= 0 for value in counts):
        raise WorkflowBlocked(f"profile {profile} split counts must be positive")
    required_count = sum(counts)
    minimum_unique = int(dataset.get("minimum_unique_scenes", required_count))
    if minimum_unique != required_count:
        raise WorkflowBlocked(
            f"profile {profile} minimum_unique_scenes must equal its locked split size"
        )
    default_split_name = (
        "graspnet_scene_split_smoke_v2.json"
        if profile == "smoke"
        else "graspnet_scene_split_v2.json"
    )
    camera = str(dataset.get("camera", config.get("camera", "")))
    if camera not in {"kinect", "realsense"}:
        raise WorkflowBlocked(f"profile {profile} camera is invalid: {camera!r}")
    frames = int(
        sampling.get("frames_per_scene", dataset.get("frames_per_scene", 16))
        if isinstance(sampling, dict)
        else dataset.get("frames_per_scene", 16)
    )
    experiment_scope = config.get("experiment_scope", {})
    source_archive = (
        str(experiment_scope.get("source_archive"))
        if isinstance(experiment_scope, dict)
        and experiment_scope.get("source_archive") is not None
        else None
    )
    return _ProfileContract(
        profile=profile,
        required_scene_archives=required_scene,
        requested_archives=tuple(
            name for name in _INDEPENDENT_ARCHIVES if name in requested_set
        ),
        required_archives=(*required_scene, *required_auxiliary),
        minimum_unique_scenes=minimum_unique,
        split_counts=counts,
        split_filename=str(split.get("manifest_filename", default_split_name)),
        camera=camera,
        frames_per_scene=frames,
        source_archive=source_archive,
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_budget(
    root: Path,
    *,
    phase: str,
    archive_bytes: int,
    selected_extraction_bytes: int = 0,
    temporary_bytes: int = 0,
    estimated_cache_bytes: int = 0,
) -> dict[str, Any]:
    budget = build_staged_disk_budget(
        root,
        phase=phase,
        archive_bytes=archive_bytes,
        selected_extraction_bytes=selected_extraction_bytes,
        temporary_bytes=temporary_bytes,
        estimated_cache_bytes=estimated_cache_bytes,
        minimum_free_reserve_gb=MINIMUM_FREE_RESERVE_GB,
    )
    audit = root / "artifacts" / "graspnet6d" / "audit"
    write_staged_disk_budget(
        budget,
        json_path=audit / "staged_disk_budget.json",
        markdown_path=audit / "staged_disk_budget.md",
    )
    _append_jsonl(
        audit / "staged_disk_budget_history.jsonl",
        {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **budget.to_record(),
        },
    )
    if not budget.allowed:
        raise WorkflowBlocked(
            f"staged disk budget failed for {phase}: "
            f"free={budget.current_free_bytes}, required={budget.required_total_bytes}, "
            f"deficit={budget.deficit_bytes}; the 20 GB reserve is mandatory"
        )
    return budget.to_record()


def _official_network_attempts_exhausted(root: Path, filename: str) -> bool:
    """Return true once any complete 12-attempt Google/JBox batch is durable."""

    path = root / "artifacts/graspnet6d/audit/download_log.jsonl"
    if not path.is_file() or path.is_symlink():
        return False
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(row, dict)
            or row.get("filename") != filename
            or row.get("source") not in {"google_drive", "jbox"}
        ):
            continue
        try:
            attempt = int(row.get("attempt"))
        except (TypeError, ValueError):
            continue
        if attempt == 1:
            if current:
                batches.append(current)
            current = []
        if current or attempt == 1:
            current.append(row)
    if current:
        batches.append(current)
    if not batches:
        return False
    return any(
        [int(row["attempt"]) for row in batch] == list(range(1, 13))
        and all(row.get("status") == "RETRYABLE_FAILURE" for row in batch)
        for batch in batches
    )


def _download_one(
    root: Path,
    filename: str,
) -> ArchiveVerification:
    spec = OFFICIAL_ARCHIVES[filename]
    destination = root / "downloads" / "graspnet" / filename
    present_bytes = (
        int(destination.stat().st_size)
        if destination.is_file() and not destination.is_symlink()
        else 0
    )
    download_root = root / "downloads" / "graspnet"
    log_path = root / "artifacts" / "graspnet6d" / "audit" / "download_log.jsonl"
    manifest_path = download_root / "download_manifest.json"
    checksum_path = download_root / "archive_checksums.sha256"
    existing_row = _download_manifest_row(root, filename)
    if existing_row is not None and destination.is_file() and not destination.is_symlink():
        verification = verify_archive(
            destination,
            minimum_plausible_bytes=spec.minimum_plausible_bytes,
            expected_bytes=spec.expected_bytes,
        )
        if (
            verification.bytes != int(existing_row["bytes"])
            or verification.sha256 != str(existing_row["sha256"])
        ):
            raise WorkflowBlocked(
                f"verified local archive no longer matches its download manifest: {filename}"
            )
        # Reuse is deliberately read-only: do not relabel an existing Google
        # or JBox transfer as a manually authenticated Baidu download.
        return verification
    if destination.is_file() and not destination.is_symlink() and spec.baidu_url:
        local = inspect_baidu_manual_archive(spec, download_root)
        if local.status == "READY_FOR_ADOPTION":
            _write_budget(
                root,
                phase=f"adopt exact-name official Baidu {filename}",
                archive_bytes=0,
            )
            return adopt_baidu_authenticated_archive(
                spec,
                download_root,
                authenticated_download_confirmed=True,
                manifest_path=manifest_path,
                checksum_path=checksum_path,
                log_path=log_path,
            )
        if local.status in {"AUTH_RESPONSE_REJECTED", "INVALID_LOCAL_ARCHIVE"}:
            raise WorkflowBlocked(
                "WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD: refusing to use or "
                f"overwrite the exact-name local file for {filename}; Baidu "
                f"manual state is {local.status}: {local.detail}; "
                f"URL={local.official_baidu_url}, expected_path={local.expected_path}"
            )
    if _official_network_attempts_exhausted(root, filename):
        if not spec.baidu_url:
            raise WorkflowBlocked(
                "WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD: Google Drive/JBox "
                f"attempts are already exhausted for {filename}; no locked official "
                "Baidu fallback is available"
            )
        manual = inspect_baidu_manual_archive(spec, download_root)
        if manual.status != "READY_FOR_ADOPTION":
            raise WorkflowBlocked(
                "WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD: Google Drive/JBox "
                f"attempts are already exhausted for {filename}; "
                f"official Baidu manual state is {manual.status}: {manual.detail}; "
                f"URL={manual.official_baidu_url}, expected_path={manual.expected_path}"
            )
        _write_budget(
            root,
            phase=f"adopt authenticated official Baidu {filename}",
            archive_bytes=0,
        )
        return adopt_baidu_authenticated_archive(
            spec,
            download_root,
            authenticated_download_confirmed=True,
            manifest_path=manifest_path,
            checksum_path=checksum_path,
            log_path=log_path,
        )
    _write_budget(
        root,
        phase=f"download {filename}",
        archive_bytes=max(0, _ARCHIVE_PLANNING_BYTES[filename] - present_bytes),
    )
    try:
        return download_archive(
            spec,
            download_root,
            manifest_path=manifest_path,
            checksum_path=checksum_path,
            log_path=log_path,
            python_executable=sys.executable,
        )
    except DownloadExhaustedError as error:
        if not spec.baidu_url:
            raise WorkflowBlocked(
                "WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD: the current "
                f"Google Drive/JBox batch exhausted for {filename}; no locked "
                "official Baidu fallback is available"
            ) from error
        manual = inspect_baidu_manual_archive(spec, download_root)
        if manual.status == "READY_FOR_ADOPTION":
            return adopt_baidu_authenticated_archive(
                spec,
                download_root,
                authenticated_download_confirmed=True,
                manifest_path=manifest_path,
                checksum_path=checksum_path,
                log_path=log_path,
            )
        raise WorkflowBlocked(
            "WAITING_FOR_AUTHENTICATED_BAIDU_DOWNLOAD: the current Google "
            f"Drive/JBox batch exhausted for {filename}; official Baidu manual "
            f"state is {manual.status}: {manual.detail}; "
            f"URL={manual.official_baidu_url}, expected_path={manual.expected_path}"
        ) from error


def _inspect_training_archive(
    root: Path, verification: ArchiveVerification
) -> ArchiveInventory:
    archive = Path(verification.path)
    inventory = inspect_archive(
        archive,
        filelist_path=archive.with_suffix(archive.suffix + ".filelist.txt"),
    )
    if inventory.archive_sha256 != verification.sha256:
        raise WorkflowBlocked(
            f"archive changed between verification and listing: {archive}"
        )
    return inventory


def _split_payload(
    inventories: list[ArchiveInventory], *, profile: str, seed: int
) -> dict[str, Any]:
    contract = _profile_contract(profile)
    objects: dict[str, tuple[int, ...]] = {}
    sources: dict[str, str] = {}
    for inventory in inventories:
        for scene in inventory.scenes:
            key = f"scene_{scene.scene_id:04d}"
            if key in objects:
                raise WorkflowBlocked(
                    f"duplicate scene across training archives: {key}"
                )
            if not scene.has_object_id_list or not scene.has_rs_wrt_kn:
                raise WorkflowBlocked(f"scene-level metadata is incomplete in {key}")
            objects[key] = scene.object_ids
            sources[key] = Path(inventory.archive_path).name
    counts = contract.split_counts
    split, unused = deterministic_stratified_scene_split(
        objects,
        train_count=counts[0],
        validation_count=counts[1],
        test_count=counts[2],
        seed=seed,
        source_profile="compact GraspNet training-scene split",
    )

    def distribution(scene_ids: tuple[str, ...]) -> dict[str, int]:
        result: dict[str, int] = {}
        for scene_id in scene_ids:
            for object_id in objects[scene_id]:
                key = str(object_id)
                result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items(), key=lambda item: int(item[0])))

    return {
        "schema_version": "graspnet6d_scene_split_v2",
        "seed": seed,
        "source_profile": split.source_profile,
        "profile": profile,
        "required_scene_archives": list(contract.required_scene_archives),
        "official_test_benchmark": False,
        "selection_strategy": "deterministic_multilabel_object_deficit_stratification",
        "counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "unused": len(unused),
        },
        "train": list(split.train),
        "validation": list(split.validation),
        "test": list(split.test),
        "unused": list(unused),
        "scene_sources": {key: sources[key] for key in sorted(sources)},
        "scene_object_ids": {key: list(objects[key]) for key in sorted(objects)},
        "object_distribution": {
            "train": distribution(split.train),
            "validation": distribution(split.validation),
            "test": distribution(split.test),
        },
    }


def _validate_extracted_files(plan: Any, data_root: Path) -> bool:
    import xml.etree.ElementTree as ET

    import numpy as np
    from PIL import Image
    from scipy.io import loadmat

    for item in plan.items:
        path = data_root / item.destination_relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item.bytes:
            return False
        suffix = path.suffix.lower()
        try:
            if suffix == ".npy":
                value = np.load(path, allow_pickle=False)
                if value.size == 0 or not np.isfinite(value).all():
                    return False
            elif suffix == ".npz":
                with np.load(path, allow_pickle=False) as value:
                    if not value.files:
                        return False
            elif suffix == ".png":
                with Image.open(path) as image:
                    image.verify()
            elif suffix == ".mat":
                if not loadmat(path):
                    return False
            elif suffix == ".xml":
                ET.parse(path)
            elif suffix in {".ply", ".obj", ".sdf", ".txt"}:
                if path.stat().st_size <= 0:
                    return False
        except Exception:
            return False
    return True


def _validate_grasp_assets(data_root: Path, object_ids: set[int]) -> None:
    import numpy as np

    for object_id in sorted(object_ids):
        path = data_root / "grasp_label" / f"{object_id:03d}_labels.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                if not {"points", "offsets", "scores"}.issubset(archive.files):
                    raise ValueError("required arrays are absent")
                points = np.asarray(archive["points"])
                offsets = np.asarray(archive["offsets"])
                scores = np.asarray(archive["scores"])
                if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                    raise ValueError(f"invalid sampled points shape {points.shape}")
                if offsets.shape[0] != len(points) or scores.shape[0] != len(points):
                    raise ValueError("grasp arrays disagree on sampled-point axis")
                if not all(
                    np.isfinite(value).all() for value in (points, offsets, scores)
                ):
                    raise ValueError("grasp arrays contain NaN or Inf")
        except Exception as error:
            raise WorkflowBlocked(
                f"grasp-label loader rejected object {object_id:03d}: {error}"
            ) from error


def _validate_model_assets(data_root: Path, object_ids: set[int]) -> None:
    for object_id in sorted(object_ids):
        model = data_root / "models" / f"{object_id:03d}"
        required = (
            model / "nontextured.ply",
            model / "textured.obj",
            model / "textured.sdf",
        )
        for path in required:
            if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
                raise WorkflowBlocked(
                    f"official evaluator model source is missing/empty: {path}"
                )
        if not (model / "nontextured.ply").read_bytes()[:16].lower().startswith(b"ply"):
            raise WorkflowBlocked(f"invalid PLY header: {model / 'nontextured.ply'}")


def _validate_dex_assets(data_root: Path, object_ids: set[int]) -> None:
    import pickletools

    for object_id in sorted(object_ids):
        path = data_root / "dex_models" / f"{object_id:03d}.pkl"
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise WorkflowBlocked(f"Dex-Net cache is missing/empty: {path}")
        try:
            with path.open("rb") as stream:
                saw_stop = any(opcode.name == "STOP" for opcode, _, _ in pickletools.genops(stream))
        except Exception as error:
            raise WorkflowBlocked(
                f"Dex-Net pickle structure is invalid for object {object_id:03d}: {error}"
            ) from error
        if not saw_stop:
            raise WorkflowBlocked(f"Dex-Net pickle lacks STOP opcode: {path}")


def _validate_collision_assets(
    data_root: Path, scene_object_ids: Mapping[int, Sequence[int]]
) -> None:
    import numpy as np

    for scene_id, object_ids in sorted(scene_object_ids.items()):
        path = data_root / "collision_label" / f"scene_{scene_id:04d}" / "collision_labels.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                expected = tuple(f"arr_{index}" for index in range(len(object_ids)))
                if set(archive.files) != set(expected) or len(archive.files) != len(
                    expected
                ):
                    raise ValueError(
                        f"expected array set {expected}, observed {tuple(archive.files)}"
                    )
                for name in expected:
                    value = np.asarray(archive[name])
                    if value.size == 0 or not np.isfinite(value).all():
                        raise ValueError(f"{name} is empty or non-finite")
        except Exception as error:
            raise WorkflowBlocked(
                f"collision-label loader rejected scene_{scene_id:04d}: {error}"
            ) from error


def _delete_verified_archive(
    root: Path,
    verification: ArchiveVerification,
    extraction: Any,
    *,
    downstream_loader_verified: bool,
) -> dict[str, Any]:
    deletion = plan_verified_archive_deletion(
        verification,
        extraction,
        downstream_loader_verified=downstream_loader_verified,
    )
    if not deletion.eligible:
        raise WorkflowBlocked(deletion.reason)
    path = Path(deletion.path)
    free_before = int(shutil.disk_usage(root).free)
    command = ("/bin/rm", "--", str(path))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    free_after = int(shutil.disk_usage(root).free)
    result = (
        "SUCCESS: verified archive removed"
        if completed.returncode == 0 and not path.exists()
        else f"FAILED: {completed.stderr.strip() or completed.stdout.strip()}"
    )
    record = CleanupRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        path=str(path),
        type="verified_official_archive",
        size_before_bytes=deletion.size_before_bytes,
        reason=deletion.reason,
        command=" ".join(command),
        reconstructable=True,
        result=result,
        free_space_before=free_before,
        free_space_after=free_after,
    )
    write_cleanup_log(
        root / "artifacts/graspnet6d/audit/disk_cleanup_log.csv",
        [record],
        append=True,
    )
    _refresh_cleanup_report(root)
    if completed.returncode != 0 or path.exists():
        raise OSError(f"verified archive deletion failed: {path}: {result}")
    return deletion.to_record()


def _extract_plan(
    root: Path,
    filename: str,
    verification: ArchiveVerification,
    plan: Any,
    *,
    semantic_validator: Callable[[Path], None] | None = None,
    delete_archive: bool = True,
) -> dict[str, Any]:
    data_root = root / "data_external" / "graspnet"
    marker = data_root / ".extracted" / f"{filename}.json"
    _write_budget(
        root,
        phase=f"selective extract {filename}",
        archive_bytes=0,
        selected_extraction_bytes=int(plan.selected_bytes),
        temporary_bytes=max((int(item.bytes) for item in plan.items), default=0),
    )
    extraction = extract_selective(
        plan,
        data_root,
        staging_root=data_root / "staging" / filename,
        manifest_path=marker,
        resume=True,
    )
    loader_passed = _validate_extracted_files(plan, data_root)
    if not loader_passed:
        raise WorkflowBlocked(
            f"downstream loaders rejected selected files from {filename}"
        )
    if semantic_validator is not None:
        semantic_validator(data_root)
    deletion = (
        _delete_verified_archive(
            root,
            verification,
            extraction,
            downstream_loader_verified=True,
        )
        if delete_archive
        else {
            "eligible": False,
            "path": verification.path,
            "size_before_bytes": verification.bytes,
            "reason": "deferred pending adaptive target-group validation",
            "reconstructable": True,
            "status": "DEFERRED_PENDING_TARGET_GROUP_GATE",
        }
    )
    return {
        "filename": filename,
        "verification": verification.to_record(),
        "extraction": extraction.to_record(),
        "archive_deletion": deletion,
    }


def _read_complete_marker(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if payload.get("complete") is True else None


def _refresh_cleanup_report(root: Path) -> None:
    log_path = root / "artifacts/graspnet6d/audit/disk_cleanup_log.csv"
    if not log_path.is_file():
        return
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    successful = [
        row for row in rows if str(row.get("result", "")).startswith("SUCCESS")
    ]
    initial_free = int(rows[0]["free_space_before"])
    final_free = int(rows[-1]["free_space_after"])
    rendered = [
        "# Disk cleanup report",
        "",
        "Status: **COMPLETE**",
        "",
        "Every action below is restricted to the user-authorised reconstructable "
        "cache/archive whitelist and is recorded in `disk_cleanup_log.csv`.",
        "",
        f"Initial free space: `{initial_free}` bytes.",
        f"Current recorded free space: `{final_free}` bytes.",
        f"Measured filesystem change: `{final_free - initial_free}` bytes.",
        "",
        "## Successful actions",
        "",
    ]
    rendered.extend(
        f"- `{row['path']}` — {row['size_before_bytes']} bytes; {row['result']}"
        for row in successful
    )
    rendered.extend(
        [
            "",
            "Failed/no-op attempts remain in the CSV and removed nothing.",
            "",
            "Personal files touched: **No**. Git-tracked files, dirty-worktree "
            "changes, checkpoints, prior results, model caches, Trash, Desktop, "
            "Documents, Pictures, Movies, Music, iCloud Drive, mail, Docker volumes, "
            "and unrelated repositories were not removed.",
        ]
    )
    atomic_text(
        root / "artifacts/graspnet6d/audit/disk_cleanup_report.md",
        "\n".join(rendered) + "\n",
    )


def _download_manifest_row(root: Path, filename: str) -> dict[str, Any] | None:
    matches = [
        row for row in _download_manifest_rows(root) if row.get("filename") == filename
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise WorkflowBlocked(f"duplicate download-manifest rows for {filename}")
    row = matches[0]
    sha = row.get("sha256")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
        or row.get("zip_test_passed") is not True
        or int(row.get("bytes", -1)) <= 0
    ):
        raise WorkflowBlocked(f"invalid verified download-manifest row for {filename}")
    return row


def _marker_plan(payload: dict[str, Any], *, filename: str) -> ExtractionPlan:
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        raise WorkflowBlocked(f"extraction marker lacks a plan for {filename}")
    try:
        items = tuple(ExtractionItem(**item) for item in plan["items"])
        parsed = ExtractionPlan(
            archive_path=str(plan["archive_path"]),
            archive_sha256=str(plan["archive_sha256"]),
            kind=str(plan["kind"]),
            items=items,
            selected_scene_ids=tuple(
                int(value) for value in plan.get("selected_scene_ids", ())
            ),
            camera=plan.get("camera"),
            frame_ids=tuple(int(value) for value in plan.get("frame_ids", ())),
            subtree=plan.get("subtree"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WorkflowBlocked(
            f"invalid extraction marker plan for {filename}: {error}"
        ) from error
    if Path(parsed.archive_path).name != filename or not parsed.items:
        raise WorkflowBlocked(
            f"extraction marker targets the wrong/empty archive: {filename}"
        )
    if payload.get("selected_file_count") != len(parsed.items):
        raise WorkflowBlocked(f"extraction marker count mismatch for {filename}")
    if payload.get("verified_file_count") != len(parsed.items):
        raise WorkflowBlocked(f"extraction marker is not fully verified for {filename}")
    return parsed


def _crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value = binascii.crc32(block, value)
    return value & 0xFFFFFFFF


def _validate_committed_extraction(
    root: Path, filename: str
) -> tuple[dict[str, Any], ExtractionPlan] | None:
    data_root = root / "data_external/graspnet"
    marker_path = data_root / ".extracted" / f"{filename}.json"
    payload = _read_complete_marker(marker_path)
    if payload is None:
        return None
    plan = _marker_plan(payload, filename=filename)
    row = _download_manifest_row(root, filename)
    if row is None or row["sha256"] != plan.archive_sha256:
        raise WorkflowBlocked(f"extraction/download SHA-256 mismatch for {filename}")
    for item in plan.items:
        path = data_root / item.destination_relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.bytes
            or _crc32(path) != item.crc32
        ):
            raise WorkflowBlocked(f"committed compact output is stale: {path}")
    if not _validate_extracted_files(plan, data_root):
        raise WorkflowBlocked(f"committed compact outputs fail loaders for {filename}")
    return payload, plan


def _verified_archive_was_deleted(root: Path, filename: str) -> bool:
    archive = (root / "downloads/graspnet" / filename).resolve()
    if archive.exists():
        return False
    log_path = root / "artifacts/graspnet6d/audit/disk_cleanup_log.csv"
    if not log_path.is_file():
        return False
    with log_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                recorded = Path(str(row.get("path", ""))).resolve()
            except (OSError, RuntimeError):
                continue
            if (
                recorded == archive
                and row.get("type") == "verified_official_archive"
                and row.get("reconstructable", "").lower() == "true"
                and str(row.get("result", "")).startswith("SUCCESS")
            ):
                return True
    return False


def _resumed_extraction_record(
    root: Path, filename: str, marker: dict[str, Any], plan: ExtractionPlan
) -> dict[str, Any]:
    if not _verified_archive_was_deleted(root, filename):
        raise WorkflowBlocked(
            f"{filename} is absent but no successful audited archive deletion exists"
        )
    marker_path = root / "data_external/graspnet/.extracted" / f"{filename}.json"
    row = _download_manifest_row(root, filename)
    assert row is not None
    return {
        "filename": filename,
        "verification": row,
        "extraction": {
            **marker,
            "manifest_path": str(marker_path),
            "manifest_sha256": sha256_file(marker_path),
            "resumed_from_verified_commit": True,
        },
        "archive_deletion": {
            "eligible": True,
            "path": str((root / "downloads/graspnet" / filename).resolve()),
            "size_before_bytes": int(row["bytes"]),
            "reason": "previously removed after verified compact extraction",
            "reconstructable": True,
            "resumed_from_cleanup_audit": True,
            "archive_sha256": plan.archive_sha256,
        },
    }


def _download_manifest_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "downloads/graspnet/download_manifest.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("archives", [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise WorkflowBlocked(f"invalid download manifest: {path}")
    return list(rows)


def _split_manifest_path(root: Path, profile: str) -> Path:
    return root / "configs/graspnet6d/splits" / _profile_contract(profile).split_filename


def _load_locked_split(path: Path, *, profile: str) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkflowBlocked(
            f"invalid compact split manifest {path}: {error}"
        ) from error
    contract = _profile_contract(profile)
    expected = contract.split_counts
    if (
        payload.get("schema_version") != "graspnet6d_scene_split_v2"
        or payload.get("seed") != 20260815
        or not isinstance(payload.get("scene_sources"), dict)
        or not isinstance(payload.get("scene_object_ids"), dict)
    ):
        raise WorkflowBlocked(f"compact split manifest has the wrong contract: {path}")
    seen: set[str] = set()
    for partition, count in zip(("train", "validation", "test"), expected):
        values = payload.get(partition)
        if not isinstance(values, list) or len(values) != count:
            raise WorkflowBlocked(
                f"compact split {partition} count is not locked to {count}: {path}"
            )
        for value in values:
            if (
                not isinstance(value, str)
                or not value.startswith("scene_")
                or value in seen
                or value not in payload["scene_sources"]
                or value not in payload["scene_object_ids"]
            ):
                raise WorkflowBlocked(f"invalid or overlapping split scene {value!r}")
            source = payload["scene_sources"][value]
            approved_sources = set(contract.required_scene_archives)
            if profile != "paper-lite-train3":
                approved_sources.add("train_2.zip")
            if source not in approved_sources:
                raise WorkflowBlocked(f"unapproved training archive in split: {source}")
            seen.add(value)
    if len(seen) != contract.minimum_unique_scenes:
        raise WorkflowBlocked(
            f"compact split union is not locked to {contract.minimum_unique_scenes} scenes"
        )
    return payload


def _selected_scene_ids(split_payload: dict[str, Any]) -> set[int]:
    return {
        int(scene_id.split("_")[1])
        for partition in ("train", "validation", "test")
        for scene_id in split_payload[partition]
    }


def _training_archive_names(split_payload: dict[str, Any]) -> list[str]:
    selected_names = {
        str(split_payload["scene_sources"][scene_id])
        for partition in ("train", "validation", "test")
        for scene_id in split_payload[partition]
    }
    return [
        filename
        for filename in ("train_4.zip", "train_3.zip", "train_2.zip")
        if filename in selected_names
    ]


def _archive_state_paths(root: Path, run_dir: Path | None) -> tuple[Path, ...]:
    paths = [root / "downloads/graspnet/archive_orchestration.json"]
    if run_dir is not None:
        paths.append(run_dir / "archive_download_states.json")
    return tuple(paths)


def _prior_archive_transitions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    rows = payload.get("transitions", [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return []
    return list(rows)


def _write_archive_state_checkpoint(
    root: Path,
    run_dir: Path | None,
    *,
    profile: str,
    statuses: Mapping[str, ArchiveStatus],
    transitions: Sequence[Mapping[str, Any]],
    complete: bool,
    unique_train_3_train_4_scenes: int | None = None,
    train_2_condition_met: bool | None = None,
) -> None:
    payload = {
        "schema_version": "graspnet6d_archive_orchestration_v1",
        "profile": profile,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": bool(complete),
        "unique_train_3_train_4_scenes": unique_train_3_train_4_scenes,
        "train_2_condition_met": train_2_condition_met,
        "states": [
            statuses[filename].to_record() for filename in _ALL_ORCHESTRATED_ARCHIVES
        ],
        "transitions": [dict(row) for row in transitions],
    }
    for path in _archive_state_paths(root, run_dir):
        atomic_json(path, payload)


def _selected_train_3_train_4_count_from_split(
    root: Path, *, profile: str
) -> int | None:
    split = _load_locked_split(_split_manifest_path(root, profile), profile=profile)
    if split is None:
        return None
    selected = [
        scene_id
        for partition in ("train", "validation", "test")
        for scene_id in split[partition]
    ]
    return sum(
        str(split["scene_sources"].get(scene_id)) in {"train_3.zip", "train_4.zip"}
        for scene_id in selected
    )


def orchestrate_archive_downloads(
    profile: str,
    *,
    root: Path | None = None,
    run_dir: Path | None = None,
) -> ArchiveOrchestration:
    """Attempt independent official archives and persist every terminal state.

    Download/verification failure of one archive is captured as evidence and
    never aborts attempts for the remaining archives.  This function does not
    enforce the extraction/smoke prerequisites; callers do that only after all
    independent attempts finish via :func:`require_verified_archives`.
    """

    contract = _profile_contract(profile)
    source_root = (root or repository_root()).resolve()
    requested = set(contract.requested_archives)
    required = set(contract.required_archives)
    statuses: dict[str, ArchiveStatus] = {}
    for filename in _ALL_ORCHESTRATED_ARCHIVES:
        is_requested = filename in requested
        statuses[filename] = ArchiveStatus(
            filename=filename,
            state=(
                ArchiveState.NOT_REQUIRED_FOR_THIS_RUN
                if profile == "paper-lite-train3" and not is_requested
                else ArchiveState.MISSING
                if filename == "train_2.zip" and not is_requested
                else ArchiveState.AVAILABLE
                if filename in OFFICIAL_ARCHIVES
                else ArchiveState.MISSING
            ),
            requested=is_requested,
            required_for_smoke=filename in required,
            attempted=False,
            adopted_from_verified_extraction=False,
        )
    state_path = source_root / "downloads/graspnet/archive_orchestration.json"
    # Per-run evidence contains only actions made for that immutable run.
    # Global prior transitions are operational history, not child-run attempts.
    transitions = [] if run_dir is not None else _prior_archive_transitions(state_path)

    def transition(status: ArchiveStatus) -> None:
        statuses[status.filename] = status
        transitions.append(
            {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                **status.to_record(),
            }
        )
        _write_archive_state_checkpoint(
            source_root,
            run_dir,
            profile=profile,
            statuses=statuses,
            transitions=transitions,
            complete=False,
        )

    _write_archive_state_checkpoint(
        source_root,
        run_dir,
        profile=profile,
        statuses=statuses,
        transitions=transitions,
        complete=False,
    )
    verifications: dict[str, ArchiveVerification] = {}
    inventories: dict[str, ArchiveInventory] = {}

    def attempt(filename: str) -> None:
        current = statuses[filename]
        archive_path = source_root / "downloads/graspnet" / filename
        try:
            committed = _validate_committed_extraction(source_root, filename)
            if committed is not None and _verified_archive_was_deleted(
                source_root, filename
            ):
                row = _download_manifest_row(source_root, filename)
                assert row is not None
                transition(
                    ArchiveStatus(
                        filename=filename,
                        state=ArchiveState.VERIFIED,
                        requested=True,
                        required_for_smoke=current.required_for_smoke,
                        attempted=False,
                        adopted_from_verified_extraction=True,
                        path=str(archive_path.resolve()),
                        bytes=int(row["bytes"]),
                        sha256=str(row["sha256"]),
                    )
                )
                return
        except Exception as error:
            transition(
                ArchiveStatus(
                    filename=filename,
                    state=ArchiveState.BLOCKED,
                    requested=True,
                    required_for_smoke=current.required_for_smoke,
                    attempted=True,
                    adopted_from_verified_extraction=False,
                    path=str(archive_path.resolve()),
                    error=f"{type(error).__name__}: {error}",
                )
            )
            return

        try:
            verification = _download_one(source_root, filename)
            verifications[filename] = verification
            transition(
                ArchiveStatus(
                    filename=filename,
                    state=ArchiveState.DOWNLOADED,
                    requested=True,
                    required_for_smoke=current.required_for_smoke,
                    attempted=True,
                    adopted_from_verified_extraction=False,
                    path=verification.path,
                    bytes=verification.bytes,
                    sha256=verification.sha256,
                )
            )
            if filename in {"train_2.zip", "train_3.zip", "train_4.zip"}:
                inventories[filename] = _inspect_training_archive(
                    source_root, verification
                )
            transition(
                ArchiveStatus(
                    filename=filename,
                    state=ArchiveState.VERIFIED,
                    requested=True,
                    required_for_smoke=current.required_for_smoke,
                    attempted=True,
                    adopted_from_verified_extraction=False,
                    path=verification.path,
                    bytes=verification.bytes,
                    sha256=verification.sha256,
                )
            )
        except Exception as error:
            present = archive_path.is_file() and not archive_path.is_symlink()
            terminal_state = (
                ArchiveState.BLOCKED
                if isinstance(error, WorkflowBlocked)
                else ArchiveState.DOWNLOADED
                if present
                else ArchiveState.BLOCKED
            )
            transition(
                ArchiveStatus(
                    filename=filename,
                    state=terminal_state,
                    requested=True,
                    required_for_smoke=current.required_for_smoke,
                    attempted=True,
                    adopted_from_verified_extraction=False,
                    path=str(archive_path.resolve()),
                    bytes=int(archive_path.stat().st_size) if present else None,
                    sha256=(
                        verifications[filename].sha256
                        if filename in verifications
                        else None
                    ),
                    error=f"{type(error).__name__}: {error}",
                )
            )

    for filename in contract.requested_archives:
        attempt(filename)

    observed_scenes = {
        scene.scene_id
        for filename, inventory in inventories.items()
        if filename in set(contract.required_scene_archives)
        for scene in inventory.scenes
    }
    persisted_count = _selected_train_3_train_4_count_from_split(
        source_root, profile=profile
    )
    unique_train_3_train_4 = (
        len(observed_scenes) if observed_scenes else int(persisted_count or 0)
    )
    training_pair_verified = all(
        statuses[filename].state is ArchiveState.VERIFIED
        for filename in ("train_3.zip", "train_4.zip")
    )
    train_2_condition_met = (
        profile == "paper-lite"
        and training_pair_verified
        and unique_train_3_train_4 < 35
    )
    if train_2_condition_met:
        statuses["train_2.zip"] = ArchiveStatus(
            filename="train_2.zip",
            state=ArchiveState.AVAILABLE,
            requested=True,
            required_for_smoke=False,
            attempted=False,
            adopted_from_verified_extraction=False,
        )
        attempt("train_2.zip")

    result = ArchiveOrchestration(
        profile=profile,
        statuses=tuple(statuses[name] for name in _ALL_ORCHESTRATED_ARCHIVES),
        verifications=dict(verifications),
        inventories=dict(inventories),
        unique_train_3_train_4_scenes=unique_train_3_train_4,
        train_2_condition_met=train_2_condition_met,
    )
    _write_archive_state_checkpoint(
        source_root,
        run_dir,
        profile=profile,
        statuses=statuses,
        transitions=transitions,
        complete=True,
        unique_train_3_train_4_scenes=unique_train_3_train_4,
        train_2_condition_met=train_2_condition_met,
    )
    return result


def require_verified_archives(
    statuses: Mapping[str, ArchiveStatus] | Sequence[ArchiveStatus],
    required: Sequence[str],
    *,
    context: str,
) -> None:
    """Fail once, with every unmet archive, at a consuming stage boundary."""

    by_name = (
        dict(statuses)
        if isinstance(statuses, Mapping)
        else {status.filename: status for status in statuses}
    )
    unmet = [
        filename
        for filename in required
        if filename not in by_name
        or by_name[filename].state is not ArchiveState.VERIFIED
    ]
    if not unmet:
        return
    details = []
    for filename in unmet:
        status = by_name.get(filename)
        details.append(
            f"{filename}="
            + (
                "MISSING"
                if status is None
                else f"{status.state.value} ({status.error or 'no verified evidence'})"
            )
        )
    raise WorkflowBlocked(
        f"{context} requires verified official archives after all independent "
        f"attempts: {'; '.join(details)}"
    )


def download(profile: str, *, run_dir: Path | None = None) -> list[dict[str, Any]]:
    """Download, compactly extract, validate, and retire official archives."""

    contract = _profile_contract(profile)
    root = repository_root()
    data_root = root / "data_external" / "graspnet"
    final_manifest = data_root / "compact_dataset_manifest.json"
    existing = _read_complete_marker(final_manifest)
    if existing is not None:
        required_scenes = contract.minimum_unique_scenes
        if (
            existing.get("profile") != profile
            or existing.get("camera") != contract.camera
            or existing.get("frames_per_scene") != contract.frames_per_scene
            or int(existing.get("scene_count", -1)) != required_scenes
        ):
            raise WorkflowBlocked(
                "existing compact dataset profile/scene-count contract does not "
                f"match requested {profile}: observed profile={existing.get('profile')!r}, "
                f"scenes={existing.get('scene_count')!r}, required scenes={required_scenes}"
            )
        for record in existing.get("extractions", []):
            filename = str(record.get("filename", ""))
            if not filename or _validate_committed_extraction(root, filename) is None:
                raise WorkflowBlocked(
                    "existing compact dataset contains a missing/stale extraction commit"
                )
        rows = _download_manifest_rows(root)
        if run_dir:
            atomic_json(run_dir / "compact_dataset_manifest.json", existing)
            record_stage(
                run_dir,
                "download",
                "COMPLETE",
                archives=rows,
                compact_manifest=existing,
            )
            update_manifest(
                run_dir,
                dataset_archive_hashes={
                    str(row["filename"]): str(row["sha256"]) for row in rows
                },
            )
        return rows

    orchestration = orchestrate_archive_downloads(profile, root=root, run_dir=run_dir)
    require_verified_archives(
        orchestration.statuses,
        contract.required_archives,
        context="compact extraction and real-data smoke",
    )
    camera = contract.camera
    frames = deterministic_frame_ids(contract.frames_per_scene)
    split_path = _split_manifest_path(root, profile)
    split_payload = _load_locked_split(split_path, profile=profile)
    inventories = dict(orchestration.inventories)
    verifications = dict(orchestration.verifications)
    if split_payload is None:
        unique_scenes = {
            scene.scene_id
            for inventory in inventories.values()
            for scene in inventory.scenes
        }
        required_scenes = contract.minimum_unique_scenes
        if len(unique_scenes) < required_scenes:
            state_summary = ", ".join(
                f"{status.filename}={status.state.value}"
                for status in orchestration.statuses
                if status.filename in {"train_2.zip", "train_3.zip", "train_4.zip"}
            )
            raise WorkflowBlocked(
                f"official selected training archives provide {len(unique_scenes)} "
                f"unique scenes; {required_scenes} are required after independent "
                f"archive attempts ({state_summary})"
            )
        write_inventory_manifest(
            root / "downloads/graspnet/archive_inventory.json",
            inventories.values(),
        )
        split_payload = _split_payload(
            list(inventories.values()), profile=profile, seed=20260815
        )
        atomic_json(split_path, split_payload)
        atomic_json(root / "artifacts/graspnet6d/split_audit_v2.json", split_payload)
        if run_dir is not None:
            atomic_json(run_dir / "split_audit.json", split_payload)

    selected = _selected_scene_ids(split_payload)
    training_names = _training_archive_names(split_payload)
    object_ids = {
        int(object_id)
        for scene_name in (
            scene_name
            for partition in ("train", "validation", "test")
            for scene_name in split_payload[partition]
        )
        for object_id in split_payload["scene_object_ids"][scene_name]
    }
    scene_object_ids = {
        int(scene_name.split("_")[1]): tuple(
            int(value) for value in split_payload["scene_object_ids"][scene_name]
        )
        for partition in ("train", "validation", "test")
        for scene_name in split_payload[partition]
    }
    extraction_records: list[dict[str, Any]] = []

    # Extract only assets referenced by the locked 30-scene universe.  The
    # formal evaluator consumes these exact object IDs; unrelated objects are
    # not needed and retaining them would violate the mandatory 20 GB reserve.
    auxiliary = (
        ("grasp_label.zip", "grasp_label"),
        ("models.zip", "models"),
    )
    validators: dict[str, Callable[[Path], None]] = {
        "grasp_label": lambda data: _validate_grasp_assets(data, object_ids),
        "models": lambda data: _validate_model_assets(data, object_ids),
    }
    for filename, kind in auxiliary:
        committed = _validate_committed_extraction(root, filename)
        if committed is not None and _verified_archive_was_deleted(root, filename):
            extraction_records.append(
                _resumed_extraction_record(root, filename, committed[0], committed[1])
            )
            continue
        verification = verifications.get(filename)
        if verification is None:
            status = orchestration.status_by_filename[filename]
            raise WorkflowBlocked(
                f"required auxiliary archive {filename} is not available for "
                f"extraction: {status.state.value} ({status.error or 'no evidence'})"
            )
        inventory = inspect_archive(
            verification.path,
            filelist_path=Path(verification.path).with_suffix(".zip.filelist.txt"),
        )
        plan = build_subtree_extraction_plan(
            inventory,
            kind,
            object_ids=object_ids,
        )
        extraction_records.append(
            _extract_plan(
                root,
                filename,
                verification,
                plan,
                semantic_validator=validators[kind],
            )
        )

    dex_status = orchestration.status_by_filename["dex_models.zip"]
    optional: dict[str, Any] = {
        "dex_models.zip": {
            "status": dex_status.state.value,
            "error": dex_status.error,
        }
    }
    try:
        committed = _validate_committed_extraction(root, "dex_models.zip")
        if committed is not None and _verified_archive_was_deleted(
            root, "dex_models.zip"
        ):
            optional["dex_models.zip"] = _resumed_extraction_record(
                root, "dex_models.zip", committed[0], committed[1]
            )
        elif "dex_models.zip" in verifications:
            verification = verifications["dex_models.zip"]
            inventory = inspect_archive(verification.path)
            plan = build_subtree_extraction_plan(
                inventory, "dex_models", object_ids=object_ids
            )
            optional["dex_models.zip"] = _extract_plan(
                root,
                "dex_models.zip",
                verification,
                plan,
                semantic_validator=lambda data: _validate_dex_assets(
                    data, object_ids
                ),
            )
    except Exception as error:
        optional["dex_models.zip"] = {
            "status": "UNAVAILABLE_NON_BLOCKING",
            "error": f"{type(error).__name__}: {error}",
        }

    filename = "collision_label.zip"
    committed = _validate_committed_extraction(root, filename)
    if committed is not None and _verified_archive_was_deleted(root, filename):
        extraction_records.append(
            _resumed_extraction_record(root, filename, committed[0], committed[1])
        )
    else:
        verification = verifications.get(filename)
        if verification is None:
            status = orchestration.status_by_filename[filename]
            raise WorkflowBlocked(
                f"required auxiliary archive {filename} is not available for "
                f"extraction: {status.state.value} ({status.error or 'no evidence'})"
            )
        inventory = inspect_archive(
            verification.path,
            filelist_path=Path(verification.path).with_suffix(".zip.filelist.txt"),
        )
        plan = build_collision_extraction_plan(inventory, selected)
        extraction_records.append(
            _extract_plan(
                root,
                filename,
                verification,
                plan,
                semantic_validator=lambda data: _validate_collision_assets(
                    data, scene_object_ids
                ),
            )
        )

    for filename in training_names:
        committed = _validate_committed_extraction(root, filename)
        if committed is not None and _verified_archive_was_deleted(root, filename):
            extraction_records.append(
                _resumed_extraction_record(root, filename, committed[0], committed[1])
            )
            continue
        verification = verifications.get(filename)
        if verification is None:
            status = orchestration.status_by_filename.get(filename)
            raise WorkflowBlocked(
                f"locked split needs {filename}, but this orchestration has no "
                f"live verified archive ({status.state.value if status else 'MISSING'})"
            )
        inventory = inventories.get(filename) or _inspect_training_archive(
            root, verification
        )
        archive_scenes = {scene.scene_id for scene in inventory.scenes}
        selected_here = sorted(selected & archive_scenes)
        if not selected_here:
            raise WorkflowBlocked(
                f"locked split references no selected scenes from {filename}"
            )
        plan = build_training_extraction_plan(
            inventory, selected_here, camera=camera, frame_ids=frames
        )

        def validate_complete_dataset(data: Path) -> None:
            from .dataset import validate_graspnet_structure

            validate_graspnet_structure(
                data,
                camera=camera,
                scene_ids=sorted(selected),
                frame_ids=frames,
                strict=True,
            )

        extraction_records.append(
            _extract_plan(
                root,
                filename,
                verification,
                plan,
                semantic_validator=validate_complete_dataset,
                delete_archive=profile != "paper-lite-train3",
            )
        )

    _write_budget(
        root,
        phase="formal cache growth preflight",
        archive_bytes=0,
        estimated_cache_bytes=FORMAL_CACHE_ESTIMATE_BYTES,
    )
    compact_size = 0
    for path in data_root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            compact_size += path.stat().st_size
    compact = {
        "schema_version": "graspnet6d_compact_dataset_manifest_v1",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_dataset_page": "https://graspnet.net/datasets.html",
        "scope": "compact_scene_disjoint_subset_of_graspnet_training_scenes",
        "scope_label": (
            "held-out GraspNet training-scene subset"
            if profile == "paper-lite-train3"
            else "compact GraspNet training-scene subset"
        ),
        "profile": profile,
        "official_test_server_result": False,
        "official_test_benchmark": False,
        "source_archives": list(contract.required_scene_archives),
        "camera": camera,
        "frames_per_scene": len(frames),
        "frame_ids": list(frames),
        "scene_count": len(selected),
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "extractions": extraction_records,
        "training_archive_retirement_pending": profile == "paper-lite-train3",
        "optional_archives": optional,
        "compact_dataset_size_bytes": compact_size,
    }
    atomic_json(final_manifest, compact)
    rows = _download_manifest_rows(root)
    if run_dir:
        atomic_json(run_dir / "compact_dataset_manifest.json", compact)
        record_stage(
            run_dir,
            "download",
            "COMPLETE",
            archives=rows,
            archive_states=orchestration.to_record(),
            compact_manifest=compact,
        )
        update_manifest(
            run_dir,
            dataset_archive_hashes={
                str(row["filename"]): str(row["sha256"]) for row in rows
            },
        )
    return rows


def prepare(
    profile: str, *, run_dir: Path | None = None, resume: bool = False
) -> list[dict[str, Any]]:
    """Verify the compact commit; extraction is owned by :func:`download`."""

    del resume
    contract = _profile_contract(profile)
    root = repository_root()
    manifest = root / "data_external/graspnet/compact_dataset_manifest.json"
    payload = _read_complete_marker(manifest)
    if payload is None:
        raise WorkflowBlocked(
            "verified compact dataset manifest is missing; run download"
        )
    if (
        payload.get("profile") != profile
        or payload.get("camera") != contract.camera
        or int(payload.get("scene_count", -1)) != contract.minimum_unique_scenes
    ):
        raise WorkflowBlocked(
            "verified compact dataset belongs to another profile/camera/scene contract"
        )
    split = Path(str(payload.get("split_manifest_path", ""))).expanduser().resolve()
    if not split.is_file() or payload.get("split_manifest_sha256") != sha256_file(
        split
    ):
        raise WorkflowBlocked("compact dataset split manifest is missing or stale")
    records = list(payload.get("extractions", []))
    if not records:
        raise WorkflowBlocked("compact dataset manifest contains no extraction records")
    if run_dir:
        atomic_json(run_dir / "compact_dataset_manifest.json", payload)
        record_stage(run_dir, "prepare", "COMPLETE", compact_manifest=payload)
    return records


def ensure_compact_training_frames(
    profile: str,
    frames_per_scene: int,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Idempotently expand the locked training archive before retirement."""

    contract = _profile_contract(profile)
    if profile != "paper-lite-train3":
        raise WorkflowBlocked("adaptive frame expansion is train3-profile only")
    allowed = {16, 24, 32}
    if frames_per_scene not in allowed:
        raise WorkflowBlocked(f"adaptive frame count must be one of {sorted(allowed)}")
    root = repository_root()
    split_path = _split_manifest_path(root, profile)
    split = _load_locked_split(split_path, profile=profile)
    if split is None:
        raise WorkflowBlocked("locked train3 split is missing")
    selected = _selected_scene_ids(split)
    filename = contract.required_scene_archives[0]
    archive = root / "downloads/graspnet" / filename
    if not archive.is_file() or archive.is_symlink():
        raise WorkflowBlocked(
            f"adaptive extraction requires retained verified archive: {archive}"
        )
    verification = _download_one(root, filename)
    inventory = _inspect_training_archive(root, verification)
    frame_ids = deterministic_frame_ids(frames_per_scene)
    plan = build_training_extraction_plan(
        inventory,
        sorted(selected),
        camera=contract.camera,
        frame_ids=frame_ids,
    )

    def validate_complete_dataset(data: Path) -> None:
        from .dataset import validate_graspnet_structure

        validate_graspnet_structure(
            data,
            camera=contract.camera,
            scene_ids=sorted(selected),
            frame_ids=frame_ids,
            strict=True,
        )

    record = _extract_plan(
        root,
        filename,
        verification,
        plan,
        semantic_validator=validate_complete_dataset,
        delete_archive=False,
    )
    manifest_path = root / "data_external/graspnet/compact_dataset_manifest.json"
    manifest = _read_complete_marker(manifest_path)
    if manifest is None or manifest.get("profile") != profile:
        raise WorkflowBlocked("compact dataset manifest is missing or belongs elsewhere")
    manifest["frames_per_scene"] = len(frame_ids)
    manifest["frame_ids"] = list(frame_ids)
    manifest["training_archive_retirement_pending"] = True
    records = [
        item
        for item in manifest.get("extractions", [])
        if item.get("filename") != filename
    ]
    records.append(record)
    manifest["extractions"] = records
    atomic_json(manifest_path, manifest)
    if run_dir is not None:
        atomic_json(run_dir / "compact_dataset_manifest.json", manifest)
    return record


def retire_compact_training_archives(
    profile: str, *, run_dir: Path | None = None
) -> tuple[dict[str, Any], ...]:
    """Delete retained train ZIPs only after the adaptive group gate passes."""

    contract = _profile_contract(profile)
    if profile != "paper-lite-train3":
        return ()
    root = repository_root()
    retired: list[dict[str, Any]] = []
    for filename in contract.required_scene_archives:
        committed = _validate_committed_extraction(root, filename)
        if committed is None:
            raise WorkflowBlocked(f"training extraction is not committed: {filename}")
        marker, plan = committed
        marker_path = root / "data_external/graspnet/.extracted" / f"{filename}.json"
        verification = _download_one(root, filename)
        extraction = ExtractionResult(
            archive_path=verification.path,
            archive_sha256=plan.archive_sha256,
            destination_root=str(root / "data_external/graspnet"),
            selected_file_count=int(marker["selected_file_count"]),
            selected_bytes=int(marker["selected_bytes"]),
            extracted_file_count=int(marker["extracted_file_count"]),
            resumed_file_count=int(marker["resumed_file_count"]),
            verified_file_count=int(marker["verified_file_count"]),
            complete=bool(marker["complete"]),
            manifest_path=str(marker_path),
            manifest_sha256=sha256_file(marker_path),
        )
        retired.append(
            _delete_verified_archive(
                root,
                verification,
                extraction,
                downstream_loader_verified=True,
            )
        )
    manifest_path = root / "data_external/graspnet/compact_dataset_manifest.json"
    manifest = _read_complete_marker(manifest_path)
    if manifest is None:
        raise WorkflowBlocked("compact dataset manifest disappeared before retirement")
    manifest["training_archive_retirement_pending"] = False
    manifest["training_archive_deletions"] = retired
    atomic_json(manifest_path, manifest)
    if run_dir is not None:
        atomic_json(run_dir / "compact_dataset_manifest.json", manifest)
    return tuple(retired)


def record_failure(
    run_dir: Path, stage: str, error: BaseException, group_id: str | None = None
) -> None:
    directory = run_dir / "errors" / stage
    directory.mkdir(parents=True, exist_ok=True)
    identity = group_id or "stage"
    payload = {
        "stage": stage,
        "group_id": group_id,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(directory / f"{identity}.json", payload)
    record_stage(
        run_dir,
        stage,
        "BLOCKED" if isinstance(error, WorkflowBlocked) else "FAILED",
        error=payload,
    )


def write_blocked_status(run_dir: Path, stage: str, reason: str) -> None:
    recorded = datetime.now(timezone.utc).isoformat()
    manifest_path = run_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    sample_counts = dict(manifest.get("sample_counts", {}))
    targets_path = run_dir / "manifests" / "target_groups.jsonl"
    target_rows: list[dict[str, Any]] = []
    if targets_path.is_file():
        for line in targets_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    target_rows.append(value)
    split_counts = {
        split: sum(str(row.get("split")) == split for row in target_rows)
        for split in ("train", "validation", "test")
    }
    completed_stages: list[str] = []
    for path in sorted((run_dir / "stages").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("status") == "COMPLETE":
            completed_stages.append(str(payload.get("stage", path.stem)))
    awaiting_review = "independent hashed review" in reason
    status_label = "AWAITING_GEOMETRY_REVIEW" if awaiting_review else "BLOCKED"
    text = f"""# FINAL STATUS: BLOCKED

Run ID: `{run_dir.name}`  
Blocked stage: `{stage}`  
Machine status: `{status_label}`  
Recorded: {recorded}

Reason: {reason}

Completed stages recorded before this stop: {", ".join(completed_stages) if completed_stages else "none"}.

No top-level formal `RESULTS.md` or evidence-based `CONCLUSIONS.md` was emitted.
Partial caches, validation figures, or analysis files (if listed above) are not
a completed paper result and must not be cited as one.

## Completion gates

The immutable manifest and per-stage JSON files are the source of truth for
what actually ran. Any stage absent from the completed-stage list remains
unexecuted or incomplete for this run.

## Resume

Follow `MANUAL_DOWNLOAD_REQUIRED.md`, then run:

```bash
PYTHONPATH=src .venv-graspnet6d/bin/python -m graspnet6d.cli all --profile paper-lite --resume --run-id {run_dir.name}
```
"""
    atomic_text(run_dir / "FINAL_STATUS.md", text)
    atomic_text(
        run_dir / "REPRODUCE.md",
        "# Reproduce / resume\n\nSee the repository-root `MANUAL_DOWNLOAD_REQUIRED.md`.\n\n"
        f"`PYTHONPATH=src .venv-graspnet6d/bin/python -m graspnet6d.cli all --profile paper-lite --resume --run-id {run_dir.name}`\n",
    )
    atomic_text(
        run_dir / "METHODS.md",
        "# Methods status\n\n"
        f"This run stopped at `{stage}` with machine status `{status_label}`. "
        f"Completed stages: {', '.join(completed_stages) if completed_stages else 'none'}. "
        "The locked protocol uses a "
        "scene-disjoint 60/20/20 held-out GraspNet training-scene split; derived, "
        "programmatically unique language; oracle and HiFi-CS masks; a fixed 0.30 m, "
        "40^3, 7.5 mm target-centred single-view TSDF retaining full local scene depth; "
        "the frozen pretrained VGN checkpoint recorded in the run manifest; deterministic "
        "score/tie-break ordering and frozen pose NMS; per-candidate official GraspNet "
        "association/collision/friction math behind geometry and parity gates; runtime-only "
        "6-DoF features; graded CPU LightGBM LambdaMART with train-only imputation and "
        "validation-only selection; three locked training seeds; scene-cluster bootstrap; "
        "and exact paired McNemar testing. Only stages explicitly listed as complete above "
        "are reported as executed. See the resolved config and coordinate/feature contracts "
        "for the complete pre-registered values.\n",
    )
    limitations = [
        "# Limitations of this blocked run",
        "",
    ]
    if not target_rows:
        limitations.extend(
            [
                "- The required official GraspNet archives/scenes have not produced a "
                "validated target manifest, so there are no formal groups, candidates, "
                "labels, or metrics. Future work: obtain/extract the archives under the "
                "official terms and resume this immutable run.",
                "- The storage decision in the audit must preserve the required 20% "
                "filesystem reserve before download/extraction.",
            ]
        )
    if not (
        run_dir / "geometry_validation" / "evaluator_geometry_contract.json"
    ).is_file():
        limitations.append(
            "- VGN-to-GraspNet gripper axes and height/depth semantics are not yet "
            "accepted by the real-data geometry gate. Future work: complete the hashed "
            "independent review and resume."
        )
    if not (run_dir / "evaluator_parity" / "evaluator_parity_gate.json").is_file():
        limitations.append(
            "- Full per-candidate association/collision/friction parity is not yet "
            "locked for this run. Future work: complete the official-source parity gate."
        )
    limitations.append(
        "- The CPU/MPS timing audit initially uses analytic TSDF tensors; the formal "
        "device decision remains CPU until real cached-TSDF parity is recorded."
    )
    atomic_text(
        run_dir / "LIMITATIONS.md",
        "\n".join(limitations) + "\n",
    )
    atomic_json(
        run_dir / "dataset_summary.json",
        {
            "status": "BLOCKED",
            "profile": manifest.get("profile"),
            "downloaded_archives": sorted(
                dict(manifest.get("dataset_archive_hashes", {}))
            ),
            "scenes": int(sample_counts.get("scenes", 0)),
            "frames": int(sample_counts.get("frames", 0)),
            "target_groups": len(target_rows),
            "train_groups": split_counts["train"],
            "validation_groups": split_counts["validation"],
            "test_groups": split_counts["test"],
            "excluded_groups": int(sample_counts.get("excluded_groups", 0)),
            "completed_stages": completed_stages,
            "blocked_stage": stage,
            "reason": reason,
        },
    )
    update_manifest(
        run_dir,
        status="BLOCKED",
        blocked_stage=stage,
        blocked_reason=reason,
        ended_at_utc=recorded,
        formal_results_emitted=False,
    )


__all__ = [
    "Archive",
    "ArchiveOrchestration",
    "ArchiveState",
    "ArchiveStatus",
    "PAPER_LITE_ARCHIVES",
    "WorkflowBlocked",
    "download",
    "orchestrate_archive_downloads",
    "prepare",
    "record_failure",
    "require_verified_archives",
    "write_blocked_status",
]
