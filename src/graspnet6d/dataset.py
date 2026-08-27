"""GraspNet-1Billion archive planning and strict on-disk validation.

Archive byte counts are the values currently published by the official
GraspNet dataset page.  They are planning evidence, not checksums: downloaded
archives still require ``unzip -t`` and a locally recorded SHA-256 digest.
"""

from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


OFFICIAL_DATASET_PAGE = "https://graspnet.net/datasets.html"
SCENE_PATTERN = re.compile(r"^scene_(\d{4})$")
BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class ArchiveSpec:
    filename: str
    published_size_bytes: int
    google_drive_id: str
    jbox_url: str
    required_for_paper_lite: bool = True


OFFICIAL_ARCHIVES: Mapping[str, ArchiveSpec] = {
    "train_2": ArchiveSpec(
        "train_2.zip",
        20_000_000_000,
        "1b1Z1goPV0o_wdwXZ8qTlHd2TBRU5-CmH",
        "https://jbox.sjtu.edu.cn/l/G57uyS",
        required_for_paper_lite=False,
    ),
    "train_3": ArchiveSpec(
        "train_3.zip",
        20_000_000_000,
        "1oNcmZno2ymsDUWTmfFOxewMBjTXhL95c",
        "https://jbox.sjtu.edu.cn/l/wJorXZ",
    ),
    "train_4": ArchiveSpec(
        "train_4.zip",
        6_803_985_160,
        "1e8Xy7-lFhiXk0ugPOKvHKDiGTparmx00",
        "https://jbox.sjtu.edu.cn/l/SHwJVL",
    ),
    "grasp_label": ArchiveSpec(
        "grasp_label.zip",
        2_059_130_127,
        "1FCV6j2J2eQpVk_ddJXljJvjRT1KU3sJ6",
        "https://jbox.sjtu.edu.cn/l/noXqUa",
    ),
    "collision_label": ArchiveSpec(
        "collision_label.zip",
        441_783_131,
        "1p43sntiN9HJZRDFDNpzaEaEYoPY6IWsu",
        "https://jbox.sjtu.edu.cn/l/DuUptQ",
    ),
    "models": ArchiveSpec(
        "models.zip",
        4_599_338_858,
        "1Gxwu2C5wRQ0QwjdA8CbMXx-bYf_wwPT5",
        "https://jbox.sjtu.edu.cn/l/jFF3no",
    ),
    "dex_models": ArchiveSpec(
        "dex_models.zip",
        9_518_063_724,
        "1RElNqUHNoA9l_muTGNu7yAc3ql_e7pL3",
        "",
        required_for_paper_lite=False,
    ),
}


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    archive_keys: tuple[str, ...]
    scene_limit: int | None
    frames_per_scene: int
    targets_per_frame: int
    group_limit: int | None
    frozen_top_k: int
    formal: bool


PROFILE_SPECS: Mapping[str, DatasetProfile] = {
    "smoke": DatasetProfile(
        "smoke",
        ("train_4", "grasp_label", "collision_label", "models"),
        3,
        16,
        1,
        24,
        20,
        False,
    ),
    "paper-lite": DatasetProfile(
        "paper-lite",
        ("train_4", "train_3", "grasp_label", "collision_label", "models"),
        35,
        16,
        3,
        3_000,
        50,
        True,
    ),
    "paper-lite-train3": DatasetProfile(
        "paper-lite-train3",
        ("train_3", "grasp_label", "collision_label", "models"),
        30,
        16,
        3,
        3_000,
        50,
        True,
    ),
    "paper-extended": DatasetProfile(
        "paper-extended",
        ("train_4", "train_3", "grasp_label", "collision_label", "models"),
        None,
        16,
        3,
        None,
        50,
        True,
    ),
}


@dataclass(frozen=True)
class StorageEstimate:
    profile: str
    compressed_bytes: int
    conservative_extracted_bytes: int
    extraction_temp_bytes: int
    cache_and_artifact_bytes: int
    total_required_bytes: int
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class DiskBudget:
    filesystem_total_bytes: int
    filesystem_free_bytes: int
    safety_reserve_bytes: int
    free_after_reserve_bytes: int
    required_bytes: int
    shortfall_bytes: int
    allowed: bool

    def to_record(self) -> dict[str, int | bool]:
        return asdict(self)


def get_profile(name: str) -> DatasetProfile:
    key = str(name).strip().lower().replace("_", "-")
    if key not in PROFILE_SPECS:
        raise ValueError(f"unknown dataset profile {name!r}; expected {sorted(PROFILE_SPECS)}")
    return PROFILE_SPECS[key]


def estimate_storage(
    profile: str | DatasetProfile,
    *,
    extracted_multiplier: float = 2.5,
    cache_and_artifact_bytes: int = 16 * BYTES_PER_GIB,
) -> StorageEstimate:
    """Build a conservative, explicit pre-download storage estimate.

    The official page publishes compressed sizes but not a stable extracted
    byte count.  Consequently the extraction multiplier and cache allowance
    remain visible assumptions rather than being presented as measured facts.
    """

    selected = get_profile(profile) if isinstance(profile, str) else profile
    if extracted_multiplier < 1:
        raise ValueError("extracted_multiplier must be at least 1")
    if cache_and_artifact_bytes < 0:
        raise ValueError("cache_and_artifact_bytes must be non-negative")
    specs = [OFFICIAL_ARCHIVES[key] for key in selected.archive_keys]
    compressed = sum(spec.published_size_bytes for spec in specs)
    extracted = int(np.ceil(compressed * float(extracted_multiplier)))
    extraction_temp = max(spec.published_size_bytes for spec in specs)
    required = compressed + extracted + extraction_temp + int(cache_and_artifact_bytes)
    return StorageEstimate(
        profile=selected.name,
        compressed_bytes=compressed,
        conservative_extracted_bytes=extracted,
        extraction_temp_bytes=extraction_temp,
        cache_and_artifact_bytes=int(cache_and_artifact_bytes),
        total_required_bytes=required,
        assumptions=(
            f"extracted size estimated as {extracted_multiplier:.2f}x compressed size",
            "largest archive retained as resumable extraction headroom",
            "Dex model archive excluded; needed objects may be built from official models",
        ),
    )


def check_disk_budget(
    filesystem_path: Path | str,
    estimate: StorageEstimate,
    *,
    safety_fraction: float = 0.20,
) -> DiskBudget:
    """Check storage while preserving a fraction of the entire filesystem."""

    if not 0 <= safety_fraction < 1:
        raise ValueError("safety_fraction must be in [0, 1)")
    target = Path(filesystem_path).expanduser().resolve()
    # Download roots commonly do not exist at audit time.  Query the nearest
    # existing ancestor without creating or mutating anything.
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    reserve = int(np.ceil(usage.total * safety_fraction))
    usable = max(0, usage.free - reserve)
    shortfall = max(0, estimate.total_required_bytes - usable)
    return DiskBudget(
        filesystem_total_bytes=usage.total,
        filesystem_free_bytes=usage.free,
        safety_reserve_bytes=reserve,
        free_after_reserve_bytes=usable,
        required_bytes=estimate.total_required_bytes,
        shortfall_bytes=shortfall,
        allowed=shortfall == 0,
    )


@dataclass
class DatasetValidationReport:
    root: str
    camera: str
    scene_ids: list[int]
    frame_ids: list[int]
    missing: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    object_ids: list[int] = field(default_factory=list)
    checked_file_count: int = 0

    @property
    def valid(self) -> bool:
        return not self.missing and not self.invalid and bool(self.scene_ids)

    def to_record(self) -> dict[str, object]:
        value = asdict(self)
        value["valid"] = self.valid
        return value


class DatasetValidationError(RuntimeError):
    """Raised with the complete list of known structural failures."""

    def __init__(self, report: DatasetValidationReport):
        self.report = report
        details = [*(f"missing: {path}" for path in report.missing), *(f"invalid: {item}" for item in report.invalid)]
        preview = "\n".join(details[:20])
        suffix = "" if len(details) <= 20 else f"\n... and {len(details) - 20} more"
        super().__init__(f"GraspNet dataset validation failed:\n{preview}{suffix}")


def discover_scene_ids(root: Path | str) -> list[int]:
    scenes_root = Path(root).expanduser().resolve() / "scenes"
    if not scenes_root.is_dir():
        return []
    result: list[int] = []
    for child in scenes_root.iterdir():
        match = SCENE_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            result.append(int(match.group(1)))
    return sorted(result)


def _expect_file(path: Path, report: DatasetValidationReport) -> bool:
    report.checked_file_count += 1
    if not path.is_file():
        report.missing.append(str(path))
        return False
    return True


def _load_array(path: Path, report: DatasetValidationReport, description: str) -> np.ndarray | None:
    if not _expect_file(path, report):
        return None
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as error:
        report.invalid.append(f"{description} {path}: {type(error).__name__}: {error}")
        return None
    if not np.all(np.isfinite(value)):
        report.invalid.append(f"{description} {path}: contains NaN or Inf")
    return value


def _validate_homogeneous(value: np.ndarray, path: Path, report: DatasetValidationReport) -> None:
    if value.shape[-2:] != (4, 4):
        report.invalid.append(f"transform {path}: expected (...,4,4), got {value.shape}")
        return
    rows = value.reshape(-1, 4, 4)[:, 3, :]
    if not np.allclose(rows, [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        report.invalid.append(f"transform {path}: invalid homogeneous last row")


def _read_object_ids(path: Path, report: DatasetValidationReport) -> list[int]:
    if not _expect_file(path, report):
        return []
    try:
        values = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        report.invalid.append(f"object ID list {path}: {type(error).__name__}: {error}")
        return []
    if not values:
        report.invalid.append(f"object ID list {path}: empty")
    if any(value < 0 or value >= 88 for value in values):
        report.invalid.append(f"object ID list {path}: IDs must be in [0, 87]")
    return values


def _read_image(path: Path, report: DatasetValidationReport, description: str) -> np.ndarray | None:
    if not _expect_file(path, report):
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            value = np.asarray(image)
    except Exception as error:
        report.invalid.append(f"{description} {path}: {type(error).__name__}: {error}")
        return None
    if value.size == 0:
        report.invalid.append(f"{description} {path}: empty image")
        return None
    return value


def _validate_frame_payloads(
    camera_root: Path,
    stem: str,
    report: DatasetValidationReport,
) -> None:
    rgb_path = camera_root / "rgb" / f"{stem}.png"
    depth_path = camera_root / "depth" / f"{stem}.png"
    label_path = camera_root / "label" / f"{stem}.png"
    meta_path = camera_root / "meta" / f"{stem}.mat"
    xml_path = camera_root / "annotations" / f"{stem}.xml"
    rgb = _read_image(rgb_path, report, "RGB image")
    depth = _read_image(depth_path, report, "depth image")
    label = _read_image(label_path, report, "instance-label image")
    if rgb is not None and (rgb.ndim != 3 or rgb.shape[2] not in {3, 4}):
        report.invalid.append(f"RGB image {rgb_path}: expected HxWx3/4, got {rgb.shape}")
    if depth is not None and (depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer)):
        report.invalid.append(f"depth image {depth_path}: expected integer HxW, got {depth.shape}/{depth.dtype}")
    if label is not None and (label.ndim != 2 or not np.issubdtype(label.dtype, np.integer)):
        report.invalid.append(f"instance-label image {label_path}: expected integer HxW, got {label.shape}/{label.dtype}")
    shapes = [value.shape[:2] for value in (rgb, depth, label) if value is not None]
    if shapes and any(shape != shapes[0] for shape in shapes[1:]):
        report.invalid.append(
            f"frame {camera_root / stem}: RGB/depth/label image shapes disagree: {shapes}"
        )

    if _expect_file(meta_path, report):
        try:
            from scipy.io import loadmat

            meta = loadmat(meta_path)
            required = {"cls_indexes", "poses", "intrinsic_matrix", "factor_depth"}
            missing = sorted(required - set(meta))
            if missing:
                raise ValueError(f"missing keys {missing}")
            classes = np.asarray(meta["cls_indexes"]).reshape(-1)
            poses = np.asarray(meta["poses"], dtype=np.float64)
            intrinsic = np.asarray(meta["intrinsic_matrix"], dtype=np.float64)
            factor = np.asarray(meta["factor_depth"], dtype=np.float64).reshape(-1)
            if not len(classes) or not np.isfinite(classes).all():
                raise ValueError("cls_indexes must be non-empty and finite")
            if poses.shape != (3, 4, len(classes)) or not np.isfinite(poses).all():
                raise ValueError(f"poses must be finite (3,4,N), got {poses.shape}")
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                raise ValueError("intrinsic_matrix must be finite 3x3")
            if factor.size != 1 or not np.isfinite(factor[0]) or factor[0] <= 0:
                raise ValueError("factor_depth must be one finite positive scalar")
        except Exception as error:
            report.invalid.append(f"frame metadata {meta_path}: {type(error).__name__}: {error}")

    if _expect_file(xml_path, report):
        try:
            root = ET.parse(xml_path).getroot()
            if len(root) == 0:
                raise ValueError("scene contains no object annotations")
            for index, item in enumerate(root):
                if len(item) < 5:
                    raise ValueError(f"object {index} has fewer than five fields")
                object_id = int(item[0].text or "")
                translation = np.asarray([float(value) for value in (item[3].text or "").split()])
                quaternion = np.asarray([float(value) for value in (item[4].text or "").split()])
                if object_id < 0 or object_id >= 88:
                    raise ValueError(f"object ID {object_id} is outside [0,87]")
                if translation.shape != (3,) or not np.isfinite(translation).all():
                    raise ValueError(f"object {index} translation is not finite xyz")
                if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) <= 0:
                    raise ValueError(f"object {index} quaternion is invalid")
        except Exception as error:
            report.invalid.append(f"scene annotation {xml_path}: {type(error).__name__}: {error}")


def _validate_npz(
    path: Path,
    report: DatasetValidationReport,
    *,
    kind: str,
    expected_arrays: int | None = None,
) -> None:
    if not _expect_file(path, report):
        return
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = tuple(archive.files)
            if kind == "grasp labels":
                required = {"points", "offsets", "scores"}
                if not required.issubset(names):
                    raise ValueError(f"missing arrays {sorted(required - set(names))}")
                points = np.asarray(archive["points"])
                offsets = np.asarray(archive["offsets"])
                scores = np.asarray(archive["scores"])
                if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                    raise ValueError(f"points must be non-empty Nx3, got {points.shape}")
                if offsets.ndim < 2 or scores.ndim < 2 or offsets.shape[0] != len(points) or scores.shape[0] != len(points):
                    raise ValueError("offsets/scores must share the sampled-point axis")
                if not all(np.isfinite(value).all() for value in (points, offsets, scores)):
                    raise ValueError("grasp arrays contain NaN or Inf")
            else:
                expected_names = tuple(f"arr_{index}" for index in range(expected_arrays or 0))
                if set(names) != set(expected_names) or len(names) != len(expected_names):
                    raise ValueError(f"expected array set {expected_names}, got {names}")
                for name in expected_names:
                    value = np.asarray(archive[name])
                    if value.size == 0 or not np.isfinite(value).all():
                        raise ValueError(f"{name} is empty or non-finite")
    except Exception as error:
        report.invalid.append(f"{kind} {path}: {type(error).__name__}: {error}")


def _validate_model_sources(model_root: Path, report: DatasetValidationReport) -> None:
    ply = model_root / "nontextured.ply"
    obj = model_root / "textured.obj"
    sdf = model_root / "textured.sdf"
    for path, description in ((ply, "PLY"), (obj, "OBJ"), (sdf, "SDF")):
        if _expect_file(path, report):
            try:
                if path.stat().st_size <= 0:
                    raise ValueError("empty file")
                head = path.read_bytes()[:256].lower()
                if description == "PLY" and not head.startswith(b"ply"):
                    raise ValueError("missing PLY header")
                if description == "OBJ" and b"v " not in head and b"#" not in head:
                    raise ValueError("missing OBJ vertex/header content")
            except Exception as error:
                report.invalid.append(f"object {description} {path}: {type(error).__name__}: {error}")


def validate_graspnet_structure(
    root: Path | str,
    *,
    camera: str = "kinect",
    scene_ids: Sequence[int] | None = None,
    frame_ids: Sequence[int] = tuple(range(256)),
    strict: bool = True,
    require_rect_labels: bool = False,
) -> DatasetValidationReport:
    """Validate every requested scene/frame and report every missing sample.

    The paths mirror ``graspnetAPI.GraspNet.checkDataCompleteness``.  This
    implementation collects failures instead of printing and continuing, so a
    missing frame can never be silently excluded from an experiment.
    """

    dataset_root = Path(root).expanduser().resolve()
    if camera not in {"kinect", "realsense"}:
        raise ValueError("camera must be 'kinect' or 'realsense'")
    selected_scenes = discover_scene_ids(dataset_root) if scene_ids is None else sorted({int(value) for value in scene_ids})
    selected_frames = sorted({int(value) for value in frame_ids})
    if any(value < 0 or value >= 256 for value in selected_frames):
        raise ValueError("frame IDs must be in [0, 255]")
    report = DatasetValidationReport(
        root=str(dataset_root), camera=camera, scene_ids=selected_scenes, frame_ids=selected_frames
    )
    if not selected_scenes:
        report.missing.append(str(dataset_root / "scenes" / "scene_####"))

    observed_objects: set[int] = set()
    for scene_id in selected_scenes:
        scene = dataset_root / "scenes" / f"scene_{scene_id:04d}"
        if not scene.is_dir():
            report.missing.append(str(scene))
            continue
        objects = _read_object_ids(scene / "object_id_list.txt", report)
        observed_objects.update(objects)
        cross_camera_path = scene / "rs_wrt_kn.npy"
        cross_camera = _load_array(cross_camera_path, report, "RealSense/Kinect transform")
        if cross_camera is not None:
            _validate_homogeneous(cross_camera, cross_camera_path, report)
        camera_root = scene / camera
        intrinsic = _load_array(camera_root / "camK.npy", report, "camera intrinsics")
        if intrinsic is not None:
            if intrinsic.shape != (3, 3):
                report.invalid.append(f"camera intrinsics {camera_root / 'camK.npy'}: expected (3,3), got {intrinsic.shape}")
            elif intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
                report.invalid.append(f"camera intrinsics {camera_root / 'camK.npy'}: focal lengths must be positive")
        poses_path = camera_root / "camera_poses.npy"
        poses = _load_array(poses_path, report, "camera poses")
        if poses is not None:
            _validate_homogeneous(poses, poses_path, report)
            if poses.ndim != 3 or poses.shape[0] <= max(selected_frames, default=-1):
                report.invalid.append(
                    f"camera poses {poses_path}: expected enough (N,4,4) poses for requested frames, got {poses.shape}"
                )
        table_path = camera_root / "cam0_wrt_table.npy"
        table = _load_array(table_path, report, "camera/table transform")
        if table is not None:
            _validate_homogeneous(table, table_path, report)

        for frame_id in selected_frames:
            stem = f"{frame_id:04d}"
            _validate_frame_payloads(camera_root, stem, report)
            if require_rect_labels:
                rect_path = camera_root / "rect" / f"{stem}.npy"
                rect = _load_array(rect_path, report, "rectangle grasp labels")
                if rect is not None and (rect.ndim < 2 or rect.size == 0):
                    report.invalid.append(f"rectangle grasp labels {rect_path}: empty or invalid shape {rect.shape}")
        _validate_npz(
            dataset_root / "collision_label" / f"scene_{scene_id:04d}" / "collision_labels.npz",
            report,
            kind="collision labels",
            expected_arrays=len(objects),
        )

    for object_id in sorted(observed_objects):
        model_root = dataset_root / "models" / f"{object_id:03d}"
        _validate_model_sources(model_root, report)
        _validate_npz(
            dataset_root / "grasp_label" / f"{object_id:03d}_labels.npz",
            report,
            kind="grasp labels",
        )
    report.object_ids = sorted(observed_objects)
    report.missing.sort()
    report.invalid.sort()
    if strict and not report.valid:
        raise DatasetValidationError(report)
    return report
