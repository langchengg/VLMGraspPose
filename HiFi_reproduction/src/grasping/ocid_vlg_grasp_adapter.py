"""Load verified OCID-VLG/HiFi-CS bundles for planar grasp sampling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from .camera_geometry import CameraIntrinsicsData, depth_mm_to_meters
from .mask_processing import MaskProcessingResult, process_mask_with_diagnostics


REQUIRED_BUNDLE_FILES = (
    "color.png",
    "depth.png",
    "target_mask.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
    "metadata.json",
    "checksums.sha256",
)


@dataclass(frozen=True)
class OcidVlgGraspSample:
    sample_id: str
    sample_index: int
    question_index: int
    scene_id: str
    query: str
    bundle_dir: Path
    rgb: np.ndarray
    depth_mm: np.ndarray
    depth_m: np.ndarray
    mask_input: np.ndarray
    mask_processing: MaskProcessingResult
    intrinsics: CameraIntrinsicsData
    intrinsics_metadata: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def target_mask_original(self) -> np.ndarray:
        return self.mask_processing.original_binary

    @property
    def target_mask_processed(self) -> np.ndarray:
        return self.mask_processing.processed

    @property
    def valid_depth_mask(self) -> np.ndarray:
        return self.mask_processing.valid_depth


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle_checksums(bundle_dir: Path) -> None:
    lines = (bundle_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed checksum line {line_number}: {line!r}")
        expected, name = parts
        name = name.lstrip("* ")
        if Path(name).name != name:
            raise ValueError(f"Unsafe checksum member: {name!r}")
        path = bundle_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe bundle member: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Bundle checksum mismatch: {path.name}")
        seen.add(name)
    expected_members = set(REQUIRED_BUNDLE_FILES) - {"checksums.sha256"}
    if seen != expected_members:
        raise ValueError(
            f"Checksum manifest members differ: missing={sorted(expected_members-seen)} "
            f"extra={sorted(seen-expected_members)}"
        )


class OcidVlgBundleIndex:
    """Manifest-backed index of the completed prediction-only input bundles."""

    def __init__(self, dataset_root: Path, mask_root: Path, *, split: str = "test"):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.mask_root = Path(mask_root).expanduser().resolve()
        self.split = split
        if split != "test":
            raise ValueError(
                "The completed HiFi prediction manifest contains only the frozen test split"
            )
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"OCID-VLG dataset root missing: {self.dataset_root}")
        manifest_path = self.mask_root / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"HiFi predicted-mask manifest missing: {manifest_path}")
        self.manifest_path = manifest_path
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rows = rows
        self.by_id = {str(row["sample_id"]): row for row in rows}
        if len(rows) != len(self.by_id):
            raise ValueError("Duplicate sample_id values in predicted-mask manifest")
        if any(
            row.get("ready") is not True
            or row.get("ready_for_anygrasp") is not True
            or row.get("blockers") not in (None, [])
            for row in rows
        ):
            raise ValueError("Predicted-mask manifest contains non-ready bundles")

    def sample_ids(self, sample_id: str | None = None, limit: int | None = None) -> list[str]:
        if sample_id is not None:
            if sample_id not in self.by_id:
                raise KeyError(f"Unknown sample_id: {sample_id}")
            return [sample_id]
        ids = [str(row["sample_id"]) for row in self.rows]
        if limit is not None:
            if limit <= 0:
                raise ValueError("sample limit must be positive")
            ids = ids[: int(limit)]
        return ids

    def iter_samples(
        self,
        sample_ids: Iterable[str],
        **load_kwargs: Any,
    ) -> Iterable[OcidVlgGraspSample]:
        for sample_id in sample_ids:
            yield self.load_sample(sample_id, **load_kwargs)

    def load_sample(
        self,
        sample_id: str,
        *,
        camera_frame: str,
        mask_source: str = "binary_prediction",
        mask_threshold: float = 0.15,
        min_component_area_px: int = 0,
        retain_largest_component: bool = False,
        mask_erode_px: int = 0,
        mask_dilate_px: int = 0,
        verify_checksums: bool = True,
    ) -> OcidVlgGraspSample:
        if sample_id not in self.by_id:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        row = self.by_id[sample_id]
        bundle = self.mask_root / sample_id
        if not bundle.is_dir() or bundle.is_symlink():
            raise FileNotFoundError(f"Bundle missing or unsafe: {bundle}")
        missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Bundle files missing for {sample_id}: {missing}")
        if verify_checksums:
            verify_bundle_checksums(bundle)

        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        intrinsics_json = json.loads(
            (bundle / "intrinsics.json").read_text(encoding="utf-8")
        )
        query = (bundle / "language.txt").read_text(encoding="utf-8").strip()
        for field in ("sample_id", "question_index", "scene_id", "query"):
            expected = row.get(field)
            actual = metadata.get(field)
            if str(actual) != str(expected):
                raise ValueError(
                    f"Bundle/manifest {field} mismatch for {sample_id}: {actual!r} != {expected!r}"
                )
        if query != str(row["query"]):
            raise ValueError(f"language.txt mismatch for {sample_id}")
        if metadata.get("mask_source") != "predicted_mask_original_resolution":
            raise ValueError(f"Bundle is not prediction-only: {sample_id}")
        if metadata.get("oracle_artifacts_exported") is not False:
            raise ValueError(f"Oracle contamination flag invalid: {sample_id}")
        if intrinsics_json.get("source") != "derived_from_organized_pcd":
            raise ValueError(f"Unverified intrinsics source: {sample_id}")
        if intrinsics_json.get("factory_calibration") is not False:
            raise ValueError(f"Invalid factory-calibration claim: {sample_id}")
        if intrinsics_json.get("depth_scale_verified") is not True:
            raise ValueError(f"Depth scale is not verified: {sample_id}")
        if float(intrinsics_json.get("depth_scale", 0.0)) != 1000.0:
            raise ValueError(f"Unexpected depth scale for {sample_id}")

        for key in ("source_rgb", "source_depth", "source_pcd"):
            source = Path(str(metadata[key])).resolve()
            try:
                source.relative_to(self.dataset_root)
            except ValueError as error:
                raise ValueError(f"{key} is outside dataset root: {source}") from error
            if not source.is_file():
                raise FileNotFoundError(f"Source provenance file missing: {source}")

        rgb = np.asarray(Image.open(bundle / "color.png").convert("RGB"), dtype=np.uint8)
        depth_mm = np.asarray(Image.open(bundle / "depth.png"))
        if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
            raise ValueError(
                f"Depth must be original uint16 millimetres, got {depth_mm.shape} {depth_mm.dtype}"
            )
        depth_m = depth_mm_to_meters(depth_mm)
        if mask_source == "binary_prediction":
            mask_input = np.asarray(Image.open(bundle / "target_mask.png"))
            threshold = 0.5
        elif mask_source == "probability":
            mask_input = np.load(bundle / "target_probability.npy", allow_pickle=False)
            threshold = float(mask_threshold)
        else:
            raise ValueError(f"Unsupported mask_source: {mask_source}")
        if mask_input.ndim != 2 or not np.all(np.isfinite(mask_input)):
            raise ValueError(f"Mask is not a finite 2D array: {sample_id}")
        if rgb.shape[:2] != depth_m.shape:
            raise ValueError(f"RGB/depth shape mismatch: {sample_id}")

        processing = process_mask_with_diagnostics(
            mask_input,
            depth_m,
            threshold=threshold,
            min_component_size_px=int(min_component_area_px),
            keep_largest_component=bool(retain_largest_component),
            erode_radius_px=int(mask_erode_px),
            dilate_radius_px=int(mask_dilate_px),
        )
        if not np.any(processing.original_binary):
            raise ValueError(f"Original target mask is empty: {sample_id}")
        if not np.any(processing.processed):
            raise ValueError(f"Processed target mask has no valid target depth: {sample_id}")

        values = dict(intrinsics_json)
        values["frame"] = camera_frame
        intrinsics = CameraIntrinsicsData.from_mapping(values)
        if (intrinsics.height, intrinsics.width) != depth_m.shape:
            raise ValueError(f"Intrinsics/depth shape mismatch: {sample_id}")
        return OcidVlgGraspSample(
            sample_id=sample_id,
            sample_index=int(row["sample_index"]),
            question_index=int(row["question_index"]),
            scene_id=str(row["scene_id"]),
            query=str(row["query"]),
            bundle_dir=bundle,
            rgb=rgb,
            depth_mm=depth_mm,
            depth_m=depth_m,
            mask_input=np.array(mask_input, copy=True),
            mask_processing=processing,
            intrinsics=intrinsics,
            intrinsics_metadata=intrinsics_json,
            metadata=metadata,
        )
