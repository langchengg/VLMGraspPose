"""Batch orchestration utilities for HiFi-CS target masks and official VGN."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)

TSDF_MODE = "single_view_adaptation"
LIMITATIONS = [
    "single-view TSDF adaptation",
    "no 6-DoF ground truth in OCID-VLG",
    "no robot execution validation",
]
SCORE_SOURCE = "official_vgn_processed_quality"

SUMMARY_FIELDS = (
    "sample_id",
    "scene_id",
    "instruction",
    "mask_area",
    "valid_target_depth_points",
    "support_plane_residual",
    "official_candidate_count",
    "target_candidate_count",
    "top1_vgn_quality",
    "top1_width_m",
    "top1_x_task",
    "top1_y_task",
    "top1_z_task",
    "processing_time_depth",
    "processing_time_tsdf",
    "processing_time_vgn",
    "status",
    "failure_reason",
)


class PipelineError(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    dataset_index: int
    scene_id: str
    instruction: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path
    bundle_dir: Path | None
    metadata_path: Path | None
    intrinsics_path: Path | None
    view: str | None
    row: Mapping[str, Any]
    metadata: Mapping[str, Any]


FIELD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "sample_id": ("sample_id", "stable_sample_id", "evaluation_sample_id", "id"),
    "dataset_index": ("dataset_index", "sample_index", "index"),
    "instruction": ("text", "instruction", "query", "language"),
    "scene_id": ("scene_id", "scene", "sequence_id"),
    "rgb_path": ("rgb_dataset_rel", "rgb_path", "color_path", "image_path"),
    "depth_path": ("depth_dataset_rel", "depth_path"),
    "mask_path": ("pred_mask_rel", "pred_mask_path", "target_mask_path", "mask_path"),
    "bundle_dir": ("output_dir", "bundle_dir", "sample_dir"),
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and isinstance(payload.get("samples"), list):
            rows = payload["samples"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise PipelineError("manifest_schema_error", "JSON manifest must be a list or contain a samples list")
    else:
        raise PipelineError("manifest_schema_error", f"Unsupported manifest format: {path.suffix}")
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise PipelineError("manifest_schema_error", "Manifest contains no object rows")
    return rows


def detect_manifest_mapping(rows: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    keys = set().union(*(row.keys() for row in rows[: min(100, len(rows))]))
    mapping: dict[str, str | None] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        matches = [name for name in aliases if name in keys]
        if len(matches) > 1:
            # Multiple known aliases are permitted only when their values agree.
            for row in rows[: min(100, len(rows))]:
                present = [row[name] for name in matches if row.get(name) not in (None, "")]
                if present and any(str(value) != str(present[0]) for value in present[1:]):
                    raise PipelineError(
                        "manifest_schema_error",
                        f"Conflicting aliases for {canonical}: {matches}",
                    )
        mapping[canonical] = matches[0] if matches else None
    required = ("sample_id", "dataset_index", "instruction", "scene_id")
    missing = [name for name in required if mapping[name] is None]
    if missing:
        raise PipelineError("manifest_schema_error", f"Manifest lacks required fields: {missing}")
    if mapping["bundle_dir"] is None and any(
        mapping[name] is None for name in ("rgb_path", "depth_path", "mask_path")
    ):
        raise PipelineError(
            "manifest_schema_error",
            "Manifest must provide explicit RGB/depth/mask fields or a recognized bundle directory",
        )
    return mapping


def _resolve_path(value: Any, bases: Sequence[Path]) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()


def _view_from_scene(scene_id: str, metadata: Mapping[str, Any]) -> str | None:
    for key in ("camera_view_from_sequence_path", "camera_view", "view", "sensor"):
        explicit = metadata.get(key)
        if explicit not in (None, ""):
            return str(explicit)
    parts = str(scene_id).replace(",", "/").split("/")
    for view in ("top", "bottom"):
        if view in parts:
            return view
    return None


def _manifest_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise PipelineError("manifest_schema_error", f"Invalid manifest boolean: {value!r}")


def _manifest_blockers(value: Any) -> list[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def load_manifest_samples(
    manifest: Path | str,
    *,
    ocid_root: Path | str,
    hifi_root: Path | str,
    logger: logging.Logger = LOGGER,
) -> tuple[list[ManifestSample], dict[str, str | None]]:
    manifest_path = Path(manifest).expanduser().resolve()
    ocid = Path(ocid_root).expanduser().resolve()
    hifi = Path(hifi_root).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not ocid.is_dir():
        raise FileNotFoundError(f"OCID-VLG root not found: {ocid}")
    rows = _read_manifest_rows(manifest_path)
    mapping = detect_manifest_mapping(rows)
    logger.info("Detected manifest fields: %s", sorted(rows[0].keys()))
    logger.info("Manifest adapter mapping:")
    for name, source in mapping.items():
        logger.info("  %-13s <- %s", name, source or "bundle-derived")

    samples: list[ManifestSample] = []
    ids: set[str] = set()
    indices: set[int] = set()
    for row_number, row in enumerate(rows, 1):
        for readiness_field in ("ready", "ready_for_anygrasp"):
            if readiness_field in row and not _manifest_bool(row[readiness_field]):
                raise PipelineError(
                    "manifest_sample_not_ready",
                    f"Row {row_number} has {readiness_field}=false",
                )
        blockers = _manifest_blockers(row.get("blockers"))
        if blockers:
            raise PipelineError(
                "manifest_sample_not_ready",
                f"Row {row_number} has blockers: {blockers}",
            )
        sample_id = str(row[str(mapping["sample_id"])])
        try:
            dataset_index = int(row[str(mapping["dataset_index"])])
        except (TypeError, ValueError) as error:
            raise PipelineError("manifest_schema_error", f"Invalid dataset index at row {row_number}") from error
        if sample_id in ids or dataset_index in indices:
            raise PipelineError("duplicate_manifest_sample", f"Duplicate sample id/index at row {row_number}")
        ids.add(sample_id)
        indices.add(dataset_index)

        bundle = None
        if mapping["bundle_dir"] is not None:
            bundle = _resolve_path(row[str(mapping["bundle_dir"])], (manifest_path.parent, hifi))
        metadata_path = bundle / "metadata.json" if bundle is not None else None
        bundle_metadata: Mapping[str, Any] = {}
        if metadata_path is not None and metadata_path.is_file():
            bundle_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # Manifest metadata is a valid source of depth/view provenance.  Bundle
        # metadata is more specific, but conflicting values are never silently
        # accepted.
        for key in ("depth_scale", "depth_unit", "mask_source"):
            if (
                key in row
                and key in bundle_metadata
                and row[key] not in (None, "")
                and bundle_metadata[key] not in (None, "")
                and str(row[key]) != str(bundle_metadata[key])
            ):
                raise PipelineError(
                    "manifest_bundle_mismatch",
                    f"{key} metadata mismatch for {sample_id}",
                )
        metadata = {**dict(row), **dict(bundle_metadata)}

        def explicit_or_bundle(name: str, bundle_name: str, raw_metadata_name: str | None = None) -> Path:
            source_field = mapping[name]
            if source_field is not None and row.get(source_field) not in (None, ""):
                return _resolve_path(row[source_field], (manifest_path.parent, hifi, ocid))
            if raw_metadata_name and metadata.get(raw_metadata_name) not in (None, ""):
                raw = _resolve_path(metadata[raw_metadata_name], (ocid, manifest_path.parent))
                try:
                    raw.relative_to(ocid)
                except ValueError as error:
                    raise PipelineError("source_path_outside_ocid_root", f"{raw} is outside {ocid}") from error
                return raw
            if bundle is None:
                raise PipelineError("manifest_schema_error", f"Cannot derive {name} at row {row_number}")
            return (bundle / bundle_name).resolve()

        rgb_path = explicit_or_bundle("rgb_path", "color.png", "source_rgb")
        depth_path = explicit_or_bundle("depth_path", "depth.png", "source_depth")
        mask_path = explicit_or_bundle("mask_path", "target_mask.png")
        missing = [path for path in (rgb_path, depth_path, mask_path) if not path.is_file()]
        if missing:
            raise PipelineError("missing_sample_file", f"Row {row_number} missing files: {missing}")
        if metadata:
            for key in ("sample_id", "scene_id", "query"):
                expected_key = {"query": "instruction"}.get(key, key)
                canonical_value = {
                    "sample_id": sample_id,
                    "scene_id": str(row[str(mapping["scene_id"])]),
                    "instruction": str(row[str(mapping["instruction"])]),
                }[expected_key]
                if key in metadata and str(metadata[key]) != canonical_value:
                    raise PipelineError("manifest_bundle_mismatch", f"{key} mismatch for {sample_id}")
            if metadata.get("mask_source") not in (None, "predicted_mask_original_resolution"):
                raise PipelineError("oracle_mask_forbidden", f"Non-predicted mask source for {sample_id}")

        mask_source_field = mapping["mask_path"]
        generic_mask_field = mask_source_field in {"target_mask_path", "mask_path"}
        declared_predicted = "predicted" in str(metadata.get("mask_source", "")).lower()
        forbidden_parts = {"gt", "ground_truth", "groundtruth", "ground-truth"}
        path_parts = {part.lower() for part in mask_path.parts}
        if path_parts & forbidden_parts or mask_path.stem.lower().startswith(("gt_", "ground_truth")):
            raise PipelineError("oracle_mask_forbidden", f"GT/oracle-looking mask path for {sample_id}")
        if bundle is None and generic_mask_field and not declared_predicted:
            raise PipelineError(
                "oracle_mask_forbidden",
                f"Generic manifest mask field for {sample_id} requires mask_source=predicted...",
            )

        intrinsics_path = bundle / "intrinsics.json" if bundle is not None else None
        samples.append(
            ManifestSample(
                sample_id=sample_id,
                dataset_index=dataset_index,
                scene_id=str(row[str(mapping["scene_id"])]),
                instruction=str(row[str(mapping["instruction"])]),
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_path=mask_path,
                bundle_dir=bundle,
                metadata_path=metadata_path if metadata_path and metadata_path.is_file() else None,
                intrinsics_path=intrinsics_path if intrinsics_path and intrinsics_path.is_file() else None,
                view=_view_from_scene(str(row[str(mapping["scene_id"])]), metadata),
                row=dict(row),
                metadata=dict(metadata),
            )
        )
    logger.info(
        "Bundle path mapping: rgb/depth <- metadata source_rgb/source_depth when present; "
        "pred_mask <- bundle target_mask.png; intrinsics <- explicit --intrinsics config"
    )
    return samples, mapping


def build_stem_manifest(
    *,
    ocid_root: Path | str,
    hifi_root: Path | str,
    report_path: Path | str,
) -> list[ManifestSample]:
    """Strict manifest-free fallback; every stem must map one-to-one."""
    ocid = Path(ocid_root).expanduser().resolve()
    hifi = Path(hifi_root).expanduser().resolve()

    def grouped(paths: Iterable[Path]) -> dict[str, list[Path]]:
        result: dict[str, list[Path]] = {}
        for path in paths:
            result.setdefault(path.stem, []).append(path.resolve())
        return result

    rgb = grouped(path for path in ocid.rglob("*.png") if path.parent.name.lower() == "rgb")
    depth = grouped(path for path in ocid.rglob("*.png") if path.parent.name.lower() == "depth")
    masks = grouped(
        path for path in hifi.rglob("*.png")
        if "mask" in path.name.lower() and "ground" not in path.name.lower() and "gt" not in path.name.lower()
    )
    all_stems = sorted(set(rgb) | set(depth) | set(masks))
    report = {
        "duplicates": {
            kind: {stem: [str(path) for path in values[stem]] for stem in sorted(values) if len(values[stem]) != 1}
            for kind, values in (("rgb", rgb), ("depth", depth), ("mask", masks))
        },
        "unmatched": {
            stem: {"rgb": len(rgb.get(stem, [])), "depth": len(depth.get(stem, [])), "mask": len(masks.get(stem, []))}
            for stem in all_stems
            if not (len(rgb.get(stem, [])) == len(depth.get(stem, [])) == len(masks.get(stem, [])) == 1)
        },
    }
    atomic_write_json(report_path, report)
    if any(report["duplicates"].values()) or report["unmatched"]:
        raise PipelineError(
            "unmatched_files",
            f"Stem fallback is not one-to-one; see {Path(report_path).resolve()}",
        )
    samples = []
    for index, stem in enumerate(all_stems):
        samples.append(
            ManifestSample(
                sample_id=stem,
                dataset_index=index,
                scene_id=str(rgb[stem][0].parent.parent),
                instruction="",
                rgb_path=rgb[stem][0],
                depth_path=depth[stem][0],
                mask_path=masks[stem][0],
                bundle_dir=None,
                metadata_path=None,
                intrinsics_path=None,
                view=_view_from_scene(str(rgb[stem][0]), {}),
                row={},
                metadata={"mask_source": "predicted_mask_stem_fallback_unverified"},
            )
        )
    return samples


def load_sample_arrays(sample: ManifestSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_hashes = (
        (sample.rgb_path, sample.metadata.get("source_rgb_sha256"), "RGB"),
        (sample.depth_path, sample.metadata.get("source_depth_sha256"), "depth"),
        (sample.mask_path, sample.metadata.get("prediction_mask_sha256"), "predicted mask"),
    )
    for path, expected, label in expected_hashes:
        if expected not in (None, ""):
            actual = sha256_file(path)
            if actual != str(expected):
                raise PipelineError(
                    "source_checksum_mismatch",
                    f"{label} SHA256 mismatch for {sample.sample_id}: {path}",
                )
    rgb = np.asarray(Image.open(sample.rgb_path).convert("RGB"), dtype=np.uint8)
    depth = np.asarray(Image.open(sample.depth_path))
    mask = np.asarray(Image.open(sample.mask_path))
    if depth.ndim != 2:
        raise PipelineError("depth_shape_error", f"Depth must be HxW, got {depth.shape}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise PipelineError("mask_depth_shape_error", f"Mask must be HxW, got {mask.shape}")
    if rgb.shape[:2] != depth.shape:
        raise PipelineError("rgb_depth_shape_error", f"RGB/depth mismatch {rgb.shape[:2]} != {depth.shape}")
    return rgb, depth, mask


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_path(destination: Path) -> tuple[int, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    return fd, Path(name)


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    destination = Path(path)
    fd, temporary = _atomic_path(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_write_csv(path: Path | str, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    destination = Path(path)
    fd, temporary = _atomic_path(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _json_safe(row.get(field, "")) for field in fields})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def candidate_records(candidates: Iterable[Any]) -> list[dict[str, Any]]:
    records = []
    for candidate in candidates:
        converter = getattr(candidate, "to_record", None)
        value = converter() if callable(converter) else (dict(candidate) if isinstance(candidate, Mapping) else vars(candidate))
        records.append(dict(_json_safe(value)))
    return records


def atomic_write_candidates_npz(path: Path | str, candidates: Iterable[Any]) -> Path:
    records = candidate_records(candidates)
    count = len(records)
    arrays = {
        "official_selection_index": np.asarray([r.get("official_selection_index", -1) for r in records], dtype=np.int64),
        "score_rank": np.asarray([r.get("score_rank", -1) for r in records], dtype=np.int64),
        "vgn_quality": np.asarray([r.get("vgn_quality", np.nan) for r in records], dtype=np.float32),
        "voxel_index_ijk": np.asarray([r.get("voxel_index_ijk", [-1, -1, -1]) for r in records], dtype=np.int16).reshape(count, 3),
        "position_task_m": np.asarray([r.get("position_task_m", [np.nan] * 3) for r in records], dtype=np.float32).reshape(count, 3),
        "quaternion_task_xyzw": np.asarray([r.get("quaternion_task_xyzw", [np.nan] * 4) for r in records], dtype=np.float32).reshape(count, 4),
        "width_m": np.asarray([r.get("width_m", np.nan) for r in records], dtype=np.float32),
        "T_task_grasp": np.asarray([r.get("T_task_grasp", np.full((4, 4), np.nan)) for r in records], dtype=np.float64).reshape(count, 4, 4),
        "T_camera_grasp": np.asarray([r.get("T_camera_grasp", np.full((4, 4), np.nan)) for r in records], dtype=np.float64).reshape(count, 4, 4),
        "inside_dilated_target_mask": np.asarray([r.get("inside_dilated_target_mask", False) for r in records], dtype=np.bool_),
    }
    destination = Path(path)
    fd, temporary = _atomic_path(destination)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_write_npz(path: Path | str, **arrays: np.ndarray) -> Path:
    destination = Path(path)
    fd, temporary = _atomic_path(destination)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **{name: np.asarray(value) for name, value in arrays.items()})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def environment_versions() -> dict[str, Any]:
    import open3d
    import scipy
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pytorch": torch.__version__,
        "open3d": open3d.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def empty_summary(sample: ManifestSample, *, status: str, reason: str) -> dict[str, Any]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        sample_id=sample.sample_id,
        scene_id=sample.scene_id,
        instruction=sample.instruction,
        official_candidate_count=0,
        target_candidate_count=0,
        status=status,
        failure_reason=reason,
    )
    return row


def candidate_status(official_candidate_count: int, target_candidate_count: int) -> str:
    """Return the clean sample status without substituting an off-target grasp."""
    official = int(official_candidate_count)
    target = int(target_candidate_count)
    if official < 0 or target < 0 or target > official:
        raise ValueError("candidate counts must satisfy 0 <= target <= official")
    if official == 0:
        return "no_official_grasp"
    if target == 0:
        return "no_target_grasp"
    return "ok"


def elapsed(start: float) -> float:
    return float(time.perf_counter() - start)
