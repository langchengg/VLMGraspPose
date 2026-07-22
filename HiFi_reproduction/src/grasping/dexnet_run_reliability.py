"""Reliability primitives for long Dex-Net candidate-generation runs.

This module deliberately contains no grasp-sampling logic.  It validates and
commits already-generated artifacts while leaving the pinned official sampler,
candidate filtering, ordering, and serialization semantics untouched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .camera_geometry import T_CAMERA_GRASP_FIXED_APPROACH_KEY
from .grasp_serialization import candidates_to_arrays


SUCCESS_NONEMPTY = "success_nonempty"
SUCCESS_EMPTY = "success_empty"
FAILED = "failed"
TERMINAL_STATUSES = frozenset({SUCCESS_NONEMPTY, SUCCESS_EMPTY, FAILED})

# These include the numerical inputs required by the existing downstream
# geometric and GQ-CNN rankers.  Diagnostic overlays are intentionally separate.
SUCCESS_REQUIRED_FILES = (
    "metadata.json",
    "raw_candidates.json",
    "mask_validated_candidates.json",
    "filtered_candidates.json",
    "topk_candidates.json",
    "candidates.json",
    "candidates.npz",
    "candidates.csv",
    "rejection_summary.json",
    "camera.intr",
    "depth_m.npy",
    "hifics_mask_processed.png",
)

VISUALIZATION_FILES = (
    "rgb.png",
    "hifics_mask_original.png",
    "valid_depth_mask.png",
    "raw_candidates_overlay.png",
    "filtered_candidates_overlay.png",
    "topk_candidates_overlay.png",
    "rejected_candidates_overlay.png",
    "depth_visualization.png",
    "mask_overlay.png",
)

NPZ_SCHEMA: dict[str, tuple[tuple[int | None, ...], np.dtype[Any]]] = {
    "center_uv": ((None, 2), np.dtype(np.float32)),
    "center_depth_m": ((None,), np.dtype(np.float32)),
    "center_camera_xyz_m": ((None, 3), np.dtype(np.float32)),
    "angle_rad": ((None,), np.dtype(np.float32)),
    "width_m": ((None,), np.dtype(np.float32)),
    "width_px": ((None,), np.dtype(np.float32)),
    "endpoints_uv": ((None, 2, 2), np.dtype(np.float32)),
    "mask_support": ((None,), np.dtype(np.float32)),
    "boundary_distance_px": ((None,), np.dtype(np.float32)),
    "gqcnn_q_value": ((None,), np.dtype(np.float32)),
    "valid": ((None,), np.dtype(np.bool_)),
    T_CAMERA_GRASP_FIXED_APPROACH_KEY: ((None, 4, 4), np.dtype(np.float64)),
}


@dataclass
class ValidationResult:
    sample_id: str
    valid: bool
    status: str | None = None
    errors: list[str] = field(default_factory=list)
    marker: dict[str, Any] | None = None
    summary_row: dict[str, Any] | None = None
    legacy: bool = False


def decide_sample_action(
    *,
    output_exists: bool,
    validation: ValidationResult | None,
    resume: bool,
    overwrite_existing: bool,
    retry_failures: bool,
) -> str:
    """Return ``process`` or ``skip`` for one output without hidden overwrite."""
    if not output_exists:
        return "process"
    if overwrite_existing:
        return "process"
    if validation is not None and validation.valid:
        if validation.status == FAILED and retry_failures:
            return "process"
        if resume:
            return "skip"
        raise FileExistsError("valid output exists; pass --resume or --overwrite-existing")
    if resume:
        return "process"
    raise FileExistsError("incomplete output exists; pass --resume or --overwrite-existing")


def has_identity_mismatch(validation: ValidationResult) -> bool:
    """Return whether invalid output belongs to a different frozen run identity."""
    identity_fragments = (
        "configuration hash mismatch",
        "config-file hash mismatch",
        "seed mismatch",
        "sampler version mismatch",
        "sampler release mismatch",
        "sampler commit mismatch",
        "sampler sampler_class mismatch",
    )
    return any(
        fragment in error
        for error in validation.errors
        for fragment in identity_fragments
    )


def utc_timestamp(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else float(timestamp)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fsync_file(path: Path) -> None:
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=True)
        + "\n",
    )


def atomic_write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=True) + "\n"
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def select_manifest_indices(
    total: int,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    start = 0 if start_index is None else int(start_index)
    end = total if end_index is None else int(end_index)
    if start < 0 or end < 0 or start > end or end > total:
        raise ValueError(f"invalid half-open range [{start}, {end}) for {total} rows")
    if (shard_index is None) != (num_shards is None):
        raise ValueError("--shard-index and --num-shards must be supplied together")
    if num_shards is not None:
        shards = int(num_shards)
        shard = int(shard_index)
        if shards <= 0 or shard < 0 or shard >= shards:
            raise ValueError("shards require num_shards>0 and 0<=shard_index<num_shards")
    else:
        shards = 1
        shard = 0
    return [index for index in range(start, end) if index % shards == shard]


def select_sample_ids(
    canonical_ids: Sequence[str],
    *,
    sample_id: str | None = None,
    sample_limit: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> list[str]:
    ids = [str(value) for value in canonical_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical manifest contains duplicate sample IDs")
    if sample_id is not None:
        if sample_id not in ids:
            raise KeyError(f"unknown sample ID: {sample_id}")
        if any(value is not None for value in (start_index, end_index, shard_index, num_shards)):
            raise ValueError("--sample-id cannot be combined with range or shard selection")
        selected = [sample_id]
    else:
        selected = [ids[index] for index in select_manifest_indices(
            len(ids),
            start_index=start_index,
            end_index=end_index,
            shard_index=shard_index,
            num_shards=num_shards,
        )]
    if sample_limit is not None:
        limit = int(sample_limit)
        if limit <= 0:
            raise ValueError("sample limit must be positive")
        selected = selected[:limit]
    return selected


def make_staging_directory(output_root: Path, sample_id: str) -> Path:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{sample_id}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    staging.mkdir()
    return staging


def _safe_child(path: Path, root: Path) -> Path:
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    if resolved.parent != resolved_root:
        raise ValueError(f"unsafe sample output path: {resolved}")
    return resolved


def recover_interrupted_backup(output_root: Path, sample_id: str) -> None:
    root = Path(output_root).resolve()
    final = _safe_child(root / sample_id, root)
    backups = sorted(root.glob(f".{sample_id}.backup.*"), key=lambda path: path.stat().st_mtime)
    if final.exists():
        return
    if backups:
        os.replace(backups[-1], final)
        fsync_directory(root)


def atomic_commit_sample(staging: Path, final: Path, output_root: Path) -> None:
    root = Path(output_root).resolve()
    final = _safe_child(final, root)
    staging = _safe_child(staging, root)
    if not staging.name.startswith(f".{final.name}.tmp."):
        raise ValueError(f"unexpected staging directory: {staging}")
    fsync_directory(staging)
    backup: Path | None = None
    if final.exists():
        backup = root / f".{final.name}.backup.{os.getpid()}.{uuid.uuid4().hex}"
        os.replace(final, backup)
        try:
            fsync_directory(root)
        except OSError:
            pass
    try:
        os.replace(staging, final)
    except BaseException:
        if backup is not None and backup.exists() and not final.exists():
            os.replace(backup, final)
            fsync_directory(root)
        raise
    # The new directory is now published. Durability maintenance and stale
    # backup cleanup are best-effort: a cleanup failure must never cause the
    # caller to replace a successfully committed sample with a failed marker.
    try:
        fsync_directory(root)
    except OSError:
        pass
    if backup is not None and backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError:
            pass


def remove_staging_directory(staging: Path, output_root: Path) -> None:
    staging = _safe_child(staging, output_root)
    if staging.name.startswith(".") and ".tmp." in staging.name and staging.exists():
        shutil.rmtree(staging)


def required_file_hashes(sample_dir: Path, files: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in files:
        path = Path(sample_dir) / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required output missing or unsafe: {path}")
        fsync_file(path)
        hashes[name] = sha256_file(path)
    return hashes


def write_completion_marker(
    sample_dir: Path,
    *,
    sample_id: str,
    question_index: int,
    configuration_hash: str,
    config_file_sha256: str,
    seed: int,
    sampler_runtime: Mapping[str, Any],
    counts: Mapping[str, Any],
    status: str,
    required_files: Sequence[str],
    visualization_files: Sequence[str] = (),
    summary_row: Mapping[str, Any] | None = None,
    failure_reason: str | None = None,
    attempt: int = 1,
) -> dict[str, Any]:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"invalid terminal status: {status}")
    required = list(dict.fromkeys(str(name) for name in required_files))
    visual = list(dict.fromkeys(str(name) for name in visualization_files))
    marker = {
        "schema_version": 1,
        "sample_id": str(sample_id),
        "question_index": int(question_index),
        "configuration_hash": str(configuration_hash),
        "config_file_sha256": str(config_file_sha256),
        "seed": int(seed),
        "sampler_version": sampler_runtime.get("version"),
        "sampler_release": sampler_runtime.get("release"),
        "sampler_commit": sampler_runtime.get("commit"),
        "sampler_class": sampler_runtime.get("sampler_class"),
        "candidate_counts": dict(counts),
        "completion_timestamp": utc_timestamp(),
        "required_files": required,
        "required_file_hashes": required_file_hashes(sample_dir, required),
        "visualization_files": visual,
        "status": status,
        "failure_reason": failure_reason,
        "attempt": int(attempt),
        "summary_row": None if summary_row is None else dict(summary_row),
    }
    atomic_write_json(Path(sample_dir) / "_SUCCESS.json", marker)
    return marker


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_npz(
    path: Path,
    expected_count: int,
    errors: list[str],
    *,
    expected_records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(NPZ_SCHEMA):
                errors.append(
                    f"NPZ keys differ: missing={sorted(set(NPZ_SCHEMA)-set(arrays.files))} "
                    f"extra={sorted(set(arrays.files)-set(NPZ_SCHEMA))}"
                )
                return
            for name, (shape_spec, dtype) in NPZ_SCHEMA.items():
                value = arrays[name]
                expected_shape = tuple(expected_count if item is None else item for item in shape_spec)
                if value.shape != expected_shape:
                    errors.append(f"NPZ {name} shape {value.shape} != {expected_shape}")
                if value.dtype != dtype:
                    errors.append(f"NPZ {name} dtype {value.dtype} != {dtype}")
            finite_keys = (
                "center_uv",
                "center_depth_m",
                "center_camera_xyz_m",
                "angle_rad",
                "width_m",
                "width_px",
                "endpoints_uv",
                "mask_support",
                "boundary_distance_px",
                T_CAMERA_GRASP_FIXED_APPROACH_KEY,
            )
            for name in finite_keys:
                if not np.all(np.isfinite(arrays[name])):
                    errors.append(f"NPZ {name} contains non-finite values")
            if expected_count and not np.all(arrays["valid"]):
                errors.append("NPZ retained candidates include invalid rows")
            if expected_records is not None:
                expected_arrays = candidates_to_arrays(expected_records)
                for name in NPZ_SCHEMA:
                    if not np.array_equal(
                        arrays[name], expected_arrays[name], equal_nan=True
                    ):
                        errors.append(f"NPZ {name} values differ from candidates.json")
    except Exception as error:
        errors.append(f"NPZ unreadable: {type(error).__name__}: {error}")


def _validate_success_payload(
    sample_dir: Path,
    sample_id: str,
    *,
    expected_configuration_hash: str | None,
    expected_seed: int | None,
    expected_sampler_runtime: Mapping[str, Any] | None,
    errors: list[str],
) -> tuple[str | None, dict[str, Any] | None]:
    missing = [name for name in SUCCESS_REQUIRED_FILES if not (sample_dir / name).is_file()]
    if missing:
        errors.append(f"missing required files: {missing}")
        return None, None
    try:
        metadata = _load_json(sample_dir / "metadata.json")
        raw = _load_json(sample_dir / "raw_candidates.json")
        mask_valid = _load_json(sample_dir / "mask_validated_candidates.json")
        filtered = _load_json(sample_dir / "filtered_candidates.json")
        topk = _load_json(sample_dir / "topk_candidates.json")
        bundle = _load_json(sample_dir / "candidates.json")
        rejection = _load_json(sample_dir / "rejection_summary.json")
    except Exception as error:
        errors.append(f"JSON unreadable: {type(error).__name__}: {error}")
        return None, None
    if metadata.get("sample_id") != sample_id:
        errors.append("metadata sample_id mismatch")
    if metadata.get("representation") != "planar_parallel_jaw_4dof":
        errors.append("representation is not planar_parallel_jaw_4dof")
    if metadata.get("approach_constraint") != "fixed_camera_optical_axis":
        errors.append("approach constraint is not fixed camera optical axis")
    if not metadata.get("camera_frame"):
        errors.append("camera frame is missing")
    pose = metadata.get("pose", {})
    if pose.get("name") != T_CAMERA_GRASP_FIXED_APPROACH_KEY:
        errors.append("fixed-approach pose name mismatch")
    if pose.get("is_freely_predicted_6dof") is not False:
        errors.append("fixed-approach pose is mislabelled as free 6-DoF")
    if expected_seed is not None and int(metadata.get("seed", -1)) != int(expected_seed):
        errors.append("seed mismatch")
    if expected_configuration_hash is not None:
        actual_hash = canonical_json_hash(metadata.get("config"))
        if actual_hash != expected_configuration_hash:
            errors.append("configuration hash mismatch")
    if expected_sampler_runtime is not None:
        metadata_runtime = metadata.get("gqcnn_runtime", {})
        for key in ("version", "release", "commit", "sampler_class"):
            expected = expected_sampler_runtime.get(key)
            if expected is not None and metadata_runtime.get(key) != expected:
                errors.append(f"metadata sampler {key} mismatch")
    if not all(isinstance(value, list) for value in (raw, mask_valid, filtered, topk)):
        errors.append("candidate stage JSON must be lists")
        return None, None
    counts = metadata.get("counts", {})
    try:
        requested = int(counts["requested"])
        raw_count = int(counts["raw"])
        mask_count = int(counts["mask_validated"])
        post_count = int(counts["post_nms"])
        top_count = int(counts["top_k"])
    except Exception as error:
        errors.append(f"metadata counts invalid: {error}")
        return None, None
    if not (0 <= post_count <= mask_count <= raw_count <= requested):
        errors.append("candidate count ordering is invalid")
    if (len(raw), len(mask_valid), len(filtered), len(topk)) != (
        raw_count,
        mask_count,
        post_count,
        top_count,
    ):
        errors.append("JSON candidate counts disagree with metadata")
    if top_count > post_count:
        errors.append("top-K count exceeds post-NMS count")
    if not isinstance(bundle, dict) or not isinstance(bundle.get("candidates"), list):
        errors.append("candidates.json bundle is malformed")
        final_records: list[Any] = []
    else:
        final_records = bundle["candidates"]
        if bundle.get("metadata") != metadata:
            errors.append("candidates.json metadata differs from metadata.json")
    if len(final_records) != post_count:
        errors.append("candidates.json count disagrees with metadata")
    final_ids = [str(item.get("candidate_id")) for item in final_records if isinstance(item, dict)]
    if len(final_ids) != len(set(final_ids)):
        errors.append("duplicate candidate IDs")
    filtered_ids = [str(item.get("candidate_id")) for item in filtered if isinstance(item, dict)]
    top_ids = [str(item.get("candidate_id")) for item in topk if isinstance(item, dict)]
    if final_ids != filtered_ids:
        errors.append("candidates.json and filtered candidate ordering differ")
    if top_ids != filtered_ids[:top_count]:
        errors.append("top-K candidates are not the filtered prefix")
    if metadata.get("rejection_summary") != rejection:
        errors.append("rejection summary mismatch")
    try:
        with (sample_dir / "candidates.csv").open(encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        if [row.get("candidate_id") for row in csv_rows] != final_ids:
            errors.append("CSV candidate count/order differs")
    except Exception as error:
        errors.append(f"CSV unreadable: {type(error).__name__}: {error}")
    _validate_npz(
        sample_dir / "candidates.npz",
        post_count,
        errors,
        expected_records=final_records,
    )
    status = SUCCESS_NONEMPTY if post_count > 0 else SUCCESS_EMPTY
    if status == SUCCESS_EMPTY and not metadata.get("failure_reason"):
        errors.append("success_empty requires a failure_reason explanation")
    summary = {
        "sample_id": sample_id,
        "query": metadata.get("query", ""),
        "mask_area_px": metadata.get("mask_area_px", ""),
        "valid_target_depth_px": metadata.get("valid_target_depth_px", ""),
        "requested_candidate_count": requested,
        "raw_candidate_count": raw_count,
        "mask_validated_count": mask_count,
        "post_nms_count": post_count,
        "scored_candidate_count": int(counts.get("scored", 0)),
        "best_gqcnn_q": "",
        "median_gqcnn_q": "",
        "generation_time_ms": metadata.get("timing_ms", {}).get("generation", ""),
        "scoring_time_ms": metadata.get("timing_ms", {}).get("scoring", ""),
        "total_time_ms": metadata.get("timing_ms", {}).get("total", ""),
        "failure_reason": metadata.get("failure_reason") or "",
        "status": status,
        "question_index": metadata.get("question_index", ""),
        "scene_id": metadata.get("scene_id", ""),
    }
    return status, summary


def validate_sample_output(
    sample_dir: Path,
    *,
    expected_sample_id: str,
    expected_configuration_hash: str | None = None,
    expected_config_file_sha256: str | None = None,
    expected_seed: int | None = None,
    expected_sampler_runtime: Mapping[str, Any] | None = None,
    expect_visualizations: bool | None = None,
    verify_hashes: bool = True,
    allow_legacy: bool = False,
) -> ValidationResult:
    directory = Path(sample_dir)
    errors: list[str] = []
    if not directory.is_dir() or directory.is_symlink():
        return ValidationResult(expected_sample_id, False, errors=["sample directory missing or unsafe"])
    marker_path = directory / "_SUCCESS.json"
    marker: dict[str, Any] | None = None
    if marker_path.is_file():
        try:
            marker = _load_json(marker_path)
        except Exception as error:
            return ValidationResult(
                expected_sample_id,
                False,
                errors=[f"completion marker unreadable: {type(error).__name__}: {error}"],
            )
        status = marker.get("status")
        if status not in TERMINAL_STATUSES:
            errors.append("completion marker status invalid")
        if marker.get("sample_id") != expected_sample_id:
            errors.append("completion marker sample_id mismatch")
        if expected_configuration_hash is not None and marker.get("configuration_hash") != expected_configuration_hash:
            errors.append("completion marker configuration hash mismatch")
        if expected_config_file_sha256 is not None and marker.get("config_file_sha256") != expected_config_file_sha256:
            errors.append("completion marker config-file hash mismatch")
        if expected_seed is not None and int(marker.get("seed", -1)) != int(expected_seed):
            errors.append("completion marker seed mismatch")
        if expected_sampler_runtime is not None:
            marker_runtime = {
                "version": marker.get("sampler_version"),
                "release": marker.get("sampler_release"),
                "commit": marker.get("sampler_commit"),
                "sampler_class": marker.get("sampler_class"),
            }
            for key, expected in expected_sampler_runtime.items():
                if key in marker_runtime and expected is not None and marker_runtime[key] != expected:
                    errors.append(f"completion marker sampler {key} mismatch")
        required = marker.get("required_files")
        hashes = marker.get("required_file_hashes")
        if not isinstance(required, list) or not isinstance(hashes, dict):
            errors.append("completion marker required-file manifest invalid")
        else:
            for name in required:
                path = directory / str(name)
                if Path(str(name)).name != str(name) or not path.is_file() or path.is_symlink():
                    errors.append(f"required file missing or unsafe: {name}")
                elif verify_hashes and hashes.get(name) != sha256_file(path):
                    errors.append(f"required file hash mismatch: {name}")
            required_set = {str(name) for name in required}
            minimum_required = (
                {"failure.json"} if status == FAILED else set(SUCCESS_REQUIRED_FILES)
            )
            missing_from_marker = sorted(minimum_required - required_set)
            if missing_from_marker:
                errors.append(
                    f"completion marker omits required files: {missing_from_marker}"
                )
            if set(hashes) != required_set:
                errors.append("completion marker hash keys differ from required files")
        visualizations = marker.get("visualization_files")
        if not isinstance(visualizations, list):
            errors.append("completion marker visualization-file manifest invalid")
        else:
            visual_set = {str(name) for name in visualizations}
            if isinstance(required, list) and not visual_set.issubset(
                {str(name) for name in required}
            ):
                errors.append("visualization files are not covered by required-file hashes")
            if expect_visualizations is True and status != FAILED:
                missing_visualizations = sorted(set(VISUALIZATION_FILES) - visual_set)
                if missing_visualizations:
                    errors.append(
                        f"visualization policy requires files: {missing_visualizations}"
                    )
            if expect_visualizations is False and visual_set:
                errors.append("visualization policy forbids per-sample overlays")
        if status == FAILED:
            if not marker.get("failure_reason"):
                errors.append("failed marker has no failure reason")
            marker_summary = marker.get("summary_row")
            if not isinstance(marker_summary, dict):
                errors.append("failed marker has no summary row")
            else:
                if marker_summary.get("sample_id") != expected_sample_id:
                    errors.append("failed marker summary sample_id mismatch")
                if marker_summary.get("status") != FAILED:
                    errors.append("failed marker summary status mismatch")
            return ValidationResult(
                expected_sample_id,
                not errors,
                status=FAILED,
                errors=errors,
                marker=marker,
                summary_row=marker.get("summary_row"),
            )
        payload_status, summary = _validate_success_payload(
            directory,
            expected_sample_id,
            expected_configuration_hash=expected_configuration_hash,
            expected_seed=expected_seed,
            expected_sampler_runtime=expected_sampler_runtime,
            errors=errors,
        )
        if payload_status != status:
            errors.append(f"marker status {status} disagrees with payload {payload_status}")
        marker_summary = marker.get("summary_row")
        if not isinstance(marker_summary, dict):
            errors.append("completion marker has no summary row")
        elif summary is not None:
            for key in (
                "sample_id",
                "status",
                "requested_candidate_count",
                "raw_candidate_count",
                "mask_validated_count",
                "post_nms_count",
            ):
                if marker_summary.get(key) != summary.get(key):
                    errors.append(f"completion marker summary {key} mismatch")
        try:
            metadata_counts = _load_json(directory / "metadata.json").get("counts", {})
            marker_counts = marker.get("candidate_counts")
            if not isinstance(marker_counts, dict):
                errors.append("completion marker candidate counts invalid")
            else:
                for key in ("requested", "raw", "mask_validated", "post_nms", "top_k"):
                    if int(marker_counts.get(key, -1)) != int(metadata_counts.get(key, -2)):
                        errors.append(f"completion marker candidate count {key} mismatch")
        except Exception as error:
            errors.append(f"completion marker count validation failed: {error}")
        return ValidationResult(
            expected_sample_id,
            not errors,
            status=status,
            errors=errors,
            marker=marker,
            summary_row=summary,
        )
    if not allow_legacy:
        return ValidationResult(expected_sample_id, False, errors=["completion marker missing"])
    status, summary = _validate_success_payload(
        directory,
        expected_sample_id,
        expected_configuration_hash=expected_configuration_hash,
        expected_seed=expected_seed,
        expected_sampler_runtime=expected_sampler_runtime,
        errors=errors,
    )
    return ValidationResult(
        expected_sample_id,
        not errors,
        status=status,
        errors=errors,
        summary_row=summary,
        legacy=True,
    )


def sample_directory_digest(sample_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in Path(sample_dir).rglob("*") if item.is_file()):
        relative = path.relative_to(sample_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
