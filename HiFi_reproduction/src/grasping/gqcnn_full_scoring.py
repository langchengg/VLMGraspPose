"""Reliable GQ-CNN scoring of immutable, already-sampled candidates.

This module is deliberately Python 3.7 compatible because the official
GQ-CNN v1.3.0 / TensorFlow 1.15 runtime is isolated in a linux/amd64 image.
It contains no sampler and no CEM policy.  Source candidate directories are
opened read-only and every scored sample is committed to a separate root.
"""

from __future__ import absolute_import, division, print_function

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
import traceback
import uuid
import zipfile
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
GQCNN_COMMIT = "499a609fe9dfb074bdfb6c4e6e33667ea50f4c21"
SCORED_NONEMPTY = "scored_nonempty"
SKIPPED_VALID_EMPTY = "skipped_valid_empty"
SCORING_FAILED = "scoring_failed"
TERMINAL_STATUSES = (SCORED_NONEMPTY, SKIPPED_VALID_EMPTY, SCORING_FAILED)
MARKER_NAME = "_SCORING_COMPLETE.json"
NONEMPTY_FILES = (
    "gqcnn_scored_candidates.npz",
    "gqcnn_scored_candidates.json",
    "gqcnn_scored_candidates.csv",
    "gqcnn_top1.json",
    "gqcnn_top5.json",
    "scoring_metadata.json",
)
EMPTY_FILES = ("scoring_metadata.json",)
FAILED_FILES = ("failure.json", "scoring_metadata.json")
POSE_ARRAY_KEYS = (
    "center_uv",
    "center_depth_m",
    "center_camera_xyz_m",
    "angle_rad",
    "width_m",
    "width_px",
    "endpoints_uv",
    "T_camera_grasp_fixed_approach",
)
SUMMARY_FIELDS = (
    "sample_id",
    "query",
    "source_candidate_count",
    "scored_candidate_count",
    "scoring_status",
    "top1_candidate_id",
    "top1_q_value",
    "top5_candidate_ids",
    "max_q_value",
    "min_q_value",
    "mean_q_value",
    "median_q_value",
    "std_q_value",
    "q_value_range",
    "scoring_time_ms",
    "preprocessing_time_ms",
    "serialization_time_ms",
    "total_time_ms",
    "model_name",
    "model_config_hash",
    "source_candidate_sha256",
    "failure_reason",
)


class ScoringValidationError(ValueError):
    """Raised when a source or committed scored output violates its schema."""


def utc_timestamp(timestamp=None):
    value = time.time() if timestamp is None else float(timestamp)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def assert_disjoint_roots(candidate_root, output_root):
    """Reject source/output aliases and ancestor relationships before any write."""

    source = Path(candidate_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    same_filesystem_object = False
    if source.exists() and output.exists():
        try:
            same_filesystem_object = os.path.samefile(str(source), str(output))
        except OSError:
            same_filesystem_object = False
    if (
        same_filesystem_object
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError(
            "candidate and scored output roots must be disjoint: %s versus %s"
            % (source, output)
        )
    return source, output


def canonical_json_hash(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strict_value(value):
    if isinstance(value, np.ndarray):
        return strict_value(value.tolist())
    if isinstance(value, np.generic):
        return strict_value(value.item())
    if isinstance(value, dict):
        return {str(key): strict_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fsync_directory(path):
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path, text):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination.name,
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(destination))
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path, value):
    payload = json.dumps(
        strict_value(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, payload)


def atomic_write_jsonl(path, rows):
    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                strict_value(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def atomic_write_csv(path, rows, fieldnames):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination.name,
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            for source in rows:
                row = {}
                for name in fieldnames:
                    value = strict_value(source.get(name))
                    if isinstance(value, (list, dict)):
                        value = json.dumps(
                            value,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    row[name] = "" if value is None else value
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(destination))
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def deterministic_save_npz(path, arrays):
    destination = Path(path)
    with zipfile.ZipFile(str(destination), mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(
                "%s.npy" % name, date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_candidate_payload(path):
    payload = _json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ScoringValidationError("%s has no candidates list" % path)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ScoringValidationError("%s metadata is not an object" % path)
    return [dict(item) for item in payload["candidates"]], dict(metadata)


def _assert_array_equal(name, actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ScoringValidationError(
            "%s shape/dtype differs: %s %s versus %s %s"
            % (name, actual.shape, actual.dtype, expected.shape, expected.dtype)
        )
    if not np.array_equal(actual, expected, equal_nan=True):
        raise ScoringValidationError("%s values differ" % name)


def load_source_sample(sample_dir, expected_entry=None, verify_hashes=True):
    """Load and fully validate one frozen candidate sample."""

    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name
    marker = _json(sample_dir / "_SUCCESS.json")
    metadata = _json(sample_dir / "metadata.json")
    records, candidate_metadata = read_candidate_payload(sample_dir / "candidates.json")
    if marker.get("sample_id") != sample_id or metadata.get("sample_id") != sample_id:
        raise ScoringValidationError("source sample_id mismatch for %s" % sample_id)
    source_status = marker.get("status")
    if source_status not in ("success_nonempty", "success_empty"):
        raise ScoringValidationError("source status is not a valid success")
    if metadata.get("representation") != "planar_parallel_jaw_4dof":
        raise ScoringValidationError("source representation mismatch")
    if metadata.get("approach_constraint") != "fixed_camera_optical_axis":
        raise ScoringValidationError("source approach constraint mismatch")
    if not metadata.get("camera_frame"):
        raise ScoringValidationError("source camera frame missing")
    if candidate_metadata and candidate_metadata.get("sample_id") != sample_id:
        raise ScoringValidationError("candidate metadata sample_id mismatch")
    with np.load(str(sample_dir / "candidates.npz"), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    missing = sorted(set(POSE_ARRAY_KEYS + ("valid",)) - set(arrays))
    if missing:
        raise ScoringValidationError("source NPZ missing %s" % missing)
    count = int(arrays["center_uv"].shape[0])
    expected_count = int(metadata.get("counts", {}).get("post_nms", -1))
    marker_count = int(marker.get("candidate_counts", {}).get("post_nms", -2))
    if count != len(records) or count != expected_count or count != marker_count:
        raise ScoringValidationError("source candidate counts disagree")
    if (source_status == "success_empty") != (count == 0):
        raise ScoringValidationError("source empty status/count disagree")
    if count == 0 and not (metadata.get("failure_reason") or marker.get("failure_reason")):
        raise ScoringValidationError("valid empty source has no empty reason")
    ids = [record.get("candidate_id") for record in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ScoringValidationError("source candidate ID missing")
    if len(ids) != len(set(ids)):
        raise ScoringValidationError("source candidate IDs are not unique")
    for name, value in arrays.items():
        if value.ndim and value.shape[0] != count:
            raise ScoringValidationError("source NPZ %s length mismatch" % name)
    for name in POSE_ARRAY_KEYS:
        if not np.all(np.isfinite(arrays[name])):
            raise ScoringValidationError("source pose %s contains non-finite values" % name)
    if count and not np.all(arrays["valid"]):
        raise ScoringValidationError("source contains invalid retained candidates")
    json_checks = {
        "center_uv": np.asarray(
            [[item["center_u_px"], item["center_v_px"]] for item in records],
            dtype=arrays["center_uv"].dtype,
        ),
        "center_depth_m": np.asarray(
            [item["center_depth_m"] for item in records],
            dtype=arrays["center_depth_m"].dtype,
        ),
        "center_camera_xyz_m": np.asarray(
            [item["center_camera_xyz_m"] for item in records],
            dtype=arrays["center_camera_xyz_m"].dtype,
        ),
        "angle_rad": np.asarray(
            [item["angle_rad"] for item in records], dtype=arrays["angle_rad"].dtype
        ),
        "width_m": np.asarray(
            [item["width_m"] for item in records], dtype=arrays["width_m"].dtype
        ),
        "width_px": np.asarray(
            [item["width_px"] for item in records], dtype=arrays["width_px"].dtype
        ),
        "endpoints_uv": np.asarray(
            [[item["endpoint_1_uv"], item["endpoint_2_uv"]] for item in records],
            dtype=arrays["endpoints_uv"].dtype,
        ),
        "T_camera_grasp_fixed_approach": np.asarray(
            [item["T_camera_grasp_fixed_approach"] for item in records],
            dtype=arrays["T_camera_grasp_fixed_approach"].dtype,
        ),
    }
    # Comprehensions over zero records naturally produce shape ``(0,)``;
    # restore the frozen multidimensional empty shape before comparison.
    for name in json_checks:
        json_checks[name] = json_checks[name].reshape(arrays[name].shape)
    for name, expected in json_checks.items():
        _assert_array_equal(name, arrays[name], expected)
    hashes = {
        "candidates_npz_sha256": sha256_file(sample_dir / "candidates.npz"),
        "candidates_json_sha256": sha256_file(sample_dir / "candidates.json"),
        "metadata_json_sha256": sha256_file(sample_dir / "metadata.json"),
        "camera_intrinsics_sha256": sha256_file(sample_dir / "camera.intr"),
        "depth_m_sha256": sha256_file(sample_dir / "depth_m.npy"),
        "processed_mask_sha256": sha256_file(
            sample_dir / "hifics_mask_processed.png"
        ),
        "completion_marker_sha256": sha256_file(sample_dir / "_SUCCESS.json"),
    }
    if verify_hashes:
        required_hashes = marker.get("required_file_hashes", {})
        for name, key in (
            ("candidates.npz", "candidates_npz_sha256"),
            ("candidates.json", "candidates_json_sha256"),
            ("metadata.json", "metadata_json_sha256"),
            ("camera.intr", "camera_intrinsics_sha256"),
            ("depth_m.npy", "depth_m_sha256"),
            ("hifics_mask_processed.png", "processed_mask_sha256"),
        ):
            if required_hashes.get(name) != hashes[key]:
                raise ScoringValidationError("source marker hash mismatch: %s" % name)
    if expected_entry is not None:
        if int(expected_entry["candidate_count"]) != count:
            raise ScoringValidationError("source manifest count mismatch")
        if expected_entry["candidate_ids"] != ids:
            raise ScoringValidationError("source manifest candidate IDs mismatch")
        for key, value in hashes.items():
            if expected_entry["source_hashes"].get(key) != value:
                raise ScoringValidationError("source manifest hash mismatch: %s" % key)
    return {
        "sample_id": sample_id,
        "source_status": source_status,
        "empty_reason": metadata.get("failure_reason") or marker.get("failure_reason"),
        "candidate_count": count,
        "candidate_ids": ids,
        "records": records,
        "metadata": metadata,
        "marker": marker,
        "arrays": arrays,
        "source_hashes": hashes,
    }


def source_manifest_entry(source, sample_index):
    metadata = source["metadata"]
    marker = source["marker"]
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_index": int(sample_index),
        "sample_id": source["sample_id"],
        "question_index": int(metadata.get("question_index", -1)),
        "query": metadata.get("query", ""),
        "source_status": source["source_status"],
        "empty_reason": source["empty_reason"],
        "candidate_count": source["candidate_count"],
        "candidate_ids": source["candidate_ids"],
        "candidate_relative_path": "%s/candidates.npz" % source["sample_id"],
        "representation": metadata.get("representation"),
        "approach_constraint": metadata.get("approach_constraint"),
        "camera_frame": metadata.get("camera_frame"),
        "source_hashes": source["source_hashes"],
        "candidate_generation_configuration_hash": marker.get("configuration_hash"),
        "candidate_generation_config_file_sha256": marker.get("config_file_sha256"),
        "candidate_generation_seed": marker.get("seed"),
        "sampler_commit": marker.get("sampler_commit"),
    }


def build_source_manifest(
    candidate_root,
    manifest_path,
    expected_samples=7675,
    expected_nonempty=7620,
    expected_empty=55,
    expected_candidates=206538,
    verify_hashes=True,
):
    """Validate every frozen sample and atomically write its immutable manifest."""

    candidate_root = Path(candidate_root).resolve()
    summary_path = candidate_root / "summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    if len(summary_rows) != int(expected_samples):
        raise ScoringValidationError(
            "expected %s source samples, found %s"
            % (expected_samples, len(summary_rows))
        )
    ids = [row["sample_id"] for row in summary_rows]
    if len(ids) != len(set(ids)):
        raise ScoringValidationError("source summary has duplicate sample IDs")
    entries = []
    nonempty = 0
    empty = 0
    candidate_total = 0
    generation_hashes = set()
    config_file_hashes = set()
    for index, row in enumerate(summary_rows):
        source = load_source_sample(
            candidate_root / row["sample_id"], verify_hashes=verify_hashes
        )
        if int(row["post_nms_count"]) != source["candidate_count"]:
            raise ScoringValidationError("summary count mismatch for %s" % row["sample_id"])
        if row["status"] != source["source_status"]:
            raise ScoringValidationError("summary status mismatch for %s" % row["sample_id"])
        entry = source_manifest_entry(source, index)
        entries.append(entry)
        candidate_total += source["candidate_count"]
        nonempty += int(source["candidate_count"] > 0)
        empty += int(source["candidate_count"] == 0)
        generation_hashes.add(entry["candidate_generation_configuration_hash"])
        config_file_hashes.add(entry["candidate_generation_config_file_sha256"])
    expected = (
        int(expected_nonempty),
        int(expected_empty),
        int(expected_candidates),
    )
    actual = (nonempty, empty, candidate_total)
    if actual != expected:
        raise ScoringValidationError(
            "frozen source gate failed: expected %s, found %s" % (expected, actual)
        )
    if len(generation_hashes) != 1 or len(config_file_hashes) != 1:
        raise ScoringValidationError("candidate-generation identity is not uniform")
    atomic_write_jsonl(manifest_path, entries)
    return entries, {
        "samples": len(entries),
        "nonempty_samples": nonempty,
        "empty_samples": empty,
        "candidate_count": candidate_total,
        "source_manifest_sha256": sha256_file(manifest_path),
        "candidate_generation_configuration_hash": next(iter(generation_hashes)),
        "candidate_generation_config_file_sha256": next(iter(config_file_hashes)),
    }


def load_source_manifest(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ScoringValidationError("source manifest has duplicate sample IDs")
    return rows


def select_entries(
    entries,
    sample_ids=None,
    sample_limit=None,
    start_index=None,
    end_index=None,
    num_shards=None,
    shard_index=None,
):
    if sample_ids:
        if any(value is not None for value in (start_index, end_index, num_shards, shard_index)):
            raise ValueError("--sample-id cannot be combined with range or shards")
        by_id = {row["sample_id"]: row for row in entries}
        missing = [value for value in sample_ids if value not in by_id]
        if missing:
            raise KeyError("unknown sample IDs: %s" % missing)
        selected = [by_id[value] for value in sample_ids]
    else:
        total = len(entries)
        start = 0 if start_index is None else int(start_index)
        end = total if end_index is None else int(end_index)
        if start < 0 or end < start or end > total:
            raise ValueError("invalid half-open range [%s, %s)" % (start, end))
        if (num_shards is None) != (shard_index is None):
            raise ValueError("--num-shards and --shard-index are required together")
        shards = 1 if num_shards is None else int(num_shards)
        shard = 0 if shard_index is None else int(shard_index)
        if shards <= 0 or shard < 0 or shard >= shards:
            raise ValueError("invalid shard selection")
        selected = [row for index, row in enumerate(entries[start:end], start) if index % shards == shard]
    if sample_limit is not None:
        if int(sample_limit) <= 0:
            raise ValueError("sample limit must be positive")
        selected = selected[: int(sample_limit)]
    return selected


def model_file_manifest(model_dir):
    model_dir = Path(model_dir).resolve()
    files = []
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(model_dir).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not files or not (model_dir / "config.json").is_file():
        raise FileNotFoundError("GQ-CNN model is incomplete: %s" % model_dir)
    return files, canonical_json_hash(files)


def make_staging_directory(output_root, sample_id):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    staging = root / (".%s.tmp.%s.%s" % (sample_id, os.getpid(), uuid.uuid4().hex))
    staging.mkdir()
    return staging


def remove_staging(staging, output_root):
    staging = Path(staging).resolve()
    root = Path(output_root).resolve()
    if staging.parent != root or not staging.name.startswith(".") or ".tmp." not in staging.name:
        raise ValueError("unsafe staging path: %s" % staging)
    if staging.exists():
        shutil.rmtree(str(staging))


def cleanup_stale_staging(output_root, sample_ids):
    """Remove interrupted per-sample staging directories for this selection."""

    root = Path(output_root).resolve()
    if not root.is_dir():
        return []
    selected = set(sample_ids)
    removed = []
    for path in root.iterdir():
        name = path.name
        if not name.startswith(".") or ".tmp." not in name:
            continue
        sample_id = name[1:].split(".tmp.", 1)[0]
        if sample_id not in selected or not path.is_dir() or path.is_symlink():
            continue
        remove_staging(path, root)
        removed.append(name)
    return sorted(removed)


def quarantine_output(path, output_root, label="corrupt"):
    path = Path(path).resolve()
    root = Path(output_root).resolve()
    if path.parent != root or path.name.startswith("."):
        raise ValueError("unsafe output path: %s" % path)
    destination = root / (
        ".%s.%s.%s.%s" % (path.name, label, int(time.time()), uuid.uuid4().hex)
    )
    os.replace(str(path), str(destination))
    _fsync_directory(root)
    return destination


def atomic_commit_sample(staging, final, output_root):
    staging = Path(staging).resolve()
    final = Path(final).resolve()
    root = Path(output_root).resolve()
    if staging.parent != root or final.parent != root:
        raise ValueError("sample paths must be direct output-root children")
    if not staging.name.startswith(".%s.tmp." % final.name):
        raise ValueError("unexpected staging directory")
    backup = None
    if final.exists():
        backup = root / (".%s.backup.%s.%s" % (final.name, os.getpid(), uuid.uuid4().hex))
        os.replace(str(final), str(backup))
    try:
        os.replace(str(staging), str(final))
        _fsync_directory(root)
    except BaseException:
        if backup is not None and backup.exists() and not final.exists():
            os.replace(str(backup), str(final))
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(str(backup))


def _file_hashes(directory, names):
    result = {}
    for name in names:
        path = Path(directory) / name
        if not path.is_file() or path.is_symlink():
            raise ScoringValidationError("required output missing: %s" % path)
        result[name] = sha256_file(path)
    return result


def write_marker(directory, metadata, required_files):
    marker = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": metadata["sample_id"],
        "scoring_status": metadata["scoring_status"],
        "source_candidate_count": metadata["source_candidate_count"],
        "gqcnn_scored_count": metadata["gqcnn_scored_count"],
        "source_candidate_sha256": metadata["source_candidate_sha256"],
        "source_candidate_json_sha256": metadata["source_candidate_json_sha256"],
        "source_depth_m_sha256": metadata["source_depth_m_sha256"],
        "source_processed_mask_sha256": metadata["source_processed_mask_sha256"],
        "model_name": metadata["model_name"],
        "model_commit": metadata["model_commit"],
        "model_config_hash": metadata["model_config_hash"],
        "model_file_manifest_hash": metadata["model_file_manifest_hash"],
        "seed": metadata["seed"],
        "top1_candidate_id": metadata.get("top1_candidate_id"),
        "failure_reason": metadata.get("failure_reason"),
        "required_files": list(required_files),
        "required_file_hashes": _file_hashes(directory, required_files),
        "completion_timestamp": utc_timestamp(),
        "summary_row": metadata["summary_row"],
    }
    atomic_write_json(Path(directory) / MARKER_NAME, marker)
    return marker


def base_metadata(entry, model_info, seed):
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": entry["sample_id"],
        "query": entry.get("query", ""),
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "camera_frame": entry.get("camera_frame"),
        "source_candidate_count": int(entry["candidate_count"]),
        "source_candidate_sha256": entry["source_hashes"]["candidates_npz_sha256"],
        "source_candidate_json_sha256": entry["source_hashes"]["candidates_json_sha256"],
        "source_metadata_sha256": entry["source_hashes"]["metadata_json_sha256"],
        "source_camera_intrinsics_sha256": entry["source_hashes"]["camera_intrinsics_sha256"],
        "source_depth_m_sha256": entry["source_hashes"]["depth_m_sha256"],
        "source_processed_mask_sha256": entry["source_hashes"]["processed_mask_sha256"],
        "model_name": model_info["model_name"],
        "model_commit": model_info["model_commit"],
        "model_config_hash": model_info["model_config_hash"],
        "model_file_manifest_hash": model_info["model_file_manifest_hash"],
        "docker_image": model_info.get("docker_image"),
        "docker_image_id": model_info.get("docker_image_id"),
        "seed": int(seed),
        "ranking": {
            "primary_key": "raw full-precision gqcnn_q_value descending",
            "tie_breaker": "candidate_id ascending for exact ties",
            "rank_indexing": "one_based",
            "q_value_rounded_before_sort": False,
        },
        "candidate_source": "frozen post-NMS candidates; no sampling, CEM, or pose alteration",
        "q_value_is_calibrated_ocid_vlg_robot_success_probability": False,
    }


def empty_summary(entry, model_info, status, failure_reason=None):
    return {
        "sample_id": entry["sample_id"],
        "query": entry.get("query", ""),
        "source_candidate_count": int(entry["candidate_count"]),
        "scored_candidate_count": 0,
        "scoring_status": status,
        "top1_candidate_id": None,
        "top1_q_value": None,
        "top5_candidate_ids": [],
        "max_q_value": None,
        "min_q_value": None,
        "mean_q_value": None,
        "median_q_value": None,
        "std_q_value": None,
        "q_value_range": None,
        "scoring_time_ms": 0.0,
        "preprocessing_time_ms": 0.0,
        "serialization_time_ms": 0.0,
        "total_time_ms": 0.0,
        "model_name": model_info["model_name"],
        "model_config_hash": model_info["model_config_hash"],
        "source_candidate_sha256": entry["source_hashes"]["candidates_npz_sha256"],
        "failure_reason": failure_reason,
    }


def write_empty_sample(staging, entry, model_info, seed):
    metadata = base_metadata(entry, model_info, seed)
    metadata.update(
        {
            "scoring_status": SKIPPED_VALID_EMPTY,
            "gqcnn_scored_count": 0,
            "top1_candidate_id": None,
            "empty_reason": entry.get("empty_reason"),
            "failure_reason": None,
        }
    )
    metadata["summary_row"] = empty_summary(
        entry, model_info, SKIPPED_VALID_EMPTY
    )
    atomic_write_json(Path(staging) / "scoring_metadata.json", metadata)
    return write_marker(staging, metadata, EMPTY_FILES)


def write_failed_sample(staging, entry, model_info, seed, error, stage):
    failure_reason = "%s: %s" % (type(error).__name__, error)
    metadata = base_metadata(entry, model_info, seed)
    metadata.update(
        {
            "scoring_status": SCORING_FAILED,
            "gqcnn_scored_count": 0,
            "top1_candidate_id": None,
            "failure_reason": failure_reason,
            "failure_stage": stage,
        }
    )
    metadata["summary_row"] = empty_summary(
        entry, model_info, SCORING_FAILED, failure_reason=failure_reason
    )
    failure = {
        "sample_id": entry["sample_id"],
        "scoring_status": SCORING_FAILED,
        "failure_stage": stage,
        "failure_type": type(error).__name__,
        "failure_reason": failure_reason,
        "traceback": traceback.format_exc(),
        "timestamp": utc_timestamp(),
    }
    atomic_write_json(Path(staging) / "failure.json", failure)
    atomic_write_json(Path(staging) / "scoring_metadata.json", metadata)
    return write_marker(staging, metadata, FAILED_FILES)


def _rank_indices(q_values, candidate_ids):
    if q_values.shape != (len(candidate_ids),) or not np.all(np.isfinite(q_values)):
        raise ScoringValidationError("q-values must be a finite N-vector")
    return sorted(
        range(len(candidate_ids)),
        key=lambda index: (-float(q_values[index]), candidate_ids[index]),
    )


def _candidate_record(source, source_index, q_value, rank, entry, model_info):
    record = dict(source)
    center = record.get("center_camera_xyz_m")
    record.update(
        {
            "source_candidate_index": int(source_index),
            "centre_u_px": float(record["center_u_px"]),
            "centre_v_px": float(record["center_v_px"]),
            "centre_depth_m": float(record["center_depth_m"]),
            "centre_camera_xyz_m": center,
            "gqcnn_q_value": float(q_value),
            "gqcnn_rank": int(rank),
            "source_candidate_sha256": entry["source_hashes"]["candidates_npz_sha256"],
            "camera_frame": entry.get("camera_frame"),
            "model_name": model_info["model_name"],
            "model_commit": model_info["model_commit"],
            "model_config_hash": model_info["model_config_hash"],
            "scoring_status": SCORED_NONEMPTY,
            "representation": "planar_parallel_jaw_4dof",
        }
    )
    return strict_value(record)


def _top_payload(records, limit, entry, model_info):
    selected = records[: min(int(limit), len(records))]
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": entry["sample_id"],
        "query": entry.get("query", ""),
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "camera_frame": entry.get("camera_frame"),
        "model_name": model_info["model_name"],
        "model_commit": model_info["model_commit"],
        "model_config_hash": model_info["model_config_hash"],
        "ranking": "raw q descending; exact ties by candidate_id ascending",
        "q_value_is_calibrated_ocid_vlg_robot_success_probability": False,
        "candidates": selected,
    }


def score_and_write_sample(
    staging,
    source_dir,
    entry,
    quality_fn,
    make_state_and_grasps,
    model_info,
    seed,
    inpaint_rescale_factor=0.5,
):
    """Score one non-empty source and serialize a validated atomic payload."""

    overall_start = time.perf_counter()
    source = load_source_sample(source_dir, expected_entry=entry, verify_hashes=True)
    if source["candidate_count"] <= 0:
        raise ScoringValidationError("non-empty scoring called for empty source")
    preprocess_start = time.perf_counter()
    state, grasps, intrinsics = make_state_and_grasps(
        Path(source_dir),
        source["arrays"],
        source["records"],
        float(inpaint_rescale_factor),
    )
    preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000.0
    scoring_start = time.perf_counter()
    q_values = np.asarray(
        quality_fn(
            state,
            grasps,
            params={"vis": {"tf_images": False, "k": min(25, len(grasps))}},
        ),
        dtype=np.float64,
    )
    scoring_ms = (time.perf_counter() - scoring_start) * 1000.0
    order = _rank_indices(q_values, source["candidate_ids"])
    rank_by_index = np.empty(len(order), dtype=np.int32)
    ranked_records = []
    for rank, source_index in enumerate(order, start=1):
        rank_by_index[source_index] = rank
        ranked_records.append(
            _candidate_record(
                source["records"][source_index],
                source_index,
                q_values[source_index],
                rank,
                entry,
                model_info,
            )
        )
    serialization_start = time.perf_counter()
    npz_arrays = {name: value.copy() for name, value in source["arrays"].items()}
    npz_arrays["candidate_id"] = np.asarray(source["candidate_ids"], dtype="<U128")
    npz_arrays["gqcnn_q_value"] = q_values.astype(np.float64, copy=True)
    npz_arrays["gqcnn_rank"] = rank_by_index
    deterministic_save_npz(Path(staging) / "gqcnn_scored_candidates.npz", npz_arrays)
    payload_metadata = base_metadata(entry, model_info, seed)
    payload_metadata.update(
        {
            "scoring_status": SCORED_NONEMPTY,
            "gqcnn_scored_count": len(ranked_records),
            "storage_order": "JSON/CSV ranked; NPZ source candidate order",
            "preprocessing": "official full-depth inpaint then GQCnnQualityFunction.grasps_to_tensors",
            "camera_intrinsics": intrinsics,
        }
    )
    atomic_write_json(
        Path(staging) / "gqcnn_scored_candidates.json",
        {"metadata": payload_metadata, "candidates": ranked_records},
    )
    fields = []
    preferred = [
        "gqcnn_rank",
        "candidate_id",
        "source_candidate_index",
        "sample_id",
        "query",
        "gqcnn_q_value",
        "centre_u_px",
        "centre_v_px",
        "centre_depth_m",
        "angle_rad",
        "angle_deg",
        "width_m",
        "width_px",
    ]
    all_fields = set()
    for record in ranked_records:
        all_fields.update(record)
    fields = preferred + sorted(all_fields - set(preferred))
    atomic_write_csv(
        Path(staging) / "gqcnn_scored_candidates.csv", ranked_records, fields
    )
    atomic_write_json(
        Path(staging) / "gqcnn_top1.json",
        _top_payload(ranked_records, 1, entry, model_info),
    )
    atomic_write_json(
        Path(staging) / "gqcnn_top5.json",
        _top_payload(ranked_records, 5, entry, model_info),
    )
    serialization_ms = (time.perf_counter() - serialization_start) * 1000.0
    top1 = ranked_records[0]
    q64 = q_values.astype(np.float64)
    summary = {
        "sample_id": entry["sample_id"],
        "query": entry.get("query", ""),
        "source_candidate_count": len(ranked_records),
        "scored_candidate_count": len(ranked_records),
        "scoring_status": SCORED_NONEMPTY,
        "top1_candidate_id": top1["candidate_id"],
        "top1_q_value": float(top1["gqcnn_q_value"]),
        "top5_candidate_ids": [item["candidate_id"] for item in ranked_records[:5]],
        "max_q_value": float(np.max(q64)),
        "min_q_value": float(np.min(q64)),
        "mean_q_value": float(np.mean(q64)),
        "median_q_value": float(np.median(q64)),
        "std_q_value": float(np.std(q64)),
        "q_value_range": float(np.max(q64) - np.min(q64)),
        "scoring_time_ms": scoring_ms,
        "preprocessing_time_ms": preprocessing_ms,
        "serialization_time_ms": serialization_ms,
        "total_time_ms": (time.perf_counter() - overall_start) * 1000.0,
        "model_name": model_info["model_name"],
        "model_config_hash": model_info["model_config_hash"],
        "source_candidate_sha256": entry["source_hashes"]["candidates_npz_sha256"],
        "failure_reason": None,
    }
    scoring_metadata = dict(payload_metadata)
    scoring_metadata.update(
        {
            "top1_candidate_id": top1["candidate_id"],
            "failure_reason": None,
            "timing_ms": {
                "preprocessing": preprocessing_ms,
                "model_inference": scoring_ms,
                "serialization": serialization_ms,
                "total": summary["total_time_ms"],
            },
            "summary_row": summary,
        }
    )
    atomic_write_json(Path(staging) / "scoring_metadata.json", scoring_metadata)
    marker = write_marker(staging, scoring_metadata, NONEMPTY_FILES)
    after_hash = sha256_file(Path(source_dir) / "candidates.npz")
    if after_hash != entry["source_hashes"]["candidates_npz_sha256"]:
        raise ScoringValidationError("source candidates.npz changed during scoring")
    return marker


def _metadata_identity_errors(metadata, status, entry, model_info, seed):
    expected = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": entry["sample_id"],
        "scoring_status": status,
        "source_candidate_count": int(entry["candidate_count"]),
        "source_candidate_sha256": entry["source_hashes"]["candidates_npz_sha256"],
        "source_candidate_json_sha256": entry["source_hashes"]["candidates_json_sha256"],
        "source_metadata_sha256": entry["source_hashes"]["metadata_json_sha256"],
        "source_camera_intrinsics_sha256": entry["source_hashes"]["camera_intrinsics_sha256"],
        "source_depth_m_sha256": entry["source_hashes"]["depth_m_sha256"],
        "source_processed_mask_sha256": entry["source_hashes"]["processed_mask_sha256"],
        "model_name": model_info["model_name"],
        "model_commit": model_info["model_commit"],
        "model_config_hash": model_info["model_config_hash"],
        "model_file_manifest_hash": model_info["model_file_manifest_hash"],
        "seed": int(seed),
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "camera_frame": entry.get("camera_frame"),
    }
    errors = []
    for name, value in expected.items():
        if metadata.get(name) != value:
            errors.append("scoring metadata %s mismatch" % name)
    expected_count = int(entry["candidate_count"]) if status == SCORED_NONEMPTY else 0
    if int(metadata.get("gqcnn_scored_count", -1)) != expected_count:
        errors.append("scoring metadata gqcnn_scored_count mismatch")
    return errors


def _top_identity_errors(payload, records, limit, entry, model_info):
    expected = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": entry["sample_id"],
        "query": entry.get("query", ""),
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "camera_frame": entry.get("camera_frame"),
        "model_name": model_info["model_name"],
        "model_commit": model_info["model_commit"],
        "model_config_hash": model_info["model_config_hash"],
    }
    errors = []
    for name, value in expected.items():
        if payload.get(name) != value:
            errors.append("Top-%s %s mismatch" % (limit, name))
    if payload.get("candidates") != records[: min(limit, len(records))]:
        errors.append("Top-%s candidate payload mismatch" % limit)
    return errors


def validate_scored_output(sample_dir, source_dir, entry, model_info, seed, verify_hashes=True):
    """Return ``(valid, status, marker, errors)`` for one committed output."""

    sample_dir = Path(sample_dir)
    errors = []
    marker = None
    if not sample_dir.is_dir() or sample_dir.is_symlink():
        return False, None, None, ["sample directory missing or unsafe"]
    try:
        marker = _json(sample_dir / MARKER_NAME)
        status = marker.get("scoring_status")
        if status not in TERMINAL_STATUSES:
            errors.append("invalid scoring status")
        checks = {
            "sample_id": entry["sample_id"],
            "source_candidate_count": int(entry["candidate_count"]),
            "source_candidate_sha256": entry["source_hashes"]["candidates_npz_sha256"],
            "source_candidate_json_sha256": entry["source_hashes"]["candidates_json_sha256"],
            "source_depth_m_sha256": entry["source_hashes"]["depth_m_sha256"],
            "source_processed_mask_sha256": entry["source_hashes"]["processed_mask_sha256"],
            "model_name": model_info["model_name"],
            "model_commit": model_info["model_commit"],
            "model_config_hash": model_info["model_config_hash"],
            "model_file_manifest_hash": model_info["model_file_manifest_hash"],
            "seed": int(seed),
        }
        for name, expected in checks.items():
            if marker.get(name) != expected:
                errors.append("marker %s mismatch" % name)
        required = marker.get("required_files", [])
        hashes = marker.get("required_file_hashes", {})
        expected_required = {
            SCORED_NONEMPTY: set(NONEMPTY_FILES),
            SKIPPED_VALID_EMPTY: set(EMPTY_FILES),
            SCORING_FAILED: set(FAILED_FILES),
        }.get(status, set())
        if set(required) != expected_required or set(hashes) != expected_required:
            errors.append("required-file manifest mismatch")
        for name in required:
            path = sample_dir / name
            if not path.is_file() or path.is_symlink():
                errors.append("required file missing: %s" % name)
            elif verify_hashes and sha256_file(path) != hashes.get(name):
                errors.append("required file hash mismatch: %s" % name)
        scoring_metadata = _json(sample_dir / "scoring_metadata.json")
        errors.extend(
            _metadata_identity_errors(
                scoring_metadata, status, entry, model_info, seed
            )
        )
        if scoring_metadata.get("summary_row") != marker.get("summary_row"):
            errors.append("marker/scoring metadata summary mismatch")
        if status == SKIPPED_VALID_EMPTY:
            if int(entry["candidate_count"]) != 0 or int(marker.get("gqcnn_scored_count", -1)) != 0:
                errors.append("empty status/count mismatch")
            if scoring_metadata.get("top1_candidate_id") is not None:
                errors.append("empty output has a Top-1")
        elif status == SCORING_FAILED:
            if not marker.get("failure_reason"):
                errors.append("failed output has no reason")
        elif status == SCORED_NONEMPTY:
            source = load_source_sample(source_dir, expected_entry=entry, verify_hashes=True)
            records, metadata = read_candidate_payload(sample_dir / "gqcnn_scored_candidates.json")
            errors.extend(
                _metadata_identity_errors(metadata, status, entry, model_info, seed)
            )
            count = int(entry["candidate_count"])
            if len(records) != count or int(marker.get("gqcnn_scored_count", -1)) != count:
                errors.append("scored JSON/count mismatch")
            with np.load(str(sample_dir / "gqcnn_scored_candidates.npz"), allow_pickle=False) as archive:
                scored = {name: np.asarray(archive[name]) for name in archive.files}
            expected_keys = set(source["arrays"]) | {"candidate_id", "gqcnn_rank"}
            if set(scored) != expected_keys:
                errors.append("scored NPZ keys mismatch")
            ids = [str(value) for value in scored.get("candidate_id", np.asarray([])).tolist()]
            if ids != source["candidate_ids"]:
                errors.append("scored NPZ candidate IDs/order mismatch")
            for name, value in source["arrays"].items():
                if name == "gqcnn_q_value" or name not in scored:
                    continue
                try:
                    _assert_array_equal(name, scored[name], value)
                except ScoringValidationError as error:
                    errors.append(str(error))
            q_values = np.asarray(scored.get("gqcnn_q_value", []), dtype=np.float64)
            ranks = np.asarray(scored.get("gqcnn_rank", []))
            if q_values.shape != (count,) or not np.all(np.isfinite(q_values)):
                errors.append("scored q-values invalid")
            if ranks.shape != (count,) or set(ranks.astype(int).tolist()) != set(range(1, count + 1)):
                errors.append("scored ranks are not a one-based permutation")
            if not errors:
                order = _rank_indices(q_values, ids)
                expected_ranks = np.empty(count, dtype=np.int32)
                for rank, index in enumerate(order, start=1):
                    expected_ranks[index] = rank
                if not np.array_equal(ranks.astype(np.int32), expected_ranks):
                    errors.append("scored ranks do not match raw q ordering")
                json_ids = [record.get("candidate_id") for record in records]
                if json_ids != [ids[index] for index in order]:
                    errors.append("ranked JSON order mismatch")
                for rank, record in enumerate(records, start=1):
                    source_index = int(record.get("source_candidate_index", -1))
                    if source_index != order[rank - 1]:
                        errors.append("JSON source_candidate_index mismatch")
                        break
                    if int(record.get("gqcnn_rank", -1)) != rank:
                        errors.append("JSON rank mismatch")
                        break
                    if float(record.get("gqcnn_q_value")) != float(q_values[source_index]):
                        errors.append("JSON q-value mismatch")
                        break
                    overwritten_fields = {
                        "gqcnn_q_value",
                        "gqcnn_rank",
                        "source_candidate_index",
                        "source_candidate_sha256",
                        "camera_frame",
                        "model_name",
                        "model_commit",
                        "model_config_hash",
                        "scoring_status",
                        "representation",
                    }
                    for name, value in source["records"][source_index].items():
                        if name in overwritten_fields:
                            continue
                        if record.get(name) != value:
                            errors.append("JSON source field mismatch: %s" % name)
                            break
                    identity = {
                        "source_candidate_sha256": entry["source_hashes"]["candidates_npz_sha256"],
                        "camera_frame": entry.get("camera_frame"),
                        "model_name": model_info["model_name"],
                        "model_commit": model_info["model_commit"],
                        "model_config_hash": model_info["model_config_hash"],
                        "scoring_status": SCORED_NONEMPTY,
                        "representation": "planar_parallel_jaw_4dof",
                    }
                    for name, value in identity.items():
                        if record.get(name) != value:
                            errors.append("JSON candidate %s mismatch" % name)
                            break
                top1 = _json(sample_dir / "gqcnn_top1.json")
                top5 = _json(sample_dir / "gqcnn_top5.json")
                errors.extend(_top_identity_errors(top1, records, 1, entry, model_info))
                errors.extend(_top_identity_errors(top5, records, 5, entry, model_info))
                with (sample_dir / "gqcnn_scored_candidates.csv").open(
                    encoding="utf-8", newline=""
                ) as stream:
                    reader = csv.DictReader(stream)
                    csv_rows = list(reader)
                    csv_fields = set(reader.fieldnames or [])
                required_csv = {
                    "candidate_id",
                    "source_candidate_index",
                    "gqcnn_rank",
                    "gqcnn_q_value",
                    "center_u_px",
                    "center_v_px",
                    "center_depth_m",
                    "angle_rad",
                    "width_m",
                    "width_px",
                    "model_name",
                    "model_commit",
                    "model_config_hash",
                }
                if not required_csv.issubset(csv_fields):
                    errors.append("CSV required fields missing")
                csv_ids = [row.get("candidate_id") for row in csv_rows]
                if csv_ids != json_ids:
                    errors.append("CSV ranked order mismatch")
                if len(csv_rows) == len(records):
                    for csv_row, record in zip(csv_rows, records):
                        try:
                            matches = (
                                int(csv_row["source_candidate_index"])
                                == int(record["source_candidate_index"])
                                and int(csv_row["gqcnn_rank"]) == int(record["gqcnn_rank"])
                                and float(csv_row["gqcnn_q_value"])
                                == float(record["gqcnn_q_value"])
                                and csv_row["model_name"] == record["model_name"]
                                and csv_row["model_commit"] == record["model_commit"]
                                and csv_row["model_config_hash"]
                                == record["model_config_hash"]
                            )
                        except (KeyError, TypeError, ValueError):
                            matches = False
                        if not matches:
                            errors.append("CSV candidate values mismatch")
                            break
            if metadata.get("sample_id") != entry["sample_id"]:
                errors.append("scored JSON metadata mismatch")
    except Exception as error:
        errors.append("%s: %s" % (type(error).__name__, error))
        status = marker.get("scoring_status") if isinstance(marker, dict) else None
    return not errors, status, marker, errors


def collect_committed(output_root, entries, candidate_root, model_info, seed, verify_hashes=False):
    rows = []
    failures = []
    manifests = []
    by_id = {entry["sample_id"]: entry for entry in entries}
    for sample_id in [entry["sample_id"] for entry in entries]:
        entry = by_id[sample_id]
        directory = Path(output_root) / sample_id
        if not directory.exists():
            continue
        valid, status, marker, errors = validate_scored_output(
            directory,
            Path(candidate_root) / sample_id,
            entry,
            model_info,
            seed,
            verify_hashes=verify_hashes,
        )
        if not valid:
            failures.append(
                {"sample_id": sample_id, "scoring_status": "corrupt", "errors": errors}
            )
            continue
        row = dict(marker["summary_row"])
        rows.append(row)
        manifests.append(
            {
                "sample_id": sample_id,
                "scoring_status": status,
                "source_candidate_count": marker["source_candidate_count"],
                "gqcnn_scored_count": marker["gqcnn_scored_count"],
                "top1_candidate_id": marker.get("top1_candidate_id"),
                "source_candidate_sha256": marker["source_candidate_sha256"],
                "model_config_hash": marker["model_config_hash"],
                "completion_timestamp": marker["completion_timestamp"],
            }
        )
        if status == SCORING_FAILED:
            failures.append(_json(directory / "failure.json"))
    return rows, manifests, failures


def collect_marker_rows(output_root, entries):
    """Read already-validated completion markers for cheap live checkpoints."""

    rows = []
    manifests = []
    failures = []
    for entry in entries:
        directory = Path(output_root) / entry["sample_id"]
        marker_path = directory / MARKER_NAME
        if not marker_path.is_file():
            continue
        try:
            marker = _json(marker_path)
            status = marker.get("scoring_status")
            if status not in TERMINAL_STATUSES or marker.get("sample_id") != entry["sample_id"]:
                raise ScoringValidationError("completion marker identity/status mismatch")
            rows.append(dict(marker["summary_row"]))
            manifests.append(
                {
                    "sample_id": entry["sample_id"],
                    "scoring_status": status,
                    "source_candidate_count": marker["source_candidate_count"],
                    "gqcnn_scored_count": marker["gqcnn_scored_count"],
                    "top1_candidate_id": marker.get("top1_candidate_id"),
                    "source_candidate_sha256": marker["source_candidate_sha256"],
                    "model_config_hash": marker["model_config_hash"],
                    "completion_timestamp": marker["completion_timestamp"],
                }
            )
            if status == SCORING_FAILED:
                failures.append(_json(directory / "failure.json"))
        except Exception as error:
            failures.append(
                {
                    "sample_id": entry["sample_id"],
                    "scoring_status": "corrupt",
                    "errors": ["%s: %s" % (type(error).__name__, error)],
                }
            )
    return rows, manifests, failures


def write_root_state(
    output_root,
    entries,
    candidate_root,
    model_info,
    seed,
    start_time,
    last_completed_sample_id=None,
    verify_hashes=False,
):
    output_root = Path(output_root)
    del candidate_root, model_info, seed, verify_hashes
    rows, manifests, failures = collect_marker_rows(output_root, entries)
    index_by_id = {entry["sample_id"]: entry["sample_index"] for entry in entries}
    rows.sort(key=lambda row: index_by_id[row["sample_id"]])
    atomic_write_csv(output_root / "summary.csv", rows, SUMMARY_FIELDS)
    atomic_write_jsonl(output_root / "scoring_manifest.jsonl", manifests)
    atomic_write_jsonl(output_root / "failures.jsonl", failures)
    elapsed = max(0.0, time.time() - float(start_time))
    completed_nonempty = sum(row["scoring_status"] == SCORED_NONEMPTY for row in rows)
    skipped_empty = sum(row["scoring_status"] == SKIPPED_VALID_EMPTY for row in rows)
    failed = sum(row["scoring_status"] == SCORING_FAILED for row in rows)
    scored = sum(int(row["scored_candidate_count"]) for row in rows)
    expected_candidates = sum(int(entry["candidate_count"]) for entry in entries)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "total_samples": len(entries),
        "completed_nonempty_samples": completed_nonempty,
        "skipped_empty_samples": skipped_empty,
        "failed_samples": failed,
        "terminal_samples": len(rows),
        "scored_candidates": scored,
        "remaining_candidates": expected_candidates - scored,
        "elapsed_seconds": elapsed,
        "throughput_candidates_per_second": 0.0 if elapsed <= 0 else scored / elapsed,
        "last_completed_sample_id": last_completed_sample_id,
        "updated_at": utc_timestamp(),
    }
    atomic_write_json(output_root / "progress.json", progress)
    return progress


def write_run_statistics(
    output_root,
    entries,
    candidate_root,
    model_info,
    seed,
    start_time,
    nearly_identical_atol=1e-12,
    low_q_thresholds=(1e-6, 1e-4, 1e-2),
):
    rows, manifests, failures = collect_committed(
        output_root,
        entries,
        candidate_root,
        model_info,
        seed,
        verify_hashes=False,
    )
    q_chunks = []
    nearly_identical = []
    below = {str(value): [] for value in low_q_thresholds}
    for row in rows:
        if row["scoring_status"] != SCORED_NONEMPTY:
            continue
        sample_id = row["sample_id"]
        with np.load(
            str(Path(output_root) / sample_id / "gqcnn_scored_candidates.npz"),
            allow_pickle=False,
        ) as archive:
            values = np.asarray(archive["gqcnn_q_value"], dtype=np.float64)
        q_chunks.append(values)
        if values.size and float(np.max(values) - np.min(values)) <= float(nearly_identical_atol):
            nearly_identical.append(sample_id)
        for threshold in low_q_thresholds:
            if values.size and bool(np.all(values < float(threshold))):
                below[str(threshold)].append(sample_id)
    q_values = np.concatenate(q_chunks) if q_chunks else np.asarray([], dtype=np.float64)
    finite = q_values[np.isfinite(q_values)]
    percentiles = {}
    if finite.size:
        for label, value in (
            ("p01", 1), ("p05", 5), ("p25", 25), ("p50", 50),
            ("p75", 75), ("p95", 95), ("p99", 99),
        ):
            percentiles[label] = float(np.percentile(finite, value))
    model_time_ms = sum(float(row.get("scoring_time_ms") or 0.0) for row in rows)
    total_time_ms = sum(float(row.get("total_time_ms") or 0.0) for row in rows)
    elapsed = max(0.0, time.time() - float(start_time))
    statistics = {
        "schema_version": SCHEMA_VERSION,
        "total_samples": len(entries),
        "terminal_samples": len(rows),
        "scored_nonempty_samples": sum(row["scoring_status"] == SCORED_NONEMPTY for row in rows),
        "skipped_valid_empty_samples": sum(row["scoring_status"] == SKIPPED_VALID_EMPTY for row in rows),
        "failed_samples": sum(row["scoring_status"] == SCORING_FAILED for row in rows),
        "corrupt_committed_samples": sum(item.get("scoring_status") == "corrupt" for item in failures),
        "expected_candidates": sum(int(entry["candidate_count"]) for entry in entries),
        "scored_candidates": int(q_values.size),
        "finite_q_values": int(finite.size),
        "invalid_q_values": int(q_values.size - finite.size),
        "q_value_min": None if not finite.size else float(np.min(finite)),
        "q_value_max": None if not finite.size else float(np.max(finite)),
        "q_value_mean": None if not finite.size else float(np.mean(finite)),
        "q_value_median": None if not finite.size else float(np.median(finite)),
        "q_value_percentiles": percentiles,
        "nearly_identical_atol": float(nearly_identical_atol),
        "samples_all_q_nearly_identical": nearly_identical,
        "samples_all_q_below_threshold": below,
        "model_inference_time_seconds": model_time_ms / 1000.0,
        "summed_sample_total_time_seconds": total_time_ms / 1000.0,
        "wall_elapsed_seconds": elapsed,
        "candidates_per_model_second": 0.0 if model_time_ms <= 0 else q_values.size / (model_time_ms / 1000.0),
        "candidates_per_wall_second": 0.0 if elapsed <= 0 else q_values.size / elapsed,
        "samples_per_wall_second": 0.0 if elapsed <= 0 else len(rows) / elapsed,
        "model_name": model_info["model_name"],
        "model_config_hash": model_info["model_config_hash"],
        "updated_at": utc_timestamp(),
    }
    atomic_write_json(Path(output_root) / "run_statistics.json", statistics)
    if finite.size:
        bins = np.linspace(0.0, 1.0, 101)
        counts, edges = np.histogram(finite, bins=bins)
        distribution = [
            {
                "lower_inclusive": float(edges[index]),
                "upper_exclusive": float(edges[index + 1]),
                "count": int(counts[index]),
            }
            for index in range(len(counts))
        ]
    else:
        distribution = []
    atomic_write_csv(
        Path(output_root) / "q_value_distribution.csv",
        distribution,
        ("lower_inclusive", "upper_exclusive", "count"),
    )
    return statistics
