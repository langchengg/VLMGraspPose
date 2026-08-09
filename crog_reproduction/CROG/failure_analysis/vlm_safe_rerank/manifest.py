"""Immutable manifests and identity checks for the safe VLM reranking run.

This module deliberately contains no provider or evaluator code.  Its job is to
freeze the local inputs that an inference process is allowed to see and to make
candidate/q drift observable before any formal request can be authorised.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
DATA_MANIFEST_KIND = "vlm_safe_rerank_data_manifest"
INFERENCE_MANIFEST_KIND = "vlm_safe_rerank_inference_manifest"
PROTOCOL_LOCK_KIND = "vlm_safe_rerank_protocol_lock"
LOCKED_MANIFEST_KIND = "vlm_safe_rerank_locked_manifest"

_RESERVED_LAYER_FIELDS = {
    "schema_version",
    "manifest_kind",
    "created_at_utc",
    "content_sha256",
}

_FORBIDDEN_FIELD_TOKENS = {
    "answer",
    "angle_error",
    "candidate_correct",
    "candidate_iou",
    "candidate_positive",
    "correct_candidate_id",
    "correctness",
    "evaluation_only",
    "evaluator",
    "first_valid_rank",
    "grasps",
    "ground_truth",
    "gt",
    "gt_angle",
    "gt_box",
    "gt_cos",
    "gt_grasp",
    "gt_mask",
    "gt_qua",
    "gt_sin",
    "gt_wid",
    "harmful",
    "j1",
    "jany",
    "label",
    "labels",
    "objid",
    "oracle",
    "program",
    "recovered",
    "target",
    "target_idx",
}

_FORBIDDEN_PATH_FRAGMENTS = {
    "evaluation_only",
    "ground_truth",
    "gt",
    "label",
    "labels",
}

_CROG_GEOMETRY_FIELDS = (
    "row",
    "col",
    "cx",
    "cy",
    "angle_rad",
    "angle_deg",
    "width_px",
    "height_px",
    "polygon",
)


class ManifestError(RuntimeError):
    """Raised when an immutable manifest or a frozen identity is invalid."""


def canonical_json(value: Any) -> str:
    """Return the single canonical JSON representation used for all hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ManifestError(f"artifact is not a regular file: {resolved}")
    return {
        "identity_kind": "file_sha256",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def verify_file_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("identity_kind") != "file_sha256":
        raise ManifestError("unsupported file identity kind")
    path = Path(str(identity.get("path", "")))
    if not path.is_absolute():
        raise ManifestError(f"file identity path must be absolute: {path}")
    if not path.is_file():
        raise ManifestError(f"bound artifact is missing: {path}")
    actual_size = path.stat().st_size
    expected_size = _positive_or_zero_int(identity.get("size_bytes"), "size_bytes")
    if actual_size != expected_size:
        raise ManifestError(
            f"bound artifact size drift for {path}: {actual_size} != {expected_size}"
        )
    actual_sha = sha256_file(path)
    expected_sha = _sha256(identity.get("sha256"), "sha256")
    if actual_sha != expected_sha:
        raise ManifestError(f"bound artifact hash drift for {path}")


def _verify_embedded_file_identities(value: Any) -> None:
    if isinstance(value, Mapping):
        if value.get("identity_kind") == "file_sha256":
            verify_file_identity(value)
            return
        for child in value.values():
            _verify_embedded_file_identities(child)
    elif isinstance(value, list):
        for child in value:
            _verify_embedded_file_identities(child)


def _sha256(value: Any, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ManifestError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ManifestError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ManifestError(f"{field} must be a positive integer")
    return parsed


def _positive_or_zero_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ManifestError(f"{field} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must be a non-negative integer") from exc
    if parsed < 0 or parsed != value:
        raise ManifestError(f"{field} must be a non-negative integer")
    return parsed


def _normalise_name(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _forbidden_name(name: Any) -> str | None:
    normalised = _normalise_name(name)
    if normalised in _FORBIDDEN_FIELD_TOKENS:
        return normalised
    parts = normalised.split("_") if normalised else []
    if "label" in parts or "labels" in parts or "gt" in parts:
        return normalised
    if normalised.startswith("evaluation_only") or normalised.endswith("evaluation_only"):
        return normalised
    return None


def _looks_like_path(value: str) -> bool:
    suffixes = (".json", ".jsonl", ".parquet", ".csv", ".npz", ".npy", ".pt", ".pth")
    return "/" in value or "\\" in value or value.lower().endswith(suffixes)


def _forbidden_path(value: str) -> str | None:
    if not _looks_like_path(value):
        return None
    for component in re.split(r"[/\\]+", value):
        stem = _normalise_name(Path(component).stem)
        parts = stem.split("_") if stem else []
        for fragment in _FORBIDDEN_PATH_FRAGMENTS:
            if stem == fragment or fragment in parts or stem.startswith(f"{fragment}_"):
                return component
    return None


def assert_inference_safe_manifest(payload: Mapping[str, Any]) -> None:
    """Reject evaluation/GT/label-bearing fields and paths recursively.

    This is intentionally a deny-list *and* a separation boundary: labels may be
    present in DATA_MANIFEST as evaluation-only artifacts, but the inference
    manifest bound into requests must never reference them.
    """

    if not isinstance(payload, Mapping):
        raise ManifestError("inference manifest must be a JSON object")

    def visit(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                forbidden = _forbidden_name(key)
                if forbidden:
                    raise ManifestError(
                        f"inference manifest contains forbidden field {key!r} at {location}"
                    )
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")
        elif isinstance(value, str):
            forbidden_component = _forbidden_path(value)
            if forbidden_component:
                raise ManifestError(
                    "inference manifest references a forbidden label/GT/evaluation-only "
                    f"path at {location}: {forbidden_component!r}"
                )

    visit(payload, "$")


def _content_sha256(payload: Mapping[str, Any]) -> str:
    without_digest = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(canonical_json(without_digest).encode("utf-8")).hexdigest()


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create a JSON file once, without a check-then-replace race."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    try:
        data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(fd)
    return destination


def build_layer(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ManifestError("manifest payload must be a mapping")
    overlap = _RESERVED_LAYER_FIELDS.intersection(payload)
    if overlap:
        raise ManifestError(f"reserved manifest fields supplied by caller: {sorted(overlap)}")
    layer = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **dict(payload),
    }
    layer["content_sha256"] = _content_sha256(layer)
    return layer


def write_layer(path: str | Path, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    layer = build_layer(kind, payload)
    _exclusive_write_json(path, layer)
    return layer


def load_json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"expected JSON object in {path}")
    return value


def verify_layer(
    path: str | Path,
    *,
    expected_kind: str | None = None,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    layer = load_json_object(path)
    if layer.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest schema in {path}")
    if expected_kind is not None and layer.get("manifest_kind") != expected_kind:
        raise ManifestError(
            f"manifest kind mismatch in {path}: {layer.get('manifest_kind')!r} != {expected_kind!r}"
        )
    expected_sha = _sha256(layer.get("content_sha256"), "content_sha256")
    if _content_sha256(layer) != expected_sha:
        raise ManifestError(f"manifest content hash mismatch in {path}")
    if verify_bound_files:
        _verify_embedded_file_identities(layer)
    return layer


def _stable_sample_id(row: Mapping[str, Any]) -> str:
    sample_id = row.get("sample_id", row.get("key"))
    if sample_id is None or str(sample_id).strip() == "":
        raise ManifestError("candidate row has no stable sample_id/key")
    return str(sample_id)


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    candidate_id = candidate.get("stable_candidate_id", candidate.get("candidate_id"))
    if candidate_id is None or str(candidate_id).strip() == "":
        raise ManifestError("candidate has no stable_candidate_id/candidate_id")
    return str(candidate_id)


def _candidate_q(candidate: Mapping[str, Any]) -> float:
    value = candidate.get("q_raw", candidate.get("q"))
    try:
        q_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError("candidate q/q_raw must be numeric") from exc
    if not math.isfinite(q_value):
        raise ManifestError("candidate q/q_raw must be finite")
    return q_value


def _candidate_checksum(candidate: Mapping[str, Any]) -> str:
    checksum = candidate.get("candidate_checksum", candidate.get("checksum"))
    if checksum is None or str(checksum).strip() == "":
        raise ManifestError("candidate has no frozen candidate_checksum/checksum")
    checksum_text = str(checksum)
    present_geometry_fields = [field for field in _CROG_GEOMETRY_FIELDS if field in candidate]
    if present_geometry_fields:
        if len(present_geometry_fields) != len(_CROG_GEOMETRY_FIELDS):
            missing = sorted(set(_CROG_GEOMETRY_FIELDS) - set(present_geometry_fields))
            raise ManifestError(f"candidate has incomplete frozen geometry: {missing}")
        geometry = {field: candidate[field] for field in _CROG_GEOMETRY_FIELDS}
        computed = hashlib.sha256(
            json.dumps(
                geometry,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if computed != checksum_text:
            raise ManifestError("candidate geometry no longer matches candidate_checksum")
    return checksum_text


def compute_candidate_q_identity(path: str | Path) -> dict[str, Any]:
    """Compute ordered candidate geometry and q-value identities from JSONL."""

    source = Path(path)
    candidate_stream: list[dict[str, Any]] = []
    q_stream: list[dict[str, Any]] = []
    combined_stream: list[dict[str, Any]] = []
    seen_samples: set[str] = set()

    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ManifestError(f"cannot read candidate feature source {source}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"invalid JSONL at {source}:{line_number}") from exc
        if not isinstance(row, Mapping):
            raise ManifestError(f"candidate row must be an object at {source}:{line_number}")
        sample_id = _stable_sample_id(row)
        if sample_id in seen_samples:
            raise ManifestError(f"duplicate candidate sample_id: {sample_id}")
        seen_samples.add(sample_id)
        candidates = row.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ManifestError(f"sample {sample_id} has no candidate list")
        if len(candidates) != 5:
            raise ManifestError(f"sample {sample_id} must have exactly five frozen candidates")

        parsed: list[tuple[str, float, str]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ManifestError(f"sample {sample_id} has a non-object candidate")
            parsed.append(
                (_candidate_id(candidate), _candidate_q(candidate), _candidate_checksum(candidate))
            )
        candidate_ids = [candidate_id for candidate_id, _, _ in parsed]
        if len(set(candidate_ids)) != 5:
            raise ManifestError(f"sample {sample_id} has duplicate stable candidate IDs")

        q_order = sorted(parsed, key=lambda item: (-item[1], item[0]))
        ranks = {candidate_id: index for index, (candidate_id, _, _) in enumerate(q_order)}
        by_id = {
            _candidate_id(candidate): candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
        }
        for candidate_id, q_value, checksum in sorted(parsed, key=lambda item: item[0]):
            stored_rank = by_id[candidate_id].get("q_rank")
            if stored_rank is not None and stored_rank != ranks[candidate_id]:
                raise ManifestError(
                    f"sample {sample_id} candidate {candidate_id} has stale q_rank"
                )
            candidate_record = {
                "sample_id": sample_id,
                "stable_candidate_id": candidate_id,
                "candidate_checksum": checksum,
            }
            q_record = {
                "sample_id": sample_id,
                "stable_candidate_id": candidate_id,
                "q_raw": q_value,
                "q_rank": ranks[candidate_id],
            }
            candidate_stream.append(candidate_record)
            q_stream.append(q_record)
            combined_stream.append({**candidate_record, **q_record})

    if not seen_samples:
        raise ManifestError(f"candidate feature source is empty: {source}")

    def stream_hash(rows: list[dict[str, Any]]) -> str:
        ordered = sorted(rows, key=lambda row: (row["sample_id"], row["stable_candidate_id"]))
        return hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()

    return {
        "sample_count": len(seen_samples),
        "candidate_count": len(candidate_stream),
        "candidates_per_sample": 5,
        "candidate_identity_sha256": stream_hash(candidate_stream),
        "q_value_sha256": stream_hash(q_stream),
        "combined_identity_sha256": stream_hash(combined_stream),
    }


def verify_candidate_q_identity(
    path: str | Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    actual = compute_candidate_q_identity(path)
    for field in (
        "sample_count",
        "candidate_count",
        "candidates_per_sample",
        "candidate_identity_sha256",
        "q_value_sha256",
        "combined_identity_sha256",
    ):
        if actual[field] != expected.get(field):
            raise ManifestError(f"candidate/q identity drift in {field}")
    return actual


def write_inference_manifest(
    path: str | Path,
    *,
    run_id: str,
    expected_denominator: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    denominator = _positive_int(expected_denominator, "expected_denominator")
    assert_inference_safe_manifest(payload)
    layer = write_layer(
        path,
        INFERENCE_MANIFEST_KIND,
        {
            "run_id": _nonempty(run_id, "run_id"),
            "expected_denominator": denominator,
            "inference_inputs": dict(payload),
        },
    )
    return layer


def verify_inference_manifest(path: str | Path) -> dict[str, Any]:
    layer = verify_layer(path, expected_kind=INFERENCE_MANIFEST_KIND)
    _positive_int(layer.get("expected_denominator"), "expected_denominator")
    inputs = layer.get("inference_inputs")
    if not isinstance(inputs, Mapping):
        raise ManifestError("inference manifest has no inference_inputs object")
    assert_inference_safe_manifest(inputs)
    return layer


def _nonempty(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ManifestError(f"{field} must be non-empty")
    return text


def write_data_manifest(
    path: str | Path,
    *,
    run_id: str,
    split_manifest: str | Path,
    candidate_sources: Mapping[str, str | Path],
    inference_manifest: str | Path,
    expected_denominator: int,
    formal_partition: str | None = None,
    evaluation_sources: Mapping[str, str | Path] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the first immutable layer, including separate evaluation inputs."""

    denominator = _positive_int(expected_denominator, "expected_denominator")
    if not candidate_sources:
        raise ManifestError("candidate_sources must not be empty")
    if formal_partition is None:
        if len(candidate_sources) != 1:
            raise ManifestError("formal_partition is required with multiple candidate sources")
        formal_partition = next(iter(candidate_sources))
    if formal_partition not in candidate_sources:
        raise ManifestError(f"unknown formal_partition: {formal_partition}")

    inference = verify_inference_manifest(inference_manifest)
    if inference["expected_denominator"] != denominator:
        raise ManifestError("inference manifest expected denominator mismatch")

    candidate_artifacts: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for partition, source_path in sorted(candidate_sources.items()):
        partition_name = _nonempty(partition, "candidate partition")
        candidate_artifacts[partition_name] = file_identity(source_path)
        identities[partition_name] = compute_candidate_q_identity(source_path)

    formal_identity = identities[formal_partition]
    if formal_identity["sample_count"] != denominator:
        raise ManifestError(
            "formal candidate source sample count does not match expected denominator"
        )

    evaluation_artifacts = {
        _nonempty(name, "evaluation source name"): file_identity(source)
        for name, source in sorted((evaluation_sources or {}).items())
    }
    payload: dict[str, Any] = {
        "run_id": _nonempty(run_id, "run_id"),
        "expected_denominator": denominator,
        "formal_partition": formal_partition,
        "split_manifest": file_identity(split_manifest),
        "candidate_sources": candidate_artifacts,
        "candidate_q_identities": identities,
        "formal_candidate_q_identity": formal_identity,
        "inference_manifest": file_identity(inference_manifest),
        "inference_manifest_content_sha256": inference["content_sha256"],
        "evaluation_sources": evaluation_artifacts,
        "metadata": dict(metadata or {}),
    }
    return write_layer(path, DATA_MANIFEST_KIND, payload)


def verify_data_manifest(path: str | Path) -> dict[str, Any]:
    manifest = verify_layer(path, expected_kind=DATA_MANIFEST_KIND)
    denominator = _positive_int(manifest.get("expected_denominator"), "expected_denominator")

    inference_identity = manifest.get("inference_manifest")
    if not isinstance(inference_identity, Mapping):
        raise ManifestError("DATA_MANIFEST has no inference manifest identity")
    inference = verify_inference_manifest(str(inference_identity.get("path", "")))
    if inference.get("content_sha256") != manifest.get("inference_manifest_content_sha256"):
        raise ManifestError("bound inference manifest semantic hash drift")
    if inference.get("expected_denominator") != denominator:
        raise ManifestError("bound inference manifest denominator drift")

    sources = manifest.get("candidate_sources")
    identities = manifest.get("candidate_q_identities")
    if not isinstance(sources, Mapping) or not isinstance(identities, Mapping):
        raise ManifestError("DATA_MANIFEST candidate sources/identities are invalid")
    if set(sources) != set(identities):
        raise ManifestError("candidate source and identity partitions differ")
    for partition, source_identity in sources.items():
        if not isinstance(source_identity, Mapping) or not isinstance(identities[partition], Mapping):
            raise ManifestError(f"invalid candidate source identity for {partition}")
        verify_candidate_q_identity(source_identity["path"], identities[partition])

    formal_partition = manifest.get("formal_partition")
    if formal_partition not in identities:
        raise ManifestError("formal candidate partition is missing")
    formal_identity = identities[formal_partition]
    if formal_identity != manifest.get("formal_candidate_q_identity"):
        raise ManifestError("formal candidate/q identity binding mismatch")
    if formal_identity.get("sample_count") != denominator:
        raise ManifestError("formal candidate count/expected denominator mismatch")
    return manifest


# Intuitive aliases used by callers and tests.
create_data_manifest = write_data_manifest
create_inference_manifest = write_inference_manifest


__all__ = [
    "DATA_MANIFEST_KIND",
    "INFERENCE_MANIFEST_KIND",
    "LOCKED_MANIFEST_KIND",
    "PROTOCOL_LOCK_KIND",
    "ManifestError",
    "assert_inference_safe_manifest",
    "build_layer",
    "canonical_json",
    "compute_candidate_q_identity",
    "create_data_manifest",
    "create_inference_manifest",
    "file_identity",
    "load_json_object",
    "sha256_file",
    "verify_candidate_q_identity",
    "verify_data_manifest",
    "verify_file_identity",
    "verify_inference_manifest",
    "verify_layer",
    "write_data_manifest",
    "write_inference_manifest",
    "write_layer",
]
