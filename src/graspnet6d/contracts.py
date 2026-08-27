"""Frozen, serialization-safe 6-DoF candidate contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import as_rotation_matrix, as_translation, rotation_geodesic_deg
from .io import atomic_json, atomic_npz, canonical_json_bytes, canonical_sha256, sha256_file


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
VoxelIndex = tuple[int, int, int]


def _vector_tuple(value: ArrayLike) -> Vector3:
    vector = as_translation(value)
    return tuple(float(item) for item in vector)  # type: ignore[return-value]


def _matrix_tuple(value: ArrayLike) -> Matrix3:
    matrix = as_rotation_matrix(value)
    return tuple(tuple(float(item) for item in row) for row in matrix)  # type: ignore[return-value]


def _voxel_tuple(value: Sequence[int]) -> VoxelIndex:
    raw = np.asarray(value)
    if raw.shape != (3,):
        raise ValueError(f"voxel_index must have shape (3,), got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("voxel_index must contain integers")
    converted = tuple(int(item) for item in raw)
    if any(item < 0 for item in converted):
        raise ValueError("voxel_index entries must be non-negative")
    return converted  # type: ignore[return-value]


def _normalized_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("conversion_provenance must be a mapping")
    # Round-trip through the canonical codec to reject NaN and mutable/non-JSON data.
    normalized = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    required = {"mapping_name", "mapping_status", "source"}
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(f"conversion_provenance missing fields: {missing}")
    if any(not str(normalized[key]).strip() for key in required):
        raise ValueError("conversion provenance identifiers must be non-empty")
    return normalized


@dataclass(frozen=True, slots=True)
class Candidate6D:
    """One immutable VGN candidate expressed in every evaluator-relevant frame.

    ``height_m`` and ``depth_m`` are frozen even though they are not native VGN
    outputs: the official GraspNet collision geometry requires them.  Their
    deterministic derivation or configured constants must be recorded in
    ``conversion_provenance``.
    """

    SCHEMA_VERSION: ClassVar[str] = "graspnet6d_candidate_v1"

    candidate_id: str
    group_id: str
    native_rank: int
    native_score: float
    translation_local_m: Vector3 | ArrayLike
    rotation_local: Matrix3 | ArrayLike
    translation_camera_m: Vector3 | ArrayLike
    rotation_camera: Matrix3 | ArrayLike
    translation_table_m: Vector3 | ArrayLike
    rotation_table: Matrix3 | ArrayLike
    width_m: float
    height_m: float
    depth_m: float
    voxel_index: VoxelIndex | Sequence[int]
    conversion_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if isinstance(self.native_rank, bool) or not isinstance(self.native_rank, int):
            raise ValueError("native_rank must be an integer")
        if self.native_rank < 0:
            raise ValueError("native_rank must be non-negative")
        if not math.isfinite(float(self.native_score)):
            raise ValueError("native_score must be finite")
        for field_name in ("width_m", "height_m", "depth_m"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "native_score", float(self.native_score))
        object.__setattr__(
            self, "translation_local_m", _vector_tuple(self.translation_local_m)
        )
        object.__setattr__(self, "rotation_local", _matrix_tuple(self.rotation_local))
        object.__setattr__(
            self, "translation_camera_m", _vector_tuple(self.translation_camera_m)
        )
        object.__setattr__(self, "rotation_camera", _matrix_tuple(self.rotation_camera))
        object.__setattr__(
            self, "translation_table_m", _vector_tuple(self.translation_table_m)
        )
        object.__setattr__(self, "rotation_table", _matrix_tuple(self.rotation_table))
        object.__setattr__(self, "voxel_index", _voxel_tuple(self.voxel_index))
        object.__setattr__(
            self,
            "conversion_provenance",
            _normalized_provenance(self.conversion_provenance),
        )

    def geometry_payload(self) -> dict[str, Any]:
        return {
            "translation_local_m": self.translation_local_m,
            "rotation_local": self.rotation_local,
            "translation_camera_m": self.translation_camera_m,
            "rotation_camera": self.rotation_camera,
            "translation_table_m": self.translation_table_m,
            "rotation_table": self.rotation_table,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "depth_m": self.depth_m,
            "voxel_index": self.voxel_index,
        }

    @property
    def geometry_sha256(self) -> str:
        return canonical_sha256(self.geometry_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "group_id": self.group_id,
            "native_rank": self.native_rank,
            "native_score": self.native_score,
            **self.geometry_payload(),
            "conversion_provenance": dict(self.conversion_provenance),
        }

    @property
    def record_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Candidate6D":
        payload = dict(value)
        schema_version = payload.pop("schema_version", cls.SCHEMA_VERSION)
        if schema_version != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported candidate schema: {schema_version}")
        return cls(**payload)


def stable_candidate_id(
    group_id: str,
    *,
    translation_local_m: ArrayLike,
    rotation_local: ArrayLike,
    width_m: float,
    height_m: float,
    depth_m: float,
    voxel_index: Sequence[int],
) -> str:
    """Derive an order-independent identity from source group and local geometry."""

    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("group_id must be non-empty")
    payload = {
        "group_id": group_id,
        "translation_local_m": _vector_tuple(translation_local_m),
        "rotation_local": _matrix_tuple(rotation_local),
        "width_m": float(width_m),
        "height_m": float(height_m),
        "depth_m": float(depth_m),
        "voxel_index": _voxel_tuple(voxel_index),
    }
    return f"{group_id}:{canonical_sha256(payload)[:24]}"


def validate_candidate_pool(candidates: Iterable[Candidate6D]) -> tuple[Candidate6D, ...]:
    values = tuple(candidates)
    identifiers = [candidate.candidate_id for candidate in values]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate_id values must be unique")
    if values:
        group_ids = {candidate.group_id for candidate in values}
        if len(group_ids) != 1:
            raise ValueError(f"one candidate pool cannot span groups: {sorted(group_ids)}")
        ranks = [candidate.native_rank for candidate in values]
        if len(ranks) != len(set(ranks)):
            raise ValueError("native_rank values must be unique within a group")
    return values


def candidate_pool_fingerprint(candidates: Iterable[Candidate6D]) -> str:
    values = validate_candidate_pool(candidates)
    records = sorted(
        (
            {
                "candidate_id": candidate.candidate_id,
                "native_rank": candidate.native_rank,
                "native_score": candidate.native_score,
                "geometry_sha256": candidate.geometry_sha256,
                "conversion_provenance": candidate.conversion_provenance,
            }
            for candidate in values
        ),
        key=lambda row: row["candidate_id"],
    )
    return canonical_sha256(records)


def assert_frozen_candidate_pool(
    native_candidates: Iterable[Candidate6D],
    reranked_candidates: Iterable[Candidate6D],
) -> None:
    """Require identical membership, geometry, native scores, and provenance."""

    native = validate_candidate_pool(native_candidates)
    reranked = validate_candidate_pool(reranked_candidates)
    native_by_id = {candidate.candidate_id: candidate for candidate in native}
    reranked_by_id = {candidate.candidate_id: candidate for candidate in reranked}
    if set(native_by_id) != set(reranked_by_id):
        missing = sorted(set(native_by_id) - set(reranked_by_id))
        added = sorted(set(reranked_by_id) - set(native_by_id))
        raise ValueError(f"candidate membership changed: missing={missing}, added={added}")
    changed_geometry = sorted(
        candidate_id
        for candidate_id in native_by_id
        if native_by_id[candidate_id].geometry_sha256
        != reranked_by_id[candidate_id].geometry_sha256
    )
    if changed_geometry:
        raise ValueError(f"candidate geometry changed: {changed_geometry}")
    changed_native = sorted(
        candidate_id
        for candidate_id in native_by_id
        if (
            native_by_id[candidate_id].native_rank,
            native_by_id[candidate_id].native_score,
            native_by_id[candidate_id].conversion_provenance,
        )
        != (
            reranked_by_id[candidate_id].native_rank,
            reranked_by_id[candidate_id].native_score,
            reranked_by_id[candidate_id].conversion_provenance,
        )
    )
    if changed_native:
        raise ValueError(f"candidate native record changed: {changed_native}")


def pose_nms(
    candidates: Iterable[Candidate6D],
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    width_threshold_m: float | None = None,
    pre_nms_max_candidates: int | None = None,
    top_k: int | None = None,
) -> list[Candidate6D]:
    """Apply deterministic score-ordered pose suppression.

    A non-``None`` ``width_threshold_m`` enables the experiment-specific rule in
    the master protocol.  The official GraspNet pose NMS compares translation
    and rotation only; call this function with ``width_threshold_m=None`` when
    reproducing that membership rule.
    """

    values = validate_candidate_pool(candidates)
    for name, value in (
        ("translation_threshold_m", translation_threshold_m),
        ("rotation_threshold_deg", rotation_threshold_deg),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if width_threshold_m is not None and (
        not math.isfinite(float(width_threshold_m)) or width_threshold_m < 0.0
    ):
        raise ValueError("width_threshold_m must be finite and non-negative")
    for name, value in (
        ("pre_nms_max_candidates", pre_nms_max_candidates),
        ("top_k", top_k),
    ):
        if value is not None and (isinstance(value, bool) or int(value) <= 0):
            raise ValueError(f"{name} must be a positive integer when provided")

    ordered = sorted(
        values,
        key=lambda candidate: (
            -candidate.native_score,
            candidate.native_rank,
            candidate.candidate_id,
        ),
    )
    if pre_nms_max_candidates is not None:
        ordered = ordered[: int(pre_nms_max_candidates)]
    kept: list[Candidate6D] = []
    for candidate in ordered:
        candidate_translation = np.asarray(candidate.translation_local_m)
        suppressed = False
        for accepted in kept:
            translation_close = (
                np.linalg.norm(candidate_translation - np.asarray(accepted.translation_local_m))
                <= translation_threshold_m
            )
            rotation_close = (
                rotation_geodesic_deg(candidate.rotation_local, accepted.rotation_local)
                <= rotation_threshold_deg
            )
            width_close = width_threshold_m is None or (
                abs(candidate.width_m - accepted.width_m) <= width_threshold_m
            )
            if translation_close and rotation_close and width_close:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
            if top_k is not None and len(kept) >= int(top_k):
                break
    return kept


def save_candidate_cache(
    npz_path: str | Path,
    candidates: Iterable[Candidate6D],
    *,
    metadata: Mapping[str, Any] | None = None,
    schema_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write a pickle-free numeric cache plus an integrity-bearing JSON sidecar."""

    values = validate_candidate_pool(candidates)
    destination = Path(npz_path).expanduser().resolve()
    sidecar = (
        Path(schema_path).expanduser().resolve()
        if schema_path is not None
        else destination.with_suffix(".json")
    )
    count = len(values)

    def stacked(name: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
        if not values:
            return np.empty((0, *shape), dtype=np.float64)
        return np.asarray([getattr(candidate, name) for candidate in values], dtype=np.float64)

    arrays: dict[str, Any] = {
        "candidate_id": np.asarray([item.candidate_id for item in values], dtype=np.str_),
        "group_id": np.asarray([item.group_id for item in values], dtype=np.str_),
        "native_rank": np.asarray([item.native_rank for item in values], dtype=np.int64),
        "native_score": np.asarray([item.native_score for item in values], dtype=np.float64),
        "translation_local_m": stacked("translation_local_m", (3,)),
        "rotation_local": stacked("rotation_local", (3, 3)),
        "translation_camera_m": stacked("translation_camera_m", (3,)),
        "rotation_camera": stacked("rotation_camera", (3, 3)),
        "translation_table_m": stacked("translation_table_m", (3,)),
        "rotation_table": stacked("rotation_table", (3, 3)),
        "width_m": np.asarray([item.width_m for item in values], dtype=np.float64),
        "height_m": np.asarray([item.height_m for item in values], dtype=np.float64),
        "depth_m": np.asarray([item.depth_m for item in values], dtype=np.float64),
        "voxel_index": np.asarray(
            [item.voxel_index for item in values], dtype=np.int64
        ).reshape(count, 3),
    }
    atomic_npz(destination, **arrays)
    manifest = {
        "schema_version": Candidate6D.SCHEMA_VERSION,
        "count": count,
        "npz_file": destination.name,
        "npz_sha256": sha256_file(destination),
        "pool_fingerprint": candidate_pool_fingerprint(values),
        "candidate_records": [
            {
                "candidate_id": item.candidate_id,
                "record_sha256": item.record_sha256,
                "conversion_provenance": item.conversion_provenance,
            }
            for item in values
        ],
        "metadata": dict(metadata or {}),
    }
    atomic_json(sidecar, manifest)
    return destination, sidecar


def load_candidate_cache(
    npz_path: str | Path,
    *,
    schema_path: str | Path | None = None,
) -> tuple[list[Candidate6D], dict[str, Any]]:
    source = Path(npz_path).expanduser().resolve()
    sidecar = (
        Path(schema_path).expanduser().resolve()
        if schema_path is not None
        else source.with_suffix(".json")
    )
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != Candidate6D.SCHEMA_VERSION:
        raise ValueError(f"unsupported candidate schema: {manifest.get('schema_version')}")
    if manifest.get("npz_sha256") != sha256_file(source):
        raise ValueError("candidate NPZ SHA-256 does not match its sidecar")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "candidate_id",
            "group_id",
            "native_rank",
            "native_score",
            "translation_local_m",
            "rotation_local",
            "translation_camera_m",
            "rotation_camera",
            "translation_table_m",
            "rotation_table",
            "width_m",
            "height_m",
            "depth_m",
            "voxel_index",
        }
        if set(archive.files) != required:
            raise ValueError(f"candidate NPZ schema mismatch: {sorted(archive.files)}")
        count = int(manifest.get("count", -1))
        if any(np.asarray(archive[name]).shape[0] != count for name in required):
            raise ValueError("candidate NPZ arrays disagree with manifest count")
        records = manifest.get("candidate_records")
        if not isinstance(records, list) or len(records) != count:
            raise ValueError("candidate sidecar record count mismatch")
        candidates = [
            Candidate6D(
                candidate_id=str(archive["candidate_id"][index]),
                group_id=str(archive["group_id"][index]),
                native_rank=int(archive["native_rank"][index]),
                native_score=float(archive["native_score"][index]),
                translation_local_m=archive["translation_local_m"][index],
                rotation_local=archive["rotation_local"][index],
                translation_camera_m=archive["translation_camera_m"][index],
                rotation_camera=archive["rotation_camera"][index],
                translation_table_m=archive["translation_table_m"][index],
                rotation_table=archive["rotation_table"][index],
                width_m=float(archive["width_m"][index]),
                height_m=float(archive["height_m"][index]),
                depth_m=float(archive["depth_m"][index]),
                voxel_index=archive["voxel_index"][index],
                conversion_provenance=records[index]["conversion_provenance"],
            )
            for index in range(count)
        ]
    for candidate, expected in zip(candidates, records, strict=True):
        if candidate.candidate_id != expected.get("candidate_id"):
            raise ValueError("candidate ordering differs between NPZ and sidecar")
        if candidate.record_sha256 != expected.get("record_sha256"):
            raise ValueError(f"candidate record hash mismatch: {candidate.candidate_id}")
    if candidate_pool_fingerprint(candidates) != manifest.get("pool_fingerprint"):
        raise ValueError("candidate pool fingerprint mismatch")
    return candidates, dict(manifest.get("metadata") or {})


# Clear aliases for callers that prefer write/read terminology.
write_candidate_cache = save_candidate_cache
read_candidate_cache = load_candidate_cache
