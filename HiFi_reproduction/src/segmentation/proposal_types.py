"""Stable candidate records shared by SAM 3 proposal-bank stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def mask_sha256(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    payload = np.asarray(value.shape, dtype=np.int64).tobytes() + np.packbits(value).tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_candidate_masks_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load legacy dense or schema-v2 packbits proposal archives safely."""

    archive = np.load(Path(path), allow_pickle=False)
    try:
        stored_shape = (
            tuple(int(value) for value in archive["__mask_shape__"])
            if "__mask_shape__" in archive.files
            else None
        )
        result: dict[str, np.ndarray] = {}
        if "__candidate_ids__" in archive.files and "__packed_masks__" in archive.files:
            identifiers = [str(value) for value in archive["__candidate_ids__"]]
            packed = archive["__packed_masks__"]
            if stored_shape is None or packed.shape[0] != len(identifiers):
                raise ValueError("schema-v3 proposal archive is inconsistent")
            for index, key in enumerate(identifiers):
                result[key] = (
                    np.unpackbits(
                        packed[index],
                        count=int(np.prod(stored_shape)),
                        bitorder="little",
                    )
                    .reshape(stored_shape)
                    .astype(bool)
                )
            return result
        for key in archive.files:
            if key.startswith("__"):
                continue
            if stored_shape is None:
                result[key] = np.asarray(archive[key], dtype=bool)
            else:
                result[key] = (
                    np.unpackbits(
                        archive[key],
                        count=int(np.prod(stored_shape)),
                        bitorder="little",
                    )
                    .reshape(stored_shape)
                    .astype(bool)
                )
        return result
    finally:
        archive.close()


def stable_candidate_id(
    sample_id: str,
    source_family: str,
    source_variant: str,
    provenance: dict[str, Any],
    mask: np.ndarray,
    *,
    mask_digest: str | None = None,
) -> str:
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(
        f"{sample_id}|{source_family}|{source_variant}|{canonical}|"
        f"{mask_digest or mask_sha256(mask)}".encode()
    ).hexdigest()[:20]
    return f"{source_family.lower()}_{digest}"


def reference_candidate_ids_from_provenance(
    index: "Any", provenance: dict[str, Any]
) -> set[str]:
    """Recover logical reference roles preserved through physical-mask deduplication."""

    result: set[str] = set()
    for raw_index, row in index.iterrows():
        candidate_id = str(row.get("candidate_id", raw_index))
        if str(row.get("source_family", "")).startswith("REFERENCE_"):
            result.add(candidate_id)
            continue
        values = provenance.get(candidate_id, [])
        for value in values if isinstance(values, list) else [values]:
            if not isinstance(value, dict):
                continue
            source_family = str(value.get("source_family", ""))
            prompt_id = str(
                value.get("sample_prompt_id", value.get("prompt_id", ""))
            )
            source_variant = str(
                value.get(
                    "sample_source_variant", value.get("source_variant", "")
                )
            )
            if (
                source_family.startswith("REFERENCE_")
                or prompt_id in {"R1", "R2"}
                or source_variant.startswith(("R1_", "R2_"))
            ):
                result.add(candidate_id)
                break
    return result


@dataclass
class ProposalCandidate:
    sample_id: str
    source_family: str
    source_variant: str
    mask: np.ndarray
    probability: np.ndarray | None = None
    sam_score: float | None = None
    presence_score: float | None = None
    mask_quality_score: float | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    canonical_text_prompt: str | None = None
    prompt_boxes: tuple[tuple[float, float, float, float], ...] = ()
    prompt_points: tuple[tuple[float, float, int], ...] = ()
    mask_threshold: float | None = None
    instance_threshold: float | None = None
    model_revision: str | None = None
    rgb_checksum: str | None = None
    eligible_final: bool = True
    raw_mask_reference: str | None = None
    source_rank: int | None = None
    provenance: list[dict[str, Any]] = field(default_factory=list)
    deduplication_parent_ids: list[str] = field(default_factory=list)
    candidate_id: str | None = None
    _geometry_cache: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 2:
            raise ValueError("candidate mask must be two-dimensional")
        if self.probability is not None:
            self.probability = np.asarray(self.probability, dtype=np.float32)
            if self.probability.shape != self.mask.shape or not np.isfinite(self.probability).all():
                raise ValueError("candidate probability must be finite and mask-aligned")
        if not self.provenance:
            self.provenance.append(
                {
                    "source_family": self.source_family,
                    "source_variant": self.source_variant,
                    "canonical_text_prompt": self.canonical_text_prompt,
                    "mask_threshold": self.mask_threshold,
                    "instance_threshold": self.instance_threshold,
                }
            )
        if self.candidate_id is None:
            digest = mask_sha256(self.mask)
            self._geometry_cache["mask_sha256"] = digest
            self.candidate_id = stable_candidate_id(
                self.sample_id,
                self.source_family,
                self.source_variant,
                self.provenance[0],
                self.mask,
                mask_digest=digest,
            )

    @property
    def area(self) -> int:
        if "area" not in self._geometry_cache:
            self._geometry_cache["area"] = int(np.count_nonzero(self.mask))
        return int(self._geometry_cache["area"])

    @property
    def mask_digest(self) -> str:
        if "mask_sha256" not in self._geometry_cache:
            self._geometry_cache["mask_sha256"] = mask_sha256(self.mask)
        return str(self._geometry_cache["mask_sha256"])

    @property
    def mask_box(self) -> tuple[int, int, int, int] | None:
        if "mask_box" not in self._geometry_cache:
            yy, xx = np.nonzero(self.mask)
            self._geometry_cache["mask_box"] = (
                None
                if not len(xx)
                else (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max()))
            )
        return self._geometry_cache["mask_box"]

    @property
    def connected_components(self) -> int:
        if "connected_components" not in self._geometry_cache:
            try:
                import cv2

                count = int(
                    cv2.connectedComponents(
                        self.mask.astype(np.uint8), connectivity=4
                    )[0]
                    - 1
                )
            except ImportError:
                from scipy import ndimage

                count = int(ndimage.label(self.mask)[1])
            self._geometry_cache["connected_components"] = count
        return int(self._geometry_cache["connected_components"])

    def to_index_record(self) -> dict[str, Any]:
        if self.box_xyxy is None and self.area:
            box = tuple(float(value) for value in self.mask_box or ())
        else:
            box = self.box_xyxy
        return {
            "sample_id": self.sample_id,
            "candidate_id": self.candidate_id,
            "source_family": self.source_family,
            "source_variant": self.source_variant,
            "canonical_text_prompt": self.canonical_text_prompt,
            "prompt_boxes_json": json.dumps(self.prompt_boxes),
            "prompt_points_json": json.dumps(self.prompt_points),
            "mask_threshold": self.mask_threshold,
            "instance_threshold": self.instance_threshold,
            "sam_score": self.sam_score,
            "presence_score": self.presence_score,
            "mask_quality_score": self.mask_quality_score,
            "box_json": json.dumps(box),
            "area": self.area,
            "connected_components": self.connected_components,
            "raw_mask_reference": self.raw_mask_reference,
            "deduplication_parent_ids_json": json.dumps(self.deduplication_parent_ids),
            "provenance_count": len(self.provenance),
            "model_revision": self.model_revision,
            "rgb_checksum": self.rgb_checksum,
            "eligible_final": bool(self.eligible_final),
            "source_rank": self.source_rank,
            "mask_sha256": self.mask_digest,
        }


__all__ = [
    "ProposalCandidate",
    "load_candidate_masks_npz",
    "mask_sha256",
    "reference_candidate_ids_from_provenance",
    "stable_candidate_id",
]
