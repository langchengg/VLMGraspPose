"""Fail-closed GT-mask mapping and post-lock pixel QA primitives.

The pre-lock registry deliberately stores only identities, paths, recorded
hashes, and declared shapes.  It never opens a referenced image.  P2 pixel QA
is a separate, purpose-bound access that may validate annotations before the
protocol lock.  Feeding those pixels to a candidate generator remains forbidden
until the final P2 artifact is bound by the P4 protocol lock.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Any

from unified_reranking.hashing import sha256_file


EXPECTED_SAMPLE_COUNT = 7_675
PREPARED_MASK_SHAPE = (352, 352)
ORIGINAL_MASK_SHAPE = (480, 640)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GTMappingError(RuntimeError):
    """A GT authority cannot be mapped without guessing."""


@dataclass(frozen=True)
class GTAuthorityColumns:
    """Explicit column contract for a path/hash-only authority table."""

    sample_id: str = "sample_id"
    scene_id: str = "scene_id"
    query_id: str = "question_index"
    target_instance_id: str = "target_object_id"
    prepared_path: str = "prepared_gt_mask_path"
    prepared_sha256: str = "prepared_gt_mask_sha256"
    source_instance_path: str = "source_instance_mask_path"
    source_instance_sha256: str = "source_instance_mask_sha256"

    def required(self) -> tuple[str, ...]:
        return tuple(self.__dict__.values())


def _nonempty(value: Any, *, field: str, sample_id: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise GTMappingError(f"{sample_id}: {field} is empty")
    return text


def _sha256(value: Any, *, field: str, sample_id: str) -> str:
    text = _nonempty(value, field=field, sample_id=sample_id).lower()
    if _SHA256.fullmatch(text) is None:
        raise GTMappingError(f"{sample_id}: {field} is not a SHA256 digest")
    return text


def _absolute_path(value: Any, *, field: str, sample_id: str) -> str:
    text = _nonempty(value, field=field, sample_id=sample_id)
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise GTMappingError(
            f"{sample_id}: {field} must be absolute; path guessing is forbidden"
        )
    return str(path.resolve(strict=False))


def _strict_instance(value: Any, *, sample_id: str) -> int:
    if isinstance(value, bool):
        raise GTMappingError(f"{sample_id}: target instance ID cannot be boolean")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise GTMappingError(f"{sample_id}: target instance ID is invalid") from error
    if isinstance(value, float) and not value.is_integer():
        raise GTMappingError(f"{sample_id}: target instance ID is not integral")
    if integer <= 0:
        raise GTMappingError(f"{sample_id}: target instance ID must be positive")
    return integer


def _index_unique(
    rows: Iterable[Mapping[str, Any]], *, key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = _nonempty(row.get(key), field=key, sample_id="<unknown>")
        if sample_id in result:
            raise GTMappingError(f"duplicate {label} sample_id: {sample_id}")
        result[sample_id] = row
    return result


def build_prelock_registry(
    expected_samples: Iterable[Mapping[str, Any]],
    authority_rows: Iterable[Mapping[str, Any]],
    *,
    columns: GTAuthorityColumns | None = None,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> list[dict[str, Any]]:
    """Build the path/hash-only registry without opening any referenced file.

    ``expected_samples`` is the denominator authority and must expose
    ``sample_id``, ``scene_id``, and ``question_index``.  The GT authority must
    additionally bind the target instance, prepared 352x352 target mask, and
    original 480x640 multi-instance map.  P2 QA derives the original-resolution
    target binary as ``instance_map == target_instance_id`` inside the new run.
    """

    fields = columns or GTAuthorityColumns()
    expected = _index_unique(expected_samples, key="sample_id", label="denominator")
    authorities = _index_unique(
        authority_rows, key=fields.sample_id, label="GT authority"
    )
    if len(expected) != expected_count:
        raise GTMappingError(
            f"denominator count {len(expected)} differs from {expected_count}"
        )
    extras = sorted(set(authorities).difference(expected))
    if extras:
        raise GTMappingError(f"GT authority has samples outside denominator: {extras[:5]}")

    registry: list[dict[str, Any]] = []
    target_keys: set[tuple[str, str, int]] = set()
    for sample_id in sorted(expected):
        sample = expected[sample_id]
        authority = authorities.get(sample_id)
        if authority is None:
            registry.append(
                {
                    "sample_id": sample_id,
                    "scene_id": str(sample.get("scene_id", "")),
                    "query_id": str(sample.get("question_index", "")),
                    "mapping_status": "gt_oracle_unavailable",
                    "mapping_reason": "missing explicit GT authority row",
                    "pixel_qa_status": "NOT_ACCESSED_PRELOCK",
                    "bulk_gt_pixels_read": False,
                }
            )
            continue

        scene_id = _nonempty(
            authority.get(fields.scene_id), field=fields.scene_id, sample_id=sample_id
        )
        query_id = _nonempty(
            authority.get(fields.query_id), field=fields.query_id, sample_id=sample_id
        )
        if scene_id != str(sample.get("scene_id", "")):
            raise GTMappingError(f"{sample_id}: scene identity differs")
        if query_id != str(sample.get("question_index", "")):
            raise GTMappingError(f"{sample_id}: query identity differs")
        target = _strict_instance(
            authority.get(fields.target_instance_id), sample_id=sample_id
        )
        target_key = (scene_id, query_id, target)
        if target_key in target_keys:
            raise GTMappingError(f"duplicate scene/query/target identity: {target_key}")
        target_keys.add(target_key)

        registry.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "query_id": query_id,
                "target_instance_id": target,
                "prepared_gt_mask_path": _absolute_path(
                    authority.get(fields.prepared_path),
                    field=fields.prepared_path,
                    sample_id=sample_id,
                ),
                "prepared_gt_mask_sha256": _sha256(
                    authority.get(fields.prepared_sha256),
                    field=fields.prepared_sha256,
                    sample_id=sample_id,
                ),
                "prepared_height": PREPARED_MASK_SHAPE[0],
                "prepared_width": PREPARED_MASK_SHAPE[1],
                "original_gt_mask_path": None,
                "original_gt_mask_sha256": None,
                "source_instance_mask_path": _absolute_path(
                    authority.get(fields.source_instance_path),
                    field=fields.source_instance_path,
                    sample_id=sample_id,
                ),
                "source_instance_mask_sha256": _sha256(
                    authority.get(fields.source_instance_sha256),
                    field=fields.source_instance_sha256,
                    sample_id=sample_id,
                ),
                "original_height": ORIGINAL_MASK_SHAPE[0],
                "original_width": ORIGINAL_MASK_SHAPE[1],
                "mapping_status": "PATH_HASH_INSTANCE_AUTHORITY_MAPPED",
                "mapping_reason": "",
                "pixel_qa_status": "NOT_ACCESSED_PRELOCK",
                "bulk_gt_pixels_read": False,
                "foreground_pixels": None,
                "foreground_fraction": None,
                "component_count": None,
                "bbox": None,
            }
        )
    return registry


def _assert_mapping_qa_access(access: Mapping[str, Any]) -> None:
    if access.get("status") != "AUTHORIZED":
        raise PermissionError("P2 GT-mask pixel QA requires explicit authorization")
    if access.get("stage") != "P2_GT_MAPPING_PASS":
        raise PermissionError("GT-mask pixel QA authorization has the wrong stage")
    if access.get("purpose") != "mapping_and_annotation_pixel_qa_only":
        raise PermissionError("GT-mask pixel QA authorization has the wrong purpose")
    if access.get("candidate_generation_allowed") is not False:
        raise PermissionError("P2 GT-mask access must forbid candidate generation")


def mapping_pixel_qa(
    row: Mapping[str, Any],
    *,
    access_authority: Mapping[str, Any],
    derived_output_dir: Path,
    prepared_transform: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Validate one GT authority under the narrow P2 annotation-QA purpose."""

    _assert_mapping_qa_access(access_authority)
    if row.get("mapping_status") != "PATH_HASH_INSTANCE_AUTHORITY_MAPPED":
        raise GTMappingError(f"{row.get('sample_id')}: GT mapping is not evaluable")

    import numpy as np
    from PIL import Image
    from scipy import ndimage

    sample_id = str(row["sample_id"])
    paths = {
        "prepared": Path(str(row["prepared_gt_mask_path"])),
        "instance": Path(str(row["source_instance_mask_path"])),
    }
    hashes = {
        "prepared": str(row["prepared_gt_mask_sha256"]),
        "instance": str(row["source_instance_mask_sha256"]),
    }
    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise GTMappingError(f"{sample_id}: {name} mask hash mismatch")

    with Image.open(paths["prepared"]) as image:
        prepared = np.asarray(image)
    with Image.open(paths["instance"]) as image:
        instance = np.asarray(image)
    if prepared.ndim != 2 or prepared.shape != PREPARED_MASK_SHAPE:
        raise GTMappingError(f"{sample_id}: prepared GT shape is {prepared.shape}")
    if instance.ndim != 2 or instance.shape != ORIGINAL_MASK_SHAPE:
        raise GTMappingError(f"{sample_id}: source instance shape is {instance.shape}")

    rgb_height = int(row.get("rgb_height", 0))
    rgb_width = int(row.get("rgb_width", 0))
    if (rgb_height, rgb_width) != ORIGINAL_MASK_SHAPE:
        raise GTMappingError(
            f"{sample_id}: RGB declared shape {(rgb_height, rgb_width)} differs "
            f"from {ORIGINAL_MASK_SHAPE}"
        )

    target = _strict_instance(row["target_instance_id"], sample_id=sample_id)
    grasp_target = _strict_instance(
        row.get("gt_grasp_target_instance_id"), sample_id=sample_id
    )
    if target != grasp_target:
        raise GTMappingError(f"{sample_id}: GT mask/grasp target identity differs")
    expected_original = instance == target
    prepared_binary = prepared != 0
    if not bool(expected_original.any()):
        raise GTMappingError(f"{sample_id}: target instance is absent")
    if not bool(prepared_binary.any()):
        raise GTMappingError(f"{sample_id}: prepared GT is empty")
    transformed = (
        np.asarray(prepared_transform(expected_original)) != 0
        if prepared_transform is not None
        else np.asarray(
            Image.fromarray(expected_original.astype(np.uint8) * 255).resize(
                (PREPARED_MASK_SHAPE[1], PREPARED_MASK_SHAPE[0]),
                resample=Image.Resampling.NEAREST,
            )
        )
        != 0
    )
    if transformed.shape != PREPARED_MASK_SHAPE or not np.array_equal(
        transformed, prepared_binary
    ):
        raise GTMappingError(f"{sample_id}: PIL-nearest prepared transform differs")

    inverse = (
        np.asarray(
            Image.fromarray(prepared_binary.astype(np.uint8) * 255).resize(
                (ORIGINAL_MASK_SHAPE[1], ORIGINAL_MASK_SHAPE[0]),
                resample=Image.Resampling.NEAREST,
            )
        )
        != 0
    )
    intersection = int(np.count_nonzero(inverse & expected_original))
    union = int(np.count_nonzero(inverse | expected_original))
    round_trip_iou = intersection / union if union else 0.0
    if round_trip_iou < 0.95:
        raise GTMappingError(
            f"{sample_id}: PIL-nearest resize/inverse round-trip IoU "
            f"{round_trip_iou:.6f} is below 0.95"
        )

    labels, component_count = ndimage.label(expected_original)
    del labels
    rows, columns = np.nonzero(expected_original)
    foreground = int(rows.size)
    derived_root = derived_output_dir.expanduser().resolve()
    derived_root.mkdir(parents=True, exist_ok=True)
    derived_name = f"{__import__('hashlib').sha256(sample_id.encode()).hexdigest()}.png"
    derived_path = derived_root / derived_name
    temporary = derived_path.with_name(f".{derived_path.name}.{os.getpid()}.tmp")
    Image.fromarray(expected_original.astype(np.uint8) * 255, mode="L").save(
        temporary, format="PNG"
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    if derived_path.exists():
        if sha256_file(derived_path) != sha256_file(temporary):
            temporary.unlink()
            raise GTMappingError(f"{sample_id}: existing derived original GT differs")
        temporary.unlink()
    else:
        os.replace(temporary, derived_path)
    return {
        **dict(row),
        "mapping_status": "PASS",
        "pixel_qa_status": "P2_MAPPING_QA_PASS",
        "bulk_gt_pixels_read": True,
        "gt_pixel_access_purpose": "mapping_and_annotation_pixel_qa_only",
        "foreground_pixels": foreground,
        "foreground_fraction": foreground / float(expected_original.size),
        "component_count": int(component_count),
        "resize_inverse_round_trip_iou": round_trip_iou,
        "xy_orientation_evidence": {
            "source_instance_height_width": list(instance.shape),
            "rgb_declared_height_width": [rgb_height, rgb_width],
            "transposed_height_width": [instance.shape[1], instance.shape[0]],
            "transposed_shape_matches_rgb": (instance.shape[1], instance.shape[0])
            == (rgb_height, rgb_width),
            "status": "PASS",
        },
        "gt_mask_grasp_target_identity_evidence": {
            "gt_mask_target_instance_id": target,
            "gt_grasp_authority_target_instance_id": grasp_target,
            "gt_grasp_set_sha256": str(row.get("gt_grasp_set_sha256", "")),
            "authority_identity_match": True,
            "gt_grasp_rows_read": False,
            "status": "PASS",
        },
        "bbox": [
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        ],
        "original_gt_mask_path": str(derived_path),
        "original_gt_mask_sha256": sha256_file(derived_path),
    }


def join_real_authority_rows(
    prepared_rows: Iterable[Mapping[str, Any]],
    visual_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join the two real locked schemas without materialising GT grasp labels."""

    prepared = _index_unique(prepared_rows, key="sample_id", label="prepared label")
    visual = _index_unique(visual_rows, key="sample_id", label="visual paired")
    if set(prepared) != set(visual):
        raise GTMappingError("prepared and visual GT authority coverage differs")
    result = []
    for sample_id in sorted(prepared):
        left, right = prepared[sample_id], visual[sample_id]
        prepared_target = _strict_instance(left.get("target_object_id"), sample_id=sample_id)
        visual_target = _strict_instance(right.get("target_instance_id"), sample_id=sample_id)
        if prepared_target != visual_target:
            raise GTMappingError(f"{sample_id}: prepared/visual target identity differs")
        for field in ("scene_id", "question_index"):
            if str(left.get(field)) != str(right.get(field)):
                raise GTMappingError(f"{sample_id}: prepared/visual {field} differs")
        result.append(
            {
                "sample_id": sample_id,
                "scene_id": left["scene_id"],
                "question_index": left["question_index"],
                "target_object_id": prepared_target,
                "prepared_gt_mask_path": left["prepared_gt_mask_path"],
                "prepared_gt_mask_sha256": left["prepared_gt_mask_sha256"],
                "source_instance_mask_path": right["gt_mask_path"],
                "source_instance_mask_sha256": right["gt_mask_sha256"],
            }
        )
    return result


__all__ = [
    "EXPECTED_SAMPLE_COUNT",
    "GTAuthorityColumns",
    "GTMappingError",
    "ORIGINAL_MASK_SHAPE",
    "PREPARED_MASK_SHAPE",
    "build_prelock_registry",
    "mapping_pixel_qa",
    "join_real_authority_rows",
]
