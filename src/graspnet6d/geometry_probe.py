"""Resumable pre-contract VGN inference for formal geometry validation.

The normal candidate stage cannot run until the VGN-to-GraspNet gripper-frame
contract has passed.  Geometry validation, in turn, needs real VGN-frame
candidates.  This module breaks that dependency without guessing the frame
mapping: it publishes only deterministic raw :class:`VGNCandidate` records.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import candidate_pool_fingerprint
from .formal_validation import RAW_VGN_VALIDATION_SCHEMA
from .io import atomic_json, canonical_sha256, sha256_file
from .stages import (
    VGN_BUNDLE_SCHEMA,
    GroupFailure,
    GroupManifest,
    StageInputError,
    StageSummary,
    _PRE_NMS_COMPLETE,
    _PRE_NMS_TEST_INCOMPLETE,
    _a7_pre_nms_payload,
    _candidate_bundle_path,
    _convert_frozen_candidates,
    _decode_a7_pre_nms_contract,
    _load_tsdf_cache,
    _raise_failures,
    _record_failure,
    _require_regular_file,
    _slug,
    _validate_resumable_candidate_bundle,
    load_evaluator_geometry_contract,
    load_target_language_jsonl,
)
from .vgn import (
    DEFAULT_CHECKPOINT,
    ExtractionConfig,
    VGNExtractionSnapshot,
    VGNCandidate,
    extract_candidate_snapshot,
    load_frozen_vgn,
    pose_nms,
    run_vgn,
    validate_extraction_snapshot,
)


GROUNDING_CONDITIONS = frozenset(
    {
        "oracle_gt_mask",
        "hifics_zero_shot_mask",
        "hifics_adapted_mask",
    }
)
GEOMETRY_PROBE_STAGE = "geometry_probe"
GEOMETRY_PROBE_PROMOTION_STAGE = "geometry_probe_promotion"
RAW_VGN_EMPTY_OBSERVATION_SCHEMA = "graspnet6d_raw_vgn_empty_observation_v1"

_RAW_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "group_id",
        "native_rank",
        "native_score",
        "translation_local_m",
        "rotation_local_vgn",
        "translation_camera_m",
        "rotation_camera_vgn",
        "translation_table_m",
        "rotation_table_vgn",
        "width_m",
        "voxel_index",
        "gripper_frame",
    }
)
_VGN_GRIPPER_FRAME = "vgn_(+Z_approach,+Y_closing)"
_EMPTY_OBSERVATION_REASON = "deterministic_extractor_returned_no_candidates"


def _condition(value: str) -> str:
    condition = str(value).strip()
    if condition not in GROUNDING_CONDITIONS:
        raise StageInputError(f"unsupported grounding condition: {condition!r}")
    return condition


def _output_path(root: Path, condition: str, group_id: str) -> Path:
    return (
        root
        / "geometry_probe"
        / "raw_vgn_candidates"
        / condition
        / f"{_slug(group_id)}.json"
    )


def _empty_output_path(root: Path, condition: str, group_id: str) -> Path:
    return (
        root
        / "geometry_probe"
        / "empty_vgn_observations"
        / condition
        / f"{_slug(group_id)}.json"
    )


def _candidate_vector(
    value: Any, shape: tuple[int, ...], description: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise StageInputError(f"raw VGN candidate has invalid {description}")
    return array


def _candidate_rotation(value: Any, description: str) -> np.ndarray:
    rotation = _candidate_vector(value, (3, 3), description)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0):
        raise StageInputError(f"raw VGN candidate has a non-orthogonal {description}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0):
        raise StageInputError(f"raw VGN candidate has an improper {description}")
    return rotation


def _validate_raw_candidates(
    candidates: Sequence[VGNCandidate],
    *,
    group_id: str,
    config: ExtractionConfig,
) -> None:
    if not candidates:
        raise StageInputError(
            "VGN geometry probe produced an empty candidate pool; the group is "
            "ineligible for geometry validation"
        )
    if len(candidates) > config.frozen_top_k:
        raise StageInputError("VGN geometry probe exceeded frozen_top_k")
    identifiers = [str(item.candidate_id).strip() for item in candidates]
    if any(not identifier for identifier in identifiers):
        raise StageInputError("raw VGN candidate has a blank candidate_id")
    if len(identifiers) != len(set(identifiers)):
        raise StageInputError("VGN geometry probe returned duplicate candidate IDs")
    if any(item.group_id != group_id for item in candidates):
        raise StageInputError(
            "VGN geometry probe returned a candidate for another group"
        )
    ranks = [int(item.native_rank) for item in candidates]
    if any(rank < 1 for rank in ranks) or ranks != sorted(set(ranks)):
        raise StageInputError(
            "raw VGN candidates are not in deterministic native-rank order"
        )
    for item in candidates:
        if not np.isfinite(float(item.native_score)):
            raise StageInputError("raw VGN candidate has a non-finite native score")
        if not np.isfinite(float(item.width_m)) or float(item.width_m) <= 0:
            raise StageInputError("raw VGN candidate has a non-positive width")
        voxel_index = tuple(item.voxel_index)
        if len(voxel_index) != 3 or any(
            int(value) != value or int(value) < 0 or int(value) >= 40
            for value in voxel_index
        ):
            raise StageInputError("raw VGN candidate has an invalid voxel index")
        _candidate_vector(item.translation_local_m, (3,), "local translation")
        _candidate_rotation(item.rotation_local_vgn, "local rotation")
        if item.translation_camera_m is None or item.rotation_camera_vgn is None:
            raise StageInputError("raw VGN candidate lacks its camera-frame transform")
        if item.translation_table_m is None or item.rotation_table_vgn is None:
            raise StageInputError("raw VGN candidate lacks its table-frame transform")
        _candidate_vector(item.translation_camera_m, (3,), "camera translation")
        _candidate_rotation(item.rotation_camera_vgn, "camera rotation")
        _candidate_vector(item.translation_table_m, (3,), "table translation")
        _candidate_rotation(item.rotation_table_vgn, "table rotation")
    expected = pose_nms(candidates, config)[: config.frozen_top_k]
    if [item.candidate_id for item in expected] != identifiers:
        raise StageInputError(
            "raw VGN candidates do not equal deterministic pose-NMS/Top-K output"
        )


def _candidate_from_record(
    value: Any,
    *,
    group_id: str,
) -> VGNCandidate:
    if not isinstance(value, Mapping):
        raise StageInputError("raw VGN candidate record must be an object")
    record = dict(value)
    if set(record) != _RAW_CANDIDATE_FIELDS:
        raise StageInputError(
            "raw VGN candidate fields differ from the frozen VGNCandidate schema: "
            f"missing={sorted(_RAW_CANDIDATE_FIELDS - set(record))}, "
            f"extra={sorted(set(record) - _RAW_CANDIDATE_FIELDS)}"
        )
    if (
        not isinstance(record["candidate_id"], str)
        or not record["candidate_id"].strip()
    ):
        raise StageInputError("raw VGN candidate has an invalid candidate_id")
    if record["group_id"] != group_id:
        raise StageInputError("raw VGN candidate belongs to another group")
    if isinstance(record["native_rank"], bool) or not isinstance(
        record["native_rank"], int
    ):
        raise StageInputError("raw VGN candidate native_rank must be an integer")
    for field in ("native_score", "width_m"):
        if isinstance(record[field], bool) or not isinstance(
            record[field], (int, float)
        ):
            raise StageInputError(f"raw VGN candidate {field} must be numeric")
    if record["gripper_frame"] != _VGN_GRIPPER_FRAME:
        raise StageInputError("raw VGN candidate has an unexpected gripper frame")
    voxel = record["voxel_index"]
    if (
        not isinstance(voxel, list)
        or len(voxel) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in voxel)
    ):
        raise StageInputError("raw VGN candidate has a non-canonical voxel index")
    candidate = VGNCandidate(
        candidate_id=record["candidate_id"],
        group_id=group_id,
        native_rank=record["native_rank"],
        native_score=float(record["native_score"]),
        translation_local_m=_candidate_vector(
            record["translation_local_m"], (3,), "local translation"
        ),
        rotation_local_vgn=_candidate_rotation(
            record["rotation_local_vgn"], "local rotation"
        ),
        width_m=float(record["width_m"]),
        voxel_index=tuple(voxel),
        translation_camera_m=_candidate_vector(
            record["translation_camera_m"], (3,), "camera translation"
        ),
        rotation_camera_vgn=_candidate_rotation(
            record["rotation_camera_vgn"], "camera rotation"
        ),
        translation_table_m=_candidate_vector(
            record["translation_table_m"], (3,), "table translation"
        ),
        rotation_table_vgn=_candidate_rotation(
            record["rotation_table_vgn"], "table rotation"
        ),
        gripper_frame=record["gripper_frame"],
    )
    if canonical_sha256(candidate.to_record()) != canonical_sha256(record):
        raise StageInputError("raw VGN candidate record is not canonically serialized")
    return candidate


def _reconstruct_raw_candidates(
    records: Sequence[Any],
    *,
    group_id: str,
    config: ExtractionConfig,
    cache: Mapping[str, np.ndarray],
) -> list[VGNCandidate]:
    candidates = [
        _candidate_from_record(record, group_id=group_id) for record in records
    ]
    _validate_raw_candidates(candidates, group_id=group_id, config=config)
    for candidate in candidates:
        local_translation = np.asarray(candidate.translation_local_m, dtype=np.float64)
        local_rotation = np.asarray(candidate.rotation_local_vgn, dtype=np.float64)
        for transform_name, translation, rotation in (
            (
                "T_local_to_camera",
                candidate.translation_camera_m,
                candidate.rotation_camera_vgn,
            ),
            (
                "T_local_to_table",
                candidate.translation_table_m,
                candidate.rotation_table_vgn,
            ),
        ):
            transform = np.asarray(cache[transform_name], dtype=np.float64)
            expected_translation = (
                transform[:3, :3] @ local_translation + transform[:3, 3]
            )
            expected_rotation = transform[:3, :3] @ local_rotation
            if not np.allclose(
                np.asarray(translation, dtype=np.float64),
                expected_translation,
                atol=1e-10,
                rtol=0,
            ):
                raise StageInputError(
                    f"raw VGN candidate translation is stale for {transform_name}"
                )
            if not np.allclose(
                np.asarray(rotation, dtype=np.float64),
                expected_rotation,
                atol=1e-10,
                rtol=0,
            ):
                raise StageInputError(
                    f"raw VGN candidate rotation is stale for {transform_name}"
                )
    reconstructed_records = [candidate.to_record() for candidate in candidates]
    if canonical_sha256(reconstructed_records) != canonical_sha256(list(records)):
        raise StageInputError("raw VGN candidate pool changed during reconstruction")
    return candidates


def _input_fingerprint(
    *,
    group: GroupManifest,
    tsdf_path: Path,
    checkpoint_sha256: str,
    device: str,
    condition: str,
    mask_input_fingerprint: str,
    mask_commit_sha256: str,
    config: ExtractionConfig,
) -> str:
    return canonical_sha256(
        {
            "schema": RAW_VGN_VALIDATION_SCHEMA,
            "group_id": group.group_id,
            "target_manifest_record": dict(group.target),
            "language_manifest_record": dict(group.language),
            "tsdf_path": str(tsdf_path),
            "tsdf_sha256": sha256_file(tsdf_path),
            "checkpoint_sha256": checkpoint_sha256,
            "device": str(device),
            "grounding_condition": condition,
            "grounding_mask_input_fingerprint": mask_input_fingerprint,
            "grounding_mask_commit_sha256": mask_commit_sha256,
            "extraction_config": asdict(config),
        }
    )


def _read_bundle(path: Path, *, group_id: str | None = None) -> dict[str, Any]:
    source = _require_regular_file(path, "geometry-probe raw candidate bundle")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageInputError(
            f"invalid geometry-probe bundle {source}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise StageInputError(f"geometry-probe bundle is not a JSON object: {source}")
    if payload.get("schema_version") != RAW_VGN_VALIDATION_SCHEMA:
        raise StageInputError(f"geometry-probe bundle schema mismatch: {source}")
    if group_id is not None and payload.get("group_id") != group_id:
        raise StageInputError(
            f"geometry-probe bundle belongs to another group: {source}"
        )
    required = {
        "group_id",
        "tsdf_path",
        "tsdf_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "device",
        "grounding_condition",
        "grounding_mask_input_fingerprint",
        "grounding_mask_commit_sha256",
        "extraction_config",
        "inference_calls_for_group",
        "candidate_count",
        "raw_vgn_candidates",
        "raw_vgn_pool_fingerprint",
        "pre_nms_snapshot_status",
        "pre_nms_candidate_count",
        "pre_nms_vgn_candidates",
        "pre_nms_pool_fingerprint",
        "a7_top_k_membership",
        "a7_top_k_membership_fingerprint",
        "input_fingerprint",
        "bundle_fingerprint",
    }
    if not required.issubset(payload):
        raise StageInputError(
            f"geometry-probe bundle lacks fields: {sorted(required - set(payload))}"
        )
    for field in (
        "input_fingerprint",
        "tsdf_sha256",
        "checkpoint_sha256",
        "grounding_mask_input_fingerprint",
        "grounding_mask_commit_sha256",
        "raw_vgn_pool_fingerprint",
        "bundle_fingerprint",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload[field])):
            raise StageInputError(f"geometry-probe bundle has an invalid {field}")
    condition = _condition(str(payload["grounding_condition"]))
    if not str(payload["device"]).strip():
        raise StageInputError("geometry-probe bundle has a blank device")
    extraction_record = payload["extraction_config"]
    if not isinstance(extraction_record, Mapping):
        raise StageInputError("geometry-probe extraction_config must be an object")
    try:
        extraction = ExtractionConfig(**dict(extraction_record))
        extraction.validate()
    except (TypeError, ValueError) as error:
        raise StageInputError(
            f"geometry-probe extraction_config is invalid: {error}"
        ) from error
    if asdict(extraction) != dict(extraction_record):
        raise StageInputError("geometry-probe extraction_config is not canonical")
    if int(payload["inference_calls_for_group"]) != 1:
        raise StageInputError("geometry-probe bundle must record exactly one inference")
    raw = payload["raw_vgn_candidates"]
    if not isinstance(raw, list) or not raw:
        raise StageInputError("geometry-probe bundle has an empty raw candidate pool")
    if int(payload["candidate_count"]) != len(raw):
        raise StageInputError("geometry-probe candidate_count does not match its pool")
    if payload["raw_vgn_pool_fingerprint"] != canonical_sha256(raw):
        raise StageInputError("geometry-probe raw pool fingerprint mismatch")
    check = dict(payload)
    observed = check.pop("bundle_fingerprint")
    if observed != canonical_sha256(check):
        raise StageInputError("geometry-probe bundle fingerprint mismatch")
    tsdf_path = _require_regular_file(payload["tsdf_path"], "geometry-probe TSDF")
    checkpoint = _require_regular_file(
        payload["checkpoint_path"], "geometry-probe checkpoint"
    )
    if payload["tsdf_sha256"] != sha256_file(tsdf_path):
        raise StageInputError("geometry-probe TSDF hash is stale")
    if payload["checkpoint_sha256"] != sha256_file(checkpoint):
        raise StageInputError("geometry-probe checkpoint hash is stale")
    cache = _load_tsdf_cache(tsdf_path, str(payload["group_id"]))
    cached_condition = str(np.asarray(cache["grounding_condition"]).item())
    cached_mask_input = str(
        np.asarray(cache["grounding_mask_input_fingerprint"]).item()
    )
    cached_mask_commit = str(np.asarray(cache["grounding_mask_commit_sha256"]).item())
    if cached_condition != condition:
        raise StageInputError("geometry-probe condition disagrees with its TSDF cache")
    if cached_mask_input != payload["grounding_mask_input_fingerprint"]:
        raise StageInputError("geometry-probe mask input lineage is stale")
    if cached_mask_commit != payload["grounding_mask_commit_sha256"]:
        raise StageInputError("geometry-probe mask commit lineage is stale")
    candidate_ids = [
        str(record.get("candidate_id", "")).strip()
        for record in raw
        if isinstance(record, Mapping)
    ]
    if len(candidate_ids) != len(raw) or any(not value for value in candidate_ids):
        raise StageInputError("geometry-probe candidates lack non-empty candidate IDs")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise StageInputError("geometry-probe candidate IDs are duplicated")
    candidate_groups = [str(record.get("group_id", "")) for record in raw]
    if any(value != payload["group_id"] for value in candidate_groups):
        raise StageInputError("geometry-probe raw candidate belongs to another group")
    try:
        native_ranks = [int(record["native_rank"]) for record in raw]
    except (KeyError, TypeError, ValueError) as error:
        raise StageInputError(
            "geometry-probe candidates have invalid native ranks"
        ) from error
    if any(rank < 1 for rank in native_ranks) or native_ranks != sorted(
        set(native_ranks)
    ):
        raise StageInputError(
            "geometry-probe candidates are not in deterministic native-rank order"
        )
    frozen = tuple(
        _candidate_from_record(record, group_id=str(payload["group_id"]))
        for record in raw
    )
    _decode_a7_pre_nms_contract(
        payload,
        group_id=str(payload["group_id"]),
        frozen=frozen,
        config=extraction,
        allow_test_incomplete=True,
    )
    return payload


def _read_empty_observation(
    path: Path, *, group_id: str | None = None
) -> dict[str, Any]:
    source = _require_regular_file(path, "geometry-probe empty observation")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageInputError(
            f"invalid geometry-probe empty observation {source}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise StageInputError(
            f"geometry-probe empty observation is not a JSON object: {source}"
        )
    required = {
        "schema_version",
        "group_id",
        "input_fingerprint",
        "tsdf_path",
        "tsdf_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "device",
        "grounding_condition",
        "grounding_mask_input_fingerprint",
        "grounding_mask_commit_sha256",
        "extraction_config",
        "inference_calls_for_group",
        "candidate_count",
        "raw_vgn_candidates",
        "raw_vgn_pool_fingerprint",
        "pre_nms_snapshot_status",
        "pre_nms_candidate_count",
        "pre_nms_vgn_candidates",
        "pre_nms_pool_fingerprint",
        "a7_top_k_membership",
        "a7_top_k_membership_fingerprint",
        "empty_observation_reason",
        "bundle_fingerprint",
    }
    if set(payload) != required:
        raise StageInputError(
            "geometry-probe empty observation fields differ from its schema: "
            f"missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}"
        )
    if payload["schema_version"] != RAW_VGN_EMPTY_OBSERVATION_SCHEMA:
        raise StageInputError(
            f"geometry-probe empty observation schema mismatch: {source}"
        )
    if group_id is not None and payload["group_id"] != group_id:
        raise StageInputError(
            f"geometry-probe empty observation belongs to another group: {source}"
        )
    if not isinstance(payload["group_id"], str) or not payload["group_id"].strip():
        raise StageInputError("geometry-probe empty observation has a blank group_id")
    for field in (
        "input_fingerprint",
        "tsdf_sha256",
        "checkpoint_sha256",
        "grounding_mask_input_fingerprint",
        "grounding_mask_commit_sha256",
        "raw_vgn_pool_fingerprint",
        "bundle_fingerprint",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(payload[field])) is None:
            raise StageInputError(
                f"geometry-probe empty observation has an invalid {field}"
            )
    condition = _condition(str(payload["grounding_condition"]))
    if not isinstance(payload["device"], str) or not payload["device"].strip():
        raise StageInputError("geometry-probe empty observation has a blank device")
    extraction_record = payload["extraction_config"]
    if not isinstance(extraction_record, Mapping):
        raise StageInputError(
            "geometry-probe empty observation extraction_config must be an object"
        )
    try:
        extraction = ExtractionConfig(**dict(extraction_record))
        extraction.validate()
    except (TypeError, ValueError) as error:
        raise StageInputError(
            f"geometry-probe empty observation extraction_config is invalid: {error}"
        ) from error
    if asdict(extraction) != dict(extraction_record):
        raise StageInputError(
            "geometry-probe empty observation extraction_config is not canonical"
        )
    if (
        isinstance(payload["inference_calls_for_group"], bool)
        or payload["inference_calls_for_group"] != 1
    ):
        raise StageInputError(
            "geometry-probe empty observation must record exactly one inference"
        )
    if isinstance(payload["candidate_count"], bool) or payload["candidate_count"] != 0:
        raise StageInputError(
            "geometry-probe empty observation candidate_count must be zero"
        )
    if payload["raw_vgn_candidates"] != []:
        raise StageInputError(
            "geometry-probe empty observation must contain an empty raw pool"
        )
    if payload["raw_vgn_pool_fingerprint"] != canonical_sha256([]):
        raise StageInputError(
            "geometry-probe empty observation raw pool fingerprint mismatch"
        )
    if payload["empty_observation_reason"] != _EMPTY_OBSERVATION_REASON:
        raise StageInputError("geometry-probe empty observation has an unknown reason")
    _decode_a7_pre_nms_contract(
        payload,
        group_id=str(payload["group_id"]),
        frozen=(),
        config=extraction,
        allow_test_incomplete=True,
    )
    check = dict(payload)
    observed = check.pop("bundle_fingerprint")
    if observed != canonical_sha256(check):
        raise StageInputError(
            "geometry-probe empty observation bundle fingerprint mismatch"
        )
    tsdf_path = _require_regular_file(
        payload["tsdf_path"], "geometry-probe empty-observation TSDF"
    )
    checkpoint = _require_regular_file(
        payload["checkpoint_path"], "geometry-probe empty-observation checkpoint"
    )
    if payload["tsdf_sha256"] != sha256_file(tsdf_path):
        raise StageInputError("geometry-probe empty-observation TSDF hash is stale")
    if payload["checkpoint_sha256"] != sha256_file(checkpoint):
        raise StageInputError(
            "geometry-probe empty-observation checkpoint hash is stale"
        )
    cache = _load_tsdf_cache(tsdf_path, payload["group_id"])
    cached_condition = str(np.asarray(cache["grounding_condition"]).item())
    cached_mask_input = str(
        np.asarray(cache["grounding_mask_input_fingerprint"]).item()
    )
    cached_mask_commit = str(np.asarray(cache["grounding_mask_commit_sha256"]).item())
    if cached_condition != condition:
        raise StageInputError(
            "geometry-probe empty observation condition disagrees with its TSDF"
        )
    if cached_mask_input != payload["grounding_mask_input_fingerprint"]:
        raise StageInputError(
            "geometry-probe empty observation mask input lineage is stale"
        )
    if cached_mask_commit != payload["grounding_mask_commit_sha256"]:
        raise StageInputError(
            "geometry-probe empty observation mask commit lineage is stale"
        )
    return payload


def _probe_resume_hit(
    raw_output: Path,
    empty_output: Path,
    *,
    group_id: str,
    input_fingerprint: str,
    resume: bool,
) -> Path | None:
    existing = [path for path in (raw_output, empty_output) if path.exists()]
    if not existing:
        return None
    if len(existing) != 1:
        raise StageInputError(
            f"geometry probe has both nonempty and empty observations for {group_id!r}"
        )
    if not resume:
        raise StageInputError(
            f"geometry-probe output already exists: {existing[0]}; use resume=True"
        )
    output = existing[0]
    payload = (
        _read_bundle(output, group_id=group_id)
        if output == raw_output
        else _read_empty_observation(output, group_id=group_id)
    )
    if payload["input_fingerprint"] != input_fingerprint:
        raise StageInputError(f"stale geometry-probe input fingerprint: {output}")
    return output


def run_geometry_probe_stage(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    output_root: Path | str,
    *,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    grounding_condition: str = "oracle_gt_mask",
    config: ExtractionConfig = ExtractionConfig(),
    resume: bool = False,
    model_loader: Callable[..., Any] = load_frozen_vgn,
    inference: Callable[..., Any] = run_vgn,
    extractor: Callable[
        ..., VGNExtractionSnapshot | Sequence[VGNCandidate]
    ] = extract_candidate_snapshot,
) -> StageSummary:
    """Publish nonempty raw VGN candidates before any evaluator conversion.

    The checkpoint is loaded once when at least one group is pending.  Each
    pending group makes exactly one call to ``inference`` and one deterministic
    extraction call.  Empty pools are recorded as group failures and are never
    materialized as successful bundles.
    """

    root = Path(output_root).expanduser().resolve()
    groups = load_target_language_jsonl(target_manifest_path, language_manifest_path)
    failures: list[GroupFailure] = []
    try:
        condition = _condition(grounding_condition)
        checkpoint_path = _require_regular_file(checkpoint, "VGN checkpoint")
        checkpoint_digest = sha256_file(checkpoint_path)
        device_name = str(device).strip()
        if not device_name:
            raise StageInputError("VGN device must be non-empty")
        config.validate()
    except Exception as error:
        for group in groups:
            failures.append(
                _record_failure(
                    root,
                    stage=GEOMETRY_PROBE_STAGE,
                    group_id=group.group_id,
                    error=error,
                )
            )
        _raise_failures(GEOMETRY_PROBE_STAGE, failures)
        raise AssertionError("unreachable")

    pending: list[tuple[GroupManifest, Path, dict[str, np.ndarray], str, str, str]] = []
    outputs: list[str] = []
    resumed = 0
    for group in groups:
        try:
            tsdf_path = _require_regular_file(
                root / "target_tsdf" / condition / f"{_slug(group.group_id)}.npz",
                "condition-specific committed TSDF cache",
            )
            cache = _load_tsdf_cache(tsdf_path, group.group_id)
            cached_condition = str(np.asarray(cache["grounding_condition"]).item())
            if cached_condition != condition:
                raise StageInputError(
                    "TSDF cache belongs to another grounding condition"
                )
            mask_input = str(
                np.asarray(cache["grounding_mask_input_fingerprint"]).item()
            )
            mask_commit = str(np.asarray(cache["grounding_mask_commit_sha256"]).item())
            fingerprint = _input_fingerprint(
                group=group,
                tsdf_path=tsdf_path,
                checkpoint_sha256=checkpoint_digest,
                device=device_name,
                condition=condition,
                mask_input_fingerprint=mask_input,
                mask_commit_sha256=mask_commit,
                config=config,
            )
            raw_output = _output_path(root, condition, group.group_id)
            empty_output = _empty_output_path(root, condition, group.group_id)
            resumed_output = _probe_resume_hit(
                raw_output,
                empty_output,
                group_id=group.group_id,
                input_fingerprint=fingerprint,
                resume=resume,
            )
            if resumed_output is not None:
                resumed += 1
                outputs.append(str(resumed_output))
                continue
            pending.append(
                (group, tsdf_path, cache, fingerprint, mask_input, mask_commit)
            )
        except Exception as error:
            failures.append(
                _record_failure(
                    root,
                    stage=GEOMETRY_PROBE_STAGE,
                    group_id=group.group_id,
                    error=error,
                )
            )
    _raise_failures(GEOMETRY_PROBE_STAGE, failures)

    model: Any | None = None
    if pending:
        try:
            model = model_loader(checkpoint=checkpoint_path, device=device_name)
        except Exception as error:
            for group, *_ in pending:
                failures.append(
                    _record_failure(
                        root,
                        stage=GEOMETRY_PROBE_STAGE,
                        group_id=group.group_id,
                        error=error,
                    )
                )
            _raise_failures(GEOMETRY_PROBE_STAGE, failures)

    for group, tsdf_path, cache, fingerprint, mask_input, mask_commit in pending:
        try:
            raw_outputs = inference(cache["tsdf"], model, device=device_name)
            extraction_result = extractor(
                cache["tsdf"],
                raw_outputs,
                group_id=group.group_id,
                config=config,
                T_local_to_camera=cache["T_local_to_camera"],
                T_local_to_table=cache["T_local_to_table"],
            )
            if isinstance(extraction_result, VGNExtractionSnapshot):
                snapshot = validate_extraction_snapshot(
                    extraction_result, group_id=group.group_id, config=config
                )
                pre_nms = list(snapshot.pre_nms_candidates)
                frozen = list(snapshot.frozen_candidates)
                pre_nms_status = _PRE_NMS_COMPLETE
            else:
                # Sequence injection is retained solely for fixture-scoped
                # tests.  Formal promotion below rejects this explicit marker.
                pre_nms = []
                frozen = list(extraction_result)
                pre_nms_status = _PRE_NMS_TEST_INCOMPLETE
            raw_records = [item.to_record() for item in frozen]
            if frozen:
                _validate_raw_candidates(frozen, group_id=group.group_id, config=config)
                schema = RAW_VGN_VALIDATION_SCHEMA
                output = _output_path(root, condition, group.group_id)
                extra: dict[str, Any] = {}
            else:
                schema = RAW_VGN_EMPTY_OBSERVATION_SCHEMA
                output = _empty_output_path(root, condition, group.group_id)
                extra = {"empty_observation_reason": _EMPTY_OBSERVATION_REASON}
            payload: dict[str, Any] = {
                "schema_version": schema,
                "group_id": group.group_id,
                "input_fingerprint": fingerprint,
                "tsdf_path": str(tsdf_path),
                "tsdf_sha256": sha256_file(tsdf_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_digest,
                "device": device_name,
                "grounding_condition": condition,
                "grounding_mask_input_fingerprint": mask_input,
                "grounding_mask_commit_sha256": mask_commit,
                "extraction_config": asdict(config),
                "inference_calls_for_group": 1,
                "candidate_count": len(frozen),
                "raw_vgn_candidates": raw_records,
                "raw_vgn_pool_fingerprint": canonical_sha256(raw_records),
                **_a7_pre_nms_payload(
                    pre_nms,
                    frozen,
                    config=config,
                    status=pre_nms_status,
                ),
                **extra,
            }
            payload["bundle_fingerprint"] = canonical_sha256(payload)
            atomic_json(output, payload)
            outputs.append(str(output))
        except Exception as error:
            failures.append(
                _record_failure(
                    root,
                    stage=GEOMETRY_PROBE_STAGE,
                    group_id=group.group_id,
                    error=error,
                )
            )
    _raise_failures(GEOMETRY_PROBE_STAGE, failures)
    return StageSummary(
        GEOMETRY_PROBE_STAGE,
        len(groups),
        len(groups) - resumed,
        resumed,
        tuple(outputs),
    )


def promote_geometry_probe_bundles(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    output_root: Path | str,
    *,
    geometry_contract_path: Path | str,
    selected_group_ids: Sequence[str],
    checkpoint: Path | str,
    device: str,
    grounding_condition: str,
    config: ExtractionConfig,
    evidence_policy: str = "formal",
) -> StageSummary:
    """Convert validated raw probe bundles without running VGN a second time.

    Promotion is deliberately narrower than :func:`run_vgn_candidate_stage`:
    every selected group must already have one intact raw-probe bundle bound to
    the exact manifest records, TSDF, checkpoint, device, condition, mask
    lineage, and extraction config supplied here.  The published payload is the
    standard VGN candidate bundle, including the exact input fingerprint used
    by the normal stage's resume validator.

    ``evidence_policy`` defaults to ``"formal"``.  ``"test"`` exists solely so
    unit tests can use an explicitly marked fixture evidence artifact; it is
    checked by the same geometry-contract loader as the normal VGN stage.
    """

    root = Path(output_root).expanduser().resolve()
    groups = load_target_language_jsonl(target_manifest_path, language_manifest_path)
    selected = tuple(str(group_id).strip() for group_id in selected_group_ids)
    if not selected or any(not group_id for group_id in selected):
        raise StageInputError("selected geometry-probe group IDs must be non-empty")
    if len(selected) != len(set(selected)):
        raise StageInputError("selected geometry-probe group IDs must be unique")
    by_group = {group.group_id: group for group in groups}
    missing = sorted(set(selected) - set(by_group))
    if missing:
        raise StageInputError(
            f"selected geometry-probe groups are absent from the manifests: {missing}"
        )
    chosen = tuple(by_group[group_id] for group_id in selected)

    failures: list[GroupFailure] = []
    try:
        condition = _condition(grounding_condition)
        checkpoint_path = _require_regular_file(checkpoint, "VGN checkpoint")
        checkpoint_digest = sha256_file(checkpoint_path)
        device_name = str(device)
        if not device_name.strip() or device_name != device_name.strip():
            raise StageInputError("VGN device must be non-empty and canonical")
        config.validate()
        if evidence_policy == "formal" and (
            config.pre_nms_max_candidates != 100 or config.frozen_top_k != 50
        ):
            raise StageInputError(
                "formal A7 requires pre_nms_max_candidates=100 and frozen_top_k=50"
            )
        geometry, geometry_evidence = load_evaluator_geometry_contract(
            geometry_contract_path, evidence_policy=evidence_policy
        )
    except Exception as error:
        for group in chosen:
            failures.append(
                _record_failure(
                    root,
                    stage=GEOMETRY_PROBE_PROMOTION_STAGE,
                    group_id=group.group_id,
                    error=error,
                )
            )
        _raise_failures(GEOMETRY_PROBE_PROMOTION_STAGE, failures)
        raise AssertionError("unreachable")

    prepared: list[tuple[Path, dict[str, Any]]] = []
    outputs: list[str] = []
    resumed = 0
    for group in chosen:
        try:
            raw_path = _output_path(root, condition, group.group_id)
            empty_path = _empty_output_path(root, condition, group.group_id)
            observed_paths = [path for path in (raw_path, empty_path) if path.exists()]
            if len(observed_paths) != 1:
                raise StageInputError(
                    "promotion requires exactly one nonempty or empty geometry-probe "
                    f"observation for {group.group_id!r}, got {len(observed_paths)}"
                )
            is_empty = observed_paths[0] == empty_path
            raw_payload = (
                _read_empty_observation(empty_path, group_id=group.group_id)
                if is_empty
                else _read_bundle(raw_path, group_id=group.group_id)
            )
            expected_tsdf = (
                root / "target_tsdf" / condition / f"{_slug(group.group_id)}.npz"
            ).resolve()
            recorded_tsdf = Path(str(raw_payload["tsdf_path"])).expanduser().resolve()
            if recorded_tsdf != expected_tsdf:
                raise StageInputError(
                    "geometry-probe bundle is bound to an unexpected TSDF path"
                )
            cache = _load_tsdf_cache(expected_tsdf, group.group_id)
            mask_input = str(
                np.asarray(cache["grounding_mask_input_fingerprint"]).item()
            )
            mask_commit = str(np.asarray(cache["grounding_mask_commit_sha256"]).item())
            if raw_payload["grounding_condition"] != condition:
                raise StageInputError(
                    "geometry-probe bundle belongs to another condition"
                )
            if (
                Path(str(raw_payload["checkpoint_path"])).expanduser().resolve()
                != checkpoint_path
            ):
                raise StageInputError(
                    "geometry-probe bundle is bound to another VGN checkpoint"
                )
            if raw_payload["checkpoint_sha256"] != checkpoint_digest:
                raise StageInputError("geometry-probe VGN checkpoint hash is stale")
            if raw_payload["device"] != device_name:
                raise StageInputError(
                    "geometry-probe bundle was created on another device"
                )
            if raw_payload["extraction_config"] != asdict(config):
                raise StageInputError(
                    "geometry-probe extraction config differs from promotion config"
                )
            expected_raw_input = _input_fingerprint(
                group=group,
                tsdf_path=expected_tsdf,
                checkpoint_sha256=checkpoint_digest,
                device=device_name,
                condition=condition,
                mask_input_fingerprint=mask_input,
                mask_commit_sha256=mask_commit,
                config=config,
            )
            if raw_payload["input_fingerprint"] != expected_raw_input:
                raise StageInputError(
                    "geometry-probe input fingerprint is stale for the manifests"
                )
            raw_values = raw_payload["raw_vgn_candidates"]
            frozen = (
                []
                if is_empty
                else _reconstruct_raw_candidates(
                    raw_values,
                    group_id=group.group_id,
                    config=config,
                    cache=cache,
                )
            )
            raw_records = [candidate.to_record() for candidate in frozen]
            if raw_payload["raw_vgn_pool_fingerprint"] != canonical_sha256(raw_records):
                raise StageInputError("geometry-probe raw candidate pool is stale")
            pre_nms_status = str(raw_payload["pre_nms_snapshot_status"])
            if evidence_policy == "formal" and pre_nms_status != _PRE_NMS_COMPLETE:
                raise StageInputError(
                    "formal promotion requires a complete same-inference pre-NMS snapshot"
                )
            pre_nms = _decode_a7_pre_nms_contract(
                raw_payload,
                group_id=group.group_id,
                frozen=frozen,
                config=config,
                allow_test_incomplete=evidence_policy == "test",
            )

            standard_input = canonical_sha256(
                {
                    "schema": VGN_BUNDLE_SCHEMA,
                    "group_id": group.group_id,
                    "tsdf_sha256": sha256_file(expected_tsdf),
                    "checkpoint_sha256": checkpoint_digest,
                    "device": device_name,
                    "extraction_config": asdict(config),
                    "geometry": geometry_evidence,
                    "grounding_condition": condition,
                    "grounding_mask_input_fingerprint": mask_input,
                    "grounding_mask_commit_sha256": mask_commit,
                }
            )
            candidates, rows = _convert_frozen_candidates(
                frozen, geometry, geometry_evidence
            )
            payload: dict[str, Any] = {
                "schema_version": VGN_BUNDLE_SCHEMA,
                "group_id": group.group_id,
                "input_fingerprint": standard_input,
                "tsdf_path": str(expected_tsdf),
                "tsdf_sha256": sha256_file(expected_tsdf),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_digest,
                "device": device_name,
                "grounding_condition": condition,
                "grounding_mask_input_fingerprint": mask_input,
                "grounding_mask_commit_sha256": mask_commit,
                "extraction_config": asdict(config),
                "generation_status": "completed_vgn_inference",
                "inference_calls_for_group": 1,
                "candidate_count": len(frozen),
                "raw_vgn_candidates": raw_records,
                "raw_vgn_pool_fingerprint": canonical_sha256(raw_records),
                "candidate_records": [candidate.to_dict() for candidate in candidates],
                "candidate_pool_fingerprint": candidate_pool_fingerprint(candidates),
                "graspnet_rows": rows,
                "geometry_contract": dict(geometry_evidence),
                **_a7_pre_nms_payload(
                    pre_nms,
                    frozen,
                    config=config,
                    status=pre_nms_status,
                ),
            }
            payload["bundle_fingerprint"] = canonical_sha256(payload)
            output = _candidate_bundle_path(root, group.group_id, condition)
            if output.exists():
                if not _validate_resumable_candidate_bundle(
                    output,
                    group_id=group.group_id,
                    input_fingerprint=standard_input,
                    resume=True,
                ):
                    raise AssertionError("existing bundle validation returned false")
                existing = _read_bundle_json(output)
                if canonical_sha256(existing) != canonical_sha256(payload):
                    raise StageInputError(
                        "existing candidate bundle differs from the exact promotion output"
                    )
                resumed += 1
                outputs.append(str(output))
            else:
                prepared.append((output, payload))
        except Exception as error:
            failures.append(
                _record_failure(
                    root,
                    stage=GEOMETRY_PROBE_PROMOTION_STAGE,
                    group_id=group.group_id,
                    error=error,
                )
            )
    _raise_failures(GEOMETRY_PROBE_PROMOTION_STAGE, failures)

    for output, payload in prepared:
        atomic_json(output, payload)
        outputs.append(str(output))
    return StageSummary(
        GEOMETRY_PROBE_PROMOTION_STAGE,
        len(chosen),
        len(prepared),
        resumed,
        tuple(outputs),
    )


def _read_bundle_json(path: Path) -> dict[str, Any]:
    """Read an already validated standard bundle for byte-semantic comparison."""

    source = _require_regular_file(path, "promoted VGN candidate bundle")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageInputError(
            f"invalid promoted VGN candidate bundle: {source}"
        ) from error
    if not isinstance(payload, dict):
        raise StageInputError("promoted VGN candidate bundle must be a JSON object")
    return payload


def discover_geometry_probe_group_ids(
    output_root: Path | str,
    *,
    grounding_condition: str,
) -> tuple[str, ...]:
    """Discover every validated probe observation, including empty pools."""

    root = Path(output_root).expanduser().resolve()
    condition = _condition(grounding_condition)
    observations: dict[str, Path] = {}
    sources = (
        (
            root / "geometry_probe" / "raw_vgn_candidates" / condition,
            _read_bundle,
            _output_path,
        ),
        (
            root / "geometry_probe" / "empty_vgn_observations" / condition,
            _read_empty_observation,
            _empty_output_path,
        ),
    )
    for directory, loader, expected_path in sources:
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise StageInputError(
                f"geometry-probe observation directory is invalid: {directory}"
            )
        for path in sorted(directory.glob("*.json")):
            payload = loader(path)
            group_id = str(payload["group_id"])
            if payload["grounding_condition"] != condition:
                raise StageInputError(
                    f"geometry-probe observation belongs to another condition: {path}"
                )
            if path.resolve() != expected_path(root, condition, group_id).resolve():
                raise StageInputError(
                    f"geometry-probe observation has a non-canonical path: {path}"
                )
            if group_id in observations:
                raise StageInputError(
                    "geometry probe has multiple observations for group "
                    f"{group_id!r}: {observations[group_id]}, {path}"
                )
            observations[group_id] = path.resolve()
    if not observations:
        raise StageInputError(
            f"no geometry-probe observations exist for condition {condition!r}"
        )
    return tuple(sorted(observations))


def select_nonempty_geometry_probe_paths(
    output_root: Path | str,
    selected_group_ids: Sequence[str],
    *,
    grounding_condition: str = "oracle_gt_mask",
    minimum_groups: int = 20,
) -> dict[str, Path]:
    """Return validated raw-bundle paths for an explicit geometry sample.

    Formal calls retain the default minimum of twenty groups.  Unit tests may
    lower it explicitly, while a zero or negative minimum is always invalid.
    """

    if minimum_groups <= 0:
        raise ValueError("minimum_groups must be positive")
    selected = tuple(str(group_id).strip() for group_id in selected_group_ids)
    if any(not group_id for group_id in selected) or len(set(selected)) != len(
        selected
    ):
        raise StageInputError(
            "selected geometry-probe group IDs must be non-empty and unique"
        )
    if len(selected) < minimum_groups:
        raise StageInputError(
            f"geometry probe requires at least {minimum_groups} selected nonempty "
            f"groups, got {len(selected)}"
        )
    root = Path(output_root).expanduser().resolve()
    condition = _condition(grounding_condition)
    result: dict[str, Path] = {}
    for group_id in selected:
        path = _output_path(root, condition, group_id)
        empty_path = _empty_output_path(root, condition, group_id)
        if empty_path.exists():
            _read_empty_observation(empty_path, group_id=group_id)
            if path.exists():
                raise StageInputError(
                    f"geometry probe has both empty and nonempty outputs for {group_id!r}"
                )
            raise StageInputError(
                f"geometry-probe group {group_id!r} has an empty candidate pool and "
                "is ineligible for geometry audit selection"
            )
        payload = _read_bundle(path, group_id=group_id)
        if payload["grounding_condition"] != condition:
            raise StageInputError(
                f"geometry-probe bundle belongs to another condition: {path}"
            )
        expected_tsdf = (
            root / "target_tsdf" / condition / f"{_slug(group_id)}.npz"
        ).resolve()
        if Path(payload["tsdf_path"]).expanduser().resolve() != expected_tsdf:
            raise StageInputError(
                f"geometry-probe bundle is bound to an unexpected TSDF path: {path}"
            )
        result[group_id] = path.resolve()
    return result


__all__ = [
    "GEOMETRY_PROBE_STAGE",
    "GEOMETRY_PROBE_PROMOTION_STAGE",
    "GROUNDING_CONDITIONS",
    "RAW_VGN_EMPTY_OBSERVATION_SCHEMA",
    "discover_geometry_probe_group_ids",
    "promote_geometry_probe_bundles",
    "run_geometry_probe_stage",
    "select_nonempty_geometry_probe_paths",
]
