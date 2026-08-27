"""Fail-closed, resumable orchestration for the real GraspNet 6-DoF stages.

The functions in this module deliberately operate on one target group at a
time.  A group output is published atomically and contains an input
fingerprint; ``resume=True`` therefore skips only an output whose complete
input contract still matches.  Missing source data is an error, never an empty
or placeholder experiment table.

Unit tests may inject small deterministic builders/inference functions.  The
defaults always call the real TSDF, frozen-VGN, and official low-level
GraspNet evaluator adapters.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import Candidate6D, candidate_pool_fingerprint
from .evaluator import (
    DEFAULT_GRASPNET_API_ROOT,
    EvaluatorParityGate,
    ensure_graspnetapi_source,
    evaluate_frozen_candidates,
    relevance_from_friction,
)
from .features import StableMissingValueImputer, assert_no_gt_leakage
from .io import (
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    canonical_sha256,
    sha256_file,
)
from .metrics import evaluate_target_rankings
from .ranker import ValidationData, contiguous_group_sizes, fit_ranker
from .splits import SceneSplit
from .tsdf import TSDFBuildResult, build_target_centered_tsdf
from .vgn import (
    DEFAULT_CHECKPOINT,
    EvaluatorGeometryContract,
    ExtractionConfig,
    VGNExtractionSnapshot,
    VGNCandidate,
    candidate_to_graspnet_row,
    extract_candidate_snapshot,
    load_frozen_vgn,
    pose_nms,
    run_vgn,
    validate_extraction_snapshot,
    vgn_candidate_from_record,
)


MASK_SCHEMA = "graspnet6d_oracle_mask_v1"
TSDF_SCHEMA = "graspnet6d_target_tsdf_v1"
GROUNDING_TERMINAL_SCHEMA = "graspnet6d_grounding_terminal_v1"
VGN_BUNDLE_SCHEMA = "graspnet6d_frozen_vgn_candidates_v2"
LABEL_BUNDLE_SCHEMA = "graspnet6d_official_candidate_labels_v1"
GEOMETRY_SCHEMA = "graspnet6d_evaluator_geometry_v1"
PARITY_SCHEMA = "graspnet6d_evaluator_parity_v1"
GEOMETRY_EVIDENCE_SCHEMA = "graspnet6d_geometry_validation_evidence_v1"
PARITY_EVIDENCE_SCHEMA = "graspnet6d_evaluator_parity_evidence_v1"

_A7_TOP_K_VALUES = (20, 50, 100)
_PRE_NMS_COMPLETE = "complete_same_inference"
_PRE_NMS_TEST_INCOMPLETE = "test_injected_post_nms_only"
_PRE_NMS_GROUNDING_TERMINAL = "not_applicable_grounding_failure"

# JSON round-trips preserve the binary64 values emitted by the conversion
# stage.  Keep these tolerances deliberately much tighter than any physical
# evaluation threshold: they tolerate only harmless floating-point
# reconstruction noise, not a changed pose or gripper geometry.
_CONVERSION_SCALAR_ATOL = 1e-12
_CONVERSION_TRANSLATION_ATOL_M = 1e-12
_CONVERSION_ROTATION_ATOL = 1e-12
_EVALUATOR_ROW_ATOL = 1e-12


class StageInputError(RuntimeError):
    """A required real input is absent, stale, or structurally invalid."""


@dataclass(frozen=True, slots=True)
class GroupFailure:
    stage: str
    group_id: str
    error_type: str
    message: str
    error_path: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class StageBatchError(RuntimeError):
    """One or more group operations failed after their errors were recorded."""

    def __init__(self, stage: str, failures: Sequence[GroupFailure]):
        self.stage = str(stage)
        self.failures = tuple(failures)
        identities = ", ".join(item.group_id for item in self.failures[:5])
        suffix = "" if len(self.failures) <= 5 else ", ..."
        super().__init__(
            f"{self.stage} failed for {len(self.failures)} group(s): {identities}{suffix}"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": "FAILED",
            "failure_count": len(self.failures),
            "failures": [item.to_record() for item in self.failures],
        }


@dataclass(frozen=True, slots=True)
class StageSummary:
    stage: str
    total_groups: int
    completed_groups: int
    resumed_groups: int
    output_paths: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "status": "COMPLETE",
            "output_paths": list(self.output_paths),
        }


@dataclass(frozen=True, slots=True)
class GroupManifest:
    group_id: str
    target: Mapping[str, Any]
    language: Mapping[str, Any]


def _require_regular_file(path: Path | str, description: str) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_symlink() or not value.is_file():
        raise StageInputError(f"missing regular {description}: {value}")
    return value


def _read_json(path: Path | str, description: str) -> dict[str, Any]:
    source = _require_regular_file(path, description)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageInputError(f"invalid {description}: {source}: {error}") from error
    if not isinstance(payload, dict):
        raise StageInputError(f"{description} must be a JSON object: {source}")
    return payload


def load_jsonl_records(
    path: Path | str, *, description: str
) -> tuple[dict[str, Any], ...]:
    """Load a non-empty JSONL file without accepting malformed/blank records."""

    source = _require_regular_file(path, description)
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise StageInputError(
                f"invalid JSON in {description} at {source}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise StageInputError(
                f"{description} record at {source}:{line_number} is not an object"
            )
        records.append(record)
    if not records:
        raise StageInputError(
            f"{description} is empty; refusing a placeholder stage: {source}"
        )
    return tuple(records)


def load_target_language_jsonl(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
) -> tuple[GroupManifest, ...]:
    """Load and one-to-one join target and derived-language manifests."""

    targets = load_jsonl_records(target_manifest_path, description="target manifest")
    languages = load_jsonl_records(
        language_manifest_path, description="language manifest"
    )

    def index(
        records: Sequence[Mapping[str, Any]], name: str
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for record in records:
            group_id = str(record.get("group_id", "")).strip()
            if not group_id:
                raise StageInputError(f"{name} record lacks a non-empty group_id")
            if group_id in result:
                raise StageInputError(f"duplicate group_id {group_id!r} in {name}")
            result[group_id] = record
        return result

    by_target = index(targets, "target manifest")
    by_language = index(languages, "language manifest")
    if set(by_target) != set(by_language):
        missing_language = sorted(set(by_target) - set(by_language))
        missing_target = sorted(set(by_language) - set(by_target))
        raise StageInputError(
            "target/language group universes differ: "
            f"missing_language={missing_language[:5]}, missing_target={missing_target[:5]}"
        )
    required_target = {
        "scene_id",
        "camera",
        "frame_id",
        "target_object_id",
        "target_instance_label",
        "depth_path",
        "instance_label_path",
        "meta_path",
        "intrinsics_path",
        "camera_pose_path",
        "table_transform_path",
    }
    groups: list[GroupManifest] = []
    for group_id in sorted(by_target):
        target = by_target[group_id]
        language = by_language[group_id]
        missing = sorted(required_target - set(target))
        if missing:
            raise StageInputError(f"target group {group_id!r} lacks fields: {missing}")
        query = str(language.get("query", "")).strip()
        resolver = language.get("resolver_result")
        if not query or language.get("is_unique") is not True:
            raise StageInputError(
                f"language group {group_id!r} is not a unique real query"
            )
        if not isinstance(resolver, list) or len(resolver) != 1:
            raise StageInputError(
                f"language group {group_id!r} lacks a unique resolver result"
            )
        if int(resolver[0]) != int(target["target_object_id"]):
            raise StageInputError(
                f"language group {group_id!r} resolves to the wrong target"
            )
        groups.append(GroupManifest(group_id, dict(target), dict(language)))
    return tuple(groups)


def _slug(group_id: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(group_id)).strip("._") or "group"
    return f"{prefix[:96]}-{canonical_sha256(str(group_id))[:12]}"


def _record_failure(
    output_root: Path,
    *,
    stage: str,
    group_id: str,
    error: BaseException,
) -> GroupFailure:
    path = output_root / "errors" / stage / f"{_slug(group_id)}.json"
    payload = {
        "schema_version": 1,
        "stage": stage,
        "group_id": group_id,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(path, payload)
    return GroupFailure(stage, group_id, type(error).__name__, str(error), str(path))


def _raise_failures(stage: str, failures: Sequence[GroupFailure]) -> None:
    if failures:
        raise StageBatchError(stage, failures)


def _npz_resume_hit(
    output: Path,
    sidecar: Path,
    *,
    schema: str,
    input_fingerprint: str,
    resume: bool,
    sidecar_payload: Mapping[str, Any],
) -> bool:
    if not resume and (output.exists() or sidecar.exists()):
        raise StageInputError(
            f"output already exists for a non-resume run: {output}; use resume=True"
        )
    if not output.exists() and not sidecar.exists():
        return False
    if sidecar.exists() and not output.exists():
        raise StageInputError(f"committed sidecar has no data archive: {sidecar}")
    try:
        with np.load(output, allow_pickle=False) as archive:
            embedded = str(np.asarray(archive["input_fingerprint"]).item())
    except (OSError, KeyError, ValueError) as error:
        raise StageInputError(
            f"invalid resumable NPZ archive {output}: {error}"
        ) from error
    if embedded != input_fingerprint:
        raise StageInputError(f"stale output input fingerprint for {output}")
    if sidecar.exists():
        manifest = _read_json(sidecar, "stage sidecar")
        if manifest.get("schema_version") != schema:
            raise StageInputError(f"stage sidecar schema mismatch: {sidecar}")
        if manifest.get("input_fingerprint") != input_fingerprint:
            raise StageInputError(
                f"stage sidecar input fingerprint mismatch: {sidecar}"
            )
        if manifest.get("output_sha256") != sha256_file(output):
            raise StageInputError(f"stage output hash mismatch: {output}")
    else:
        # The NPZ was atomically published but the process stopped before the
        # commit sidecar.  Finalise it without recomputing the expensive stage.
        atomic_json(
            sidecar,
            {
                **dict(sidecar_payload),
                "schema_version": schema,
                "input_fingerprint": input_fingerprint,
                "output_sha256": sha256_file(output),
                "resumed_uncommitted_atomic_output": True,
            },
        )
    return True


def _write_npz_commit(
    output: Path,
    sidecar: Path,
    *,
    schema: str,
    input_fingerprint: str,
    arrays: Mapping[str, Any],
    sidecar_payload: Mapping[str, Any],
) -> None:
    atomic_npz(output, **dict(arrays), input_fingerprint=np.asarray(input_fingerprint))
    atomic_json(
        sidecar,
        {
            **dict(sidecar_payload),
            "schema_version": schema,
            "input_fingerprint": input_fingerprint,
            "output_sha256": sha256_file(output),
        },
    )


def run_oracle_mask_stage(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    output_root: Path | str,
    *,
    resume: bool = False,
) -> StageSummary:
    """Create one boolean oracle mask from each official instance-label image."""

    from PIL import Image

    stage = "oracle_masks"
    root = Path(output_root).expanduser().resolve()
    groups = load_target_language_jsonl(target_manifest_path, language_manifest_path)
    outputs: list[str] = []
    resumed = 0
    failures: list[GroupFailure] = []
    for group in groups:
        try:
            label_path = _require_regular_file(
                group.target["instance_label_path"], "instance-label image"
            )
            instance_label = int(group.target["target_instance_label"])
            if instance_label <= 0:
                raise StageInputError("target_instance_label must be positive")
            input_fingerprint = canonical_sha256(
                {
                    "schema": MASK_SCHEMA,
                    "group_id": group.group_id,
                    "instance_label": instance_label,
                    "instance_label_sha256": sha256_file(label_path),
                }
            )
            output = root / stage / f"{_slug(group.group_id)}.npz"
            sidecar = output.with_suffix(".json")
            base = {
                "group_id": group.group_id,
                "source_instance_label_path": str(label_path),
                "target_instance_label": instance_label,
            }
            if _npz_resume_hit(
                output,
                sidecar,
                schema=MASK_SCHEMA,
                input_fingerprint=input_fingerprint,
                resume=resume,
                sidecar_payload=base,
            ):
                resumed += 1
                outputs.append(str(output))
                continue
            label = np.asarray(Image.open(label_path))
            if label.ndim != 2 or not np.issubdtype(label.dtype, np.integer):
                raise StageInputError(
                    f"instance label must be an integer HxW image: {label_path}"
                )
            mask = label == instance_label
            if not bool(mask.any()):
                raise StageInputError(
                    f"target instance label {instance_label} is absent from {label_path}"
                )
            base = {
                **base,
                "shape": list(mask.shape),
                "mask_pixel_count": int(mask.sum()),
            }
            _write_npz_commit(
                output,
                sidecar,
                schema=MASK_SCHEMA,
                input_fingerprint=input_fingerprint,
                arrays={
                    "mask": mask.astype(np.uint8),
                    "target_instance_label": np.int64(instance_label),
                },
                sidecar_payload=base,
            )
            outputs.append(str(output))
        except Exception as error:  # record exact group before batch failure
            failures.append(
                _record_failure(root, stage=stage, group_id=group.group_id, error=error)
            )
    _raise_failures(stage, failures)
    return StageSummary(
        stage, len(groups), len(groups) - resumed, resumed, tuple(outputs)
    )


def _scalar_factor_depth(meta: Mapping[str, Any]) -> float:
    if "factor_depth" not in meta:
        raise StageInputError("frame metadata lacks factor_depth")
    flat = np.asarray(meta["factor_depth"], dtype=np.float64).reshape(-1)
    if flat.size != 1 or not np.isfinite(flat[0]) or flat[0] <= 0:
        raise StageInputError("factor_depth must be one finite positive scalar")
    return float(flat[0])


def _load_mask_cache(path: Path, expected_group_id: str) -> np.ndarray:
    sidecar = path.with_suffix(".json")
    manifest = _read_json(sidecar, "oracle-mask sidecar")
    if (
        manifest.get("schema_version") != MASK_SCHEMA
        or manifest.get("group_id") != expected_group_id
    ):
        raise StageInputError(
            f"oracle-mask cache belongs to another schema/group: {path}"
        )
    if manifest.get("output_sha256") != sha256_file(path):
        raise StageInputError(f"oracle-mask cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        mask = np.asarray(archive["mask"])
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise StageInputError(f"oracle-mask cache is invalid or empty: {path}")
    return mask.astype(bool)


def _load_selected_grounding_mask(
    output_root: Path,
    group_id: str,
    *,
    grounding_condition: str,
    mask_output_root: Path,
) -> tuple[np.ndarray, dict[str, str], dict[str, str]]:
    """Load a hash-verified oracle or HiFi mask without crossing tracks."""

    condition = str(grounding_condition).strip()
    if condition == "oracle_gt_mask":
        mask_path = _require_regular_file(
            mask_output_root / "oracle_masks" / f"{_slug(group_id)}.npz",
            "oracle-mask cache",
        )
        sidecar_path = _require_regular_file(
            mask_path.with_suffix(".json"), "oracle-mask sidecar"
        )
        sidecar = _read_json(sidecar_path, "oracle-mask sidecar")
        return (
            _load_mask_cache(mask_path, group_id),
            {
                str(mask_path): sha256_file(mask_path),
                str(sidecar_path): sha256_file(sidecar_path),
            },
            {
                "grounding_mask_input_fingerprint": str(sidecar["input_fingerprint"]),
                "grounding_mask_commit_sha256": sha256_file(sidecar_path),
            },
        )
    if condition not in {"hifics_zero_shot_mask", "hifics_adapted_mask"}:
        raise StageInputError(f"unsupported grounding condition: {condition!r}")
    # Lazy import avoids making the core oracle/TSDF route depend on Torch.
    from .formal_inputs import load_committed_predicted_mask, predicted_mask_paths

    _, _, sidecar_path = predicted_mask_paths(mask_output_root, condition, group_id)
    committed = load_committed_predicted_mask(
        sidecar_path,
        expected_group_id=group_id,
        expected_condition=condition,
    )
    sources = {
        str(path): sha256_file(path)
        for path in (
            committed.sidecar_path,
            committed.probability_path,
            committed.mask_path,
        )
    }
    return (
        np.asarray(committed.binary_mask, dtype=bool),
        sources,
        {
            "grounding_mask_input_fingerprint": str(
                committed.sidecar["input_fingerprint"]
            ),
            "grounding_mask_commit_sha256": sha256_file(committed.sidecar_path),
        },
    )


def grounding_terminal_path(
    output_root: Path | str, group_id: str, grounding_condition: str
) -> Path:
    """Return the canonical record for an observed, non-fabricated empty pool."""

    return (
        Path(output_root).expanduser().resolve()
        / "grounding_terminal"
        / str(grounding_condition)
        / f"{_slug(group_id)}.json"
    )


def load_grounding_terminal(
    path: Path | str,
    *,
    group_id: str,
    grounding_condition: str,
    expected_input_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Revalidate an expected predicted-grounding failure and all its sources."""

    source = _require_regular_file(path, "grounding terminal record")
    payload = _read_json(source, "grounding terminal record")
    if payload.get("schema_version") != GROUNDING_TERMINAL_SCHEMA:
        raise StageInputError(f"grounding terminal schema mismatch: {source}")
    if payload.get("group_id") != group_id:
        raise StageInputError(f"grounding terminal group mismatch: {source}")
    condition = str(grounding_condition).strip()
    if condition not in {"hifics_zero_shot_mask", "hifics_adapted_mask"}:
        raise StageInputError(
            "grounding terminals are allowed only for predicted masks"
        )
    if payload.get("grounding_condition") != condition:
        raise StageInputError(f"grounding terminal condition mismatch: {source}")
    if payload.get("status") != "empty_pool":
        raise StageInputError(f"grounding terminal status mismatch: {source}")
    reason = payload.get("reason")
    if reason not in {"empty_predicted_mask", "no_valid_predicted_mask_depth"}:
        raise StageInputError(f"grounding terminal reason is invalid: {source}")
    input_fingerprint = str(payload.get("input_fingerprint", ""))
    if re.fullmatch(r"[0-9a-f]{64}", input_fingerprint) is None:
        raise StageInputError(
            f"grounding terminal input fingerprint is invalid: {source}"
        )
    if (
        expected_input_fingerprint is not None
        and input_fingerprint != expected_input_fingerprint
    ):
        raise StageInputError(
            f"grounding terminal input fingerprint mismatch: {source}"
        )
    for field in (
        "grounding_mask_input_fingerprint",
        "grounding_mask_commit_sha256",
        "mask_sha256",
        "depth_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field, ""))) is None:
            raise StageInputError(f"grounding terminal has invalid {field}: {source}")
    foreground_count = payload.get("mask_foreground_pixel_count")
    valid_depth_count = payload.get("valid_target_depth_pixel_count")
    if (
        isinstance(foreground_count, bool)
        or not isinstance(foreground_count, int)
        or foreground_count < 0
        or isinstance(valid_depth_count, bool)
        or not isinstance(valid_depth_count, int)
        or valid_depth_count != 0
    ):
        raise StageInputError(f"grounding terminal pixel counts are invalid: {source}")
    if reason == "empty_predicted_mask" and foreground_count != 0:
        raise StageInputError(
            f"empty-mask terminal records foreground pixels: {source}"
        )
    if reason == "no_valid_predicted_mask_depth" and foreground_count <= 0:
        raise StageInputError(
            f"no-depth terminal has an empty foreground mask: {source}"
        )
    if (
        payload.get("tsdf_constructed") is not False
        or payload.get("vgn_inference_calls_for_group") != 0
        or payload.get("candidate_count") != 0
        or payload.get("candidate_ids") != []
    ):
        raise StageInputError(
            f"grounding terminal fabricates downstream work: {source}"
        )
    mask_path = _require_regular_file(
        payload.get("mask_path", ""), "terminal predicted-mask source"
    )
    if sha256_file(mask_path) != payload["mask_sha256"]:
        raise StageInputError(f"grounding terminal mask source changed: {source}")
    depth_path = _require_regular_file(
        payload.get("depth_path", ""), "terminal depth source"
    )
    if sha256_file(depth_path) != payload["depth_sha256"]:
        raise StageInputError(f"grounding terminal depth source changed: {source}")
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise StageInputError(f"grounding terminal lacks source hashes: {source}")
    if (
        source_hashes.get(str(mask_path)) != payload["mask_sha256"]
        or source_hashes.get(str(depth_path)) != payload["depth_sha256"]
    ):
        raise StageInputError(
            f"grounding terminal mask/depth bindings are incomplete: {source}"
        )
    for raw_path, expected in source_hashes.items():
        if re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None:
            raise StageInputError(
                f"grounding terminal source hash is invalid: {source}"
            )
        item = _require_regular_file(raw_path, "grounding terminal source")
        if sha256_file(item) != expected:
            raise StageInputError(f"grounding terminal source changed: {item}")
    try:
        from PIL import Image

        with Image.open(mask_path) as image:
            mask_pixels = np.asarray(image.convert("L"))
        with Image.open(depth_path) as image:
            depth_pixels = np.asarray(image)
    except (OSError, ValueError) as error:
        raise StageInputError(
            f"cannot decode grounding terminal mask/depth sources: {error}"
        ) from error
    if (
        mask_pixels.ndim != 2
        or depth_pixels.ndim != 2
        or mask_pixels.shape != depth_pixels.shape
        or not np.isin(mask_pixels, [0, 255]).all()
    ):
        raise StageInputError(
            f"grounding terminal mask/depth pixels are invalid: {source}"
        )
    observed_mask = mask_pixels != 0
    observed_foreground = int(np.count_nonzero(observed_mask))
    observed_valid_depth = int(np.count_nonzero(observed_mask & (depth_pixels > 0)))
    if (
        observed_foreground != foreground_count
        or observed_valid_depth != valid_depth_count
    ):
        raise StageInputError(f"grounding terminal pixel evidence mismatch: {source}")
    check = dict(payload)
    observed = check.pop("commit_fingerprint", None)
    if observed != canonical_sha256(check):
        raise StageInputError(f"grounding terminal fingerprint mismatch: {source}")
    return payload


def run_tsdf_stage(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    output_root: Path | str,
    *,
    resume: bool = False,
    grounding_condition: str = "oracle_gt_mask",
    mask_output_root: Path | str | None = None,
    builder: Callable[..., TSDFBuildResult] = build_target_centered_tsdf,
) -> StageSummary:
    """Build target-centred single-view TSDFs from each *complete* depth image."""

    from PIL import Image
    from scipy.io import loadmat

    stage = "target_tsdf"
    root = Path(output_root).expanduser().resolve()
    mask_root = (
        root
        if mask_output_root is None
        else Path(mask_output_root).expanduser().resolve()
    )
    groups = load_target_language_jsonl(target_manifest_path, language_manifest_path)
    outputs: list[str] = []
    resumed = 0
    failures: list[GroupFailure] = []
    for group in groups:
        try:
            mask, mask_hashes, mask_binding = _load_selected_grounding_mask(
                root,
                group.group_id,
                grounding_condition=grounding_condition,
                mask_output_root=mask_root,
            )
            depth_path = _require_regular_file(
                group.target["depth_path"], "depth image"
            )
            meta_path = _require_regular_file(
                group.target["meta_path"], "frame metadata"
            )
            intrinsics_path = _require_regular_file(
                group.target["intrinsics_path"], "camera intrinsics"
            )
            camera_pose_path = _require_regular_file(
                group.target["camera_pose_path"], "camera poses"
            )
            table_path = _require_regular_file(
                group.target["table_transform_path"], "table transform"
            )
            inputs = {
                str(path): sha256_file(path)
                for path in (
                    depth_path,
                    meta_path,
                    intrinsics_path,
                    camera_pose_path,
                    table_path,
                )
            }
            inputs.update(mask_hashes)
            input_fingerprint = canonical_sha256(
                {
                    "schema": TSDF_SCHEMA,
                    "group_id": group.group_id,
                    "grounding_condition": str(grounding_condition),
                    **mask_binding,
                    "inputs": inputs,
                }
            )
            output = (
                root / stage / str(grounding_condition) / f"{_slug(group.group_id)}.npz"
            )
            sidecar = output.with_suffix(".json")
            terminal = grounding_terminal_path(
                root, group.group_id, str(grounding_condition)
            )
            base = {
                "group_id": group.group_id,
                "grounding_condition": str(grounding_condition),
                **mask_binding,
                "source_hashes": inputs,
            }
            if terminal.exists():
                if output.exists() or sidecar.exists():
                    raise StageInputError(
                        "group has both a TSDF and a grounding terminal record"
                    )
                if not resume:
                    raise StageInputError(
                        f"grounding terminal already exists for non-resume run: {terminal}"
                    )
                load_grounding_terminal(
                    terminal,
                    group_id=group.group_id,
                    grounding_condition=str(grounding_condition),
                    expected_input_fingerprint=input_fingerprint,
                )
                resumed += 1
                outputs.append(str(terminal))
                continue
            if _npz_resume_hit(
                output,
                sidecar,
                schema=TSDF_SCHEMA,
                input_fingerprint=input_fingerprint,
                resume=resume,
                sidecar_payload=base,
            ):
                resumed += 1
                outputs.append(str(output))
                continue
            depth = np.asarray(Image.open(depth_path))
            if depth.ndim != 2 or depth.shape != mask.shape:
                raise StageInputError(
                    f"complete depth/mask shapes disagree: {depth.shape} vs {mask.shape}"
                )
            # The untouched array is passed below.  The mask is used only by
            # the builder to centre the workspace, never to erase neighbours.
            complete_depth = depth.copy()
            meta = loadmat(meta_path)
            factor_depth = _scalar_factor_depth(meta)
            if "intrinsic_matrix" not in meta:
                raise StageInputError("frame metadata lacks intrinsic_matrix")
            meta_intrinsics = np.asarray(meta["intrinsic_matrix"], dtype=np.float64)
            stored_intrinsics = np.asarray(
                np.load(intrinsics_path, allow_pickle=False), dtype=np.float64
            )
            if (
                meta_intrinsics.shape != (3, 3)
                or stored_intrinsics.shape != (3, 3)
                or not np.allclose(
                    meta_intrinsics, stored_intrinsics, atol=1e-6, rtol=0
                )
            ):
                raise StageInputError("meta intrinsic_matrix disagrees with camK.npy")
            camera_poses = np.asarray(
                np.load(camera_pose_path, allow_pickle=False), dtype=np.float64
            )
            frame_id = int(group.target["frame_id"])
            if (
                camera_poses.ndim != 3
                or camera_poses.shape[1:] != (4, 4)
                or not 0 <= frame_id < len(camera_poses)
            ):
                raise StageInputError(
                    "camera_poses.npy does not contain the requested 4x4 pose"
                )
            table = np.asarray(
                np.load(table_path, allow_pickle=False), dtype=np.float64
            )
            if table.shape != (4, 4):
                raise StageInputError(
                    "cam0_wrt_table.npy must contain one 4x4 transform"
                )
            T_camera_to_table = table @ camera_poses[frame_id]
            if not np.all(np.isfinite(T_camera_to_table)) or not np.allclose(
                T_camera_to_table[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7
            ):
                raise StageInputError("composed camera-to-table transform is invalid")
            valid_target_depth_count = int(
                np.count_nonzero(mask & (complete_depth > 0))
            )
            if (
                str(grounding_condition)
                in {"hifics_zero_shot_mask", "hifics_adapted_mask"}
                and valid_target_depth_count == 0
            ):
                foreground_count = int(np.count_nonzero(mask))
                binary_mask_paths = [
                    Path(raw_path)
                    for raw_path in mask_hashes
                    if Path(raw_path).suffix.lower() == ".png"
                ]
                if len(binary_mask_paths) != 1:
                    raise StageInputError(
                        "predicted grounding terminal requires one committed binary PNG"
                    )
                binary_mask_path = binary_mask_paths[0]
                terminal_payload: dict[str, Any] = {
                    "schema_version": GROUNDING_TERMINAL_SCHEMA,
                    "group_id": group.group_id,
                    "grounding_condition": str(grounding_condition),
                    "status": "empty_pool",
                    "reason": (
                        "empty_predicted_mask"
                        if foreground_count == 0
                        else "no_valid_predicted_mask_depth"
                    ),
                    "input_fingerprint": input_fingerprint,
                    **mask_binding,
                    "mask_path": str(binary_mask_path),
                    "mask_sha256": sha256_file(binary_mask_path),
                    "depth_path": str(depth_path),
                    "depth_sha256": sha256_file(depth_path),
                    "source_hashes": inputs,
                    "mask_foreground_pixel_count": foreground_count,
                    "valid_target_depth_pixel_count": 0,
                    "tsdf_constructed": False,
                    "vgn_inference_calls_for_group": 0,
                    "candidate_count": 0,
                    "candidate_ids": [],
                }
                terminal_payload["commit_fingerprint"] = canonical_sha256(
                    terminal_payload
                )
                atomic_json(terminal, terminal_payload)
                load_grounding_terminal(
                    terminal,
                    group_id=group.group_id,
                    grounding_condition=str(grounding_condition),
                    expected_input_fingerprint=input_fingerprint,
                )
                outputs.append(str(terminal))
                continue
            result = builder(
                complete_depth,
                mask,
                meta_intrinsics,
                T_camera_to_table=T_camera_to_table,
                depth_scale=factor_depth,
                source_view_id=frame_id,
            )
            if not isinstance(result, TSDFBuildResult):
                raise TypeError("TSDF builder must return TSDFBuildResult")
            if result.full_scene_depth_integrated is not True:
                raise StageInputError(
                    "TSDF builder did not attest complete-scene depth integration"
                )
            if (
                result.tsdf.shape != (1, 40, 40, 40)
                or not np.isfinite(result.tsdf).all()
            ):
                raise StageInputError(
                    "TSDF output violates frozen VGN shape/finite contract"
                )
            arrays = result.cache_record()
            arrays.update(
                {
                    "group_id": np.asarray(group.group_id),
                    "grounding_condition": np.asarray(str(grounding_condition)),
                    "grounding_mask_input_fingerprint": np.asarray(
                        mask_binding["grounding_mask_input_fingerprint"]
                    ),
                    "grounding_mask_commit_sha256": np.asarray(
                        mask_binding["grounding_mask_commit_sha256"]
                    ),
                    "source_complete_depth_sha256": np.asarray(sha256_file(depth_path)),
                }
            )
            _write_npz_commit(
                output,
                sidecar,
                schema=TSDF_SCHEMA,
                input_fingerprint=input_fingerprint,
                arrays=arrays,
                sidecar_payload={
                    **base,
                    "full_scene_depth_integrated": True,
                    "valid_voxel_fraction": float(result.valid_voxel_fraction),
                    "depth_scale": factor_depth,
                    "frame_id": frame_id,
                },
            )
            outputs.append(str(output))
        except Exception as error:
            failures.append(
                _record_failure(root, stage=stage, group_id=group.group_id, error=error)
            )
    _raise_failures(stage, failures)
    return StageSummary(
        stage, len(groups), len(groups) - resumed, resumed, tuple(outputs)
    )


def _resolve_evidence_path(contract_path: Path, raw: Any, description: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise StageInputError(f"{description} must name a non-empty evidence file")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = contract_path.parent / candidate
    return _require_regular_file(candidate, description)


def _sha256_text(value: Any, name: str) -> str:
    text = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise StageInputError(f"{name} must be one full SHA-256 digest")
    return text


def _bound_file(
    evidence_path: Path,
    raw_path: Any,
    expected_sha256: Any,
    description: str,
) -> Path:
    path = _resolve_evidence_path(evidence_path, raw_path, description)
    expected = _sha256_text(expected_sha256, f"{description} SHA-256")
    if sha256_file(path) != expected:
        raise StageInputError(f"{description} hash does not match its evidence: {path}")
    return path


def _boolean_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise StageInputError(f"validation table lacks boolean column {column!r}")
    values: list[bool] = []
    for value in frame[column].tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            values.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {
            "true",
            "false",
            "0",
            "1",
        }:
            values.append(value.strip().lower() in {"true", "1"})
        else:
            raise StageInputError(
                f"validation column {column!r} contains a non-boolean value"
            )
    return np.asarray(values, dtype=bool)


def _formal_geometry_evidence(
    evidence_path: Path,
    payload: Mapping[str, Any],
    contract: EvaluatorGeometryContract,
) -> dict[str, Any]:
    if payload.get("schema_version") != GEOMETRY_EVIDENCE_SCHEMA:
        raise StageInputError(
            "geometry evidence does not use the formal evidence schema"
        )
    if (
        payload.get("scope") != "formal_real_data"
        or payload.get("fixture_only") is True
    ):
        raise StageInputError("geometry evidence is not scoped to formal real data")
    if str(payload.get("status", "")).upper() != "PASSED":
        raise StageInputError("geometry evidence did not pass")
    group_ids = payload.get("validated_group_ids")
    if (
        not isinstance(group_ids, list)
        or not group_ids
        or len({str(value) for value in group_ids}) != len(group_ids)
    ):
        raise StageInputError(
            "geometry evidence requires unique real validated_group_ids"
        )
    bindings = payload.get("bindings")
    required_bindings = {
        "data_manifest_sha256",
        "group_manifest_sha256",
        "tsdf_config_sha256",
        "extraction_config_sha256",
        "upstream_versions_sha256",
    }
    if not isinstance(bindings, Mapping) or not required_bindings.issubset(bindings):
        raise StageInputError(
            f"geometry evidence lacks provenance bindings: {sorted(required_bindings)}"
        )
    checked_bindings = {
        key: _sha256_text(bindings[key], f"geometry binding {key}")
        for key in sorted(required_bindings)
    }
    binding_paths = payload.get("binding_paths")
    if not isinstance(binding_paths, Mapping) or not required_bindings.issubset(
        binding_paths
    ):
        raise StageInputError(
            "geometry evidence lacks files for its provenance bindings"
        )
    checked_binding_paths = {
        key: str(
            _bound_file(
                evidence_path,
                binding_paths[key],
                checked_bindings[key],
                f"geometry binding {key}",
            )
        )
        for key in sorted(required_bindings)
    }
    evidence_rotation = np.asarray(
        payload.get("R_vgn_gripper_to_graspnet_gripper"), dtype=np.float64
    )
    if not np.allclose(
        evidence_rotation,
        contract.R_vgn_gripper_to_graspnet_gripper,
        atol=1e-12,
        rtol=0,
    ):
        raise StageInputError(
            "geometry evidence conversion matrix differs from the contract"
        )
    if not np.isclose(
        float(payload.get("height_m", np.nan)), contract.height_m, atol=1e-12, rtol=0
    ):
        raise StageInputError("geometry evidence height differs from the contract")
    if not np.isclose(
        float(payload.get("depth_m", np.nan)), contract.depth_m, atol=1e-12, rtol=0
    ):
        raise StageInputError("geometry evidence depth differs from the contract")

    samples = _bound_file(
        evidence_path,
        payload.get("sample_metrics_path"),
        payload.get("sample_metrics_sha256"),
        "geometry sample metrics CSV",
    )
    frame = pd.read_csv(samples)
    required_columns = {
        "group_id",
        "reprojection_error_px",
        "camera_table_round_trip_error_m",
        "local_table_round_trip_error_m",
        "rotation_orthogonality_error",
        "rotation_determinant",
        "candidate_center_inside_workspace",
        "projected_center_reasonable",
        "approach_visual_audit_passed",
    }
    if frame.empty or not required_columns.issubset(frame.columns):
        raise StageInputError("geometry sample metrics are empty or incomplete")
    if not set(frame["group_id"].astype(str)).issubset(
        {str(value) for value in group_ids}
    ):
        raise StageInputError("geometry sample rows are outside validated_group_ids")
    numeric_columns = sorted(
        required_columns
        - {
            "group_id",
            "candidate_center_inside_workspace",
            "projected_center_reasonable",
            "approach_visual_audit_passed",
        }
    )
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise StageInputError("geometry sample metrics contain non-finite values")
    reprojection = numeric["reprojection_error_px"].to_numpy(float)
    if (
        float(np.median(reprojection)) >= 1.0
        or float(np.quantile(reprojection, 0.95)) >= 2.0
    ):
        raise StageInputError("geometry reprojection thresholds did not pass")
    if (
        max(
            float(numeric["camera_table_round_trip_error_m"].max()),
            float(numeric["local_table_round_trip_error_m"].max()),
        )
        > 1e-6
    ):
        raise StageInputError("geometry transform round-trip error exceeds 1e-6 m")
    if float(numeric["rotation_orthogonality_error"].max()) > 1e-5:
        raise StageInputError("geometry rotation orthogonality error exceeds 1e-5")
    if (
        float(np.max(np.abs(numeric["rotation_determinant"].to_numpy(float) - 1.0)))
        > 1e-5
    ):
        raise StageInputError("geometry rotation determinant differs from +1")
    for column in (
        "candidate_center_inside_workspace",
        "projected_center_reasonable",
        "approach_visual_audit_passed",
    ):
        if not _boolean_column(frame, column).all():
            raise StageInputError(f"geometry evidence check {column!r} did not pass")

    figure_paths = payload.get("audit_figure_paths")
    figure_hashes = payload.get("audit_figure_sha256")
    if (
        not isinstance(figure_paths, list)
        or len(figure_paths) < 20
        or len(set(map(str, figure_paths))) != len(figure_paths)
        or not isinstance(figure_hashes, Mapping)
    ):
        raise StageInputError(
            "formal geometry evidence requires at least twenty unique audit figures"
        )
    checked_figures: dict[str, str] = {}
    for raw in figure_paths:
        key = str(raw)
        if key not in figure_hashes:
            raise StageInputError(f"geometry audit figure lacks a hash: {key}")
        figure = _bound_file(
            evidence_path, raw, figure_hashes[key], "geometry audit figure"
        )
        checked_figures[str(figure)] = sha256_file(figure)
    return {
        "evidence_schema": GEOMETRY_EVIDENCE_SCHEMA,
        "validated_group_ids": list(map(str, group_ids)),
        "bindings": checked_bindings,
        "binding_paths": checked_binding_paths,
        "sample_metrics_path": str(samples),
        "sample_metrics_sha256": sha256_file(samples),
        "audit_figure_sha256": checked_figures,
        "reprojection_median_error_px": float(np.median(reprojection)),
        "reprojection_p95_error_px": float(np.quantile(reprojection, 0.95)),
    }


def _formal_parity_evidence(
    evidence_path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if payload.get("schema_version") != PARITY_EVIDENCE_SCHEMA:
        raise StageInputError(
            "evaluator parity evidence does not use the formal evidence schema"
        )
    if (
        payload.get("scope") != "formal_real_data"
        or payload.get("fixture_only") is True
    ):
        raise StageInputError(
            "evaluator parity evidence is not scoped to formal real data"
        )
    if str(payload.get("status", "")).upper() != "PASSED":
        raise StageInputError("evaluator parity evidence did not pass")
    if not str(payload.get("group_id", "")).strip():
        raise StageInputError(
            "evaluator parity evidence lacks its official example group_id"
        )
    pool_fingerprint = _sha256_text(
        payload.get("candidate_pool_fingerprint"), "parity candidate pool fingerprint"
    )
    bindings = payload.get("bindings")
    required_bindings = {
        "official_api_source_sha256",
        "adapter_source_sha256",
        "dataset_manifest_sha256",
    }
    if not isinstance(bindings, Mapping) or not required_bindings.issubset(bindings):
        raise StageInputError("evaluator parity evidence lacks source/data bindings")
    checked_bindings = {
        key: _sha256_text(bindings[key], f"parity binding {key}")
        for key in sorted(required_bindings)
    }
    binding_paths = payload.get("binding_paths")
    if not isinstance(binding_paths, Mapping) or not required_bindings.issubset(
        binding_paths
    ):
        raise StageInputError(
            "evaluator parity evidence lacks files for its provenance bindings"
        )
    checked_binding_paths = {
        key: str(
            _bound_file(
                evidence_path,
                binding_paths[key],
                checked_bindings[key],
                f"parity binding {key}",
            )
        )
        for key in sorted(required_bindings)
    }
    comparison = _bound_file(
        evidence_path,
        payload.get("comparison_csv_path"),
        payload.get("comparison_csv_sha256"),
        "evaluator parity comparison CSV",
    )
    report = _bound_file(
        evidence_path,
        payload.get("report_path"),
        payload.get("report_sha256"),
        "evaluator parity report",
    )
    if not report.read_text(encoding="utf-8").strip():
        raise StageInputError("evaluator parity report is empty")
    frame = pd.read_csv(comparison)
    required_columns = {
        "candidate_id",
        "official_associated_object_id",
        "adapter_associated_object_id",
        "official_collision",
        "adapter_collision",
        "official_friction_score",
        "adapter_friction_score",
        "official_binary_valid",
        "adapter_binary_valid",
    }
    candidate_count = int(payload.get("candidate_count", 0))
    if (
        frame.empty
        or candidate_count <= 0
        or len(frame) != candidate_count
        or not required_columns.issubset(frame.columns)
        or frame["candidate_id"].astype(str).duplicated().any()
    ):
        raise StageInputError(
            "evaluator parity comparison is empty, incomplete, or duplicated"
        )
    official_association = pd.to_numeric(
        frame["official_associated_object_id"], errors="coerce"
    ).to_numpy(float)
    adapter_association = pd.to_numeric(
        frame["adapter_associated_object_id"], errors="coerce"
    ).to_numpy(float)
    official_friction = pd.to_numeric(
        frame["official_friction_score"], errors="coerce"
    ).to_numpy(float)
    adapter_friction = pd.to_numeric(
        frame["adapter_friction_score"], errors="coerce"
    ).to_numpy(float)
    if not all(
        np.isfinite(values).all()
        for values in (
            official_association,
            adapter_association,
            official_friction,
            adapter_friction,
        )
    ):
        raise StageInputError("evaluator parity comparison contains non-finite values")
    friction_atol = float(payload.get("friction_atol", np.nan))
    if not np.isfinite(friction_atol) or not 0 <= friction_atol <= 1e-9:
        raise StageInputError(
            "formal evaluator parity friction_atol must be in [0, 1e-9]"
        )
    association_mismatches = int(
        np.count_nonzero(official_association != adapter_association)
    )
    collision_mismatches = int(
        np.count_nonzero(
            _boolean_column(frame, "official_collision")
            != _boolean_column(frame, "adapter_collision")
        )
    )
    validity_mismatches = int(
        np.count_nonzero(
            _boolean_column(frame, "official_binary_valid")
            != _boolean_column(frame, "adapter_binary_valid")
        )
    )
    friction_max_abs_error = float(np.max(np.abs(official_friction - adapter_friction)))
    if association_mismatches or collision_mismatches or validity_mismatches:
        raise StageInputError("evaluator parity has discrete output mismatches")
    if friction_max_abs_error > friction_atol:
        raise StageInputError("evaluator parity friction mismatch exceeds tolerance")
    claimed = payload.get("recomputed_summary")
    observed_summary = {
        "association_mismatches": association_mismatches,
        "collision_mismatches": collision_mismatches,
        "binary_validity_mismatches": validity_mismatches,
        "friction_max_abs_error": friction_max_abs_error,
    }
    if not isinstance(claimed, Mapping):
        raise StageInputError("evaluator parity evidence lacks a recomputed_summary")
    for key, observed in observed_summary.items():
        if key not in claimed or not np.isclose(
            float(claimed[key]), observed, atol=1e-15, rtol=0
        ):
            raise StageInputError(
                f"evaluator parity claimed summary disagrees for {key}"
            )
    return {
        "evidence_schema": PARITY_EVIDENCE_SCHEMA,
        "group_id": str(payload["group_id"]),
        "candidate_count": candidate_count,
        "candidate_pool_fingerprint": pool_fingerprint,
        "bindings": checked_bindings,
        "binding_paths": checked_binding_paths,
        "comparison_csv_path": str(comparison),
        "comparison_csv_sha256": sha256_file(comparison),
        "report_path": str(report),
        "report_sha256": sha256_file(report),
        **observed_summary,
    }


def load_evaluator_geometry_contract(
    path: Path | str,
    *,
    evidence_policy: str = "formal",
) -> tuple[EvaluatorGeometryContract, dict[str, Any]]:
    """Load the explicit real-data geometry acceptance gate."""

    source = _require_regular_file(path, "evaluator geometry contract")
    payload = _read_json(source, "evaluator geometry contract")
    if payload.get("schema_version") != GEOMETRY_SCHEMA:
        raise StageInputError(
            f"unsupported evaluator geometry contract schema: {source}"
        )
    artifact = _resolve_evidence_path(
        source, payload.get("validation_artifact"), "geometry validation artifact"
    )
    contract = EvaluatorGeometryContract(
        validated=payload.get("validated") is True,
        validation_artifact=str(artifact),
        R_vgn_gripper_to_graspnet_gripper=np.asarray(
            payload.get("R_vgn_gripper_to_graspnet_gripper"), dtype=np.float64
        ),
        height_m=float(payload.get("height_m", float("nan"))),
        depth_m=float(payload.get("depth_m", float("nan"))),
    )
    contract.validate()
    if evidence_policy not in {"formal", "test"}:
        raise ValueError("evidence_policy must be 'formal' or 'test'")
    artifact_payload = _read_json(artifact, "geometry validation artifact")
    if evidence_policy == "formal":
        verified_evidence = _formal_geometry_evidence(
            artifact, artifact_payload, contract
        )
    else:
        if artifact_payload.get("fixture_only") is not True:
            raise StageInputError(
                "test evidence policy requires fixture_only:true evidence"
            )
        verified_evidence = {
            "evidence_schema": "explicit_test_fixture_only",
            "fixture_only": True,
        }
    evidence = {
        "contract_path": str(source),
        "contract_sha256": sha256_file(source),
        "validation_artifact": str(artifact),
        "validation_artifact_sha256": sha256_file(artifact),
        "evidence_policy": evidence_policy,
        "verified_evidence": verified_evidence,
    }
    return contract, evidence


def load_evaluator_parity_gate(
    path: Path | str,
    *,
    evidence_policy: str = "formal",
) -> tuple[EvaluatorParityGate, dict[str, Any]]:
    source = _require_regular_file(path, "evaluator parity gate")
    payload = _read_json(source, "evaluator parity gate")
    if payload.get("schema_version") != PARITY_SCHEMA:
        raise StageInputError(f"unsupported evaluator parity schema: {source}")
    artifact = _resolve_evidence_path(
        source, payload.get("artifact_path"), "evaluator parity artifact"
    )
    gate = EvaluatorParityGate(payload.get("validated") is True, str(artifact))
    gate.validate()
    if evidence_policy not in {"formal", "test"}:
        raise ValueError("evidence_policy must be 'formal' or 'test'")
    artifact_payload = _read_json(artifact, "evaluator parity artifact")
    if evidence_policy == "formal":
        verified_evidence = _formal_parity_evidence(artifact, artifact_payload)
    else:
        if artifact_payload.get("fixture_only") is not True:
            raise StageInputError(
                "test evidence policy requires fixture_only:true evidence"
            )
        verified_evidence = {
            "evidence_schema": "explicit_test_fixture_only",
            "fixture_only": True,
        }
    return gate, {
        "gate_path": str(source),
        "gate_sha256": sha256_file(source),
        "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "evidence_policy": evidence_policy,
        "verified_evidence": verified_evidence,
    }


def _load_tsdf_cache(path: Path, group_id: str) -> dict[str, np.ndarray]:
    manifest = _read_json(path.with_suffix(".json"), "TSDF sidecar")
    if (
        manifest.get("schema_version") != TSDF_SCHEMA
        or manifest.get("group_id") != group_id
    ):
        raise StageInputError(f"TSDF cache belongs to another schema/group: {path}")
    if manifest.get("output_sha256") != sha256_file(path):
        raise StageInputError(f"TSDF cache hash mismatch: {path}")
    grounding_condition = str(manifest.get("grounding_condition", "")).strip()
    if grounding_condition not in {
        "oracle_gt_mask",
        "hifics_zero_shot_mask",
        "hifics_adapted_mask",
    }:
        raise StageInputError(f"TSDF cache has invalid grounding condition: {path}")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "tsdf",
        "T_local_to_camera",
        "T_local_to_table",
        "full_scene_depth_integrated",
        "grounding_condition",
        "grounding_mask_input_fingerprint",
        "grounding_mask_commit_sha256",
    }
    if not required.issubset(values):
        raise StageInputError(
            f"TSDF cache lacks arrays: {sorted(required - set(values))}"
        )
    if values["tsdf"].shape != (1, 40, 40, 40) or not np.isfinite(values["tsdf"]).all():
        raise StageInputError(f"TSDF cache violates VGN input contract: {path}")
    if not bool(np.asarray(values["full_scene_depth_integrated"]).item()):
        raise StageInputError(
            f"TSDF cache was not built from complete scene depth: {path}"
        )
    for transform in ("T_local_to_camera", "T_local_to_table"):
        if (
            values[transform].shape != (4, 4)
            or not np.isfinite(values[transform]).all()
        ):
            raise StageInputError(f"TSDF cache has invalid {transform}: {path}")
    embedded_condition = str(np.asarray(values.get("grounding_condition", "")).item())
    if embedded_condition != grounding_condition:
        raise StageInputError(f"TSDF cache grounding condition mismatch: {path}")
    for field in (
        "grounding_mask_input_fingerprint",
        "grounding_mask_commit_sha256",
    ):
        embedded = str(np.asarray(values[field]).item())
        if not re.fullmatch(r"[0-9a-f]{64}", embedded):
            raise StageInputError(f"TSDF cache has invalid {field}: {path}")
        if embedded != str(manifest.get(field, "")):
            raise StageInputError(f"TSDF cache/sidecar disagree on {field}: {path}")
    return values


def _candidate_bundle_path(root: Path, group_id: str, grounding_condition: str) -> Path:
    return (
        root / "vgn_candidates" / str(grounding_condition) / f"{_slug(group_id)}.json"
    )


def _a7_pre_nms_payload(
    pre_nms: Sequence[VGNCandidate],
    frozen: Sequence[VGNCandidate],
    *,
    config: ExtractionConfig,
    status: str,
) -> dict[str, Any]:
    """Serialize the Top-K sensitivity source pool and exact derived memberships."""

    pre_records = [item.to_record() for item in pre_nms]
    frozen_ids = [item.candidate_id for item in frozen]
    memberships: dict[str, dict[str, Any]] = {}
    if status == _PRE_NMS_COMPLETE:
        for top_k in _A7_TOP_K_VALUES:
            selected = pose_nms(tuple(pre_nms)[:top_k], config)[: config.frozen_top_k]
            selected_ids = [item.candidate_id for item in selected]
            if any(candidate_id not in frozen_ids for candidate_id in selected_ids):
                raise StageInputError(
                    "A7 Top-K derivation is not a subset of the frozen K=100 pool"
                )
            memberships[str(top_k)] = {
                "candidate_count": len(selected_ids),
                "candidate_ids": selected_ids,
                "candidate_ids_sha256": canonical_sha256(selected_ids),
            }
    elif status in {_PRE_NMS_TEST_INCOMPLETE, _PRE_NMS_GROUNDING_TERMINAL}:
        memberships = {
            str(top_k): {
                "candidate_count": 0,
                "candidate_ids": [],
                "candidate_ids_sha256": canonical_sha256([]),
            }
            for top_k in _A7_TOP_K_VALUES
        }
    else:  # pragma: no cover - internal enum guard
        raise StageInputError(f"unsupported pre-NMS snapshot status: {status}")
    return {
        "pre_nms_snapshot_status": status,
        "pre_nms_candidate_count": len(pre_records),
        "pre_nms_vgn_candidates": pre_records,
        "pre_nms_pool_fingerprint": canonical_sha256(pre_records),
        "a7_top_k_membership": memberships,
        "a7_top_k_membership_fingerprint": canonical_sha256(memberships),
    }


def _decode_a7_pre_nms_contract(
    payload: Mapping[str, Any],
    *,
    group_id: str,
    frozen: Sequence[VGNCandidate],
    config: ExtractionConfig,
    allow_test_incomplete: bool,
) -> tuple[VGNCandidate, ...]:
    """Recompute every serialized A7 membership from the bound pre-NMS pool."""

    status = payload.get("pre_nms_snapshot_status")
    raw = payload.get("pre_nms_vgn_candidates")
    count = payload.get("pre_nms_candidate_count")
    if (
        not isinstance(raw, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
    ):
        raise StageInputError("candidate bundle has an invalid pre-NMS pool contract")
    if count != len(raw) or count > config.pre_nms_max_candidates:
        raise StageInputError("candidate bundle pre-NMS count is inconsistent")
    if canonical_sha256(raw) != payload.get("pre_nms_pool_fingerprint"):
        raise StageInputError("candidate bundle pre-NMS pool fingerprint mismatch")
    memberships = payload.get("a7_top_k_membership")
    if (
        not isinstance(memberships, Mapping)
        or set(memberships) != {str(value) for value in _A7_TOP_K_VALUES}
        or canonical_sha256(memberships)
        != payload.get("a7_top_k_membership_fingerprint")
    ):
        raise StageInputError("candidate bundle A7 membership fingerprint mismatch")

    if status == _PRE_NMS_TEST_INCOMPLETE:
        if not allow_test_incomplete or raw:
            raise StageInputError("incomplete pre-NMS snapshots are fixture-only")
        expected = _a7_pre_nms_payload(
            (), frozen, config=config, status=_PRE_NMS_TEST_INCOMPLETE
        )
    elif status == _PRE_NMS_GROUNDING_TERMINAL:
        if raw or frozen:
            raise StageInputError(
                "grounding-terminal pre-NMS contract fabricates candidates"
            )
        expected = _a7_pre_nms_payload(
            (), (), config=config, status=_PRE_NMS_GROUNDING_TERMINAL
        )
    elif status == _PRE_NMS_COMPLETE:
        try:
            pre_nms = tuple(
                vgn_candidate_from_record(record, expected_group_id=group_id)
                for record in raw
            )
            validate_extraction_snapshot(
                VGNExtractionSnapshot(pre_nms, tuple(frozen)),
                group_id=group_id,
                config=config,
            )
        except Exception as error:
            raise StageInputError(
                f"invalid pre-NMS extraction snapshot: {error}"
            ) from error
        expected = _a7_pre_nms_payload(
            pre_nms, frozen, config=config, status=_PRE_NMS_COMPLETE
        )
    else:
        raise StageInputError("candidate bundle has an unsupported pre-NMS status")
    for field in (
        "pre_nms_candidate_count",
        "pre_nms_pool_fingerprint",
        "a7_top_k_membership",
        "a7_top_k_membership_fingerprint",
    ):
        if payload.get(field) != expected[field]:
            raise StageInputError(f"candidate bundle has a stale A7 field: {field}")
    return tuple(
        vgn_candidate_from_record(record, expected_group_id=group_id) for record in raw
    )


def _validate_resumable_candidate_bundle(
    path: Path, *, group_id: str, input_fingerprint: str, resume: bool
) -> bool:
    if not path.exists():
        return False
    if not resume:
        raise StageInputError(
            f"candidate bundle already exists for non-resume run: {path}; use resume=True"
        )
    payload = _read_json(path, "candidate bundle")
    if payload.get("schema_version") != VGN_BUNDLE_SCHEMA:
        raise StageInputError(f"candidate bundle schema mismatch: {path}")
    if (
        payload.get("group_id") != group_id
        or payload.get("input_fingerprint") != input_fingerprint
    ):
        raise StageInputError(
            f"candidate bundle group/input fingerprint mismatch: {path}"
        )
    check = dict(payload)
    observed = check.pop("bundle_fingerprint", None)
    if observed != canonical_sha256(check):
        raise StageInputError(f"candidate bundle fingerprint mismatch: {path}")
    records = payload.get("candidate_records")
    raw = payload.get("raw_vgn_candidates")
    if (
        not isinstance(records, list)
        or not isinstance(raw, list)
        or len(records) != len(raw)
    ):
        raise StageInputError(
            f"candidate bundle has inconsistent candidate arrays: {path}"
        )
    count = payload.get("candidate_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(records):
        raise StageInputError(f"candidate bundle count mismatch: {path}")
    candidates = [Candidate6D.from_dict(record) for record in records]
    if candidate_pool_fingerprint(candidates) != payload.get(
        "candidate_pool_fingerprint"
    ):
        raise StageInputError(f"candidate pool fingerprint mismatch: {path}")
    if canonical_sha256(raw) != payload.get("raw_vgn_pool_fingerprint"):
        raise StageInputError(f"raw VGN pool fingerprint mismatch: {path}")
    extraction_record = payload.get("extraction_config")
    if not isinstance(extraction_record, Mapping):
        raise StageInputError(f"candidate extraction config is invalid: {path}")
    try:
        extraction = ExtractionConfig(**dict(extraction_record))
        extraction.validate()
    except (TypeError, ValueError) as error:
        raise StageInputError(
            f"candidate extraction config is invalid: {path}"
        ) from error
    if asdict(extraction) != dict(extraction_record):
        raise StageInputError(f"candidate extraction config is not canonical: {path}")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    candidate_ranks = [candidate.native_rank for candidate in candidates]
    if [str(item.get("candidate_id", "")) for item in raw] != candidate_ids or [
        int(item.get("native_rank", -1)) for item in raw
    ] != candidate_ranks:
        raise StageInputError(f"raw/converted candidate membership mismatch: {path}")
    try:
        frozen_vgn = tuple(
            vgn_candidate_from_record(item, expected_group_id=group_id) for item in raw
        )
    except Exception as error:
        raise StageInputError(
            f"invalid raw frozen VGN pool: {path}: {error}"
        ) from error
    geometry_record = payload.get("geometry_contract")
    allow_test_incomplete = bool(
        isinstance(geometry_record, Mapping)
        and geometry_record.get("evidence_policy") == "test"
    )
    _decode_a7_pre_nms_contract(
        payload,
        group_id=group_id,
        frozen=frozen_vgn,
        config=extraction,
        allow_test_incomplete=allow_test_incomplete,
    )
    raw_rows = payload.get("graspnet_rows")
    rows = (
        np.empty((0, 17), dtype=np.float64)
        if count == 0 and raw_rows == []
        else np.asarray(raw_rows, dtype=np.float64)
    )
    if rows.shape != (count, 17) or not np.isfinite(rows).all():
        raise StageInputError(f"candidate evaluator rows are invalid: {path}")
    expected_rows = np.asarray(
        [
            [
                candidate.native_score,
                candidate.width_m,
                candidate.height_m,
                candidate.depth_m,
                *np.asarray(candidate.rotation_camera, dtype=np.float64).reshape(-1),
                *np.asarray(candidate.translation_camera_m, dtype=np.float64),
                -1.0,
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    ).reshape(count, 17)
    if not np.allclose(rows, expected_rows, atol=1e-12, rtol=0.0):
        raise StageInputError(
            f"candidate evaluator rows disagree with geometry: {path}"
        )
    condition = str(payload.get("grounding_condition", ""))
    if condition not in {
        "oracle_gt_mask",
        "hifics_zero_shot_mask",
        "hifics_adapted_mask",
    }:
        raise StageInputError(f"candidate grounding condition is invalid: {path}")
    for field in (
        "grounding_mask_input_fingerprint",
        "grounding_mask_commit_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field, ""))) is None:
            raise StageInputError(f"candidate bundle has invalid {field}: {path}")
    checkpoint = _require_regular_file(
        payload.get("checkpoint_path", ""), "candidate VGN checkpoint"
    )
    if sha256_file(checkpoint) != payload.get("checkpoint_sha256"):
        raise StageInputError(f"candidate checkpoint source changed: {path}")
    generation_status = payload.get("generation_status")
    if generation_status == "completed_vgn_inference":
        if payload.get("inference_calls_for_group") != 1:
            raise StageInputError(f"candidate inference count is invalid: {path}")
        tsdf = _require_regular_file(payload.get("tsdf_path", ""), "candidate TSDF")
        if sha256_file(tsdf) != payload.get("tsdf_sha256"):
            raise StageInputError(f"candidate TSDF source changed: {path}")
        cache = _load_tsdf_cache(tsdf, group_id)
        if str(np.asarray(cache["grounding_condition"]).item()) != condition:
            raise StageInputError(f"candidate TSDF condition mismatch: {path}")
        _validate_raw_to_evaluator_geometry(
            payload,
            group_id=group_id,
            raw_candidates=frozen_vgn,
            candidates=candidates,
            evaluator_rows=rows,
            allow_explicit_fixture_only=False,
        )
    elif generation_status == "skipped_grounding_failure":
        if (
            count != 0
            or payload.get("inference_calls_for_group") != 0
            or payload.get("tsdf_path") is not None
            or payload.get("tsdf_sha256") is not None
        ):
            raise StageInputError(f"skipped candidate bundle fabricates work: {path}")
        terminal = _require_regular_file(
            payload.get("grounding_terminal_path", ""),
            "candidate grounding terminal",
        )
        if sha256_file(terminal) != payload.get("grounding_terminal_sha256"):
            raise StageInputError(f"candidate grounding terminal changed: {path}")
        terminal_payload = load_grounding_terminal(
            terminal, group_id=group_id, grounding_condition=condition
        )
        if terminal_payload.get("reason") != payload.get("grounding_failure_reason"):
            raise StageInputError(
                f"candidate grounding failure reason mismatch: {path}"
            )
    else:
        raise StageInputError(f"candidate generation_status is invalid: {path}")
    return True


def _convert_frozen_candidates(
    values: Sequence[VGNCandidate],
    geometry: EvaluatorGeometryContract,
    geometry_evidence: Mapping[str, Any],
) -> tuple[list[Candidate6D], list[list[float]]]:
    conversion = np.asarray(
        geometry.R_vgn_gripper_to_graspnet_gripper, dtype=np.float64
    )
    converted: list[Candidate6D] = []
    rows: list[list[float]] = []
    provenance = {
        "mapping_name": "validated_vgn_to_graspnet_gripper_axes",
        "mapping_status": "validated_real_data_artifact",
        "source": str(geometry_evidence["contract_path"]),
        **dict(geometry_evidence),
        "height_source": "EvaluatorGeometryContract.height_m",
        "depth_source": "EvaluatorGeometryContract.depth_m",
    }
    for candidate in values:
        if (
            candidate.translation_camera_m is None
            or candidate.rotation_camera_vgn is None
            or candidate.translation_table_m is None
            or candidate.rotation_table_vgn is None
        ):
            raise StageInputError(
                f"VGN candidate {candidate.candidate_id} lacks camera/table transforms"
            )
        item = Candidate6D(
            candidate_id=candidate.candidate_id,
            group_id=candidate.group_id,
            native_rank=candidate.native_rank,
            native_score=candidate.native_score,
            translation_local_m=candidate.translation_local_m,
            rotation_local=candidate.rotation_local_vgn @ conversion,
            translation_camera_m=candidate.translation_camera_m,
            rotation_camera=candidate.rotation_camera_vgn @ conversion,
            translation_table_m=candidate.translation_table_m,
            rotation_table=candidate.rotation_table_vgn @ conversion,
            width_m=candidate.width_m,
            height_m=geometry.height_m,
            depth_m=geometry.depth_m,
            voxel_index=candidate.voxel_index,
            conversion_provenance=provenance,
        )
        row = candidate_to_graspnet_row(candidate, geometry)
        if not np.allclose(row[4:13].reshape(3, 3), np.asarray(item.rotation_camera)):
            raise AssertionError(
                "candidate record and evaluator row conversion disagree"
            )
        converted.append(item)
        rows.append(row.tolist())
    return converted, rows


def _validate_raw_to_evaluator_geometry(
    payload: Mapping[str, Any],
    *,
    group_id: str,
    raw_candidates: Sequence[VGNCandidate],
    candidates: Sequence[Candidate6D],
    evaluator_rows: np.ndarray,
    allow_explicit_fixture_only: bool,
) -> None:
    """Recompute the complete raw-VGN to evaluator conversion boundary.

    Fingerprints only prove that a serialized object is self-consistent.  They
    cannot prove that a consistently re-hashed converted pose was derived from
    the frozen raw VGN pose.  This check therefore reloads the *bound* geometry
    contract and evidence, converts every raw candidate again without network
    inference, and compares all evaluator-relevant fields at locked numeric
    tolerances.

    A legacy, explicitly marked unit-test fixture may omit the raw geometry and
    contract files.  Such incompleteness is never accepted when the record
    claims the formal evidence policy, and callers must opt in explicitly.
    """

    geometry_record = payload.get("geometry_contract")
    if not isinstance(geometry_record, Mapping):
        raise StageInputError("candidate bundle lacks a geometry contract binding")
    fixture_only = geometry_record.get("fixture_only") is True
    evidence_policy = geometry_record.get("evidence_policy")
    if fixture_only:
        if evidence_policy == "formal" or not allow_explicit_fixture_only:
            raise StageInputError(
                "formal candidate conversion cannot use incomplete fixture geometry"
            )
        return
    if evidence_policy not in {"formal", "test"}:
        raise StageInputError(
            "candidate geometry binding lacks an explicit evidence policy"
        )
    try:
        geometry, checked_evidence = load_evaluator_geometry_contract(
            geometry_record.get("contract_path", ""),
            evidence_policy=str(evidence_policy),
        )
    except Exception as error:
        raise StageInputError(
            f"candidate geometry contract/evidence cannot be revalidated: {error}"
        ) from error
    if canonical_sha256(dict(geometry_record)) != canonical_sha256(checked_evidence):
        raise StageInputError(
            "candidate geometry contract/evidence binding is stale or incomplete"
        )
    if evidence_policy == "formal" and (
        checked_evidence.get("verified_evidence", {}).get("fixture_only") is True
    ):
        raise StageInputError("formal candidate geometry evidence is fixture-only")
    if len(raw_candidates) != len(candidates):
        raise StageInputError("raw and converted candidate counts disagree")
    try:
        expected_candidates, expected_row_records = _convert_frozen_candidates(
            raw_candidates, geometry, checked_evidence
        )
    except Exception as error:
        raise StageInputError(
            f"raw VGN candidate conversion cannot be reconstructed: {error}"
        ) from error
    expected_rows = np.asarray(expected_row_records, dtype=np.float64).reshape(
        len(expected_candidates), 17
    )
    observed_rows = np.asarray(evaluator_rows, dtype=np.float64)
    if observed_rows.shape != expected_rows.shape or not np.allclose(
        observed_rows,
        expected_rows,
        atol=_EVALUATOR_ROW_ATOL,
        rtol=0.0,
    ):
        raise StageInputError(
            "evaluator rows do not match raw VGN candidates under the bound geometry"
        )

    scalar_fields = ("native_score", "width_m", "height_m", "depth_m")
    translation_fields = (
        "translation_local_m",
        "translation_camera_m",
        "translation_table_m",
    )
    rotation_fields = ("rotation_local", "rotation_camera", "rotation_table")
    for observed, expected in zip(candidates, expected_candidates, strict=True):
        if (
            observed.candidate_id != expected.candidate_id
            or observed.group_id != group_id
            or observed.group_id != expected.group_id
            or observed.native_rank != expected.native_rank
            or observed.voxel_index != expected.voxel_index
        ):
            raise StageInputError(
                "converted candidate identity/rank/voxel disagrees with raw VGN"
            )
        for field in scalar_fields:
            if not np.isclose(
                float(getattr(observed, field)),
                float(getattr(expected, field)),
                atol=_CONVERSION_SCALAR_ATOL,
                rtol=0.0,
            ):
                raise StageInputError(
                    f"converted candidate {field} disagrees with raw VGN/geometry"
                )
        for field in translation_fields:
            if not np.allclose(
                np.asarray(getattr(observed, field), dtype=np.float64),
                np.asarray(getattr(expected, field), dtype=np.float64),
                atol=_CONVERSION_TRANSLATION_ATOL_M,
                rtol=0.0,
            ):
                raise StageInputError(
                    f"converted candidate {field} disagrees with raw VGN/geometry"
                )
        for field in rotation_fields:
            if not np.allclose(
                np.asarray(getattr(observed, field), dtype=np.float64),
                np.asarray(getattr(expected, field), dtype=np.float64),
                atol=_CONVERSION_ROTATION_ATOL,
                rtol=0.0,
            ):
                raise StageInputError(
                    f"converted candidate {field} disagrees with raw VGN/geometry"
                )
        if canonical_sha256(dict(observed.conversion_provenance)) != canonical_sha256(
            dict(expected.conversion_provenance)
        ):
            raise StageInputError(
                "converted candidate provenance disagrees with the bound geometry"
            )


def run_vgn_candidate_stage(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    output_root: Path | str,
    *,
    geometry_contract_path: Path | str,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    grounding_condition: str = "oracle_gt_mask",
    config: ExtractionConfig = ExtractionConfig(),
    resume: bool = False,
    evidence_policy: str = "formal",
    model_loader: Callable[..., Any] = load_frozen_vgn,
    inference: Callable[..., Any] = run_vgn,
    extractor: Callable[
        ..., VGNExtractionSnapshot | Sequence[VGNCandidate]
    ] = extract_candidate_snapshot,
) -> StageSummary:
    """Run frozen VGN once for each uncommitted group and freeze its pool.

    The geometry contract is mandatory: this formal-stage API never publishes
    evaluator-frame candidates using an unvalidated axis/height/depth guess.
    Raw VGN-frame poses are retained alongside the converted records.
    """

    stage = "vgn_candidates"
    root = Path(output_root).expanduser().resolve()
    condition = str(grounding_condition).strip()
    if condition not in {
        "oracle_gt_mask",
        "hifics_zero_shot_mask",
        "hifics_adapted_mask",
    }:
        raise StageInputError(f"unsupported grounding condition: {condition!r}")
    groups = load_target_language_jsonl(target_manifest_path, language_manifest_path)
    failures: list[GroupFailure] = []
    try:
        geometry, geometry_evidence = load_evaluator_geometry_contract(
            geometry_contract_path, evidence_policy=evidence_policy
        )
        checkpoint_path = _require_regular_file(checkpoint, "VGN checkpoint")
        checkpoint_digest = sha256_file(checkpoint_path)
        config.validate()
        if evidence_policy == "formal" and (
            config.pre_nms_max_candidates != 100 or config.frozen_top_k != 50
        ):
            raise StageInputError(
                "formal A7 requires pre_nms_max_candidates=100 and frozen_top_k=50"
            )
    except Exception as error:
        for group in groups:
            failures.append(
                _record_failure(root, stage=stage, group_id=group.group_id, error=error)
            )
        _raise_failures(stage, failures)
        raise AssertionError("unreachable")

    pending: list[tuple[GroupManifest, Path, dict[str, np.ndarray], str]] = []
    outputs: list[str] = []
    resumed = 0
    for group in groups:
        try:
            raw_tsdf_path = (
                root / "target_tsdf" / condition / f"{_slug(group.group_id)}.npz"
            )
            terminal_path = grounding_terminal_path(root, group.group_id, condition)
            if raw_tsdf_path.exists() and terminal_path.exists():
                raise StageInputError(
                    "candidate generation requires exactly one of TSDF or grounding terminal"
                )
            if terminal_path.exists():
                terminal = load_grounding_terminal(
                    terminal_path,
                    group_id=group.group_id,
                    grounding_condition=condition,
                )
                terminal_sha = sha256_file(terminal_path)
                input_fingerprint = canonical_sha256(
                    {
                        "schema": VGN_BUNDLE_SCHEMA,
                        "group_id": group.group_id,
                        "grounding_terminal_sha256": terminal_sha,
                        "checkpoint_sha256": checkpoint_digest,
                        "device": str(device),
                        "extraction_config": asdict(config),
                        "geometry": geometry_evidence,
                        "grounding_condition": condition,
                        "grounding_mask_input_fingerprint": terminal[
                            "grounding_mask_input_fingerprint"
                        ],
                        "grounding_mask_commit_sha256": terminal[
                            "grounding_mask_commit_sha256"
                        ],
                    }
                )
                output = _candidate_bundle_path(root, group.group_id, condition)
                if _validate_resumable_candidate_bundle(
                    output,
                    group_id=group.group_id,
                    input_fingerprint=input_fingerprint,
                    resume=resume,
                ):
                    resumed += 1
                    outputs.append(str(output))
                    continue
                empty_records: list[dict[str, Any]] = []
                payload: dict[str, Any] = {
                    "schema_version": VGN_BUNDLE_SCHEMA,
                    "group_id": group.group_id,
                    "input_fingerprint": input_fingerprint,
                    "generation_status": "skipped_grounding_failure",
                    "grounding_failure_reason": terminal["reason"],
                    "grounding_terminal_path": str(terminal_path),
                    "grounding_terminal_sha256": terminal_sha,
                    "tsdf_path": None,
                    "tsdf_sha256": None,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_digest,
                    "device": str(device),
                    "grounding_condition": condition,
                    "grounding_mask_input_fingerprint": terminal[
                        "grounding_mask_input_fingerprint"
                    ],
                    "grounding_mask_commit_sha256": terminal[
                        "grounding_mask_commit_sha256"
                    ],
                    "extraction_config": asdict(config),
                    "inference_calls_for_group": 0,
                    "candidate_count": 0,
                    "candidate_ids": [],
                    "raw_vgn_candidates": empty_records,
                    "raw_vgn_pool_fingerprint": canonical_sha256(empty_records),
                    "candidate_records": [],
                    "candidate_pool_fingerprint": candidate_pool_fingerprint([]),
                    "graspnet_rows": [],
                    "geometry_contract": dict(geometry_evidence),
                    **_a7_pre_nms_payload(
                        (),
                        (),
                        config=config,
                        status=_PRE_NMS_GROUNDING_TERMINAL,
                    ),
                }
                payload["bundle_fingerprint"] = canonical_sha256(payload)
                atomic_json(output, payload)
                outputs.append(str(output))
                continue
            tsdf_path = _require_regular_file(raw_tsdf_path, "TSDF cache")
            cache = _load_tsdf_cache(tsdf_path, group.group_id)
            cached_condition = str(np.asarray(cache["grounding_condition"]).item())
            if cached_condition != condition:
                raise StageInputError(
                    "TSDF cache belongs to another grounding condition"
                )
            input_fingerprint = canonical_sha256(
                {
                    "schema": VGN_BUNDLE_SCHEMA,
                    "group_id": group.group_id,
                    "tsdf_sha256": sha256_file(tsdf_path),
                    "checkpoint_sha256": checkpoint_digest,
                    "device": str(device),
                    "extraction_config": asdict(config),
                    "geometry": geometry_evidence,
                    "grounding_condition": condition,
                    "grounding_mask_input_fingerprint": str(
                        np.asarray(cache["grounding_mask_input_fingerprint"]).item()
                    ),
                    "grounding_mask_commit_sha256": str(
                        np.asarray(cache["grounding_mask_commit_sha256"]).item()
                    ),
                }
            )
            output = _candidate_bundle_path(root, group.group_id, condition)
            if _validate_resumable_candidate_bundle(
                output,
                group_id=group.group_id,
                input_fingerprint=input_fingerprint,
                resume=resume,
            ):
                resumed += 1
                outputs.append(str(output))
                continue
            pending.append((group, tsdf_path, cache, input_fingerprint))
        except Exception as error:
            failures.append(
                _record_failure(root, stage=stage, group_id=group.group_id, error=error)
            )
    if failures:
        _raise_failures(stage, failures)

    model: Any | None = None
    if pending:
        try:
            model = model_loader(checkpoint=checkpoint_path, device=device)
        except Exception as error:
            for group, _, _, _ in pending:
                failures.append(
                    _record_failure(
                        root, stage=stage, group_id=group.group_id, error=error
                    )
                )
            _raise_failures(stage, failures)
    for group, tsdf_path, cache, input_fingerprint in pending:
        try:
            # Exactly one dense forward call occurs in this group operation.
            raw_outputs = inference(cache["tsdf"], model, device=device)
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
                if evidence_policy != "test":
                    raise StageInputError(
                        "formal candidate extraction requires a complete same-inference "
                        "pre-NMS snapshot"
                    )
                pre_nms = []
                frozen = list(extraction_result)
                pre_nms_status = _PRE_NMS_TEST_INCOMPLETE
            if any(item.group_id != group.group_id for item in frozen):
                raise StageInputError(
                    "VGN extractor returned a candidate for another group"
                )
            identifiers = [item.candidate_id for item in frozen]
            ranks = [int(item.native_rank) for item in frozen]
            if len(identifiers) != len(set(identifiers)):
                raise StageInputError("VGN extractor returned duplicate candidate IDs")
            if any(rank < 1 for rank in ranks) or ranks != sorted(set(ranks)):
                raise StageInputError(
                    "frozen candidates are not in deterministic native-rank order"
                )
            if len(frozen) > config.frozen_top_k:
                raise StageInputError("VGN extractor exceeded frozen_top_k")
            expected_frozen = pose_nms(frozen, config)[: config.frozen_top_k]
            if [item.candidate_id for item in expected_frozen] != identifiers:
                raise StageInputError(
                    "VGN extractor output does not equal deterministic pose-NMS/Top-K"
                )
            candidates, rows = _convert_frozen_candidates(
                frozen, geometry, geometry_evidence
            )
            raw_records = [item.to_record() for item in frozen]
            payload: dict[str, Any] = {
                "schema_version": VGN_BUNDLE_SCHEMA,
                "group_id": group.group_id,
                "input_fingerprint": input_fingerprint,
                "tsdf_path": str(tsdf_path),
                "tsdf_sha256": sha256_file(tsdf_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_digest,
                "device": str(device),
                "grounding_condition": condition,
                "grounding_mask_input_fingerprint": str(
                    np.asarray(cache["grounding_mask_input_fingerprint"]).item()
                ),
                "grounding_mask_commit_sha256": str(
                    np.asarray(cache["grounding_mask_commit_sha256"]).item()
                ),
                "extraction_config": asdict(config),
                "generation_status": "completed_vgn_inference",
                "inference_calls_for_group": 1,
                "candidate_count": len(frozen),
                "raw_vgn_candidates": raw_records,
                "raw_vgn_pool_fingerprint": canonical_sha256(raw_records),
                "candidate_records": [item.to_dict() for item in candidates],
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
            atomic_json(output, payload)
            outputs.append(str(output))
        except Exception as error:
            failures.append(
                _record_failure(root, stage=stage, group_id=group.group_id, error=error)
            )
    _raise_failures(stage, failures)
    return StageSummary(
        stage, len(groups), len(groups) - resumed, resumed, tuple(outputs)
    )


def _load_candidate_bundle(path: Path, group_id: str) -> dict[str, Any]:
    payload = _read_json(path, "frozen candidate bundle")
    fingerprint = payload.get("input_fingerprint")
    if not isinstance(fingerprint, str) or not _validate_resumable_candidate_bundle(
        path, group_id=group_id, input_fingerprint=fingerprint, resume=True
    ):
        raise StageInputError(f"invalid frozen candidate bundle: {path}")
    count = int(payload["candidate_count"])
    if payload.get("grounding_condition") not in {
        "oracle_gt_mask",
        "hifics_zero_shot_mask",
        "hifics_adapted_mask",
    }:
        raise StageInputError(
            f"candidate bundle has invalid grounding condition: {path}"
        )
    for field in (
        "grounding_mask_input_fingerprint",
        "grounding_mask_commit_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field, ""))) is None:
            raise StageInputError(f"candidate bundle has invalid {field}: {path}")
    generation_status = payload.get("generation_status", "completed_vgn_inference")
    if generation_status == "skipped_grounding_failure":
        if (
            count != 0
            or payload.get("inference_calls_for_group") != 0
            or payload.get("candidate_records") != []
            or payload.get("raw_vgn_candidates") != []
            or payload.get("graspnet_rows") != []
            or payload.get("tsdf_path") is not None
            or payload.get("tsdf_sha256") is not None
        ):
            raise StageInputError(
                f"skipped candidate bundle contains fabricated work: {path}"
            )
        terminal_path = _require_regular_file(
            payload.get("grounding_terminal_path", ""),
            "candidate grounding terminal source",
        )
        if sha256_file(terminal_path) != payload.get("grounding_terminal_sha256"):
            raise StageInputError(
                f"candidate grounding terminal source hash mismatch: {path}"
            )
        terminal = load_grounding_terminal(
            terminal_path,
            group_id=group_id,
            grounding_condition=str(payload["grounding_condition"]),
        )
        if (
            terminal.get("reason") != payload.get("grounding_failure_reason")
            or terminal.get("grounding_mask_input_fingerprint")
            != payload["grounding_mask_input_fingerprint"]
            or terminal.get("grounding_mask_commit_sha256")
            != payload["grounding_mask_commit_sha256"]
        ):
            raise StageInputError(
                f"candidate bundle grounding terminal lineage mismatch: {path}"
            )
    elif generation_status == "completed_vgn_inference":
        if payload.get("inference_calls_for_group", 1) != 1:
            raise StageInputError(
                f"candidate bundle inference count is invalid: {path}"
            )
        tsdf_path = _require_regular_file(
            payload.get("tsdf_path", ""), "candidate TSDF source"
        )
        if sha256_file(tsdf_path) != payload.get("tsdf_sha256"):
            raise StageInputError(f"candidate TSDF source hash mismatch: {path}")
        tsdf_sidecar = _read_json(
            tsdf_path.with_suffix(".json"), "candidate TSDF sidecar"
        )
        if (
            tsdf_sidecar.get("grounding_condition") != payload["grounding_condition"]
            or tsdf_sidecar.get("grounding_mask_input_fingerprint")
            != payload["grounding_mask_input_fingerprint"]
            or tsdf_sidecar.get("grounding_mask_commit_sha256")
            != payload["grounding_mask_commit_sha256"]
        ):
            raise StageInputError(
                f"candidate bundle grounding lineage mismatch: {path}"
            )
    else:
        raise StageInputError(f"candidate bundle generation status is invalid: {path}")
    raw_rows = payload.get("graspnet_rows")
    rows = (
        np.empty((0, 17), dtype=np.float64)
        if count == 0 and raw_rows == []
        else np.asarray(raw_rows, dtype=np.float64)
    )
    if rows.shape != (count, 17):
        raise StageInputError(f"candidate evaluator rows have the wrong shape: {path}")
    payload["_rows_array"] = rows
    return payload


def load_frozen_candidate_bundle(
    path: Path | str, *, group_id: str, grounding_condition: str | None = None
) -> dict[str, Any]:
    """Public strict loader used by indexes and downstream stage boundaries."""

    source = _require_regular_file(path, "frozen candidate bundle")
    payload = _load_candidate_bundle(source, group_id)
    if grounding_condition is not None and payload.get("grounding_condition") != str(
        grounding_condition
    ):
        raise StageInputError(
            f"candidate bundle belongs to another grounding condition: {source}"
        )
    return payload


@lru_cache(maxsize=4096)
def _source_digest_cached(
    path: str, size: int, mtime_ns: int, ctime_ns: int
) -> str:
    del size, mtime_ns, ctime_ns
    return sha256_file(path)


def _source_digest(path: Path | str) -> str:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    return _source_digest_cached(
        str(source), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)
    )


@lru_cache(maxsize=128)
def _official_object_evaluator_asset(
    dataset_root: str, object_id: int, api_root: str
) -> tuple[np.ndarray, Any, tuple[str, str, str], dict[str, str]]:
    ensure_graspnetapi_source(api_root)
    try:
        import open3d as o3d
        from graspnetAPI.utils.eval_utils import load_dexnet_model
    except (ImportError, ModuleNotFoundError) as error:
        raise StageInputError(
            f"official evaluator dependency is unavailable: {error}"
        ) from error
    model_dir = Path(dataset_root) / "models" / f"{int(object_id):03d}"
    ply = _require_regular_file(model_dir / "nontextured.ply", "official object PLY")
    dex_prefix = model_dir / "textured"
    obj = _require_regular_file(
        dex_prefix.with_suffix(".obj"), "Dex-Net OBJ source"
    )
    sdf = _require_regular_file(
        dex_prefix.with_suffix(".sdf"), "Dex-Net SDF source"
    )
    cloud = o3d.io.read_point_cloud(str(ply))
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise StageInputError(f"official object PLY is empty or invalid: {ply}")
    paths = (str(ply), str(obj), str(sdf))
    hashes = {path: _source_digest(path) for path in paths}
    return points, load_dexnet_model(str(dex_prefix)), paths, hashes


def _official_group_evaluator_inputs(
    group: GroupManifest,
    *,
    dataset_root: Path,
    api_root: Path | str = DEFAULT_GRASPNET_API_ROOT,
) -> dict[str, Any]:
    """Load official XML poses, PLY models, Dex-Net sources, and table points.

    Dex-Net models are built from official ``textured.obj/.sdf`` sources.  This
    path intentionally does not load downloaded pickle caches.
    """

    ensure_graspnetapi_source(api_root)
    try:
        from graspnetAPI.utils.config import get_config
        from graspnetAPI.utils.eval_utils import (
            create_table_points,
            parse_posevector,
            transform_points,
        )
        from graspnetAPI.utils.xmlhandler import xmlReader
    except (ImportError, ModuleNotFoundError) as error:
        raise StageInputError(
            f"official evaluator dependency is unavailable: {error}"
        ) from error

    scene_id = str(group.target["scene_id"])
    camera = str(group.target["camera"])
    frame_id = int(group.target["frame_id"])
    annotation = _require_regular_file(
        dataset_root
        / "scenes"
        / scene_id
        / camera
        / "annotations"
        / f"{frame_id:04d}.xml",
        "official scene annotation",
    )
    posevectors = xmlReader(str(annotation)).getposevectorlist()
    if not posevectors:
        raise StageInputError(f"official annotation contains no objects: {annotation}")
    object_ids: list[int] = []
    poses: list[np.ndarray] = []
    models: list[np.ndarray] = []
    dexnet_models: list[Any] = []
    source_files: list[Path] = [annotation]
    for posevector in posevectors:
        object_id, pose = parse_posevector(posevector)
        object_id = int(object_id)
        points, dexnet_model, asset_paths, _ = _official_object_evaluator_asset(
            str(dataset_root.resolve()), object_id, str(Path(api_root).resolve())
        )
        object_ids.append(object_id)
        poses.append(np.asarray(pose, dtype=np.float64))
        models.append(points)
        dexnet_models.append(dexnet_model)
        source_files.extend(Path(value) for value in asset_paths)

    camera_pose = _require_regular_file(
        group.target["camera_pose_path"], "camera poses"
    )
    table_transform = _require_regular_file(
        group.target["table_transform_path"], "table transform"
    )
    camera_poses = np.asarray(
        np.load(camera_pose, allow_pickle=False), dtype=np.float64
    )
    if (
        camera_poses.ndim != 3
        or camera_poses.shape[1:] != (4, 4)
        or not 0 <= frame_id < len(camera_poses)
    ):
        raise StageInputError("camera pose source does not contain the evaluator frame")
    align = np.asarray(np.load(table_transform, allow_pickle=False), dtype=np.float64)
    if align.shape != (4, 4):
        raise StageInputError("table transform must be 4x4")
    table = create_table_points(
        1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008
    )
    table_camera = transform_points(
        table, np.linalg.inv(align @ camera_poses[frame_id])
    )
    source_files.extend((camera_pose, table_transform))
    unique_sources = sorted(set(source_files), key=str)
    return {
        "models_object_m": models,
        "dexnet_models": dexnet_models,
        "poses_object_to_camera": poses,
        "object_ids": object_ids,
        "dexnet_config": get_config(),
        "table_points_camera_m": table_camera,
        "source_files": [str(path) for path in unique_sources],
        "source_hashes": {str(path): _source_digest(path) for path in unique_sources},
        "dexnet_source_kind": "official_textured_obj_and_sdf_no_pickle",
    }


def _labels_resume_hit(
    output: Path,
    *,
    group_id: str,
    target_object_id: int,
    candidate_path: Path,
    candidate_bundle: Mapping[str, Any],
    candidate_sha256: str,
    parity_evidence: Mapping[str, Any],
    grounding_condition: str,
    resume: bool,
) -> bool:
    if not output.exists():
        return False
    if not resume:
        raise StageInputError(f"label bundle already exists: {output}; use resume=True")
    payload = _read_json(output, "candidate label bundle")
    if (
        payload.get("schema_version") != LABEL_BUNDLE_SCHEMA
        or payload.get("group_id") != group_id
    ):
        raise StageInputError(f"candidate label bundle schema/group mismatch: {output}")
    if payload.get("candidate_bundle_sha256") != candidate_sha256:
        raise StageInputError(
            f"candidate labels are stale for the frozen pool: {output}"
        )
    declared_candidate_path = Path(
        str(payload.get("candidate_bundle_path", ""))
    ).expanduser()
    if not declared_candidate_path.is_absolute():
        declared_candidate_path = output.parent / declared_candidate_path
    if declared_candidate_path.resolve() != candidate_path.resolve():
        raise StageInputError(f"candidate labels reference another bundle: {output}")
    if payload.get("target_object_id") != int(target_object_id):
        raise StageInputError(f"candidate labels reference another target: {output}")
    expected_ids = [
        str(record["candidate_id"]) for record in candidate_bundle["candidate_records"]
    ]
    if (
        payload.get("candidate_pool_fingerprint")
        != candidate_bundle.get("candidate_pool_fingerprint")
        or payload.get("candidate_count") != len(expected_ids)
        or payload.get("candidate_ids") != expected_ids
    ):
        raise StageInputError(
            f"candidate labels changed frozen membership/order: {output}"
        )
    if payload.get("grounding_condition") != grounding_condition:
        raise StageInputError(
            f"candidate labels belong to another grounding condition: {output}"
        )
    if payload.get("parity_gate") != dict(parity_evidence):
        raise StageInputError(
            f"candidate labels use another evaluator parity gate: {output}"
        )
    source_hashes = payload.get("official_source_hashes")
    if not isinstance(source_hashes, dict):
        raise StageInputError(
            f"candidate labels lack official source fingerprints: {output}"
        )
    skipped = payload.get("label_generation_status") == "skipped_empty_pool"
    if skipped:
        expected_reason = (
            "grounding_failure:" + str(candidate_bundle["grounding_failure_reason"])
            if candidate_bundle.get("generation_status") == "skipped_grounding_failure"
            else "vgn_no_candidates"
        )
        if (
            source_hashes != {}
            or payload.get("candidate_count") != 0
            or payload.get("labels") != []
            or payload.get("evaluator_calls_for_group") != 0
            or payload.get("empty_pool_reason") != expected_reason
        ):
            raise StageInputError(
                f"skipped empty-pool labels contain fabricated evaluator work: {output}"
            )
        terminal_fields = (
            "grounding_failure_reason",
            "grounding_terminal_path",
            "grounding_terminal_sha256",
        )
        if candidate_bundle.get("generation_status") == "skipped_grounding_failure":
            if any(
                payload.get(field) != candidate_bundle.get(field)
                for field in terminal_fields
            ):
                raise StageInputError(
                    f"skipped labels do not bind the grounding terminal: {output}"
                )
        elif any(field in payload for field in terminal_fields):
            raise StageInputError(
                f"zero-output VGN labels claim a grounding terminal: {output}"
            )
    elif not source_hashes:
        raise StageInputError(
            f"candidate labels lack official source fingerprints: {output}"
        )
    else:
        labels = payload.get("labels")
        if (
            payload.get("label_generation_status")
            != "completed_official_low_level_evaluation"
            or payload.get("evaluator_operation")
            != "per_candidate_low_level_no_eval_grasp_no_nms_no_topk"
            or payload.get("evaluator_calls_for_group") != 1
            or not isinstance(labels, list)
            or [str(item.get("candidate_id", "")) for item in labels] != expected_ids
            or any(
                int(item.get("target_object_id", -1)) != int(target_object_id)
                for item in labels
            )
        ):
            raise StageInputError(
                f"candidate labels violate official-evaluator semantics: {output}"
            )
        for index, (candidate_id, item) in enumerate(
            zip(expected_ids, labels, strict=True)
        ):
            if (
                not isinstance(item, Mapping)
                or item.get("candidate_id") != candidate_id
            ):
                raise StageInputError(
                    f"candidate labels changed evaluator row ordering: {output}"
                )

            def strict_int(field: str) -> int:
                value = item.get(field)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise StageInputError(
                        f"candidate label {field} must be an integer: {output}"
                    )
                return value

            def strict_bool(field: str) -> bool:
                value = item.get(field)
                if not isinstance(value, bool):
                    raise StageInputError(
                        f"candidate label {field} must be boolean: {output}"
                    )
                return value

            def strict_float(field: str) -> float:
                value = item.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise StageInputError(
                        f"candidate label {field} must be numeric: {output}"
                    )
                converted = float(value)
                if not np.isfinite(converted):
                    raise StageInputError(
                        f"candidate label {field} must be finite: {output}"
                    )
                return converted

            if strict_int("candidate_index") != index:
                raise StageInputError(
                    f"candidate labels changed candidate_index ordering: {output}"
                )
            strict_int("associated_instance_index")
            row_target_id = strict_int("target_object_id")
            associated_id = strict_int("associated_object_id")
            target_match = strict_bool("target_match")
            correct_target = strict_bool("correct_target")
            if (
                row_target_id != int(target_object_id)
                or target_match != (associated_id == int(target_object_id))
                or correct_target != target_match
            ):
                raise StageInputError(
                    f"candidate label association aliases disagree: {output}"
                )
            pose_valid = strict_bool("pose_valid")
            valid_geometry = strict_bool("valid_geometry")
            if pose_valid != valid_geometry:
                raise StageInputError(
                    f"candidate label pose-valid aliases disagree: {output}"
                )
            collision = strict_bool("collision")
            strict_bool("empty_grasp")
            friction = strict_float("friction_required")
            native_friction = strict_float("friction_score")
            if friction != native_friction:
                raise StageInputError(
                    f"candidate label friction aliases disagree: {output}"
                )
            relevance = strict_int("relevance")
            expected_relevance = relevance_from_friction(
                friction,
                correct_target=target_match,
                collision=collision,
                valid_geometry=pose_valid,
            )
            if relevance != expected_relevance:
                raise StageInputError(
                    f"candidate label relevance disagrees with raw outcomes: {output}"
                )
            for threshold in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
                reported_success = strict_bool(f"success_mu_{threshold:.1f}")
                expected_success = bool(
                    target_match
                    and pose_valid
                    and not collision
                    and 0 < friction <= threshold
                )
                if reported_success != expected_success:
                    raise StageInputError(
                        f"candidate label success alias disagrees with raw outcomes: {output}"
                    )
    for raw_path, expected in source_hashes.items():
        path = _require_regular_file(raw_path, "saved official evaluator source")
        if sha256_file(path) != expected:
            raise StageInputError(
                f"official evaluator source changed since labeling: {path}"
            )
    check = dict(payload)
    observed = check.pop("bundle_fingerprint", None)
    if observed != canonical_sha256(check):
        raise StageInputError(f"candidate label bundle fingerprint mismatch: {output}")
    if len(payload.get("labels", [])) != int(payload.get("candidate_count", -1)):
        raise StageInputError(f"candidate label count mismatch: {output}")
    return True


def run_official_label_stage(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    output_root: Path | str,
    *,
    dataset_root: Path | str,
    parity_gate_path: Path | str,
    grounding_condition: str = "oracle_gt_mask",
    api_root: Path | str = DEFAULT_GRASPNET_API_ROOT,
    resume: bool = False,
    evidence_policy: str = "formal",
    input_loader: Callable[..., Mapping[str, Any]] = _official_group_evaluator_inputs,
    evaluator: Callable[..., Sequence[Any]] = evaluate_frozen_candidates,
) -> StageSummary:
    """Label every frozen row with official low-level evaluator operations.

    This stage never invokes the official high-level ``eval_grasp`` function,
    whose NMS and Top-K operations would mutate frozen candidate membership.
    """

    stage = "official_labels"
    root = Path(output_root).expanduser().resolve()
    data = Path(dataset_root).expanduser().resolve()
    condition = str(grounding_condition).strip()
    if condition not in {
        "oracle_gt_mask",
        "hifics_zero_shot_mask",
        "hifics_adapted_mask",
    }:
        raise StageInputError(f"unsupported grounding condition: {condition!r}")
    groups = load_target_language_jsonl(target_manifest_path, language_manifest_path)
    failures: list[GroupFailure] = []
    try:
        if not data.is_dir():
            raise StageInputError(f"official GraspNet dataset root is absent: {data}")
        parity_gate, parity_evidence = load_evaluator_parity_gate(
            parity_gate_path, evidence_policy=evidence_policy
        )
    except Exception as error:
        for group in groups:
            failures.append(
                _record_failure(root, stage=stage, group_id=group.group_id, error=error)
            )
        _raise_failures(stage, failures)
        raise AssertionError("unreachable")
    outputs: list[str] = []
    resumed = 0
    for group in groups:
        try:
            candidate_path = _require_regular_file(
                _candidate_bundle_path(root, group.group_id, condition),
                "frozen candidate bundle",
            )
            candidate_bundle = _load_candidate_bundle(candidate_path, group.group_id)
            if candidate_bundle["grounding_condition"] != condition:
                raise StageInputError(
                    "candidate bundle belongs to another grounding condition"
                )
            candidate_sha = sha256_file(candidate_path)
            output = root / stage / condition / f"{_slug(group.group_id)}.json"
            if _labels_resume_hit(
                output,
                group_id=group.group_id,
                target_object_id=int(group.target["target_object_id"]),
                candidate_path=candidate_path,
                candidate_bundle=candidate_bundle,
                candidate_sha256=candidate_sha,
                parity_evidence=parity_evidence,
                grounding_condition=condition,
                resume=resume,
            ):
                resumed += 1
                outputs.append(str(output))
                continue
            if candidate_bundle.get("candidate_count") == 0:
                if candidate_bundle.get("_rows_array", np.empty((1, 17))).shape != (
                    0,
                    17,
                ):
                    raise StageInputError(
                        "zero-candidate bundle is not an exact empty pool"
                    )
                grounding_skipped = (
                    candidate_bundle.get("generation_status")
                    == "skipped_grounding_failure"
                )
                payload = {
                    "schema_version": LABEL_BUNDLE_SCHEMA,
                    "group_id": group.group_id,
                    "target_object_id": int(group.target["target_object_id"]),
                    "candidate_bundle_path": str(candidate_path),
                    "candidate_bundle_sha256": candidate_sha,
                    "candidate_pool_fingerprint": candidate_bundle[
                        "candidate_pool_fingerprint"
                    ],
                    "grounding_condition": candidate_bundle["grounding_condition"],
                    "candidate_count": 0,
                    "candidate_ids": [],
                    "labels": [],
                    "label_generation_status": "skipped_empty_pool",
                    "empty_pool_reason": (
                        "grounding_failure:"
                        + str(candidate_bundle["grounding_failure_reason"])
                        if grounding_skipped
                        else "vgn_no_candidates"
                    ),
                    "official_source_hashes": {},
                    "dexnet_source_kind": "not_loaded_empty_frozen_pool",
                    "evaluator_operation": "not_called_empty_frozen_pool",
                    "evaluator_calls_for_group": 0,
                    "parity_gate": dict(parity_evidence),
                }
                if grounding_skipped:
                    payload.update(
                        {
                            "grounding_failure_reason": candidate_bundle[
                                "grounding_failure_reason"
                            ],
                            "grounding_terminal_path": candidate_bundle[
                                "grounding_terminal_path"
                            ],
                            "grounding_terminal_sha256": candidate_bundle[
                                "grounding_terminal_sha256"
                            ],
                        }
                    )
                payload["bundle_fingerprint"] = canonical_sha256(payload)
                atomic_json(output, payload)
                outputs.append(str(output))
                continue
            inputs = dict(input_loader(group, dataset_root=data, api_root=api_root))
            required = {
                "models_object_m",
                "dexnet_models",
                "poses_object_to_camera",
                "object_ids",
                "dexnet_config",
                "table_points_camera_m",
                "source_hashes",
                "dexnet_source_kind",
            }
            missing = sorted(required - set(inputs))
            if missing:
                raise StageInputError(
                    f"official evaluator loader omitted inputs: {missing}"
                )
            if (
                not isinstance(inputs["source_hashes"], Mapping)
                or not inputs["source_hashes"]
            ):
                raise StageInputError("official evaluator inputs lack source hashes")
            for raw_path, expected in inputs["source_hashes"].items():
                source = _require_regular_file(raw_path, "official evaluator source")
                if _source_digest(source) != expected:
                    raise StageInputError(
                        f"official source fingerprint is stale: {source}"
                    )
            rows = candidate_bundle.pop("_rows_array")
            evaluations = list(
                evaluator(
                    rows,
                    models_object_m=inputs["models_object_m"],
                    dexnet_models=inputs["dexnet_models"],
                    poses_object_to_camera=inputs["poses_object_to_camera"],
                    object_ids=inputs["object_ids"],
                    target_object_id=int(group.target["target_object_id"]),
                    dexnet_config=inputs["dexnet_config"],
                    table_points_camera_m=inputs["table_points_camera_m"],
                    api_root=api_root,
                    parity_gate=parity_gate,
                    validation_probe=False,
                )
            )
            if len(evaluations) != len(rows):
                raise StageInputError(
                    "official evaluator changed frozen candidate membership"
                )
            candidate_ids = [
                record["candidate_id"]
                for record in candidate_bundle["candidate_records"]
            ]
            labels: list[dict[str, Any]] = []
            for index, (candidate_id, evaluation) in enumerate(
                zip(candidate_ids, evaluations, strict=True)
            ):
                if int(evaluation.candidate_index) != index:
                    raise StageInputError(
                        "official evaluator changed frozen candidate ordering"
                    )
                record = evaluation.to_record()
                if int(record["target_object_id"]) != int(
                    group.target["target_object_id"]
                ):
                    raise StageInputError(
                        "official evaluator returned the wrong target identity"
                    )
                labels.append({"candidate_id": candidate_id, **record})
            payload: dict[str, Any] = {
                "schema_version": LABEL_BUNDLE_SCHEMA,
                "group_id": group.group_id,
                "target_object_id": int(group.target["target_object_id"]),
                "candidate_bundle_path": str(candidate_path),
                "candidate_bundle_sha256": candidate_sha,
                "candidate_pool_fingerprint": candidate_bundle[
                    "candidate_pool_fingerprint"
                ],
                "grounding_condition": candidate_bundle["grounding_condition"],
                "candidate_count": len(candidate_ids),
                "candidate_ids": candidate_ids,
                "labels": labels,
                "official_source_hashes": dict(inputs["source_hashes"]),
                "dexnet_source_kind": str(inputs["dexnet_source_kind"]),
                "evaluator_operation": "per_candidate_low_level_no_eval_grasp_no_nms_no_topk",
                "label_generation_status": "completed_official_low_level_evaluation",
                "evaluator_calls_for_group": 1,
                "parity_gate": dict(parity_evidence),
            }
            payload["bundle_fingerprint"] = canonical_sha256(payload)
            atomic_json(output, payload)
            outputs.append(str(output))
        except Exception as error:
            failures.append(
                _record_failure(root, stage=stage, group_id=group.group_id, error=error)
            )
    _raise_failures(stage, failures)
    return StageSummary(
        stage, len(groups), len(groups) - resumed, resumed, tuple(outputs)
    )


def _read_table(path: Path | str, description: str) -> pd.DataFrame:
    source = _require_regular_file(path, description)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        frame = pd.DataFrame(load_jsonl_records(source, description=description))
    elif suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise StageInputError(f"unsupported {description} format: {source.suffix}")
    if frame.empty:
        raise StageInputError(
            f"{description} is empty; refusing placeholder rows: {source}"
        )
    return frame


def _write_table(path: Path | str, frame: pd.DataFrame) -> Path:
    destination = Path(path).expanduser().resolve()
    if frame.empty:
        raise StageInputError("refusing to publish an empty experiment table")
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".jsonl":
        atomic_jsonl(destination, frame.to_dict(orient="records"))
        return destination
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        if suffix == ".csv":
            frame.to_csv(temporary, index=False)
        elif suffix in {".parquet", ".pq"}:
            frame.to_parquet(temporary, index=False)
        else:
            raise StageInputError(
                f"unsupported experiment-table output format: {suffix}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def assemble_labeled_feature_rows(
    feature_table_path: Path | str,
    label_bundle_paths: Iterable[Path | str],
    output_path: Path | str,
) -> Path:
    """Strictly attach real evaluator labels to an identical feature universe."""

    features = _read_table(feature_table_path, "candidate feature table")
    required = {"group_id", "candidate_id"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise StageInputError(f"candidate feature table lacks columns: {missing}")
    if features[list(required)].astype(str).duplicated().any():
        raise StageInputError(
            "candidate feature table contains duplicate group/candidate rows"
        )
    label_rows: list[dict[str, Any]] = []
    for raw_path in label_bundle_paths:
        payload = _read_json(raw_path, "candidate label bundle")
        if payload.get("schema_version") != LABEL_BUNDLE_SCHEMA:
            raise StageInputError(f"unsupported candidate label bundle: {raw_path}")
        labels = payload.get("labels")
        if not isinstance(labels, list) or len(labels) != int(
            payload.get("candidate_count", -1)
        ):
            raise StageInputError(f"candidate label bundle is incomplete: {raw_path}")
        label_rows.extend({"group_id": payload["group_id"], **row} for row in labels)
    labels = pd.DataFrame(label_rows)
    if labels.empty:
        raise StageInputError("no real evaluator label rows were provided")
    if labels[["group_id", "candidate_id"]].astype(str).duplicated().any():
        raise StageInputError("candidate label bundles contain duplicate rows")
    feature_keys = set(
        map(tuple, features[["group_id", "candidate_id"]].astype(str).to_numpy())
    )
    label_keys = set(
        map(tuple, labels[["group_id", "candidate_id"]].astype(str).to_numpy())
    )
    if feature_keys != label_keys:
        raise StageInputError(
            "feature/label candidate universes differ; refusing an inner join that would hide rows"
        )
    result = features.merge(
        labels,
        on=["group_id", "candidate_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_official"),
    )
    if len(result) != len(features) or result["relevance"].isna().any():
        raise AssertionError("strict feature/label merge lost a real candidate")
    return _write_table(output_path, result)


def run_train_ranker_hook(
    training_rows_path: Path | str,
    validation_rows_path: Path | str,
    output_root: Path | str,
    *,
    split_manifest_path: Path | str,
    feature_columns: Sequence[str],
    config: Mapping[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    """Fit the existing graded ranker from complete, non-empty real rows."""

    train_path = _require_regular_file(training_rows_path, "training rows")
    validation_path = _require_regular_file(validation_rows_path, "validation rows")
    split_path = _require_regular_file(split_manifest_path, "locked scene split")
    split = SceneSplit.from_dict(_read_json(split_path, "locked scene split"))
    columns = assert_no_gt_leakage(feature_columns)
    train = _read_table(train_path, "training rows")
    validation = _read_table(validation_path, "validation rows")
    required = {"group_id", "candidate_id", "scene_id", "split", "relevance", *columns}
    for name, frame in (("training", train), ("validation", validation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise StageInputError(f"{name} rows lack columns: {missing}")
        if frame["candidate_id"].astype(str).duplicated().any():
            raise StageInputError(f"{name} rows contain duplicate candidate IDs")
    if set(train["split"].astype(str)) != {"train"}:
        raise StageInputError(
            "training rows must contain only split='train'; test access is prohibited"
        )
    if set(validation["split"].astype(str)) != {"validation"}:
        raise StageInputError(
            "validation rows must contain only split='validation'; test access is prohibited"
        )
    train_scenes = set(train["scene_id"].astype(str))
    validation_scenes = set(validation["scene_id"].astype(str))
    if train_scenes & validation_scenes:
        raise StageInputError("training and validation scenes overlap")
    if not train_scenes.issubset(set(split.train)):
        raise StageInputError(
            "training rows contain scenes outside the locked training split"
        )
    if not validation_scenes.issubset(set(split.validation)):
        raise StageInputError(
            "validation rows contain scenes outside the locked validation split"
        )
    if set(train["group_id"].astype(str)) & set(validation["group_id"].astype(str)):
        raise StageInputError("training and validation groups overlap")
    if set(train["candidate_id"].astype(str)) & set(
        validation["candidate_id"].astype(str)
    ):
        raise StageInputError("training and validation candidates overlap")
    input_fingerprint = canonical_sha256(
        {
            "training_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(validation_path),
            "split_manifest_sha256": sha256_file(split_path),
            "feature_columns": columns,
            "config": dict(config),
        }
    )
    root = Path(output_root).expanduser().resolve()
    model_path = root / "ranker" / "model.txt"
    artifact_path = root / "ranker" / "model.json"
    if artifact_path.exists():
        if not resume:
            raise StageInputError(f"ranker artifact already exists: {artifact_path}")
        artifact = _read_json(artifact_path, "ranker artifact")
        if (
            artifact.get("input_fingerprint") != input_fingerprint
            or not model_path.is_file()
            or artifact.get("model_sha256") != sha256_file(model_path)
        ):
            raise StageInputError("saved ranker is stale or corrupt")
        return artifact

    sort_columns = (
        ["group_id"]
        + (["native_rank"] if "native_rank" in train else [])
        + ["candidate_id"]
    )
    train = train.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    validation_sort = (
        ["group_id"]
        + (["native_rank"] if "native_rank" in validation else [])
        + ["candidate_id"]
    )
    validation = validation.sort_values(validation_sort, kind="mergesort").reset_index(
        drop=True
    )
    imputer = StableMissingValueImputer()
    train_features = imputer.fit_transform(train[list(columns)])
    validation_features = imputer.transform(validation[list(columns)])
    train_groups = contiguous_group_sizes(train["group_id"], length=len(train))
    validation_groups = contiguous_group_sizes(
        validation["group_id"], length=len(validation)
    )
    model = fit_ranker(
        train_features.to_numpy(float),
        train["relevance"].to_numpy(),
        train_groups,
        ValidationData(
            validation_features.to_numpy(float),
            validation["relevance"].to_numpy(),
            validation_groups,
        ),
        dict(config),
    )
    if model.model is None:
        raise RuntimeError("ranker fit returned no model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    try:
        model.model.booster_.save_model(str(temporary))
        os.replace(temporary, model_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    artifact = {
        "schema_version": "graspnet6d_ranker_artifact_v1",
        "input_fingerprint": input_fingerprint,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "ranker": model.artifact(),
        "imputer": imputer.artifact(),
        "training_row_count": len(train),
        "validation_row_count": len(validation),
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "training_candidate_fingerprint": canonical_sha256(
            train["candidate_id"].astype(str).tolist()
        ),
        "validation_candidate_fingerprint": canonical_sha256(
            validation["candidate_id"].astype(str).tolist()
        ),
    }
    atomic_json(artifact_path, artifact)
    return artifact


def run_evaluate_hook(
    raw_prediction_rows_path: Path | str,
    group_universe_path: Path | str,
    output_root: Path | str,
    *,
    score_column: str,
    max_k: int = 50,
    resume: bool = False,
) -> dict[str, Any]:
    """Recompute metrics only from supplied raw evaluator/prediction rows."""

    raw_path = _require_regular_file(raw_prediction_rows_path, "raw prediction rows")
    universe_path = _require_regular_file(group_universe_path, "group universe")
    raw = _read_table(raw_path, "raw prediction rows")
    universe = _read_table(universe_path, "group universe")
    input_fingerprint = canonical_sha256(
        {
            "raw_prediction_rows_sha256": sha256_file(raw_path),
            "group_universe_sha256": sha256_file(universe_path),
            "score_column": str(score_column),
            "max_k": int(max_k),
        }
    )
    root = Path(output_root).expanduser().resolve() / "evaluation"
    artifact_path = root / "metrics.json"
    if artifact_path.exists():
        if not resume:
            raise StageInputError(
                f"evaluation artifact already exists: {artifact_path}"
            )
        saved = _read_json(artifact_path, "evaluation artifact")
        saved_per_group = _require_regular_file(
            saved.get("per_group_path", ""), "saved per-group metrics"
        )
        if saved.get("input_fingerprint") != input_fingerprint or saved.get(
            "per_group_sha256"
        ) != sha256_file(saved_per_group):
            raise StageInputError("saved evaluation is stale or corrupt")
        return saved
    metrics, per_group = evaluate_target_rankings(
        raw, universe, score_column=score_column, max_k=max_k
    )
    per_group_path = root / "per_group.jsonl"
    atomic_jsonl(per_group_path, per_group.to_dict(orient="records"))
    payload = {
        "schema_version": "graspnet6d_evaluation_v1",
        "input_fingerprint": input_fingerprint,
        "raw_prediction_rows_sha256": sha256_file(raw_path),
        "group_universe_sha256": sha256_file(universe_path),
        "score_column": score_column,
        "max_k": int(max_k),
        "metrics": metrics,
        "per_group_path": str(per_group_path),
        "per_group_sha256": sha256_file(per_group_path),
    }
    atomic_json(artifact_path, payload)
    return payload


__all__ = [
    "GEOMETRY_SCHEMA",
    "GroupFailure",
    "GroupManifest",
    "GROUNDING_TERMINAL_SCHEMA",
    "LABEL_BUNDLE_SCHEMA",
    "MASK_SCHEMA",
    "PARITY_SCHEMA",
    "StageBatchError",
    "StageInputError",
    "StageSummary",
    "TSDF_SCHEMA",
    "VGN_BUNDLE_SCHEMA",
    "assemble_labeled_feature_rows",
    "grounding_terminal_path",
    "load_evaluator_geometry_contract",
    "load_evaluator_parity_gate",
    "load_frozen_candidate_bundle",
    "load_grounding_terminal",
    "load_jsonl_records",
    "load_target_language_jsonl",
    "run_evaluate_hook",
    "run_official_label_stage",
    "run_oracle_mask_stage",
    "run_train_ranker_hook",
    "run_tsdf_stage",
    "run_vgn_candidate_stage",
]
