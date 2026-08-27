"""Fail-closed predicted masks and runtime-only formal feature tables.

The expensive stages in this module publish one group at a time.  Data files
are written atomically and a JSON sidecar is published last as the commit
marker.  A consumer never infers provenance from a filename: it re-hashes the
source manifests, RGB/depth files, model weights, mask archive, frozen
candidate bundle, and the committed output before accepting a cache hit.

Ground-truth instance images are intentionally not opened by either public
stage.  Predicted conditions accept only the schema emitted here, so an oracle
mask cache cannot be relabelled or accidentally routed into a predicted track.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat

from .contracts import Candidate6D, candidate_pool_fingerprint, validate_candidate_pool
from .feature_extraction import RuntimeObservation, extract_candidate_features
from .features import default_feature_schema, feature_schema_sha256
from .geometry import CameraIntrinsics, as_transform, compose_transforms
from .grounding import (
    DEFAULT_CLIP_WEIGHT,
    DEFAULT_HIFI_CHECKPOINT,
    EXPECTED_CLIP_WEIGHT_SHA256,
    EXPECTED_HIFI_CHECKPOINT_SHA256,
    FOREGROUND_THRESHOLD,
    IMAGE_RESOLUTION,
    GroundingPrediction,
    HiFiModelBundle,
    load_hifi_model,
    predict_grounding,
    save_grounding_prediction,
    validate_adaptation_splits,
)
from .io import atomic_json, canonical_sha256, sha256_file
from .stages import (
    MASK_SCHEMA,
    TSDF_SCHEMA,
    VGN_BUNDLE_SCHEMA,
    GroupFailure,
    GroupManifest,
    StageBatchError,
    StageInputError,
    StageSummary,
    _decode_a7_pre_nms_contract,
    _validate_raw_to_evaluator_geometry,
    load_grounding_terminal,
    load_jsonl_records,
    load_target_language_jsonl,
)
from .vgn import ExtractionConfig, vgn_candidate_from_record


PREDICTED_MASK_SCHEMA = "graspnet6d_hifics_predicted_mask_v1"
FORMAL_FEATURE_SCHEMA = "graspnet6d_runtime_feature_table_v1"
ADAPTATION_EVIDENCE_SCHEMA = "graspnet6d_hifi_adaptation_evidence_v1"

GroundingCondition = Literal[
    "oracle_gt_mask", "hifics_zero_shot_mask", "hifics_adapted_mask"
]
PredictedCondition = Literal["hifics_zero_shot_mask", "hifics_adapted_mask"]
SceneCloudKind = Literal[
    "full_depth_backprojection",
    "deterministic_voxel_downsample_of_full_depth",
]

PREDICTED_CONDITIONS: frozenset[str] = frozenset(
    {"hifics_zero_shot_mask", "hifics_adapted_mask"}
)
GROUNDING_CONDITIONS: frozenset[str] = frozenset(
    {"oracle_gt_mask", *PREDICTED_CONDITIONS}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ValidatedAdaptationEvidence:
    """A decoder checkpoint bound to independently checked split evidence."""

    evidence_path: Path
    evidence_sha256: str
    evidence_fingerprint: str
    checkpoint_path: Path
    checkpoint_sha256: str
    train_manifest_path: Path
    train_manifest_sha256: str
    validation_manifest_path: Path
    validation_manifest_sha256: str
    best_validation_mean_iou: float
    split_contract: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CommittedPredictedMask:
    """A fully revalidated predicted probability cache."""

    probability: np.ndarray
    binary_mask: np.ndarray
    foreground_threshold: float
    condition: PredictedCondition
    group_id: str
    probability_path: Path
    mask_path: Path
    sidecar_path: Path
    sidecar: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CommittedOracleMask:
    """A revalidated official instance mask for the named oracle condition."""

    probability: np.ndarray
    binary_mask: np.ndarray
    foreground_threshold: float
    condition: Literal["oracle_gt_mask"]
    group_id: str
    probability_path: Path
    mask_path: Path
    sidecar_path: Path
    sidecar: Mapping[str, Any]


def _require_regular_file(path: str | os.PathLike[str], description: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise StageInputError(f"{description} must not be a symlink: {raw}")
    source = raw.resolve()
    if not source.is_file():
        raise StageInputError(f"missing regular {description}: {source}")
    return source


def _read_json(path: str | os.PathLike[str], description: str) -> dict[str, Any]:
    source = _require_regular_file(path, description)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageInputError(f"invalid {description}: {source}: {error}") from error
    if not isinstance(value, dict):
        raise StageInputError(f"{description} must be a JSON object: {source}")
    return value


def _digest(value: Any, name: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise StageInputError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _predicted_condition(value: str) -> PredictedCondition:
    normalized = str(value).strip()
    if normalized not in PREDICTED_CONDITIONS:
        raise StageInputError(
            f"predicted feature tracks require one of {sorted(PREDICTED_CONDITIONS)}, "
            f"received {value!r}"
        )
    return normalized  # type: ignore[return-value]


def _grounding_condition(value: str) -> GroundingCondition:
    normalized = str(value).strip()
    if normalized not in GROUNDING_CONDITIONS:
        raise StageInputError(
            f"grounding condition must be one of {sorted(GROUNDING_CONDITIONS)}, "
            f"received {value!r}"
        )
    return normalized  # type: ignore[return-value]


def group_artifact_slug(group_id: str) -> str:
    """Return the stable collision-resistant basename shared by both stages."""

    normalized = str(group_id).strip()
    if not normalized:
        raise StageInputError("group_id must be non-empty")
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip("._") or "group"
    return f"{prefix[:96]}-{canonical_sha256(normalized)[:12]}"


def _record_group_failure(
    output_root: Path,
    *,
    stage: str,
    group_id: str,
    error: BaseException,
) -> GroupFailure:
    path = output_root / "errors" / stage / f"{group_artifact_slug(group_id)}.json"
    atomic_json(
        path,
        {
            "schema_version": 1,
            "stage": stage,
            "group_id": group_id,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return GroupFailure(
        stage=stage,
        group_id=group_id,
        error_type=type(error).__name__,
        message=str(error),
        error_path=str(path),
    )


def predicted_mask_paths(
    output_root: str | os.PathLike[str], condition: str, group_id: str
) -> tuple[Path, Path, Path]:
    selected = _predicted_condition(condition)
    stem = Path(output_root).expanduser().resolve() / "predicted_masks" / selected
    slug = group_artifact_slug(group_id)
    return stem / f"{slug}.npz", stem / f"{slug}.png", stem / f"{slug}.json"


def formal_feature_paths(
    output_root: str | os.PathLike[str], condition: str, group_id: str
) -> tuple[Path, Path]:
    selected = _grounding_condition(condition)
    stem = Path(output_root).expanduser().resolve() / "candidate_features" / selected
    slug = group_artifact_slug(group_id)
    return stem / f"{slug}.parquet", stem / f"{slug}.json"


def oracle_mask_paths(
    output_root: str | os.PathLike[str], group_id: str
) -> tuple[Path, Path]:
    stem = Path(output_root).expanduser().resolve() / "oracle_masks"
    slug = group_artifact_slug(group_id)
    return stem / f"{slug}.npz", stem / f"{slug}.json"


def _resolve_manifest_member(
    evidence_path: Path, raw_path: Any, description: str
) -> Path:
    value = Path(str(raw_path)).expanduser()
    if not value.is_absolute():
        value = evidence_path.parent / value
    return _require_regular_file(value, description)


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise StageInputError(
            f"cannot safely load adaptation checkpoint {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise StageInputError("adaptation checkpoint must contain a mapping")
    return payload


def _validate_adaptation_row_sources(
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    *,
    role: str,
) -> dict[str, str]:
    """Re-hash the exact RGB and instance pixels consumed by adaptation."""

    observed: dict[str, str] = {}
    forbidden_alternate_targets = {"target", "target_mask", "mask_path", "gt_mask_path"}
    for index, row in enumerate(rows):
        if forbidden_alternate_targets.intersection(row):
            raise StageInputError(
                f"{role} adaptation row {index} bypasses the hashed official instance label"
            )
        row_paths: dict[str, Path] = {}
        for path_field, hash_field, description in (
            ("rgb_path", "rgb_sha256", "adaptation RGB image"),
            (
                "instance_label_path",
                "instance_label_sha256",
                "adaptation instance-label image",
            ),
        ):
            raw = Path(str(row.get(path_field, ""))).expanduser()
            if not raw.is_absolute():
                raw = manifest_path.parent / raw
            path = _require_regular_file(raw, description)
            expected = _digest(row.get(hash_field), f"{role} row {index} {hash_field}")
            if sha256_file(path) != expected:
                raise StageInputError(
                    f"{role} adaptation row {index} has stale {path_field}"
                )
            observed[str(path)] = expected
            row_paths[path_field] = path
        raw_label = row.get("target_instance_label")
        if isinstance(raw_label, bool):
            raise StageInputError("target_instance_label must be a positive integer")
        try:
            instance_label = int(raw_label)
        except (TypeError, ValueError) as error:
            raise StageInputError(
                f"{role} adaptation row {index} lacks target_instance_label"
            ) from error
        if instance_label <= 0:
            raise StageInputError("target_instance_label must be a positive integer")
        rgb_path = row_paths["rgb_path"]
        label_path = row_paths["instance_label_path"]
        with Image.open(rgb_path) as image:
            rgb_size = (image.height, image.width)
        with Image.open(label_path) as image:
            labels = np.asarray(image)
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise StageInputError("adaptation instance-label image must be integer HxW")
        if labels.shape != rgb_size:
            raise StageInputError("adaptation RGB/instance-label dimensions differ")
        if not bool(np.any(labels == instance_label)):
            raise StageInputError(
                f"adaptation target instance {instance_label} is absent from its label image"
            )
        if not str(row.get("query", row.get("text", ""))).strip():
            raise StageInputError(
                f"{role} adaptation row {index} has no language query"
            )
    return observed


def validate_adaptation_evidence(
    evidence_path: str | os.PathLike[str],
    *,
    adapted_checkpoint_path: str | os.PathLike[str] | None = None,
) -> ValidatedAdaptationEvidence:
    """Validate all evidence needed before an adapted mask may be inferred.

    The two row manifests are re-read and passed through the same strict split
    validator used during adaptation.  That proves that only explicit train
    and validation rows were supplied, their scenes/identities are disjoint,
    and no test row was available to optimization or model selection.
    """

    evidence_source = _require_regular_file(evidence_path, "adaptation evidence")
    evidence = _read_json(evidence_source, "adaptation evidence")
    if evidence.get("schema_version") != ADAPTATION_EVIDENCE_SCHEMA:
        raise StageInputError(
            f"unsupported adaptation evidence schema: {evidence.get('schema_version')!r}"
        )
    check = dict(evidence)
    observed_fingerprint = _digest(
        check.pop("evidence_fingerprint", ""), "evidence_fingerprint"
    )
    if observed_fingerprint != canonical_sha256(check):
        raise StageInputError("adaptation evidence fingerprint mismatch")

    recorded_checkpoint = _resolve_manifest_member(
        evidence_source,
        evidence.get("adaptation_checkpoint_path", ""),
        "adaptation checkpoint",
    )
    if adapted_checkpoint_path is not None:
        requested = _require_regular_file(
            adapted_checkpoint_path, "requested adaptation checkpoint"
        )
        if requested != recorded_checkpoint:
            raise StageInputError(
                "requested adaptation checkpoint differs from the evidence manifest"
            )
    checkpoint_digest = sha256_file(recorded_checkpoint)
    if checkpoint_digest != _digest(
        evidence.get("adaptation_checkpoint_sha256"),
        "adaptation_checkpoint_sha256",
    ):
        raise StageInputError("adaptation checkpoint SHA-256 mismatch")

    train_manifest = _resolve_manifest_member(
        evidence_source,
        evidence.get("train_manifest_path", ""),
        "adaptation train manifest",
    )
    validation_manifest = _resolve_manifest_member(
        evidence_source,
        evidence.get("validation_manifest_path", ""),
        "adaptation validation manifest",
    )
    train_digest = sha256_file(train_manifest)
    validation_digest = sha256_file(validation_manifest)
    if train_digest != _digest(
        evidence.get("train_manifest_sha256"), "train_manifest_sha256"
    ):
        raise StageInputError("adaptation train manifest SHA-256 mismatch")
    if validation_digest != _digest(
        evidence.get("validation_manifest_sha256"),
        "validation_manifest_sha256",
    ):
        raise StageInputError("adaptation validation manifest SHA-256 mismatch")

    train_rows = load_jsonl_records(
        train_manifest, description="adaptation train manifest"
    )
    validation_rows = load_jsonl_records(
        validation_manifest, description="adaptation validation manifest"
    )
    adaptation_source_hashes = {
        **_validate_adaptation_row_sources(train_rows, train_manifest, role="train"),
        **_validate_adaptation_row_sources(
            validation_rows, validation_manifest, role="validation"
        ),
    }
    try:
        independently_observed_contract = dict(
            validate_adaptation_splits(train_rows, validation_rows)
        )
    except (TypeError, ValueError) as error:
        raise StageInputError(f"invalid adaptation split evidence: {error}") from error

    checkpoint = _checkpoint_payload(recorded_checkpoint)
    if checkpoint.get("format") != "graspnet6d_hifi_decoder_adaptation_v1":
        raise StageInputError("unsupported adapted HiFi checkpoint format")
    if checkpoint.get("base_checkpoint_sha256") != EXPECTED_HIFI_CHECKPOINT_SHA256:
        raise StageInputError(
            "adaptation checkpoint is not based on the retained HiFi model"
        )
    if checkpoint.get("clip_weight_sha256") != EXPECTED_CLIP_WEIGHT_SHA256:
        raise StageInputError(
            "adaptation checkpoint is not bound to the retained CLIP weight"
        )
    state = checkpoint.get("trainable_state")
    metadata = checkpoint.get("metadata")
    if not isinstance(state, Mapping) or not state:
        raise StageInputError("adaptation checkpoint has no trainable_state")
    if not isinstance(metadata, Mapping):
        raise StageInputError("adaptation checkpoint has no metadata")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise StageInputError(
                "adaptation state must map parameter names to tensors"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise StageInputError(f"non-finite adapted parameter: {name}")

    selection_metric = str(metadata.get("selection_metric", ""))
    selection_split = str(metadata.get("selection_split", ""))
    if selection_metric != "validation_mean_iou" or selection_split != "val":
        raise StageInputError("adaptation was not selected solely by validation mIoU")
    if int(metadata.get("test_rows_consumed", -1)) != 0:
        raise StageInputError("adaptation checkpoint reports consuming test rows")
    checkpoint_contract = metadata.get("split_contract")
    if not isinstance(checkpoint_contract, Mapping):
        raise StageInputError("adaptation checkpoint has no split_contract")
    if dict(checkpoint_contract) != independently_observed_contract:
        raise StageInputError(
            "adaptation checkpoint split contract disagrees with the hashed row manifests"
        )
    if (
        evidence.get("selection_metric") != "validation_mean_iou"
        or evidence.get("selection_split") != "val"
    ):
        raise StageInputError(
            "adaptation evidence does not declare validation-mIoU selection"
        )
    if int(evidence.get("test_rows_consumed", -1)) != 0:
        raise StageInputError("adaptation evidence does not prove test exclusion")
    if evidence.get("input_splits") != ["train", "val"]:
        raise StageInputError(
            "adaptation evidence input_splits must be exactly ['train', 'val']"
        )
    if dict(evidence.get("split_contract") or {}) != independently_observed_contract:
        raise StageInputError("adaptation evidence split contract mismatch")
    if evidence.get("base_checkpoint_sha256") != EXPECTED_HIFI_CHECKPOINT_SHA256:
        raise StageInputError("adaptation evidence base checkpoint mismatch")
    if evidence.get("clip_weight_sha256") != EXPECTED_CLIP_WEIGHT_SHA256:
        raise StageInputError("adaptation evidence CLIP weight mismatch")

    best_iou = float(metadata.get("best_validation_mean_iou", float("nan")))
    if not np.isfinite(best_iou) or not 0.0 <= best_iou <= 1.0:
        raise StageInputError("adaptation checkpoint has invalid best validation mIoU")
    if float(evidence.get("best_validation_mean_iou", float("nan"))) != best_iou:
        raise StageInputError("adaptation evidence best validation mIoU mismatch")
    if int(metadata.get("best_epoch", -1)) < 0:
        raise StageInputError("adaptation checkpoint has no validation-selected epoch")
    for raw_path, expected in adaptation_source_hashes.items():
        if (
            sha256_file(_require_regular_file(raw_path, "adaptation source"))
            != expected
        ):
            raise StageInputError(
                f"adaptation source changed during validation: {raw_path}"
            )

    return ValidatedAdaptationEvidence(
        evidence_path=evidence_source,
        evidence_sha256=sha256_file(evidence_source),
        evidence_fingerprint=observed_fingerprint,
        checkpoint_path=recorded_checkpoint,
        checkpoint_sha256=checkpoint_digest,
        train_manifest_path=train_manifest,
        train_manifest_sha256=train_digest,
        validation_manifest_path=validation_manifest,
        validation_manifest_sha256=validation_digest,
        best_validation_mean_iou=best_iou,
        split_contract=MappingProxyType(independently_observed_contract),
    )


def load_validated_adapted_hifi(
    evidence_path: str | os.PathLike[str],
    *,
    adapted_checkpoint_path: str | os.PathLike[str] | None = None,
    device: str = "cpu",
    base_checkpoint_path: str | os.PathLike[str] = DEFAULT_HIFI_CHECKPOINT,
    clip_weight_path: str | os.PathLike[str] = DEFAULT_CLIP_WEIGHT,
) -> tuple[HiFiModelBundle, ValidatedAdaptationEvidence]:
    """Load, validate, apply, and freeze a validation-selected decoder state."""

    evidence = validate_adaptation_evidence(
        evidence_path, adapted_checkpoint_path=adapted_checkpoint_path
    )
    bundle = load_hifi_model(
        device=device,
        mode="zero_shot",
        checkpoint_path=base_checkpoint_path,
        clip_weight_path=clip_weight_path,
    )
    checkpoint = _checkpoint_payload(evidence.checkpoint_path)
    state = checkpoint["trainable_state"]
    expected_names = set(bundle.adaptation_parameter_names)
    observed_names = set(map(str, state))
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise StageInputError(
            "adapted state does not match the retained decoder: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model_state = bundle.model.state_dict()
    for name in bundle.adaptation_parameter_names:
        tensor = state[name]
        if tensor.shape != model_state[name].shape:
            raise StageInputError(f"adapted tensor shape mismatch: {name}")
        model_state[name] = tensor
    incompatible = bundle.model.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise StageInputError("strict adapted decoder load failed")
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    bundle.model.eval()
    bundle.model.clip_model.eval()
    adapted_metadata = {
        "adaptation_evidence_path": str(evidence.evidence_path),
        "adaptation_evidence_sha256": evidence.evidence_sha256,
        "adaptation_evidence_fingerprint": evidence.evidence_fingerprint,
        "best_validation_mean_iou": evidence.best_validation_mean_iou,
        "split_contract": dict(evidence.split_contract),
    }
    return (
        replace(
            bundle,
            checkpoint_path=evidence.checkpoint_path,
            checkpoint_sha256=evidence.checkpoint_sha256,
            checkpoint_metadata=MappingProxyType(adapted_metadata),
        ),
        evidence,
    )


def _model_provenance(bundle: HiFiModelBundle) -> dict[str, Any]:
    checkpoint_path = _require_regular_file(
        bundle.checkpoint_path, "HiFi model checkpoint"
    )
    clip_path = _require_regular_file(bundle.clip_weight_path, "CLIP model weight")
    checkpoint_sha = sha256_file(checkpoint_path)
    clip_sha = sha256_file(clip_path)
    if checkpoint_sha != _digest(bundle.checkpoint_sha256, "bundle checkpoint_sha256"):
        raise StageInputError("loaded HiFi checkpoint changed after model construction")
    if clip_sha != _digest(bundle.clip_weight_sha256, "bundle clip_weight_sha256"):
        raise StageInputError("loaded CLIP weight changed after model construction")
    if bundle.mode != "zero_shot":
        raise StageInputError("inference bundle must be frozen and inference-only")
    if any(parameter.requires_grad for parameter in bundle.model.parameters()):
        raise StageInputError(
            "predicted mask inference model contains trainable parameters"
        )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "clip_weight_path": str(clip_path),
        "clip_weight_sha256": clip_sha,
    }


def _mask_input_payload(
    group: GroupManifest,
    *,
    condition: PredictedCondition,
    target_manifest_sha256: str,
    language_manifest_sha256: str,
    rgb_path: Path,
    rgb_sha256: str,
    model: Mapping[str, Any],
    foreground_threshold: float,
    adaptation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": PREDICTED_MASK_SCHEMA,
        "group_id": group.group_id,
        "condition": condition,
        "target_manifest_sha256": target_manifest_sha256,
        "language_manifest_sha256": language_manifest_sha256,
        "target_record_sha256": canonical_sha256(dict(group.target)),
        "language_record_sha256": canonical_sha256(dict(group.language)),
        "rgb_path": str(rgb_path),
        "rgb_sha256": rgb_sha256,
        "query": str(group.language["query"]).strip(),
        "foreground_threshold": float(foreground_threshold),
        "model": dict(model),
        "adaptation": dict(adaptation) if adaptation is not None else None,
        "ground_truth_inputs_consumed": [],
    }


def _validate_commit_fingerprint(payload: Mapping[str, Any], field: str) -> None:
    check = dict(payload)
    observed = _digest(check.pop(field, ""), field)
    if observed != canonical_sha256(check):
        raise StageInputError(f"{field} mismatch")


def _committed_mask_resume_hit(
    sidecar_path: Path,
    *,
    expected_input_fingerprint: str,
    group_id: str,
    condition: PredictedCondition,
    resume: bool,
) -> bool:
    probability_path, mask_path, _ = predicted_mask_paths(
        sidecar_path.parents[2], condition, group_id
    )
    existing = (probability_path.exists(), mask_path.exists(), sidecar_path.exists())
    if not any(existing):
        return False
    if not resume:
        raise StageInputError(
            f"predicted mask output already exists for {group_id}; use resume=True"
        )
    if not sidecar_path.exists():
        # No commit marker: the interrupted files are safe to replace atomically.
        return False
    payload = _read_json(sidecar_path, "predicted mask commit")
    if payload.get("input_fingerprint") != expected_input_fingerprint:
        raise StageInputError(f"stale predicted mask input fingerprint: {sidecar_path}")
    load_committed_predicted_mask(
        sidecar_path,
        expected_group_id=group_id,
        expected_condition=condition,
        expected_input_fingerprint=expected_input_fingerprint,
    )
    return True


def _run_predicted_mask_group(
    group: GroupManifest,
    *,
    output_root: str | os.PathLike[str],
    condition: PredictedCondition,
    target_manifest_sha256: str,
    language_manifest_sha256: str,
    bundle: HiFiModelBundle,
    model_record: Mapping[str, Any],
    adaptation_record: Mapping[str, Any] | None,
    foreground_threshold: float,
    resume: bool,
    predictor: Callable[..., GroundingPrediction],
) -> tuple[str, bool]:
    rgb_path = _require_regular_file(group.target.get("rgb_path", ""), "RGB image")
    rgb_sha = sha256_file(rgb_path)
    input_payload = _mask_input_payload(
        group,
        condition=condition,
        target_manifest_sha256=target_manifest_sha256,
        language_manifest_sha256=language_manifest_sha256,
        rgb_path=rgb_path,
        rgb_sha256=rgb_sha,
        model=model_record,
        foreground_threshold=float(foreground_threshold),
        adaptation=adaptation_record,
    )
    input_fingerprint = canonical_sha256(input_payload)
    probability_path, mask_path, sidecar_path = predicted_mask_paths(
        output_root, condition, group.group_id
    )
    if _committed_mask_resume_hit(
        sidecar_path,
        expected_input_fingerprint=input_fingerprint,
        group_id=group.group_id,
        condition=condition,
        resume=resume,
    ):
        return str(sidecar_path), True

    prediction = predictor(
        bundle,
        rgb_path,
        str(group.language["query"]),
        foreground_threshold=float(foreground_threshold),
    )
    if prediction.query != str(group.language["query"]).strip():
        raise StageInputError(
            "grounding predictor changed the committed language query"
        )
    if prediction.device != str(bundle.device):
        raise StageInputError(
            "grounding prediction reports a different execution device"
        )
    with Image.open(rgb_path) as image:
        expected_size = (image.height, image.width)
    if (prediction.native_height, prediction.native_width) != expected_size:
        raise StageInputError("grounding prediction shape differs from the source RGB")
    if sha256_file(rgb_path) != rgb_sha:
        raise StageInputError("RGB input changed during grounding inference")
    if (
        sha256_file(model_record["checkpoint_path"])
        != model_record["checkpoint_sha256"]
    ):
        raise StageInputError("HiFi checkpoint changed during grounding inference")
    if (
        sha256_file(model_record["clip_weight_path"])
        != model_record["clip_weight_sha256"]
    ):
        raise StageInputError("CLIP weight changed during grounding inference")

    stored = save_grounding_prediction(
        prediction, probability_path=probability_path, mask_path=mask_path
    )
    sidecar: dict[str, Any] = {
        "schema_version": PREDICTED_MASK_SCHEMA,
        "group_id": group.group_id,
        "scene_id": str(group.target["scene_id"]),
        "split": str(group.target.get("split", "")),
        "condition": condition,
        "mask_origin": "hifics_inference",
        "ground_truth_inputs_consumed": [],
        "input_fingerprint": input_fingerprint,
        "input_contract": input_payload,
        "probability_file": probability_path.name,
        "probability_sha256": stored["probability_sha256"],
        "mask_file": mask_path.name,
        "mask_sha256": stored["mask_sha256"],
        "native_height": int(prediction.native_height),
        "native_width": int(prediction.native_width),
        "foreground_threshold": float(prediction.foreground_threshold),
        "foreground_probability": "sigmoid(-background_logit)",
        "model": dict(model_record),
        "adaptation": dict(adaptation_record)
        if adaptation_record is not None
        else None,
    }
    sidecar["commit_fingerprint"] = canonical_sha256(sidecar)
    atomic_json(sidecar_path, sidecar)
    return str(sidecar_path), False


def run_predicted_mask_stage(
    target_manifest_path: str | os.PathLike[str],
    language_manifest_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    condition: PredictedCondition = "hifics_zero_shot_mask",
    device: str = "cpu",
    foreground_threshold: float = FOREGROUND_THRESHOLD,
    resume: bool = False,
    checkpoint_path: str | os.PathLike[str] = DEFAULT_HIFI_CHECKPOINT,
    clip_weight_path: str | os.PathLike[str] = DEFAULT_CLIP_WEIGHT,
    adaptation_evidence_path: str | os.PathLike[str] | None = None,
    adapted_checkpoint_path: str | os.PathLike[str] | None = None,
    model_loader: Callable[..., HiFiModelBundle] = load_hifi_model,
    adapted_model_loader: Callable[
        ..., tuple[HiFiModelBundle, ValidatedAdaptationEvidence]
    ] = load_validated_adapted_hifi,
    predictor: Callable[..., GroundingPrediction] = predict_grounding,
) -> StageSummary:
    """Infer and commit exactly one HiFi probability mask per target group."""

    selected = _predicted_condition(condition)
    if not 0.0 < float(foreground_threshold) < 1.0:
        raise StageInputError("foreground_threshold must be strictly inside (0, 1)")
    target_manifest = _require_regular_file(target_manifest_path, "target manifest")
    language_manifest = _require_regular_file(
        language_manifest_path, "language manifest"
    )
    groups = load_target_language_jsonl(target_manifest, language_manifest)
    target_manifest_sha = sha256_file(target_manifest)
    language_manifest_sha = sha256_file(language_manifest)

    root = Path(output_root).expanduser().resolve()
    failures: list[GroupFailure] = []
    adaptation_record: dict[str, Any] | None = None
    try:
        if selected == "hifics_zero_shot_mask":
            if (
                adaptation_evidence_path is not None
                or adapted_checkpoint_path is not None
            ):
                raise StageInputError(
                    "zero-shot masks must not receive adaptation artifacts"
                )
            bundle = model_loader(
                device=device,
                mode="zero_shot",
                checkpoint_path=checkpoint_path,
                clip_weight_path=clip_weight_path,
            )
        else:
            if adaptation_evidence_path is None:
                raise StageInputError(
                    "adapted masks require a versioned adaptation evidence manifest"
                )
            bundle, evidence = adapted_model_loader(
                adaptation_evidence_path,
                adapted_checkpoint_path=adapted_checkpoint_path,
                device=device,
                base_checkpoint_path=checkpoint_path,
                clip_weight_path=clip_weight_path,
            )
            adaptation_record = {
                "evidence_path": str(evidence.evidence_path),
                "evidence_sha256": evidence.evidence_sha256,
                "evidence_fingerprint": evidence.evidence_fingerprint,
                "train_manifest_sha256": evidence.train_manifest_sha256,
                "validation_manifest_sha256": evidence.validation_manifest_sha256,
                "best_validation_mean_iou": evidence.best_validation_mean_iou,
                "test_rows_consumed": 0,
                "selection_metric": "validation_mean_iou",
                "selection_split": "val",
            }
        model_record = _model_provenance(bundle)
    except Exception as error:
        failures.extend(
            _record_group_failure(
                root,
                stage="predicted_masks",
                group_id=group.group_id,
                error=error,
            )
            for group in groups
        )
        raise StageBatchError("predicted_masks", failures) from error

    output_paths: list[str] = []
    resumed = 0
    for group in groups:
        try:
            output_path, was_resumed = _run_predicted_mask_group(
                group,
                output_root=root,
                condition=selected,
                target_manifest_sha256=target_manifest_sha,
                language_manifest_sha256=language_manifest_sha,
                bundle=bundle,
                model_record=model_record,
                adaptation_record=adaptation_record,
                foreground_threshold=float(foreground_threshold),
                resume=resume,
                predictor=predictor,
            )
            output_paths.append(output_path)
            resumed += int(was_resumed)
        except Exception as error:
            failures.append(
                _record_group_failure(
                    root,
                    stage="predicted_masks",
                    group_id=group.group_id,
                    error=error,
                )
            )
    if (
        sha256_file(target_manifest) != target_manifest_sha
        or sha256_file(language_manifest) != language_manifest_sha
    ):
        error = StageInputError(
            "target/language manifest changed during predicted mask stage"
        )
        failures.extend(
            _record_group_failure(
                root,
                stage="predicted_masks",
                group_id=group.group_id,
                error=error,
            )
            for group in groups
            if group.group_id not in {failure.group_id for failure in failures}
        )
    if failures:
        raise StageBatchError("predicted_masks", failures)
    return StageSummary(
        "predicted_masks",
        len(groups),
        len(groups) - resumed,
        resumed,
        tuple(output_paths),
    )


def load_committed_predicted_mask(
    sidecar_path: str | os.PathLike[str],
    *,
    expected_group_id: str,
    expected_condition: PredictedCondition,
    expected_input_fingerprint: str | None = None,
) -> CommittedPredictedMask:
    """Load only a committed HiFi cache; oracle schemas are unconditionally rejected."""

    selected = _predicted_condition(expected_condition)
    sidecar_source = _require_regular_file(sidecar_path, "predicted mask commit")
    sidecar = _read_json(sidecar_source, "predicted mask commit")
    if sidecar.get("schema_version") != PREDICTED_MASK_SCHEMA:
        raise StageInputError(
            "predicted tracks reject non-HiFi mask caches (including oracle_gt_mask)"
        )
    _validate_commit_fingerprint(sidecar, "commit_fingerprint")
    if sidecar.get("group_id") != expected_group_id:
        raise StageInputError("predicted mask group_id mismatch")
    if sidecar.get("condition") != selected:
        raise StageInputError("predicted mask condition mismatch")
    if sidecar.get("mask_origin") != "hifics_inference":
        raise StageInputError("predicted mask cache was not emitted by HiFi inference")
    if sidecar.get("ground_truth_inputs_consumed") != []:
        raise StageInputError("predicted mask cache reports consuming ground truth")
    input_contract = sidecar.get("input_contract")
    if not isinstance(input_contract, Mapping):
        raise StageInputError("predicted mask cache has no input contract")
    committed_input_fingerprint = _digest(
        sidecar.get("input_fingerprint"), "input_fingerprint"
    )
    if canonical_sha256(dict(input_contract)) != committed_input_fingerprint:
        raise StageInputError("predicted mask input contract fingerprint mismatch")
    if (
        input_contract.get("schema") != PREDICTED_MASK_SCHEMA
        or input_contract.get("group_id") != expected_group_id
        or input_contract.get("condition") != selected
        or input_contract.get("ground_truth_inputs_consumed") != []
    ):
        raise StageInputError("predicted mask input contract violates track isolation")
    if (
        expected_input_fingerprint is not None
        and sidecar.get("input_fingerprint") != expected_input_fingerprint
    ):
        raise StageInputError("predicted mask input fingerprint mismatch")

    probability_name = str(sidecar.get("probability_file", ""))
    mask_name = str(sidecar.get("mask_file", ""))
    if (
        Path(probability_name).name != probability_name
        or Path(mask_name).name != mask_name
    ):
        raise StageInputError(
            "predicted mask commit contains a non-local artifact path"
        )
    probability_path = _require_regular_file(
        sidecar_source.parent / probability_name, "predicted probability archive"
    )
    mask_path = _require_regular_file(
        sidecar_source.parent / mask_name, "predicted binary mask"
    )
    if sha256_file(probability_path) != _digest(
        sidecar.get("probability_sha256"), "probability_sha256"
    ):
        raise StageInputError("predicted probability archive SHA-256 mismatch")
    if sha256_file(mask_path) != _digest(sidecar.get("mask_sha256"), "mask_sha256"):
        raise StageInputError("predicted binary mask SHA-256 mismatch")

    try:
        with np.load(probability_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "probability_352",
                "native_probability",
                "foreground_threshold",
            }:
                raise StageInputError("predicted probability archive schema mismatch")
            model_probability = np.asarray(archive["probability_352"], dtype=np.float32)
            native_probability = np.asarray(
                archive["native_probability"], dtype=np.float32
            )
            threshold = float(np.asarray(archive["foreground_threshold"]).item())
    except (OSError, ValueError) as error:
        if isinstance(error, StageInputError):
            raise
        raise StageInputError(
            f"invalid predicted probability archive: {error}"
        ) from error
    declared_shape = (
        int(sidecar.get("native_height", -1)),
        int(sidecar.get("native_width", -1)),
    )
    if model_probability.shape != (IMAGE_RESOLUTION, IMAGE_RESOLUTION):
        raise StageInputError("predicted model-resolution probability shape mismatch")
    if native_probability.shape != declared_shape:
        raise StageInputError("predicted native probability shape mismatch")
    if not np.isfinite(native_probability).all() or np.any(
        (native_probability < 0) | (native_probability > 1)
    ):
        raise StageInputError("predicted probability contains invalid values")
    if not np.isfinite(model_probability).all() or np.any(
        (model_probability < 0) | (model_probability > 1)
    ):
        raise StageInputError("model-resolution probability contains invalid values")
    if threshold != float(sidecar.get("foreground_threshold", float("nan"))):
        raise StageInputError("predicted mask threshold mismatch")
    with Image.open(mask_path) as image:
        pixels = np.asarray(image.convert("L"))
    if pixels.shape != declared_shape or not np.isin(pixels, [0, 255]).all():
        raise StageInputError("predicted binary PNG must contain only 0 and 255")
    binary = pixels != 0
    if not np.array_equal(binary, native_probability >= threshold):
        raise StageInputError("predicted PNG disagrees with probability and threshold")

    model = sidecar.get("model")
    if not isinstance(model, Mapping):
        raise StageInputError("predicted mask commit has no model provenance")
    if dict(input_contract.get("model") or {}) != dict(model):
        raise StageInputError(
            "predicted mask model provenance differs from its input contract"
        )
    rgb_source = _require_regular_file(
        input_contract.get("rgb_path", ""), "mask RGB source"
    )
    if sha256_file(rgb_source) != _digest(
        input_contract.get("rgb_sha256"), "mask rgb_sha256"
    ):
        raise StageInputError("predicted mask RGB source is stale")
    for prefix in ("checkpoint", "clip_weight"):
        path = _require_regular_file(
            model.get(f"{prefix}_path", ""), f"{prefix} source"
        )
        expected = _digest(model.get(f"{prefix}_sha256"), f"{prefix}_sha256")
        if sha256_file(path) != expected:
            raise StageInputError(f"predicted mask {prefix} source is stale")
    if selected == "hifics_zero_shot_mask":
        if sidecar.get("adaptation") is not None:
            raise StageInputError(
                "zero-shot predicted mask unexpectedly has adaptation evidence"
            )
        if model.get("checkpoint_sha256") != EXPECTED_HIFI_CHECKPOINT_SHA256:
            raise StageInputError(
                "zero-shot mask does not use the retained HiFi checkpoint"
            )
        if model.get("clip_weight_sha256") != EXPECTED_CLIP_WEIGHT_SHA256:
            raise StageInputError(
                "zero-shot mask does not use the retained CLIP weight"
            )
    else:
        adaptation = sidecar.get("adaptation")
        if not isinstance(adaptation, Mapping):
            raise StageInputError("adapted mask has no adaptation evidence")
        if dict(input_contract.get("adaptation") or {}) != dict(adaptation):
            raise StageInputError(
                "adaptation provenance differs from the predicted-mask input contract"
            )
        validated = validate_adaptation_evidence(
            adaptation.get("evidence_path", ""),
            adapted_checkpoint_path=model.get("checkpoint_path", ""),
        )
        if validated.evidence_sha256 != adaptation.get("evidence_sha256"):
            raise StageInputError("adaptation evidence SHA-256 is stale")
        if validated.evidence_fingerprint != adaptation.get("evidence_fingerprint"):
            raise StageInputError("adaptation evidence fingerprint mismatch")

    return CommittedPredictedMask(
        probability=native_probability,
        binary_mask=binary,
        foreground_threshold=threshold,
        condition=selected,
        group_id=expected_group_id,
        probability_path=probability_path,
        mask_path=mask_path,
        sidecar_path=sidecar_source,
        sidecar=MappingProxyType(sidecar),
    )


def load_committed_oracle_mask(
    sidecar_path: str | os.PathLike[str],
    *,
    expected_group_id: str,
    expected_instance_label_path: str | os.PathLike[str],
    expected_target_instance_label: int,
) -> CommittedOracleMask:
    """Load the official oracle artifact only for ``oracle_gt_mask``.

    Unlike predicted tracks, this explicitly named counterfactual is allowed
    to read the official instance-label pixels.  The selected binary mask is
    recomputed from those pixels and must match the committed archive exactly.
    No object ID, pose, mesh, collision, or friction field is exposed to the
    feature frame.
    """

    source = _require_regular_file(sidecar_path, "oracle-mask commit")
    sidecar = _read_json(source, "oracle-mask commit")
    if sidecar.get("schema_version") != MASK_SCHEMA:
        raise StageInputError(
            "oracle feature track requires the official oracle-mask schema"
        )
    if sidecar.get("group_id") != expected_group_id:
        raise StageInputError("oracle-mask group_id mismatch")
    instance_label = int(expected_target_instance_label)
    if (
        instance_label <= 0
        or int(sidecar.get("target_instance_label", -1)) != instance_label
    ):
        raise StageInputError("oracle-mask target instance label mismatch")
    label_path = _require_regular_file(
        expected_instance_label_path, "official instance-label image"
    )
    recorded_label_path = _require_regular_file(
        sidecar.get("source_instance_label_path", ""),
        "oracle-mask recorded instance-label image",
    )
    if recorded_label_path != label_path:
        raise StageInputError("oracle-mask commit names another instance-label image")
    label_sha = sha256_file(label_path)
    expected_input_fingerprint = canonical_sha256(
        {
            "schema": MASK_SCHEMA,
            "group_id": expected_group_id,
            "instance_label": instance_label,
            "instance_label_sha256": label_sha,
        }
    )
    if sidecar.get("input_fingerprint") != expected_input_fingerprint:
        raise StageInputError("oracle-mask input fingerprint is stale")
    archive_path = _require_regular_file(
        source.with_suffix(".npz"), "oracle-mask archive"
    )
    if sha256_file(archive_path) != _digest(
        sidecar.get("output_sha256"), "oracle output_sha256"
    ):
        raise StageInputError("oracle-mask archive SHA-256 mismatch")
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "mask",
                "target_instance_label",
                "input_fingerprint",
            }:
                raise StageInputError("oracle-mask archive schema mismatch")
            mask = np.asarray(archive["mask"])
            embedded_label = int(np.asarray(archive["target_instance_label"]).item())
            embedded_fingerprint = str(np.asarray(archive["input_fingerprint"]).item())
    except (OSError, ValueError) as error:
        if isinstance(error, StageInputError):
            raise
        raise StageInputError(f"invalid oracle-mask archive: {error}") from error
    if (
        embedded_label != instance_label
        or embedded_fingerprint != expected_input_fingerprint
    ):
        raise StageInputError("oracle-mask archive provenance mismatch")
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all() or not bool(mask.any()):
        raise StageInputError(
            "oracle-mask archive must contain one non-empty binary mask"
        )
    with Image.open(label_path) as image:
        label_pixels = np.asarray(image)
    if label_pixels.ndim != 2 or not np.issubdtype(label_pixels.dtype, np.integer):
        raise StageInputError("official instance-label image must be integer HxW")
    recomputed = label_pixels == instance_label
    binary = mask.astype(bool)
    if not np.array_equal(binary, recomputed):
        raise StageInputError(
            "oracle-mask pixels differ from the official instance label"
        )
    if sidecar.get("shape") != list(binary.shape) or int(
        sidecar.get("mask_pixel_count", -1)
    ) != int(binary.sum()):
        raise StageInputError("oracle-mask sidecar shape/count mismatch")
    enriched = {
        **sidecar,
        "condition": "oracle_gt_mask",
        "mask_origin": "official_instance_label_equality",
        "ground_truth_inputs_consumed": [
            "official_instance_label_pixels_for_explicit_oracle_gt_mask_only"
        ],
        "source_instance_label_sha256": label_sha,
        "commit_fingerprint": canonical_sha256(sidecar),
    }
    return CommittedOracleMask(
        probability=binary.astype(np.float32),
        binary_mask=binary,
        foreground_threshold=0.5,
        condition="oracle_gt_mask",
        group_id=expected_group_id,
        probability_path=archive_path,
        mask_path=archive_path,
        sidecar_path=source,
        sidecar=MappingProxyType(enriched),
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_sha256(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _depth_observation(
    group: GroupManifest,
    mask: CommittedPredictedMask | CommittedOracleMask,
    *,
    scene_cloud_kind: SceneCloudKind,
    voxel_size_m: float,
) -> tuple[RuntimeObservation, dict[str, str], dict[str, Any]]:
    target = group.target
    paths = {
        "depth": _require_regular_file(target["depth_path"], "depth image"),
        "meta": _require_regular_file(target["meta_path"], "frame metadata"),
        "intrinsics": _require_regular_file(
            target["intrinsics_path"], "camera intrinsics"
        ),
        "camera_pose": _require_regular_file(
            target["camera_pose_path"], "camera poses"
        ),
        "table_transform": _require_regular_file(
            target["table_transform_path"], "camera/table transform"
        ),
    }
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    with Image.open(paths["depth"]) as image:
        depth_raw = np.asarray(image)
    if depth_raw.ndim != 2 or not np.issubdtype(depth_raw.dtype, np.integer):
        raise StageInputError("GraspNet depth must be a two-dimensional integer image")
    if depth_raw.shape != mask.probability.shape:
        raise StageInputError("predicted mask and actual depth dimensions differ")
    metadata = loadmat(paths["meta"])
    factor_values = np.asarray(
        metadata.get("factor_depth", []), dtype=np.float64
    ).reshape(-1)
    if (
        factor_values.size != 1
        or not np.isfinite(factor_values[0])
        or factor_values[0] <= 0
    ):
        raise StageInputError("frame metadata factor_depth must be one positive scalar")
    factor_depth = float(factor_values[0])
    depth_m = depth_raw.astype(np.float64) / factor_depth
    if not np.isfinite(depth_m).all() or np.any(depth_m < 0) or not np.any(depth_m > 0):
        raise StageInputError(
            "depth conversion produced no finite positive scene depth"
        )

    matrix = np.asarray(
        np.load(paths["intrinsics"], allow_pickle=False), dtype=np.float64
    )
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise StageInputError("camera intrinsics must be a finite 3x3 matrix")
    if "intrinsic_matrix" in metadata and not np.allclose(
        matrix,
        np.asarray(metadata["intrinsic_matrix"], dtype=np.float64),
        atol=1e-6,
        rtol=0,
    ):
        raise StageInputError("camK.npy disagrees with frame metadata intrinsics")
    intrinsics = CameraIntrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        width=int(depth_raw.shape[1]),
        height=int(depth_raw.shape[0]),
    )
    if not np.allclose(matrix, intrinsics.matrix, atol=1e-8, rtol=0):
        raise StageInputError("camera intrinsic matrix violates the pinhole contract")

    rows, columns = np.nonzero(depth_m > 0)
    z = depth_m[rows, columns]
    full_scene = np.column_stack(
        (
            (columns - intrinsics.cx) * z / intrinsics.fx,
            (rows - intrinsics.cy) * z / intrinsics.fy,
            z,
        )
    ).astype(np.float64, copy=False)
    if scene_cloud_kind == "full_depth_backprojection":
        scene = full_scene
        downsample_record: dict[str, Any] = {
            "algorithm": "none_full_valid_depth_row_major",
            "voxel_size_m": None,
        }
    elif scene_cloud_kind == "deterministic_voxel_downsample_of_full_depth":
        if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise StageInputError("voxel_size_m must be finite and positive")
        voxel_keys = np.floor(full_scene / float(voxel_size_m)).astype(np.int64)
        _, first_indices = np.unique(voxel_keys, axis=0, return_index=True)
        first_indices.sort()
        scene = full_scene[first_indices]
        downsample_record = {
            "algorithm": "first_row_major_depth_point_per_floor_quantized_xyz_voxel",
            "voxel_size_m": float(voxel_size_m),
        }
    else:
        raise StageInputError(f"unsupported scene_cloud_kind: {scene_cloud_kind!r}")
    if not len(scene):
        raise StageInputError("scene cloud is empty after deterministic construction")

    camera_poses = np.asarray(
        np.load(paths["camera_pose"], allow_pickle=False), dtype=np.float64
    )
    frame_id = int(target["frame_id"])
    if (
        camera_poses.ndim != 3
        or camera_poses.shape[1:] != (4, 4)
        or not 0 <= frame_id < len(camera_poses)
    ):
        raise StageInputError("camera_poses.npy does not contain the selected frame")
    table = as_transform(np.load(paths["table_transform"], allow_pickle=False))
    T_table_camera = compose_transforms(table, camera_poses[frame_id])
    table_normal_camera = T_table_camera[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    gravity_camera = -table_normal_camera
    scene_content_sha = _array_sha256(scene)
    provenance = {
        "factor_depth": factor_depth,
        "full_valid_depth_point_count": int(len(full_scene)),
        "scene_point_count": int(len(scene)),
        "scene_cloud_kind": scene_cloud_kind,
        "scene_cloud_content_sha256": scene_content_sha,
        "depth_source_sha256": source_hashes["depth"],
        "intrinsics_source_sha256": source_hashes["intrinsics"],
        "downsample": downsample_record,
        "table_normal_derivation": "R_table_camera.T @ table_positive_z",
        "gravity_derivation": "negative_table_normal_camera",
    }
    scene_source_sha = canonical_sha256(provenance)
    observation = RuntimeObservation(
        intrinsics=intrinsics,
        mask_probability=mask.probability,
        depth_m=depth_m,
        scene_points_camera_m=scene,
        table_normal_camera=table_normal_camera,
        gravity_camera=gravity_camera,
        grounding_condition=mask.condition,
        mask_source_sha256=str(
            mask.sidecar.get(
                "probability_sha256", mask.sidecar.get("output_sha256", "")
            )
        ),
        depth_source_sha256=source_hashes["depth"],
        scene_points_source_sha256=scene_source_sha,
        scene_points_source_kind=scene_cloud_kind,
    ).validated()
    return observation, source_hashes, provenance


def _load_candidate_bundle(
    path: str | os.PathLike[str], group_id: str
) -> tuple[tuple[Candidate6D, ...], dict[str, Any], Path, dict[str, str]]:
    source = _require_regular_file(path, "frozen VGN candidate bundle")
    payload = _read_json(source, "frozen VGN candidate bundle")
    if payload.get("schema_version") != VGN_BUNDLE_SCHEMA:
        raise StageInputError("unsupported frozen candidate bundle schema")
    if payload.get("group_id") != group_id:
        raise StageInputError("frozen candidate bundle group_id mismatch")
    _validate_commit_fingerprint(payload, "bundle_fingerprint")
    records = payload.get("candidate_records")
    raw = payload.get("raw_vgn_candidates")
    raw_count = payload.get("candidate_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        raise StageInputError("frozen candidate bundle count must be an integer")
    count = raw_count
    if not isinstance(records, list) or not isinstance(raw, list):
        raise StageInputError("frozen candidate bundle lacks candidate records")
    if count < 0 or len(records) != count or len(raw) != count:
        raise StageInputError("frozen candidate bundle has an invalid candidate count")
    try:
        candidates = validate_candidate_pool(
            Candidate6D.from_dict(record) for record in records
        )
    except (TypeError, ValueError) as error:
        raise StageInputError(f"invalid frozen candidate record: {error}") from error
    if any(candidate.group_id != group_id for candidate in candidates):
        raise StageInputError("frozen candidate bundle contains another group")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if candidate_pool_fingerprint(candidates) != payload.get(
        "candidate_pool_fingerprint"
    ):
        raise StageInputError("frozen candidate pool fingerprint mismatch")
    rows = np.asarray(payload.get("graspnet_rows"), dtype=np.float64)
    if count == 0 and rows.size == 0:
        rows = rows.reshape(0, 17)
    if rows.shape != (count, 17) or not np.isfinite(rows).all():
        raise StageInputError("frozen candidate evaluator rows must be finite N x 17")
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
            "frozen evaluator rows disagree with candidate score/geometry/membership"
        )
    if candidate_ids != [str(record.get("candidate_id", "")) for record in records]:
        raise StageInputError("frozen candidate ordering changed during decoding")
    geometry_record = payload.get("geometry_contract")
    fixture_bundle = bool(
        isinstance(geometry_record, Mapping)
        and geometry_record.get("fixture_only") is True
    )
    frozen_vgn = ()
    if not fixture_bundle:
        if canonical_sha256(raw) != payload.get("raw_vgn_pool_fingerprint"):
            raise StageInputError("frozen raw VGN pool fingerprint mismatch")
        if [str(record.get("candidate_id", "")) for record in raw] != candidate_ids or [
            int(record.get("native_rank", -1)) for record in raw
        ] != [candidate.native_rank for candidate in candidates]:
            raise StageInputError(
                "raw VGN and converted candidate membership/order disagree"
            )
        if "generation_status" not in payload:
            raise StageInputError("formal candidate bundle lacks generation_status")
        try:
            frozen_vgn = tuple(
                vgn_candidate_from_record(record, expected_group_id=group_id)
                for record in raw
            )
        except Exception as error:
            raise StageInputError(
                f"invalid frozen raw VGN candidate: {error}"
            ) from error
    a7_fields = {
        "pre_nms_snapshot_status",
        "pre_nms_candidate_count",
        "pre_nms_vgn_candidates",
        "pre_nms_pool_fingerprint",
        "a7_top_k_membership",
        "a7_top_k_membership_fingerprint",
    }
    present_a7_fields = a7_fields.intersection(payload)
    if present_a7_fields and present_a7_fields != a7_fields:
        raise StageInputError("frozen candidate bundle has a partial A7 contract")
    if not fixture_bundle and present_a7_fields != a7_fields:
        raise StageInputError(
            "formal frozen candidate bundle lacks the mandatory A7 pre-NMS contract"
        )
    if present_a7_fields:
        extraction_record = payload.get("extraction_config")
        if not isinstance(extraction_record, Mapping):
            raise StageInputError(
                "frozen candidate A7 contract lacks extraction config"
            )
        try:
            extraction = ExtractionConfig(**dict(extraction_record))
            extraction.validate()
            if fixture_bundle:
                frozen_vgn = tuple(
                    vgn_candidate_from_record(record, expected_group_id=group_id)
                    for record in raw
                )
        except Exception as error:
            raise StageInputError(
                f"invalid frozen candidate A7 source: {error}"
            ) from error
        _decode_a7_pre_nms_contract(
            payload,
            group_id=group_id,
            frozen=frozen_vgn,
            config=extraction,
            allow_test_incomplete=bool(
                fixture_bundle
                or (
                    isinstance(geometry_record, Mapping)
                    and geometry_record.get("evidence_policy") == "test"
                )
            ),
        )
    generation_status = payload.get("generation_status", "completed_vgn_inference")
    upstream_source_hashes: dict[str, str]
    if generation_status == "skipped_grounding_failure":
        if (
            count != 0
            or payload.get("inference_calls_for_group") != 0
            or payload.get("tsdf_path") is not None
            or payload.get("tsdf_sha256") is not None
        ):
            raise StageInputError(
                "grounding-failure candidate bundle contains fabricated work"
            )
        terminal_path = _require_regular_file(
            payload.get("grounding_terminal_path", ""),
            "candidate grounding terminal source",
        )
        terminal_sha = sha256_file(terminal_path)
        if terminal_sha != _digest(
            payload.get("grounding_terminal_sha256"),
            "candidate grounding_terminal_sha256",
        ):
            raise StageInputError("candidate grounding terminal SHA-256 mismatch")
        terminal = load_grounding_terminal(
            terminal_path,
            group_id=group_id,
            grounding_condition=str(payload.get("grounding_condition", "")),
        )
        if (
            terminal.get("reason") != payload.get("grounding_failure_reason")
            or terminal.get("grounding_mask_input_fingerprint")
            != payload.get("grounding_mask_input_fingerprint")
            or terminal.get("grounding_mask_commit_sha256")
            != payload.get("grounding_mask_commit_sha256")
        ):
            raise StageInputError("candidate/grounding-terminal lineage mismatch")
        upstream_source_hashes = {str(terminal_path): terminal_sha}
    elif generation_status == "completed_vgn_inference":
        if payload.get("inference_calls_for_group", 1) != 1:
            raise StageInputError("candidate VGN inference count is invalid")
        tsdf_path = _require_regular_file(
            payload.get("tsdf_path", ""), "candidate TSDF source"
        )
        tsdf_sha = sha256_file(tsdf_path)
        if tsdf_sha != _digest(payload.get("tsdf_sha256"), "candidate tsdf_sha256"):
            raise StageInputError("candidate TSDF source SHA-256 mismatch")
        tsdf_sidecar_path = _require_regular_file(
            tsdf_path.with_suffix(".json"), "candidate TSDF commit"
        )
        tsdf_sidecar = _read_json(tsdf_sidecar_path, "candidate TSDF commit")
        if (
            tsdf_sidecar.get("schema_version") != TSDF_SCHEMA
            or tsdf_sidecar.get("group_id") != group_id
            or tsdf_sidecar.get("output_sha256") != tsdf_sha
        ):
            raise StageInputError("candidate TSDF commit schema/group/hash mismatch")
        for field in (
            "grounding_condition",
            "grounding_mask_input_fingerprint",
            "grounding_mask_commit_sha256",
        ):
            if tsdf_sidecar.get(field) != payload.get(field):
                raise StageInputError(
                    f"candidate/TSDF grounding lineage mismatch: {field}"
                )
        upstream_source_hashes = {
            str(tsdf_path): tsdf_sha,
            str(tsdf_sidecar_path): sha256_file(tsdf_sidecar_path),
        }
        _validate_raw_to_evaluator_geometry(
            payload,
            group_id=group_id,
            raw_candidates=frozen_vgn,
            candidates=candidates,
            evaluator_rows=rows,
            allow_explicit_fixture_only=fixture_bundle,
        )
    else:
        raise StageInputError("candidate generation_status is unsupported")
    checkpoint_path = _require_regular_file(
        payload.get("checkpoint_path", ""), "frozen VGN checkpoint source"
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != _digest(
        payload.get("checkpoint_sha256"), "candidate checkpoint_sha256"
    ):
        raise StageInputError("frozen VGN checkpoint source SHA-256 mismatch")
    return (
        candidates,
        payload,
        source,
        {
            **upstream_source_hashes,
            str(checkpoint_path): checkpoint_sha,
        },
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            suffix=".parquet",
            prefix=f".{destination.stem}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=False)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _expected_feature_columns() -> tuple[str, ...]:
    return (
        "group_id",
        "candidate_id",
        "scene_id",
        "split",
        "condition",
        *(spec.name for spec in default_feature_schema()),
    )


def load_committed_formal_feature_table(
    sidecar_path: str | os.PathLike[str],
    *,
    expected_group_id: str | None = None,
    expected_condition: GroundingCondition | None = None,
    expected_input_fingerprint: str | None = None,
    expected_candidate_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Revalidate one committed feature table and its exact candidate universe."""

    source = _require_regular_file(sidecar_path, "formal feature commit")
    payload = _read_json(source, "formal feature commit")
    if payload.get("schema_version") != FORMAL_FEATURE_SCHEMA:
        raise StageInputError("unsupported formal feature commit schema")
    _validate_commit_fingerprint(payload, "commit_fingerprint")
    if expected_group_id is not None and payload.get("group_id") != expected_group_id:
        raise StageInputError("formal feature group_id mismatch")
    if expected_condition is not None and payload.get(
        "condition"
    ) != _grounding_condition(expected_condition):
        raise StageInputError("formal feature condition mismatch")
    if (
        expected_input_fingerprint is not None
        and payload.get("input_fingerprint") != expected_input_fingerprint
    ):
        raise StageInputError("formal feature input fingerprint mismatch")
    name = str(payload.get("feature_file", ""))
    if Path(name).name != name:
        raise StageInputError("formal feature commit contains a non-local table path")
    table_path = _require_regular_file(source.parent / name, "formal feature table")
    if sha256_file(table_path) != _digest(
        payload.get("feature_sha256"), "feature_sha256"
    ):
        raise StageInputError("formal feature table SHA-256 mismatch")
    try:
        frame = pd.read_parquet(table_path)
    except Exception as error:
        raise StageInputError(f"cannot read formal feature table: {error}") from error
    if tuple(map(str, frame.columns)) != _expected_feature_columns():
        raise StageInputError("formal feature table column schema/order mismatch")
    if len(frame) != int(payload.get("candidate_count", -1)):
        raise StageInputError("formal feature table candidate count mismatch")
    candidate_ids = frame["candidate_id"].astype(str).tolist()
    recorded_ids = [str(value) for value in payload.get("candidate_ids", [])]
    if candidate_ids != recorded_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise StageInputError("formal feature candidate membership/order mismatch")
    if expected_candidate_ids is not None and candidate_ids != list(
        expected_candidate_ids
    ):
        raise StageInputError(
            "formal feature table differs from the frozen candidate universe"
        )
    for column, expected in (
        ("group_id", payload.get("group_id")),
        ("scene_id", payload.get("scene_id")),
        ("split", payload.get("split")),
        ("condition", payload.get("condition")),
    ):
        if frame[column].astype(str).tolist() != [str(expected)] * len(frame):
            raise StageInputError(
                f"formal feature identifier column mismatch: {column}"
            )
    if payload.get("feature_schema_sha256") != feature_schema_sha256(
        default_feature_schema()
    ):
        raise StageInputError("formal feature runtime schema fingerprint mismatch")
    return frame


def _source_hashes_unchanged(source_hashes: Mapping[str, str]) -> None:
    for raw_path, expected in source_hashes.items():
        path = _require_regular_file(raw_path, "formal feature source")
        if sha256_file(path) != expected:
            raise StageInputError(
                f"formal feature source changed during extraction: {path}"
            )


def _run_formal_feature_stage_impl(
    target_manifest_path: str | os.PathLike[str],
    language_manifest_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    condition: GroundingCondition,
    mask_output_root: str | os.PathLike[str] | None = None,
    candidate_output_root: str | os.PathLike[str] | None = None,
    scene_cloud_kind: SceneCloudKind = "deterministic_voxel_downsample_of_full_depth",
    voxel_size_m: float = 0.005,
    resume: bool = False,
    extractor: Callable[
        [Sequence[Candidate6D], RuntimeObservation], pd.DataFrame
    ] = extract_candidate_features,
    only_group_id: str | None = None,
) -> StageSummary:
    """Build hash-bound runtime features without opening GT labels or evaluator outputs."""

    selected = _grounding_condition(condition)
    target_manifest = _require_regular_file(target_manifest_path, "target manifest")
    language_manifest = _require_regular_file(
        language_manifest_path, "language manifest"
    )
    groups = load_target_language_jsonl(target_manifest, language_manifest)
    if only_group_id is not None:
        groups = tuple(group for group in groups if group.group_id == only_group_id)
        if len(groups) != 1:
            raise StageInputError(
                f"formal feature group is absent or duplicated: {only_group_id!r}"
            )
    target_manifest_sha = sha256_file(target_manifest)
    language_manifest_sha = sha256_file(language_manifest)
    mask_root = output_root if mask_output_root is None else mask_output_root
    candidate_root = (
        output_root if candidate_output_root is None else candidate_output_root
    )

    outputs: list[str] = []
    resumed = 0
    for group in groups:
        if selected == "oracle_gt_mask":
            _, mask_sidecar_path = oracle_mask_paths(mask_root, group.group_id)
            mask = load_committed_oracle_mask(
                mask_sidecar_path,
                expected_group_id=group.group_id,
                expected_instance_label_path=group.target["instance_label_path"],
                expected_target_instance_label=int(
                    group.target["target_instance_label"]
                ),
            )
            expected_mask_fingerprint = str(mask.sidecar["input_fingerprint"])
            ground_truth_inputs = [
                "official_instance_label_pixels_for_explicit_oracle_gt_mask_only"
            ]
            condition_candidate_root = (
                Path(candidate_root).expanduser().resolve()
                / "vgn_candidates"
                / selected
            )
            candidate_path = (
                condition_candidate_root / f"{group_artifact_slug(group.group_id)}.json"
            )
        else:
            _, _, mask_sidecar_path = predicted_mask_paths(
                mask_root, selected, group.group_id
            )
            raw_mask_sidecar = _read_json(mask_sidecar_path, "predicted mask commit")
            mask_contract = raw_mask_sidecar.get("input_contract")
            if not isinstance(mask_contract, Mapping):
                raise StageInputError("predicted mask commit lacks an input contract")
            model_record = raw_mask_sidecar.get("model")
            if not isinstance(model_record, Mapping):
                raise StageInputError("predicted mask commit lacks model provenance")
            rgb_path = _require_regular_file(
                group.target.get("rgb_path", ""), "RGB image"
            )
            expected_mask_payload = _mask_input_payload(
                group,
                condition=selected,
                target_manifest_sha256=target_manifest_sha,
                language_manifest_sha256=language_manifest_sha,
                rgb_path=rgb_path,
                rgb_sha256=sha256_file(rgb_path),
                model=model_record,
                foreground_threshold=float(
                    raw_mask_sidecar.get("foreground_threshold", float("nan"))
                ),
                adaptation=raw_mask_sidecar.get("adaptation"),
            )
            expected_mask_fingerprint = canonical_sha256(expected_mask_payload)
            if dict(mask_contract) != expected_mask_payload:
                raise StageInputError(
                    "predicted mask input contract is stale or was rewritten"
                )
            mask = load_committed_predicted_mask(
                mask_sidecar_path,
                expected_group_id=group.group_id,
                expected_condition=selected,
                expected_input_fingerprint=expected_mask_fingerprint,
            )
            ground_truth_inputs = []
            # Predicted masks must generate their own target-centred TSDF and
            # frozen VGN pool.  A legacy unscoped bundle is oracle-derived and
            # is never accepted for a predicted condition.
            candidate_path = (
                Path(candidate_root).expanduser().resolve()
                / "vgn_candidates"
                / selected
                / f"{group_artifact_slug(group.group_id)}.json"
            )
        candidates, candidate_payload, candidate_source, candidate_source_hashes = (
            _load_candidate_bundle(candidate_path, group.group_id)
        )
        declared_candidate_condition = candidate_payload.get("grounding_condition")
        if declared_candidate_condition != selected:
            raise StageInputError("candidate pool grounding condition mismatch")
        if selected != "oracle_gt_mask":
            if candidate_payload.get("grounding_mask_input_fingerprint") != (
                expected_mask_fingerprint
            ):
                raise StageInputError(
                    "predicted candidate pool is centred from another mask"
                )
            if candidate_payload.get("grounding_mask_commit_sha256") != sha256_file(
                mask.sidecar_path
            ):
                raise StageInputError(
                    "predicted candidate pool mask commit hash mismatch"
                )
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        skipped_grounding = (
            candidate_payload.get("generation_status") == "skipped_grounding_failure"
        )
        observation: RuntimeObservation | None
        if skipped_grounding:
            if candidates:
                raise StageInputError(
                    "grounding-failure bundle contains nonempty candidates"
                )
            terminal_path = _require_regular_file(
                candidate_payload.get("grounding_terminal_path", ""),
                "candidate grounding terminal source",
            )
            terminal = load_grounding_terminal(
                terminal_path,
                group_id=group.group_id,
                grounding_condition=selected,
            )
            if (
                Path(str(terminal.get("mask_path", ""))).resolve()
                != mask.mask_path.resolve()
                or terminal.get("mask_sha256") != sha256_file(mask.mask_path)
                or terminal.get("grounding_mask_input_fingerprint")
                != expected_mask_fingerprint
                or terminal.get("grounding_mask_commit_sha256")
                != sha256_file(mask.sidecar_path)
            ):
                raise StageInputError(
                    "grounding terminal does not bind the committed feature mask"
                )
            target_sources = {
                "depth": _require_regular_file(
                    group.target["depth_path"], "depth image"
                ),
                "meta": _require_regular_file(
                    group.target["meta_path"], "frame metadata"
                ),
                "intrinsics": _require_regular_file(
                    group.target["intrinsics_path"], "camera intrinsics"
                ),
                "camera_pose": _require_regular_file(
                    group.target["camera_pose_path"], "camera poses"
                ),
                "table_transform": _require_regular_file(
                    group.target["table_transform_path"], "table transform"
                ),
            }
            observation_hashes = {
                name: sha256_file(path) for name, path in target_sources.items()
            }
            terminal_sources = terminal.get("source_hashes")
            if not isinstance(terminal_sources, Mapping) or any(
                terminal_sources.get(str(path)) != observation_hashes[name]
                for name, path in target_sources.items()
            ):
                raise StageInputError(
                    "grounding terminal does not bind the feature depth/geometry sources"
                )
            observation = None
            scene_provenance = {
                "scene_feature_status": "unavailable_expected_grounding_failure",
                "reason": terminal["reason"],
                "mask_foreground_pixel_count": terminal["mask_foreground_pixel_count"],
                "valid_target_depth_pixel_count": 0,
                "tsdf_constructed": False,
                "vgn_inference_calls_for_group": 0,
                "scene_cloud_kind": "not_constructed_empty_grounding_pool",
                "grounding_terminal_path": str(terminal_path),
                "grounding_terminal_sha256": sha256_file(terminal_path),
            }
        else:
            observation, observation_hashes, scene_provenance = _depth_observation(
                group,
                mask,
                scene_cloud_kind=scene_cloud_kind,
                voxel_size_m=float(voxel_size_m),
            )
        split = str(group.target.get("split", "")).strip()
        if split not in {"train", "validation", "test"}:
            raise StageInputError(f"target group has unsupported split {split!r}")
        source_hashes = {
            str(target_manifest): target_manifest_sha,
            str(language_manifest): language_manifest_sha,
            str(mask.sidecar_path): sha256_file(mask.sidecar_path),
            str(mask.probability_path): sha256_file(mask.probability_path),
            str(mask.mask_path): sha256_file(mask.mask_path),
            str(candidate_source): sha256_file(candidate_source),
            **candidate_source_hashes,
            **{
                str(_require_regular_file(group.target[key], key)): digest
                for key, digest in (
                    ("depth_path", observation_hashes["depth"]),
                    ("meta_path", observation_hashes["meta"]),
                    ("intrinsics_path", observation_hashes["intrinsics"]),
                    ("camera_pose_path", observation_hashes["camera_pose"]),
                    ("table_transform_path", observation_hashes["table_transform"]),
                )
            },
        }
        input_contract = {
            "schema": FORMAL_FEATURE_SCHEMA,
            "group_id": group.group_id,
            "scene_id": str(group.target["scene_id"]),
            "split": split,
            "condition": selected,
            "mask_input_fingerprint": expected_mask_fingerprint,
            "mask_commit_fingerprint": mask.sidecar["commit_fingerprint"],
            "candidate_pool_fingerprint": candidate_payload[
                "candidate_pool_fingerprint"
            ],
            "candidate_count": len(candidates),
            "candidate_ids": candidate_ids,
            "feature_schema_sha256": feature_schema_sha256(default_feature_schema()),
            "scene_provenance": scene_provenance,
            "source_hashes": source_hashes,
            "ground_truth_inputs_consumed": ground_truth_inputs,
            "official_evaluator_outputs_consumed": [],
        }
        input_fingerprint = canonical_sha256(input_contract)
        feature_path, feature_sidecar_path = formal_feature_paths(
            output_root, selected, group.group_id
        )
        if feature_sidecar_path.exists():
            if not resume:
                raise StageInputError(
                    f"formal feature output already exists for {group.group_id}; use resume=True"
                )
            load_committed_formal_feature_table(
                feature_sidecar_path,
                expected_group_id=group.group_id,
                expected_condition=selected,
                expected_input_fingerprint=input_fingerprint,
                expected_candidate_ids=candidate_ids,
            )
            _source_hashes_unchanged(source_hashes)
            resumed += 1
            outputs.append(str(feature_sidecar_path))
            continue

        if skipped_grounding:
            runtime = pd.DataFrame(
                columns=[spec.name for spec in default_feature_schema()],
                dtype=np.float64,
            )
        else:
            if observation is None:
                raise AssertionError("nonempty candidate route lacks an observation")
            runtime = extractor(candidates, observation)
        if not isinstance(runtime, pd.DataFrame):
            raise StageInputError("runtime feature extractor must return a DataFrame")
        expected_runtime_columns = tuple(spec.name for spec in default_feature_schema())
        if tuple(map(str, runtime.columns)) != expected_runtime_columns:
            raise StageInputError(
                "runtime feature extractor violated the versioned schema"
            )
        if len(runtime) != len(candidates):
            raise StageInputError(
                "runtime feature extractor changed candidate membership"
            )
        identifiers = pd.DataFrame(
            {
                "group_id": [group.group_id] * len(candidates),
                "candidate_id": candidate_ids,
                "scene_id": [str(group.target["scene_id"])] * len(candidates),
                "split": [split] * len(candidates),
                "condition": [selected] * len(candidates),
            }
        )
        frame = pd.concat(
            [identifiers.reset_index(drop=True), runtime.reset_index(drop=True)], axis=1
        )
        if tuple(map(str, frame.columns)) != _expected_feature_columns():
            raise AssertionError("formal feature assembly changed the declared schema")
        if frame["candidate_id"].astype(str).tolist() != candidate_ids:
            raise AssertionError("formal feature assembly changed candidate ordering")
        _source_hashes_unchanged(source_hashes)
        # Re-run the complete mask trust chain immediately before publication;
        # this re-hashes RGB/model/adaptation sources that are transitively
        # bound by the mask commit but are not direct feature-file inputs.
        if selected == "oracle_gt_mask":
            load_committed_oracle_mask(
                mask.sidecar_path,
                expected_group_id=group.group_id,
                expected_instance_label_path=group.target["instance_label_path"],
                expected_target_instance_label=int(
                    group.target["target_instance_label"]
                ),
            )
        else:
            load_committed_predicted_mask(
                mask.sidecar_path,
                expected_group_id=group.group_id,
                expected_condition=selected,
                expected_input_fingerprint=expected_mask_fingerprint,
            )
        _atomic_parquet(feature_path, frame)
        commit: dict[str, Any] = {
            "schema_version": FORMAL_FEATURE_SCHEMA,
            "group_id": group.group_id,
            "scene_id": str(group.target["scene_id"]),
            "split": split,
            "condition": selected,
            "input_fingerprint": input_fingerprint,
            "input_contract": input_contract,
            "feature_file": feature_path.name,
            "feature_sha256": sha256_file(feature_path),
            "feature_schema_sha256": feature_schema_sha256(default_feature_schema()),
            "candidate_count": len(candidates),
            "candidate_ids": candidate_ids,
            "candidate_pool_fingerprint": candidate_payload[
                "candidate_pool_fingerprint"
            ],
            "ground_truth_inputs_consumed": ground_truth_inputs,
            "official_evaluator_outputs_consumed": [],
        }
        commit["commit_fingerprint"] = canonical_sha256(commit)
        atomic_json(feature_sidecar_path, commit)
        outputs.append(str(feature_sidecar_path))

    if (
        sha256_file(target_manifest) != target_manifest_sha
        or sha256_file(language_manifest) != language_manifest_sha
    ):
        raise StageInputError(
            "target/language manifest changed during formal feature stage"
        )
    return StageSummary(
        "formal_features",
        len(groups),
        len(groups) - resumed,
        resumed,
        tuple(outputs),
    )


def run_formal_feature_stage(
    target_manifest_path: str | os.PathLike[str],
    language_manifest_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    condition: GroundingCondition,
    mask_output_root: str | os.PathLike[str] | None = None,
    candidate_output_root: str | os.PathLike[str] | None = None,
    scene_cloud_kind: SceneCloudKind = "deterministic_voxel_downsample_of_full_depth",
    voxel_size_m: float = 0.005,
    resume: bool = False,
    extractor: Callable[
        [Sequence[Candidate6D], RuntimeObservation], pd.DataFrame
    ] = extract_candidate_features,
) -> StageSummary:
    """Run every group independently and atomically record complete failures."""

    selected = _grounding_condition(condition)
    target_manifest = _require_regular_file(target_manifest_path, "target manifest")
    language_manifest = _require_regular_file(
        language_manifest_path, "language manifest"
    )
    groups = load_target_language_jsonl(target_manifest, language_manifest)
    root = Path(output_root).expanduser().resolve()
    failures: list[GroupFailure] = []
    output_paths: list[str] = []
    completed = 0
    resumed = 0
    for group in groups:
        try:
            result = _run_formal_feature_stage_impl(
                target_manifest,
                language_manifest,
                root,
                condition=selected,
                mask_output_root=mask_output_root,
                candidate_output_root=candidate_output_root,
                scene_cloud_kind=scene_cloud_kind,
                voxel_size_m=voxel_size_m,
                resume=resume,
                extractor=extractor,
                only_group_id=group.group_id,
            )
            completed += result.completed_groups
            resumed += result.resumed_groups
            output_paths.extend(result.output_paths)
        except Exception as error:
            failures.append(
                _record_group_failure(
                    root,
                    stage="formal_features",
                    group_id=group.group_id,
                    error=error,
                )
            )
    if failures:
        raise StageBatchError("formal_features", failures)
    return StageSummary(
        "formal_features",
        len(groups),
        completed,
        resumed,
        tuple(output_paths),
    )


__all__ = [
    "ADAPTATION_EVIDENCE_SCHEMA",
    "FORMAL_FEATURE_SCHEMA",
    "GROUNDING_CONDITIONS",
    "PREDICTED_CONDITIONS",
    "PREDICTED_MASK_SCHEMA",
    "CommittedOracleMask",
    "CommittedPredictedMask",
    "ValidatedAdaptationEvidence",
    "formal_feature_paths",
    "group_artifact_slug",
    "load_committed_formal_feature_table",
    "load_committed_oracle_mask",
    "load_committed_predicted_mask",
    "load_validated_adapted_hifi",
    "oracle_mask_paths",
    "predicted_mask_paths",
    "run_formal_feature_stage",
    "run_predicted_mask_stage",
    "validate_adaptation_evidence",
]
