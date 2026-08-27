"""Leakage-safe orchestration for HiFi adaptation and grounding evaluation.

This module is deliberately separate from model code.  It builds explicit
train/validation row manifests, runs the retained decoder-only adaptation,
evaluates committed predicted masks against GraspNet labels, and selects the
formal predicted-mask track from validation mIoU only.  Test masks or metrics
are structurally unavailable to the selector.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .grounding import (
    DEFAULT_CLIP_WEIGHT,
    DEFAULT_HIFI_CHECKPOINT,
    EXPECTED_CLIP_WEIGHT_SHA256,
    EXPECTED_HIFI_CHECKPOINT_SHA256,
    AdaptationResult,
    adapt_hifi_decoder,
    compute_mask_metrics,
    summarize_mask_metrics,
    validate_adaptation_splits,
)
from .io import atomic_json, atomic_jsonl, canonical_sha256, sha256_file
from .stages import StageInputError, load_jsonl_records, load_target_language_jsonl


ADAPTATION_EVIDENCE_SCHEMA = "graspnet6d_hifi_adaptation_evidence_v1"
GROUNDING_METRICS_SCHEMA = "graspnet6d_grounding_metrics_v1"
PREDICTED_SELECTION_SCHEMA = "graspnet6d_predicted_condition_selection_v1"
PREDICTED_CONDITIONS = (
    "hifics_zero_shot_mask",
    "hifics_adapted_mask",
)


def _regular_file(path: Path | str, description: str) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise StageInputError(f"{description} must not be a symlink: {value}")
    source = value.resolve()
    if not source.is_file():
        raise StageInputError(f"missing regular {description}: {source}")
    return source


def _read_json(path: Path | str, description: str) -> dict[str, Any]:
    source = _regular_file(path, description)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageInputError(f"invalid {description}: {source}: {error}") from error
    if not isinstance(payload, dict):
        raise StageInputError(f"{description} must be a JSON object: {source}")
    return payload


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _condition(value: str) -> str:
    selected = str(value).strip()
    if selected not in PREDICTED_CONDITIONS:
        raise StageInputError(
            f"predicted condition must be one of {PREDICTED_CONDITIONS}, got {value!r}"
        )
    return selected


def _source_hashes_unchanged(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        for prefix in ("rgb", "instance_label"):
            path = _regular_file(row[f"{prefix}_path"], f"adaptation {prefix} source")
            expected = str(row.get(f"{prefix}_sha256", ""))
            if sha256_file(path) != expected:
                raise StageInputError(f"adaptation {prefix} source hash changed: {path}")


def build_adaptation_manifests(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Commit exact train/validation-only rows for decoder adaptation."""

    target_manifest = _regular_file(target_manifest_path, "target manifest")
    language_manifest = _regular_file(language_manifest_path, "language manifest")
    groups = load_target_language_jsonl(target_manifest, language_manifest)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for group in groups:
        source_split = str(group.target.get("split", ""))
        if source_split == "test":
            continue
        if source_split not in {"train", "validation"}:
            raise StageInputError(f"unsupported adaptation source split: {source_split!r}")
        destination_split = "train" if source_split == "train" else "val"
        rgb = _regular_file(group.target.get("rgb_path", ""), "adaptation RGB")
        label = _regular_file(
            group.target.get("instance_label_path", ""), "adaptation instance label"
        )
        target_instance_label = int(group.target.get("target_instance_label", 0))
        if target_instance_label <= 0:
            raise StageInputError("adaptation target_instance_label must be positive")
        rows[destination_split].append(
            {
                "sample_id": group.group_id,
                "group_id": group.group_id,
                "scene_id": str(group.target["scene_id"]),
                "split": destination_split,
                "rgb_path": str(rgb),
                "rgb_sha256": sha256_file(rgb),
                "instance_label_path": str(label),
                "instance_label_sha256": sha256_file(label),
                "target_instance_label": target_instance_label,
                "query": str(group.language["query"]),
                "template_family": str(group.language.get("template_family", "unknown")),
                "target_manifest_sha256": sha256_file(target_manifest),
                "language_manifest_sha256": sha256_file(language_manifest),
            }
        )
    try:
        split_contract = dict(validate_adaptation_splits(rows["train"], rows["val"]))
    except ValueError as error:
        raise StageInputError(f"adaptation split contract failed: {error}") from error
    root = Path(output_root).expanduser().resolve() / "grounding_adaptation"
    train_path = root / "train_rows.jsonl"
    validation_path = root / "validation_rows.jsonl"
    atomic_jsonl(train_path, rows["train"])
    atomic_jsonl(validation_path, rows["val"])
    _source_hashes_unchanged([*rows["train"], *rows["val"]])
    return {
        "train_manifest_path": str(train_path),
        "train_manifest_sha256": sha256_file(train_path),
        "validation_manifest_path": str(validation_path),
        "validation_manifest_sha256": sha256_file(validation_path),
        "split_contract": split_contract,
        "train_rows": len(rows["train"]),
        "validation_rows": len(rows["val"]),
        "test_rows_consumed": 0,
    }


def _record_stage_failure(output_root: Path, stage: str, error: BaseException) -> None:
    atomic_json(
        output_root / "errors" / stage / "stage.json",
        {
            "schema_version": 1,
            "stage": stage,
            "group_id": None,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        },
    )


def run_hifi_adaptation_stage(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    output_root: Path | str,
    *,
    device: str = "cpu",
    resume: bool = False,
    checkpoint_path: Path | str = DEFAULT_HIFI_CHECKPOINT,
    clip_weight_path: Path | str = DEFAULT_CLIP_WEIGHT,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    max_epochs: int = 30,
    patience: int = 5,
    minimum_improvement: float = 1e-5,
    seed: int = 20260815,
    trainer: Callable[..., AdaptationResult] = adapt_hifi_decoder,
    evidence_validator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Train only the retained decoder and publish validation-bound evidence."""

    root = Path(output_root).expanduser().resolve()
    stage_root = root / "grounding_adaptation"
    evidence_path = stage_root / "adaptation_evidence.json"
    if evidence_validator is None:
        from .formal_inputs import validate_adaptation_evidence

        evidence_validator = validate_adaptation_evidence
    try:
        manifests = build_adaptation_manifests(
            target_manifest_path, language_manifest_path, root
        )
        checkpoint = _regular_file(checkpoint_path, "base HiFi checkpoint")
        clip_weight = _regular_file(clip_weight_path, "CLIP weight")
        if sha256_file(checkpoint) != EXPECTED_HIFI_CHECKPOINT_SHA256:
            raise StageInputError("base HiFi checkpoint hash mismatch")
        if sha256_file(clip_weight) != EXPECTED_CLIP_WEIGHT_SHA256:
            raise StageInputError("CLIP weight hash mismatch")
        if evidence_path.exists():
            if not resume:
                raise StageInputError(
                    f"adaptation evidence already exists: {evidence_path}; use resume=True"
                )
            validated = evidence_validator(evidence_path)
            payload = _read_json(evidence_path, "adaptation evidence")
            payload["evidence_path"] = str(evidence_path)
            payload["evidence_sha256"] = sha256_file(evidence_path)
            payload["resumed"] = True
            del validated
            return payload
        train_rows = load_jsonl_records(
            manifests["train_manifest_path"], description="adaptation train rows"
        )
        validation_rows = load_jsonl_records(
            manifests["validation_manifest_path"],
            description="adaptation validation rows",
        )
        _source_hashes_unchanged([*train_rows, *validation_rows])
        output_checkpoint = stage_root / "hifics_decoder_adapted.pth"
        result = trainer(
            train_rows,
            validation_rows,
            device=device,
            checkpoint_path=checkpoint,
            clip_weight_path=clip_weight,
            output_checkpoint=output_checkpoint,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            minimum_improvement=minimum_improvement,
            seed=seed,
        )
        produced = _regular_file(output_checkpoint, "adapted HiFi checkpoint")
        if result.output_checkpoint is None or Path(result.output_checkpoint).resolve() != produced:
            raise StageInputError("adaptation trainer did not publish the requested checkpoint")
        if dict(result.split_contract) != manifests["split_contract"]:
            raise StageInputError("adaptation result split contract changed")
        if not 0.0 <= float(result.best_validation_miou) <= 1.0:
            raise StageInputError("adaptation result has invalid validation mIoU")
        _source_hashes_unchanged([*train_rows, *validation_rows])
        history_path = stage_root / "training_history.json"
        atomic_json(
            history_path,
            {
                "schema_version": "graspnet6d_hifi_adaptation_history_v1",
                "selection_split": "val",
                "test_rows_consumed": 0,
                "history": [_json_value(record) for record in result.history],
            },
        )
        evidence: dict[str, Any] = {
            "schema_version": ADAPTATION_EVIDENCE_SCHEMA,
            "scope": "formal_real_data",
            "fixture_only": False,
            "adaptation_checkpoint_path": str(produced),
            "adaptation_checkpoint_sha256": sha256_file(produced),
            "base_checkpoint_path": str(checkpoint),
            "base_checkpoint_sha256": sha256_file(checkpoint),
            "clip_weight_path": str(clip_weight),
            "clip_weight_sha256": sha256_file(clip_weight),
            "train_manifest_path": manifests["train_manifest_path"],
            "train_manifest_sha256": manifests["train_manifest_sha256"],
            "validation_manifest_path": manifests["validation_manifest_path"],
            "validation_manifest_sha256": manifests["validation_manifest_sha256"],
            "training_history_path": str(history_path),
            "training_history_sha256": sha256_file(history_path),
            "selection_metric": "validation_mean_iou",
            "selection_split": "val",
            "best_validation_mean_iou": float(result.best_validation_miou),
            "best_epoch": int(result.best_epoch),
            "epochs_completed": int(result.epochs_completed),
            "stopped_early": bool(result.stopped_early),
            "input_splits": ["train", "val"],
            "test_rows_consumed": 0,
            "split_contract": manifests["split_contract"],
        }
        evidence["evidence_fingerprint"] = canonical_sha256(evidence)
        atomic_json(evidence_path, evidence)
        evidence_validator(evidence_path)
        return {
            **evidence,
            "evidence_path": str(evidence_path),
            "evidence_sha256": sha256_file(evidence_path),
            "resumed": False,
        }
    except Exception as error:
        _record_stage_failure(root, "hifi_adaptation", error)
        raise


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _default_mask_loader(
    output_root: Path | str, condition: str, group_id: str
) -> Any:
    from .formal_inputs import load_committed_predicted_mask, predicted_mask_paths

    _, _, sidecar = predicted_mask_paths(output_root, condition, group_id)
    return load_committed_predicted_mask(
        sidecar,
        expected_group_id=group_id,
        expected_condition=condition,
    )


def run_grounding_metric_stage(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    mask_output_root: Path | str,
    output_root: Path | str,
    *,
    condition: str,
    included_splits: Sequence[str] = ("train", "validation", "test"),
    selection_scope: bool = False,
    resume: bool = False,
    mask_loader: Callable[[Path | str, str, str], Any] = _default_mask_loader,
) -> dict[str, Any]:
    """Recompute mask metrics from committed predictions and GT evaluator masks."""

    selected = _condition(condition)
    allowed = tuple(str(item) for item in included_splits)
    if not allowed or len(set(allowed)) != len(allowed):
        raise StageInputError("included_splits must be unique and non-empty")
    if not set(allowed).issubset({"train", "validation", "test"}):
        raise StageInputError("included_splits contains an unsupported split")
    if selection_scope and allowed != ("validation",):
        raise StageInputError(
            "predicted-condition selection metrics must contain validation only"
        )
    target_manifest = _regular_file(target_manifest_path, "target manifest")
    language_manifest = _regular_file(language_manifest_path, "language manifest")
    groups = [
        group
        for group in load_target_language_jsonl(target_manifest, language_manifest)
        if str(group.target.get("split")) in set(allowed)
    ]
    if not groups:
        raise StageInputError("grounding metric stage has no groups in the requested splits")
    metric_inputs: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {
        str(target_manifest): sha256_file(target_manifest),
        str(language_manifest): sha256_file(language_manifest),
    }
    for group in groups:
        prediction = mask_loader(mask_output_root, selected, group.group_id)
        label_path = _regular_file(
            group.target.get("instance_label_path", ""), "grounding GT instance label"
        )
        depth_path = _regular_file(
            group.target.get("depth_path", ""), "grounding metric depth"
        )
        with Image.open(label_path) as image:
            labels = np.asarray(image)
        with Image.open(depth_path) as image:
            depth = np.asarray(image)
        if labels.ndim != 2 or depth.shape != labels.shape:
            raise StageInputError(f"grounding label/depth shape mismatch: {group.group_id}")
        target = labels == int(group.target["target_instance_label"])
        other = (labels > 0) & ~target
        if not bool(target.any()):
            raise StageInputError(f"grounding GT target is empty: {group.group_id}")
        metrics = dict(
            compute_mask_metrics(
                prediction.binary_mask,
                target,
                valid_depth=depth,
                non_target_mask=other,
            )
        )
        split = str(group.target["split"])
        template = str(group.language.get("template_family", "unknown"))
        metric_inputs.append(
            {
                "sample_id": group.group_id,
                "group": split,
                "template": template,
                "prediction": prediction.binary_mask,
                "target": target,
                "valid_depth": depth,
                "non_target_mask": other,
            }
        )
        raw_rows.append(
            {
                "group_id": group.group_id,
                "scene_id": str(group.target["scene_id"]),
                "split": split,
                "condition": selected,
                "template_family": template,
                **metrics,
            }
        )
        for path in (
            prediction.sidecar_path,
            prediction.probability_path,
            prediction.mask_path,
            label_path,
            depth_path,
        ):
            source = _regular_file(path, "grounding metric source")
            source_hashes[str(source)] = sha256_file(source)
    summary = _json_value(summarize_mask_metrics(metric_inputs))
    input_fingerprint = canonical_sha256(
        {
            "schema": GROUNDING_METRICS_SCHEMA,
            "condition": selected,
            "included_splits": list(allowed),
            "selection_scope": bool(selection_scope),
            "source_hashes": source_hashes,
        }
    )
    scope_name = "selection_validation" if selection_scope else "all_requested_splits"
    stage_root = (
        Path(output_root).expanduser().resolve()
        / "grounding_metrics"
        / scope_name
        / selected
    )
    artifact_path = stage_root / "summary.json"
    raw_path = stage_root / "raw_metrics.csv"
    if artifact_path.exists():
        if not resume:
            raise StageInputError(f"grounding metrics already exist: {artifact_path}")
        artifact = _read_json(artifact_path, "grounding metric artifact")
        saved_raw = _regular_file(artifact.get("raw_metrics_path", ""), "saved grounding rows")
        if (
            artifact.get("input_fingerprint") != input_fingerprint
            or artifact.get("raw_metrics_sha256") != sha256_file(saved_raw)
        ):
            raise StageInputError("saved grounding metrics are stale or corrupt")
        return artifact
    _atomic_csv(raw_path, pd.DataFrame(raw_rows))
    artifact = {
        "schema_version": GROUNDING_METRICS_SCHEMA,
        "scope": "formal_real_data",
        "fixture_only": False,
        "condition": selected,
        "included_splits": list(allowed),
        "selection_scope": bool(selection_scope),
        "test_rows_consumed_for_selection": 0,
        "input_fingerprint": input_fingerprint,
        "source_hashes": source_hashes,
        "raw_metrics_path": str(raw_path),
        "raw_metrics_sha256": sha256_file(raw_path),
        "sample_count": len(raw_rows),
        "metrics": summary,
    }
    atomic_json(artifact_path, artifact)
    return artifact


def select_predicted_condition(
    zero_shot_validation_metrics_path: Path | str,
    adapted_validation_metrics_path: Path | str,
    output_path: Path | str,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Lock the primary predicted track from validation mIoU and nothing else."""

    sources = {
        "hifics_zero_shot_mask": _regular_file(
            zero_shot_validation_metrics_path, "zero-shot validation metrics"
        ),
        "hifics_adapted_mask": _regular_file(
            adapted_validation_metrics_path, "adapted validation metrics"
        ),
    }
    values: dict[str, float] = {}
    for condition, path in sources.items():
        payload = _read_json(path, f"{condition} validation metrics")
        if (
            payload.get("schema_version") != GROUNDING_METRICS_SCHEMA
            or payload.get("condition") != condition
            or payload.get("included_splits") != ["validation"]
            or payload.get("selection_scope") is not True
            or int(payload.get("test_rows_consumed_for_selection", -1)) != 0
            or payload.get("fixture_only") is not False
        ):
            raise StageInputError(
                f"{condition} metrics are not validation-only formal evidence"
            )
        try:
            value = float(payload["metrics"]["by_group"]["validation"]["mean_iou"])
        except (KeyError, TypeError, ValueError) as error:
            raise StageInputError(f"{condition} validation mIoU is missing") from error
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise StageInputError(f"{condition} validation mIoU is invalid")
        values[condition] = value
    # Prefer the immutable zero-shot checkpoint on an exact tie; adaptation is
    # selected only when validation provides a strictly higher mIoU.
    selected = (
        "hifics_adapted_mask"
        if values["hifics_adapted_mask"] > values["hifics_zero_shot_mask"]
        else "hifics_zero_shot_mask"
    )
    artifact: dict[str, Any] = {
        "schema_version": PREDICTED_SELECTION_SCHEMA,
        "scope": "formal_real_data",
        "fixture_only": False,
        "selection_metric": "validation_mean_iou",
        "selection_split": "validation",
        "test_rows_consumed": 0,
        "tie_break": "prefer_hifics_zero_shot_mask",
        "validation_mean_iou": values,
        "selected_condition": selected,
        "source_paths": {key: str(value) for key, value in sources.items()},
        "source_sha256": {key: sha256_file(value) for key, value in sources.items()},
    }
    artifact["selection_fingerprint"] = canonical_sha256(artifact)
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        saved = _read_json(destination, "predicted-condition selection")
        if not resume:
            raise StageInputError(f"predicted-condition selection exists: {destination}")
        if saved != artifact:
            raise StageInputError("saved predicted-condition selection is stale")
        return saved
    atomic_json(destination, artifact)
    return artifact


__all__ = [
    "ADAPTATION_EVIDENCE_SCHEMA",
    "GROUNDING_METRICS_SCHEMA",
    "PREDICTED_CONDITIONS",
    "PREDICTED_SELECTION_SCHEMA",
    "build_adaptation_manifests",
    "run_grounding_metric_stage",
    "run_hifi_adaptation_stage",
    "select_predicted_condition",
]
