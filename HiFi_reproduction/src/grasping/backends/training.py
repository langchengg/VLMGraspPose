"""Deterministic OCID-VLG fine-tuning for official GR-ConvNet/GG-CNN2 models."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..common import EvaluatorConfig, NMSConfig, decode_quality_maps
from ..common import evaluate_ocid_predictions, non_maximum_suppression
from ..common.sample_io import CompactSampleLoader, aligned_labels
from ..common.training_targets import build_dense_grasp_targets
from .conditioning import ConditionedInput, condition_rgbd
from .network_utils import (
    gated_quality_map,
    load_ggcnn2_state_dict,
    load_trusted_grconvnet_full_pickle,
    official_gaussian_post_process,
    rescore_candidates_with_mask_support,
    select_device,
    synchronize_device,
)


BackendName = Literal["grconvnet", "ggcnn2"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    backend: BackendName
    input_size: int
    conditioning_variant: Literal["hard_mask", "dilated_crop"]
    dilation_fraction: float = 0.15
    minimum_crop_side_px: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 0
    seed: int = 20260803
    num_workers: int = 0
    device: str = "auto"
    quality_threshold: float = 0.2
    min_peak_distance_px: int = 20
    max_raw_candidates: int = 100
    width_scale_px: float = 150.0
    fixed_height_px: float = 20.0
    center_gate_exponent: float = 1.0
    jaw_gate_exponent: float = 0.0
    nms_center_distance_px: float = 8.0
    nms_angle_distance_deg: float = 15.0
    nms_width_distance_px: float = 10.0
    nms_iou_threshold: float = 0.25
    grconvnet_source_checkpoint: str | None = None
    grconvnet_source_checkpoint_sha256: str | None = None
    grconvnet_input_channels: int = 4

    def __post_init__(self) -> None:
        if self.backend not in ("grconvnet", "ggcnn2"):
            raise ValueError("unsupported backend")
        if self.conditioning_variant not in ("hard_mask", "dilated_crop"):
            raise ValueError("unsupported conditioning variant")
        if min(self.input_size, self.max_epochs, self.patience) <= 0:
            raise ValueError("sizes, epochs and patience must be positive")
        if self.batch_size < 0 or self.num_workers < 0:
            raise ValueError("batch size/workers cannot be negative")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer settings")
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in (self.center_gate_exponent, self.jaw_gate_exponent)
        ):
            raise ValueError("gate exponents must be finite and non-negative")
        if self.backend == "grconvnet" and self.grconvnet_input_channels not in (1, 4):
            raise ValueError("GR-ConvNet channels must be 1 or 4")


@dataclass(slots=True)
class TrainingExample:
    sample_id: str
    model_input: np.ndarray
    targets: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    conditioned: ConditionedInput
    gt_corners: Sequence[Sequence[Sequence[float]]]


class OcidGraspTrainingDataset(Dataset[TrainingExample | None]):
    """Prediction-conditioned inputs and separate GT grasp supervision."""

    def __init__(
        self,
        deployment: Sequence[Mapping[str, Any]],
        labels: Sequence[Mapping[str, Any]],
        config: TrainingConfig,
    ) -> None:
        self.deployment = list(deployment)
        self.labels = list(aligned_labels(self.deployment, labels))
        self.config = config
        self.loader = CompactSampleLoader()

    def __len__(self) -> int:
        return len(self.deployment)

    def __getitem__(self, index: int) -> TrainingExample | None:
        row = self.deployment[index]
        label = self.labels[index]
        arrays = self.loader.load(row, mask_source="predicted", labels=None)
        try:
            conditioned = condition_rgbd(
                rgb=arrays.rgb,
                depth_m=arrays.depth_m,
                binary_mask=arrays.binary_mask,
                probability=arrays.probability,
                variant=self.config.conditioning_variant,
                output_size=self.config.input_size,
                dilation_fraction=self.config.dilation_fraction,
                minimum_side_px=self.config.minimum_crop_side_px,
            )
        except ValueError as error:
            if str(error) in ("empty_mask", "invalid_depth", "invalid_target_depth"):
                return None
            raise
        dense = build_dense_grasp_targets(
            label["gt_grasp_rectangles"],
            transform=conditioned.transform,
            output_shape=(self.config.input_size, self.config.input_size),
            width_scale_px=self.config.width_scale_px,
            fixed_height_px=self.config.fixed_height_px,
        )
        if self.config.backend == "grconvnet" and self.config.grconvnet_input_channels == 4:
            # Official GraspDataset concatenates [depth, R, G, B].
            model_input = np.concatenate(
                (conditioned.depth_chw, conditioned.rgb_chw), axis=0
            )
        else:
            model_input = conditioned.depth_chw
        return TrainingExample(
            sample_id=arrays.sample_id,
            model_input=np.ascontiguousarray(model_input, dtype=np.float32),
            targets=(dense.quality, dense.cos_2theta, dense.sin_2theta, dense.width),
            conditioned=conditioned,
            gt_corners=label["gt_grasp_rectangles"],
        )


def collate_training_examples(
    values: Sequence[TrainingExample | None],
) -> dict[str, Any] | None:
    examples = [value for value in values if value is not None]
    if not examples:
        return None
    model_input = torch.from_numpy(np.stack([item.model_input for item in examples]))
    targets = tuple(
        torch.from_numpy(np.stack([item.targets[index] for item in examples]))[:, None]
        for index in range(4)
    )
    return {"input": model_input, "targets": targets, "examples": examples}


def _loss(
    backend: BackendName,
    outputs: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    loss_function = F.smooth_l1_loss if backend == "grconvnet" else F.mse_loss
    components = tuple(loss_function(output, target) for output, target in zip(outputs, targets))
    return sum(components), components


def _nms(config: TrainingConfig) -> NMSConfig:
    return NMSConfig(
        center_distance_px=config.nms_center_distance_px,
        angle_distance_deg=config.nms_angle_distance_deg,
        width_distance_px=config.nms_width_distance_px,
        rectangle_iou_threshold=config.nms_iou_threshold,
    )


def _model_from_official(config: TrainingConfig) -> tuple[nn.Module, str, str]:
    if config.backend == "grconvnet":
        kwargs: dict[str, Any] = {
            "device": "cpu",
            "expected_input_channels": config.grconvnet_input_channels,
        }
        if config.grconvnet_source_checkpoint is not None:
            kwargs["checkpoint_path"] = config.grconvnet_source_checkpoint
        if config.grconvnet_source_checkpoint_sha256 is not None:
            kwargs["expected_sha256"] = config.grconvnet_source_checkpoint_sha256
        model, digest = load_trusted_grconvnet_full_pickle(**kwargs)
        source = "official_grconvnet_jacquard_rgbd"
    else:
        model, digest = load_ggcnn2_state_dict(device="cpu")
        source = "official_ggcnn2_cornell_depth"
    model.requires_grad_(True)
    return model, digest, source


def load_finetuned_model(
    checkpoint_path: str | Path,
    *,
    backend: BackendName,
    device: str = "cpu",
) -> tuple[nn.Module, str, Mapping[str, Any]]:
    """Strictly restore a locally trained state_dict over its official initializer."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 2
        or payload.get("backend") != backend
    ):
        raise ValueError("fine-tuned checkpoint backend mismatch")
    saved_lineage = payload.get("data_lineage")
    if not isinstance(saved_lineage, Mapping):
        raise ValueError("fine-tuned checkpoint lacks training data lineage")
    lineage_without_digest = dict(saved_lineage)
    claimed_lineage_sha = lineage_without_digest.pop("content_sha256", None)
    if (
        claimed_lineage_sha != _canonical_sha256(lineage_without_digest)
        or payload.get("data_lineage_sha256") != claimed_lineage_sha
    ):
        raise ValueError("fine-tuned checkpoint training data lineage mismatch")
    if backend == "grconvnet":
        saved_config = payload.get("training_config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("fine-tuned GR checkpoint lacks training config")
        loader_kwargs: dict[str, Any] = {
            "device": "cpu",
            "expected_input_channels": int(
                saved_config.get("grconvnet_input_channels", 4)
            ),
        }
        if saved_config.get("grconvnet_source_checkpoint"):
            loader_kwargs["checkpoint_path"] = saved_config[
                "grconvnet_source_checkpoint"
            ]
        if saved_config.get("grconvnet_source_checkpoint_sha256"):
            loader_kwargs["expected_sha256"] = saved_config[
                "grconvnet_source_checkpoint_sha256"
            ]
        model, source_sha = load_trusted_grconvnet_full_pickle(**loader_kwargs)
    else:
        model, source_sha = load_ggcnn2_state_dict(device="cpu")
    if str(payload.get("source_checkpoint_sha256")) != source_sha:
        raise ValueError("fine-tuned checkpoint source lineage mismatch")
    best_validation = payload.get("best_validation")
    if not isinstance(best_validation, Mapping):
        raise ValueError("fine-tuned checkpoint lacks validation evidence")
    _assert_finite_metrics(best_validation)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False).eval()
    model = model.to(device=select_device(device), dtype=torch.float32)
    return model, digest, payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_ids(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> tuple[str, ...]:
    ids = tuple(str(row["sample_id"]) for row in rows)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{label} sample IDs must be non-empty and unique")
    return ids


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in ids).encode()).hexdigest()


def _validated_data_lineage(
    *,
    train_deployment: Sequence[Mapping[str, Any]],
    train_labels: Sequence[Mapping[str, Any]],
    validation_deployment: Sequence[Mapping[str, Any]],
    validation_labels: Sequence[Mapping[str, Any]],
    declared: Mapping[str, Any] | None,
) -> dict[str, Any]:
    train_ids = _ordered_ids(train_deployment, label="train deployment")
    train_label_ids = _ordered_ids(train_labels, label="train labels")
    validation_ids = _ordered_ids(
        validation_deployment, label="validation deployment"
    )
    validation_label_ids = _ordered_ids(validation_labels, label="validation labels")
    if train_ids != train_label_ids or validation_ids != validation_label_ids:
        raise ValueError("deployment/label sample order mismatch")
    if set(train_ids) & set(validation_ids):
        raise ValueError("train/validation sample IDs overlap")
    identity = {
        "train": {
            "count": len(train_ids),
            "ordered_sample_ids_sha256": _ids_sha256(train_ids),
        },
        "validation": {
            "count": len(validation_ids),
            "ordered_sample_ids_sha256": _ids_sha256(validation_ids),
        },
    }
    if declared is None:
        lineage: dict[str, Any] = {
            "schema_version": 1,
            "source": "validated_in_memory_sequences",
            "identity": identity,
            "train_validation_overlap": 0,
            "test_manifest_read": False,
        }
        lineage["content_sha256"] = _canonical_sha256(lineage)
        return lineage
    lineage = json.loads(
        json.dumps(declared, sort_keys=True, default=str, allow_nan=False)
    )
    claimed_digest = lineage.pop("content_sha256", None)
    if (
        lineage.get("schema_version") != 1
        or lineage.get("identity") != identity
        or lineage.get("train_validation_overlap") != 0
        or lineage.get("test_manifest_read") is not False
        or claimed_digest != _canonical_sha256(lineage)
    ):
        raise ValueError("declared training data lineage does not match loaded records")
    lineage["content_sha256"] = claimed_digest
    return lineage


def _assert_finite_metrics(row: Mapping[str, Any]) -> None:
    numeric_fields = (
        "training_loss",
        "training_position_loss",
        "training_cos_loss",
        "training_sin_loss",
        "training_width_loss",
        "validation_loss",
        "j_at_1",
        "j_at_5",
        "non_empty_rate",
        "validation_elapsed_seconds",
        "elapsed_seconds",
    )
    for field in numeric_fields:
        if field in row and not math.isfinite(float(row[field])):
            raise FloatingPointError(f"non-finite training metric: {field}")


def _epoch_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:epoch-v1".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def _atomic_curve_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("training curve schema changed between epochs")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _score_tuple(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row["j_at_1"]),
        float(row["j_at_5"]),
        float(row["non_empty_rate"]),
        -float(row["validation_loss"]),
        -float(row.get("validation_elapsed_seconds", float("inf"))),
    )


def _run_validation(
    model: nn.Module,
    loader: DataLoader[Any],
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    loss_total = 0.0
    loss_examples = 0
    j1 = j5 = non_empty = evaluated_count = 0
    nms_config = _nms(config)
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            inputs = batch["input"].to(device=device, dtype=torch.float32)
            targets = tuple(
                value.to(device=device, dtype=torch.float32) for value in batch["targets"]
            )
            outputs = model(inputs)
            batch_loss, _ = _loss(config.backend, outputs, targets)
            if not bool(torch.isfinite(batch_loss).all()):
                raise FloatingPointError("non-finite validation loss")
            count = len(batch["examples"])
            loss_total += float(batch_loss.detach().cpu()) * count
            loss_examples += count
            for index, example in enumerate(batch["examples"]):
                sliced = tuple(value[index : index + 1] for value in outputs)
                network_quality, cos_map, sin_map, width_map = official_gaussian_post_process(
                    sliced,
                    expected_spatial_shape=(config.input_size, config.input_size),
                )
                quality = gated_quality_map(
                    network_quality,
                    probability_gate=example.conditioned.gate_map,
                    valid_depth_gate=example.conditioned.valid_depth_map,
                    center_gate_exponent=config.center_gate_exponent,
                )
                raw = decode_quality_maps(
                    quality,
                    cos_map,
                    sin_map,
                    width_map,
                    sample_id=example.sample_id,
                    backend=config.backend,
                    transform=example.conditioned.transform,
                    quality_threshold=config.quality_threshold,
                    min_peak_distance_px=config.min_peak_distance_px,
                    max_peaks=config.max_raw_candidates,
                    width_scale=1.0,
                    fixed_height_px=config.fixed_height_px,
                )
                raw = [
                    candidate
                    for candidate in rescore_candidates_with_mask_support(
                        raw,
                        network_quality_map=network_quality,
                        probability_gate=example.conditioned.gate_map,
                        transform=example.conditioned.transform,
                        center_gate_exponent=config.center_gate_exponent,
                        jaw_gate_exponent=config.jaw_gate_exponent,
                    )
                    if candidate.score > config.quality_threshold
                ]
                kept = non_maximum_suppression(raw, nms_config)
                outcome = evaluate_ocid_predictions(
                    kept,
                    example.gt_corners,
                    EvaluatorConfig(fixed_height_px=config.fixed_height_px),
                )
                evaluated_count += 1
                non_empty += bool(kept)
                j1 += outcome.j_at_1
                j5 += outcome.j_at_5
    synchronize_device(device)
    # Keep invalid/empty deployment inputs in the all-sample denominator as
    # failures.  Only the dense validation loss excludes samples for which an
    # input tensor cannot be constructed.
    manifest_count = len(loader.dataset)
    denominator = max(manifest_count, 1)
    return {
        "validation_loss": loss_total / max(loss_examples, 1),
        "validation_evaluated": manifest_count,
        "validation_valid_inputs": evaluated_count,
        "j_at_1": j1 / denominator,
        "j_at_5": j5 / denominator,
        "non_empty_rate": non_empty / denominator,
    }


def _run_training_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float | int]:
    model.train()
    loss_total = 0.0
    component_totals = np.zeros(4, dtype=np.float64)
    examples = skipped_batches = 0
    for batch in loader:
        if batch is None:
            skipped_batches += 1
            continue
        inputs = batch["input"].to(device=device, dtype=torch.float32)
        targets = tuple(
            value.to(device=device, dtype=torch.float32) for value in batch["targets"]
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss, components = _loss(config.backend, outputs, targets)
        if not bool(torch.isfinite(loss).all()) or any(
            not bool(torch.isfinite(component).all()) for component in components
        ):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        optimizer.step()
        count = len(batch["examples"])
        loss_total += float(loss.detach().cpu()) * count
        component_totals += np.asarray(
            [float(value.detach().cpu()) for value in components]
        ) * count
        examples += count
    synchronize_device(device)
    divisor = max(examples, 1)
    return {
        "training_loss": loss_total / divisor,
        "training_position_loss": float(component_totals[0] / divisor),
        "training_cos_loss": float(component_totals[1] / divisor),
        "training_sin_loss": float(component_totals[2] / divisor),
        "training_width_loss": float(component_totals[3] / divisor),
        "training_examples": int(examples),
        "skipped_batches": int(skipped_batches),
    }


def _is_oom_runtime_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "mps backend out of memory",
            "not enough memory",
            "allocator ran out",
        )
    )


def _cpu_clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_tree(item) for item in value)
    return value


def _probe_optimizer_step(
    model: nn.Module,
    dataset: OcidGraspTrainingDataset,
    config: TrainingConfig,
    device: torch.device,
    batch_size: int,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_training_examples,
        drop_last=True,
    )
    batch = next(
        value
        for value in loader
        if value is not None and len(value["examples"]) == batch_size
    )
    inputs = batch["input"].to(device=device, dtype=torch.float32)
    targets = tuple(
        value.to(device=device, dtype=torch.float32) for value in batch["targets"]
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss, _ = _loss(config.backend, outputs, targets)
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("non-finite auto-batch probe loss")
    loss.backward()
    optimizer.step()
    synchronize_device(device)


def _auto_batch_size(
    model: nn.Module,
    dataset: OcidGraspTrainingDataset,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    """Try 8/4/2/1 with a full real batch and the first Adam update."""

    attempts: list[dict[str, Any]] = []
    baseline = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

    def restore_model() -> None:
        model.load_state_dict(baseline, strict=True)
        model.zero_grad(set_to_none=True)

    for candidate in (8, 4, 2, 1):
        oom_attempt: dict[str, Any] | None = None
        try:
            restore_model()
            model.train()
            _probe_optimizer_step(model, dataset, config, device, candidate)
        except StopIteration as error:
            restore_model()
            raise RuntimeError(
                f"could not construct a full valid probe batch of size {candidate}"
            ) from error
        except RuntimeError as error:
            if not _is_oom_runtime_error(error):
                restore_model()
                raise RuntimeError(
                    f"auto-batch probe failed for a non-OOM reason at batch {candidate}"
                ) from error
            oom_attempt = {
                "batch_size": candidate,
                "status": "OOM",
                "error": f"{type(error).__name__}:{error}",
            }
        if oom_attempt is not None:
            attempts.append(oom_attempt)
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            restore_model()
            continue
        attempts.append({"batch_size": candidate, "status": "PASS"})
        restore_model()
        return candidate, attempts
    raise RuntimeError(f"all preregistered batch sizes failed: {attempts}")


def _epoch_training_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    config: TrainingConfig,
    epoch: int,
) -> DataLoader[Any]:
    """Build an epoch-local loader whose order is independent of resume history."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_training_examples,
        generator=torch.Generator().manual_seed(_epoch_seed(config.seed, epoch)),
        persistent_workers=False,
    )


def train_finetuned_backend(
    *,
    train_deployment: Sequence[Mapping[str, Any]],
    train_labels: Sequence[Mapping[str, Any]],
    validation_deployment: Sequence[Mapping[str, Any]],
    validation_labels: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    config: TrainingConfig,
    data_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one registered configuration and preserve only best + resume state."""

    validated_lineage = _validated_data_lineage(
        train_deployment=train_deployment,
        train_labels=train_labels,
        validation_deployment=validation_deployment,
        validation_labels=validation_labels,
        declared=data_lineage,
    )
    lineage_sha = str(validated_lineage["content_sha256"])
    config_value = json.loads(
        json.dumps(asdict(config), sort_keys=True, default=str, allow_nan=False)
    )
    _seed_everything(config.seed)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    complete_path = destination / "training_complete.json"
    if complete_path.exists():
        raise FileExistsError(f"training job already complete: {complete_path}")
    model, source_checkpoint_sha, initialization = _model_from_official(config)
    device = select_device(config.device)
    model = model.to(device=device, dtype=torch.float32)
    train_dataset = OcidGraspTrainingDataset(train_deployment, train_labels, config)
    validation_dataset = OcidGraspTrainingDataset(
        validation_deployment, validation_labels, config
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    resume_path = destination / "resume_state.pt"
    best_path = destination / "best_state_dict.pt"
    curves_path = destination / "training_curves.csv"
    start_epoch = 1
    best_row: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(resume, Mapping)
            or resume.get("schema_version") != 2
            or resume.get("backend") != config.backend
            or resume.get("training_config") != config_value
            or resume.get("data_lineage") != validated_lineage
            or resume.get("data_lineage_sha256") != lineage_sha
            or resume.get("source_checkpoint_sha256") != source_checkpoint_sha
            or resume.get("rng_schedule") != "sha256(seed,epoch)-v1"
        ):
            raise ValueError("resume config/data/source lineage mismatch")
        batch_size = int(resume.get("batch_size", 0))
        batch_size_attempts = [
            dict(row) for row in resume.get("batch_size_attempts", [])
        ]
        if (
            batch_size <= 0
            or (config.batch_size and batch_size != config.batch_size)
            or (not config.batch_size and batch_size not in (8, 4, 2, 1))
            or not batch_size_attempts
        ):
            raise ValueError("resume batch-size contract mismatch")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_row = resume.get("best_row")
        stale_epochs = int(resume.get("stale_epochs", 0))
        history = [dict(row) for row in resume.get("history", [])]
        if not history or int(history[-1]["epoch"]) != int(resume["epoch"]):
            raise ValueError("resume history/epoch mismatch")
        if [int(row["epoch"]) for row in history] != list(
            range(1, int(resume["epoch"]) + 1)
        ):
            raise ValueError("resume epoch sequence mismatch")
        for row in history:
            _assert_finite_metrics(row)
        if (
            not isinstance(best_row, Mapping)
            or dict(best_row) != max(history, key=_score_tuple)
            or stale_epochs
            != int(resume["epoch"]) - int(best_row.get("epoch", -1))
            or not best_path.is_file()
            or resume.get("best_checkpoint_sha256")
            != hashlib.sha256(best_path.read_bytes()).hexdigest()
        ):
            raise ValueError("resume best-checkpoint/history mismatch")
        _atomic_curve_csv(curves_path, history)
    else:
        if config.batch_size:
            batch_size = config.batch_size
            batch_size_attempts = [
                {"batch_size": batch_size, "status": "EXPLICIT"}
            ]
        else:
            batch_size, batch_size_attempts = _auto_batch_size(
                model, train_dataset, config, device
            )

    started = time.perf_counter()
    elapsed_before_resume = float(history[-1]["elapsed_seconds"]) if history else 0.0
    termination_reason = (
        "early_stopping_resume_boundary"
        if stale_epochs >= config.patience
        else "max_epochs_resume_boundary"
        if start_epoch > config.max_epochs
        else None
    )
    for epoch in range(start_epoch, config.max_epochs + 1):
        if termination_reason is not None:
            break
        epoch_seed = _epoch_seed(config.seed, epoch)
        while True:
            _seed_everything(epoch_seed)
            model_before_epoch = _cpu_clone_tree(model.state_dict())
            optimizer_before_epoch = _cpu_clone_tree(optimizer.state_dict())
            train_loader = _epoch_training_loader(
                train_dataset,
                batch_size=batch_size,
                config=config,
                epoch=epoch,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collate_training_examples,
                generator=torch.Generator().manual_seed(epoch_seed ^ 0x5DEECE66D),
                persistent_workers=False,
            )
            try:
                train_metrics = _run_training_epoch(
                    model, train_loader, optimizer, config, device
                )
                validation_started = time.perf_counter()
                validation_metrics = _run_validation(
                    model, validation_loader, config, device
                )
            except RuntimeError as error:
                sequence = (8, 4, 2, 1)
                if (
                    config.batch_size
                    or not _is_oom_runtime_error(error)
                    or batch_size not in sequence
                    or batch_size == 1
                ):
                    raise
                next_batch_size = sequence[sequence.index(batch_size) + 1]
                model.zero_grad(set_to_none=True)
                if device.type == "mps":
                    torch.mps.empty_cache()
                model.load_state_dict(model_before_epoch, strict=True)
                optimizer.load_state_dict(optimizer_before_epoch)
                batch_size_attempts.append(
                    {
                        "batch_size": batch_size,
                        "status": "OOM_DURING_EPOCH_ROLLED_BACK",
                        "epoch": epoch,
                        "error": f"{type(error).__name__}:{error}",
                    }
                )
                batch_size = next_batch_size
                batch_size_attempts.append(
                    {
                        "batch_size": batch_size,
                        "status": "RETRY_AFTER_EPOCH_OOM",
                        "epoch": epoch,
                    }
                )
                continue
            break
        row: dict[str, Any] = {
            "epoch": epoch,
            **train_metrics,
            **validation_metrics,
            "validation_elapsed_seconds": time.perf_counter() - validation_started,
            "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
        }
        _assert_finite_metrics(row)
        history.append(row)
        improved = best_row is None or _score_tuple(row) > _score_tuple(best_row)
        if improved:
            best_row = row
            stale_epochs = 0
            _atomic_torch_save(
                {
                    "schema_version": 2,
                    "backend": config.backend,
                    "model_state_dict": model.state_dict(),
                    "source_checkpoint_sha256": source_checkpoint_sha,
                    "initialization": initialization,
                    "training_config": config_value,
                    "data_lineage": validated_lineage,
                    "data_lineage_sha256": lineage_sha,
                    "best_epoch": epoch,
                    "best_validation": best_row,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        resume_payload: dict[str, Any] = {
            "schema_version": 2,
            "backend": config.backend,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_row": best_row,
            "stale_epochs": stale_epochs,
            "training_config": config_value,
            "data_lineage": validated_lineage,
            "data_lineage_sha256": lineage_sha,
            "source_checkpoint_sha256": source_checkpoint_sha,
            "batch_size": batch_size,
            "batch_size_attempts": batch_size_attempts,
            "best_checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
            "rng_schedule": "sha256(seed,epoch)-v1",
            "epoch_seed": epoch_seed,
            "torch_rng_state": torch.get_rng_state(),
            "history": history,
        }
        if device.type == "mps":
            resume_payload["mps_rng_state"] = torch.mps.get_rng_state()
        _atomic_torch_save(resume_payload, resume_path)
        _atomic_curve_csv(curves_path, history)
        print(
            json.dumps(
                {
                    "status": "EPOCH_COMPLETE",
                    "backend": config.backend,
                    "epoch": epoch,
                    "j_at_1": row["j_at_1"],
                    "j_at_5": row["j_at_5"],
                    "validation_loss": row["validation_loss"],
                    "stale_epochs": stale_epochs,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stale_epochs >= config.patience:
            termination_reason = "early_stopping"
            break
    if termination_reason is None:
        termination_reason = "max_epochs"
    if best_row is None or not best_path.is_file():
        raise RuntimeError("training produced no valid checkpoint")
    _assert_finite_metrics(best_row)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(best_payload, Mapping)
        or best_payload.get("schema_version") != 2
        or best_payload.get("backend") != config.backend
        or best_payload.get("training_config") != config_value
        or best_payload.get("data_lineage") != validated_lineage
        or best_payload.get("data_lineage_sha256") != lineage_sha
        or best_payload.get("best_validation") != best_row
        or int(best_payload.get("best_epoch", -1)) != int(best_row["epoch"])
    ):
        raise ValueError("best checkpoint evidence mismatch")
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    _atomic_curve_csv(curves_path, history)
    curves_sha = hashlib.sha256(curves_path.read_bytes()).hexdigest()
    finalized_best_payload = dict(best_payload)
    finalized_best_payload.update(
        {
            "training_curves_sha256": curves_sha,
            "epochs_completed": len(history),
            "last_epoch": int(history[-1]["epoch"]),
            "termination_reason": termination_reason,
        }
    )
    _atomic_torch_save(finalized_best_payload, best_path)
    best_sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    result = {
        "schema_version": 2,
        "status": "COMPLETE",
        "backend": config.backend,
        "device": device.type,
        "batch_size": batch_size,
        "batch_size_attempts": batch_size_attempts,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": best_sha,
        "best_validation": best_row,
        "source_checkpoint_sha256": source_checkpoint_sha,
        "training_config": config_value,
        "data_lineage": validated_lineage,
        "data_lineage_sha256": lineage_sha,
        "epochs_completed": len(history),
        "last_epoch": int(history[-1]["epoch"]),
        "termination_reason": termination_reason,
        "elapsed_seconds": float(history[-1]["elapsed_seconds"]),
        "artifacts": {
            "best_state_dict.pt": best_sha,
            "training_curves.csv": curves_sha,
        },
    }
    temporary = complete_path.with_name(f".{complete_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, complete_path)
    resume_path.unlink(missing_ok=True)
    return result
