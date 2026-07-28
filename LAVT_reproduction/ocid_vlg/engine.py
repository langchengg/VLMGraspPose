"""Shared LAVT/OCID-VLG training and evaluation engine."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import logging
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset

from data.dataset_ocid_vlg_bert import OCIDVLGLAVTDataset
from lib import segmentation

from .device import resolve_device, should_pin_memory
from .losses import build_loss
from .metrics import aggregate_metrics, compute_sample_metrics


LOGGER = logging.getLogger("lavt.ocid_vlg")


def configure_logging(log_path: Path | None = None) -> logging.Logger:
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
    return LOGGER


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    try:
        disk = shutil.disk_usage(Path.cwd())
        disk_free = disk.free
    except OSError:
        disk_free = None
    memory_bytes = None
    try:
        import psutil

        memory_bytes = psutil.virtual_memory().total
    except Exception:
        pass
    chip = None
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "chip": chip,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "memory_bytes": memory_bytes,
        "free_disk_bytes": disk_free,
    }


def _dataset_kwargs(args: Namespace, split: str, manifest: str) -> dict[str, Any]:
    values = {
        "ocid_root": args.ocid_root,
        "root_dir": args.ocid_root,
        "root": args.ocid_root,
        "ocid_api_root": args.ocid_api_root,
        "api_root": args.ocid_api_root,
        "split": split,
        "manifest_path": manifest or None,
        "manifest": manifest or None,
        "dataset_version": args.ocid_version,
        "version": args.ocid_version,
        "img_size": args.img_size,
        "image_size": args.img_size,
        "max_tokens": args.max_tokens,
        "tokenizer_name": args.bert_tokenizer,
        "tokenizer_name_or_path": args.bert_tokenizer,
        "bert_tokenizer": args.bert_tokenizer,
    }
    signature = inspect.signature(OCIDVLGLAVTDataset)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return {
        key: value
        for key, value in values.items()
        if value is not None and (accepts_kwargs or key in signature.parameters)
    }


def build_dataset(
    args: Namespace,
    split: str,
    manifest: str,
    limit: int | None = None,
) -> Dataset[Any]:
    dataset: Dataset[Any] = OCIDVLGLAVTDataset(**_dataset_kwargs(args, split, manifest))
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"sample limit must be positive, got {limit}")
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def build_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    train: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=should_pin_memory(device),
        drop_last=False,
    )


def build_model(
    args: Namespace,
    device: torch.device,
    *,
    initialize_backbone: bool,
) -> torch.nn.Module:
    if args.model not in segmentation.__dict__:
        raise ValueError(f"unknown LAVT model: {args.model}")
    pretrained = args.pretrained_swin_weights if initialize_backbone else ""
    if initialize_backbone and not pretrained:
        raise FileNotFoundError(
            "Swin initialization is mandatory; --pretrained_swin_weights is empty"
        )
    if initialize_backbone and not Path(pretrained).is_file():
        raise FileNotFoundError(f"Swin initialization checkpoint is missing: {pretrained}")
    model = segmentation.__dict__[args.model](pretrained=pretrained, args=args)
    return model.to(device)


def official_parameter_groups(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Reproduce LAVT's trainable parameter groups, including BERT layers 0-9."""

    unwrapped = model.module if hasattr(model, "module") else model
    backbone_no_decay: list[torch.nn.Parameter] = []
    backbone_decay: list[torch.nn.Parameter] = []
    for name, parameter in unwrapped.backbone.named_parameters():
        if "norm" in name or "absolute_pos_embed" in name or "relative_position_bias_table" in name:
            backbone_no_decay.append(parameter)
        else:
            backbone_decay.append(parameter)
    groups: list[dict[str, Any]] = [
        {"params": backbone_no_decay, "weight_decay": 0.0},
        {"params": backbone_decay},
        {"params": [p for p in unwrapped.classifier.parameters() if p.requires_grad]},
    ]
    if hasattr(unwrapped, "text_encoder"):
        bert_parameters = [
            parameter
            for layer_index in range(10)
            for parameter in unwrapped.text_encoder.encoder.layer[layer_index].parameters()
            if parameter.requires_grad
        ]
        groups.append({"params": bert_parameters})
    else:
        raise ValueError("OCID-VLG primary training requires model=lavt_one")
    return groups


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "driver_allocated_memory"):
            return int(torch.mps.driver_allocated_memory())
        if hasattr(torch.mps, "current_allocated_memory"):
            return int(torch.mps.current_allocated_memory())
    return None


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    non_blocking = device.type == "cuda"
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if input_ids.ndim == 3 and input_ids.shape[1] == 1:
        input_ids = input_ids[:, 0]
    if attention_mask.ndim == 3 and attention_mask.shape[1] == 1:
        attention_mask = attention_mask[:, 0]
    return (
        batch["image"].to(device, non_blocking=non_blocking),
        batch["target_model_resolution"].to(device, non_blocking=non_blocking).long(),
        input_ids.to(device, non_blocking=non_blocking).long(),
        attention_mask.to(device, non_blocking=non_blocking).long(),
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    epoch: int,
    grad_accum_steps: int,
    print_freq: int,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    started = time.perf_counter()
    optimizer_steps = 0
    for batch_index, batch in enumerate(loader):
        image, target, input_ids, attention_mask = _move_batch(batch, device)
        logits = model(image, input_ids, l_mask=attention_mask)
        raw_loss = criterion(logits, target)
        if not torch.isfinite(raw_loss):
            raise FloatingPointError(
                f"non-finite training loss at epoch={epoch} batch={batch_index}: {raw_loss}"
            )
        accumulation_group_start = (
            batch_index // grad_accum_steps
        ) * grad_accum_steps
        accumulation_group_size = min(
            grad_accum_steps, len(loader) - accumulation_group_start
        )
        (raw_loss / accumulation_group_size).backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"non-finite gradient at epoch={epoch} batch={batch_index}"
                )
        should_step = (
            (batch_index + 1) % grad_accum_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
        losses.append(float(raw_loss.detach().cpu()))
        if batch_index % max(1, print_freq) == 0:
            LOGGER.info(
                "epoch=%d batch=%d/%d loss=%.6f lr=%.8g",
                epoch,
                batch_index + 1,
                len(loader),
                losses[-1],
                optimizer.param_groups[0]["lr"],
            )
    synchronize(device)
    return {
        "epoch": epoch,
        "train_loss": float(statistics.fmean(losses)),
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "peak_memory_bytes": peak_memory_bytes(device),
    }


def _original_target(batch: Mapping[str, Any], index: int) -> torch.Tensor:
    value = batch["target_original_resolution"]
    if isinstance(value, torch.Tensor):
        return value[index].detach().cpu().long()
    return torch.as_tensor(value[index]).long()


def _sample_field(batch: Mapping[str, Any], name: str, index: int) -> Any:
    value = batch[name]
    if isinstance(value, torch.Tensor):
        item = value[index].detach().cpu()
        return item.item() if item.ndim == 0 else item.tolist()
    return value[index]


def _prediction(
    logits: torch.Tensor,
    *,
    policy: str,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits, dim=1)
    foreground = probabilities[:, 1]
    if policy == "argmax":
        prediction = probabilities.argmax(dim=1)
    elif policy == "threshold":
        prediction = (foreground >= threshold).long()
    else:
        raise ValueError(f"unknown prediction policy: {policy}")
    return foreground, prediction


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    prediction_policy: str,
    threshold: float,
    collect_arrays: bool = False,
    sample_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    model_records: list[dict[str, Any]] = []
    original_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        image, _, input_ids, attention_mask = _move_batch(batch, device)
        synchronize(device)
        started = time.perf_counter()
        logits = model(image, input_ids, l_mask=attention_mask)
        synchronize(device)
        elapsed = time.perf_counter() - started
        per_sample_seconds = elapsed / image.shape[0]
        model_probability, model_prediction = _prediction(
            logits, policy=prediction_policy, threshold=threshold
        )
        for index in range(image.shape[0]):
            original_gt = _original_target(batch, index)
            original_size_value = _sample_field(batch, "original_size", index)
            original_size = tuple(int(value) for value in original_size_value)
            if tuple(original_gt.shape[-2:]) != original_size:
                raise ValueError(
                    f"original GT shape {tuple(original_gt.shape[-2:])} does not "
                    f"match RGB-derived size {original_size}"
                )
            original_logits = F.interpolate(
                logits[index : index + 1],
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            original_probability, original_prediction = _prediction(
                original_logits, policy=prediction_policy, threshold=threshold
            )
            model_gt = batch["target_model_resolution"][index].detach().cpu()
            model_metric = compute_sample_metrics(
                model_prediction[index].detach().cpu(),
                model_gt,
                foreground_probability=model_probability[index].detach().cpu(),
                inference_time_seconds=per_sample_seconds,
            )
            original_metric = compute_sample_metrics(
                original_prediction[0].detach().cpu(),
                original_gt,
                foreground_probability=original_probability[0].detach().cpu(),
                inference_time_seconds=per_sample_seconds,
            )
            model_records.append(model_metric)
            original_records.append(original_metric)
            row: dict[str, Any] = {
                "sent_id": str(_sample_field(batch, "sent_id", index)),
                "raw_question_index": _sample_field(batch, "raw_question_index", index)
                if "raw_question_index" in batch
                else None,
                "scene_id": str(_sample_field(batch, "scene_id", index)),
                "sentence": str(_sample_field(batch, "sentence", index)),
                "image_path": str(_sample_field(batch, "image_path", index)),
                "mask_path": str(_sample_field(batch, "mask_path", index)),
                "objID": _sample_field(batch, "objID", index)
                if "objID" in batch
                else None,
                "intersection": original_metric["intersection"],
                "union": original_metric["union"],
                "IoU": original_metric["iou"],
                "predicted_foreground_pixels": original_metric[
                    "predicted_foreground_pixels"
                ],
                "gt_foreground_pixels": original_metric["gt_foreground_pixels"],
                "inference_time_ms": per_sample_seconds * 1000.0,
                "prediction_status": "ok",
            }
            if collect_arrays:
                row["_probability_original"] = (
                    original_probability[0].detach().cpu().float().numpy()
                )
                row["_prediction_original"] = (
                    original_prediction[0].detach().cpu().byte().numpy()
                )
                row["_probability_model"] = (
                    model_probability[index].detach().cpu().float().numpy()
                )
            if sample_callback is not None:
                sample_callback(row)
                rows.append(
                    {key: value for key, value in row.items() if not key.startswith("_")}
                )
            else:
                rows.append(row)
    return aggregate_metrics(model_records), aggregate_metrics(original_records), rows


def save_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def save_yaml(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def save_metrics_tables(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    serializable = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    frame = pd.DataFrame(serializable)
    frame.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    frame.to_parquet(output_dir / "per_sample_metrics.parquet", index=False)


class PredictionExporter:
    """Write each prediction immediately so full-test RAM stays bounded."""

    def __init__(
        self,
        output_dir: Path,
        *,
        checkpoint_path: Path,
        model_name: str,
        backbone: str,
        prediction_policy: str,
        threshold: float,
    ) -> None:
        self.output_dir = output_dir
        self.prediction_root = output_dir / "predictions"
        self.prediction_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.backbone = backbone
        self.prediction_policy = prediction_policy
        self.threshold = threshold
        self.manifest_rows: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        sample_dir = self.prediction_root / row["sent_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        probability_original = np.asarray(row["_probability_original"], dtype=np.float32)
        probability_model = np.asarray(row["_probability_model"], dtype=np.float32)
        binary = np.asarray(row["_prediction_original"], dtype=np.uint8)
        if not np.isfinite(probability_original).all() or not np.isfinite(
            probability_model
        ).all():
            raise ValueError(f"non-finite probability for {row['sent_id']}")
        probability_path = sample_dir / "target_probability.npy"
        mask_path = sample_dir / "target_mask.png"
        model_probability_path = sample_dir / "predicted_probability_model_resolution.npy"
        np.save(probability_path, probability_original)
        np.save(model_probability_path, probability_model)
        Image.fromarray(binary * 255, mode="L").save(mask_path)
        metadata = {
            "sent_id": row["sent_id"],
            "scene_id": row["scene_id"],
            "raw_sentence": row["sentence"],
            "image_path": row["image_path"],
            "original_size": list(probability_original.shape),
            "model_size": list(probability_model.shape),
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "model_name": self.model_name,
            "backbone": self.backbone,
            "threshold": (
                self.threshold if self.prediction_policy == "threshold" else None
            ),
            "prediction_policy": self.prediction_policy,
            "per_sample_iou": row["IoU"],
            "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        save_json(sample_dir / "metadata.json", metadata)
        self.manifest_rows.append(
            {
                **metadata,
                "target_probability": str(probability_path.resolve()),
                "target_mask": str(mask_path.resolve()),
                "predicted_probability_model_resolution": str(
                    model_probability_path.resolve()
                ),
                "metadata": str((sample_dir / "metadata.json").resolve()),
            }
        )

    def finalize(self) -> Path:
        manifest_path = self.output_dir / "predictions_manifest.jsonl"
        save_jsonl(manifest_path, self.manifest_rows)
        return manifest_path


def export_predictions(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    checkpoint_path: Path,
    model_name: str,
    backbone: str,
    prediction_policy: str,
    threshold: float,
) -> Path:
    exporter = PredictionExporter(
        output_dir,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        backbone=backbone,
        prediction_policy=prediction_policy,
        threshold=threshold,
    )
    for row in rows:
        exporter.write(row)
    manifest_path = exporter.finalize()
    if len(exporter.manifest_rows) != len(rows):
        raise RuntimeError("prediction export count mismatch")
    return manifest_path


def verify_prediction_export(manifest_path: Path, expected_count: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sent_id = row["sent_id"]
        if sent_id in seen:
            errors.append(f"duplicate sent_id: {sent_id}")
        seen.add(sent_id)
        try:
            probability = np.load(row["target_probability"])
            model_probability = np.load(row["predicted_probability_model_resolution"])
            with Image.open(row["target_mask"]) as image:
                mask = np.asarray(image)
            expected_original_shape = tuple(int(value) for value in row["original_size"])
            expected_model_shape = tuple(int(value) for value in row["model_size"])
            if probability.shape != expected_original_shape:
                errors.append(
                    f"{sent_id}: original probability shape={probability.shape}, "
                    f"expected={expected_original_shape}"
                )
            if model_probability.shape != expected_model_shape:
                errors.append(
                    f"{sent_id}: model probability shape={model_probability.shape}, "
                    f"expected={expected_model_shape}"
                )
            if probability.dtype != np.float32 or model_probability.dtype != np.float32:
                errors.append(f"{sent_id}: probability dtype")
            if not np.isfinite(probability).all() or not np.isfinite(
                model_probability
            ).all():
                errors.append(f"{sent_id}: non-finite probability")
            if probability.min() < 0 or probability.max() > 1:
                errors.append(f"{sent_id}: probability outside [0,1]")
            if mask.shape != expected_original_shape:
                errors.append(
                    f"{sent_id}: mask shape={mask.shape}, "
                    f"expected={expected_original_shape}"
                )
            if not set(np.unique(mask)).issubset({0, 255}):
                errors.append(f"{sent_id}: non-binary PNG")
        except Exception as error:
            errors.append(f"{sent_id}: {type(error).__name__}: {error}")
    if len(rows) != expected_count:
        errors.append(f"manifest count {len(rows)} != expected {expected_count}")
    result = {
        "expected_count": expected_count,
        "actual_count": len(rows),
        "unique_sent_ids": len(seen),
        "errors": errors,
        "status": "ok" if not errors else "failed",
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result
