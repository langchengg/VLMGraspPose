from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import utils.config as config
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.dataset import OCIDVLGDataset

from failure_analysis.reranking.exporter import DEFAULT_CHECKPOINT, DEFAULT_CONFIG

from .aligned_crops import CROP_CHANNELS, build_aligned_crop
from .artifacts import ArtifactRun
from .extract import (
    REPO_ROOT,
    _device,
    _frozen_export_batch_size,
    _model_inputs,
    _postprocess_maps,
    _verify_forward_candidate_identity,
)
from .inference import load_local_model_artifact
from .models.rgbd_critic import RGBDGraspCritic
from .models.uncertainty import perturbation_geometries
from .schema import append_jsonl_record, atomic_write_json, read_jsonl, stable_sample_id


def _critic_models(
    paths: list[str | Path], device: torch.device
) -> list[tuple[RGBDGraspCritic, tuple[int, ...]]]:
    result = []
    for path in paths:
        artifact = load_local_model_artifact(path)
        channels = tuple(map(int, artifact["channel_indices"]))
        model = RGBDGraspCritic(
            int(artifact["input_channels"]),
            int(artifact.get("embedding_dim", 64)),
        )
        model.load_state_dict(artifact["model_state_dict"])
        model.to(device).eval()
        result.append((model, channels))
    if len(result) < 3:
        raise ValueError("stability requires at least three critic seeds")
    return result


@torch.no_grad()
def _score_crops(
    models: list[tuple[RGBDGraspCritic, tuple[int, ...]]],
    crops: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values = torch.from_numpy(np.asarray(crops, dtype=np.float32)).to(device)
    scores = []
    for model, channels in models:
        logits, _ = model(values[:, channels])
        scores.append(logits.float().cpu().numpy())
    return np.stack(scores)


def _sample_statistics(
    frozen: dict[str, Any],
    *,
    crop_inputs: dict[str, Any],
    critic_models: list[tuple[RGBDGraspCritic, tuple[int, ...]]],
    device: torch.device,
    crop_size: int,
    kappa: float,
) -> dict[str, Any]:
    crops = []
    candidate_slices = []
    definitions = None
    failure_reasons = []
    for candidate in frozen["candidates"]:
        start = len(crops)
        local_definitions = []
        for definition, geometry in perturbation_geometries(candidate):
            try:
                crop, _ = build_aligned_crop(
                    geometry,
                    output_size=crop_size,
                    **crop_inputs,
                )
                crops.append(crop)
                local_definitions.append(definition)
            except (ValueError, FloatingPointError) as error:
                failure_reasons.append(type(error).__name__)
        candidate_slices.append((start, len(crops)))
        definitions = definitions or local_definitions
    seed_scores = _score_crops(
        critic_models, np.stack(crops), device
    ) if crops else np.empty((len(critic_models), 0))
    rows = []
    for start, end in candidate_slices:
        values = seed_scores[:, start:end]
        if not values.size:
            rows.append(
                {
                    "mean": 0.0,
                    "standard_deviation": 0.0,
                    "minimum": 0.0,
                    "valid_fraction": 0.0,
                    "stability_penalty": 0.0,
                    "stable_score": 0.0,
                    "ensemble_disagreement": 0.0,
                }
            )
            continue
        mean = float(values.mean())
        standard_deviation = float(values.std())
        rows.append(
            {
                "mean": mean,
                "standard_deviation": standard_deviation,
                "minimum": float(values.min()),
                "valid_fraction": (end - start) / 17.0,
                "stability_penalty": float(kappa) * standard_deviation,
                "stable_score": mean
                - float(kappa) * standard_deviation,
                "ensemble_disagreement": float(
                    values[:, 0].std()
                ),
            }
        )
    return {
        "candidate_statistics": rows,
        "perturbation_definitions": definitions or [],
        "failed_perturbation_count": len(failure_reasons),
        "failure_reasons": sorted(set(failure_reasons)),
    }


@torch.no_grad()
def run_stability_extraction(
    *,
    split: str,
    frozen_features_path: str | Path,
    split_manifest_path: str | Path,
    critic_model_paths: list[str | Path],
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "auto",
    batch_size: int = 8,
    workers: int = 0,
    crop_size: int = 32,
    kappa: float = 1.0,
    max_samples: int | None = None,
    resume: bool = False,
    seed: int = 31,
) -> dict[str, Any]:
    frozen_features_path = Path(frozen_features_path).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = (REPO_ROOT / config_path).resolve()
    checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    critic_model_paths = [Path(path).resolve() for path in critic_model_paths]
    metadata_path = frozen_features_path.parent / "metadata.json"
    frozen_batch_size = _frozen_export_batch_size(frozen_features_path)
    if (
        frozen_batch_size is not None
        and int(batch_size) != frozen_batch_size
    ):
        raise ValueError(
            "stability replay batch size must match the frozen exporter "
            f"metadata: requested={batch_size}, frozen={frozen_batch_size}"
        )
    torch_device = _device(device)
    artifact_inputs = [frozen_features_path, config_path, *critic_model_paths]
    if metadata_path.exists():
        artifact_inputs.append(metadata_path)
    run = ArtifactRun(
        output_dir,
        kind="perturbation_ensemble_stability",
        repo_root=REPO_ROOT,
        config={
            "split": split,
            "batch_size": int(batch_size),
            "workers": int(workers),
            "crop_size": int(crop_size),
            "kappa": float(kappa),
            "perturbations_per_candidate": 17,
            "max_samples": max_samples,
        },
        inputs=tuple(artifact_inputs),
        split_manifest=split_manifest_path,
        checkpoint=checkpoint_path,
        evaluator_source=REPO_ROOT / "utils" / "grasp_metrics.py",
        seed=seed,
        device=str(torch_device),
        resume=resume,
    )
    manifest = run.prepare()
    if run.is_complete:
        return manifest
    cfg = config.load_cfg_from_cfg_file(str(config_path))
    dataset = OCIDVLGDataset(
        root_dir=str((REPO_ROOT / cfg.root_path).resolve()),
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        split=split,
        version=cfg.version,
    )
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    stats_path = output_dir / "statistics.jsonl"
    existing_records = (
        list(read_jsonl(stats_path)) if stats_path.exists() else []
    )
    completed = len(existing_records)
    manifest_completed = int(manifest.get("row_count", 0))
    if manifest_completed > completed:
        raise ValueError(
            "stability progress manifest is ahead of its durable statistics"
        )
    existing_ids = [str(record["sample_id"]) for record in existing_records]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("stability resume contains duplicate sample IDs")
    if completed != manifest_completed:
        manifest["row_count"] = completed
        atomic_write_json(run.manifest_path, manifest)
    loader = DataLoader(
        Subset(dataset, list(range(completed, total))),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        collate_fn=OCIDVLGDataset.collate_fn,
    )
    frozen_iterator = read_jsonl(frozen_features_path)
    for _ in range(completed):
        next(frozen_iterator)
    crog, _ = build_crog(cfg)
    crog.to(torch_device).eval()
    load_checkpoint(checkpoint_path, crog, torch_device, strict=True)
    critics = _critic_models(critic_model_paths, torch_device)
    mode = "a" if resume and stats_path.exists() else "x"
    all_ids = existing_ids.copy()
    failed_samples = sum(
        int(record.get("failed_perturbation_count", 0) > 0)
        for record in existing_records
    )
    with stats_path.open(mode, encoding="utf-8") as handle:
        for data in tqdm(loader, desc=f"V2 stability {split}", ncols=100):
            inputs = _model_inputs(data, torch_device)
            pred, _ = crog(*inputs)
            restored = _postprocess_maps(pred, inputs[0], data)
            frozen_batch = [
                next(frozen_iterator) for _ in range(len(restored))
            ]
            for batch_index, maps in enumerate(restored):
                frozen = frozen_batch[batch_index]
                local_id = int(data["sent_id"][batch_index])
                if local_id != int(frozen["sample_id"]):
                    raise ValueError("dataset/frozen artifact order mismatch")
                ins, quality, sin_map, cos_map, width_map = maps
                _verify_forward_candidate_identity(
                    frozen["candidates"],
                    quality,
                    sin_map,
                    cos_map,
                    width_map,
                )
                image_bgr = cv2.imread(
                    str(data["img_path"][batch_index]), cv2.IMREAD_COLOR
                )
                if image_bgr is None:
                    raise FileNotFoundError(data["img_path"][batch_index])
                result = _sample_statistics(
                    frozen,
                    crop_inputs={
                        "rgb": cv2.cvtColor(
                            image_bgr, cv2.COLOR_BGR2RGB
                        ),
                        "depth_m": data["depth"][batch_index]
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        "mask_probability": ins,
                        "quality": quality,
                        "sin_2theta": sin_map,
                        "cos_2theta": cos_map,
                        "width_probability": width_map,
                    },
                    critic_models=critics,
                    device=torch_device,
                    crop_size=crop_size,
                    kappa=kappa,
                )
                sample_id = stable_sample_id(split, local_id)
                append_jsonl_record(
                    handle,
                    {
                        "schema_version": "2.0.0",
                        "kind": "perturbation_stability_statistics",
                        "sample_id": sample_id,
                        "candidate_ids": [
                            candidate["candidate_id"]
                            for candidate in frozen["candidates"]
                        ],
                        **result,
                    },
                )
                all_ids.append(sample_id)
                completed += 1
                failed_samples += int(
                    result["failed_perturbation_count"] > 0
                )
            handle.flush()
            os.fsync(handle.fileno())
            manifest["row_count"] = completed
            manifest["failed_samples"] = failed_samples
            atomic_write_json(run.manifest_path, manifest)
    records = list(read_jsonl(stats_path))
    if len(records) != total:
        raise AssertionError(f"stability wrote {len(records)}/{total}")
    stable_scores = np.asarray(
        [
            [
                candidate["stable_score"]
                for candidate in record["candidate_statistics"]
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    npz_path = output_dir / "stability.npz"
    temporary = npz_path.with_name(f".{npz_path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(
            temporary,
            sample_ids=np.asarray(all_ids, dtype="U64"),
            stable_scores=stable_scores,
        )
        os.replace(temporary, npz_path)
    finally:
        temporary.unlink(missing_ok=True)
    return run.complete(
        outputs=(stats_path, npz_path),
        row_count=total,
        unique_ids=all_ids,
        extra={
            "failed_samples": failed_samples,
            "coverage": (total - failed_samples) / total,
            "candidate_count": total * 5,
            "critic_seed_count": len(critics),
            "crop_channels": CROP_CHANNELS,
        },
    )
