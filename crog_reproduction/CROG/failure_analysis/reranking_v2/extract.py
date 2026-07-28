from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import utils.config as config
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.dataset import OCIDVLGDataset
from utils.device import move_to_device
from utils.grasp_eval import detect_grasp_candidates

from failure_analysis.reranking.exporter import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    _restore_original,
)

from .aligned_crops import CROP_CHANNELS, build_aligned_crop
from .artifacts import ArtifactRun
from .latent_roi import CROGLatentCapture, pool_candidate_rois
from .schema import (
    append_jsonl_record,
    atomic_write_json,
    read_jsonl,
    stable_sample_id,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def _model_inputs(data, device):
    values = move_to_device(
        (
            data["img"],
            data["word_vec"],
            data["mask"],
            data["grasp_masks"]["qua"],
            data["grasp_masks"]["sin"],
            data["grasp_masks"]["cos"],
            data["grasp_masks"]["wid"],
        ),
        device,
    )
    image, text, mask, qua, sin, cos, wid = values
    return image, text, mask.unsqueeze(1), qua.unsqueeze(1), sin.unsqueeze(1), cos.unsqueeze(1), wid.unsqueeze(1)


def _postprocess_maps(pred, image, data):
    ins, quality, sin, cos, width = pred
    ins = torch.sigmoid(ins)
    quality = torch.sigmoid(quality)
    width = torch.sigmoid(width)
    if ins.shape[-2:] != image.shape[-2:]:
        kwargs = {
            "size": image.shape[-2:],
            "mode": "bicubic",
            "align_corners": True,
        }
        values = [
            F.interpolate(value, **kwargs).squeeze(1)
            for value in (ins, quality, sin, cos, width)
        ]
    else:
        values = [value.squeeze(1) for value in (ins, quality, sin, cos, width)]
    restored = []
    for batch_index in range(values[0].shape[0]):
        inverse = data["inverse"][batch_index]
        height, width_px = map(int, data["ori_size"][batch_index])
        restored.append(
            tuple(
                _restore_original(
                    value[batch_index].detach().cpu().numpy(),
                    inverse,
                    width_px,
                    height,
                )
                for value in values
            )
        )
    return restored


def _assert_hook_non_mutating(model, inputs, capture: CROGLatentCapture):
    with torch.no_grad():
        with capture.suspended():
            without_capture, _ = model(*inputs)
        with_capture, _ = model(*inputs)
    for left, right in zip(without_capture, with_capture, strict=True):
        if not torch.equal(left, right):
            maximum = float((left - right).abs().max().cpu())
            raise AssertionError(
                f"latent hook changed CROG output; max_abs_difference={maximum}"
            )
    capture.feature_maps()


def _verify_forward_candidate_identity(
    frozen: list[dict[str, Any]],
    quality: np.ndarray,
    sin_map: np.ndarray,
    cos_map: np.ndarray,
    width_map: np.ndarray,
    *,
    tolerance: float = 5e-4,
    value_tolerances: dict[str, float] | None = None,
    coordinate_tolerance: int = 0,
    sample_context: str = "unknown",
) -> dict[str, Any]:
    current, _ = detect_grasp_candidates(
        quality, sin_map, cos_map, width_map, num_grasps=5
    )
    if len(current) != len(frozen):
        raise AssertionError(
            f"{sample_context}: forward candidate count {len(current)} "
            f"!= frozen {len(frozen)}"
        )
    tolerances = value_tolerances or {
        "q_raw": float(tolerance),
        "angle_deg": float(tolerance),
        "width_px": float(tolerance),
    }
    maximum = 0.0
    maximum_by_field = {
        "q_raw": 0.0,
        "angle_deg": 0.0,
        "width_px": 0.0,
    }
    maximum_coordinate_difference = 0
    exact_coordinate_match = True
    for rank, (old, new) in enumerate(zip(frozen, current, strict=True)):
        old_coordinate = (int(old["row"]), int(old["col"]))
        new_coordinate = (int(new["row"]), int(new["col"]))
        coordinate_difference = max(
            abs(old_coordinate[0] - new_coordinate[0]),
            abs(old_coordinate[1] - new_coordinate[1]),
        )
        maximum_coordinate_difference = max(
            maximum_coordinate_difference, coordinate_difference
        )
        exact_coordinate_match &= coordinate_difference == 0
        if coordinate_difference > int(coordinate_tolerance):
            old_coordinates = [
                (int(item["row"]), int(item["col"])) for item in frozen
            ]
            new_coordinates = [
                (int(item["row"]), int(item["col"])) for item in current
            ]
            raise AssertionError(
                f"{sample_context}: forward peak coordinate/order differs at "
                f"rank {rank} beyond tolerance {coordinate_tolerance}; "
                f"frozen={old_coordinates}; current={new_coordinates}"
            )
        for name in ("q_raw", "angle_deg", "width_px"):
            difference = abs(float(old[name]) - float(new[name]))
            maximum = max(maximum, difference)
            maximum_by_field[name] = max(
                maximum_by_field[name], difference
            )
            if difference > float(tolerances[name]):
                raise AssertionError(
                    f"{sample_context}: forward {name} differs from frozen "
                    f"artifact by {difference} at rank {rank}; "
                    f"tolerance={tolerances[name]}"
                )
    return {
        "maximum_value_difference": maximum,
        "maximum_value_difference_by_field": maximum_by_field,
        "maximum_coordinate_difference": maximum_coordinate_difference,
        "exact_coordinate_match": bool(exact_coordinate_match),
        "candidate_count": len(current),
    }


def _frozen_export_device(frozen_features_path: Path) -> str | None:
    metadata_path = frozen_features_path.parent / "metadata.json"
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = payload.get("runtime", {}).get("device")
    return None if value is None else str(value)


def _frozen_export_batch_size(
    frozen_features_path: str | Path,
) -> int | None:
    metadata_path = Path(frozen_features_path).resolve().parent / "metadata.json"
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = payload.get("output_config", {}).get("batch_size")
    return None if value is None else int(value)


def _write_shard(
    output_path: Path,
    records: list[dict[str, Any]],
    crops: list[np.ndarray],
    latent_pre: list[np.ndarray],
    latent_post: list[np.ndarray],
) -> None:
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(
            temporary,
            crops=np.stack(crops).astype(np.float16),
            latent_pre=np.stack(latent_pre).astype(np.float16),
            latent_post=np.stack(latent_post).astype(np.float16),
            sample_ids=np.asarray(
                [record["sample_id"] for record in records], dtype="U64"
            ),
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


@torch.no_grad()
def extract_enhanced_features(
    *,
    split: str,
    frozen_features_path: str | Path,
    output_dir: str | Path,
    split_manifest_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "auto",
    batch_size: int = 8,
    workers: int = 0,
    crop_size: int = 32,
    roi_size: int = 5,
    shard_samples: int = 64,
    max_samples: int | None = None,
    resume: bool = False,
    seed: int = 19,
) -> dict[str, Any]:
    frozen_features_path = Path(frozen_features_path).resolve()
    output_dir = Path(output_dir).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()
    config_path = (REPO_ROOT / config_path).resolve()
    checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    cfg = config.load_cfg_from_cfg_file(str(config_path))
    root = (REPO_ROOT / cfg.root_path).resolve()
    torch_device = _device(device)
    frozen_device = _frozen_export_device(frozen_features_path)
    frozen_batch_size = _frozen_export_batch_size(frozen_features_path)
    if (
        frozen_batch_size is not None
        and int(batch_size) != frozen_batch_size
    ):
        raise ValueError(
            "enhanced replay batch size must match the frozen exporter "
            f"metadata: requested={batch_size}, frozen={frozen_batch_size}"
        )
    cross_device_replay = (
        frozen_device is not None
        and frozen_device != str(torch_device)
    )
    metadata_path = frozen_features_path.parent / "metadata.json"
    artifact_inputs = [frozen_features_path, config_path]
    if metadata_path.exists():
        artifact_inputs.append(metadata_path)
    run = ArtifactRun(
        output_dir,
        kind="enhanced_candidate_features",
        repo_root=REPO_ROOT,
        config={
            "split": split,
            "batch_size": int(batch_size),
            "workers": int(workers),
            "crop_size": int(crop_size),
            "roi_size": int(roi_size),
            "shard_samples": int(shard_samples),
            "max_samples": max_samples,
            "crop_channels": CROP_CHANNELS,
            "latent_layers": (
                "neck_pre_decoder_fq",
                "decoder_post_cross_attention_fq",
            ),
            "frozen_export_device": frozen_device,
            "cross_device_replay": cross_device_replay,
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
    dataset = OCIDVLGDataset(
        root_dir=str(root),
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        split=split,
        version=cfg.version,
    )
    total = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    index_path = output_dir / "index.jsonl"
    existing_records = (
        list(read_jsonl(index_path)) if index_path.exists() else []
    )
    completed = len(existing_records)
    manifest_completed = int(manifest.get("row_count", 0))
    if manifest_completed > completed:
        raise ValueError(
            "enhanced progress manifest is ahead of its durable index"
        )
    existing_ids = [str(record["sample_id"]) for record in existing_records]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("enhanced resume index contains duplicate sample IDs")
    if completed != manifest_completed:
        manifest["row_count"] = completed
        atomic_write_json(run.manifest_path, manifest)
    selected = list(range(completed, total))
    if not selected:
        if completed != total:
            raise ValueError("incomplete manifest has no remaining samples")
        shard_dir = output_dir / "shards"
        outputs = [index_path, *sorted(shard_dir.glob("shard_*.npz"))]
        return run.complete(
            outputs=outputs,
            row_count=completed,
            unique_ids=existing_ids,
            extra={
                "hook_non_mutating_checked": bool(
                    manifest.get("hook_non_mutating_checked", False)
                ),
                "forward_identity_max_difference": float(
                    manifest.get("forward_identity_max_difference", 0.0)
                ),
                "forward_identity_max_difference_by_field": manifest.get(
                    "forward_identity_max_difference_by_field",
                    {"q_raw": 0.0, "angle_deg": 0.0, "width_px": 0.0},
                ),
                "forward_identity_max_coordinate_difference": int(
                    manifest.get(
                        "forward_identity_max_coordinate_difference", 0
                    )
                ),
                "forward_identity_coordinate_mismatch_count": int(
                    manifest.get(
                        "forward_identity_coordinate_mismatch_count", 0
                    )
                ),
                "cross_device_replay": cross_device_replay,
                "candidate_count": completed * 5,
                "crop_channels": CROP_CHANNELS,
                "latent_shapes": {
                    "pre_decoder": [5, 1024],
                    "post_decoder": [5, 1024],
                },
            },
        )
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=False,
        collate_fn=OCIDVLGDataset.collate_fn,
    )
    frozen_iterator = read_jsonl(frozen_features_path)
    for _ in range(completed):
        next(frozen_iterator)
    model, _ = build_crog(cfg)
    model = model.to(torch_device).eval()
    load_checkpoint(checkpoint_path, model, torch_device, strict=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and index_path.exists() else "x"
    records: list[dict[str, Any]] = []
    crop_rows: list[np.ndarray] = []
    pre_rows: list[np.ndarray] = []
    post_rows: list[np.ndarray] = []
    all_ids = existing_ids.copy()
    forward_maximum = float(manifest.get("forward_identity_max_difference", 0.0))
    forward_maximum_by_field = {
        name: float(value)
        for name, value in manifest.get(
            "forward_identity_max_difference_by_field",
            {"q_raw": 0.0, "angle_deg": 0.0, "width_px": 0.0},
        ).items()
    }
    coordinate_maximum = int(
        manifest.get("forward_identity_max_coordinate_difference", 0)
    )
    coordinate_mismatch_count = int(
        manifest.get("forward_identity_coordinate_mismatch_count", 0)
    )
    hook_checked = bool(manifest.get("hook_non_mutating_checked", False))
    shard_index = len(list(shard_dir.glob("shard_*.npz")))
    with CROGLatentCapture(model) as capture, index_path.open(
        mode, encoding="utf-8"
    ) as index_handle:
        for data in tqdm(loader, desc=f"V2 enhanced {split}", ncols=100):
            inputs = _model_inputs(data, torch_device)
            pred, _ = model(*inputs)
            if not hook_checked:
                _assert_hook_non_mutating(model, inputs, capture)
                # Repopulate capture for the batch being written.
                pred, _ = model(*inputs)
                hook_checked = True
            pre_map, post_map = capture.feature_maps()
            restored = _postprocess_maps(pred, inputs[0], data)
            frozen_batch = [next(frozen_iterator) for _ in range(len(restored))]
            candidates_by_batch = [
                record["candidates"] for record in frozen_batch
            ]
            image_shapes = [
                tuple(map(int, data["ori_size"][index]))
                for index in range(len(restored))
            ]
            pooled_pre = (
                pool_candidate_rois(
                    pre_map,
                    candidates_by_batch,
                    image_shapes=image_shapes,
                    roi_size=roi_size,
                )
                .float()
                .cpu()
                .numpy()
            )
            pooled_post = (
                pool_candidate_rois(
                    post_map,
                    candidates_by_batch,
                    image_shapes=image_shapes,
                    roi_size=roi_size,
                )
                .float()
                .cpu()
                .numpy()
            )
            for batch_index, maps in enumerate(restored):
                frozen = frozen_batch[batch_index]
                local_id = int(data["sent_id"][batch_index])
                if local_id != int(frozen["sample_id"]):
                    raise ValueError("dataset/frozen artifact order mismatch")
                ins, quality, sin_map, cos_map, width_map = maps
                identity = _verify_forward_candidate_identity(
                    frozen["candidates"],
                    quality,
                    sin_map,
                    cos_map,
                    width_map,
                    value_tolerances=(
                        {
                            "q_raw": 2e-3,
                            "angle_deg": 0.2,
                            "width_px": 0.2,
                        }
                        if cross_device_replay
                        else None
                    ),
                    coordinate_tolerance=2 if cross_device_replay else 0,
                    sample_context=f"{split}:{local_id}",
                )
                forward_maximum = max(
                    forward_maximum, identity["maximum_value_difference"]
                )
                for name, value in identity[
                    "maximum_value_difference_by_field"
                ].items():
                    forward_maximum_by_field[name] = max(
                        forward_maximum_by_field[name], value
                    )
                coordinate_maximum = max(
                    coordinate_maximum,
                    identity["maximum_coordinate_difference"],
                )
                coordinate_mismatch_count += int(
                    not identity["exact_coordinate_match"]
                )
                image_bgr = cv2.imread(
                    str(data["img_path"][batch_index]), cv2.IMREAD_COLOR
                )
                if image_bgr is None:
                    raise FileNotFoundError(data["img_path"][batch_index])
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                depth_m = (
                    data["depth"][batch_index].cpu().numpy().astype(np.float32)
                )
                candidate_crops = []
                crop_metadata = []
                for candidate in frozen["candidates"]:
                    crop, metadata = build_aligned_crop(
                        candidate,
                        rgb=image_rgb,
                        depth_m=depth_m,
                        mask_probability=ins,
                        quality=quality,
                        sin_2theta=sin_map,
                        cos_2theta=cos_map,
                        width_probability=width_map,
                        output_size=crop_size,
                    )
                    candidate_crops.append(crop)
                    crop_metadata.append(metadata)
                sample_id = stable_sample_id(split, local_id)
                record = {
                    "schema_version": "2.0.0",
                    "kind": "enhanced_candidate_index",
                    "sample_id": sample_id,
                    "source_sample_id": local_id,
                    "split": split,
                    "frame_id": frozen["scene_id"],
                    "candidate_ids": [
                        item["candidate_id"] for item in frozen["candidates"]
                    ],
                    "candidate_checksums": [
                        item["candidate_checksum"] for item in frozen["candidates"]
                    ],
                    "shard": f"shard_{shard_index:05d}.npz",
                    "offset": len(records),
                    "crop_metadata": crop_metadata,
                }
                records.append(record)
                crop_rows.append(np.stack(candidate_crops))
                pre_rows.append(pooled_pre[batch_index])
                post_rows.append(pooled_post[batch_index])
                all_ids.append(sample_id)
                if len(records) >= int(shard_samples):
                    shard_path = shard_dir / f"shard_{shard_index:05d}.npz"
                    _write_shard(
                        shard_path, records, crop_rows, pre_rows, post_rows
                    )
                    for item in records:
                        append_jsonl_record(index_handle, item)
                    index_handle.flush()
                    os.fsync(index_handle.fileno())
                    completed += len(records)
                    manifest["row_count"] = completed
                    manifest["forward_identity_max_difference"] = forward_maximum
                    manifest[
                        "forward_identity_max_difference_by_field"
                    ] = forward_maximum_by_field
                    manifest[
                        "forward_identity_max_coordinate_difference"
                    ] = coordinate_maximum
                    manifest[
                        "forward_identity_coordinate_mismatch_count"
                    ] = coordinate_mismatch_count
                    manifest["hook_non_mutating_checked"] = hook_checked
                    # A running manifest is a resumable progress checkpoint.
                    atomic_write_json(run.manifest_path, manifest)
                    shard_index += 1
                    records, crop_rows, pre_rows, post_rows = [], [], [], []
        if records:
            shard_path = shard_dir / f"shard_{shard_index:05d}.npz"
            _write_shard(shard_path, records, crop_rows, pre_rows, post_rows)
            for item in records:
                append_jsonl_record(index_handle, item)
            index_handle.flush()
            os.fsync(index_handle.fileno())
            completed += len(records)
            manifest["row_count"] = completed
            manifest["forward_identity_max_difference"] = forward_maximum
            manifest[
                "forward_identity_max_difference_by_field"
            ] = forward_maximum_by_field
            manifest[
                "forward_identity_max_coordinate_difference"
            ] = coordinate_maximum
            manifest[
                "forward_identity_coordinate_mismatch_count"
            ] = coordinate_mismatch_count
            manifest["hook_non_mutating_checked"] = hook_checked
            atomic_write_json(run.manifest_path, manifest)
    if completed != total:
        raise AssertionError(f"enhanced extraction wrote {completed}/{total}")
    outputs = [index_path, *sorted(shard_dir.glob("shard_*.npz"))]
    return run.complete(
        outputs=outputs,
        row_count=completed,
        unique_ids=all_ids,
        extra={
            "hook_non_mutating_checked": hook_checked,
            "forward_identity_max_difference": forward_maximum,
            "forward_identity_max_difference_by_field": (
                forward_maximum_by_field
            ),
            "forward_identity_max_coordinate_difference": coordinate_maximum,
            "forward_identity_coordinate_mismatch_count": (
                coordinate_mismatch_count
            ),
            "cross_device_replay": cross_device_replay,
            "candidate_count": completed * 5,
            "crop_channels": CROP_CHANNELS,
            "latent_shapes": {
                "pre_decoder": [5, 1024],
                "post_decoder": [5, 1024],
            },
        },
    )
