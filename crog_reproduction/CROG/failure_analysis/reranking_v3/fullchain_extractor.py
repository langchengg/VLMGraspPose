from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from tqdm import tqdm

import utils.config as config
from failure_analysis.reranking.exporter import DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from failure_analysis.reranking_v2.extract import (
    _frozen_export_batch_size,
    _postprocess_maps,
    _verify_forward_candidate_identity,
)
from failure_analysis.reranking_v2.schema import atomic_write_json as mutable_atomic_write_json
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.dataset import tokenize

from .aligned_crops import CROP_CHANNELS, build_fullchain_crop
from .artifacts import code_fingerprint
from .coordinate_mapping import forward_from_inverse
from .crog_hooks import FullChainCapture, assert_capture_non_mutating
from .depth_geometry import DEPTH_FEATURE_NAMES, depth_geometry_features
from .latent_roi import POOL_REGIONS, pool_attention_features, pool_fullchain_features
from .output_map_features import extract_head_features
from .schema import artifact_identity, canonical_json, read_jsonl, sha256_bytes, sha256_file, stable_sample_id


REPO_ROOT = Path(__file__).resolve().parents[2]
MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
FLOAT16_STORAGE_FIELDS = (
    "latent_rois", "attention_rois", "tokens", "sentence", "dynamic", "crops",
)
_MILLIMETRE_UNITS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}
_METRE_UNITS = {"m", "meter", "meters", "metre", "metres"}


def _depth_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for container_name in ("metadata", "depth_metadata"):
        container = record.get(container_name)
        if isinstance(container, Mapping):
            for key in ("depth_unit", "depth_units", "depth_scale_to_m"):
                if key in container:
                    result[key] = container[key]
    for key in ("depth_unit", "depth_units", "depth_scale_to_m"):
        if key in record:
            result[key] = record[key]
    return result


def _missing_depth(
    image_shape: tuple[int, int], *, reason: str, path: str | None,
    raw_dtype: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return np.zeros(image_shape, dtype=np.float32), {
        "available": False,
        "reason": str(reason),
        "path": path,
        "raw_dtype": raw_dtype,
        "unit": None,
        "scale_to_m": None,
        "valid_fraction": 0.0,
    }


def _normalise_depth_array(
    depth_raw: np.ndarray,
    *,
    image_shape: tuple[int, int],
    metadata: Mapping[str, Any] | None = None,
    path: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Conservatively decode a depth array to metres or return missing evidence."""
    raw = np.asarray(depth_raw)
    raw_dtype = str(raw.dtype)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 2 or tuple(raw.shape) != tuple(image_shape):
        return _missing_depth(
            image_shape, reason="shape_mismatch", path=path, raw_dtype=raw_dtype,
        )
    values = raw.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        return _missing_depth(
            image_shape, reason="nonfinite_values", path=path, raw_dtype=raw_dtype,
        )
    if np.any(values < 0):
        return _missing_depth(
            image_shape, reason="negative_values", path=path, raw_dtype=raw_dtype,
        )
    depth_metadata = {} if metadata is None else dict(metadata)
    raw_unit = depth_metadata.get("depth_unit", depth_metadata.get("depth_units"))
    unit = None if raw_unit is None else str(raw_unit).strip().lower()
    if unit in _MILLIMETRE_UNITS:
        unit_scale = 0.001
    elif unit in _METRE_UNITS:
        unit_scale = 1.0
    elif unit:
        return _missing_depth(
            image_shape, reason="unsupported_unit_metadata", path=path,
            raw_dtype=raw_dtype,
        )
    else:
        unit_scale = None
    raw_scale = depth_metadata.get("depth_scale_to_m")
    if raw_scale is not None:
        try:
            explicit_scale = float(raw_scale)
        except (TypeError, ValueError):
            explicit_scale = float("nan")
        if not np.isfinite(explicit_scale) or explicit_scale <= 0.0:
            return _missing_depth(
                image_shape, reason="invalid_scale_metadata", path=path,
                raw_dtype=raw_dtype,
            )
        if unit_scale is not None and not np.isclose(explicit_scale, unit_scale):
            return _missing_depth(
                image_shape, reason="conflicting_unit_metadata", path=path,
                raw_dtype=raw_dtype,
            )
        scale = explicit_scale
        scale_source = "metadata_scale"
    elif unit_scale is not None:
        scale = unit_scale
        scale_source = "metadata_unit"
    elif raw.dtype == np.uint16:
        scale = 0.001
        scale_source = "known_uint16_millimetres"
    elif np.issubdtype(raw.dtype, np.floating):
        scale = 1.0
        scale_source = "known_float_metres"
    else:
        return _missing_depth(
            image_shape, reason="unsupported_dtype", path=path, raw_dtype=raw_dtype,
        )
    depth_m64 = values * scale
    positive = depth_m64 > 0.0
    if not positive.any():
        return _missing_depth(
            image_shape, reason="no_positive_values", path=path, raw_dtype=raw_dtype,
        )
    positive_values = depth_m64[positive]
    if float(positive_values.max()) > 100.0:
        return _missing_depth(
            image_shape, reason="implausible_metric_range", path=path,
            raw_dtype=raw_dtype,
        )
    depth_m = depth_m64.astype(np.float32)
    return depth_m, {
        "available": True,
        "reason": None,
        "path": path,
        "raw_dtype": raw_dtype,
        "unit": "m",
        "scale_to_m": float(scale),
        "scale_source": scale_source,
        "valid_fraction": float(positive.mean()),
        "positive_min_m": float(positive_values.min()),
        "positive_max_m": float(positive_values.max()),
    }


def _load_depth_m(
    record: Mapping[str, Any], *, image_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_path = record.get("depth_path")
    if raw_path is None or not str(raw_path).strip():
        return _missing_depth(image_shape, reason="depth_path_missing", path=None)
    path = str(Path(str(raw_path)).expanduser().resolve())
    if not Path(path).is_file():
        return _missing_depth(image_shape, reason="depth_file_missing", path=path)
    try:
        depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except (cv2.error, OSError):
        depth_raw = None
    if depth_raw is None:
        return _missing_depth(image_shape, reason="depth_unreadable", path=path)
    return _normalise_depth_array(
        depth_raw,
        image_shape=image_shape,
        metadata=_depth_metadata(record),
        path=path,
    )


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    result = torch.device(requested)
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return result


def _affine(image_shape: tuple[int, int], input_size: int) -> tuple[np.ndarray, np.ndarray]:
    ori_h, ori_w = image_shape
    scale = min(input_size / ori_h, input_size / ori_w)
    new_h, new_w = ori_h * scale, ori_w * scale
    bias_x, bias_y = (input_size-new_w)/2.0, (input_size-new_h)/2.0
    src = np.asarray([[0,0],[ori_w,0],[0,ori_h]], np.float32)
    dst = np.asarray([[bias_x,bias_y],[new_w+bias_x,bias_y],[bias_x,new_h+bias_y]], np.float32)
    forward = cv2.getAffineTransform(src,dst); inverse = cv2.getAffineTransform(dst,src)
    return forward, inverse


def preprocess_inference_record(record: dict[str, Any], *, input_size: int, word_length: int) -> dict[str, Any]:
    image_bgr = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(record["image_path"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    depth_m, depth_status = _load_depth_m(record, image_shape=image_rgb.shape[:2])
    forward, inverse = _affine(image_rgb.shape[:2], input_size)
    warped = cv2.warpAffine(
        image_rgb, forward, (input_size,input_size), flags=cv2.INTER_CUBIC,
        borderValue=tuple((MEAN[:,0,0]*255).tolist()),
    )
    normalized = (warped.transpose(2,0,1).astype(np.float32)/255.0-MEAN)/STD
    token_ids = tokenize(str(record["language_instruction"]), word_length, True).squeeze(0)
    return {
        "record": record,
        "image": torch.from_numpy(normalized),
        "rgb": torch.from_numpy(image_rgb.transpose(2,0,1).astype(np.float32)/255.0),
        "depth": torch.from_numpy(depth_m[None]),
        "depth_status": depth_status,
        "token_ids": token_ids.long(),
        "forward": forward,
        "inverse": inverse,
        "ori_size": tuple(map(int,image_rgb.shape[:2])),
    }


def _preserved_batch_slices(
    *,
    record_count: int,
    batch_size: int,
    skip: int,
    max_samples: int | None,
    shard_id: int,
    num_shards: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Plan frozen batches without changing their membership or order.

    The returned tuple is ``(source_start, source_stop, emit_start, emit_stop)``.
    The model must see the complete source interval; only the local emit interval is
    written. Sharding therefore happens over frozen batch indices, never samples.
    """
    selected_seen = 0
    for source_start in range(0, int(record_count), int(batch_size)):
        batch_index = source_start // int(batch_size)
        source_stop = min(source_start + int(batch_size), int(record_count))
        if batch_index % int(num_shards) != int(shard_id):
            continue
        group_size = source_stop - source_start
        selected_start = selected_seen
        selected_stop = selected_start + group_size
        selected_seen = selected_stop
        emit_start = max(int(skip) - selected_start, 0)
        emit_stop = group_size
        if max_samples is not None:
            emit_stop = min(emit_stop, int(max_samples) - selected_start)
        emit_start = min(max(emit_start, 0), group_size)
        emit_stop = min(max(emit_stop, 0), group_size)
        if emit_start < emit_stop:
            yield source_start, source_stop, emit_start, emit_stop
        if max_samples is not None and selected_stop >= int(max_samples):
            break


def _batches(
    path: Path,
    *,
    record_count: int,
    batch_size: int,
    input_size: int,
    word_length: int,
    skip: int,
    max_samples: int | None,
    shard_id: int,
    num_shards: int,
    allowed_ids: set[str] | None = None,
) -> Iterator[tuple[list[dict[str, Any]], range]]:
    selected_seen = 0
    raw_batch: list[dict[str, Any]] = []
    for global_index, record in enumerate(read_jsonl(path)):
        raw_batch.append(record)
        is_last = global_index + 1 == int(record_count)
        if len(raw_batch) != int(batch_size) and not is_last:
            continue
        batch_index = (global_index + 1 - len(raw_batch)) // int(batch_size)
        if batch_index % int(num_shards) == int(shard_id):
            local_indices = [
                index for index, value in enumerate(raw_batch)
                if allowed_ids is None or stable_sample_id(value["split"], value["sample_id"]) in allowed_ids
            ]
            selected_start = selected_seen
            selected_seen += len(local_indices)
            local_indices = [
                value for offset, value in enumerate(local_indices, start=selected_start)
                if offset >= int(skip) and (max_samples is None or offset < int(max_samples))
            ]
            if local_indices:
                batch = [
                    preprocess_inference_record(
                        value, input_size=input_size, word_length=word_length
                    )
                    for value in raw_batch
                ]
                yield batch, range(local_indices[0], local_indices[-1] + 1) if local_indices == list(range(local_indices[0], local_indices[-1] + 1)) else local_indices
        raw_batch = []
        if max_samples is not None and selected_seen >= int(max_samples):
            break


def _model_inputs(batch: list[dict[str, Any]], device: torch.device):
    images = torch.stack([item["image"] for item in batch]).to(device)
    tokens = torch.stack([item["token_ids"] for item in batch]).to(device)
    return (images, tokens, None, None, None, None, None)


def _write_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_index(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record)+"\n")
        handle.flush(); os.fsync(handle.fileno())


def _materialize_fullchain_shard(
    pending_records: list[dict[str, Any]],
    pending_arrays: dict[str, list[np.ndarray]],
    *,
    precast_reference: bool,
) -> dict[str, np.ndarray]:
    """Materialize one cache shard directly from the original float32 arrays.

    The diagnostic reference calls this before the persisted-cache call.  Thus
    its float32 values are captured before the first float16 cast rather than
    reconstructed by round-tripping the cache.
    """
    float_storage_dtype = np.float32 if precast_reference else np.float16
    result = {
        "sample_ids": np.asarray([value["sample_id"] for value in pending_records], dtype="U64"),
        "head_features": np.stack(pending_arrays["head_features"]).astype(np.float32),
        "depth_features": np.stack(pending_arrays["depth_features"]).astype(np.float32),
        "token_ids": np.stack(pending_arrays["token_ids"]),
    }
    for name in FLOAT16_STORAGE_FIELDS:
        result[name] = np.stack(pending_arrays[name]).astype(float_storage_dtype)
    return result


@torch.no_grad()
def extract_fullchain_features(
    *,
    frozen_features_path: str | Path,
    split_manifest_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "auto",
    batch_size: int = 16,
    crop_size: int = 32,
    roi_size: int = 7,
    channel_bins: int = 32,
    shard_samples: int = 64,
    max_samples: int | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
    seed: int = 20260801,
    resume: bool = False,
    allowed_ids: set[str] | None = None,
    precast_reference_dir: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(frozen_features_path).resolve(); split_manifest = Path(split_manifest_path).resolve()
    output = Path(output_dir).resolve(); cfg_path = (REPO_ROOT/config_path).resolve(); checkpoint = (REPO_ROOT/checkpoint_path).resolve()
    precast_reference = None if precast_reference_dir is None else Path(precast_reference_dir).resolve()
    if precast_reference is not None:
        if resume:
            raise ValueError("pre-cast diagnostic capture is intentionally non-resumable")
        if precast_reference == output:
            raise ValueError("pre-cast diagnostic reference and float16 cache directories must differ")
        if precast_reference.exists():
            raise FileExistsError(precast_reference)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard-id must be in [0,num-shards)")
    frozen_batch_size = _frozen_export_batch_size(source)
    if frozen_batch_size is None:
        raise FileNotFoundError(
            f"frozen exporter metadata with output_config.batch_size is required: {source.parent / 'metadata.json'}"
        )
    if int(batch_size) != int(frozen_batch_size):
        raise ValueError(
            f"batch-size {batch_size} would change frozen forward semantics; "
            f"the source artifact requires {frozen_batch_size}"
        )
    record_count = sum(1 for _ in read_jsonl(source))
    cfg = config.load_cfg_from_cfg_file(str(cfg_path)); torch_device = _device(device)
    fingerprint_payload = {
        "source": artifact_identity(source), "split_manifest": artifact_identity(split_manifest),
        "config": artifact_identity(cfg_path), "checkpoint": artifact_identity(checkpoint),
        "code": code_fingerprint(), "batch_size": batch_size, "crop_size": crop_size,
        "roi_size": roi_size, "channel_bins": channel_bins, "max_samples": max_samples,
        "shard_id": shard_id, "num_shards": num_shards, "seed": seed,
        "record_count": record_count,
        "allowed_ids_sha256": None if allowed_ids is None else sha256_bytes("\n".join(sorted(allowed_ids)).encode()),
        "allowed_id_count": None if allowed_ids is None else len(allowed_ids),
        "batching_strategy": "frozen_contiguous_batches_sharded_by_batch_index_v1",
        "precast_reference_capture": precast_reference is not None,
    }
    fingerprint = sha256_bytes(canonical_json(fingerprint_payload).encode())
    manifest_path = output/"artifact_manifest.json"; index_path = output/"index.jsonl"; schema_path = output/"feature_schema.json"
    if output.exists() and not resume:
        raise FileExistsError(f"full-chain output exists; pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True); (output/"shards").mkdir(exist_ok=True)
    reference_manifest_path = None
    reference_index_path = None
    reference_schema_path = None
    reference_manifest: dict[str, Any] | None = None
    if precast_reference is not None:
        precast_reference.mkdir(parents=True, exist_ok=False)
        (precast_reference/"shards").mkdir()
        reference_manifest_path = precast_reference/"artifact_manifest.json"
        reference_index_path = precast_reference/"index.jsonl"
        reference_schema_path = precast_reference/"feature_schema.json"
        reference_manifest = {
            "schema_version":"3.0.0",
            "artifact_type":"fullchain_candidate_features_precast_float32_reference",
            "status":"running",
            "fingerprint":fingerprint,
            "created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "row_count":0,
            "candidate_count":0,
            "missing_count":0,
            "fallback_count":0,
            "missing_depth_reason_counts":{},
            "labels_read":False,
            "capture_stage":"materialized directly from extractor pending float32 arrays before the first float16 cast",
            "float16_storage_fields":list(FLOAT16_STORAGE_FIELDS),
            **fingerprint_payload,
        }
        mutable_atomic_write_json(reference_manifest_path, reference_manifest)
    existing = list(read_jsonl(index_path)) if index_path.exists() else []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("full-chain resume fingerprint mismatch")
        if manifest.get("status") == "complete":
            for identity in manifest["outputs"]:
                if sha256_file(identity["path"]) != identity["sha256"]:
                    raise ValueError("completed full-chain output changed")
            return manifest
    else:
        manifest = {
            "schema_version":"3.0.0", "artifact_type":"fullchain_candidate_features",
            "status":"running", "fingerprint":fingerprint, "created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "row_count":0, "candidate_count":0, "missing_count":0, "fallback_count":0,
            "labels_read":False,
            **fingerprint_payload,
        }
        mutable_atomic_write_json(manifest_path,manifest)
    completed = len(existing)
    if completed != int(manifest.get("row_count",0)):
        raise ValueError("manifest/index progress mismatch")
    missing_depth_count = int(manifest.get("missing_count", 0))
    missing_depth_reason_counts = {
        str(key): int(value)
        for key, value in manifest.get("missing_depth_reason_counts", {}).items()
    }
    model,_ = build_crog(cfg); model=model.to(torch_device).eval(); load_checkpoint(checkpoint,model,torch_device,strict=True)
    layer_names = None; feature_names = None; hook_differences = manifest.get("hook_max_abs_difference")
    pending_records=[]; pending_arrays={name:[] for name in ("head_features","depth_features","latent_rois","attention_rois","tokens","sentence","dynamic","token_ids","crops")}
    shard_number=len(list((output/"shards").glob("shard_*.npz"))); all_ids=[str(item["sample_id"]) for item in existing]
    expected_total = max_samples
    progress = tqdm(total=expected_total, initial=completed, desc="V3 full-chain", ncols=100)
    with FullChainCapture(model) as capture:
        for batch, emit_indices in _batches(
            source,
            record_count=record_count,
            batch_size=batch_size,
            input_size=cfg.input_size,
            word_length=cfg.word_len,
            skip=completed,
            max_samples=max_samples,
            shard_id=shard_id,
            num_shards=num_shards,
            allowed_ids=allowed_ids,
        ):
            inputs=_model_inputs(batch,torch_device); capture.clear(); pred,_=model(*inputs)
            if hook_differences is None:
                hook_differences=assert_capture_non_mutating(model,inputs,capture)
                pred,_=model(*inputs)
            maps=capture.feature_maps(); attentions=capture.attention_maps()
            candidates=[item["record"]["candidates"] for item in batch]
            affines=[item["forward"] for item in batch]
            layer_names,latent=pool_fullchain_features(maps,candidates,affines,roi_size=roi_size,channel_bins=channel_bins)
            attention=pool_attention_features(attentions,candidates,affines,roi_size=roi_size)
            restored=_postprocess_maps(pred,inputs[0],{"inverse":[item["inverse"] for item in batch],"ori_size":[item["ori_size"] for item in batch]})
            tokens=capture.tokens(); dynamic=capture.dynamic()
            # sentence state is exactly the EOT row projected by the frozen matrix.
            eot=inputs[1].argmax(dim=-1); projection=capture.base.backbone.text_projection
            sentence=tokens[torch.arange(len(batch),device=tokens.device),eot]@projection
            raw_heads=capture.raw_heads()
            for batch_index in emit_indices:
                item = batch[batch_index]
                record=item["record"]; stable_id=stable_sample_id(record["split"],record["sample_id"])
                ins,q_act,sin_map,cos_map,w_act=restored[batch_index]
                identity=_verify_forward_candidate_identity(record["candidates"],q_act,sin_map,cos_map,w_act,sample_context=stable_id)
                sample_crops=[]; sample_head=[]; sample_depth=[]; crop_metadata=[]
                rgb=item["rgb"].unsqueeze(0).to(torch_device); depth=item["depth"].unsqueeze(0).to(torch_device)
                heads=tuple(value[batch_index:batch_index+1] for value in raw_heads)
                for candidate in record["candidates"]:
                    crop,metadata=build_fullchain_crop(candidate,rgb=rgb,depth_m=depth,raw_heads=heads,forward_affine=item["forward"],output_size=crop_size)
                    metadata["depth_source_available"] = bool(item["depth_status"]["available"])
                    metadata["depth_fallback_reason"] = item["depth_status"]["reason"]
                    names,head_values=extract_head_features(crop,candidate,record["candidates"],image_shape=item["ori_size"])
                    if feature_names is None: feature_names=names
                    elif feature_names!=names: raise AssertionError("head feature schema changed between samples")
                    sample_crops.append(crop); sample_head.append(head_values); sample_depth.append(depth_geometry_features(crop)); crop_metadata.append(metadata)
                index_record={
                    "schema_version":"3.0.0","kind":"fullchain_candidate_index","sample_id":stable_id,
                    "source_sample_id":int(record["sample_id"]),"split":record["split"],"frame_id":record["scene_id"],
                    "language_instruction":record["language_instruction"],
                    "candidate_ids":[value["candidate_id"] for value in record["candidates"]],
                    "candidate_checksums":[value["candidate_checksum"] for value in record["candidates"]],
                    "forward_affine":item["forward"].tolist(),"inverse_affine":item["inverse"].tolist(),
                    "original_size":list(item["ori_size"]),"model_input_size":[cfg.input_size,cfg.input_size],
                    "depth_source":dict(item["depth_status"]),
                    "shard":f"shard_{shard_number:05d}.npz","offset":len(pending_records),
                    "candidate_identity":identity,"crop_metadata":crop_metadata,
                }
                if not item["depth_status"]["available"]:
                    missing_depth_count += 1
                    reason = str(item["depth_status"]["reason"])
                    missing_depth_reason_counts[reason] = missing_depth_reason_counts.get(reason, 0) + 1
                pending_records.append(index_record); all_ids.append(stable_id)
                for key,value in (
                    ("head_features",np.stack(sample_head)),("depth_features",np.stack(sample_depth)),
                    ("latent_rois",latent[batch_index].float().cpu().numpy()),("attention_rois",attention[batch_index].float().cpu().numpy()),
                    ("tokens",tokens[batch_index].float().cpu().numpy()),("sentence",sentence[batch_index].float().cpu().numpy()),
                    ("dynamic",dynamic[batch_index].float().cpu().numpy()),("token_ids",item["token_ids"].numpy()),
                    ("crops",np.stack(sample_crops)),
                ): pending_arrays[key].append(value)
                if len(pending_records)>=shard_samples:
                    shard_path=output/"shards"/f"shard_{shard_number:05d}.npz"
                    if precast_reference is not None:
                        assert reference_index_path is not None and reference_manifest_path is not None and reference_manifest is not None
                        _write_npz(
                            precast_reference/"shards"/shard_path.name,
                            **_materialize_fullchain_shard(pending_records, pending_arrays, precast_reference=True),
                        )
                        _append_index(reference_index_path,pending_records)
                        reference_manifest.update({
                            "row_count":completed+len(pending_records),
                            "candidate_count":(completed+len(pending_records))*5,
                            "missing_count":missing_depth_count,
                            "fallback_count":missing_depth_count,
                            "missing_depth_reason_counts":missing_depth_reason_counts,
                        })
                        mutable_atomic_write_json(reference_manifest_path,reference_manifest)
                    _write_npz(
                        shard_path,
                        **_materialize_fullchain_shard(pending_records, pending_arrays, precast_reference=False),
                    )
                    _append_index(index_path,pending_records); completed+=len(pending_records); progress.update(len(pending_records))
                    manifest.update({
                        "row_count":completed,"candidate_count":completed*5,
                        "missing_count":missing_depth_count,"fallback_count":missing_depth_count,
                        "missing_depth_reason_counts":missing_depth_reason_counts,
                        "hook_max_abs_difference":hook_differences,
                    })
                    mutable_atomic_write_json(manifest_path,manifest); shard_number+=1; pending_records=[]; pending_arrays={name:[] for name in pending_arrays}
            del latent,attention,tokens,dynamic,sentence
        if pending_records:
            shard_path=output/"shards"/f"shard_{shard_number:05d}.npz"
            if precast_reference is not None:
                assert reference_index_path is not None and reference_manifest_path is not None and reference_manifest is not None
                _write_npz(
                    precast_reference/"shards"/shard_path.name,
                    **_materialize_fullchain_shard(pending_records, pending_arrays, precast_reference=True),
                )
                _append_index(reference_index_path,pending_records)
                reference_manifest.update({
                    "row_count":completed+len(pending_records),
                    "candidate_count":(completed+len(pending_records))*5,
                    "missing_count":missing_depth_count,
                    "fallback_count":missing_depth_count,
                    "missing_depth_reason_counts":missing_depth_reason_counts,
                })
                mutable_atomic_write_json(reference_manifest_path,reference_manifest)
            _write_npz(
                shard_path,
                **_materialize_fullchain_shard(pending_records, pending_arrays, precast_reference=False),
            )
            _append_index(index_path,pending_records); completed+=len(pending_records); progress.update(len(pending_records))
    progress.close()
    if max_samples is not None and completed!=int(max_samples):
        raise AssertionError(f"full-chain extraction wrote {completed}/{max_samples}")
    if len(all_ids)!=len(set(all_ids)):
        raise AssertionError("full-chain extraction duplicated sample IDs")
    schema={
        "head_feature_names":feature_names,"depth_feature_names":list(DEPTH_FEATURE_NAMES),
        "latent_layer_names":layer_names,"latent_pool_regions":list(POOL_REGIONS),"latent_channel_bins":channel_bins,
        "crop_channels":list(CROP_CHANNELS),"token_length":cfg.word_len,
    }
    if not schema_path.exists(): mutable_atomic_write_json(schema_path,schema)
    elif json.loads(schema_path.read_text())!=schema: raise ValueError("feature schema changed on resume")
    if precast_reference is not None:
        assert reference_schema_path is not None and reference_manifest_path is not None and reference_manifest is not None
        mutable_atomic_write_json(reference_schema_path,schema)
        reference_output_paths=[
            reference_index_path, reference_schema_path,
            *sorted((precast_reference/"shards").glob("shard_*.npz")),
        ]
        reference_manifest.update({
            "status":"complete",
            "completed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "row_count":completed,
            "candidate_count":completed*5,
            "unique_sample_count":len(set(all_ids)),
            "unique_candidate_count":completed*5,
            "missing_count":missing_depth_count,
            "fallback_count":missing_depth_count,
            "missing_depth_reason_counts":missing_depth_reason_counts,
            "feature_schema_hash":sha256_file(reference_schema_path),
            "outputs":[artifact_identity(path) for path in reference_output_paths],
        })
        reference_manifest["content_sha256"]=sha256_bytes(canonical_json({key:value for key,value in reference_manifest.items() if key!="content_sha256"}).encode())
        mutable_atomic_write_json(reference_manifest_path,reference_manifest)
    output_paths=[index_path,schema_path,*sorted((output/"shards").glob("shard_*.npz"))]
    manifest.update({
        "status":"complete","completed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"row_count":completed,"candidate_count":completed*5,
        "unique_sample_count":len(set(all_ids)),"unique_candidate_count":completed*5,
        "missing_count":missing_depth_count,"fallback_count":missing_depth_count,
        "missing_depth_reason_counts":missing_depth_reason_counts,
        "hook_max_abs_difference":hook_differences,"feature_schema_hash":sha256_file(schema_path),
        "outputs":[artifact_identity(path) for path in output_paths],
    })
    if reference_manifest_path is not None:
        manifest["precast_reference_manifest"]=artifact_identity(reference_manifest_path)
    manifest["content_sha256"]=sha256_bytes(canonical_json({key:value for key,value in manifest.items() if key!="content_sha256"}).encode())
    mutable_atomic_write_json(manifest_path,manifest)
    return manifest
