"""Extract true label-free CROG/G1/C1 dense evidence at every canonical anchor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional
import cv2

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HIFI_ROOT = ROOT / "HiFi_reproduction"
CROG_ROOT = ROOT / "crog_reproduction" / "CROG"
for item in (CROG_ROOT, SRC, HIFI_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))
CROG_SITE_PACKAGES = (
    CROG_ROOT
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if CROG_SITE_PACKAGES.is_dir() and str(CROG_SITE_PACKAGES) not in sys.path:
    # Keep the current interpreter's audited scientific stack ahead of this
    # path; it supplies only CROG's pure-Python tokenizer/runtime extras.
    sys.path.append(str(CROG_SITE_PACKAGES))

import utils.config as crog_config
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.simple_tokenizer import SimpleTokenizer

from src.grasping.backends.conditioning import condition_rgbd
from src.grasping.common.geometry import CropTransform
from src.grasping.common.sample_io import CompactSampleLoader
from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.feature_extractors.consensus import BACKENDS
from unified_reranking.feature_extractors.tri_backend_dense import (
    tri_backend_dense_features,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log


ID_COLUMNS = ("sample_id", "candidate_id")
DEFAULT_CONFIG = "config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml"
DEFAULT_CHECKPOINT = (
    "exp/OCID-VLG_multiple_mac/"
    "CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth"
)
CROG_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
CROG_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
CROG_TOKENIZER = SimpleTokenizer()


class _IdentityTransform:
    def native_to_model_point(self, x: float, y: float) -> tuple[float, float]:
        return float(x), float(y)

    def model_to_native_point(
        self, x: float, y: float, *, clip: bool = False
    ) -> tuple[float, float]:
        return float(x), float(y)

    def model_to_native_pose(
        self,
        x: float,
        y: float,
        angle: float,
        width: float,
        *,
        clip: bool = False,
    ) -> tuple[float, float, float, float]:
        return float(x), float(y), float(angle), float(width)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    result = torch.device(requested)
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return result


def _crog_affine(
    image_shape: tuple[int, int], input_size: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    scale = min(input_size / height, input_size / width)
    new_height, new_width = height * scale, width * scale
    bias_x = (input_size - new_width) / 2.0
    bias_y = (input_size - new_height) / 2.0
    source = np.asarray([[0, 0], [width, 0], [0, height]], dtype=np.float32)
    target = np.asarray(
        [[bias_x, bias_y], [new_width + bias_x, bias_y], [bias_x, new_height + bias_y]],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(source, target), cv2.getAffineTransform(target, source)


def preprocess_inference_record(
    record: dict[str, Any], *, input_size: int, word_length: int
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(record["image_path"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    forward, inverse = _crog_affine(image_rgb.shape[:2], input_size)
    warped = cv2.warpAffine(
        image_rgb,
        forward,
        (input_size, input_size),
        flags=cv2.INTER_CUBIC,
        borderValue=tuple((CROG_MEAN[:, 0, 0] * 255).tolist()),
    )
    normalized = (
        warped.transpose(2, 0, 1).astype(np.float32) / 255.0 - CROG_MEAN
    ) / CROG_STD
    start_token = CROG_TOKENIZER.encoder["<|startoftext|>"]
    end_token = CROG_TOKENIZER.encoder["<|endoftext|>"]
    tokens = [
        start_token,
        *CROG_TOKENIZER.encode(str(record["language_instruction"])),
        end_token,
    ]
    if len(tokens) > word_length:
        tokens = tokens[:word_length]
        tokens[-1] = end_token
    token_ids = torch.zeros(word_length, dtype=torch.long)
    token_ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return {
        "record": record,
        "image": torch.from_numpy(normalized),
        "token_ids": token_ids,
        "forward": forward,
        "inverse": inverse,
        "ori_size": tuple(map(int, image_rgb.shape[:2])),
    }


def _model_inputs(
    batch: list[dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor, ...]:
    images = torch.stack([item["image"] for item in batch]).to(device)
    tokens = torch.stack([item["token_ids"] for item in batch]).to(device)
    return images, tokens, None, None, None, None, None


def _postprocess_maps(
    prediction: tuple[torch.Tensor, ...],
    image: torch.Tensor,
    data: dict[str, Any],
) -> list[tuple[np.ndarray, ...]]:
    instance, quality, sine, cosine, width = prediction
    instance = torch.sigmoid(instance)
    quality = torch.sigmoid(quality)
    width = torch.sigmoid(width)
    values = [instance, quality, sine, cosine, width]
    if instance.shape[-2:] != image.shape[-2:]:
        values = [
            torch_functional.interpolate(
                value,
                size=image.shape[-2:],
                mode="bicubic",
                align_corners=True,
            )
            for value in values
        ]
    values = [value.squeeze(1) for value in values]
    restored: list[tuple[np.ndarray, ...]] = []
    for batch_index in range(values[0].shape[0]):
        height, width_px = map(int, data["ori_size"][batch_index])
        inverse = data["inverse"][batch_index]
        restored.append(
            tuple(
                cv2.warpAffine(
                    value[batch_index].detach().cpu().numpy(),
                    inverse,
                    (width_px, height),
                    flags=cv2.INTER_CUBIC,
                )
                for value in values
            )
        )
    return restored


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fair-test-source", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="formal")
    return parser.parse_args()


def _source_paths(
    run_dir: Path, fair_test_source: Path, route: str, split: str
) -> tuple[Path, Path]:
    if split == "test":
        root = fair_test_source / "02_predictions" / "native_work" / route
    else:
        root = run_dir / "02_candidates" / "native_work" / f"{route}_{split}"
    return root / "candidates.parquet", root / "per_sample.parquet"


def _load_config(fair_test_source: Path, route: str) -> tuple[dict[str, Any], Path]:
    path = fair_test_source / "config" / f"{route.upper()}_validation_selected.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return value, path


def _transform_from_json(value: str) -> CropTransform:
    payload = json.loads(value)
    return CropTransform(**payload)


def _stored_transform(
    source_group: pd.DataFrame,
    *,
    expected_shape: tuple[int, int],
) -> CropTransform | None:
    if source_group.empty:
        return None
    values = source_group["transform_json"].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise RuntimeError("backend sample has inconsistent persisted CropTransforms")
    transform = _transform_from_json(values[0])
    if (transform.model_height, transform.model_width) != expected_shape:
        raise RuntimeError("persisted CropTransform and raw dense-map shapes differ")
    for candidate in source_group.itertuples(index=False):
        model_x, model_y = transform.native_to_model_point(candidate.cx_px, candidate.cy_px)
        if abs(model_x - float(candidate.source_column)) > 1e-6 or abs(model_y - float(candidate.source_row)) > 1e-6:
            raise RuntimeError("persisted CropTransform does not reproduce frozen peak coordinate")
        native_x, native_y = transform.model_to_native_point(model_x, model_y)
        if np.hypot(native_x - candidate.cx_px, native_y - candidate.cy_px) > 1e-6:
            raise RuntimeError("persisted CropTransform original/model/original round-trip failed")
    return transform


def _fallback_transform(
    row: dict[str, Any], config: dict[str, Any], loader: CompactSampleLoader
) -> CropTransform:
    arrays = loader.load(row, mask_source="predicted", labels=None, load_intrinsics=False)
    conditioned = condition_rgbd(
        rgb=arrays.rgb,
        depth_m=arrays.depth_m,
        binary_mask=arrays.binary_mask,
        probability=arrays.probability,
        variant=str(config["conditioning_variant"]),
        output_size=int(config["input_size"]),
        dilation_fraction=float(config["dilation_fraction"]),
        minimum_side_px=int(config["minimum_crop_side_px"]),
    )
    return conditioned.transform


def _archive_maps(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("quality_post", "cos_2theta_post", "sin_2theta_post", "width_px_post")
        missing = set(required).difference(archive.files)
        if missing:
            raise RuntimeError(f"backend raw map archive misses {sorted(missing)}: {path}")
        return {name: np.asarray(archive[name]) for name in required}


def _crog_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "split": split,
        "image_path": str(row["source_rgb_path"]),
        "depth_path": str(row["source_depth_path"]),
        "language_instruction": str(row["language"]),
    }


def _sample_assets(
    rows: list[dict[str, Any]],
    raw_paths: dict[str, dict[str, str]],
    map_states: dict[str, dict[str, dict[str, str]]],
    cache: dict[str, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        rgb_path = str(Path(str(row["source_rgb_path"])).resolve())
        observed_rgb = cache.get(rgb_path)
        if observed_rgb is None:
            observed_rgb = sha256_file(rgb_path)
            cache[rgb_path] = observed_rgb
        if observed_rgb != str(row["source_rgb_sha256"]):
            raise RuntimeError(f"CROG RGB input drift: {sample_id}")
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "rgb": {"path": rgb_path, "sha256": observed_rgb},
            "language_sha256": str(row["language_sha256"]),
            "backend_maps": {},
        }
        for route in ("g1", "c1"):
            raw_path = raw_paths[route].get(sample_id)
            state = map_states[route].get(sample_id)
            if state is None:
                raise RuntimeError(f"backend per-sample record is missing: {route}/{sample_id}")
            if raw_path is None:
                if state["status"] not in {"no_output", "technical_failure"}:
                    raise RuntimeError(
                        f"backend map is absent without an audited failure: {route}/{sample_id}"
                    )
                record["backend_maps"][route] = {
                    "availability": "MISSING_BACKEND_MAP",
                    "status": state["status"],
                    "failure_reason": state["failure_reason"],
                }
            else:
                path = str(Path(raw_path).resolve())
                observed = sha256_file(path)
                record["backend_maps"][route] = {
                    "availability": "AVAILABLE",
                    "path": path,
                    "sha256": observed,
                    "status": state["status"],
                }
        result.append(record)
    return result


def _valid_shard(
    marker: Path,
    paths: dict[str, Path],
    expected: dict[str, Any],
) -> bool:
    if not marker.is_file() or any(not path.is_file() for path in paths.values()):
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("status") != "COMPLETE" or any(value.get(key) != item for key, item in expected.items()):
        return False
    artifacts = value.get("artifacts", {})
    for route, path in paths.items():
        record = artifacts.get(route, {})
        if record.get("sha256") != sha256_file(path):
            return False
        frame = pd.read_parquet(path, columns=list(ID_COLUMNS))
        keys = frame[list(ID_COLUMNS)].astype(str).sort_values(list(ID_COLUMNS)).to_dict("records")
        if record.get("candidate_keys_sha256") != canonical_sha256(keys):
            return False
    return True


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("batch and chunk sizes must be positive")
    if args.tag == "formal" and args.limit is not None:
        raise ValueError("limited tri-backend extraction requires a non-formal tag")
    run_dir = args.run_dir.resolve()
    fair_source = args.fair_test_source.resolve()
    paired_path = run_dir / "01_manifests" / f"paired_{args.split}.parquet"
    paired = pd.read_parquet(paired_path).to_dict("records")
    if args.limit is not None:
        paired = paired[: args.limit]
    allowed_ids = {str(row["sample_id"]) for row in paired}
    pools = {
        route: pd.read_parquet(run_dir / "02_candidates" / f"{route}_{args.split}_top5.parquet")
        for route in BACKENDS
    }
    if args.limit is not None:
        pools = {route: frame.loc[frame["sample_id"].astype(str).isin(allowed_ids)].copy() for route, frame in pools.items()}
    calibration_suffix = "train_oof" if args.split == "train" else args.split
    calibrations = {
        route: pd.read_parquet(run_dir / "05_calibration" / f"{route}_{calibration_suffix}.parquet")
        for route in BACKENDS
    }
    if args.limit is not None:
        calibrations = {route: frame.loc[frame["sample_id"].astype(str).isin(allowed_ids)].copy() for route, frame in calibrations.items()}

    source_frames: dict[str, pd.DataFrame] = {}
    per_sample_frames: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, Path] = {}
    per_sample_paths: dict[str, Path] = {}
    raw_paths: dict[str, dict[str, str]] = {}
    map_states: dict[str, dict[str, dict[str, str]]] = {}
    for route in ("g1", "c1"):
        source_path, per_sample_path = _source_paths(run_dir, fair_source, route, args.split)
        source_frames[route] = pd.read_parquet(source_path)
        per_sample_frames[route] = pd.read_parquet(per_sample_path)
        source_paths[route], per_sample_paths[route] = source_path, per_sample_path
        raw_paths[route] = {
            str(row.sample_id): str(row.raw_maps_path)
            for row in per_sample_frames[route].itertuples(index=False)
            if isinstance(row.raw_maps_path, str) and row.raw_maps_path
        }
        map_states[route] = {
            str(row.sample_id): {
                "status": str(row.status),
                "failure_reason": str(row.failure_reason),
            }
            for row in per_sample_frames[route].itertuples(index=False)
        }
    configs: dict[str, dict[str, Any]] = {}
    config_paths: dict[str, Path] = {}
    for route in ("g1", "c1"):
        configs[route], config_paths[route] = _load_config(fair_source, route)

    crog_cfg_path = (CROG_ROOT / DEFAULT_CONFIG).resolve()
    crog_checkpoint = (CROG_ROOT / DEFAULT_CHECKPOINT).resolve()
    cfg = crog_config.load_cfg_from_cfg_file(str(crog_cfg_path))
    if not Path(str(cfg.clip_pretrain)).is_absolute():
        cfg.clip_pretrain = str((CROG_ROOT / str(cfg.clip_pretrain)).resolve())
    device = _device(args.device)
    model, _ = build_crog(cfg)
    model = model.to(device).eval()
    load_checkpoint(crog_checkpoint, model, device, strict=True)
    loader = CompactSampleLoader()
    grouped_pools = {
        route: {str(key): value.copy() for key, value in frame.groupby("sample_id", sort=False)}
        for route, frame in pools.items()
    }
    grouped_calibrations = {
        route: {str(key): value.copy() for key, value in frame.groupby("sample_id", sort=False)}
        for route, frame in calibrations.items()
    }
    grouped_source = {
        route: {str(key): value.copy() for key, value in frame.groupby("sample_id", sort=False)}
        for route, frame in source_frames.items()
    }
    empty_pool = {route: pools[route].iloc[0:0].copy() for route in BACKENDS}
    empty_calibration = {route: calibrations[route].iloc[0:0].copy() for route in BACKENDS}

    output_name = args.split if args.tag == "formal" else f"{args.split}_{args.tag}"
    output = run_dir / "03_features" / "tri_backend_dense" / output_name
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    common_signature = {
        "paired_manifest_sha256": sha256_file(paired_path),
        "candidate_manifests": {
            route: sha256_file(run_dir / "02_candidates" / f"{route}_{args.split}_top5.parquet")
            for route in BACKENDS
        },
        "calibration_artifacts": {
            route: sha256_file(run_dir / "05_calibration" / f"{route}_{calibration_suffix}.parquet")
            for route in BACKENDS
        },
        "source_candidates": {route: sha256_file(path) for route, path in source_paths.items()},
        "per_sample_maps": {route: sha256_file(path) for route, path in per_sample_paths.items()},
        "backend_configs": {route: sha256_file(path) for route, path in config_paths.items()},
        "crog_config_sha256": sha256_file(crog_cfg_path),
        "crog_checkpoint_sha256": sha256_file(crog_checkpoint),
        "feature_extractor_sha256": sha256_file(
            ROOT / "src" / "unified_reranking" / "feature_extractors" / "tri_backend_dense.py"
        ),
        "backend_map_extractor_sha256": sha256_file(
            ROOT / "src" / "unified_reranking" / "feature_extractors" / "backend_maps.py"
        ),
        "consensus_extractor_sha256": sha256_file(
            ROOT / "src" / "unified_reranking" / "feature_extractors" / "consensus.py"
        ),
        "tool_sha256": sha256_file(Path(__file__)),
        "requested_device": args.device,
        "resolved_device": str(device),
        "numerical_precision": "model_default_float32",
        "batch_size": int(args.batch_size),
    }
    rgb_hash_cache: dict[str, str] = {}
    all_asset_records: list[dict[str, Any]] = []
    expected_paths: dict[str, list[Path]] = {route: [] for route in BACKENDS}
    for start in range(0, len(paired), args.chunk_size):
        stop = min(start + args.chunk_size, len(paired))
        rows = paired[start:stop]
        sample_ids = [str(row["sample_id"]) for row in rows]
        asset_records = _sample_assets(rows, raw_paths, map_states, rgb_hash_cache)
        all_asset_records.extend(asset_records)
        chunk_paths = {
            route: shards / f"{start:08d}_{stop:08d}_{route}.parquet" for route in BACKENDS
        }
        for route, path in chunk_paths.items():
            expected_paths[route].append(path)
        marker = shards / f"{start:08d}_{stop:08d}.json"
        expected = {
            **common_signature,
            "start": start,
            "stop": stop,
            "sample_identity_sha256": canonical_sha256(sample_ids),
            "asset_identity_sha256": canonical_sha256(asset_records),
        }
        if _valid_shard(marker, chunk_paths, expected):
            continue
        pieces: dict[str, list[pd.DataFrame]] = {route: [] for route in BACKENDS}
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            batch = [
                preprocess_inference_record(
                    _crog_record(row, args.split),
                    input_size=int(cfg.input_size),
                    word_length=int(cfg.word_len),
                )
                for row in batch_rows
            ]
            inputs = _model_inputs(batch, device)
            pred, _ = model(*inputs)
            restored = _postprocess_maps(
                pred,
                inputs[0],
                {
                    "inverse": [item["inverse"] for item in batch],
                    "ori_size": [item["ori_size"] for item in batch],
                },
            )
            for local_index, row in enumerate(batch_rows):
                sample_id = str(row["sample_id"])
                ins, quality, sine, cosine, width_probability = restored[local_index]
                crog_maps = {
                    "quality_post": np.asarray(quality),
                    "cos_2theta_post": np.asarray(cosine),
                    "sin_2theta_post": np.asarray(sine),
                    "width_px_post": np.asarray(width_probability) * 100.0,
                }
                dense_maps: dict[str, dict[str, np.ndarray] | None] = {
                    "crog": crog_maps,
                    "g1": None
                    if sample_id not in raw_paths["g1"]
                    else _archive_maps(Path(raw_paths["g1"][sample_id])),
                    "c1": None
                    if sample_id not in raw_paths["c1"]
                    else _archive_maps(Path(raw_paths["c1"][sample_id])),
                }
                transforms: dict[str, Any] = {"crog": _IdentityTransform()}
                for route in ("g1", "c1"):
                    if dense_maps[route] is None:
                        transforms[route] = None
                        continue
                    source_group = grouped_source[route].get(
                        sample_id, source_frames[route].iloc[0:0]
                    )
                    transform = _stored_transform(
                        source_group,
                        expected_shape=dense_maps[route]["quality_post"].shape,
                    )
                    if transform is None:
                        transform = _fallback_transform(row, configs[route], loader)
                    transforms[route] = transform
                sample_pools = {
                    route: grouped_pools[route].get(sample_id, empty_pool[route])
                    for route in BACKENDS
                }
                sample_calibrations = {
                    route: grouped_calibrations[route].get(
                        sample_id, empty_calibration[route]
                    )
                    for route in BACKENDS
                }
                for anchor_route in BACKENDS:
                    anchor = sample_pools[anchor_route]
                    if anchor.empty:
                        continue
                    pieces[anchor_route].append(
                        tri_backend_dense_features(
                            anchor_route=anchor_route,
                            anchor_candidates=anchor,
                            candidate_pools=sample_pools,
                            calibrations=sample_calibrations,
                            dense_maps=dense_maps,
                            transforms=transforms,
                        )
                    )
        artifacts: dict[str, Any] = {}
        for route, path in chunk_paths.items():
            frame = pd.concat(pieces[route], ignore_index=True) if pieces[route] else pd.DataFrame()
            _atomic_parquet(path, frame)
            keys = (
                frame[list(ID_COLUMNS)].astype(str).sort_values(list(ID_COLUMNS)).to_dict("records")
                if len(frame)
                else []
            )
            artifacts[route] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "candidate_keys_sha256": canonical_sha256(keys),
                "rows": len(frame),
            }
        atomic_json(marker, {"status": "COMPLETE", **expected, "artifacts": artifacts})
        print(f"[tri_backend_{args.split}] samples {start}:{stop}", flush=True)

    route_artifacts: dict[str, Any] = {}
    model_columns: tuple[str, ...] | None = None
    for route in BACKENDS:
        parts = [pd.read_parquet(path) for path in expected_paths[route]]
        frame = pd.concat([part for part in parts if len(part)], ignore_index=True)
        expected_keys = set(
            map(tuple, pools[route][list(ID_COLUMNS)].astype(str).to_numpy())
        )
        actual_keys = set(map(tuple, frame[list(ID_COLUMNS)].astype(str).to_numpy()))
        if expected_keys != actual_keys or frame.duplicated(list(ID_COLUMNS)).any():
            raise RuntimeError(f"tri-backend dense features changed {route} candidate membership")
        columns = assert_model_feature_columns(
            column for column in frame.columns if column not in ID_COLUMNS
        )
        if model_columns is None:
            model_columns = columns
        elif columns != model_columns:
            raise RuntimeError("tri-backend dense feature schema differs across anchor routes")
        route_dir = output / route
        artifact = route_dir / "candidate_features.parquet"
        _atomic_parquet(artifact, frame)
        route_manifest = {
            "status": "COMPLETE",
            "route": route,
            "split": args.split,
            "track_component": "tri_backend_dense",
            "candidate_rows": len(frame),
            "model_feature_columns": list(columns),
            "model_feature_schema_sha256": canonical_sha256(columns),
            "artifact": {"path": str(artifact.resolve()), "sha256": sha256_file(artifact)},
            "dense_sampling_verified": True,
            "original_model_original_roundtrip_verified": True,
            "nearest_candidates_frozen": True,
            "new_peak_search": False,
            "candidate_geometry_changed": False,
            "candidate_test_labels_read": False if args.split == "test" else None,
            "requested_device": args.device,
            "resolved_device": str(device),
            "numerical_precision": "model_default_float32",
            "source_signature": common_signature,
            "asset_identity_sha256": canonical_sha256(all_asset_records),
        }
        atomic_json(route_dir / "feature_manifest.json", route_manifest)
        route_artifacts[route] = route_manifest["artifact"]
    result = {
        "status": "COMPLETE",
        "split": args.split,
        "tag": args.tag,
        "routes": list(BACKENDS),
        "model_feature_columns": list(model_columns or ()),
        "model_feature_schema_sha256": canonical_sha256(model_columns or ()),
        "artifacts": route_artifacts,
        "source_signature": common_signature,
        "asset_identity_sha256": canonical_sha256(all_asset_records),
        "candidate_test_labels_read": False if args.split == "test" else None,
        "requested_device": args.device,
        "resolved_device": str(device),
        "numerical_precision": "model_default_float32",
    }
    atomic_json(output / "manifest.json", result)
    if args.split == "test" and args.tag == "formal":
        append_access_log(
            run_dir,
            {
                "event": "prelock_label_free_test_stage",
                "stage": "tri_backend_dense_test",
                "inputs": ["candidate_geometry", "RGB", "language", "G1/C1 dense maps"],
                "output_manifest": str((output / "manifest.json").resolve()),
                "output_manifest_sha256": sha256_file(output / "manifest.json"),
                "candidate_labels_opened_as_table": False,
            },
        )
    return result


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P3_P4",
        substage=f"tri_backend_dense_{args.split}_{args.tag}",
        evidence_track="T3_tri_backend",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(args)
        output_name = args.split if args.tag == "formal" else f"{args.split}_{args.tag}"
        artifact = run_dir / "03_features" / "tri_backend_dense" / output_name / "manifest.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
