#!/usr/bin/env python3
"""Resumable, native-score-only G1/C1 inference.

This intentionally bypasses the project gated quality map, mask/jaw candidate
rescoring, and project NMS.  Target masks still define the already selected
``dilated_crop`` model input contract; they do not modify native output scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from skimage.feature import peak_local_max


ROOT = Path(__file__).resolve().parents[2]
HIFI_ROOT = ROOT / "HiFi_reproduction"
sys.path.insert(0, str(HIFI_ROOT))

from src.grasping.backends import BackendSample  # noqa: E402
from src.grasping.backends.base import NetworkBackendConfig, prepare_conditioned_input  # noqa: E402
from src.grasping.backends.network_utils import (  # noqa: E402
    official_gaussian_post_process,
    synchronize_device,
)
from src.grasping.backends.training import load_finetuned_model  # noqa: E402
from src.grasping.common.sample_io import (  # noqa: E402
    CompactSampleLoader,
    aligned_labels,
    read_deployment_manifest,
    read_label_manifest,
)


DECODER = {
    "quality_source": "raw network q after upstream Gaussian sigma=2; no sigmoid",
    "quality_threshold": 0.2,
    "min_peak_distance_px": 20,
    "max_peaks": 100,
    "exclude_border": True,
    "angle": "Gaussian sigma=2 on 0.5*atan2(sin,cos)",
    "width": "raw width*150 then Gaussian sigma=1",
    "rectangle_height": "native jaw width / 2",
    "post_peak_nms": False,
    "mask_score_gating": False,
    "candidate_rescoring": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    os.replace(temporary, path)


def load_config(method: str, selected_path: Path, oracle: bool) -> tuple[NetworkBackendConfig, dict[str, Any]]:
    selected = json.loads(selected_path.read_text())
    expected_method = "G1" if method == "g1" else "C1"
    if selected.get("method_id") != expected_method:
        raise ValueError("selected config method mismatch")
    # Input conditioning remains the validation-selected deployed contract.
    config = NetworkBackendConfig(
        conditioning_variant=str(selected["conditioning_variant"]),
        input_size=int(selected["input_size"]),
        dilation_fraction=float(selected["dilation_fraction"]),
        minimum_crop_side_px=int(selected["minimum_crop_side_px"]),
        minimum_mask_area_px=int(selected["minimum_mask_area_px"]),
        quality_threshold=DECODER["quality_threshold"],
        min_peak_distance_px=DECODER["min_peak_distance_px"],
        max_raw_candidates=DECODER["max_peaks"],
        fixed_height_px=20.0,
        center_gate_exponent=0.0,
        jaw_gate_exponent=0.0,
        allow_oracle=oracle,
        device="mps",
    )
    if method == "g1" and int(selected.get("input_channels", 4)) != 4:
        raise ValueError("G1 selected checkpoint is not the audited RGB-D model")
    return config, selected


def raw_maps(outputs: object) -> dict[str, np.ndarray]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise ValueError("expected four model output tensors")
    names = ("quality_raw", "cos_2theta_raw", "sin_2theta_raw", "width_raw")
    result = {}
    for name, tensor in zip(names, outputs, strict=True):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError("invalid model output tensor")
        result[name] = tensor.detach().float().cpu().numpy()[0, 0].astype(np.float32)
    return result


def infer_one(
    *,
    method: str,
    model: torch.nn.Module,
    device: torch.device,
    config: NetworkBackendConfig,
    sample: BackendSample,
    raw_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    conditioning_seconds = model_seconds = decoder_seconds = 0.0
    try:
        stage = time.perf_counter()
        conditioned = prepare_conditioned_input(sample, config)
        conditioning_seconds = time.perf_counter() - stage
        model_input = conditioned.depth_chw
        if method == "g1":
            model_input = np.concatenate((conditioned.depth_chw, conditioned.rgb_chw), axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(model_input)).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=torch.float32)
        stage = time.perf_counter()
        with torch.inference_mode():
            outputs = model(tensor)
        synchronize_device(device)
        model_seconds = time.perf_counter() - stage
        maps = raw_maps(outputs)
        stage = time.perf_counter()
        quality, cos_map, sin_map, width_map = official_gaussian_post_process(
            outputs,
            expected_spatial_shape=(config.input_size, config.input_size),
        )
        peaks = peak_local_max(
            quality,
            min_distance=int(DECODER["min_peak_distance_px"]),
            threshold_abs=float(DECODER["quality_threshold"]),
            num_peaks=int(DECODER["max_peaks"]),
        )
        # Preserve the native score order with deterministic row/column tie-breaks.
        ordered = sorted(
            ((int(row), int(column)) for row, column in peaks),
            key=lambda rc: (-float(quality[rc]), rc[0], rc[1]),
        )
        candidates: list[dict[str, Any]] = []
        for peak_rank, (row, column) in enumerate(ordered, 1):
            angle = float(np.degrees(0.5 * np.arctan2(sin_map[row, column], cos_map[row, column])))
            model_width = float(width_map[row, column])
            if not np.isfinite(model_width) or model_width <= 0:
                continue
            cx, cy, native_angle, native_width = conditioned.transform.model_to_native_pose(
                column, row, angle, model_width, clip=False
            )
            values = (cx, cy, native_angle, native_width, float(quality[row, column]))
            if not all(np.isfinite(values)) or native_width <= 0:
                continue
            rank = len(candidates) + 1
            candidates.append(
                {
                    "sample_id": sample.sample_id,
                    "method": method.upper(),
                    "candidate_id": f"native_peak_{peak_rank:03d}",
                    "native_rank": rank,
                    "native_score": float(quality[row, column]),
                    "cx_px": float(cx),
                    "cy_px": float(cy),
                    "theta_deg": float((native_angle + 90.0) % 180.0 - 90.0),
                    "jaw_width_px": float(native_width),
                    "rectangle_height_px": float(native_width / 2.0),
                    "source_row": row,
                    "source_column": column,
                    "status": "ok",
                    "failure_reason": None,
                    "transform_json": json.dumps(conditioned.transform.__dict__ if hasattr(conditioned.transform, "__dict__") else {
                        key: getattr(conditioned.transform, key)
                        for key in ("crop_x", "crop_y", "crop_width", "crop_height", "model_width", "model_height", "native_width", "native_height")
                    }, sort_keys=True),
                }
            )
        decoder_seconds = time.perf_counter() - stage
        if raw_path is not None:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.tmp.npz")
            np.savez_compressed(
                temporary,
                **maps,
                quality_post=quality.astype(np.float32),
                cos_2theta_post=cos_map.astype(np.float32),
                sin_2theta_post=sin_map.astype(np.float32),
                width_px_post=width_map.astype(np.float32),
            )
            os.replace(temporary, raw_path)
        failure = None if candidates else "no_candidate_generated"
        status = "ok" if candidates else "no_output"
        sample_row = {
            "sample_id": sample.sample_id,
            "method": method.upper(),
            "status": status,
            "failure_reason": failure,
            "candidate_count": len(candidates),
            "conditioning_seconds": conditioning_seconds,
            "model_seconds": model_seconds,
            "decoder_seconds": decoder_seconds,
            "total_seconds": time.perf_counter() - started,
            "mask_source": sample.mask_source,
            "mask_area_px": conditioned.mask_area_px,
            "valid_depth_fraction": conditioned.valid_depth_fraction,
            "raw_maps_path": None if raw_path is None else str(raw_path),
        }
        return sample_row, candidates
    except Exception as error:  # retain every failure in the formal denominator
        reason = f"{type(error).__name__}:{error}"
        expected_no_output = type(error).__name__ == "BackendInputError" and str(error) in {
            "empty_mask", "mask_too_small", "missing_predicted_mask",
            "missing_probability_map", "invalid_depth", "invalid_target_depth",
        }
        return (
            {
                "sample_id": sample.sample_id,
                "method": method.upper(),
                "status": "no_output" if expected_no_output else "technical_failure",
                "failure_reason": reason,
                "candidate_count": 0,
                "conditioning_seconds": conditioning_seconds,
                "model_seconds": model_seconds,
                "decoder_seconds": decoder_seconds,
                "total_seconds": time.perf_counter() - started,
                "mask_source": sample.mask_source,
                "mask_area_px": None,
                "valid_depth_fraction": None,
                "raw_maps_path": None,
            },
            [],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--method", choices=("g1", "c1"), required=True)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-raw-maps", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_run.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    variant = f"{args.method}_gtmask_oracle" if args.oracle else args.method
    output = run / "02_predictions" / "native_work" / variant
    output.mkdir(parents=True, exist_ok=True)
    samples_file = source / "manifests/test_samples.parquet"
    labels_file = source / "manifests/test_labels.parquet"
    deployment = read_deployment_manifest(samples_file)
    labels = aligned_labels(deployment, read_label_manifest(labels_file))
    if args.limit is not None:
        deployment, labels = deployment[: args.limit], labels[: args.limit]
    selected_path = source / "selected_configs" / f"{args.method.upper()}.json"
    config, selected = load_config(args.method, selected_path, args.oracle)
    checkpoint = Path(selected["finetuned_checkpoint"]).resolve()
    expected_checkpoint_sha = str(selected["finetuned_checkpoint_sha256"])
    if sha256_file(checkpoint) != expected_checkpoint_sha:
        raise ValueError("fine-tuned checkpoint hash mismatch")
    model, observed_sha, payload = load_finetuned_model(
        checkpoint,
        backend="grconvnet" if args.method == "g1" else "ggcnn2",
        device="mps",
    )
    if observed_sha != expected_checkpoint_sha:
        raise ValueError("loaded fine-tuned checkpoint hash mismatch")
    device = torch.device("mps")
    rows_path = output / "per_sample.parquet"
    candidates_path = output / "candidates.parquet"
    sample_rows = pq.read_table(rows_path).to_pylist() if args.resume and rows_path.exists() else []
    candidate_rows = pq.read_table(candidates_path).to_pylist() if args.resume and candidates_path.exists() else []
    done = {str(row["sample_id"]) for row in sample_rows}
    loader = CompactSampleLoader()
    started = time.perf_counter()
    expected = len(deployment)
    for index, (row, label) in enumerate(zip(deployment, labels, strict=True), 1):
        sample_id = str(row["sample_id"])
        if sample_id in done:
            continue
        mask_source = "gt_mask_oracle" if args.oracle else "predicted"
        arrays = loader.load(row, mask_source=mask_source, labels=label if args.oracle else None)
        sample = BackendSample(
            sample_id=sample_id,
            rgb=arrays.rgb,
            depth_m=arrays.depth_m,
            predicted_mask=None if args.oracle else arrays.binary_mask,
            probability_map=None if args.oracle else arrays.probability,
            mask_source=mask_source,
            oracle_mask=arrays.binary_mask if args.oracle else None,
            metadata={"scene_id": arrays.scene_id},
        )
        raw_path = None
        if args.save_raw_maps and not args.oracle:
            raw_path = output / "raw_maps" / sample_id[:3] / f"{sample_id}.npz"
        sample_result, candidates = infer_one(
            method=args.method,
            model=model,
            device=device,
            config=config,
            sample=sample,
            raw_path=raw_path,
        )
        sample_result.update(
            {
                "checkpoint_sha256": observed_sha,
                "selected_config_sha256": sha256_file(selected_path),
                "native_decoder_config_sha256": canonical_hash(DECODER),
            }
        )
        for item in candidates:
            item.update(
                {
                    "checkpoint_sha256": observed_sha,
                    "selected_config_sha256": sha256_file(selected_path),
                    "native_decoder_config_sha256": canonical_hash(DECODER),
                }
            )
        sample_rows.append(sample_result)
        candidate_rows.extend(candidates)
        done.add(sample_id)
        completed = len(done)
        if completed % 25 == 0 or completed == expected:
            atomic_parquet(rows_path, sample_rows)
            atomic_parquet(candidates_path, candidate_rows)
            atomic_json(
                output / "progress.json",
                {
                    "status": "COMPLETE" if completed == expected else "RUNNING",
                    "variant": variant,
                    "completed": completed,
                    "expected": expected,
                    "candidate_rows": len(candidate_rows),
                    "technical_failures": sum(row["status"] == "technical_failure" for row in sample_rows),
                    "elapsed_seconds_this_invocation": time.perf_counter() - started,
                    "pid": os.getpid(),
                },
            )
            print(f"[{variant}] {completed}/{expected}, candidates={len(candidate_rows)}", flush=True)
    if len(done) != expected:
        raise RuntimeError(f"incomplete run: {len(done)}/{expected}")
    atomic_json(
        output / "run_manifest.json",
        {
            "status": "COMPLETE",
            "variant": variant,
            "sample_count": expected,
            "candidate_count": len(candidate_rows),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": observed_sha,
            "checkpoint_validation": payload.get("best_validation"),
            "selected_config": str(selected_path),
            "selected_config_sha256": sha256_file(selected_path),
            "native_decoder": DECODER,
            "native_decoder_config_sha256": canonical_hash(DECODER),
            "raw_maps_saved": bool(args.save_raw_maps and not args.oracle),
            "source_samples_sha256": sha256_file(samples_file),
            "source_labels_sha256": sha256_file(labels_file),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
