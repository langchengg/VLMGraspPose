#!/usr/bin/env python3
"""Extract label-free 64x64 candidate-aligned multi-channel crop shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.backends.conditioning import resize_probability_to_native  # noqa: E402
from src.grasping.common.sample_io import CompactSampleLoader, read_deployment_manifest  # noqa: E402
from src.grasping.g1_c1_safe_rerank.contracts import canonical_sha256, sha256_file  # noqa: E402


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("G1", "C1"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _aligned_crop(image: np.ndarray, row: Mapping[str, Any], *, interpolation: int) -> np.ndarray:
    height, width = image.shape[:2]
    center = (float(row["center_x"]), float(row["center_y"]))
    rotation = cv2.getRotationMatrix2D(center, float(row["angle_deg"]), 1.0)
    rotated = cv2.warpAffine(image, rotation, (width, height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    side = int(np.clip(round(max(float(row["width_px"]) * 1.5, 48.0)), 48, 256))
    patch = cv2.getRectSubPix(rotated, (side, side), center)
    return cv2.resize(patch, (64, 64), interpolation=interpolation)


def _candidate_crops(arrays: Any, candidates: pd.DataFrame) -> np.ndarray:
    probability = resize_probability_to_native(arrays.probability, arrays.depth_m.shape)
    gray = cv2.cvtColor(arrays.rgb, cv2.COLOR_RGB2GRAY)
    channels = (
        gray.astype(np.uint8),
        np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8),
        arrays.binary_mask.astype(np.uint8) * 255,
        np.rint(np.clip(arrays.depth_m, 0.0, 3.0) / 3.0 * 255.0).astype(np.uint8),
    )
    crops = np.empty((len(candidates), 4, 64, 64), dtype=np.uint8)
    for index, row in enumerate(candidates.to_dict(orient="records")):
        for channel, image in enumerate(channels):
            interpolation = cv2.INTER_NEAREST if channel == 2 else cv2.INTER_LINEAR
            crops[index, channel] = _aligned_crop(image, row, interpolation=interpolation)
    return crops


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    base = args.base_run.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    backend = args.backend.lower()
    deployment_path = base / "manifests" / f"{args.split}_samples.parquet"
    deployment = list(read_deployment_manifest(deployment_path))
    candidate_path = run / "data" / f"frozen_{backend}_{args.split}_candidates.parquet"
    candidates = pd.read_parquet(candidate_path)
    by_sample = {str(sample): group for sample, group in candidates.groupby("sample_id", sort=False)}
    output = run / "02_features" / args.split / backend / "candidate_crops_64"
    chunks = (len(deployment) + args.chunk_size - 1) // args.chunk_size
    loader = CompactSampleLoader()
    total_rows = 0
    shard_records = []
    for chunk in range(chunks):
        start, stop = chunk * args.chunk_size, min(len(deployment), (chunk + 1) * args.chunk_size)
        path = output / "shards" / f"part-{chunk:05d}.npz"
        marker = output / "markers" / f"part-{chunk:05d}.json"
        local_ids = [str(row["sample_id"]) for row in deployment[start:stop]]
        local_candidates = candidates.loc[candidates["sample_id"].astype(str).isin(local_ids)]
        identity_sha = canonical_sha256(sorted(local_candidates["candidate_identity_sha256"].astype(str)))
        if args.resume and path.is_file() and marker.is_file():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record.get("candidate_identity_sha256") == identity_sha and sha256_file(path) == record.get("sha256"):
                total_rows += int(record["candidate_rows"])
                shard_records.append(record)
                continue
            raise RuntimeError(f"stale crop shard: {path}")
        crop_parts, candidate_ids, sample_ids = [], [], []
        for deployment_row in deployment[start:stop]:
            sample_id = str(deployment_row["sample_id"])
            group = by_sample.get(sample_id)
            if group is None or group.empty:
                continue
            arrays = loader.load(deployment_row, mask_source="predicted", load_intrinsics=False)
            crop_parts.append(_candidate_crops(arrays, group))
            candidate_ids.extend(group["stable_candidate_id"].astype(str).tolist())
            sample_ids.extend([sample_id] * len(group))
        crops = np.concatenate(crop_parts, axis=0) if crop_parts else np.empty((0, 4, 64, 64), dtype=np.uint8)
        if len(crops) != len(candidate_ids) or len(crops) != len(sample_ids) or len(crops) != len(local_candidates):
            raise AssertionError("crop shard candidate coverage mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        np.savez_compressed(
            temporary,
            sample_id=np.asarray(sample_ids, dtype=str),
            candidate_id=np.asarray(candidate_ids, dtype=str),
            crop=crops,
        )
        os.replace(temporary, path)
        record = {"chunk": chunk, "start": start, "stop": stop, "candidate_rows": len(crops), "candidate_identity_sha256": identity_sha, "sha256": sha256_file(path), "path": str(path)}
        _atomic_json(marker, record)
        shard_records.append(record)
        total_rows += len(crops)
        print(json.dumps({"backend": args.backend, "split": args.split, "completed": stop, "total": len(deployment), "crop_rows": total_rows}), flush=True)
    if total_rows != len(candidates):
        raise AssertionError(f"crop coverage drift: {total_rows} != {len(candidates)}")
    _atomic_json(output / "COMPLETE.json", {"status": "COMPLETE", "backend": args.backend, "split": args.split, "candidate_rows": total_rows, "shape": [4, 64, 64], "dtype": "uint8", "channels": ["rgb_grayscale", "hifi_probability", "predicted_mask", "depth_0_to_3m"], "alignment": "candidate centre; rotate grasp angle horizontal; square side=clip(1.5*width,48,256)", "source_candidate_sha256": sha256_file(candidate_path), "source_deployment_sha256": sha256_file(deployment_path), "shards": shard_records, "gt_loaded": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
