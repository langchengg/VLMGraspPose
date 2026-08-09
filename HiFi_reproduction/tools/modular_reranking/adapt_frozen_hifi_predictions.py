#!/usr/bin/env python3
"""Create a compact read-only index over the frozen full-test HiFi bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.identity import (  # noqa: E402
    sha256_file,
    stable_sample_id,
)


def atomic_text(path: Path, value: str, *, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    temporary = tmp_root / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any, *, tmp_root: Path) -> None:
    atomic_text(
        path,
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        tmp_root=tmp_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    manifest_path = args.frozen_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    tmp_root = args.tmp_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite adapter output: {output_root}")
    rows_dir = output_root / "rows"
    rows_dir.mkdir(parents=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_manifest_sha256 = sha256_file(manifest_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]

    rows: list[dict[str, Any]] = []
    checkpoint_hashes: set[str] = set()
    for sample_index, record in enumerate(records):
        sample_id = stable_sample_id(
            str(record["scene_id"]), int(record["question_index"])
        )
        sample_root = source_root / sample_id
        metadata_path = sample_root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata["sample_id"] != sample_id
            or int(metadata["question_index"]) != int(record["question_index"])
            or metadata["scene_id"] != record["scene_id"]
            or metadata["query"] != record["text"]
            or metadata.get("oracle_artifacts_exported") is not False
        ):
            raise ValueError(f"frozen HiFi metadata mismatch: {sample_id}")
        probability_path = (sample_root / "target_probability.npy").resolve()
        mask_path = (sample_root / "target_mask.png").resolve()
        rgb_path = Path(metadata["source_rgb"]).resolve()
        depth_path = Path(metadata["source_depth"]).resolve()
        pcd_path = Path(metadata["source_pcd"]).resolve()
        required = [
            metadata_path,
            probability_path,
            mask_path,
            rgb_path,
            depth_path,
            pcd_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"frozen HiFi sources missing: {missing}")
        probability_hash = sha256_file(probability_path)
        mask_hash = sha256_file(mask_path)
        if probability_hash != metadata["prediction_probability_sha256"]:
            raise ValueError(f"frozen probability hash mismatch: {sample_id}")
        if mask_hash != metadata["prediction_mask_sha256"]:
            raise ValueError(f"frozen mask hash mismatch: {sample_id}")
        checkpoint_hashes.add(str(metadata["checkpoint_sha256"]))
        probability = np.load(
            probability_path, mmap_mode="r", allow_pickle=False
        )
        if probability.shape != (352, 352) or probability.dtype != np.float32:
            raise ValueError(
                f"frozen probability protocol mismatch: {sample_id}: "
                f"{probability.shape}/{probability.dtype}"
            )
        if (
            float(metadata.get("mask_binary_threshold", -1.0)) != 0.5
            or metadata.get("mask_native_resize")
            != "nearest_neighbor_from_352_binary_mask"
        ):
            raise ValueError(f"frozen mask protocol mismatch: {sample_id}")
        contract = {
            "foreground_probability": "sigmoid(-background_logit)",
            "foreground_threshold": 0.5,
            "model_mask": "probability>=threshold_at_352",
            "native_mask": "nearest_resize_of_binary_352_mask_to_640x480",
            "probability_shape": [352, 352],
            "native_mask_shape": [480, 640],
        }
        contract_sha = hashlib.sha256(
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        row = {
            "schema_version": 1,
            "split": args.split,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "question_index": int(record["question_index"]),
            "scene_id": str(record["scene_id"]),
            "query": str(record["text"]),
            "processed_rgb_path": str((sample_root / "color.png").resolve()),
            "source_rgb_path": str(rgb_path),
            "source_depth_path": str(depth_path),
            "source_pcd_path": str(pcd_path),
            "source_rgb_sha256": str(metadata["source_rgb_sha256"]),
            "source_depth_sha256": str(metadata["source_depth_sha256"]),
            "source_pcd_sha256": str(metadata["source_pcd_sha256"]),
            "probability_path": str(probability_path),
            "probability_shape": [352, 352],
            "probability_dtype": "float32",
            "probability_sha256": probability_hash,
            "native_mask_path": str(mask_path),
            "native_mask_shape": [480, 640],
            "native_mask_sha256": mask_hash,
            "foreground_threshold": 0.5,
            "threshold_comparison": ">=",
            "model_mask_transform": "probability>=threshold_at_352",
            "native_mask_transform": (
                "nearest_resize_of_binary_352_mask_to_640x480"
            ),
            "inference_contract": contract,
            "inference_contract_sha256": contract_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": frozen_manifest_sha256,
            "checkpoint_path": str(metadata["checkpoint_path"]),
            "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
            "gt_artifacts_exported": False,
            "adapted_from_frozen_hifi_output": True,
            "frozen_source_metadata_path": str(metadata_path),
            "frozen_source_metadata_sha256": sha256_file(metadata_path),
            "ready": True,
        }
        atomic_json(
            rows_dir / f"{sample_id}.json", row, tmp_root=tmp_root
        )
        rows.append(row)
        if (sample_index + 1) % 1000 == 0:
            print(f"indexed {sample_index + 1}/{len(records)}", flush=True)
    if len(checkpoint_hashes) != 1:
        raise ValueError(f"frozen HiFi bundle has mixed checkpoints: {checkpoint_hashes}")
    manifest_lines = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    )
    output_manifest = output_root / "manifest.jsonl"
    atomic_text(output_manifest, manifest_lines, tmp_root=tmp_root)
    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "split": args.split,
        "samples": len(rows),
        "source_root": str(source_root),
        "source_root_modified": False,
        "frozen_manifest": str(manifest_path),
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "output_manifest": str(output_manifest.resolve()),
        "output_manifest_sha256": sha256_file(output_manifest),
        "gt_artifacts_exported": False,
        "inference_rerun": False,
        "adapter_only": True,
    }
    atomic_json(output_root / "summary.json", summary, tmp_root=tmp_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
