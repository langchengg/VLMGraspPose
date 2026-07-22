#!/usr/bin/env python3
"""Verify a portable GPU SAM 3 result bundle before importing it on the Mac host."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.sam3_serialization import (  # noqa: E402
    sha256_file,
    verify_file_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPO_ROOT / "outputs" / "sam3_refined_masks",
    )
    parser.add_argument("--require-real-inference", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite imported outputs: {destination}")
    run = json.loads((source / "run_metadata.json").read_text(encoding="utf-8"))
    if args.require_real_inference and run.get("experiment_status") != "real_sam3_inference_completed":
        raise RuntimeError("bundle contains no completed real SAM 3 inference")
    rows = [
        json.loads(line)
        for line in (source / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    for row in rows:
        sample_id = str(row["sample_id"])
        sample = source / sample_id
        metadata_path = sample / "refinement_metadata.json"
        if sha256_file(metadata_path) != row["metadata_sha256"]:
            raise ValueError(f"metadata checksum mismatch for {sample_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        verify_file_manifest(sample, metadata["outputs"])
        rgb = np.asarray(Image.open(sample / "rgb.png").convert("RGB"))
        refined = np.asarray(Image.open(sample / "refined_mask.png"))
        probability = np.load(sample / "refined_probability.npy", allow_pickle=False)
        if rgb.shape[:2] != refined.shape or refined.shape != probability.shape:
            raise ValueError(f"unaligned imported outputs for {sample_id}")
        if probability.dtype != np.float32 or not np.all(np.isfinite(probability)):
            raise ValueError(f"invalid refined probability for {sample_id}")
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"remove stale incomplete import first: {temporary}")
    shutil.copytree(source, temporary, symlinks=False)
    temporary.rename(destination)
    print(json.dumps({"status": "IMPORTED", "samples": len(rows), "destination": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

