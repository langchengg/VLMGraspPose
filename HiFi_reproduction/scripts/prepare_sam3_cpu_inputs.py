#!/usr/bin/env python3
"""Build a prediction-only SAM 3 CPU input manifest without copying protected data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.sam3_cpu_serialization import (  # noqa: E402
    assert_no_ground_truth_leakage,
    atomic_output_directory,
    sha256_file,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "hifics_ocidvlg_20260711_112921"
        / "anygrasp_input_predicted_mask",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sam3_cpu_inputs",
    )
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--sample-limit", type=int)
    return parser.parse_args()


def _source_rows(root: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or len(rows) != len({row["sample_id"] for row in rows}):
        raise ValueError("source manifest must contain unique samples")
    return rows


def _requested_ids(args: argparse.Namespace, rows: list[dict]) -> list[str]:
    if args.sample_ids_file and args.sample_limit:
        raise ValueError("choose --sample-ids-file or --sample-limit")
    if args.sample_ids_file:
        result = [
            line.strip()
            for line in args.sample_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.sample_limit:
        if args.sample_limit <= 0:
            raise ValueError("--sample-limit must be positive")
        result = [str(row["sample_id"]) for row in rows[: args.sample_limit]]
    else:
        result = [str(row["sample_id"]) for row in rows]
    if not result or len(result) != len(set(result)):
        raise ValueError("requested sample IDs must be unique and non-empty")
    return result


def build_manifest(args: argparse.Namespace) -> list[dict]:
    source_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_rows = _source_rows(source_root)
    by_id = {str(row["sample_id"]): row for row in source_rows}
    requested = _requested_ids(args, source_rows)
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise KeyError(f"unknown sample IDs: {unknown}")
    output: list[dict] = []
    for sample_id in requested:
        bundle = source_root / sample_id
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        paths = {
            "rgb_path": bundle / "color.png",
            "probability_path": bundle / "target_probability.npy",
            "coarse_mask_path": bundle / "target_mask.png",
            "depth_path": bundle / "depth.png",
            "intrinsics_path": bundle / "intrinsics.json",
            "language_path": bundle / "language.txt",
            "metadata_path": bundle / "metadata.json",
        }
        if any(path.is_symlink() or not path.is_file() for path in paths.values()):
            raise ValueError(f"missing or unsafe file in {bundle}")
        rgb = np.asarray(Image.open(paths["rgb_path"]).convert("RGB"))
        probability = np.load(paths["probability_path"], allow_pickle=False)
        mask = np.asarray(Image.open(paths["coarse_mask_path"])) > 0
        if probability.dtype != np.float32 or not np.isfinite(probability).all():
            raise ValueError(f"invalid probability map: {sample_id}")
        if rgb.shape[:2] != probability.shape or probability.shape != mask.shape:
            raise ValueError(f"unaligned input shapes: {sample_id}")
        threshold = float(
            metadata.get(
                "prediction_threshold",
                metadata.get("mask_threshold", 0.15000000000000002),
            )
        )
        if not np.array_equal(mask, probability >= threshold):
            raise ValueError(f"coarse mask threshold mismatch: {sample_id}")
        row = {
            "schema_version": 1,
            "sample_id": sample_id,
            "scene_id": str(metadata["scene_id"]),
            "query": (bundle / "language.txt").read_text(encoding="utf-8").strip(),
            "bundle_path": str(bundle),
            **{key: str(value) for key, value in paths.items()},
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "coarse_mask_threshold": threshold,
            "checksums": {key: sha256_file(path) for key, path in paths.items()},
            "inference_inputs": [
                "rgb_path",
                "probability_path",
                "coarse_mask_path",
                "language_path",
            ],
            "ground_truth_or_annotations_included": False,
        }
        assert_no_ground_truth_leakage(
            {
                key: value
                for key, value in row.items()
                if key != "ground_truth_or_annotations_included"
            },
            context="SAM 3 CPU input manifest",
        )
        output.append(row)
    with atomic_output_directory(output_root) as temporary:
        write_jsonl(temporary / "manifest.jsonl", output)
    return output


def main() -> int:
    args = parse_args()
    rows = build_manifest(args)
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "samples": len(rows),
                "output": str(args.output_root.expanduser().resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
