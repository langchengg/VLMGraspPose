#!/usr/bin/env python3
"""Create portable, prediction-only SAM 3 inputs from frozen HiFi-CS bundles."""

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
    assert_no_ground_truth_leakage,
    file_manifest,
    save_strict_json,
    sha256_file,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "hifics_ocidvlg_20260711_112921"
        / "anygrasp_input_predicted_mask",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        help="Explicit ordered cohort; defaults to configs/sam3_ten_sample_ids.txt",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Use the first N frozen source-manifest rows instead of the default ten-sample cohort",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sam3_refinement_inputs",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.15000000000000002)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def _load_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("sample ID file must contain unique non-empty IDs")
    return ids


def _read_manifest(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("source manifest contains duplicate sample IDs")
    return by_id


def prepare_inputs(args: argparse.Namespace) -> list[dict]:
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite input bundle: {output_root}")
    temporary = output_root.with_name(output_root.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"remove stale incomplete bundle first: {temporary}")
    by_id = _read_manifest(source_root / "manifest.jsonl")
    if args.sample_ids_file is not None and args.sample_limit is not None:
        raise ValueError("choose either --sample-ids-file or --sample-limit")
    if args.sample_limit is not None:
        if args.sample_limit <= 0:
            raise ValueError("--sample-limit must be positive")
        sample_ids = list(by_id)[: args.sample_limit]
    else:
        ids_file = args.sample_ids_file or REPO_ROOT / "configs" / "sam3_ten_sample_ids.txt"
        sample_ids = _load_ids(ids_file)
    unknown = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if unknown:
        raise KeyError(f"sample IDs absent from source manifest: {unknown}")
    rows: list[dict] = []
    temporary.mkdir(parents=True)
    try:
        for sample_id in sample_ids:
            source = source_root / sample_id
            source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
            sample = temporary / sample_id
            sample.mkdir()
            copies = {
                "rgb": (source / "color.png", sample / "rgb.png"),
                "coarse_mask": (source / "target_mask.png", sample / "coarse_mask.png"),
                "coarse_probability": (
                    source / "target_probability.npy",
                    sample / "coarse_probability.npy",
                ),
            }
            for source_path, destination in copies.values():
                if source_path.is_symlink() or not source_path.is_file():
                    raise ValueError(f"missing or unsafe prediction input: {source_path}")
                shutil.copy2(source_path, destination)
            rgb = np.asarray(Image.open(sample / "rgb.png").convert("RGB"), dtype=np.uint8)
            mask = np.asarray(Image.open(sample / "coarse_mask.png"))
            probability = np.load(sample / "coarse_probability.npy", allow_pickle=False)
            if rgb.shape[:2] != mask.shape or mask.shape != probability.shape:
                raise ValueError(f"unaligned inputs for {sample_id}")
            if probability.dtype != np.float32 or not np.all(np.isfinite(probability)):
                raise ValueError(f"probability must be finite float32 for {sample_id}")
            if not np.array_equal(mask > 0, probability >= float(args.mask_threshold)):
                raise ValueError(f"binary mask does not match preserved probability threshold for {sample_id}")
            scene_id = str(source_metadata["scene_id"])
            image_id = scene_id.rsplit(",", 1)[-1]
            metadata = {
                "schema_version": 1,
                "sample_id": sample_id,
                "sample_index": int(source_metadata["sample_index"]),
                "question_index": int(source_metadata["question_index"]),
                "query": str(source_metadata["query"]),
                "split": str(args.split),
                "scene_id": scene_id,
                "image_id": image_id,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "mask_threshold": float(args.mask_threshold),
                "source_rgb": str(source_metadata["source_rgb"]),
                "source_depth": str(source_metadata["source_depth"]),
                "source_coarse_mask": str(source / "target_mask.png"),
                "source_coarse_probability": str(source / "target_probability.npy"),
                "source_hashes": {
                    "rgb": sha256_file(source / "color.png"),
                    "depth": sha256_file(source / "depth.png"),
                    "coarse_mask": sha256_file(source / "target_mask.png"),
                    "coarse_probability": sha256_file(source / "target_probability.npy"),
                },
                "input_files": file_manifest(
                    {name: destination for name, (_, destination) in copies.items()}
                ),
                "prompt_inputs": ["rgb", "coarse_mask", "coarse_probability"],
                "evaluation_artifacts_exported": False,
            }
            assert_no_ground_truth_leakage(metadata, context=f"SAM 3 input {sample_id}")
            save_strict_json(sample / "metadata.json", metadata)
            row = {
                "sample_id": sample_id,
                "sample_index": metadata["sample_index"],
                "question_index": metadata["question_index"],
                "query": metadata["query"],
                "scene_id": scene_id,
                "image_id": image_id,
                "bundle": str(output_root / sample_id),
                "ready": True,
                "prompt_inputs_prediction_only": True,
                "metadata_sha256": sha256_file(sample / "metadata.json"),
            }
            assert_no_ground_truth_leakage(row, context="SAM 3 input manifest")
            rows.append(row)
        write_jsonl(temporary / "manifest.jsonl", rows)
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return rows


def main() -> int:
    args = parse_args()
    rows = prepare_inputs(args)
    print(json.dumps({"status": "PREPARED", "samples": len(rows), "output": str(args.output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
