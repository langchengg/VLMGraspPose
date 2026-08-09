#!/usr/bin/env python3
"""Build label-only mask diagnostics for representative validation-pilot selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.ocid_annotations import mask_iou  # noqa: E402
from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402


def _load_probability(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return np.asarray(loaded, dtype=np.float32)
    with loaded as archive:
        if "probability" not in archive.files:
            raise ValueError(f"probability archive has no probability key: {path}")
        return np.asarray(archive["probability"], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--gt-manifest", type=Path, required=True)
    parser.add_argument("--hifi-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    prediction_root = args.prediction_root.expanduser().resolve()
    gt_manifest_path = args.gt_manifest.expanduser().resolve()
    hifi_root = args.hifi_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    gt_rows = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(gt_rows, list) or not gt_rows:
        raise ValueError("GT manifest must be a non-empty list")
    gt_by_index = {int(row["num"]): row for row in gt_rows}
    if len(gt_by_index) != len(gt_rows):
        raise ValueError("GT manifest contains duplicate sample indices")

    records: list[dict] = []
    source_hashes: list[dict] = []
    row_paths = sorted((prediction_root / "rows").glob("*.json"))
    if not row_paths:
        raise ValueError("prediction root has no compact row JSON")
    for row_path in row_paths:
        prediction = json.loads(row_path.read_text(encoding="utf-8"))
        sample_index = int(prediction["sample_index"])
        gt = gt_by_index.get(sample_index)
        if (
            gt is None
            or int(gt["question_index"]) != int(prediction["question_index"])
            or str(gt["text"]) != str(prediction["query"])
            or str(gt["scene_id"]) != str(prediction["scene_id"])
        ):
            raise ValueError(
                f"prediction/GT validation join mismatch: {prediction['sample_id']}"
            )
        raw_gt_path = Path(str(gt["mask_path"])).expanduser()
        gt_path = (
            raw_gt_path.resolve()
            if raw_gt_path.is_absolute()
            else (hifi_root / raw_gt_path).resolve()
        )
        gt_mask = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8) > 0
        probability_path = Path(str(prediction["probability_path"])).resolve()
        native_mask_path = Path(str(prediction["native_mask_path"])).resolve()
        predicted = (
            np.asarray(Image.open(native_mask_path).convert("L"), dtype=np.uint8)
            > 0
        )
        if (
            predicted.shape != gt_mask.shape
            or predicted.ndim != 2
            or prediction.get("native_mask_transform")
            != "nearest_resize_of_binary_352_mask_to_640x480"
            or sha256_file(native_mask_path)
            != prediction.get("native_mask_sha256")
        ):
            raise ValueError(
                f"prediction/GT mask shape or value mismatch: {prediction['sample_id']}"
            )
        records.append(
            {
                "sample_id": str(prediction["sample_id"]),
                "hifi_mask_iou": mask_iou(predicted, gt_mask),
                "target_area_px": int(np.count_nonzero(gt_mask)),
            }
        )
        source_hashes.append(
            {
                "sample_id": str(prediction["sample_id"]),
                "prediction_row_sha256": sha256_file(row_path),
                "probability_sha256": sha256_file(probability_path),
                "native_mask_sha256": sha256_file(native_mask_path),
                "gt_mask_sha256": sha256_file(gt_path),
            }
        )
    frame = pd.DataFrame(records).sort_values("sample_id", kind="mergesort")
    if frame["sample_id"].duplicated().any():
        raise ValueError("diagnostics contain duplicate sample IDs")
    output_path = output_root / "pilot_diagnostics.parquet"
    temporary = output_root / f".pilot_diagnostics.{os.getpid()}.tmp.parquet"
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output_path)
    manifest = {
        "schema_version": 1,
        "label_only_offline_selection_artifact": True,
        "inference_or_vlm_prompt_allowed": False,
        "sample_count": len(frame),
        "columns": list(frame.columns),
        "prediction_root": str(prediction_root),
        "gt_manifest": str(gt_manifest_path),
        "gt_manifest_sha256": sha256_file(gt_manifest_path),
        "per_sample_sources": source_hashes,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sample_count": len(frame),
                "hifi_mask_iou_min": float(frame["hifi_mask_iou"].min()),
                "hifi_mask_iou_median": float(frame["hifi_mask_iou"].median()),
                "hifi_mask_iou_max": float(frame["hifi_mask_iou"].max()),
                "target_area_px_min": int(frame["target_area_px"].min()),
                "target_area_px_median": float(frame["target_area_px"].median()),
                "target_area_px_max": int(frame["target_area_px"].max()),
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
