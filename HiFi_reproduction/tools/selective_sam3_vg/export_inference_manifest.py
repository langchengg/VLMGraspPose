#!/usr/bin/env python3
"""Export a compact prediction manifest to a minimal GT-free inference list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.sam3_cpu_serialization import atomic_write_jsonl  # noqa: E402
from segmentation.selective_sam3_vg.io import load_compact_manifest  # noqa: E402


CHECKPOINT_SHA256 = "b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601"
SPLIT_SHA256 = {
    "val": "573c6ecd9ed9963eda525162279836b7649d163d83c57f164598604579b8b84a",
    "test": "915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409",
}
COUNTS = {"val": 3778, "test": 7675}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = load_compact_manifest(
        args.input,
        expected_split=args.split,
        expected_count=COUNTS[args.split],
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_manifest_sha256=SPLIT_SHA256[args.split],
    )
    rows = [
        {
            "sample_order": index,
            "sample_index": item.sample_index,
            "sample_id": item.sample_id,
            "question_index": item.question_index,
            "scene_id": item.scene_id,
            "query": item.query,
            "rgb_path": str(item.rgb_path),
            "depth_path": str(item.depth_path),
            "probability_path": str(item.probability_path),
            "native_mask_path": str(item.native_mask_path),
            "checkpoint_sha256": item.checkpoint_sha256,
            "split_manifest_sha256": item.manifest_sha256,
        }
        for index, item in enumerate(predictions)
    ]
    forbidden = {"gt", "ground_truth", "iou", "threshold_success", "label"}
    if any(any(token in key.lower() for token in forbidden) for row in rows for key in row):
        raise RuntimeError("forbidden GT/evaluation field in formal inference manifest")
    atomic_write_jsonl(args.output, rows)
    print(json.dumps({"status": "COMPLETED", "split": args.split, "rows": len(rows), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
