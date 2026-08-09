#!/usr/bin/env python3
"""Reuse locked formal SAM hypotheses inside the later oracle diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.sam3_cpu_serialization import atomic_output_directory  # noqa: E402


def _link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--box-expansion", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    reused = 0
    for row in rows:
        source = ROOT / "outputs/selective_sam3_vg/formal_test_masks" / row["sample_id"]
        destination = args.output_root / row["sample_id"] / args.family
        if destination.exists() or not (source / "all_sam_hypotheses.npz").is_file():
            continue
        with atomic_output_directory(destination) as temporary:
            for name in (
                "all_sam_hypotheses.npz", "candidate_features.csv", "pre_sam_features.json",
                "prompt_metadata.json", "timing.json", "memory.json",
            ):
                if (source / name).is_file():
                    _link(source / name, temporary / name)
            provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
            reused_provenance = {
                "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                "query": row["query"], "prompt_family": args.family,
                "box_expansion_fraction": float(args.box_expansion),
                "model_revision": provenance["sam_model_revision"],
                "model_backend": "tracker", "config_sha256": provenance["configuration_hash"],
                "prediction_probability_path": row["probability_path"],
                "prediction_mask_path": row["native_mask_path"],
                "ground_truth_loaded": False, "oracle": True,
                "reused_from_locked_formal_inference": True,
            }
            (temporary / "provenance.json").write_text(
                json.dumps(reused_provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (temporary / "status.json").write_text(
                json.dumps(
                    {"status": "COMPLETED", "hypothesis_count": 3, "oracle": True, "reused": True},
                    indent=2, sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
        reused += 1
    print(json.dumps({"status": "COMPLETED", "oracle": True, "reused_configurations": reused}, indent=2))


if __name__ == "__main__":
    main()
