#!/usr/bin/env python3
"""Render prediction-only SAM 3 prompts without loading model weights."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.sam3_prompt_builder import build_visual_prompt  # noqa: E402
from src.segmentation.sam3_refiner import load_refinement_input  # noqa: E402
from src.segmentation.sam3_serialization import save_strict_json, sha256_file, write_jsonl  # noqa: E402
from src.segmentation.sam3_visualization import save_prompt_visualization  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=REPO_ROOT / "outputs" / "sam3_refinement_inputs"
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "outputs" / "sam3_prompt_previews"
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "sam3_refinement.yaml"
    )
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prompt previews: {output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["prompt"]
    rows = [
        json.loads(line)
        for line in (args.input_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    temporary = output.with_name(output.name + ".incomplete")
    temporary.mkdir(parents=True)
    manifest: list[dict] = []
    try:
        for row in rows:
            sample_id = str(row["sample_id"])
            loaded = load_refinement_input(args.input_root / sample_id)
            prompt = build_visual_prompt(loaded["coarse_probability"], config)
            sample = temporary / sample_id
            sample.mkdir()
            visualization = save_prompt_visualization(
                loaded["rgb"], prompt, sample / "prompt_visualization.png"
            )
            metadata = save_strict_json(sample / "prompt_metadata.json", prompt.to_dict())
            manifest.append(
                {
                    "sample_id": sample_id,
                    "prompt_strategy": prompt.strategy,
                    "positive_point_count": len(prompt.positive_points_xy),
                    "negative_point_count": len(prompt.negative_points_xy),
                    "visualization_sha256": sha256_file(visualization),
                    "metadata_sha256": sha256_file(metadata),
                    "sam3_inference_ran": False,
                }
            )
        write_jsonl(temporary / "manifest.jsonl", manifest)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": "PROMPTS_RENDERED", "samples": len(manifest), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

