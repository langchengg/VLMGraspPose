#!/usr/bin/env python3
"""Verify automatic SAM 3 proposals when sharing the official Tracker model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.sam3_automatic_proposals import (  # noqa: E402
    OfficialSam3AutomaticProposalGenerator,
)
from src.segmentation.sam3_embedding_cache import Sam3EmbeddingCache  # noqa: E402
from src.segmentation.sam3_visual_proposals import (  # noqa: E402
    OfficialSam3VisualProposalGenerator,
)
from src.segmentation.selective_sam3_vg.io import load_compact_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/sam3_proposal_bank_p90_v1/proposal_generation.yaml",
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/validation_pilot/"
        "automatic_shared_model_equivalence.json",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    if config["device"] != "cpu" or config["dtype"] != "float32":
        raise ValueError("benchmark requires the locked CPU/float32 path")
    rows = load_compact_manifest(
        PROJECT_ROOT
        / "runs/modular_reranking_repeatedfilm_v1_20260729_203147/"
        "compact_inputs/val/manifest.jsonl",
        expected_split="val",
    )
    row = rows[0]
    image = Image.open(row.rgb_path).convert("RGB")
    rgb_sha = str(row.raw["source_rgb_sha256"])
    model_path = (PROJECT_ROOT / config["model"]["local_path"]).resolve()
    visual = OfficialSam3VisualProposalGenerator(
        model_path,
        revision=config["model"]["revision"],
        cache=Sam3EmbeddingCache(
            PROJECT_ROOT
            / "outputs/sam3_proposal_bank_p90_v1/cache/automatic_sharing_benchmark"
        ),
        processor_sha256=config["model"]["tracker_processor_sha256"],
        num_threads=int(args.num_threads),
    )
    shared = OfficialSam3AutomaticProposalGenerator(
        model_path,
        revision=config["model"]["revision"],
        num_threads=int(args.num_threads),
        shared_tracker_model=visual.model,
        shared_tracker_processor=visual.processor,
    )
    independent = OfficialSam3AutomaticProposalGenerator(
        model_path,
        revision=config["model"]["revision"],
        num_threads=int(args.num_threads),
    )
    automatic = config["automatic"]
    kwargs = {
        "points_per_side": int(automatic["points_per_side"]),
        "points_per_batch": int(automatic["points_per_batch"]),
        "score_threshold": float(automatic["score_threshold"]),
        "stability_threshold": float(automatic["stability_threshold"]),
        "nms_threshold": float(automatic["nms_threshold"]),
        "minimum_mask_area": int(automatic["minimum_mask_area_px"]),
        "maximum_proposals": int(automatic["maximum_proposals_per_image"]),
    }
    reference, reference_runtime = independent.generate(
        "automatic_sharing_benchmark", image, rgb_sha, **kwargs
    )
    reused, reused_runtime = shared.generate(
        "automatic_sharing_benchmark", image, rgb_sha, **kwargs
    )
    left = {candidate.source_variant: candidate for candidate in reference}
    right = {candidate.source_variant: candidate for candidate in reused}
    shared_keys = sorted(set(left) & set(right))
    mask_mismatches = sum(
        not np.array_equal(left[key].mask, right[key].mask) for key in shared_keys
    )
    candidate_id_mismatches = sum(
        left[key].candidate_id != right[key].candidate_id for key in shared_keys
    )
    max_score_difference = max(
        (
            abs(float(left[key].sam_score) - float(right[key].sam_score))
            for key in shared_keys
        ),
        default=0.0,
    )
    max_box_difference = max(
        (
            float(
                np.abs(
                    np.asarray(left[key].box_xyxy, dtype=np.float64)
                    - np.asarray(right[key].box_xyxy, dtype=np.float64)
                ).max(initial=0.0)
            )
            for key in shared_keys
        ),
        default=0.0,
    )
    passed = bool(
        set(left) == set(right)
        and mask_mismatches == 0
        and candidate_id_mismatches == 0
        and max_score_difference <= 1e-7
        and max_box_difference <= 1e-6
        and shared.shared_tracker_model
        and not independent.shared_tracker_model
        and shared.model is visual.model
        and independent.model is not visual.model
    )
    result = {
        "status": "PASS_NUMERICALLY_EQUIVALENT" if passed else "FAIL",
        "independent_candidate_count": len(reference),
        "shared_candidate_count": len(reused),
        "matching_candidate_count": len(shared_keys),
        "missing_from_shared": len(set(left) - set(right)),
        "extra_in_shared": len(set(right) - set(left)),
        "mask_mismatches": int(mask_mismatches),
        "candidate_id_mismatches": int(candidate_id_mismatches),
        "maximum_score_abs_difference": max_score_difference,
        "maximum_box_abs_difference_px": max_box_difference,
        "reference_runtime": reference_runtime,
        "shared_runtime": reused_runtime,
        "shared_model_object_identity": shared.model is visual.model,
        "independent_model_object_identity": independent.model is visual.model,
        "shared_model_class": (
            f"{shared.model.__class__.__module__}."
            f"{shared.model.__class__.__qualname__}"
        ),
        "independent_model_class": (
            f"{independent.model.__class__.__module__}."
            f"{independent.model.__class__.__qualname__}"
        ),
        "uses_ground_truth": False,
        "device": "cpu",
        "dtype": "float32",
        "model_revision": config["model"]["revision"],
        "official_interfaces": [
            "transformers.Sam3TrackerModel",
            "transformers.MaskGenerationPipeline",
        ],
    }
    _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
