#!/usr/bin/env python3
"""Verify PCS prompt micro-batching against strict batch-size-one inference."""

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

from src.segmentation.query_semantics import parse_query  # noqa: E402
from src.segmentation.sam3_embedding_cache import Sam3EmbeddingCache  # noqa: E402
from src.segmentation.sam3_proposal_generator import build_text_prompt_specs  # noqa: E402
from src.segmentation.sam3_text_proposals import (  # noqa: E402
    OfficialSam3TextProposalGenerator,
    TextPromptSpec,
)
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_compact_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/sam3_proposal_bank_p90_v1/proposal_generation.yaml",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Candidate micro-batch size; defaults to prompt-count for compatibility.",
    )
    parser.add_argument("--frame-count", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/validation_pilot/"
        "text_microbatch_equivalence.json",
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


def _candidate_key(candidate: Any) -> tuple[str, str, str]:
    return (
        str(candidate.source_family),
        str(candidate.source_variant),
        str(candidate.canonical_text_prompt),
    )


def _maximum_abs(left: Any, right: Any) -> float:
    values = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    return float(values.max(initial=0.0))


def _compare_candidates(single: list[Any], batched: list[Any]) -> dict[str, Any]:
    left = {_candidate_key(candidate): candidate for candidate in single}
    right = {_candidate_key(candidate): candidate for candidate in batched}
    if len(left) != len(single) or len(right) != len(batched):
        raise RuntimeError("benchmark candidate comparison key is not unique")
    shared = sorted(set(left) & set(right))
    result: dict[str, Any] = {
        "single_candidate_count": len(single),
        "batched_candidate_count": len(batched),
        "shared_candidate_count": len(shared),
        "missing_from_batched": len(set(left) - set(right)),
        "extra_in_batched": len(set(right) - set(left)),
        "mask_mismatches": 0,
        "candidate_id_mismatches": 0,
        "maximum_probability_abs_difference": 0.0,
        "maximum_score_abs_difference": 0.0,
        "maximum_presence_abs_difference": 0.0,
        "maximum_box_abs_difference_px": 0.0,
    }
    for key in shared:
        one = left[key]
        many = right[key]
        result["mask_mismatches"] += int(not np.array_equal(one.mask, many.mask))
        result["candidate_id_mismatches"] += int(one.candidate_id != many.candidate_id)
        result["maximum_probability_abs_difference"] = max(
            result["maximum_probability_abs_difference"],
            _maximum_abs(one.probability, many.probability),
        )
        result["maximum_score_abs_difference"] = max(
            result["maximum_score_abs_difference"],
            abs(float(one.sam_score) - float(many.sam_score)),
        )
        if one.presence_score is not None and many.presence_score is not None:
            result["maximum_presence_abs_difference"] = max(
                result["maximum_presence_abs_difference"],
                abs(float(one.presence_score) - float(many.presence_score)),
            )
        result["maximum_box_abs_difference_px"] = max(
            result["maximum_box_abs_difference_px"],
            _maximum_abs(one.box_xyxy, many.box_xyxy),
        )
    return result


def main() -> int:
    args = parse_args()
    if args.prompt_count < 2:
        raise ValueError("prompt-count must be at least two")
    candidate_batch_size = int(args.batch_size or args.prompt_count)
    if not 1 <= candidate_batch_size <= 16:
        raise ValueError("batch-size must be within [1,16]")
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["device"] != "cpu" or config["dtype"] != "float32":
        raise ValueError("benchmark requires the locked CPU/float32 path")
    manifest_path = PROJECT_ROOT / (
        "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/"
        f"{args.split}/manifest.jsonl"
    )
    rows = load_compact_manifest(manifest_path, expected_split=args.split)
    rows_by_frame: dict[str, list[Any]] = {}
    for row in rows:
        rows_by_frame.setdefault(row.scene_id, []).append(row)
    if args.frame_count < 1:
        raise ValueError("frame-count must be positive")
    chosen_frames: list[tuple[list[Any], list[str]]] = []
    for frame_rows in rows_by_frame.values():
        texts = sorted(
            {
                spec.text
                for row in frame_rows
                for spec in build_text_prompt_specs(parse_query(row.query))
            },
            key=lambda value: (value.lower(), value),
        )
        if len(texts) >= args.prompt_count:
            chosen_frames.append((frame_rows, texts[: args.prompt_count]))
            if len(chosen_frames) >= args.frame_count:
                break
    if len(chosen_frames) != args.frame_count:
        raise RuntimeError("no frame contains enough distinct deterministic prompts")
    cache = Sam3EmbeddingCache(
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/cache/microbatch_benchmark"
    )
    generator = OfficialSam3TextProposalGenerator(
        (PROJECT_ROOT / config["model"]["local_path"]).resolve(),
        revision=config["model"]["revision"],
        cache=cache,
        processor_sha256=config["model"]["pcs_processor_sha256"],
        num_threads=int(args.num_threads),
        micro_batch_size=1,
    )
    common = {
        "instance_thresholds": tuple(
            float(value) for value in config["pcs"]["instance_thresholds"]
        ),
        "mask_thresholds": tuple(
            float(value) for value in config["pcs"]["mask_thresholds"]
        ),
        "maximum_instances_per_prompt": int(
            config["pcs"]["maximum_instances_per_prompt"]
        ),
    }
    thresholds = {
        "maximum_probability_abs_difference": 1e-5,
        "maximum_score_abs_difference": 1e-6,
        "maximum_presence_abs_difference": 1e-6,
        "maximum_box_abs_difference_px": 1e-4,
    }
    per_frame: list[dict[str, Any]] = []
    for frame_index, (frame_rows, chosen_texts) in enumerate(chosen_frames):
        first = frame_rows[0]
        image = Image.open(first.rgb_path).convert("RGB")
        rgb_sha = str(first.raw["source_rgb_sha256"])
        prompts = [
            TextPromptSpec(
                prompt_id=f"MICROBATCH_{index:02d}",
                source_family="MICROBATCH_BENCHMARK",
                source_variant=f"prompt={index:02d}",
                text=text,
            )
            for index, text in enumerate(chosen_texts)
        ]
        generator.micro_batch_size = 1
        single, single_runtime = generator.generate(
            f"microbatch_benchmark_{frame_index}",
            image,
            rgb_sha,
            prompts,
            **common,
        )
        generator.micro_batch_size = candidate_batch_size
        batched, batched_runtime = generator.generate(
            f"microbatch_benchmark_{frame_index}",
            image,
            rgb_sha,
            prompts,
            **common,
        )
        comparison = _compare_candidates(single, batched)
        per_frame.append(
            {
                "scene_id": first.scene_id,
                "rgb_sha256": rgb_sha,
                "prompt_texts": chosen_texts,
                "single_runtime": single_runtime,
                "batched_runtime": batched_runtime,
                **comparison,
            }
        )
    aggregate = {
        "single_candidate_count": sum(x["single_candidate_count"] for x in per_frame),
        "batched_candidate_count": sum(x["batched_candidate_count"] for x in per_frame),
        "shared_candidate_count": sum(x["shared_candidate_count"] for x in per_frame),
        "missing_from_batched": sum(x["missing_from_batched"] for x in per_frame),
        "extra_in_batched": sum(x["extra_in_batched"] for x in per_frame),
        "mask_mismatches": sum(x["mask_mismatches"] for x in per_frame),
        "candidate_id_mismatches": sum(
            x["candidate_id_mismatches"] for x in per_frame
        ),
        "maximum_probability_abs_difference": max(
            x["maximum_probability_abs_difference"] for x in per_frame
        ),
        "maximum_score_abs_difference": max(
            x["maximum_score_abs_difference"] for x in per_frame
        ),
        "maximum_presence_abs_difference": max(
            x["maximum_presence_abs_difference"] for x in per_frame
        ),
        "maximum_box_abs_difference_px": max(
            x["maximum_box_abs_difference_px"] for x in per_frame
        ),
    }
    passed = bool(
        aggregate["missing_from_batched"] == 0
        and aggregate["extra_in_batched"] == 0
        and aggregate["mask_mismatches"] == 0
        and aggregate["candidate_id_mismatches"] == 0
        and aggregate["maximum_probability_abs_difference"]
        <= thresholds["maximum_probability_abs_difference"]
        and aggregate["maximum_score_abs_difference"]
        <= thresholds["maximum_score_abs_difference"]
        and aggregate["maximum_presence_abs_difference"]
        <= thresholds["maximum_presence_abs_difference"]
        and aggregate["maximum_box_abs_difference_px"]
        <= thresholds["maximum_box_abs_difference_px"]
    )
    single_decoder_seconds = sum(
        float(x["single_runtime"]["decoder_seconds"]) for x in per_frame
    )
    batched_decoder_seconds = sum(
        float(x["batched_runtime"]["decoder_seconds"]) for x in per_frame
    )
    result = {
        "status": "PASS_OPERATIONALLY_EQUIVALENT" if passed else "FAIL",
        "safe_micro_batch_size": candidate_batch_size if passed else 1,
        "prompt_count_per_frame": int(args.prompt_count),
        "split": args.split,
        "frame_count": len(per_frame),
        **aggregate,
        "acceptance_thresholds": thresholds,
        "single_decoder_seconds": single_decoder_seconds,
        "batched_decoder_seconds": batched_decoder_seconds,
        "decoder_speedup_ratio": single_decoder_seconds
        / max(batched_decoder_seconds, 1e-12),
        "per_frame": per_frame,
        "uses_ground_truth": False,
        "device": "cpu",
        "dtype": "float32",
        "model_revision": config["model"]["revision"],
    }
    _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
