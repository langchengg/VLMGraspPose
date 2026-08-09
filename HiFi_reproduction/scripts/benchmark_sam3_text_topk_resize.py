#!/usr/bin/env python3
"""Verify that PCS score selection before resize preserves retained masks."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/validation_pilot/"
        "text_topk_resize_equivalence.json",
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
    if args.prompt_count < 1:
        raise ValueError("prompt-count must be positive")
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["device"] != "cpu" or config["dtype"] != "float32":
        raise ValueError("benchmark requires CPU/float32")
    rows = load_compact_manifest(
        PROJECT_ROOT
        / "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs"
        / args.split
        / "manifest.jsonl",
        expected_split=args.split,
    )
    selected = None
    prompt_texts: list[str] = []
    for row in rows:
        texts = sorted(
            {spec.text for spec in build_text_prompt_specs(parse_query(row.query))},
            key=lambda value: (value.lower(), value),
        )
        if len(texts) >= args.prompt_count:
            selected = row
            prompt_texts = texts[: args.prompt_count]
            break
    if selected is None:
        raise RuntimeError("no sample contains enough deterministic prompts")

    import torch

    cache = Sam3EmbeddingCache(
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/cache/topk_resize_benchmark"
    )
    generator = OfficialSam3TextProposalGenerator(
        (PROJECT_ROOT / config["model"]["local_path"]).resolve(),
        revision=config["model"]["revision"],
        cache=cache,
        processor_sha256=config["model"]["pcs_processor_sha256"],
        num_threads=int(args.num_threads),
        micro_batch_size=int(args.prompt_count),
    )
    image = Image.open(selected.rgb_path).convert("RGB")
    width, height = image.size
    vision, cache_hit, vision_seconds = generator.vision_features(
        image, str(selected.raw["source_rgb_sha256"])
    )
    text_inputs = generator.processor(
        text=prompt_texts, return_tensors="pt", padding=True
    )
    repeated = generator._repeat_vision(vision, len(prompt_texts))
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = generator.model(
            vision_embeds=repeated,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )
    decoder_seconds = time.perf_counter() - started
    scores = outputs.pred_logits.sigmoid()
    presence = (
        outputs.presence_logits.sigmoid()
        if outputs.presence_logits is not None
        else None
    )
    if presence is not None:
        scores = scores * presence
    maximum = int(config["pcs"]["maximum_instances_per_prompt"])

    started = time.perf_counter()
    legacy_probabilities = torch.nn.functional.interpolate(
        outputs.pred_masks.sigmoid(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    legacy_resize_seconds = time.perf_counter() - started
    started = time.perf_counter()
    order, optimized_probabilities, selected_boxes = (
        generator._select_and_resize_outputs(
            outputs,
            scores,
            maximum_instances_per_prompt=maximum,
            output_size=(height, width),
        )
    )
    optimized_resize_seconds = time.perf_counter() - started

    probability_maximum_abs_difference = 0.0
    box_maximum_abs_difference = 0.0
    mask_mismatches = 0
    mask_comparisons = 0
    thresholds = tuple(float(value) for value in config["pcs"]["mask_thresholds"])
    for batch_index in range(scores.shape[0]):
        for rank, raw_index in enumerate(order[batch_index].tolist()):
            legacy = legacy_probabilities[batch_index, raw_index]
            optimized = optimized_probabilities[batch_index, rank]
            probability_maximum_abs_difference = max(
                probability_maximum_abs_difference,
                float((legacy - optimized).abs().max()),
            )
            box_maximum_abs_difference = max(
                box_maximum_abs_difference,
                float(
                    (
                        outputs.pred_boxes[batch_index, raw_index]
                        - selected_boxes[batch_index, rank]
                    )
                    .abs()
                    .max()
                ),
            )
            for threshold in thresholds:
                mask_comparisons += 1
                mask_mismatches += int(
                    not torch.equal(legacy > threshold, optimized > threshold)
                )
    raw_masks = int(outputs.pred_masks.shape[0] * outputs.pred_masks.shape[1])
    retained_masks = int(order.numel())
    passed = bool(
        mask_mismatches == 0
        and box_maximum_abs_difference == 0.0
        and probability_maximum_abs_difference <= 1e-5
    )
    result = {
        "status": "PASS_OPERATIONALLY_EQUIVALENT" if passed else "FAIL",
        "optimization": "stable score top-k before bilinear mask resize",
        "sample_id": selected.sample_id,
        "split": args.split,
        "prompt_count": len(prompt_texts),
        "prompt_texts": prompt_texts,
        "raw_masks_resized_legacy": raw_masks,
        "retained_masks_resized_optimized": retained_masks,
        "resize_work_reduction_factor": raw_masks / retained_masks,
        "legacy_resize_seconds": legacy_resize_seconds,
        "optimized_resize_seconds": optimized_resize_seconds,
        "resize_speedup": legacy_resize_seconds / optimized_resize_seconds,
        "mask_thresholds": list(thresholds),
        "mask_comparisons": mask_comparisons,
        "mask_mismatches": mask_mismatches,
        "maximum_probability_abs_difference": probability_maximum_abs_difference,
        "maximum_box_abs_difference": box_maximum_abs_difference,
        "decoder_seconds": decoder_seconds,
        "vision_cache_hit": cache_hit,
        "vision_seconds": vision_seconds,
        "device": config["device"],
        "dtype": config["dtype"],
        "model_revision": config["model"]["revision"],
        "transformers_version": __import__("transformers").__version__,
        "candidate_semantics_changed": False,
    }
    _atomic_json(args.output.expanduser().resolve(), result)
    del legacy_probabilities, optimized_probabilities, outputs, repeated
    gc.collect()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
