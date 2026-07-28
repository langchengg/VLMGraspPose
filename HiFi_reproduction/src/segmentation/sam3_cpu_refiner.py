"""One-sample no-GT SAM 3 CPU refinement with explicit safe fallback."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .sam3_cpu_model import Sam3CpuInferenceResult, TransformersSam3Cpu
from .sam3_cpu_serialization import (
    assert_no_ground_truth_leakage,
    atomic_output_directory,
    file_manifest,
    save_strict_json,
    sha256_file,
)
from .sam3_cpu_visualization import (
    save_candidate_grid,
    save_coarse_vs_refined,
    save_mask_overlay,
    save_prompt_visualization,
)
from .sam3_mask_selector import select_refined_mask
from .sam3_prompt_builder import build_visual_prompt


REQUIRED_INPUTS = (
    "color.png",
    "depth.png",
    "target_mask.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
    "metadata.json",
)


def load_hifics_bundle(sample_dir: Path | str) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    for name in REQUIRED_INPUTS:
        path = sample_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or unsafe HiFi-CS input: {path}")
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    assert_no_ground_truth_leakage(
        {
            "sample_id": metadata["sample_id"],
            "query": metadata["query"],
            "paths": list(REQUIRED_INPUTS),
        },
        context="SAM 3 CPU inference bundle",
    )
    rgb = np.asarray(Image.open(sample_dir / "color.png").convert("RGB"), dtype=np.uint8)
    coarse_mask = np.asarray(Image.open(sample_dir / "target_mask.png")) > 0
    coarse_probability = np.load(
        sample_dir / "target_probability.npy", allow_pickle=False
    )
    if coarse_probability.dtype != np.float32 or not np.isfinite(coarse_probability).all():
        raise ValueError("target_probability.npy must contain finite float32 values")
    if rgb.shape[:2] != coarse_mask.shape or coarse_mask.shape != coarse_probability.shape:
        raise ValueError("RGB, target mask, and target probability are not aligned")
    threshold = float(
        metadata.get(
            "prediction_threshold",
            metadata.get("mask_threshold", 0.15000000000000002),
        )
    )
    if not np.array_equal(coarse_mask, coarse_probability >= threshold):
        raise ValueError("stored coarse mask does not match its recorded probability threshold")
    return {
        "metadata": metadata,
        "rgb": rgb,
        "coarse_mask": coarse_mask,
        "coarse_probability": coarse_probability,
        "threshold": threshold,
        "query": (sample_dir / "language.txt").read_text(encoding="utf-8").strip(),
    }


def _empty_result(model: TransformersSam3Cpu) -> Sam3CpuInferenceResult:
    return Sam3CpuInferenceResult(
        masks=(),
        probabilities=(),
        qualities=(),
        backend=model.backend,
        model_class=model.runtime_metadata["model_class"],
        processor_class=model.runtime_metadata["processor_class"],
        timings={
            "preprocess_seconds": 0.0,
            "inference_seconds": 0.0,
            "postprocess_seconds": 0.0,
            "total_sample_seconds": 0.0,
            "model_load_seconds": float(model.load_time_seconds),
        },
        memory={
            "rss_before_inference_bytes": 0,
            "peak_rss_bytes": 0,
            "rss_after_inference_bytes": 0,
        },
        output_schema={"keys": [], "tensors": {}},
        runtime_metadata=dict(model.runtime_metadata),
    )


def refine_cpu_sample(
    sample_dir: Path | str,
    output_dir: Path | str,
    *,
    model: TransformersSam3Cpu,
    config: Mapping[str, Any],
    prompt_mode: str,
    save_all_hypotheses: bool = True,
    fallback_to_hifics: bool = True,
    short_text: str | None = None,
) -> dict[str, Any]:
    sample_dir, output_dir = Path(sample_dir), Path(output_dir)
    loaded = load_hifics_bundle(sample_dir)
    prompt_config = dict(config["prompt"])
    prompt_config["coarse_mask_threshold"] = loaded["threshold"]
    prompt_config["strategy"] = (
        "box_positive_negative_points"
        if prompt_mode == "box_positive_negative_points"
        else "box_positive_points"
    )
    prompt = build_visual_prompt(loaded["coarse_probability"], prompt_config)
    depth_m = None
    if bool(config["selector"]["depth_consistency"].get("enabled", False)):
        depth_m = (
            np.asarray(Image.open(sample_dir / "depth.png"), dtype=np.float32) / 1000.0
        )
    inference_error: str | None = None
    try:
        result = model.infer(
            Image.fromarray(loaded["rgb"], mode="RGB"),
            prompt,
            prompt_mode=prompt_mode,
            short_text=short_text,
        )
    except Exception as error:
        if not fallback_to_hifics:
            raise
        inference_error = f"{type(error).__name__}: {error}"
        result = _empty_result(model)
    accepted_source = (
        "sam3_tracker_cpu" if model.backend == "tracker" else "sam3_pcs_cpu"
    )
    selection = select_refined_mask(
        result.masks,
        result.probabilities,
        result.qualities,
        coarse_mask=loaded["coarse_mask"],
        coarse_probability=loaded["coarse_probability"],
        prompt=prompt,
        depth_m=depth_m,
        config=config["selector"],
        accepted_source=accepted_source,
    )
    if inference_error is not None:
        fallback_reason = f"model_inference_failed|{inference_error}"
    else:
        fallback_reason = selection.fallback_reason

    with atomic_output_directory(output_dir) as temporary:
        shutil.copy2(sample_dir / "color.png", temporary / "rgb.png")
        shutil.copy2(sample_dir / "target_mask.png", temporary / "coarse_mask.png")
        shutil.copy2(
            sample_dir / "target_probability.npy", temporary / "coarse_probability.npy"
        )
        save_prompt_visualization(loaded["rgb"], prompt, temporary / "prompt_overlay.png")
        save_strict_json(temporary / "prompt_metadata.json", prompt.to_dict())
        qualities = np.asarray(
            [np.nan if value is None else value for value in result.qualities],
            dtype=np.float32,
        )
        if save_all_hypotheses:
            np.savez_compressed(
                temporary / "sam3_candidate_masks.npz",
                candidate_id=np.asarray(
                    [item["candidate_id"] for item in selection.candidate_metrics]
                ),
                masks=np.asarray(result.masks, dtype=np.uint8),
                probabilities=np.asarray(result.probabilities, dtype=np.float32),
                model_quality=qualities,
            )
        fields = (
            list(selection.candidate_metrics[0])
            if selection.candidate_metrics
            else ["candidate_id"]
        )
        with (temporary / "sam3_candidate_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selection.candidate_metrics)
        Image.fromarray(selection.selected_mask.astype(np.uint8) * 255, mode="L").save(
            temporary / "refined_mask.png"
        )
        np.save(
            temporary / "refined_probability.npy",
            selection.selected_probability.astype(np.float32),
            allow_pickle=False,
        )
        save_candidate_grid(
            loaded["rgb"],
            result.masks,
            selection.candidate_metrics,
            temporary / "sam3_candidates_grid.png",
        )
        save_mask_overlay(
            loaded["rgb"],
            selection.selected_mask,
            temporary / "refined_overlay.png",
            title=f"Selected: {selection.selected_mask_source}",
        )
        save_coarse_vs_refined(
            loaded["rgb"],
            loaded["coarse_mask"],
            selection.selected_mask,
            temporary / "coarse_vs_refined.png",
        )
        save_strict_json(temporary / "timing.json", result.timings)
        save_strict_json(temporary / "memory.json", result.memory)
        output_files = {
            path.name: path
            for path in temporary.iterdir()
            if path.is_file() and path.name != "refinement_metadata.json"
        }
        selected_metrics = next(
            (
                item
                for item in selection.candidate_metrics
                if item["candidate_id"] == selection.selected_hypothesis_id
            ),
            None,
        )
        metadata = {
            "schema_version": 1,
            "sample_id": str(loaded["metadata"]["sample_id"]),
            "query": loaded["query"],
            "model_id": result.runtime_metadata["model_id"],
            "model_revision_sha": result.runtime_metadata["model_revision"],
            "model_class": result.model_class,
            "processor_class": result.processor_class,
            "transformers_version": result.runtime_metadata["transformers_version"],
            "pytorch_version": result.runtime_metadata["torch_version"],
            "device": "cpu",
            "dtype": "float32",
            "processor_size": result.runtime_metadata["processor_size"],
            "cpu_thread_count": result.runtime_metadata["torch_num_threads"],
            "interop_thread_count": result.runtime_metadata[
                "torch_num_interop_threads"
            ],
            "prompt_mode": prompt_mode,
            "positive_points_xy": [list(point) for point in prompt.positive_points_xy],
            "negative_points_xy": [list(point) for point in prompt.negative_points_xy],
            "bounding_box_xyxy": list(prompt.expanded_box_xyxy),
            "hypothesis_count": len(result.masks),
            "selected_hypothesis_id": selection.selected_hypothesis_id,
            "model_quality": (
                None if selected_metrics is None else selected_metrics["sam_quality"]
            ),
            "refinement_score": selection.refinement_score,
            "selected_mask_source": selection.selected_mask_source,
            "fallback": selection.selected_mask_source == "hifics_fallback",
            "fallback_reason": fallback_reason,
            "inference_succeeded": inference_error is None,
            "inference_error": inference_error,
            "model_load_time_seconds": result.timings["model_load_seconds"],
            "preprocess_time_seconds": result.timings["preprocess_seconds"],
            "inference_time_seconds": result.timings["inference_seconds"],
            "postprocess_time_seconds": result.timings["postprocess_seconds"],
            "peak_rss_bytes": result.memory["peak_rss_bytes"],
            "output_schema": result.output_schema,
            "input_checksums": {
                name: sha256_file(sample_dir / name) for name in REQUIRED_INPUTS
            },
            "outputs": file_manifest(output_files),
            "ground_truth_used_for_prompt_or_selection": False,
        }
        assert_no_ground_truth_leakage(
            {
                key: value
                for key, value in metadata.items()
                if key != "ground_truth_used_for_prompt_or_selection"
            },
            context="SAM 3 CPU refinement metadata",
        )
        save_strict_json(temporary / "refinement_metadata.json", metadata)
    return metadata

