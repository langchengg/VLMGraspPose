"""Single-sample SAM 3 refinement orchestration."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .sam3_mask_selector import select_refined_mask
from .sam3_model import OfficialSam3Tracker, Sam3InferenceResult
from .sam3_prompt_builder import build_visual_prompt
from .sam3_serialization import file_manifest, save_strict_json, sha256_file, verify_file_manifest
from .sam3_visualization import (
    save_candidate_grid,
    save_coarse_vs_refined,
    save_mask_overlay,
    save_prompt_visualization,
)


def load_refinement_input(sample_dir: Path | str) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    verify_file_manifest(sample_dir, metadata["input_files"])
    rgb = np.asarray(Image.open(sample_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    coarse_mask = np.asarray(Image.open(sample_dir / "coarse_mask.png")) > 0
    probability = np.load(sample_dir / "coarse_probability.npy", allow_pickle=False)
    if rgb.shape[:2] != coarse_mask.shape or probability.shape != coarse_mask.shape:
        raise ValueError("RGB, coarse mask, and probability are not aligned")
    if probability.dtype != np.float32 or not np.all(np.isfinite(probability)):
        raise ValueError("coarse probability must be finite float32")
    return {"metadata": metadata, "rgb": rgb, "coarse_mask": coarse_mask, "coarse_probability": probability}


def refine_sample(
    sample_dir: Path | str,
    output_dir: Path | str,
    *,
    model: OfficialSam3Tracker,
    config: Mapping[str, Any],
    depth_m: np.ndarray | None = None,
) -> dict[str, Any]:
    sample_dir, output_dir = Path(sample_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refinement output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    loaded = load_refinement_input(sample_dir)
    rgb, coarse_mask, coarse_probability = loaded["rgb"], loaded["coarse_mask"], loaded["coarse_probability"]
    prompt = build_visual_prompt(coarse_probability, config["prompt"])
    save_prompt_visualization(rgb, prompt, output_dir / "prompt_visualization.png")
    save_strict_json(output_dir / "prompt_metadata.json", prompt.to_dict())
    inference_error: str | None = None
    try:
        result = model.infer(Image.fromarray(rgb, mode="RGB"), prompt)
    except Exception as error:
        inference_error = f"{type(error).__name__}: {error}"
        result = Sam3InferenceResult(
            masks=(),
            probabilities=(),
            qualities=(),
            runtime_seconds=0.0,
            runtime_metadata=dict(model.runtime_metadata),
        )
    selection = select_refined_mask(
        result.masks,
        result.probabilities,
        result.qualities,
        coarse_mask=coarse_mask,
        coarse_probability=coarse_probability,
        prompt=prompt,
        depth_m=depth_m,
        config=config["selector"],
    )
    shutil.copy2(sample_dir / "rgb.png", output_dir / "rgb.png")
    shutil.copy2(sample_dir / "coarse_mask.png", output_dir / "coarse_mask.png")
    shutil.copy2(sample_dir / "coarse_probability.npy", output_dir / "coarse_probability.npy")
    np.savez_compressed(
        output_dir / "sam3_candidate_masks.npz",
        candidate_id=np.asarray([item["candidate_id"] for item in selection.candidate_metrics]),
        masks=np.asarray(result.masks, dtype=np.uint8),
        probabilities=np.asarray(result.probabilities, dtype=np.float32),
        sam_quality=np.asarray(result.qualities, dtype=np.float32),
    )
    metric_fields = list(selection.candidate_metrics[0]) if selection.candidate_metrics else ["candidate_id"]
    with (output_dir / "sam3_candidate_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(selection.candidate_metrics)
    Image.fromarray(selection.selected_mask.astype(np.uint8) * 255, mode="L").save(output_dir / "refined_mask.png")
    np.save(output_dir / "refined_probability.npy", selection.selected_probability.astype(np.float32), allow_pickle=False)
    save_candidate_grid(rgb, result.masks, selection.candidate_metrics, output_dir / "sam3_candidates_grid.png")
    save_mask_overlay(rgb, selection.selected_mask, output_dir / "refined_overlay.png", title=f"Selected: {selection.selected_mask_source}")
    save_coarse_vs_refined(rgb, coarse_mask, selection.selected_mask, output_dir / "coarse_vs_refined.png")
    output_files = {
        path.name: path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "refinement_metadata.json"
    }
    metadata = {
        "schema_version": 1,
        "sample_id": loaded["metadata"]["sample_id"],
        "model": result.runtime_metadata,
        "prompt": prompt.to_dict(),
        "number_of_sam_hypotheses": len(result.masks),
        "selected_hypothesis_id": selection.selected_hypothesis_id,
        "selected_mask_source": selection.selected_mask_source,
        "sam_model_score": None if selection.selected_hypothesis_id is None else next(
            item["sam_quality"] for item in selection.candidate_metrics if item["candidate_id"] == selection.selected_hypothesis_id
        ),
        "refinement_score": selection.refinement_score,
        "fallback": selection.selected_mask_source == "hifics_fallback",
        "fallback_reason": (
            f"model_inference_failed|{inference_error}"
            if inference_error is not None
            else selection.fallback_reason
        ),
        "inference_succeeded": inference_error is None,
        "inference_error": inference_error,
        "runtime_seconds": result.runtime_seconds,
        "input_metadata_sha256": sha256_file(sample_dir / "metadata.json"),
        "inputs": loaded["metadata"]["input_files"],
        "outputs": file_manifest(output_files),
    }
    save_strict_json(output_dir / "refinement_metadata.json", metadata)
    return metadata
