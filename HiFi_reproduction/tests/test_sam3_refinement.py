from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.segmentation.sam3_mask_selector import (
    area_ratio_penalty,
    depth_consistency_score,
    score_candidate_mask,
    select_refined_mask,
)
from src.segmentation.sam3_model import (
    OfficialSam3Tracker,
    Sam3InferenceResult,
    build_tracker_processor_inputs,
    restore_tracker_hypotheses,
)
from src.segmentation.sam3_prompt_builder import (
    build_visual_prompt,
    clean_coarse_mask,
    expand_box_xyxy,
)
from src.segmentation.sam3_refiner import load_refinement_input, refine_sample
from src.segmentation.sam3_serialization import (
    assert_no_ground_truth_leakage,
    file_manifest,
    save_strict_json,
    verify_file_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((REPO_ROOT / "configs" / "sam3_refinement.yaml").read_text())


def probability_fixture() -> np.ndarray:
    probability = np.zeros((80, 100), dtype=np.float32)
    probability[20:60, 25:75] = 0.9
    return probability


def prompt_fixture(strategy: str = "box_positive_points"):
    config = dict(CONFIG["prompt"])
    config["strategy"] = strategy
    return build_visual_prompt(probability_fixture(), config)


def test_coarse_mask_thresholding_and_small_component_removal():
    probability = probability_fixture()
    probability[2:4, 2:4] = 0.95
    cleaned, stats = clean_coarse_mask(
        probability, threshold=0.15, minimum_component_area_px=20
    )
    assert cleaned[30, 40]
    assert not cleaned[2, 2]
    assert stats == {
        "original_component_count": 2,
        "component_count": 1,
        "removed_component_count": 1,
    }


def test_connected_components_are_preserved_when_large():
    probability = probability_fixture()
    probability[5:15, 5:15] = 0.8
    prompt = build_visual_prompt(probability, CONFIG["prompt"])
    assert prompt.component_count == 2
    assert prompt.cleaned_mask[8, 8]
    assert prompt.cleaned_mask[30, 40]


def test_distance_transform_points_are_inside_and_main_is_deepest():
    prompt = prompt_fixture()
    assert len(prompt.positive_points_xy) >= 1
    assert all(prompt.cleaned_mask[y, x] for x, y in prompt.positive_points_xy)
    main_x, main_y = prompt.positive_points_xy[0]
    assert 43 <= main_x <= 56
    assert 38 <= main_y <= 41


def test_expanded_box_clips_to_image():
    assert expand_box_xyxy((0, 0, 99, 79), (80, 100), 0.1) == (0, 0, 99, 79)


def test_negative_points_are_outside_coarse_mask():
    prompt = prompt_fixture("box_positive_negative_points")
    assert prompt.negative_points_xy
    assert all(not prompt.cleaned_mask[y, x] for x, y in prompt.negative_points_xy)


def test_prompt_serialization_and_official_nested_inputs(tmp_path: Path):
    prompt = prompt_fixture()
    output = save_strict_json(tmp_path / "prompt.json", prompt.to_dict())
    assert json.loads(output.read_text())["strategy"] == "box_positive_points"
    processor_inputs = build_tracker_processor_inputs(Image.new("RGB", (100, 80)), prompt)
    assert processor_inputs["input_boxes"] == [[list(prompt.expanded_box_xyxy)]]
    assert processor_inputs["input_points"] == [[[list(point) for point in prompt.positive_points_xy]]]
    assert processor_inputs["input_labels"] == [[[1] * len(prompt.positive_points_xy)]]


def test_result_resolution_restoration():
    logits = np.zeros((1, 3, 80, 100), dtype=np.float32)
    logits[:, 1, 20:60, 25:75] = 10.0
    probabilities, masks = restore_tracker_hypotheses(logits, expected_shape=(80, 100))
    assert probabilities.shape == masks.shape == (3, 80, 100)
    assert masks[1, 30, 40]
    with pytest.raises(RuntimeError, match="expected"):
        restore_tracker_hypotheses(logits, expected_shape=(40, 50))


def test_candidate_scoring_and_deterministic_tie_break():
    prompt = prompt_fixture()
    mask = prompt.cleaned_mask
    probability = mask.astype(np.float32)
    metrics = score_candidate_mask(
        "sam3_000",
        mask,
        probability,
        0.8,
        coarse_mask=mask,
        prompt=prompt,
        depth_m=None,
        config=CONFIG["selector"],
    )
    assert metrics["coarse_iou"] == 1.0
    result = select_refined_mask(
        [mask, mask],
        [probability, probability],
        [0.8, 0.8],
        coarse_mask=mask,
        coarse_probability=probability,
        prompt=prompt,
        depth_m=None,
        config=CONFIG["selector"],
    )
    assert result.selected_hypothesis_id == "sam3_000"
    assert result.selected_mask_source == "sam3"


def test_fallback_rules_no_mask_and_missing_main_point():
    prompt = prompt_fixture()
    coarse = prompt.cleaned_mask
    probability = coarse.astype(np.float32)
    no_masks = select_refined_mask(
        [],
        [],
        [],
        coarse_mask=coarse,
        coarse_probability=probability,
        prompt=prompt,
        depth_m=None,
        config=CONFIG["selector"],
    )
    assert no_masks.selected_mask_source == "hifics_fallback"
    assert no_masks.fallback_reason == "no_sam_masks_returned"
    empty = np.zeros_like(coarse)
    rejected = select_refined_mask(
        [empty],
        [empty.astype(np.float32)],
        [0.99],
        coarse_mask=coarse,
        coarse_probability=probability,
        prompt=prompt,
        depth_m=None,
        config=CONFIG["selector"],
    )
    assert rejected.selected_mask_source == "hifics_fallback"
    assert "main_positive_point_missing" in rejected.fallback_reason


def test_invalid_probability_is_retained_then_falls_back():
    prompt = prompt_fixture()
    coarse = prompt.cleaned_mask
    invalid = coarse.astype(np.float32)
    invalid[0, 0] = np.nan
    result = select_refined_mask(
        [coarse],
        [invalid],
        [0.9],
        coarse_mask=coarse,
        coarse_probability=coarse.astype(np.float32),
        prompt=prompt,
        depth_m=None,
        config=CONFIG["selector"],
    )
    assert result.selected_mask_source == "hifics_fallback"
    assert result.candidate_metrics[0]["valid_hypothesis"] is False
    assert "invalid_sam_hypothesis" in result.fallback_reason


def test_empty_coarse_mask_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        build_visual_prompt(np.zeros((20, 20), dtype=np.float32), CONFIG["prompt"])


def test_area_ratio_penalty():
    config = CONFIG["selector"]["area_ratio_penalty"]
    assert area_ratio_penalty(1.0, config) == 0.0
    assert area_ratio_penalty(0.0, config) == 1.0
    assert area_ratio_penalty(8.0, config) == 1.0


def test_depth_consistency():
    coarse = np.zeros((20, 20), dtype=bool)
    coarse[5:15, 5:15] = True
    candidate = coarse.copy()
    depth = np.ones((20, 20), dtype=np.float32)
    config = dict(CONFIG["selector"]["depth_consistency"])
    config["enabled"] = True
    assert depth_consistency_score(candidate, coarse, depth, config) == 1.0
    depth[candidate] = 2.0
    depth[coarse] = 1.0
    candidate[0:2, 0:2] = True
    assert 0.9 < depth_consistency_score(candidate, coarse, depth, config) <= 1.0


def test_mask_checksums_and_manifest_integrity(tmp_path: Path):
    path = tmp_path / "mask.bin"
    path.write_bytes(b"mask")
    manifest = file_manifest({"mask": path})
    verify_file_manifest(tmp_path, manifest)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest mismatch"):
        verify_file_manifest(tmp_path, manifest)


def test_no_ground_truth_leakage():
    assert_no_ground_truth_leakage({"coarse_mask": "coarse_mask.png"})
    with pytest.raises(ValueError, match="forbidden key"):
        assert_no_ground_truth_leakage({"ground_truth_mask": "x.png"})
    with pytest.raises(ValueError, match="evaluation-only"):
        assert_no_ground_truth_leakage({"path": "ground_truth_mask.png"})


def test_real_prepared_ten_sample_bundle_is_prediction_only():
    root = REPO_ROOT / "outputs" / "sam3_refinement_inputs"
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    assert [row["sample_id"] for row in rows] == [
        line.strip()
        for line in (REPO_ROOT / "configs" / "sam3_ten_sample_ids.txt").read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        assert_no_ground_truth_leakage(row)
        loaded = load_refinement_input(root / row["sample_id"])
        assert loaded["rgb"].shape == (480, 640, 3)
        assert loaded["coarse_probability"].dtype == np.float32


class FakeModel:
    runtime_metadata = {
        "model_id_or_path": "fake-test-double",
        "model_revision": "test-only",
        "transformers_version": "test-only",
        "torch_version": "test-only",
        "cuda_version": None,
        "gpu_name": None,
        "inference_precision": "fp32",
    }

    def infer(self, image, prompt):
        mask = prompt.cleaned_mask
        return Sam3InferenceResult(
            masks=(mask,),
            probabilities=(mask.astype(np.float32),),
            qualities=(0.9,),
            runtime_seconds=0.01,
            runtime_metadata=dict(self.runtime_metadata),
        )


def synthetic_input_bundle(sample: Path) -> None:
    sample.mkdir()
    rgb = np.full((80, 100, 3), 127, dtype=np.uint8)
    probability = probability_fixture()
    Image.fromarray(rgb).save(sample / "rgb.png")
    Image.fromarray((probability >= 0.15).astype(np.uint8) * 255).save(sample / "coarse_mask.png")
    np.save(sample / "coarse_probability.npy", probability, allow_pickle=False)
    metadata = {
        "sample_id": "synthetic",
        "input_files": file_manifest(
            {
                "rgb": sample / "rgb.png",
                "coarse_mask": sample / "coarse_mask.png",
                "coarse_probability": sample / "coarse_probability.npy",
            }
        ),
    }
    save_strict_json(sample / "metadata.json", metadata)


def test_synthetic_end_to_end_serialization(tmp_path: Path):
    sample = tmp_path / "input"
    synthetic_input_bundle(sample)
    output = tmp_path / "output"
    result = refine_sample(sample, output, model=FakeModel(), config=CONFIG)
    assert result["selected_mask_source"] == "sam3"
    assert result["inference_succeeded"] is True
    assert (output / "sam3_candidate_masks.npz").is_file()
    assert (output / "coarse_vs_refined.png").is_file()
    verify_file_manifest(output, result["outputs"])


class FailingModel(FakeModel):
    def infer(self, image, prompt):
        raise RuntimeError("synthetic CUDA failure")


def test_model_failure_writes_explicit_fallback(tmp_path: Path):
    sample = tmp_path / "input"
    synthetic_input_bundle(sample)
    output = tmp_path / "fallback_output"
    result = refine_sample(sample, output, model=FailingModel(), config=CONFIG)
    assert result["selected_mask_source"] == "hifics_fallback"
    assert result["inference_succeeded"] is False
    assert result["fallback_reason"].startswith("model_inference_failed|RuntimeError")
    assert np.array_equal(
        np.asarray(Image.open(output / "refined_mask.png")) > 0,
        np.asarray(Image.open(sample / "coarse_mask.png")) > 0,
    )


GPU_ENABLED = os.environ.get("RUN_SAM3_GPU_TESTS") == "1"


@pytest.mark.skipif(not GPU_ENABLED, reason="requires authorized official weights and NVIDIA CUDA")
def test_official_model_loading_and_one_gpu_inference():
    model_path = os.environ["SAM3_MODEL_PATH"]
    revision = os.environ["SAM3_MODEL_REVISION"]
    model = OfficialSam3Tracker(
        model_path,
        revision=revision,
        local_files_only=True,
        precision="fp32",
    )
    prompt = prompt_fixture()
    result = model.infer(Image.new("RGB", (100, 80), "gray"), prompt)
    assert result.masks
    assert all(mask.shape == (80, 100) for mask in result.masks)


@pytest.mark.skipif(
    not (REPO_ROOT / "outputs" / "sam3_refined_masks" / "run_metadata.json").is_file(),
    reason="requires imported real SAM 3 outputs",
)
def test_real_downstream_artifact_gate():
    run = json.loads(
        (REPO_ROOT / "outputs" / "sam3_refined_masks" / "run_metadata.json").read_text()
    )
    assert run["experiment_status"] == "real_sam3_inference_completed"
