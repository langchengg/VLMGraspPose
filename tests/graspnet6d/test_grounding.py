from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from graspnet6d import grounding
from graspnet6d.grounding import (
    EXPECTED_CHECKPOINT_FORMAT,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_CLIP_WEIGHT_SHA256,
    EXPECTED_HIFI_CHECKPOINT_SHA256,
    GroundingPrediction,
    compute_mask_metrics,
    load_hifi_model,
    save_grounding_prediction,
    summarize_mask_metrics,
    validate_adaptation_splits,
)


def test_real_retained_hifi_checkpoint_loads_strictly_and_zero_shot_is_frozen():
    bundle = load_hifi_model(device="cpu", mode="zero_shot")

    assert bundle.device == torch.device("cpu")
    assert bundle.mode == "zero_shot"
    assert bundle.checkpoint_sha256 == EXPECTED_HIFI_CHECKPOINT_SHA256
    assert bundle.clip_weight_sha256 == EXPECTED_CLIP_WEIGHT_SHA256
    assert bundle.checkpoint_metadata["global_step"] == EXPECTED_CHECKPOINT_STEP
    assert bundle.checkpoint_metadata["weight_sources"]["clip"]["cache_sha256"] == (
        EXPECTED_CLIP_WEIGHT_SHA256
    )
    assert len(bundle.adaptation_parameter_names) == 92
    assert bundle.model.__class__.__name__ == "HierarchicalCLIPDensePredT"
    assert tuple(bundle.model.extract_layers) == (1, 3, 5, 7, 9)
    assert tuple(bundle.model.projection_layer_indices) == (9, 7, 5, 3, 1)
    assert len(bundle.model.film_stages) == 5
    assert not bundle.model.training
    assert not bundle.model.clip_model.training
    assert all(not parameter.requires_grad for parameter in bundle.model.parameters())

    checkpoint = torch.load(
        bundle.checkpoint_path, map_location="cpu", weights_only=True
    )
    assert checkpoint["format"] == EXPECTED_CHECKPOINT_FORMAT
    first_name = bundle.adaptation_parameter_names[0]
    torch.testing.assert_close(
        dict(bundle.model.named_parameters())[first_name].detach().cpu(),
        checkpoint["trainable_state"][first_name],
    )
    del checkpoint, bundle
    gc.collect()


def test_mask_metrics_are_derived_from_pixels_and_grouped_by_template():
    zero = np.zeros((2, 2), dtype=bool)
    target_a = zero.copy()
    target_a[0, 0] = True

    prediction_a = target_a.copy()  # IoU 1.0
    prediction_b = np.array([[True, True], [False, False]])  # IoU 0.5
    prediction_c = np.array([[False, True], [False, False]])  # IoU 0.0
    prediction_d = zero.copy()  # empty union -> IoU 1.0
    other = np.array([[False, True], [False, False]])

    result = summarize_mask_metrics(
        [
            {
                "sample_id": "a",
                "group": "seen",
                "template": "name",
                "prediction": prediction_a,
                "target": target_a,
                "non_target_mask": other,
                "valid_depth": np.ones((2, 2), dtype=bool),
            },
            {
                "sample_id": "b",
                "group": "seen",
                "template": "spatial",
                "prediction": prediction_b,
                "target": target_a,
                "non_target_mask": other,
                "valid_depth": np.array([[True, False], [False, False]]),
            },
            {
                "sample_id": "c",
                "group": "novel",
                "template": "name",
                "prediction": prediction_c,
                "target": target_a,
                "non_target_mask": other,
                "valid_depth": zero,
            },
            {
                "sample_id": "d",
                "group": "novel",
                "template": "spatial",
                "prediction": prediction_d,
                "target": zero,
                "non_target_mask": other,
                "valid_depth": zero,
            },
        ]
    )

    overall = result["overall"]
    assert overall["mean_iou"] == pytest.approx(0.625)
    # The retained HiFi evaluator uses strict IoU > threshold, so IoU 0.5
    # does not count at P@50.
    assert overall["p_at_50"] == pytest.approx(0.5)
    assert overall["p_at_70"] == pytest.approx(0.5)
    assert overall["p_at_80"] == pytest.approx(0.5)
    assert overall["p_at_90"] == pytest.approx(0.5)
    assert overall["empty_predictions"] == 1
    assert overall["empty_prediction_rate"] == pytest.approx(0.25)
    assert overall["wrong_object_count"] == 1
    assert overall["wrong_object_rate"] == pytest.approx(0.25)
    assert overall["valid_depth_evaluable"] == 3
    assert overall["mean_valid_depth_coverage"] == pytest.approx(0.5)
    assert result["by_group"]["seen"]["mean_iou"] == pytest.approx(0.75)
    assert result["by_group"]["novel"]["mean_iou"] == pytest.approx(0.5)
    assert result["by_template"]["name"]["wrong_object_count"] == 1
    assert set(result["by_group_template"]) == {
        "novel::name",
        "novel::spatial",
        "seen::name",
        "seen::spatial",
    }


def test_single_mask_metric_reports_empty_and_valid_depth_contract():
    empty = np.zeros((3, 4), dtype=np.uint8)
    metrics = compute_mask_metrics(empty, empty, valid_depth=empty)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["empty_prediction"] is True
    assert metrics["valid_depth_coverage"] is None
    assert metrics["wrong_object"] is None

    with pytest.raises(ValueError, match="shape"):
        compute_mask_metrics(empty, np.zeros((4, 3), dtype=bool))


def test_adaptation_split_guard_is_scene_disjoint_and_never_accepts_test():
    train = [
        {"sample_id": "train-a", "scene_id": "scene-001", "split": "train"},
        {"sample_id": "train-b", "scene_id": "scene-002", "split": "train"},
    ]
    validation = [
        {"sample_id": "val-a", "scene_id": "scene-101", "split": "val"},
        {"sample_id": "val-b", "scene_id": "scene-102", "split": "val"},
    ]
    contract = validate_adaptation_splits(train, validation)
    assert contract["train_rows"] == 2
    assert contract["validation_rows"] == 2
    assert contract["scene_overlap"] == 0
    assert contract["test_rows_consumed"] == 0
    assert contract["early_stopping_split"] == "val"

    leaked = [
        {"sample_id": "val-c", "scene_id": "scene-001", "split": "val"}
    ]
    with pytest.raises(ValueError, match="scene_overlap"):
        validate_adaptation_splits(train, leaked)

    test_row = [
        {"sample_id": "test-a", "scene_id": "scene-201", "split": "test"}
    ]
    with pytest.raises(ValueError, match="test and implicit splits are forbidden"):
        validate_adaptation_splits(train, test_row)


def test_prediction_npz_and_mask_are_published_atomically(tmp_path: Path):
    probability_352 = np.full((352, 352), 0.25, dtype=np.float32)
    native_probability = np.array(
        [[0.2, 0.6, 0.7], [0.1, 0.4, 0.9]], dtype=np.float32
    )
    prediction = GroundingPrediction(
        probability_352=probability_352,
        native_probability=native_probability,
        native_mask=native_probability >= 0.5,
        native_height=2,
        native_width=3,
        foreground_threshold=0.5,
        query="the red mug",
        device="cpu",
    )
    probability_path = tmp_path / "prediction.npz"
    mask_path = tmp_path / "mask.png"
    manifest = save_grounding_prediction(
        prediction,
        probability_path=probability_path,
        mask_path=mask_path,
    )

    with np.load(probability_path, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["probability_352"], probability_352)
        np.testing.assert_array_equal(
            stored["native_probability"], native_probability
        )
        assert float(stored["foreground_threshold"]) == pytest.approx(0.5)
    mask = np.asarray(Image.open(mask_path).convert("L")) != 0
    np.testing.assert_array_equal(mask, prediction.native_mask)
    assert len(manifest["probability_sha256"]) == 64
    assert len(manifest["mask_sha256"]) == 64
    assert list(tmp_path.glob(".*")) == []


def test_adaptation_rows_derive_binary_target_from_official_instance_label(
    tmp_path: Path,
):
    rgb_path = tmp_path / "rgb.png"
    label_path = tmp_path / "label.png"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(rgb_path)
    labels = np.zeros((4, 5), dtype=np.uint16)
    labels[1:3, 2:4] = 7
    Image.fromarray(labels).save(label_path)
    dataset = grounding._AdaptationDataset(
        [
            {
                "rgb_path": str(rgb_path),
                "instance_label_path": str(label_path),
                "target_instance_label": 7,
                "query": "Pick the object.",
            }
        ]
    )

    image, target, query = dataset[0]

    assert tuple(image.shape) == (3, 352, 352)
    assert tuple(target.shape) == (1, 352, 352)
    assert 0 < int(target.sum()) < target.numel()
    assert query == "Pick the object."
