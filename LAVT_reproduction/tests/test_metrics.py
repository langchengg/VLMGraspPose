import math

import pytest
import torch

from ocid_vlg.losses import multiclass_dice_loss, weighted_cross_entropy
from ocid_vlg.metrics import (
    SegmentationMetrics,
    aggregate_metrics,
    compute_sample_metrics,
    logits_to_probabilities_and_mask,
    precision_at_thresholds,
    prediction_from_probabilities,
    resize_logits,
    resize_probabilities,
)


def test_losses_are_finite_on_cpu_and_backward_is_finite():
    logits = torch.zeros((2, 2, 3, 4), requires_grad=True)
    targets = torch.stack(
        (torch.zeros((3, 4), dtype=torch.long), torch.ones((3, 4), dtype=torch.long))
    )

    ce = weighted_cross_entropy(logits, targets)
    dice = multiclass_dice_loss(logits, targets)
    total = ce + dice
    total.backward()

    assert math.isfinite(ce.item())
    assert math.isfinite(dice.item())
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("foreground_pixels", [0, 1, 12])
def test_two_class_dice_is_finite_for_empty_and_nonempty_masks(foreground_pixels):
    target = torch.zeros((1, 3, 4), dtype=torch.long)
    target.flatten()[:foreground_pixels] = 1
    logits = torch.randn((1, 2, 3, 4), requires_grad=True)
    loss = multiclass_dice_loss(logits, target, epsilon=1e-6)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_original_resolution_logits_and_probabilities_use_bilinear_resize():
    logits = torch.tensor(
        [[[[4.0, -4.0], [4.0, -4.0]], [[-4.0, 4.0], [-4.0, 4.0]]]]
    )
    resized_logits = resize_logits(logits, (4, 6))
    probability, prediction = logits_to_probabilities_and_mask(
        logits, output_size=(4, 6)
    )
    resized_probability = resize_probabilities(
        torch.softmax(logits, dim=1), (4, 6)
    )
    probability_prediction = prediction_from_probabilities(
        torch.softmax(logits, dim=1), output_size=(4, 6)
    )

    assert resized_logits.shape == (1, 2, 4, 6)
    assert probability.shape == prediction.shape == (1, 4, 6)
    assert resized_probability.shape == (1, 2, 4, 6)
    assert torch.equal(prediction, resized_logits.argmax(dim=1))
    assert torch.equal(probability_prediction, resized_probability.argmax(dim=1))
    assert torch.all((probability >= 0) & (probability <= 1))


@pytest.mark.parametrize("threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_precision_uses_strict_float32_threshold_boundary(threshold):
    result = precision_at_thresholds([threshold], thresholds=(threshold,))
    assert result[threshold] == 0.0


def test_miou_overall_iou_probability_empty_and_timing_aggregates():
    samples = [
        compute_sample_metrics(
            [[1]], [[1]], foreground_probability=[[0.2]], inference_time_seconds=1.0
        ),
        compute_sample_metrics(
            [[1, 0, 0]],
            [[0, 1, 0]],
            foreground_probability=[[0.6, 0.6, 0.6]],
            inference_time_seconds=2.0,
        ),
        compute_sample_metrics(
            [[0]], [[0]], foreground_probability=[[0.8]], inference_time_seconds=9.0
        ),
    ]
    metrics = aggregate_metrics(samples)

    assert metrics["mean_iou"] == pytest.approx(2 / 3)
    assert metrics["overall_iou"] == pytest.approx(1 / 3)
    assert metrics["p_at_50"] == pytest.approx(2 / 3)
    assert metrics["mean_foreground_probability"] == pytest.approx(2.8 / 5)
    assert metrics["empty_prediction_count"] == 1
    assert metrics["empty_gt_count"] == 1
    assert metrics["missing_prediction_count"] == 0
    assert metrics["inference_time_mean"] == pytest.approx(4.0)
    assert metrics["inference_time_median"] == pytest.approx(2.0)
    assert metrics["inference_time_p95"] == pytest.approx(8.3)
    assert metrics["precision_at_threshold_comparison"] == ">"


def test_missing_prediction_is_counted_and_scored_as_empty():
    sample = compute_sample_metrics(None, [[1]], missing_prediction=True)
    metrics = aggregate_metrics([sample])
    assert sample["iou"] == 0.0
    assert metrics["missing_prediction_count"] == 1
    assert metrics["empty_prediction_count"] == 1


def test_stateful_accumulator_accepts_logits_and_resizes_to_target():
    logits = torch.tensor([[[[3.0]], [[-3.0]]], [[[-3.0]], [[3.0]]]])
    target = torch.stack(
        (
            torch.zeros((3, 5), dtype=torch.long),
            torch.ones((3, 5), dtype=torch.long),
        )
    )
    accumulator = SegmentationMetrics()
    rows = accumulator.update(logits, target, inference_time_seconds=[0.1, 0.2])

    assert len(rows) == 2
    assert accumulator.compute()["mean_iou"] == pytest.approx(1.0)
    assert accumulator.compute()["test_sample_count"] == 2
