from types import SimpleNamespace

import torch
import pytest

import numpy as np

from train_crog_mac import _should_validate_epoch
from engine.crog_engine import _as_metric_tensor, _freeze_batch_norm_1d, train_with_grasp
from utils.misc import concat_all_gather


class ConstantGradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, image, text, mask, qua, sin, cos, wid):
        pred = self.weight.expand_as(mask)
        loss = self.weight * image.mean()
        predictions = (pred.detach(), qua.detach(), sin.detach(), cos.detach(), wid.detach())
        targets = (mask, qua, sin, cos, wid)
        losses = {"m_ins": 0.0, "m_qua": 0.0, "m_sin": 0.0, "m_cos": 0.0, "m_wid": 0.0}
        return predictions, targets, loss, losses


class SchedulerStub:
    def get_last_lr(self):
        return [0.1]


class NonFiniteLossModel(ConstantGradientModel):
    def forward(self, image, text, mask, qua, sin, cos, wid):
        pred = self.weight.expand_as(mask)
        loss = self.weight * torch.tensor(float("nan"))
        predictions = (pred.detach(), qua.detach(), sin.detach(), cos.detach(), wid.detach())
        targets = (mask, qua, sin, cos, wid)
        losses = {"m_ins": 0.0, "m_qua": 0.0, "m_sin": 0.0, "m_cos": 0.0, "m_wid": 0.0}
        return predictions, targets, loss, losses


def make_batch():
    mask = torch.zeros(1, 2, 2)
    return {
        "img": torch.ones(1, 3, 2, 2),
        "word_vec": torch.ones(1, 2, dtype=torch.long),
        "mask": mask,
        "grasp_masks": {
            "qua": mask.clone(),
            "sin": mask.clone(),
            "cos": mask.clone(),
            "wid": mask.clone(),
        },
    }


def test_gradient_accumulation_steps_final_partial_window():
    model = ConstantGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = SimpleNamespace(
        accumulation_steps=2,
        device=torch.device("cpu"),
        epochs=1,
        freeze_bn1d_stats=False,
        max_norm=0.0,
        print_freq=100,
    )

    train_with_grasp(
        [make_batch() for _ in range(5)],
        model,
        optimizer,
        SchedulerStub(),
        None,
        1,
        args,
    )

    assert torch.isclose(model.weight.detach(), torch.tensor(-0.3), atol=1e-6)


def test_mid_epoch_checkpoint_callback_runs_after_optimizer_steps():
    model = ConstantGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    calls = []
    args = SimpleNamespace(
        accumulation_steps=2,
        device=torch.device("cpu"),
        epochs=1,
        freeze_bn1d_stats=False,
        max_norm=0.0,
        mid_epoch_checkpoint_callback=lambda epoch, iteration, total: calls.append(
            (epoch, iteration, total)
        ),
        print_freq=100,
    )

    train_with_grasp(
        [make_batch() for _ in range(5)],
        model,
        optimizer,
        SchedulerStub(),
        None,
        1,
        args,
    )

    assert calls == [(1, 2, 5), (1, 4, 5), (1, 5, 5)]


def test_freeze_batch_norm_1d_keeps_affine_parameters_trainable():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    model.train()

    _freeze_batch_norm_1d(model)

    batch_norm = model[1]
    assert model.training is True
    assert batch_norm.training is False
    assert batch_norm.weight.requires_grad is True
    assert batch_norm.bias.requires_grad is True


def test_concat_all_gather_is_identity_without_distributed_initialization():
    tensor = torch.tensor([1.0, 2.0])

    assert torch.equal(concat_all_gather(tensor), tensor)


def test_numpy_metrics_are_cast_to_mps_supported_float32():
    values = np.asarray([0.0, 0.5], dtype=np.float64)

    tensor = _as_metric_tensor(values, torch.device("cpu"))

    assert tensor.dtype == torch.float32


def test_training_stops_on_non_finite_loss():
    model = NonFiniteLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = SimpleNamespace(
        accumulation_steps=1,
        device=torch.device("cpu"),
        epochs=1,
        freeze_bn1d_stats=False,
        max_norm=0.0,
        memory_sample_interval=1000,
        print_freq=100,
    )

    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        train_with_grasp(
            [make_batch()], model, optimizer, SchedulerStub(), None, 1, args
        )


def test_validation_interval_keeps_final_epoch_validated():
    args = SimpleNamespace(validation_interval=5, epochs=12)

    assert not _should_validate_epoch(1, args)
    assert _should_validate_epoch(5, args)
    assert _should_validate_epoch(10, args)
    assert _should_validate_epoch(12, args)
