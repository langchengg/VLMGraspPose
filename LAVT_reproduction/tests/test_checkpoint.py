import random

import numpy as np
import pytest
import torch

import ocid_vlg.checkpoint as checkpoint_module
from ocid_vlg.checkpoint import load_checkpoint, save_checkpoint


def _training_state(device="cpu"):
    model = torch.nn.Linear(3, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )
    return model, optimizer, scheduler


def test_checkpoint_roundtrip_restores_full_state_and_next_epoch(tmp_path):
    model, optimizer, scheduler = _training_state()
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    model(inputs).square().mean().backward()
    optimizer.step()
    scheduler.step()
    expected = model(inputs).detach().clone()

    path = tmp_path / "checkpoint.pth"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=4,
        best_metrics={"miou": 0.73},
        config={"loss": "dice", "epochs": 40},
        seed=42,
        device="cpu",
    )

    restored_model, restored_optimizer, restored_scheduler = _training_state()
    payload = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        device="cpu",
    )

    torch.testing.assert_close(restored_model(inputs), expected)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert payload["epoch"] == 4
    assert payload["next_epoch"] == 5
    assert payload["best_metrics"] == {"miou": 0.73}
    assert payload["config"]["loss"] == "dice"
    assert payload["seed"] == 42
    assert payload["device"] == "cpu"


def test_checkpoint_payload_has_required_named_components(tmp_path):
    model, optimizer, scheduler = _training_state()
    path = tmp_path / "required.pth"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=0,
        best={"miou": 0.0},
        config={},
        seed=7,
        device="cpu",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert {
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "best",
        "config",
        "seed",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "device",
        "device_info",
    } <= payload.keys()


def test_checkpoint_restores_python_numpy_and_torch_rng(tmp_path):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model, optimizer, scheduler = _training_state()
    path = tmp_path / "rng.pth"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=2,
        config={},
        seed=17,
        device="cpu",
    )
    expected = (random.random(), np.random.random(), torch.rand(3))
    for _ in range(10):
        random.random()
        np.random.random()
        torch.rand(3)

    load_checkpoint(
        path, model, optimizer, scheduler, device="cpu", restore_rng=True
    )
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_atomic_save_preserves_existing_checkpoint_on_failure(tmp_path, monkeypatch):
    model, _, _ = _training_state()
    path = tmp_path / "atomic.pth"
    save_checkpoint(path, model, epoch=0, config={}, seed=1, device="cpu")
    before = path.read_bytes()

    def failing_save(payload, handle):
        handle.write(b"incomplete")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", failing_save)
    with pytest.raises(RuntimeError, match="simulated"):
        save_checkpoint(path, model, epoch=1, config={}, seed=1, device="cpu")

    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".atomic.pth.*.tmp"))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_mps_checkpoint_roundtrip_is_supported(tmp_path):
    model, optimizer, scheduler = _training_state("mps")
    path = tmp_path / "mps.pth"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=0,
        config={},
        seed=3,
        device="mps",
    )
    payload = load_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        device="mps",
    )
    assert payload["next_epoch"] == 1
