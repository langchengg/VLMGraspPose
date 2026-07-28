from collections import OrderedDict

import pytest
import torch

from utils.checkpoint import copy_checkpoint_atomic, load_checkpoint, save_checkpoint


def test_load_checkpoint_strips_data_parallel_prefix(tmp_path):
    source = torch.nn.Linear(2, 1)
    prefixed = OrderedDict(
        (f"module.{key}", value.clone()) for key, value in source.state_dict().items()
    )
    path = tmp_path / "ddp.pth"
    torch.save({"state_dict": prefixed, "epoch": 7}, path)
    target = torch.nn.Linear(2, 1)

    checkpoint = load_checkpoint(path, target, torch.device("cpu"))

    assert checkpoint["epoch"] == 7
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_save_checkpoint_uses_unwrapped_state_dict(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "model.pth"

    save_checkpoint(path, model, epoch=3, best_iou=0.5)
    checkpoint = torch.load(path, map_location="cpu")

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_iou"] == 0.5
    assert all(not key.startswith("module.") for key in checkpoint["state_dict"])


def test_save_checkpoint_cleans_temporary_file_on_failure(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "model.pth"

    def failing_save(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "save", failing_save)

    with pytest.raises(RuntimeError, match="boom"):
        save_checkpoint(path, model, epoch=1)

    assert not path.exists()
    assert not list(tmp_path.glob(".model.pth.*.tmp"))


def test_copy_checkpoint_atomic_replaces_destination_and_cleans_tmp(tmp_path):
    source = tmp_path / "source.pth"
    destination = tmp_path / "destination.pth"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")

    copy_checkpoint_atomic(source, destination)

    assert destination.read_bytes() == b"new"
    assert not list(tmp_path.glob(".destination.pth.*.tmp"))
