from unittest.mock import Mock

import torch

from utils.device import empty_cache, get_device, get_memory_stats, move_to_device


def test_get_device_prefers_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert get_device().type == "mps"


def test_get_device_can_prefer_cuda(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert get_device(prefer_mps=False).type == "cuda"


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_device().type == "cpu"


def test_move_to_device_recurses_without_changing_metadata():
    tensor = torch.tensor([1.0])
    batch = {
        "tensor": tensor,
        "nested": [tensor, (tensor, "sentence")],
        "metadata": {"path": "image.png"},
    }

    moved = move_to_device(batch, torch.device("cpu"))

    assert moved["tensor"].device.type == "cpu"
    assert moved["nested"][1][1] == "sentence"
    assert moved["metadata"] == {"path": "image.png"}


def test_empty_cache_dispatches_by_device(monkeypatch):
    cuda_empty = Mock()
    mps_empty = Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", cuda_empty)
    monkeypatch.setattr(torch.mps, "empty_cache", mps_empty)

    empty_cache(torch.device("cpu"))
    empty_cache(torch.device("cuda"))
    empty_cache(torch.device("mps"))

    cuda_empty.assert_called_once_with()
    mps_empty.assert_called_once_with()


def test_get_memory_stats_reads_mps_allocator(monkeypatch):
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 100)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 200)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1000)

    stats = get_memory_stats(torch.device("mps"))

    assert stats == {
        "allocated_bytes": 100,
        "driver_bytes": 200,
        "recommended_bytes": 1000,
    }
