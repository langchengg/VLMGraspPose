import pytest
import torch

from ocid_vlg.device import (
    cuda_device_ids,
    resolve_device,
    should_pin_memory,
    should_use_cuda_ddp,
)


def test_explicit_cpu_is_always_available():
    assert resolve_device("cpu") == torch.device("cpu")
    assert should_pin_memory("cpu") is False


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_auto_uses_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_explicit_unavailable_accelerator_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


def test_only_cuda_multi_gpu_can_enable_ddp(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    assert should_use_cuda_ddp("cuda") is True
    assert cuda_device_ids("cuda") == [0, 1, 2]
    assert should_use_cuda_ddp("cuda", single_process=True) is False
    assert should_use_cuda_ddp("cpu") is False


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_real_mps_selection_when_available():
    assert resolve_device("mps") == torch.device("mps")
